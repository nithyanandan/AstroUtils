from __future__ import print_function, division, unicode_literals, absolute_import
import glob, copy
import argparse
import yaml, h5py
import healpy as HP
import numpy as NP
from scipy import interpolate
from astropy import units as U
from astropy.time import Time
import astropy.cosmology as cosmology
import astropy.constants as FCNST
import astropy.convolution as CONV
from astropy.coordinates import Galactic, FK5, ICRS, SkyCoord, AltAz, EarthLocation
from astropy import wcs
from astropy.io import fits
import progressbar as PGB
import time, warnings
from astroutils import cosmotile
from astroutils import mathops as OPS
from astroutils import DSP_modules as DSP
from astroutils import constants as CNST
from astroutils import catalog as SM
from astroutils import geometry as GEOM
from astroutils import ephemeris_timing as ET
from astroutils import catalog as SM
import astroutils
import ipdb as PDB

cosmoPlanck15 = cosmology.Planck15 # Planck 2015 cosmology
cosmo100_Planck15 = cosmoPlanck15.clone(name='Modified Planck 2015 cosmology with h=1.0', H0=100.0) # Modified Planck 2015 cosmology with h=1.0, H= 100 km/s/Mpc

#################################################################################

def read_raw_lightcone_cube(parms):

    cube_source = parms['cube_source']
    if not isinstance(cube_source, str):
        raise TypeError('Input cube_source must be a string')
    if cube_source.lower() not in ['21cmfast']:
        raise ValueError('{0} cubes currently not supported'.format(cube_source))

    indir = parms['input']['indir']
    infile_prefix = parms['input']['infile_prefix']
    if infile_prefix is None:
        infile_prefix = ''
    infile_suffix = parms['input']['infile_suffix']
    if infile_suffix is None:
        infile_suffix = ''

    fname_delimiter = parms['parseinfo']['delimiter']
    zstart_pos = parms['parseinfo']['zstart_pos']
    zend_pos = parms['parseinfo']['zend_pos']
    zstart_identifier = parms['parseinfo']['zstart_identifier']
    zend_identifier = parms['parseinfo']['zend_identifier']
    zstart_identifier_pos = parms['parseinfo']['zstart_identifier_pos']
    zend_identifier_pos = parms['parseinfo']['zend_identifier_pos']
    if zstart_identifier is not None:
        if zstart_identifier_pos.lower() not in ['before', 'after']:
            raise ValueError('zstart_identifier_pos must be set to "before" or "after"')
        elif zstart_identifier_pos.lower() == 'before':
            zstart_value_place = 1
        else:
            zstart_value_place = 0

    if zend_identifier is not None:
        if zend_identifier_pos.lower() not in ['before', 'after']:
            raise ValueError('zend_identifier_pos must be set to "before" or "after"')
        elif zend_identifier_pos.lower() == 'before':
            zend_value_place = 1
        else:
            zend_value_place = 0
            
    if cube_source.lower() not in ['21cmfast']:
        raise ValueError('{0} cubes currently not supported'.format(cube_source))

    fullfnames = glob.glob(indir + infile_prefix + '*' + infile_suffix)
    if len(fullfnames) == 0:
        raise IOError('No raw lightcone files found for input at the specified location: {0}. Check the input directory and the filenames.'.format(indir))
    fullfnames = NP.asarray(fullfnames)
    fnames = [fname.split('/')[-1] for fname in fullfnames]
    fnames = NP.asarray(fnames)
    if fnames[0].split(fname_delimiter)[-1] == 'lighttravel':
        dim = int(fnames[0].split(fname_delimiter)[-3])
        boxsize = float(fnames[0].split(fname_delimiter)[-2][:-3])
    else:
        dim = int(fnames[0].split(fname_delimiter)[-2])
        boxsize = float(fnames[0].split(fname_delimiter)[-1][:-3])
    boxres = boxsize / dim
    if zstart_identifier is not None:
        zstart_str = [fname.replace(fname_delimiter,' ').split()[zstart_pos].split(zstart_identifier)[zstart_value_place] for fname in fnames]
    else:
        zstart_str = [fname.replace(fname_delimiter,' ').split()[zstart_pos] for fname in fnames]
    if zend_identifier is not None:
        zend_str = [fname.replace(fname_delimiter,' ').split()[zend_pos].split(zend_identifier)[zend_value_place] for fname in fnames]
    else:
        zend_str = [fname.replace(fname_delimiter,' ').split()[zend_pos] for fname in fnames]
    
    zstart = NP.asarray(map(float, zstart_str))
    zend = NP.asarray(map(float, zend_str))
    sortind = NP.argsort(zstart)
    zstart = zstart[sortind]
    zend = zend[sortind]
    fnames = fnames[sortind]
    fullfnames = fullfnames[sortind]
    # lightcone_cube = None
    lc_cube = []
    for fi,fullfname in enumerate(fullfnames):
        lc_cube += [cosmotile.fastread_21cmfast_cube(fullfname)]
    return (NP.concatenate(lc_cube, axis=0), boxres, zstart.min())

#################################################################################

def write_lightcone_cube(lightcone_cube, boxres, zmin, outfile_prefix, 
                         zbins=None, cosmo=cosmo100_Planck15,
                         rest_freq=CNST.rest_freq_HI*U.Hz):

    """
    Write a lightcone cube to a file in HDF5 format
    """

    cosmo100 = cosmo.clone(name='Modified cosmology with h=1.0', H0=100.0) # Modified cosmology with h=1.0, H= 100 km/s/Mpc

    # monopole_Tb = monopole_temperature(lightcone_cube)

    if not isinstance(lightcone_cube, NP.ndarray):
        raise TypeError('Input lightcone_cube must be a numpy array')

    boxres_ax_unittypes = []
    if isinstance(boxres, (int,float)):
        boxres = [boxres * U.Mpc] * 3
        boxres_ax_unittypes = ['length'] * 3
    elif isinstance(boxres, U.Quantity):
        if U.get_physical_type(boxres.unit) != 'length':
            raise U.UnitConversionError('Incompatible dimension unit for input boxres')
        if isinstance(boxres.value, (int,float)):
            boxres = [boxres] * 3
            boxres_ax_unittypes = ['length'] * 3
        elif isinstance(boxres.value, NP.ndarray):
            try:
                boxres = boxres.to('Mpc') * NP.ones(3)
                boxres_ax_unittypes = ['length'] * 3
            except (U.UnitConversionError, ValueError) as exption:
                raise exption
        else:
            raise TypeError('Invalid type for input boxres')
    elif isinstance(boxres, list):
        if len(boxres) != 3:
            raise ValueError('Input boxres must be a 3-element list')
        for ax,scale in enumerate(boxres):
            if isinstance(scale, U.Quantity):
                boxres_ax_unittype = U.get_physical_type(scale.unit)
            elif scale is None:
                raise TypeError('Scale in axis {0:0d} found to be None'.format(ax))
            else:
                boxres_ax_unittype = 'dimensionless'
            if ax == 0:
                if boxres_ax_unittype not in ['dimensionless', 'frequency', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0} in input boxres'.format(ax))
            else:
                if boxres_ax_unittype not in ['angle', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input boxres'.format(ax))
            boxres_ax_unittypes += [boxres_ax_unittype]

    if not isinstance(rest_freq, U.Quantity):
        if isinstance(rest_freq, (int,float)):
            rest_freq = rest_freq * U.Hz
        else:
            raise TypeError('Input rest_freq must be a scalar')
    else:
        if U.get_physical_type(rest_freq.unit) != 'frequency':
            raise TypeError('Input rest_freq must have units of frequency')

    dz = None
    df = None
    ddlos = None

    if boxres_ax_unittypes[0] == 'dimensionless':
        dz = boxres[0]
        redshifts_in = zmin + dz * NP.arange(lightcone_cube.shape[0])
    elif boxres_ax_unittypes[0] == 'length':
        ddlos = boxres[0]
        d_los_min = cosmo100.comoving_distance(zmin)
        d_los = d_los_min + (cosmo.h*ddlos) * NP.arange(lightcone_cube.shape[0])
        redshifts_in = NP.asarray([cosmology.z_at_value(cosmo100.comoving_distance, dist) for dist in d_los])
    else:
        df = boxres[0]
        freq_max = rest_freq / (1+zmin)
        freqs_in = freq_max - df * NP.arange(lightcone_cube.shape[0])
        redshifts_in = rest_freq/freqs_in - 1

    eps_z = 1e-6
    if zbins is not None:
        if not isinstance(zbins, list):
            raise TypeError('Input zbins must be provided as a list')
    else:
        zbins = [[zmin-eps_z, redshifts_in.max()+eps_z]]

    for zbin in zbins:
        zind = NP.where(NP.logical_and(redshifts_in >= zbin[0], redshifts_in < zbin[1]))[0]
        zstr = '_z_{0[0]:.3f}-{0[1]:.3f}'.format(zbin)

        with h5py.File(outfile_prefix+zstr+'.hdf5', 'w') as fileobj:
            hdr_grp = fileobj.create_group('header')
            for ax,bres in enumerate(boxres):
                if boxres_ax_unittypes[ax] == 'dimensionless':
                    hdr_grp['boxres_{0:0d}'.format(ax)] = float(bres)
                    hdr_grp['boxres_{0:0d}'.format(ax)].attrs['units'] = ''
                elif boxres_ax_unittypes[ax] == 'length':
                    hdr_grp['boxres_{0:0d}'.format(ax)] = cosmo.h * bres.to('Mpc').value
                    hdr_grp['boxres_{0:0d}'.format(ax)].attrs['units'] = 'Mpc'
                elif boxres_ax_unittypes[ax] == 'frequency':
                    hdr_grp['boxres_{0:0d}'.format(ax)] = bres.to('Hz').value
                    hdr_grp['boxres_{0:0d}'.format(ax)].attrs['units'] = 'Hz'
                elif boxres_ax_unittypes[ax] == 'angle':
                    hdr_grp['boxres_{0:0d}'.format(ax)] = bres.to('arcmin').value
                    hdr_grp['boxres_{0:0d}'.format(ax)].attrs['units'] = 'arcmin'
                else:
                    raise TypeError('Unrecognized unit type in boxres in axis {0:0d}'.format(ax))
                    
            hdr_grp['zmin'] = zbin[0]
            cosmo_grp = fileobj.create_group('cosmo')
            cosmo_grp['H0'] = cosmo100.H0.value
            cosmo_grp['H0'].attrs['units'] = '{0}'.format(cosmo100.H0.unit)
            cosmo_grp['Om0'] = cosmo100.Om0
            cosmo_grp['Ode0'] = cosmo100.Ode0
            cosmo_grp['Ob0'] = cosmo100.Ob0
            cosmo_grp['w0'] = cosmo100.w(0.0)
            cosmo_grp['Tcmb0'] = cosmo100.Tcmb0.value
            cosmo_grp['Tcmb0'].attrs['units'] = cosmo100.Tcmb0.unit.to_string()
            # cosmo_grp['Tcmb0'].attrs['units'] = '{0}'.format(cosmo100.Tcmb0.unit)
            cube_dset = fileobj.create_dataset('cube', lightcone_cube[zind,:,:].shape, data=lightcone_cube[zind,:,:].si.value)
            cube_dset.attrs['units'] = lightcone_cube.si.unit.to_string()
            # monopole_dset = fileobj.create_dataset('monopole_temperature', monopole_Tb.shape, data=monopole_Tb)
            # monopole_dset.attrs['units'] = 'mK'

#################################################################################

def read_HDF5_lightcone_cube(infile):

    with h5py.File(infile, 'r') as fileobj:
        hdrinfo = {key: fileobj['header'][key].value for key in fileobj['header']}
        cosmoinfo = {key: fileobj['cosmo'][key].value for key in fileobj['cosmo']}
        lightcone_cube_units = fileobj['cube'].attrs['units']
        lightcone_cube = fileobj['cube'].value * U.Unit(lightcone_cube_units)
        if 'monopole_temperature' in fileobj:
            monopole_Tb = fileobj['monopole_temperature'] * U.Unit(lightcone_cube_units)
        else:
            monopole_Tb = None

        boxres = [hdrinfo['boxres_{0:0d}'.format(bi)] * U.Unit(fileobj['header']['boxres_{0:0d}'.format(bi)].attrs['units']) for bi in range(3)]
    zmin = hdrinfo['zmin']
    cosmo = cosmology.wCDM(cosmoinfo['H0'], cosmoinfo['Om0'], cosmoinfo['Ode0'], w0=cosmoinfo['w0'], Ob0=cosmoinfo['Ob0'], Tcmb0=cosmoinfo['Tcmb0']*U.K)
    
    return (lightcone_cube, monopole_Tb, boxres, zmin, cosmo)

#################################################################################

def smooth_resample_lightcone(lightcone, boxres_in, zmin,
                              cosmo=cosmo100_Planck15, zyx_scale=None,
                              resample=True):

    if not isinstance(lightcone.value, NP.ndarray):
        raise TypeError('Input lightcone must be a numpy array')
    lightcone = lightcone.si
    lc_unit = lightcone.si.unit

    if not isinstance(zmin, (int,float)):
        raise TypeError('Input zmin must be a scalar')

    boxres_ax_unittypes = []
    if isinstance(boxres_in, (int,float)):
        boxres_in = [boxres_in * U.Mpc] * 3
        boxres_ax_unittypes = ['length'] * 3
    elif isinstance(boxres_in, U.Quantity):
        if U.get_physical_type(boxres_in.unit) != 'length':
            raise U.UnitConversionError('Incompatible dimension unit for input boxres_in')
        if isinstance(boxres_in.value, (int,float)):
            boxres_in = [boxres_in] * 3
            boxres_ax_unittypes = ['length'] * 3
        elif isinstance(boxres_in.value, NP.ndarray):
            try:
                boxres_in = boxres_in.to('Mpc') * NP.ones(3)
                boxres_ax_unittypes = ['length'] * 3
            except (U.UnitConversionError, ValueError) as exption:
                raise exption
        else:
            raise TypeError('Invalid type for input boxres_in')
    elif isinstance(boxres_in, list):
        if len(boxres_in) != 3:
            raise ValueError('Input boxres_in must be a 3-element list')
        for ax,scale in enumerate(boxres_in):
            if isinstance(scale, U.Quantity):
                boxres_ax_unittype = U.get_physical_type(scale.unit)
            elif scale is None:
                raise TypeError('Scale in axis {0:0d} found to be None'.format(ax))
            else:
                boxres_ax_unittype = 'dimensionless'
            if ax == 0:
                if boxres_ax_unittype not in ['dimensionless', 'frequency', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0} in input boxres_in'.format(ax))
            else:
                if boxres_ax_unittype not in ['angle', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input boxres_in'.format(ax))
            boxres_ax_unittypes += [boxres_ax_unittype]

    if not isinstance(cosmo, cosmology.FLRW):
        raise TypeError('Input cosmo must be an instance of class astropy.cosmology.FLRW')

    exptions = []
    dz = None
    df = None
    ddlos = None

    if boxres_ax_unittypes[0] == 'dimensionless':
        dz = boxres_in[0]
        redshifts_in = zmin + dz * NP.arange(lightcone_cube.shape[0])
    elif boxres_ax_unittypes[0] == 'length':
        ddlos = boxres_in[0]
        d_los_min = cosmo.comoving_distance(zmin)
        d_los = d_los_min + (cosmo.h*ddlos) * NP.arange(lightcone_cube.shape[0])
        redshifts_in = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in d_los])
    else:
        df = boxres_in[0]
        if isinstance(rest_freq, U.Quantity):
            freq_max = rest_freq / (1+zmin)
        else:
            freq_max = rest_freq * U.Hz / (1+zmin)
        freqs_in = freq_max - df * NP.arange(lightcone_cube.shape[0])
        redshifts_in = rest_freq/freqs_in - 1

    if zyx_scale is None:
        warnings.warn('No smoothing scale provided. Returning without any changes.')
        return (lightcone, boxres_in)

    zyx_ax_unittypes = []
    if isinstance(zyx_scale, (int,float)):
        zyx_scale = [zyx_scale * U.Mpc] * 3
        zyx_ax_unittypes = ['length'] * 3
    elif isinstance(zyx_scale, U.Quantity):
        if isinstance(zyx_scale.value, (int,float)):
            try:
                zyx_scale = [zyx_scale.to('Mpc')] * 3
                zyx_ax_unittypes = ['length'] * 3
            except U.UnitConversionError as exption:
                raise exption
        elif isinstance(zyx_scale.value, NP.ndarray):
            try:
                zyx_scale = zyx_scale.to('Mpc') * NP.ones(3)
                zyx_ax_unittypes = ['length'] * 3
            except (U.UnitConversionError, ValueError) as exption:
                raise exption
        else:
            raise TypeError('Invalid type for input zyx_scale')
    elif isinstance(zyx_scale, list):
        if len(zyx_scale) != 3:
            raise ValueError('Input zyx_scale must be a 3-element list')
        for ax,scale in enumerate(zyx_scale):
            if isinstance(scale, U.Quantity):
                zyx_ax_unittype = U.get_physical_type(scale.unit)
            elif scale is None:
                zyx_ax_unittype = None
            else:
                zyx_ax_unittype = 'dimensionless'
            if ax == 0:
                if zyx_ax_unittype not in [None, 'dimensionless', 'frequency', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input zyx_scale'.format(ax))
            else:
                if zyx_ax_unittype not in [None, 'angle', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input zyx_scale'.format(ax))
            zyx_ax_unittypes += [zyx_ax_unittype]
    
    for ax in range(len(zyx_scale)):
        zyx_ax_unittype = ''
        boxres_ax_unittype = ''
        if zyx_scale[ax] is not None:
            if isinstance(boxres_in[ax], U.Quantity):
                boxres_ax_unittype = U.get_physical_type(boxres_in[ax].unit)
            else:
                boxres_ax_unittype = 'dimensionless'
            if isinstance(zyx_scale[ax], U.Quantity):
                zyx_ax_unittype = U.get_physical_type(zyx_scale[ax].unit)
            else:
                zyx_ax_unittype = 'dimensionless'
            if zyx_ax_unittype != boxres_ax_unittype:
                if ax == 0:
                    raise U.UnitConversionError('Units incompatible between inputs boxres_in and zyx_scale for axis={0:0d}'.format(ax))
                else:
                    if zyx_ax_unittype == 'length':
                        zyx_scale[ax] = zyx_scale[ax] / cosmo.kpc_comoving_per_arcmin(redshifts_in.max())
                    else:
                        zyx_scale[ax] = zyx_scale[ax] * cosmo.kpc_comoving_per_arcmin(zmin)

    if not isinstance(resample, bool):
        raise TypeError('Input resample must be boolean')

    zyx_scale_rms = [None] * 3
    for ax in range(len(boxres_in)):
        if zyx_scale[ax] is not None:
            zyx_scale_rms[ax] = zyx_scale[ax]/(2.0*NP.sqrt(2*NP.log(2.0))) # Get standard deviation from FWHM

    kernel3Darr = NP.ones(1).reshape(1,1,1)
    for ax,bxreso in enumerate(zyx_scale_rms):
        if bxreso is not None:
            gkern = CONV.Gaussian1DKernel((bxreso / boxres_in[ax]).decompose())
            shape_tuple = [1,1,1]
            shape_tuple[ax] = -1
            shape_tuple = tuple(shape_tuple)
            kernel3Darr = kernel3Darr * gkern.array.reshape(shape_tuple)
    kernel3Darr /= NP.sum(kernel3Darr)
    kernel3D = CONV.CustomKernel(kernel3Darr)

    if kernel3D.shape[0] == 1:
        lightcone_out = NP.empty(lightcone.shape, dtype=lightcone.value.dtype)
        if lightcone.shape[0] <= 10:
            chunksize = 1
        else:
            chunksize = NP.floor(lightcone.shape[0]/10).astype(NP.int)
        chunk_edges = NP.arange(0, lightcone.shape[0], chunksize)
        chunk_edges = NP.append(chunk_edges, lightcone.shape[0])
        n_chunks = chunk_edges.size - 1
        # progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} chunks '.format(n_chunks), PGB.ETA()], maxval=n_chunks).start()
        # for chunkind in range(n_chunks):
        #     ind = NP.arange(chunk_edges[chunkind], chunk_edges[chunkind+1])
        #     # lightcone_out_1[ind,:,:] = CONV.convolve(lightcone.si.value[ind,:,:], kernel3D, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate')
        #     lightcone_out_2[ind,:,:] = CONV.convolve_fft(lightcone.si.value[ind,:,:], kernel3D, boundary='fill', normalize_kernel=True, nan_treatment='interpolate', allow_huge=True)
        #     progress.update(chunkind+1)
        # progress.finish()
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} chunks '.format(n_chunks), PGB.ETA()], maxval=n_chunks).start()
        for chunkind in range(n_chunks):
            ind = NP.arange(chunk_edges[chunkind], chunk_edges[chunkind+1])
            lightcone_out[ind,:,:] = CONV.convolve(lightcone.si.value[ind,:,:], kernel3D, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate')
            # lightcone_out_2[ind,:,:] = CONV.convolve_fft(lightcone.si.value[ind,:,:], kernel3D, boundary='fill', normalize_kernel=True, nan_treatment='interpolate', allow_huge=True)
            progress.update(chunkind+1)
        progress.finish()
    else:
        lightcone_out = CONV.convolve(lightcone.si.value, kernel3D, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate')

    resample_factor = [1.0] * 3
    if resample:
        for ax in range(len(resample_factor)):
            if zyx_scale[ax] is not None:
                resample_factor[ax] = (zyx_scale[ax]/boxres_in[ax]).decompose().value
                eps = 1e-10
                if lightcone_out.shape[0] <= 10:
                    chunksize = 1
                else:
                    chunksize = NP.floor(lightcone_out.shape[0]/10).astype(NP.int)
                chunk_edges = NP.arange(0, lightcone_out.shape[0], chunksize)
                chunk_edges = NP.append(chunk_edges, lightcone_out.shape[0])
                n_chunks = chunk_edges.size - 1
                tmparr_list = []
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} chunks '.format(n_chunks), PGB.ETA()], maxval=n_chunks).start()
                for chunkind in range(n_chunks):
                    ind = NP.arange(chunk_edges[chunkind], chunk_edges[chunkind+1])
                    if resample_factor[ax] > 1.0:
                        tmparr_list += [DSP.downsampler(lightcone_out[ind,:,:], resample_factor[ax], axis=ax, method='interp', kind='linear', fill_value=NP.nan)]
                    elif resample_factor[ax] < 1.0:
                        tmparr_list += [DSP.upsampler(lightcone_out[ind,:,:], resample_factor[ax], axis=ax, kind='linear', fill_value=NP.nan)]
                    progress.update(chunkind+1)
                progress.finish()
                lightcone_out = NP.concatenate(tmparr_list, axis=0)
            
    lightcone_out = lightcone_out * lc_unit
    boxres_out = [boxres_in[ax] * resample_factor[ax] for ax in range(len(resample_factor))]
    return (lightcone_out, boxres_out)

#################################################################################

def lightcone_LOS_interpolation(lightcone, boxres_in, zmin, losinfo=None, 
                                kind='linear', rest_freq=1420405751.77*U.Hz, 
                                cosmo=cosmo100_Planck15):

    lcunit = 1.0
    if isinstance(lightcone, U.Quantity):
        lcunit = lightcone.unit
        lightcone = lightcone.value
    if not isinstance(lightcone, NP.ndarray):
        raise TypeError('Input lightcone must be a numpy array')

    if not isinstance(zmin, (int,float)):
        raise TypeError('Input zmin must be a scalar')

    if not isinstance(losinfo, dict):
        raise TypeError('Input losinfo must be a dictionary')

    loskeys = ['los_type', 'los_unit', 'min_los', 'los_increment', 'n_los']
    for loskey in loskeys:
        if loskey not in losinfo:
            raise KeyError('Key {0} not found in input losinfo'.format(loskey))
        if loskey in ['min_los', 'n_los', 'los_increment']:
            if not isinstance(losinfo[loskey], (int,float)):
                raise TypeError('Key {0} in input losinfo must hold a scalar valaue'.format(loskey))
        if loskey in ['los_type', 'los_unit']:
            if not isinstance(losinfo[loskey], str):
                raise TypeError('Key {0} in input losinfo must hold a string'.format(loskey))

    if losinfo['los_type'] not in ['frequency', 'length', 'dimensionless']:
        raise ValueError('Invalid los_type {0} specified in input losinfo'.format(losinfo['los_type']))

    if U.get_physical_type(U.Unit(losinfo['los_unit'])) != losinfo['los_type']:
        raise U.UnitConversionError('LOS axis unit {0} and unit type {1} incompatible'.format(losinfo['los_unit'], losinfo['los_type']))

    min_los = losinfo['min_los'] * U.Unit(losinfo['los_unit'])
    n_los = losinfo['n_los']
    los_increment = losinfo['los_increment'] * U.Unit(losinfo['los_unit'])
    los_out = min_los + los_increment * NP.arange(n_los)

    boxres_ax_unittypes = []
    if isinstance(boxres_in, (int,float)):
        boxres_in = [boxres_in * U.Mpc] * 3
        boxres_ax_unittypes = ['length'] * 3
    elif isinstance(boxres_in, U.Quantity):
        if U.get_physical_type(boxres_in.unit) != 'length':
            raise U.UnitConversionError('Incompatible dimension unit for input boxres_in')
        if isinstance(boxres_in.value, (int,float)):
            boxres_in = [boxres_in] * 3
            boxres_ax_unittypes = ['length'] * 3
        elif isinstance(boxres_in.value, NP.ndarray):
            try:
                boxres_in = boxres_in.to('Mpc') * NP.ones(3)
                boxres_ax_unittypes = ['length'] * 3
            except (U.UnitConversionError, ValueError) as exption:
                raise exption
        else:
            raise TypeError('Invalid type for input boxres_in')
    elif isinstance(boxres_in, list):
        if len(boxres_in) != 3:
            raise ValueError('Input boxres_in must be a 3-element list')
        for ax,scale in enumerate(boxres_in):
            if isinstance(scale, U.Quantity):
                boxres_ax_unittype = U.get_physical_type(scale.unit)
            elif scale is None:
                raise TypeError('Scale in axis {0:0d} found to be None'.format(ax))
            else:
                boxres_ax_unittype = 'dimensionless'
            if ax == 0:
                if boxres_ax_unittype not in ['dimensionless', 'frequency', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0} in input boxres_in'.format(ax))
            else:
                if boxres_ax_unittype not in ['angle', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input boxres_in'.format(ax))
            boxres_ax_unittypes += [boxres_ax_unittype]

    if not isinstance(kind, str):
        raise TypeError('Input kind must be a string')

    if not isinstance(cosmo, cosmology.FLRW):
        raise TypeError('Input cosmo must be an instance of class astropy.cosmology.FLRW')

    if losinfo['los_type'] == 'frequency':
        if U.get_physical_type(boxres_in[0].unit) == 'frequency': # freq -> freq
            freq_max = rest_freq / (1 + zmin)
            inlocs = freq_max - boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = los_out
            zout_min = rest_freq / outlocs.max() - 1
        if U.get_physical_type(boxres_in[0].unit) == 'dimensionless': # freq -> redshift
            inlocs = zmin + boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = rest_freq / los_out - 1
            zout_min = outlocs.min()
        if U.get_physical_type(boxres_in[0].unit) == 'length': # freq -> LOS distance
            d_los_min = cosmo.comoving_distance(zmin)
            inlocs = d_los_min + boxres_in[0] * NP.arange(lightcone.shape[0])
            redshifts_out = rest_freq / los_out - 1
            outlocs = cosmo.comoving_distance(redshifts_out.value)
            zout_min = redshifts_out.min()
    elif losinfo['los_type'] == 'length':
        if U.get_physical_type(boxres_in[0].unit) == 'frequency':
            freq_max = rest_freq / (1 + zmin)
            inlocs = freq_max - boxres_in[0] * NP.arange(lightcone.shape[0])
            
            redshifts_out = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in los_out])
            outlocs = rest_freq / (1 + redshifts_out)
            outlocs = NP.sort(outlocs)
            zout_min = redshifts_out.min()
        if U.get_physical_type(boxres_in[0].unit) == 'dimensionless':
            inlocs = zmin + boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in los_out])
            zout_min = outlocs.min()
        if U.get_physical_type(boxres_in[0].unit) == 'length':
            d_los_min = cosmo.comoving_distance(zmin)
            inlocs = d_los_min + boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = los_out
            zout_min = cosmology.z_at_value(cosmo.comoving_distance, min_los)
    elif losinfo['los_type'] == 'dimensionless':
        zout_min = min_los
        if U.get_physical_type(boxres_in[0].unit) == 'frequency':
            freq_max = rest_freq / (1 + zmin)
            inlocs = freq_max - boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = rest_freq / (1 + los_out)
            outlocs = NP.sort(outlocs)
        if U.get_physical_type(boxres_in[0].unit) == 'dimensionless':
            inlocs = zmin + boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = los_out
        if U.get_physical_type(boxres_in[0].unit) == 'length':
            d_los_min = cosmo.comoving_distance(zmin)
            inlocs = d_los_min + boxres_in[0] * NP.arange(lightcone.shape[0])
            outlocs = cosmo.comoving_distance(los_out.value)

    if isinstance(zout_min, U.Quantity):
        zout_min = float(zout_min)
    boxres_out = copy.deepcopy(boxres_in)
    boxres_out[0] = los_increment
    lightcone_out = OPS.interpolate_array(lightcone, inlocs.si.value, outlocs.si.value, axis=0, kind=kind) * lcunit
    return (lightcone_out, boxres_out, zout_min)

#################################################################################

def convert_lightcone_comoving_to_angular(lightcone, boxres_in, zmin, 
                                          rest_freq=CNST.rest_freq_HI*U.Hz,
                                          cosmo=cosmo100_Planck15, interp=None):

    lcunit = 1.0
    if isinstance(lightcone, U.Quantity):
        lcunit = lightcone.unit
        lightcone = lightcone.value
    if not isinstance(lightcone, NP.ndarray):
        raise TypeError('Input lightcone must be a numpy array')

    if not isinstance(zmin, (int,float)):
        raise TypeError('Input zmin must be a scalar')

    boxres_ax_unittypes = []
    if isinstance(boxres_in, (int,float)):
        boxres_in = [boxres_in * U.Mpc] * 3
        boxres_ax_unittypes = ['length'] * 3
    elif isinstance(boxres_in, U.Quantity):
        if U.get_physical_type(boxres_in.unit) != 'length':
            raise U.UnitConversionError('Incompatible dimension unit for input boxres_in')
        if isinstance(boxres_in.value, (int,float)):
            boxres_in = [boxres_in] * 3
            boxres_ax_unittypes = ['length'] * 3
        elif isinstance(boxres_in.value, NP.ndarray):
            try:
                boxres_in = boxres_in.to('Mpc') * NP.ones(3)
                boxres_ax_unittypes = ['length'] * 3
            except (U.UnitConversionError, ValueError) as exption:
                raise exption
        else:
            raise TypeError('Invalid type for input boxres_in')
    elif isinstance(boxres_in, list):
        if len(boxres_in) != 3:
            raise ValueError('Input boxres_in must be a 3-element list')
        for ax,scale in enumerate(boxres_in):
            if isinstance(scale, U.Quantity):
                boxres_ax_unittype = U.get_physical_type(scale.unit)
            elif scale is None:
                raise TypeError('Scale in axis {0:0d} found to be None'.format(ax))
            else:
                boxres_ax_unittype = 'dimensionless'
            if ax == 0:
                if boxres_ax_unittype not in ['dimensionless', 'frequency', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0} in input boxres_in'.format(ax))
            else:
                if boxres_ax_unittype not in ['angle', 'length']:
                    raise ValueError('Incompatible dimension unit for axis {0:0d} in input boxres_in'.format(ax))
            boxres_ax_unittypes += [boxres_ax_unittype]

    if (boxres_ax_unittypes[1] != 'length') or (boxres_ax_unittypes[2] != 'length'):
        raise TypeError('Box resolution in transverse directions must have units of length for conversion to angles')

    # if zbins is None:
    #     if nzbins is None:
    #         nzbins = 1
    #     else:
    #         if not isinstance(nzbins, int):
    #             raise TypeError('Input nzbins must be an integer')
    #         if nzbins <= 0:
    #             raise ValueError('Input nzbins must be positive')
    # else:
    #     if not isinstance(zbins, list):
    #         raise TypeError('Input zbins must be provided as a list')
    #     nzbins = None
    
    if not isinstance(rest_freq, U.Quantity):
        if isinstance(rest_freq, (int,float)):
            rest_freq = rest_freq * U.Hz
        else:
            raise TypeError('Input rest_freq must be a scalar')
    else:
        if U.get_physical_type(rest_freq.unit) != 'frequency':
            raise TypeError('Input rest_freq must have units of frequency')

    if not isinstance(cosmo, cosmology.FLRW):
        raise TypeError('Input cosmo must be an instance of class astropy.cosmology.FLRW')
    # cosmoinfo = {'Om0': cosmo.Om0, 'Ode0': cosmo.Ode0, 'h': cosmo.h, 'Ob0': cosmo.Ob0, 'w0': cosmo.w(0.0)}

    # outdir = parms['dirstruct']['outdir']
    # outfile_prefix = parms['dirstruct']['outfile_prefix']
    # if outfile_prefix is None:
    #     outfile = outdir + 'light_cone_sphangles'
    # elif isinstance(outfile_prefix, str):
    #     outfile = outdir + outfile_prefix + '_light_cone_sphangles'
    # else:
    #     raise TypeError('Output filename prefix must be set to None or a string')
    
    # if not isinstance(boxres.value, (int,float)):
    #     raise TypeError('Input boxres must be a scalar')

    # d_los_min = cosmo.comoving_distance(zmin)
    # d_los = d_los_min + boxres * NP.arange(lightcone.shape[0])
    # redshifts_in = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in d_los])

    if boxres_ax_unittypes[0] == 'dimensionless':
        dz = boxres_in[0]
        redshifts_in = zmin + dz * NP.arange(lightcone.shape[0])
    elif boxres_ax_unittypes[0] == 'length':
        ddlos = boxres_in[0]
        d_los_min = cosmo.comoving_distance(zmin)
        d_los = d_los_min + (cosmo.h*ddlos) * NP.arange(lightcone.shape[0])
        redshifts_in = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in d_los])
    else:
        df = boxres_in[0]
        if isinstance(rest_freq, U.Quantity):
            freq_max = rest_freq / (1+zmin)
        else:
            freq_max = rest_freq * U.Hz / (1+zmin)
        freqs_in = freq_max - df * NP.arange(lightcone.shape[0])
        freqs_in = NP.sort(freqs_in)
        redshifts_in = rest_freq/freqs_in - 1

    kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshifts_in.value)
    factor = kpc_per_arcmin / kpc_per_arcmin.max()

    boxres_out = copy.deepcopy(boxres_in)
    if interp is None:
        ind_los_center = NP.floor(0.5*redshifts_in.size).astype(int)
        for ax in [1,2]:
            boxres_out[ax] = boxres_in[ax] / kpc_per_arcmin[ind_los_center]
            boxres_out[ax] = boxres_out[ax].to('arcmin')
        return (lightcone*lcunit, boxres_out)
    elif not isinstance(interp, str):
        raise TypeError('Input interp must be a string')
    else:
        for ax in [1,2]:
            boxres_out[ax] = boxres_in[ax] / kpc_per_arcmin.max()
            boxres_out[ax] = boxres_out[ax].to('arcmin')

        xin = boxres_in[2] * NP.arange(lightcone.shape[2]) 
        yin = boxres_in[1] * NP.arange(lightcone.shape[1])
        
        xcent = 0.5 * xin.max()
        ycent = 0.5 * yin.max()
    
        xin -= xcent
        yin -= ycent
    
        xarr, yarr = NP.meshgrid(xin, yin)
    
        az = NP.arctan2(xarr, yarr)
    
        # theta_x = xarr / kpc_per_arcmin.max()
        # theta_y = yarr / kpc_per_arcmin.max()
    
        # za = NP.sqrt(xarr**2 + yarr**2) / kpc_per_arcmin.max()

        xout = xarr[NP.newaxis,:,:] * factor[:,NP.newaxis,NP.newaxis] # nz x ny x nx
        yout = yarr[NP.newaxis,:,:] * factor[:,NP.newaxis,NP.newaxis] # nz x ny x nx
        zyxout = NP.concatenate((yout[...,NP.newaxis].to('Mpc').value, xout[...,NP.newaxis].to('Mpc').value), axis=-1) * U.Mpc # nz x ny x nx x 2
        lightcone_out = NP.empty((redshifts_in.size,lightcone.shape[1],lightcone.shape[2]))

        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Redshifts '.format(redshifts_in.size), PGB.ETA()], maxval=redshifts_in.size).start()
        for i in range(redshifts_in.size):
            lightcone_out[i,...] = interpolate.interpn((yin.to('Mpc').value,xin.to('Mpc').value), lightcone[i,...], zyxout[i,...].to('Mpc').value, method=interp, bounds_error=False)
            progress.update(i+1)
        progress.finish()
    
        lightcone_out = lightcone_out * lcunit
        return (lightcone_out, boxres_out)
    
#################################################################################

def tile_lightcone(lightcone, zyx_factor=None, zyx_out=None, boxres=None, mirror_axes=None):

    lcunit = 1.0
    if isinstance(lightcone, U.Quantity):
        lcunit = lightcone.unit
        lightcone = lightcone.value
    if not isinstance(lightcone, NP.ndarray):
        raise TypeError('Input lightcone must be a numpy array')

    if (zyx_factor is None) and (zyx_out is None):
        warnings.warn('No tiling information specified. Returning input lightcone without modification')
        return lightcone*lcunit

    if (zyx_factor is not None) and (zyx_out is not None):
        raise ValueError('One and only one of zyx_factor or zyx_out must be specified')

    if zyx_factor is not None:
        if isinstance(zyx_factor, (int,float)):
            reps = zyx_factor + NP.zeros(3)
        elif isinstance(zyx_factor, (list, NP.ndarray)):
            if NP.asarray(zyx_factor).size != 3:
                raise ValueError('Input zyx_factor must be a 3-element array')
            reps = []
            for factor in zyx_factor:
                if factor is None:
                    reps += [1]
                elif isinstance(factor, (int,float)):
                    reps += [factor]
                else:
                    raise TypeError('Values in zyx_factor must be scalars')
        else:
            raise TypeError('Input zyx_factor must be NoneType, list or numpy array')

    else:
        reps = []
        if boxres is None:
            raise TypeError('Input boxres must be specified if input zyx_out is to be used')

        boxres_ax_unittypes = []
        if isinstance(boxres, (int,float)):
            boxres = [boxres * U.Mpc] * 3
            boxres_ax_unittypes = ['length'] * 3
        elif isinstance(boxres, U.Quantity):
            if U.get_physical_type(boxres.unit) != 'length':
                raise U.UnitConversionError('Incompatible dimension unit for input boxres')
            if isinstance(boxres.value, (int,float)):
                boxres = [boxres] * 3
                boxres_ax_unittypes = ['length'] * 3
            elif isinstance(boxres.value, NP.ndarray):
                try:
                    boxres = boxres.to('Mpc') * NP.ones(3)
                    boxres_ax_unittypes = ['length'] * 3
                except (U.UnitConversionError, ValueError) as exption:
                    raise exption
            else:
                raise TypeError('Invalid type for input boxres')
        elif isinstance(boxres, list):
            if len(boxres) != 3:
                raise ValueError('Input boxres must be a 3-element list')
            for ax,scale in enumerate(boxres):
                if isinstance(scale, U.Quantity):
                    boxres_ax_unittype = U.get_physical_type(scale.unit)
                elif scale is None:
                    raise TypeError('Scale in axis {0:0d} found to be None'.format(ax))
                else:
                    boxres_ax_unittype = 'dimensionless'
                if ax == 0:
                    if boxres_ax_unittype not in ['dimensionless', 'frequency', 'length']:
                        raise ValueError('Incompatible dimension unit for axis {0} in input boxres'.format(ax))
                else:
                    if boxres_ax_unittype not in ['angle', 'length']:
                        raise ValueError('Incompatible dimension unit for axis {0:0d} in input boxres'.format(ax))
                boxres_ax_unittypes += [boxres_ax_unittype]

        zyx_ax_unittypes = []
        if isinstance(zyx_out, (int,float)):
            zyx_out = [zyx_out * U.Mpc] * 3
            zyx_ax_unittypes = ['length'] * 3
        elif isinstance(zyx_out, U.Quantity):
            if U.get_physical_type(zyx_out.unit) != 'length':
                raise U.UnitConversionError('Incompatible dimension unit for input zyx_out')
            if isinstance(zyx_out.value, (int,float)):
                zyx_out = [zyx_out] * 3
                zyx_ax_unittypes = ['length'] * 3
            elif isinstance(zyx_out.value, NP.ndarray):
                try:
                    zyx_out = zyx_out.to('Mpc') * NP.ones(3)
                    zyx_ax_unittypes = ['length'] * 3
                except (U.UnitConversionError, ValueError) as exption:
                    raise exption
            else:
                raise TypeError('Invalid type for input zyx_out')
        elif isinstance(zyx_out, list):
            if len(zyx_out) != 3:
                raise ValueError('Input zyx_out must be a 3-element list')
            for ax,scale in enumerate(zyx_out):
                if isinstance(scale, U.Quantity):
                    zyx_ax_unittype = U.get_physical_type(scale.unit)
                elif scale is None:
                    zyx_ax_unittype = None
                else:
                    zyx_ax_unittype = 'dimensionless'
                zyx_ax_unittypes += [zyx_ax_unittype]

        for ax in range(len(zyx_out)):
            if zyx_out[ax] is not None:
                if zyx_ax_unittypes[ax] != boxres_ax_unittypes[ax]:
                    raise U.UnitConversionError('Values in zyx_out must have same unit type as those in boxres')
                reps += [(zyx_out[ax]/(lightcone.shape[ax]*boxres[ax])).decompose().value]
            else:
                reps += [1]
            
    reps = NP.asarray(reps).ravel().astype(float)
    lightcone_out = OPS.tile(lightcone, reps, mirror_axes=mirror_axes)
    return lightcone_out * lcunit

#################################################################################

def write_lightcone_to_sphangles(angular_lightcone, theta, phi, redshifts,
                                 cosmoinfo, rest_freq, outfile):

    if not isinstance(angular_lightcone, U.Quantity):
        raise TypeError('Input angular_lightcone must be an instance of class astropy.units.Quantity')
    if not isinstance(theta, U.Quantity):
        raise TypeError('Input theta must be an instance of class astropy.units.Quantity')
    if not isinstance(phi, U.Quantity):
        raise TypeError('Input phi must be an instance of class astropy.units.Quantity')
    if not isinstance(rest_freq, U.Quantity):
        raise TypeError('Input rest_freq must be an instance of class astropy.units.Quantity')
    if not isinstance(redshifts, NP.ndarray):
        raise TypeError('Input redshifts must be a numpy array')

    if not isinstance(cosmoinfo, dict):
        raise TypeError('Input cosmoinfo must be a dictionary')

    if angular_lightcone.ndim != 3:
        raise ValueError('Input angular_lightcone does not have appropriate dimensions')
    if theta.ndim != 2:
        raise ValueError('Input theta does not have appropriate dimensions')
    if phi.ndim != 2:
        raise ValueError('Input phi does not have appropriate dimensions')
    if theta.shape != phi.shape:
        raise ValueError('Inputs theta and phi must have the same shape')
    if angular_lightcone.shape != (redshifts.size, theta.shape[0], theta.shape[1]):
        raise ValueError('Inputs angular_lightcone has shape incompatible with that expected from the shapes of redshifts, theta and phi')

    with h5py.File(outfile, 'w') as fileobj:
        hdr_grp = fileobj.create_group('header')
        hdr_grp['rest_freq'] = rest_freq.to('Hz').value
        hdr_grp['rest_freq'].attrs['units'] = 'Hz'

        cosmo_grp = fileobj.create_group('cosmology')
        for key in cosmoinfo:
            cosmo_grp[key] = cosmoinfo[key]
        
        lightcone_grp = fileobj.create_group('lightcone')
        cube_dset = lightcone_grp.create_dataset('cube', angular_lightcone.shape, data=angular_lightcone.to('mK').value, chunks=(1,theta.shape[0],theta.shape[1]), compression='gzip', compression_opts=9)
        cube_dset.attrs['units'] = 'mK'
        lightcone_grp['theta'] = theta.to('radian').value
        lightcone_grp['theta'].attrs['units'] = 'radian'
        lightcone_grp['phi'] = phi.to('radian').value
        lightcone_grp['phi'].attrs['units'] = 'radian'
        lightcone_grp['z'] = redshifts

#################################################################################

def read_lightcone_sphangles(infile):

    with h5py.File(infile, 'r') as fileobj:
        hdr_grp = fileobj['header']
        rest_freq = U.Quantity(hdr_grp['rest_freq'].value, hdr_grp['rest_freq'].attrs['units'])

        cosmo_grp = fileobj['cosmology']
        cosmoinfo = {cosmo_grp[key].value for key in cosmo_grp}

        lc_grp = fileobj['lightcone']
        lc_cube = U.Quantity(lc_grp['cube'].value, lc_grp['cube'].attrs['units'])
        theta = U.Quantity(lc_grp['theta'].value, lc_grp['theta'].attrs['units'])
        phi = U.Quantity(lc_grp['phi'].value, lc_grp['phi'].attrs['units'])
        redshifts = lc_grp['z'].value

    return (lc_cube, theta, phi, redshifts, cosmoinfo)

#################################################################################

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to tile cosmological lightcone cubes to cubes in spherical angles (Alt-Az)')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', required=False, type=str, default='/data3/t_nithyanandan/codes/mine/python/projects/cosmotiling/lightcone_to_sphangles_parms.yaml', help='Config file for processing cosmological lightcone cubes')
    # input_group.add_argument('-p', '--parmsfile', dest='parmsfile', required=False, type=str, default=astroutils_path+'examples/cosmotile/lightcone_to_sphshells_parms.yaml', help='Config file for processing cosmological lightcone cubes')
    
    args = vars(parser.parse_args())
    parmsfile = args['parmsfile']
    with open(parmsfile, 'r') as pfile:
        parms = yaml.safe_load(pfile)

    cosmoparms = parms['sim']['cosmo']
    if cosmoparms['name'] is None:
        cosmo = None
    elif cosmoparms['name'].lower() == 'custom':
        h = cosmoparms['h']
        H0 = 100.0 * h
        Om0 = cosmoparms['Om0']
        Ode0 = cosmoparms['Ode0']
        if Ode0 is None:
            Ode0 = 1.0 - Om0
        Ob0 = cosmoparms['Ob0'] / h**2
        w0 = cosmoparms['w0']
        Tcmb0 = cosmoparms['Tcmb0']
        cosmo = cosmology.wCDM(H0, Om0, Ode0, w0=w0, Ob0=Ob0, Tcmb0=Tcmb0)
    elif cosmoparms['name'].lower() == 'wmap9':
        cosmo = cosmology.WMAP9
    elif cosmoparms['name'].lower() == 'planck15':
        cosmo = cosmology.Planck15
    else:
        raise ValueError('{0} preset not currently accepted for cosmology'.format(cosmoparms['name'].lower()))
    units_in = parms['sim']['units_in']
    units_in['frequency'] = 'Hz'
    rest_freq = parms['sim']['rest_freq']
    if rest_freq is not None:
        rest_freq = rest_freq * U.Hz
    else:
        rest_freq = CNST.rest_freq_HI * U.Hz

    actions = [key for key in parms['actions'] if parms['actions'][key]['act']]
    if 'process_raw_LC' in actions:
        lightcone_cube, boxres, zmin = read_raw_lightcone_cube(parms['actions']['process_raw_LC'])
        zbins = parms['actions']['process_raw_LC']['proc']['zbins']
        outparms = parms['actions']['process_raw_LC']['output']
        outdir = outparms['outdir']
        outfile_prefix = outparms['outfile_prefix']
        if outfile_prefix is None:
            outfile_prefix = outdir + 'light_cone'
        elif isinstance(outfile_prefix, str):
            outfile_prefix = outdir + outfile_prefix + '_light_cone'
        else:
            raise TypeError('Output filename prefix must be set to None or a string')
        write_lightcone_cube(lightcone_cube*U.Unit(units_in['boxval']), boxres*U.Unit(units_in['boxdim']), zmin, outfile_prefix, zbins=zbins, cosmo=cosmo)

    # Smooth and resample lightcone (along transverse direction)
    if 'smooth_resample' in actions:
        input_parms_LC_HDF5 = parms['actions']['smooth_resample']['input']
        indir_LC_HDF5 = input_parms_LC_HDF5['indir']
        infile_prefix_LC_HDF5 = input_parms_LC_HDF5['infile_prefix']
        outparms_smooth_resample = parms['actions']['smooth_resample']['output']
        outdir_smooth_resample = outparms_smooth_resample['outdir']
        outfile_prefix = outparms_smooth_resample['outfile_prefix']

        lightcone_cube, monopole_Tb, boxres, zmin, cosmo = read_HDF5_lightcone_cube(indir_LC_HDF5+infile_prefix_LC_HDF5+'.hdf5')
        
        zyx_scale = parms['actions']['smooth_resample']['proc']['zyx_scale']
        zyx_units = parms['actions']['smooth_resample']['proc']['zyx_units']

        if zyx_scale is not None:
            if isinstance(zyx_scale, (int,float)):
                if zyx_units is None:
                    zyx_scale = zyx_scale * U.Mpc
                elif isinstance(zyx_units, str):
                    if U.get_physical_type(U.Unit(zyx_units)) == 'length':
                        zyx_scale = zyx_scale * U.Unit(zyx_units)
                    else:
                        raise U.UnitConversionError('zyx_units must be a unit of length')
                else:
                    raise TypeError('zyx_units must same dimensions as zyx_scale')
            elif isinstance(zyx_scale, list):
                if len(zyx_scale) != 3:
                    raise ValueError('Input zyx_scale must be a three element list')
                if zyx_units is None:
                    zyx_units = ['Mpc'] * 3
                if not isinstance(zyx_units, list):
                    raise TypeError('Input zyx_units must have same type as zyx_scale, a 3-element list')
                for ax in range(len(zyx_scale)):
                    if zyx_scale[ax] is not None:
                        if not isinstance(zyx_scale[ax], (int,float)):
                            raise TypeError('Values in input zyx_scale must be a scalar')
                        zyx_scale[ax] = zyx_scale[ax] * U.Unit(zyx_units[ax])

        lightcone_cube_smoothed, boxres_smoothed = smooth_resample_lightcone(lightcone_cube, boxres, zmin, cosmo=cosmo, zyx_scale=zyx_scale, resample=True)

        if outfile_prefix is None:
            outfile_prefix = 'smooth_resampled_lightcone'
        elif not isinstance(outfile_prefix, str):
            raise TypeError('Output filename prefix must be set to None or a string')
        outfile_prefix = outdir_smooth_resample + outfile_prefix + '_{0:.1f}_{1:.1f}_{2:.1f}'.format(boxres_smoothed[0].value, boxres_smoothed[2].value, boxres_smoothed[2].value)

        write_lightcone_cube(lightcone_cube_smoothed, boxres_smoothed, zmin, outfile_prefix, zbins=None, cosmo=cosmo)

    # LOS interpolation

    if 'los_interp' in actions:
        los_interp_parms = parms['actions']['los_interp']

        input_parms_LC_HDF5 = los_interp_parms['input']
        indir_LC_HDF5 = input_parms_LC_HDF5['indir']
        infile_prefix_LC_HDF5 = input_parms_LC_HDF5['infile_prefix']
        outparms_los_interp = los_interp_parms['output']
        outdir_los_interp = outparms_los_interp['outdir']
        outfile_prefix = outparms_los_interp['outfile_prefix']

        los_interp_proc_parms = los_interp_parms['proc']
        los_info = los_interp_proc_parms['los_info']

        PDB.set_trace()
        lightcone_cube_smoothed, monopole_Tb, boxres_smoothed, zmin, cosmo = read_HDF5_lightcone_cube(indir_LC_HDF5+infile_prefix_LC_HDF5+'.hdf5')

        lightcone_cube_specinterp, boxres_specinterp, zmin_specinterp = lightcone_LOS_interpolation(lightcone_cube_smoothed, boxres_smoothed, zmin, losinfo=los_info, kind='cubic', cosmo=cosmo)

        if outfile_prefix is None:
            outfile_prefix = 'LOS_interpolated_lightcone'
        elif not isinstance(outfile_prefix, str):
            raise TypeError('Output filename prefix must be set to None or a string')
        outfile_prefix = outdir_los_interp + outfile_prefix + '_{0:0d}x{1:.1f}{2}'.format(los_info['n_los'], los_info['los_increment'], los_info['los_unit'])

        write_lightcone_cube(lightcone_cube_specinterp, boxres_specinterp, zmin_specinterp, outfile_prefix, zbins=None, cosmo=cosmo)

    # Convert transverse axes of lightcones from comoving distances to
    # angles

    if 'LC_to_angle' in actions:

        lc_to_angle_parms = parms['actions']['LC_to_angle']

        input_parms_LC_HDF5 = lc_to_angle_parms['input']
        indir_LC_HDF5 = input_parms_LC_HDF5['indir']
        infile_prefix_LC_HDF5 = input_parms_LC_HDF5['infile_prefix']
        outparms_LC_to_angle = lc_to_angle_parms['output']
        outdir_LC_to_angle = outparms_LC_to_angle['outdir']
        outfile_prefix = outparms_LC_to_angle['outfile_prefix']

        lc_to_angle_proc_parms = lc_to_angle_parms['proc']
        interp_method = lc_to_angle_proc_parms['interp']
        if interp_method is not None:
            if not isinstance(interp_method, str):
                raise TypeError('Input interp must be a string')

        PDB.set_trace()
        lightcone_cube_specinterp, monopole_Tb, boxres_specinterp, zmin_specinterp, cosmo = read_HDF5_lightcone_cube(indir_LC_HDF5+infile_prefix_LC_HDF5+'.hdf5')

        angular_lightcone, boxres_angular = convert_lightcone_comoving_to_angular(lightcone_cube_specinterp, boxres_specinterp, zmin_specinterp, rest_freq=rest_freq, cosmo=cosmo, interp=interp_method)
        
        if outfile_prefix is None:
            outfile_prefix = 'angular_lightcone'
        elif not isinstance(outfile_prefix, str):
            raise TypeError('Output filename prefix must be set to None or a string')
        outfile_prefix = outdir_LC_to_angle + outfile_prefix + '_{0:.1f}{1}x{2:.1f}{3}x{4:.1f}{5}'.format(boxres_angular[0].value, boxres_angular[0].unit.to_string(), boxres_angular[1].to('arcmin').value, boxres_angular[1].to('arcmin').unit.to_string(), boxres_angular[2].to('arcmin').value, boxres_angular[2].to('arcmin').unit.to_string())

        write_lightcone_cube(angular_lightcone, boxres_angular, zmin_specinterp, outfile_prefix, zbins=None, cosmo=cosmo)

    # Transverse tiling

    if 'tiling' in actions:
        
        tiling_parms = parms['actions']['tiling']

        input_parms_LC_HDF5 = tiling_parms['input']
        indir_LC_HDF5 = input_parms_LC_HDF5['indir']
        infile_prefix_LC_HDF5 = input_parms_LC_HDF5['infile_prefix']
        outparms_tiling = tiling_parms['output']
        outdir_tiling = outparms_tiling['outdir']
        outfile_prefix = outparms_tiling['outfile_prefix']

        tiling_proc_parms = tiling_parms['proc']
        zyx_factor = tiling_proc_parms['zyx_factor']
        zyx_out = tiling_proc_parms['zyx_out']
        zyx_units = tiling_proc_parms['zyx_units']
        mirror_axes = tiling_proc_parms['mirror_axes']

        if zyx_out is not None:
            if not isinstance(zyx_out, list):
                raise TypeError('Input zyx_out must be a list')
            if not isinstance(zyx_units, list):
                raise TypeError('Input zyx_units must be a list')
            if len(zyx_out) != 3:
                raise ValueError('Input zyx_out must contain 3 elements')
            if len(zyx_out) != len(zyx_units):
                raise ValueError('Inputs zyx_out and zyx_units must have same length')
            for ax in range(len(zyx_out)):
                if zyx_out[ax] is not None:
                    zyx_out[ax] = zyx_out[ax] * U.Unit(zyx_units[ax])

        PDB.set_trace()
        angular_lightcone, monopole_Tb, angular_boxres, angular_zmin, cosmo = read_HDF5_lightcone_cube(indir_LC_HDF5+infile_prefix_LC_HDF5+'.hdf5')

        do_tiling = (zyx_factor is None) + (zyx_out is None)
        if do_tiling:
            lightcone_tiled = tile_lightcone(angular_lightcone, zyx_factor=zyx_factor, zyx_out=zyx_out, boxres=angular_boxres, mirror_axes=mirror_axes)
            
        if outfile_prefix is None:
            outfile_prefix = 'tiled_lightcone'
        elif not isinstance(outfile_prefix, str):
            raise TypeError('Output filename prefix must be set to None or a string')
        outfile_prefix = outdir_tiling + outfile_prefix + '_{0:0d}x{1:.1f}{2}_{3:0d}x{4:.1f}{5}_{6:0d}x{7:.1f}{8}'.format(lightcone_tiled.shape[0], angular_boxres[0].value, angular_boxres[0].unit.to_string(), lightcone_tiled.shape[1], angular_boxres[1].to('arcmin').value, angular_boxres[1].to('arcmin').unit.to_string(), lightcone_tiled.shape[2], angular_boxres[2].to('arcmin').value, angular_boxres[2].to('arcmin').unit.to_string())

        write_lightcone_cube(lightcone_tiled, angular_boxres, angular_zmin, outfile_prefix, zbins=None, cosmo=cosmo)

    # Convert lightcone to observer's cube
    if 'LC_to_obscube' in actions:
        
        lc_to_obscube_parms = parms['actions']['LC_to_obscube']
        input_parms_LC_HDF5 = lc_to_obscube_parms['input']
        indir_LC_HDF5 = input_parms_LC_HDF5['indir']
        infile_prefix_LC_HDF5 = input_parms_LC_HDF5['infile_prefix']
        outparms_lc_to_obscube = lc_to_obscube_parms['output']
        outdir_lc_to_obscube = outparms_lc_to_obscube['outdir']
        outfile_prefix = outparms_lc_to_obscube['outfile_prefix']

        tiled_lightcone, monopole_Tb, tiled_boxres, tiled_zmin, cosmo = read_HDF5_lightcone_cube(indir_LC_HDF5+infile_prefix_LC_HDF5+'.hdf5')

        if outfile_prefix is None:
            outfile_prefix = 'tiled_lightcone'
        elif not isinstance(outfile_prefix, str):
            raise TypeError('Output filename prefix must be set to None or a string')
        outfile_prefix = outdir_lc_to_obscube + outfile_prefix + '_{0:0d}x{1:.1f}{2}_{3:0d}x{4:.1f}{5}_{6:0d}x{7:.1f}{8}'.format(tiled_lightcone.shape[0], tiled_boxres[0].value, tiled_boxres[0].unit.to_string(), tiled_lightcone.shape[1], tiled_boxres[1].to('arcmin').value, tiled_boxres[1].to('arcmin').unit.to_string(), tiled_lightcone.shape[2], tiled_boxres[2].to('arcmin').value, tiled_boxres[2].to('arcmin').unit.to_string())

        lc_to_obscube_proc_parms = lc_to_obscube_parms['proc']
        wcsinfo = lc_to_obscube_proc_parms['wcsinfo']
        hdrinfo = lc_to_obscube_proc_parms['hdrinfo']

        obsdate = hdrinfo['obsdate']
        latitude = hdrinfo['latitude'] * U.deg
        longitude = hdrinfo['longitude'] * U.deg
        height = hdrinfo['altitude'] * U.m
        telescope_location = EarthLocation(lon=longitude, lat=latitude, height=height)
            
        tobj0 = Time(obsdate.replace('/', '-'), format='iso', scale='utc', location=telescope_location)
        ra_deg_ref = wcsinfo['RA_ref']
        dec_deg_ref = wcsinfo['dec_ref']
        equinox_ref = hdrinfo['equinox']
        radec_ref = SkyCoord(ra=ra_deg_ref*U.deg, dec=dec_deg_ref*U.deg, frame='icrs', equinox='J{0:.1f}'.format(equinox_ref))
        radec_ref_on_obsdate = radec_ref.transform_to(FK5(equinox=tobj0))
        radec_center_on_obsdate = SkyCoord(ra=radec_ref_on_obsdate.ra, dec=latitude, frame='fk5', equinox=tobj0)
        radec_center_on_equinox = radec_center_on_obsdate.icrs
        lst_of_radec_ref_on_obsdate = radec_ref_on_obsdate.ra.to('hourangle')
        jd_obs = ET.julian_date_from_LAST(lst_of_radec_ref_on_obsdate.hour, tobj0.jd, longitude.to('deg').value/15.0) # Julian date at beginning of observation
        jd_obs = jd_obs[0]
        tobj_obs = Time(jd_obs, format='jd', scale='utc', location=telescope_location) # Time object at the time of observation
        lst_obs = tobj_obs.sidereal_time('apparent').deg # Update LST init
        
        freqs = wcsinfo['freq_ref']*U.Hz + tiled_boxres[0] * NP.arange(tiled_lightcone.shape[0])
        pix_solid_angle = tiled_boxres[1] * tiled_boxres[2]
        fluxdensity = tiled_lightcone * (2.0* FCNST.k_B * freqs.reshape(-1,1,1)**2 / FCNST.c**2) * pix_solid_angle.to('', equivalencies=U.dimensionless_angles())
        bunit = hdrinfo['bunit']
        if bunit.lower() not in ['k', 'jy']:
            raise ValueError('Invalid value in bunit')
        
        w = wcs.WCS(naxis=3)
        w.wcs.crpix = [0.5*tiled_lightcone.shape[2], 0.5*tiled_lightcone.shape[1], 1]
        w.wcs.cdelt = NP.array([-tiled_boxres[2].to('deg').value, tiled_boxres[1].to('deg').value, tiled_boxres[0].to('Hz').value])
        w.wcs.crval = [radec_center_on_equinox.ra.deg, radec_center_on_equinox.dec.deg, wcsinfo['freq_ref']]
        w.wcs.cunit = ['deg', 'deg', 'Hz']
        w.wcs.ctype = ['RA---{0}'.format(wcsinfo['projection'].upper()), 'DEC--{0}'.format(wcsinfo['projection'].upper()), 'FREQ']

        hdr = w.to_header()
        hdr['SIMPLE'] = True
        hdr['EXTEND'] = True
        hdr['NAXIS'] = len(tiled_boxres)
        for ax in range(len(tiled_boxres)):
            hdr['NAXIS{0}'.format(ax+1)] = tiled_lightcone.shape[-(ax+1)]
        hdr['RADESYS'] = hdrinfo['radesys']
        hdr['EQUINOX'] = hdrinfo['equinox']
        hdr['SPECSYS'] = (hdrinfo['specsys'], 'Spectral reference frame')
        if bunit.lower() == 'k':
            hdr['BUNIT'] = (tiled_lightcone.to('K').unit.to_string(), 'Brightness (pixel) unit')
            hdr['BTYPE'] = 'Temperature'
        else:
            hdr['BUNIT'] = (fluxdensity.to('Jy').unit.to_string()+'/PIXEL', 'Brightness (pixel) unit')
            hdr['BTYPE'] = 'Intensity'
        hdr['LATPOLE'] = 0.0
        hdr['LONPOLE'] = 180.0
        hdr['RESTFRQ'] = (rest_freq.to('Hz').value, 'Rest Frequency (Hz)')
        hdr['DATE-OBS'] = tobj_obs.isot
        hdr['TELESCOP'] = hdrinfo['telescope']

        wcs_final = wcs.WCS(hdr)

        if outparms_lc_to_obscube['fits']:
            if bunit.lower() == 'k':
                hdu = fits.PrimaryHDU(tiled_lightcone.to('K').value)
            else:
                hdu = fits.PrimaryHDU(fluxdensity.to('Jy').value)
            hdu.header = hdr
            hdu.writeto(outfile_prefix+'.fits', output_verify='fix', overwrite=True)
        PDB.set_trace()
        if outparms_lc_to_obscube['skymodel']:
            theta_x = (NP.arange(tiled_lightcone.shape[2]) - 0.5*tiled_lightcone.shape[2]) * tiled_boxres[2]
            theta_y = (NP.arange(tiled_lightcone.shape[1]) - 0.5*tiled_lightcone.shape[1]) * tiled_boxres[1]
            za = NP.sqrt(theta_x.reshape(1,-1)**2 + theta_y.reshape(-1,1)**2)
            az = NP.arctan2(theta_y.to('deg').value.reshape(-1,1), theta_x.to('deg').value.reshape(1,-1)) * U.rad
            alt = 90 * U.deg - za

            altaz = AltAz(alt=alt, az=az, obstime=tobj_obs, location=telescope_location, obswl=NP.mean(FCNST.c/freqs))
            radec = altaz.transform_to(ICRS)

            catlabel = '21cmfast_FaintGalaxies_fiducial'
            flux_unit = 'Jy'
            spec_type = 'spectrum'
            spec_parms = {}
            skymod_init_parms = {'name': catlabel, 'frequency': freqs.to('Hz').value, 'location': NP.hstack((radec.ra.deg.reshape(-1,1), radec.dec.deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'spectrum': fluxdensity.to('Jy').value.reshape(freqs.size,-1).T, 'epoch': 'J{0:.1f}'.format(equinox_ref), 'coords': 'radec', 'src_shape': NP.hstack((tiled_boxres[1].to('deg').value+NP.zeros(radec.size).reshape(-1,1), tiled_boxres[1].to('deg').value+NP.zeros(radec.size).reshape(-1,1),NP.zeros(radec.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}
            skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

            skymod.save(outfile_prefix, fileformat='hdf5')

    if 'sphang_to_hpx' in actions:
        lc_sphang, theta, phi, redshifts, cosmoinfo = read_lightcone_sphangles(parms['actions']['sphang_to_hpx']['infile'])
        use_specparms = parms['actions']['sphang_to_hpx']['proc']['use_specparms']
        if specparms:
            freqs = parms['actions']['sphang_to_hpx']['proc']['freqs']
            freq_units = parms['actions']['sphang_to_hpx']['proc']['freq_units']
            if not isinstance(freq_units, str):
                raise TypeError('Input freq_units must be a string')
            if freq_units.lower() not in ['hz', 'mhz', 'ghz']:
                raise ValueError('Input freq_units must be set to "Hz", "MHz" or "GHz"')
            if freqs is not None:
                if not isinstance(freqs, NP.ndarray):
                    raise TypeError('Input freqs must be a numpy array')
            else:
                f0 = parms['actions']['sphang_to_hpx']['proc']['f0']
                freq_res = parms['actions']['sphang_to_hpx']['proc']['freq_res']
                nchan = parms['actions']['sphang_to_hpx']['proc']['nchan']
                freqs = (NP.arange(nchan) - 0.5 * nchan) * freq_res + f0
            freqs = U.Quantity(freqs, freq_units)
            rest_freq = U.Quantity(parms['actions']['sphang_to_hpx']['proc']['rest_freq'], freq_units)
            zout = rest_freq / freqs - 1

            if zout.size > redshifts.size:
                transverse_op = 'first'
            else:
                transverse_op = 'second'
        else:
            transverse_op = 'first'
                
        hpxmap = GEOM.project_sphangles_to_healpix(theta.to('radian').value.ravel(), phi.to('radian').value.ravel(), lc_sphang.to('mK').value[0,:,:].ravel(), parms['actions']['sphang_to_hpx']['proc']['nside'], angunits='radians', interp_method=parms['actions']['sphang_to_hpx']['proc']['interp_method'], fill_value=0.0, max_in_map=None, positivity=False)
        hpxmap = hpxmap * U.mK

    indir = parms['dirstruct']['indir']
    infile_prefix = parms['dirstruct']['infile_prefix']
    if infile_prefix is None:
        infile_prefix = ''
    infile_suffix = parms['dirstruct']['infile_suffix']
    if infile_suffix is None:
        infile_suffix = ''
    save = parms['dirstruct']['write']
    outdir = parms['dirstruct']['outdir']
    outfile_prefix = parms['dirstruct']['outfile_prefix']
    if outfile_prefix is None:
        outfile = outdir + 'light_cone_sphangles'
    elif isinstance(outfile_prefix, str):
        outfile = outdir + outfile_prefix + '_light_cone_sphangles'
    else:
        raise TypeError('Output filename prefix must be set to None or a string')

    cube_source = parms['sim']['source']
    nzbins = parms['output']['nzbins']
    if nzbins is not None:
        if not isinstance(nzbins, int):
            raise TypeError('nzbins must be an integer')
        if nzbins < 1:
            warnings.warn('nzbins not found to be positive. Setting it to 1.')
            nzbins = 1
    else:
        nzbins = 1
        
    rest_freq = parms['output']['rest_freq'] * U.Hz
    # nside = parms['output']['nside']
    if not isinstance(sim_units, str):
        raise TypeError('Input sim_units must be a string')
    if sim_units not in ['mK', 'K']:
        raise ValueError('Supported sim_units are "mK" and "K"')

    if sim_units == 'mK':
        cube_units = U.mK
    else:
        cube_units = U.K
    # is_healpix = False
    # if nside is not None:
    #     if HP.isnsideok(nside):
    #         is_healpix = True
    #     else:
    #         raise ValueError('Invalid nside presented')

    # zout = parms['output']['redshifts']
    # ofreqs = parms['output']['frequencies']
    # save_as_skymodel = parms['output']['skymodel']
    # if zout is None:
    #     if ofreqs is None:
    #         nchan = parms['output']['nchan']
    #         f0 = parms['output']['f0'] * U.Hz
    #         df = parms['output']['freq_resolution'] * U.Hz
    #         ofreqs = (f0 + (NP.arange(nchan) - 0.5 * nchan) * df) # in Hz
    #         zout = rest_freq / ofreqs - 1
    #     else:
    #         ofreqs = NP.asarray(ofreqs) * U.MHz # Input in MHz
    #         zout = rest_freq / ofreqs - 1
    # else:
    #     zout = NP.asarray(zout).reshape(-1)
    #     ofreqs = rest_freq / (1+zout)
    # if NP.any(zout < 0.0):
    #     raise ValueError('redshifts must not be negative')
    # if NP.any(ofreqs < 0.0):
    #     raise ValueError('Output frequencies must not be negative')

    # write_mode = parms['processing']['write_mode']
    # if write_mode not in [None, 'append']:
    #     raise ValueError('Input write_mode is invalid')
            
    # parallel = parms['processing']['parallel']
    # prll_type = parms['processing']['prll_type']
    # nproc = parms['processing']['nproc']
    wait_after_run = parms['processing']['wait_after_run']
    fname_delimiter = parms['format']['delimiter']
    zstart_pos = parms['format']['zstart_pos']
    zend_pos = parms['format']['zend_pos']
    zstart_identifier = parms['format']['zstart_identifier']
    zend_identifier = parms['format']['zend_identifier']
    zstart_identifier_pos = parms['format']['zstart_identifier_pos']
    zend_identifier_pos = parms['format']['zend_identifier_pos']
    if zstart_identifier is not None:
        if zstart_identifier_pos.lower() not in ['before', 'after']:
            raise ValueError('zstart_identifier_pos must be set to "before" or "after"')
        elif zstart_identifier_pos.lower() == 'before':
            zstart_value_place = 1
        else:
            zstart_value_place = 0

    if zend_identifier is not None:
        if zend_identifier_pos.lower() not in ['before', 'after']:
            raise ValueError('zend_identifier_pos must be set to "before" or "after"')
        elif zend_identifier_pos.lower() == 'before':
            zend_value_place = 1
        else:
            zend_value_place = 0
            
    if cube_source.lower() not in ['21cmfast']:
        raise ValueError('{0} cubes currently not supported'.format(cube_source))

    fullfnames = glob.glob(indir + infile_prefix + '*' + infile_suffix)
    fullfnames = NP.asarray(fullfnames)
    fnames = [fname.split('/')[-1] for fname in fullfnames]
    fnames = NP.asarray(fnames)
    if fnames[0].split(fname_delimiter)[-1] == 'lighttravel':
        dim = int(fnames[0].split(fname_delimiter)[-3])
        boxsize = float(fnames[0].split(fname_delimiter)[-2][:-3])
    else:
        dim = int(fnames[0].split(fname_delimiter)[-2])
        boxsize = float(fnames[0].split(fname_delimiter)[-1][:-3])
    boxsize = boxsize * U.Mpc # Input in Mpc
    cuberes = boxsize / dim 
    if zstart_identifier is not None:
        zstart_str = [fname.replace(fname_delimiter,' ').split()[zstart_pos].split(zstart_identifier)[zstart_value_place] for fname in fnames]
    else:
        zstart_str = [fname.replace(fname_delimiter,' ').split()[zstart_pos] for fname in fnames]
    if zend_identifier is not None:
        zend_str = [fname.replace(fname_delimiter,' ').split()[zend_pos].split(zend_identifier)[zend_value_place] for fname in fnames]
    else:
        zend_str = [fname.replace(fname_delimiter,' ').split()[zend_pos] for fname in fnames]
    
    zstart = NP.asarray(map(float, zstart_str))
    zend = NP.asarray(map(float, zend_str))
    sortind = NP.argsort(zstart)
    zstart = zstart[sortind]
    zend = zend[sortind]
    fnames = fnames[sortind]
    fullfnames = fullfnames[sortind]
    d_C = []
    lightcone_cube = None
    for fi,fullfname in enumerate(fullfnames):
        d_C_start = cosmo.comoving_distance(zstart[fi])
        if len(d_C) > 0:
            if d_C_start.to('Mpc').value <= d_C[-1]:
                raise ValueError('Overlapping comoving distances found between input cubes. Verify if this is intended behavior before proceeding.')
        d_C_in_cube = d_C_start + cuberes * NP.arange(dim)
        d_C += d_C_in_cube.to('Mpc').value.tolist()
        lc_cube = cosmotile.fastread_21cmfast_cube(fullfname)
        if lightcone_cube is None:
            lightcone_cube = NP.copy(lc_cube)
        else:
            lightcone_cube = NP.concatenate((lightcone_cube, lc_cube), axis=0) # Line of sight seems to be the first axis
    lightcone_cube = lightcone_cube * cube_units
    d_C = NP.asarray(d_C) * U.Mpc
    redshifts_in = NP.asarray([cosmology.z_at_value(cosmo.comoving_distance, dist) for dist in d_C])
    # freqs_in = CNST.rest_freq_HI / (1 + redshifts_in)

    xin = cuberes * NP.arange(dim)
    yin = cuberes * NP.arange(dim)
    
    xcent = 0.5 * xin.max()
    ycent = 0.5 * yin.max()

    xin -= xcent
    yin -= ycent

    xarr, yarr = NP.meshgrid(xin, yin)

    kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshifts_in)
    factor = kpc_per_arcmin / kpc_per_arcmin.max()

    cosmoinfo = {'Om0': cosmo.Om0, 'Ode0': cosmo.Ode0, 'h': cosmo.h, 'Ob0': cosmo.Ob0, 'w0': cosmo.w(0.0)}

    eps_z = 1e-6
    zbins = NP.linspace(redshifts_in.min()-eps_z, redshifts_in.max()+eps_z, nzbins+1, endpoint=True)
    counts, bin_edges, binnum, rvind = OPS.binned_statistic(redshifts_in, statistic='count', bins=zbins)

    phi = NP.arctan2(xarr, yarr)
    PDB.set_trace()
    for ind in range(counts.size):
        if counts[ind] > 0:
            print('Processing redshift bin {0:.1f} <= z < {1:.1f}'.format(zbins[ind], zbins[ind+1]))
            zind = rvind[rvind[ind]:rvind[ind+1]]
            theta = NP.sqrt(xarr**2 + yarr**2) / kpc_per_arcmin[zind].max()

            xout = xarr[NP.newaxis,:,:] * factor[zind,NP.newaxis,NP.newaxis]
            yout = yarr[NP.newaxis,:,:] * factor[zind,NP.newaxis,NP.newaxis]
            zyxout = NP.concatenate((yout[...,NP.newaxis].to('Mpc').value, xout[...,NP.newaxis].to('Mpc').value), axis=-1) * U.Mpc # nz x ny x nx x 2
            lightcone_out = NP.empty((zind.size,lightcone_cube.shape[1],lightcone_cube.shape[2]))
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Redshifts'.format(zind.size), PGB.ETA()], maxval=zind.size).start()
            for i in range(zind.size):
                lightcone_out[i,...] = interpolate.interpn((yin.to('Mpc').value,xin.to('Mpc').value), lightcone_cube[i,...].to(sim_units).value, zyxout[i,...].to('Mpc').value, bounds_error=False)
                progress.update(i+1)
            progress.finish()

            lightcone_out = lightcone_out * cube_units

            write_lightcone_to_sphangles(lightcone_out, theta, phi, redshifts_in[zind], cosmoinfo, rest_freq, outfile+'_zbin_{0:.1f}_{1:.1f}.hdf5'.format(zbins[ind], zbins[ind+1]))

    PDB.set_trace()
    
