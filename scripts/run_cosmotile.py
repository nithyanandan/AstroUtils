#!python

import os, sys, glob
import argparse
import yaml
import multiprocessing as MP
import itertools as IT
import time 
import healpy as HP
import numpy as NP
import astropy.cosmology as cosmology
import astropy.constants as FCNST
import progressbar as PGB
import warnings
from astroutils import cosmotile
from astroutils import constants as CNST
from astroutils import catalog as SM
import astroutils
import ipdb as PDB

astroutils_path = astroutils.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to tile cosmological coeval cubes to lightcone healpix cube')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', required=False, type=str, default=astroutils_path+'examples/cosmotile/cosmotile_parms.yaml', help='Config file for processing cosmological coeval cubes')

    args = vars(parser.parse_args())
    parmsfile = args['parmsfile']
    with open(parmsfile, 'r') as pfile:
        parms = yaml.safe_load(pfile)

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
        outfile = outdir + 'light_cone_surfaces'
    elif isinstance(outfile_prefix, str):
        outfile = outdir + outfile_prefix + '_light_cone_surfaces'
    else:
        raise TypeError('Output filename prefix must be set to None or a string')

    cube_source = parms['sim']['source']
    rest_freq = parms['output']['rest_frequency']
    nside = parms['output']['nside']
    theta_range = parms['output']['theta_range']
    phi_range = parms['output']['phi_range']
    angres = parms['output']['angres']
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
        cosmo = cosmology.wCDM(H0, Om0, Ode0, w0=w0, Ob0=Ob0)
    elif cosmoparms['name'].lower() == 'wmap9':
        cosmo = cosmology.WMAP9
    else:
        raise ValueError('{0} preset not currently accepted for cosmology'.format(cosmoparms['name'].lower()))
    process_stage = parms['sim']['process_stage']
    units = parms['sim']['units']
    if not isinstance(units, str):
        raise TypeError('Input units must be a string')
    if units not in ['mK', 'K']:
        raise ValueError('Supported units are "mK" and "K"')

    if units == 'mK':
        conv_factor = 1e-3
        units = 'K'
    else:
        conv_factor = 1.0
    is_healpix = False
    if nside is not None:
        if HP.isnsideok(nside):
            is_healpix = True
        else:
            raise ValueError('Invalid nside presented')
        theta_phi = None
    else:
        theta_range = NP.asarray(theta_range)
        phi_range = NP.asarray(phi_range)
        theta_range = NP.sort(theta_range)
        phi_range = NP.sort(phi_range)
        nside_patch = 1
        angres_patch = HP.nside2resol(nside_patch)
        while angres_patch > NP.radians(angres):
            nside_patch *= 2
            angres_patch = HP.nside2resol(nside_patch)
        pixarea_patch = HP.nside2pixarea(nside_patch)
        theta, phi = HP.pix2ang(nside_patch, NP.arange(HP.nside2npix(nside_patch)))
        select_ind = NP.logical_and(NP.logical_and(theta >= NP.radians(theta_range[0]), theta <= NP.radians(theta_range[1])), NP.logical_and(phi >= NP.radians(phi_range[0]), phi <= NP.radians(phi_range[1])))
        theta = theta[select_ind]
        phi = phi[select_ind]
        theta_phi = NP.degrees(NP.hstack((theta.reshape(-1,1), phi.reshape(-1,1))))

    zout = parms['output']['redshifts']
    ofreqs = parms['output']['frequencies']
    save_as_skymodel = parms['output']['skymodel']
    if zout is None:
        if ofreqs is None:
            nchan = parms['output']['nchan']
            f0 = parms['output']['f0']
            df = parms['output']['freq_resolution']
            ofreqs = (f0 + (NP.arange(nchan) - 0.5 * nchan) * df) # in Hz
            zout = rest_freq / ofreqs - 1
        else:
            ofreqs = NP.asarray(ofreqs)
            zout = rest_freq / ofreqs - 1
    else:
        zout = NP.asarray(zout).reshape(-1)
        ofreqs = rest_freq / (1+zout)
    if NP.any(zout < 0.0):
        raise ValueError('redshifts must not be negative')
    if NP.any(ofreqs < 0.0):
        raise ValueError('Output frequencies must not be negative')

    parallel = parms['processing']['parallel']
    nproc = parms['processing']['nproc']
    wait_after_run = parms['processing']['wait_after_run']
    fname_delimiter = parms['format']['delimiter']
    z_placeholder = parms['format']['z_placeholder']
    z_identifier = parms['format']['z_identifier']
    z_identifier_place = parms['format']['z_identifier_place']
    if z_identifier is not None:
        if z_identifier_place.lower() not in ['before', 'after']:
            raise ValueError('z_identifier_place must be set to "before" or "after"')
        elif z_identifier_place.lower() == 'before':
            redshift_value_place = 1
        else:
            redshift_value_place = 0

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
    cuberes = boxsize / dim # in Mpc
    if z_identifier is not None:
        zstr = [fname.split(fname_delimiter)[z_placeholder].split(z_identifier)[redshift_value_place] for fname in fnames]
    else:
        zstr = [fname.split(fname_delimiter)[z_placeholder] for fname in fnames]
    zin = NP.asarray(map(float, zstr))
    infreqs = rest_freq / (1+zin)

    # ind = NP.logical_and(infreqs >= freq_min, infreqs <= freq_max)
    # fnames = fnames[ind]
    # zin = zin[ind]
    # infreqs = infreqs[ind]

    sortind = NP.argsort(infreqs)
    fnames = fnames[sortind]
    zin = zin[sortind]
    infreqs = infreqs[sortind]

    sortind_z_asc = NP.argsort(zin)

    interpdicts = []
    tiledicts = []
    for zind,redshift in enumerate(zout):
        idict = {'outvals': NP.asarray(redshift).reshape(-1), 'inpcubes': None, 'cubedims': None, 'cube_source': cube_source, 'process_stage': process_stage, 'interp_method':'linear', 'outfiles': None, 'returncubes': True}
        tdict = {'inpres': cuberes, 'nside': nside, 'theta_phi': theta_phi, 'redshift': redshift, 'freq': None, 'method': 'linear', 'rest_freq': rest_freq, 'cosmo': cosmo}
        if redshift <= zin.min():
            idict['invals'] = [zin.min()]
            idict['cubefiles'] = [indir+fnames[-1]]
        elif redshift >= zin.max():
            idict['invals'] = [zin.max()]
            idict['cubefiles'] = [indir+fnames[0]]
        else:
            insert_ind = NP.searchsorted(infreqs, ofreqs[zind])
            idict['invals'] = [zin[insert_ind], zin[insert_ind-1]]
            idict['cubefiles'] = [indir+fnames[insert_ind], indir+fnames[insert_ind-1]]
        interpdicts += [idict]
        tiledicts += [tdict]

    if save_as_skymodel:
        if nside is not None:
            angres_patch = NP.degrees(HP.nside2resol(nside))
            pixarea_patch = HP.nside2pixarea(nside)
            theta, phi = NP.degrees(HP.pix2ang(nside, NP.arange(HP.nside2npix(nside))))
        else:
            theta = NP.degrees(theta)
            phi = NP.degrees(phi)
        wl = FCNST.c.to('m/s').value / ofreqs
        dJy_dK = 2 * FCNST.k_B.to('J/K').value * pixarea_patch / wl**2 / CNST.Jy # nchan (in Jy/K)
        radec = NP.hstack((phi.reshape(-1,1), 90.0 - theta.reshape(-1,1)))

    sphsurfaces = []
    if parallel:
        ts = time.time()
        if nproc is None:
            nproc = MP.cpu_count()
        assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
        nproc = min([nproc, zout.size])
        try:
            pool = MP.Pool(processes=nproc)
            sphsurfaces = pool.map(cosmotile.coeval_interp_cube_to_sphere_surface_wrapper_arg_splitter, IT.izip(interpdicts, tiledicts), chunksize=zout.size/nproc)
            # progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} frequencies'.format(zout.size), PGB.ETA()], maxval=zout.size).start()
            # for i,_ in enumerate(sphsurfaces):
            #     print '{0:0d}/{1:0d} completed'.format(i, len(interpdicts))
            #     progress.update(i+1)
            # progress.finish()

            pool.close()
            pool.join()
            te = time.time()
            print 'Time consumed: {0:.1f} seconds'.format(te-ts)
        except MemoryError:
            parallel = False
            pool.close()
            pool.join()
            del pool
            sphsurfaces = []
            warnings.warn('Memory requirements too high. Downgrading to serial processing.')
        
    if not parallel:
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(ofreqs.size), PGB.ETA()], maxval=ofreqs.size).start()
        for ind in xrange(len(interpdicts)):
            # sphsurfaces += [cosmotile.coeval_interp_cube_to_sphere_surface_wrapper(interpdicts[ind], tiledicts[ind])]
            sphsurface = cosmotile.coeval_interp_cube_to_sphere_surface_wrapper(interpdicts[ind], tiledicts[ind])
            if save_as_skymodel:
                init_parms = {'name': cube_source, 'frequency': ofreqs[ind], 'location': radec, 'spec_type': 'spectrum', 'spectrum': dJy_dK[ind]*sphsurface.reshape(-1,1), 'src_shape': NP.hstack((angres_patch+NP.zeros(phi.size).reshape(-1,1), angres_patch+NP.zeros(phi.size).reshape(-1,1), NP.zeros(phi.size).reshape(-1,1))), 'epoch': 'J2000', 'coords': 'radec', 'src_shape_units': ('degree', 'degree', 'degree')}
                skymod = SM.SkyModel(init_file=None, init_parms=init_parms)
                if ind == 0:
                    skymod.save(outfile, fileformat='hdf5')
                    # cosmotile.write_lightcone_catalog(init_parms, outfile=outfile, action='store')
                else:
                    SM.append_SkyModel_file(outfile, skymod, 'freq', filemode='a')
            else:
                cosmotile.write_lightcone_surfaces(sphpatches, units, outfile, ofreqs, cosmo=cosmo, is_healpix=is_healpix)

            progress.update(ind+1)
        progress.finish()

    sphpatches = NP.asarray([sphsurf for sphsurf in sphsurfaces])
    sphpatches = conv_factor * NP.asarray(sphpatches)
    if save:
        if save_as_skymodel:
            if nside is not None:
                angres_patch = NP.degrees(HP.nside2resol(nside))
                pixarea_patch = HP.nside2pixarea(nside)
                theta, phi = NP.degrees(HP.pix2ang(nside, NP.arange(HP.nside2npix(nside))))
            else:
                theta = NP.degrees(theta)
                phi = NP.degrees(phi)
            wl = FCNST.c.to('m/s').value / ofreqs
            dJy_dK = 2 * FCNST.k_B.to('J/K').value * pixarea_patch / wl**2 / CNST.Jy # nchan (in Jy/K)
            radec = NP.hstack((phi.reshape(-1,1), 90.0 - theta.reshape(-1,1)))
            init_parms = {'name': cube_source, 'frequency': ofreqs, 'location': radec, 'spec_type': 'spectrum', 'spectrum': dJy_dK.reshape(1,-1)*sphpatches.T, 'src_shape': NP.hstack((angres_patch+NP.zeros(phi.size).reshape(-1,1), angres_patch+NP.zeros(phi.size).reshape(-1,1), NP.zeros(phi.size).reshape(-1,1))), 'epoch': 'J2000', 'coords': 'radec', 'src_shape_units': ('degree', 'degree', 'degree')}
            cosmotile.write_lightcone_catalog(init_parms, outfile=outfile, action='store')
        else:
            cosmotile.write_lightcone_surfaces(sphpatches, units, outfile, ofreqs, cosmo=cosmo, is_healpix=is_healpix)

    if wait_after_run:
        PDB.set_trace()
