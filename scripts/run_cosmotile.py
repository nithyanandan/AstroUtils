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
import progressbar as PGB
from astroutils import cosmotile
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
    outdir = parms['dirstruct']['outdir']
    outfile_prefix = parms['dirstruct']['outfile_prefix']
    if outfile_prefix is None:
        outfile = outdir + 'light_cone_surfaces.hdf5'
    elif isinstance(outfile_prefix, str):
        outfile = outdir + outfile_prefix + '_light_cone_surfaces.hdf5'
    else:
        raise TypeError('Output filename prefix must be set to None or a string')

    cube_source = parms['sim']['source']
    rest_freq = parms['output']['rest_frequency']
    nside = parms['output']['nside']
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

    zout = parms['output']['redshifts']
    ofreqs = parms['output']['frequencies']
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
        idict = {'outvals': NP.asarray(redshift).reshape(-1), 'inpcubes': None, 'cubedims': None, 'cube_source': '21cmfast', 'interp_method':'linear', 'outfiles': None, 'returncubes': True}
        tdict = {'inpres': cuberes, 'nside': nside, 'redshift': redshift, 'freq': None, 'method': 'linear', 'rest_freq': rest_freq, 'cosmo': cosmo}
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

    hpxsurfaces = []
    if parallel:
        ts = time.time()
        if nproc is None:
            nproc = MP.cpu_count()
        assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
        nproc = min([nproc, zout.size])
        try:
            pool = MP.Pool(processes=nproc)
            hpxsurfaces = pool.map(cosmotile.coevalcube_interp_tile2hpx_wrapper_arg_splitter, IT.izip(interpdicts, tiledicts))
            # hpxsurfaces = pool.map(convert_coevalcube_to_healpix_arg_splitter, IT.izip(list_inpcubes, inpres, list_nsides, list_freqs, list_redshifts, list_methods, list_rest_freqs, list_cosmo))
            pool.close()
            pool.join()
            te = time.time()
            print 'Time consumed: {0:.1f} seconds'.format(te-ts)
        except MemoryError:
            parallel = False
            del pool
            hpxsurfaces = []
            warnings.warn('Memory requirements too high. Downgrading to serial processing.')
        
    if not parallel:
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency channels '.format(ofreqs.size), PGB.ETA()], maxval=ofreqs.size).start()
        for ind in range(len(interpdicts)):
            hpxsurfaces += [cosmotile.coevalcube_interp_tile2hpx_wrapper(interpdicts[ind], tiledicts[ind])]
            progress.update(ind+1)
        progress.finish()

    hpxsurfaces = conv_factor * NP.asarray(hpxsurfaces)
    cosmotile.write_lightcone_surfaces(hpxsurfaces, units, outfile, ofreqs, cosmo=cosmo, is_healpix=is_healpix)

    if wait_after_run:
        PDB.set_trace()
