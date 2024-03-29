#!python

from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import map
import os, sys, glob
import argparse
import yaml
import multiprocessing as MP
import itertools as IT
import time 
import healpy as HP
import numpy as NP
import astropy.cosmology as cosmology
from astropy import units as U
import progressbar as PGB
from tqdm import tqdm
import warnings
from astroutils import cosmotile
import astroutils
import ipdb as PDB

astroutils_path = astroutils.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to smooth cosmological cubes to given resolution, downsample and save')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', required=False, type=str, default=astroutils_path+'examples/cosmotile/cosmosmooth_parms.yaml', help='Config file for smoothing cosmological cubes')

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
        outfile_prefix = 'smoothed_cosmocube_'
    elif not isinstance(outfile_prefix, str):
        raise TypeError('Output filename prefix must be set to None or a string')

    nside = parms['output']['nside']
    if nside is not None:
        if isinstance(nside, int):
            if HP.isnsideok(nside):
                fwhm_angres = HP.nside2resol(nside, arcmin=True) * U.arcmin # arcmin
            else:
                raise ValueError('NSIDE parameter invalid')
        else:
            raise TypeError('NSIDE parameter must be an integer')
    else:
        fwhm_angres = parms['output']['angres']
        if not isinstance(fwhm_angres, (int,float)):
            raise TypeError('Angular resolution parameter must be a scalar')
        fwhm_angres = fwhm_angres * U.arcmin

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
    zin = NP.asarray(list(map(float, zstr)))

    sigma_angres = fwhm_angres / (2 * NP.sqrt(2.0 * NP.log(2.0)))
    cmv_trns_dist_sigma = sigma_angres * cosmo.kpc_comoving_per_arcmin(zin)
    cmv_trns_dist_res = cuberes * U.Mpc
    smoothing_pix_sigma = cmv_trns_dist_sigma / cmv_trns_dist_res
    smoothing_pix_sigma = smoothing_pix_sigma.decompose().value
    smoothing_pix_fwhm = smoothing_pix_sigma * 2 * NP.sqrt(2 * NP.log(2.0))
    downsample_factor = max([1.0, smoothing_pix_fwhm.min()])

    smthinfo = []
    for fi,fname in enumerate(fnames):
        smthdict = {'infile': indir+fname, 'outfile': outdir+outfile_prefix+fname, 'smooth_scale': smoothing_pix_sigma[fi], 'smooth_axes': [0,1,2], 'downsample_factor': downsample_factor, 'downsample_axes': [0,1,2], 'inpres': cuberes, 'units': units}
        smthinfo += [smthdict]

    if parallel:
        ts = time.time()
        if nproc is None:
            nproc = MP.cpu_count()
        assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
        nproc = min([nproc, len(fnames)])
        try:
            pool = MP.Pool(processes=nproc)
            with tqdm(total=len(smthinfo)) as pbar:
                for iternum in tqdm(pool.imap(cosmotile.cube_smooth_downsample_save, smthinfo, chunksize=len(smthinfo)/nproc)):
                    pbar.update()
            pbar.close()
            # for iternum in tqdm(pool.imap(cosmotile.cube_smooth_downsample_save, smthinfo, chunksize=len(smthinfo)/nproc), total=len(smthinfo)):
                
            pool.close()
            pool.join()
            te = time.time()
            print('Time consumed: {0:.1f} seconds'.format(te-ts))
        except MemoryError:
            parallel = False
            pool.close()
            pool.join()
            del pool
            warnings.warn('Memory requirements too high. Downgrading to serial processing.')
    else:
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} cubes '.format(len(smthinfo)), PGB.ETA()], maxval=len(smthinfo)).start()
        for fi, fname in enumerate(fnames):
            cosmotile.cube_smooth_downsample_save(smthinfo[fi])
            progress.update(fi+1)
        progress.finish()

    if wait_after_run:
        PDB.set_trace()
