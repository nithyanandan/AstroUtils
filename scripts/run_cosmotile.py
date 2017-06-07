#!python

import os, sys, glob
import argparse
import yaml
import multiprocessing as MP
import astropy.cosmology as cosmology
from astroutils import cosmotile
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

    indir = parms['indir']
    infile_prefix = parms['infile_prefix']
    infile_suffix = parms['infile_suffix']

    cube_source = parms['source']
    rest_freq = parms['rest_freq']
    nside = parms['nside']
    cosmoparms = parms['cosmo']
    if cosmoparms is not None:
        h = cosmoparms['h']
        H0 = 100.0 * h
        Om0 = cosmoparms['Om0']
        Ode0 = 1.0 - cosmoparms['Ode0']
        Ob0 = cosmoparms['Ob0'] / h**2
        wl = cosmoparms['wl']
        cosmo = cosmology.wCDM(H0, Om0, Ode0, w0=wl, Ob0=Ob0)
    else:
        cosmo = None

    parallel = parms['parallel']
    nproc = parms['nproc']
    ofreqs = parms['freqs']
    zout = parms['redshifts']
    if (zout is None) and (ofreqs is None):
        raise ValueError('One of "freqs" or "redshifts" must be specified')
    if (zout is not None) and (ofreqs is not None):
        raise ValueError('One and only one of "freqs" or "redshifts" must be specified')
    if zout is None:
        if NP.any(ofreqs <= 0.0):
            raise ValueError('Output frequencies must be positive')
        zout = rest_freq / ofreqs - 1
    else:
        if NP.any(zout <= 0.0):
            raise ValueError('Output redshifts must be positive')
        ofreqs = rest_freq / (1+zout)
        
    redshift_placeholder = parms['redshift_placeholder']
    redshift_identifier = parms['redshift_identifier']
    redshift_identifier_place = parms['redshift_identifier_place']
    if redshift_identifier is not None:
        if redshift_identifier_place.lower() not in ['before', 'after']:
            raise ValueError('redshift_identifier_place must be set to "before" or "after"')
        elif redshift_identifier_place.lower() == 'before':
            redshift_value_place = 1
        else:
            redshift_value_place = 0

    # if cube_source.lower() not in ['21cmfast']:
    #     raise ValueError('{0} cubes currently not supported'.format(cube_source))

    fullfnames = glob.glob(indir + prefix + '*' + suffix)
    fullfnames = NP.asarray(fullfnames)
    fnames = [fname.split('/')[-1] for fname in fullfnames]
    fnames = NP.asarray(fnames)
    if fnames.split('_')[-1] == 'lighttravel':
        dim = int(fname.split('_')[-3])
        boxsize = float(fname.split('_')[-2][:-3])
    else:
        dim = int(infile.split('_')[-2])
        boxsize = float(fname.split('_')[-1][:-3])
    cuberes = boxsize / dim # in Mpc
    if redshift_identifier is not None:
        zstr = [fname.split('_')[redshift_placeholder].split(redshift_identifer)[redshift_value_place] for fname in fnames]
    else:
        zstr = [fname.split('_')[redshift_placeholder] for fname in fnames]
    zin = NP.asarray(map(float, zstr))
    infreqs = rest_freq / (1+zin)

    ind = NP.logical_and(infreqs >= freq_min, infreqs <= freq_max)
    fnames = fnames[ind]
    zin = zin[ind]
    infreqs = infreqs[ind]

    sortind = NP.argsort(infreqs)
    fnames = fnames[sortind]
    zin = zin[sortind]
    infreqs = infreqs[sortind]

    sortind_z_asc = NP.argsort(zin)

    interpdicts = []
    tiledicts = []
    for zind,redshift in enumerate(zout):
        idict = {'outvals': NP.asarray(redshift).reshape(-1), 'inpcubes': None, 'cubedims': None, 'cube_source': '21cmfast', 'method'='linear', 'outfiles': None, 'returncubes': True}
        tdict = {'inpres': cuberes, 'nside': nside, 'redshift': redshift, 'method': 'linear', 'rest_freq': rest_freq, 'cosmo': cosmo}
        if redshift <= zin.min():
            idict['invals'] = [zin.min()]
            idict['cubefiles'] = [indir+fnames[-1]]
        elif redshift >= zin.max():
            idict['invals'] = [zin.max()]
            idict['cubefiles'] = [indir+fnames[0]]
        else:
            insert_ind = NP.searchsorted(infreqs, ofreqs[zind])
            idict['invals'] = [zin[insert_ind+1], zin[insert_ind]]
            idict['cubefiles'] = [indir+fnames[insert_ind+1], indir+fnames[insert_ind]]
        interpdicts += [idict]
        tiledicts += [tdict]

    hpxsurfaces = []
    if parallel:
        if nproc is None:
            nproc = MP.cpu_count()
        assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
        nproc = min([nproc, redshifts.size])
        try:
            pool = MP.Pool(processes=nproc)
            hpxsurfaces = pool.map(convert_coevalcube_to_healpix_arg_splitter, IT.izip(list_inpcubes, inpres, list_nsides, list_freqs, list_redshifts, list_methods, list_rest_freqs, list_cosmo))
            pool.close()
            pool.join()
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

    if wait_after_run:
        PDB.set_trace()
