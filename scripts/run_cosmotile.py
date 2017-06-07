#!python

import os, glob
import argparse
import yaml
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
    freq_min = parms['freq_min']
    freq_max = parms['freq_max']
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
    if redshift_identifier is not None:
        zstr = [fname.split('_')[redshift_placeholder].split(redshift_identifer)[redshift_value_place] for fname in fnames]
    else:
        zstr = [fname.split('_')[redshift_placeholder] for fname in fnames]
    redshifts = NP.asarray(map(float, zstr))
    infreqs = rest_freq / (1+redshifts)

    ind = NP.logical_and(infreqs >= freq_min, infreqs <= freq_max)
    fnames = fnames[ind]
    redshifts = redshifts[ind]
    infreqs = infreqs[ind]

    sortind = NP.argsort(infreqs)
    fnames = fnames[sortind]
    redshifts = redshifts[sortind]
    infreqs = infreqs[sortind]

    
