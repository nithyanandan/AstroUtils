#!python

import yaml, argparse
import numpy as NP
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy import units as U
from astroutils import geometry as GEOM
from astroutils import nonmathops as NMO
import astroutils

astroutils_path = astroutils.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to match input catalog positions with specified catalogs')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=astroutils_path+'examples/catalogops/catalog_match_parms.yaml', type=file, required=False, help='File specifying input parameters')
    
    args = vars(parser.parse_args())
    
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)
    
    dirinfo = parms['directory']
    projectdir = dirinfo['projectdir']
    outdir = projectdir
    outfile = outdir + dirinfo['outfile'] + '.hdf5'

    refcat = parms['refcat']
    refcatfile = refcat['catfile']
    reftable = ascii.read(refcatfile)
    obj_colname = refcat['obj_colname']
    ra_colname = refcat['RA_colname']
    dec_colname = refcat['Dec_colname']
    if refcat['RA_units'] == 'hms':
        ra_units = U.hourangle
    if (refcat['Dec_units'] == 'dms') or (refcat['Dec_units'] == 'deg'):
        dec_units = U.deg

    refRA = reftable[ra_colname]
    refDec = reftable[dec_colname]
    refObj = reftable[obj_colname]
    refcoords = SkyCoord(refRA, refDec, unit=(ra_units, dec_units), equinox=refcat['epoch'], frame='icrs')

    subsetinfo = parms['subset']
    subparnames = subsetinfo['parmnames']
    select = NP.ones(len(reftable), dtype=NP.bool)
    if len(subparnames) > 0:
        parmranges = subsetinfo['parmrange']
        for i,prm in enumerate(subparnames):
            subdat = reftable[prm]
            if (subdat.dtype == NP.float) or (subdat.dtype == NP.int):
                select[NP.logical_or(subdat < parmranges[i][0], subdat > parmranges[i][1])] = False
            else:
                for prmstr in parmranges[i]:
                    if prmstr[0] == '!':
                        pstr = prmstr[1:]
                        select = NP.logical_and(select, NP.logical_not(NP.asarray([pstr in subdat[j] for j in range(len(subdat))])))
                    else:
                        pstr = prmstr
                        select = NP.logical_and(select, NP.asarray([pstr in subdat[j] for j in range(len(subdat))]))

    select_ind = NP.where(select)[0]
    select_reftable = reftable[select_ind]
    select_refcoords = refcoords[select_ind]
    
    radiocats = parms['radiocats']

    matchinfo = {}
    for radcatkey in radiocats:
        if radiocats[radcatkey]['action']:
            if radiocats[radcatkey]['searchrad'] is not None:
                matchrad = radiocats[radcatkey]['searchrad']
            else:
                psfhwhm = 0.5 * radiocats[radcatkey]['psffwhm']
                matchrad = NP.sqrt(radiocats[radcatkey]['poserr']**2 + psfhwhm**2)

            min_fpeak = radiocats[radcatkey]['min_fpeak']
            max_fpeak = radiocats[radcatkey]['max_fpeak']
            min_fint = radiocats[radcatkey]['min_fint']
            max_fint = radiocats[radcatkey]['max_fint']
            fpeak = None
            fint = None

            if radcatkey.lower() == 'nvss':
                hdulist = fits.open(radiocats[radcatkey]['catfile'])
                ra_deg_radcat = hdulist[1].data['RA(2000)']
                dec_deg_radcat = hdulist[1].data['DEC(2000)']
                fpeak = 1e3 * hdulist[1].data['PEAK INT'] # mJy/beam
                rmajax = hdulist[1].header['BM_MAJOR'] # restoring beam major axis in degrees
                rminax = hdulist[1].header['BM_MINOR'] # restoring beam minor axis in degrees
                fmajax = hdulist[1].data['MAJOR AX'] # fitted beam major axis in degrees
                fminax = hdulist[1].data['MINOR AX'] # fitted beam minor axis in degrees
                fint = fpeak * (fmajax * fminax) / (rmajax * rminax) # from NVSS catalog description document
            elif radcatkey.lower() == 'first':
                hdulist = fits.open(radiocats[radcatkey]['catfile'])
                ra_deg_radcat = hdulist[1].data['RA']
                dec_deg_radcat = hdulist[1].data['DEC']
                fpeak = hdulist[1].data['FPEAK'] # mJy/beam
                fint = hdulist[1].data['FINT'] # mJy
            elif radcatkey.lower() == 'tgss':
                hdulist = fits.open(radiocats[radcatkey]['catfile'])
                ra_deg_radcat = hdulist[1].data['RA']
                dec_deg_radcat = hdulist[1].data['DEC']
                fpeak = hdulist[1].data['Peak_flux'] # mJy/beam
                fint = hdulist[1].data['Total_flux'] # mJy
            elif radcatkey.lower() == 'gleam':
                hdulist = fits.open(radiocats[radcatkey]['catfile'])
                ra_deg_radcat = hdulist[1].data['RAJ2000']
                dec_deg_radcat = hdulist[1].data['DEJ2000']
                fpeak = 1e3 * hdulist[1].data['peak_flux_{0:.0f}'.format(radiocats[radcatkey]['freq'])] # mJy/beam
                fint = 1e3 * hdulist[1].data['int_flux_{0:.0f}'.format(radiocats[radcatkey]['freq'])] # mJy
            elif radcatkey.lower() == 'mwa-eor':
                hdulist = fits.open(radiocats[radcatkey]['catfile'])
                ra_deg_radcat = hdulist[1].data['RAJ2000']
                dec_deg_radcat = hdulist[1].data['DECJ2000']
                fint = 1e3 * hdulist[1].data['S_{0:.0f}'.format(radiocats[radcatkey]['freq'])] # mJy

            eps = 1e-10
            if min_fpeak is None:
                if fpeak is not None:
                    min_fpeak = NP.nanmin(NP.abs(fpeak)) - eps
            if max_fpeak is None:
                if fpeak is not None:
                    max_fpeak = NP.nanmax(NP.abs(fpeak)) + eps

            if min_fint is None:
                if fint is not None:
                    min_fint = NP.nanmin(NP.abs(fint)) - eps
            if max_fint is None:
                if fint is not None:
                    max_fint = NP.nanmax(NP.abs(fint)) + eps
                
            if fpeak is not None:
                ind_fpeak = NP.where(NP.logical_and(fpeak >= min_fpeak, fpeak <= max_fpeak))[0]
            else:
                ind_fpeak = NP.arange(ra_deg_radcat.size)
            if fint is not None:
                ind_fint = NP.where(NP.logical_and(fint >= min_fint, fint <= max_fint))[0]
            else:
                ind_fint = NP.arange(ra_deg_radcat.size)

            ind_flux_cut = NP.intersect1d(ind_fpeak, ind_fint)
                
            ra_deg_radcat = ra_deg_radcat[ind_flux_cut]
            dec_deg_radcat = dec_deg_radcat[ind_flux_cut]
            if fpeak is not None:
                fpeak = fpeak[ind_flux_cut]
            if fint is not None:
                fint = fint[ind_flux_cut]

            nnearest = radiocats[radcatkey]['nnearest']
            maxmatches = radiocats[radcatkey]['maxmatches']

            mref, mrad, d12 = GEOM.spherematch(select_refcoords.ra.deg, select_refcoords.dec.deg, ra_deg_radcat, dec_deg_radcat, matchrad=matchrad/3.6e3, nnearest=nnearest, maxmatches=maxmatches)

            matchinfo[radcatkey] = {}
            matchinfo[radcatkey]['radius'] = matchrad
            matchinfo[radcatkey]['nnearest'] = nnearest
            matchinfo[radcatkey]['maxmatches'] = maxmatches

            if len(mref) > 0:
                mref = NP.asarray(mref)
                mrad = NP.asarray(mrad)
    
                matchinfo[radcatkey]['freq'] = radiocats[radcatkey]['freq']
                matchinfo[radcatkey]['iref'] = select_ind[mref]
                matchinfo[radcatkey]['icat'] = ind_flux_cut[mrad]
                matchinfo[radcatkey]['objname'] = refObj.data[select_ind[mref]].data
                matchinfo[radcatkey]['refRA'] = select_refcoords.ra.deg[mref]
                matchinfo[radcatkey]['refDec'] = select_refcoords.dec.deg[mref]
                matchinfo[radcatkey]['catRA'] = ra_deg_radcat[mrad]
                matchinfo[radcatkey]['catDec'] = dec_deg_radcat[mrad]
                matchinfo[radcatkey]['dist'] = d12 * 3.6e3
                if fpeak is not None:
                    matchinfo[radcatkey]['fpeak'] = fpeak[mrad]
                if fint is not None:
                    matchinfo[radcatkey]['fint'] = fint[mrad]
                if min_fpeak is not None:
                    matchinfo[radcatkey]['min_peak'] = min_fpeak
                    matchinfo[radcatkey]['max_peak'] = max_fpeak
                if min_fint is not None:
                    matchinfo[radcatkey]['min_fint'] = min_fint
                    matchinfo[radcatkey]['max_fint'] = max_fint
    NMO.save_dict_to_hdf5(matchinfo, outfile)
