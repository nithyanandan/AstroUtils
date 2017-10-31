from __future__ import division
import numpy as NP
import astropy.cosmology as CP

cosmo100 = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)  # Using H0 = 100 km/s/Mpc

def k_perp(uv_length, redshift, cosmo=cosmo100):

    """
    ---------------------------------------------------------------------------
    Compute transverse wavenumbers (h/Mpc) corresponding to specified 
    UV lengths, and redshifts using the relationship between UV lengths and 
    cosmological factors

    Inputs:

    uv_length [numpy array] UV lengths (in numbers of wavelengths) 

    redshift  [numpy array] redshift

    cosmo     [instance of cosmology class from astropy] An instance of class
              FLRW or default_cosmology of astropy cosmology module. Default
              uses Flat lambda CDM cosmology with Omega_m=0.27, H0=100 km/s/Mpc

    Outputs:

    Numpy array containing k_perp values is returned. It will be of shape 
    (nz,nuv)
    ---------------------------------------------------------------------------
    """

    try:
        uv_length, redshift
    except NameError:
        raise NameError('Inputs uv_length, redshift, and wavelength must be specified')

    if not isinstance(uv_length, (int,float,NP.ndarray)):
        raise TypeError('Input uv_length must be a scalar or numpy array')
    uv_length = NP.asarray(uv_length).reshape(-1)

    if not isinstance(redshift, (int,float,NP.ndarray)):
        raise TypeError('Input redshift must be a scalar or numpy array')
    redshift = NP.asarray(redshift).reshape(-1)

    if not isinstance(cosmo, (CP.FLRW, CP.default_cosmology)):
        raise TypeError('Input cosmology must be a cosmology class defined in Astropy')
    
    kperp = 2 * NP.pi * uv_length[NP.newaxis,:] / cosmo.comoving_transverse_distance(redshift)[:,NP.newaxis]
    return kperp
    
###############################################################################

