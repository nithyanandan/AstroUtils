from __future__ import print_function, division, unicode_literals, absolute_import
import healpy as HP
import numpy as NP

#################################################################################

def power_law_spectral_index(freq, flux, verbose=True):
    
    """
    -----------------------------------------------------------------------------
    Compute the power law spectral index when flux densities and frequencies are
    specified. If only two frequencies are provided, a power law is computed from 
    these two frequencies but if more than two frequencies are specified, the
    power law is computed from a polynomial fit of log(flux) vs. log(freq)

    Inputs:

    freq        [list, tuple or numpy array] Frequencies at which flux densities
                are specified. Must contain at least 2 frequencies.

    flux        [list, tuple or numpy array] Flux densities at the specified
                frequencies. Can specify multiple sources at the same. Each
                source's flux densities must be of the same size as that of freq. 
                For instance, if freq is a N-element vector (N >= 2), flux must
                be a list of N-element lists or tuples or a MxN numpy array where 
                M is the number of sources.

    Keyword Inputs:

    verbose     [boolean] If set to True (default) print diagnostic and progress
                messages, otherwise suppress (False)

    Output:

    Spectral index for each of the sources whose flux densities are fitted as a
    power law function of frequency in log-log units. 

    -----------------------------------------------------------------------------
    """

    if verbose:
        print '\nRunning power_law_spectral_index() for estimating a single power law spectral index...'

    try:
        freq, flux
    except NameError:
        raise NameError('freq and flux must be defined.')

    if verbose:
        print '\tVerifying input data parameters for compatibility...'

    if isinstance(freq, (list, tuple)):
        freq = NP.asarray(freq).ravel()
    elif isinstance(freq, NP.ndarray):
        freq.ravel()
    else:
        raise TypeError('freq must be a list, tuple or numpy array')

    if freq.size < 2:
        raise ValueError('freq must contain at least 2 elements.')

    if NP.any(freq <= 0.0):
        raise ValueError('All elements in freq must be positive.')

    if isinstance(flux, (list, tuple)):
        flux = NP.asarray(flux)
    elif not isinstance(flux, NP.ndarray):
        raise TypeError('flux must be a list, tuple or numpy array')

    if flux.shape[1] != freq.size:
        raise ValueError('flux and freq have incompatible sizes')

    if NP.any(flux <= 0.0):
        raise ValueError('All elements in flux must be positive.')

    if verbose:
        print '\tInput data parameters verified to be compatible.'
        print '\tProceeding with power law spectral index estimation...'

    flux = flux.reshape(-1,freq.size)

    if freq.size == 2:
        alpha = NP.log10(flux[:,1]/flux[:,0]) / NP.log10(freq[1]/freq[0])
        if verbose:
            print '\tPower law spectral index estimated for {0} objects.'.format(flux.shape[0])
    else:
        coeffs = NP.polyfit(NP.log10(freq), NP.log10(flux.T), 1)
        alpha = coeffs[0,:]
        if verbose:
            print '\tPower law spectral index estimated for {0} objects by a linear \n\t\tpolynomial in log-log units.'.format(flux.shape[0])
        
    if verbose:
        print 'Power law spectral index estimation completed successfully.\n'
    return alpha
    
#################################################################################
    
