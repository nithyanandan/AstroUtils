import numpy as NP
import healpy as HP
import astropy.cosmology as cosmology
import multiprocessing as MP
import warnings
import constants as CNST

#################################################################################

def convert_coevalcube_to_healpix_arg_splitter(args, **kwargs):
    return convert_coevalcube_to_healpix(*args, **kwargs)

def convert_coevalcube_to_healpix(inpcube, inpres, nside, freq=None, redshift=None,
                                 method='linear', rest_freq=CNST.rest_freq_HI,
                                 cosmo=None):

    """
    -----------------------------------------------------------------------------
    Covert a cosmological coeval cube at a given resolution (in physical comoving 
    distance) to HEALPIX coordinates of a specified nside covering the whole sky

    Inputs:

    inpcube     [numpy array] Cosmological cube in three dimensions of comoving
                distance 

    inpres      [scalar or tuple or list or numpy array] Input cube pixel 
                resolution (in comoving Mpc). If specified as scalar, it is 
                applied to all three dimensions. Otherwise a three-element tuple, 
                list or numpy array must be specified one for each dimension

    nside       [scalar] HEALPIX nside parameter for output HEALPIX map

    freq        [scalar] Frequency (in Hz) to be processed. One and only one of
                inputs freq or z (see below) must be set in order to determined
                the redshift at which this processing is to take place. Redshift
                is necessary to determine the cosmology. If set to None, 
                redshift must be specified (see below)

    redshift    [scalar] Redshift to be processed. One and only one of inputs
                freq (see above) or redshift must be specified. If set to None, 
                freq must be specified (see above)

    method      [string] Method of interpolation from cube to healpix pixels. 
                Accepted values are 'nearest_rounded' (fastest but not 
                accurate), and those accepted by the input keyword method in 
                scipy.interpolate.interpn(), namely, 'linear' and 'nearest', and 
                'splinef2d'. 'splinef2d' is only supported for 2-dimensional 
                data. Default='linear'

    rest_freq   [scalar] Rest frame frequency (in Hz) to be used in 
                determination of redshift. Will be used only if freq is set and 
                redshift is set to None. Default=1420405751.77 Hz (the rest 
                frame frequency of neutral Hydrogen spin flip transition)

    cosmo       [instance of class astropy.cosmology] Instance of class 
                astropy.cosmology to determine comoving distance for a given
                redshift. By default (None) it is set to WMAP9

    Output:

    HEALPIX lightcone cube of specified nside parameter. It is of shape npix
    -----------------------------------------------------------------------------
    """

    try:
        inpcube, nside, inpres
    except NameError:
        raise NameError('Inputs inpcube, nside and inpres must be specified')

    assert isinstance(inpcube, NP.ndarray), 'Input cube must be a numpy array'
    assert inpcube.ndim==3, 'Input cube must be a 3D numpy array'

    assert isinstance(nside, int), 'Parameter nside must be a scalar'
    assert HP.isnsideok(nside), 'Invalid nside parameter specified'

    assert isinstance(method, str), 'Method of interpolation must be a string'

    if cosmo is None:
        cosmo = cosmology.WMAP9
    assert isinstance(cosmo, cosmology.FLRW), 'Input cosmology must be an instance of class astropy.cosmology.FLRW' 

    if isinstance(inpres, (int,float)):
        inpres = NP.zeros(3) + inpres
    elif isinstance(inpres, (tuple,list,NP.ndarray)):
        inpres = NP.asarray(inpres).ravel()
        assert inpres.size==3, 'Input resolution must be a 3-element tuple, list or array'
    else:
        raise TypeError('Input resolution must be a scalar, list or numpy array')

    if (freq is None) and (redshift is None):
        raise ValueError('One and only one of redshift or freq must be specified')
    elif (freq is not None) and (redshift is not None):
        raise ValueError('One and only one of redshift or freq must be specified')
    else:
        if freq is not None:
            assert isinstance(freq, (int,float)), 'Input freq must be a scalar'

            redshift = rest_freq / freq - 1
        assert isinstance(redshift, (int,float)), 'Redshift must be a scalar'
        if redshift < 0.0:
            raise ValueError('Redshift must be positive')

    comoving_distance = cosmo.comoving_distance(redshift).value
    x, y, z = HP.pix2vec(nside, np.arange(HP.nside2npix(nside)))
    xmod = NP.mod(x*comoving_distance, inpres[0]*inpcube.shape[0])
    ymod = NP.mod(y*comoving_distance, inpres[1]*inpcube.shape[1])
    zmod = NP.mod(z*comoving_distance, inpres[2]*inpcube.shape[2])

    if method == 'nearest_rounded':
        xi = xmod / inpres[0]
        yi = ymod / inpres[1]
        zi = zmod / inpres[2]
        hpx = inpcube[xi.astype(int), yi.astype(int), zi.astype(int)]
    else:
        xyz_mod = NP.hstack((xmod.reshape(-1,1)), ymod.reshape(-1,1), zmod.reshape(-1,1))
        hpx = interpolate.interpn((inpres[0]*NP.arange(inpcube.shape[0]), inpres[1]*NP.arange(inpcube.shape[1]), inpres[2]*NP.arange(inpcube.shape[2])), inpcube, xyz_mod, method=method, bounds_error=False, fill_value=None)

    return hpx

#################################################################################

def convert_coevalcubes_to_healpix_surfaces(inpcubes, inpres, nside, redshifts=None,
                                            freqs=None, los_axis=-1, method='linear',
                                            rest_freq=CNST.rest_freq_HI, cosmo=None,
                                            parallel=False, nproc=None):

    """
    -----------------------------------------------------------------------------
    Covert array of comoving coeval cosmological cubes at a given resolution 
    (in physical comoving distance) to HEALPIX coordinates of a specified nside 
    covering the whole sky as lightcone cube

    Inputs:

    inpcubes    [numpy array] Cosmological evolving cubes in three dimensions of 
                comoving distance 

    inpres      [scalar or tuple or list or numpy array] Input cube pixel 
                resolution (in comoving Mpc). If specified as scalar, it is 
                applied to all three dimensions. Otherwise a three-element tuple, 
                list or numpy array must be specified one for each dimension

    nside       [scalar] HEALPIX nside parameter for output HEALPIX map

    freqs       [scalar] Frequency (in Hz) to be processed. One and only one of
                inputs freq or z (see below) must be set in order to determined
                the redshift at which this processing is to take place. Redshift
                is necessary to determine the cosmology. If set to None, 
                redshifts must be specified (see below)

    redshifts   [scalar] Redshift to be processed. One and only one of inputs
                freqs (see above) or redshifts must be specified. If set to 
                None, freqs must be specified (see above)

    los_axis    [integer] Denotes the axis that is along the line of sight.
                Default=-1 (last axis)

    method      [string] Method of interpolation from cube to healpix pixels. 
                Accepted values are 'nearest_rounded' (fastest but not 
                accurate), and those accepted by the input keyword method in 
                scipy.interpolate.interpn(), namely, 'linear' and 'nearest', and 
                'splinef2d'. 'splinef2d' is only supported for 2-dimensional 
                data. Default='linear'

    rest_freq   [scalar] Rest frame frequency (in Hz) to be used in 
                determination of redshift. Will be used only if freq is set and 
                redshifts is set to None. Default=1420405751.77 Hz (the rest 
                frame frequency of neutral Hydrogen spin flip transition)

    cosmo       [instance of class astropy.cosmology] Instance of class 
                astropy.cosmology to determine comoving distance for a given
                redshift. When set to None (default) it is set to WMAP9. 

    parallel    [boolean] If set to False (default), do serial processing. If
                set to True, do parallel processing with number of threads as
                specified in nproc

    nproc       [scalar] Number of parallel threads to use. Default=None means
                it will be set to the number of cores in the system.

    Output:

    HEALPIX maps of specified nside parameter for each of the redshifts or 
    frequencies as lightcone cube. It will be a numpy array of shape nchan x npix
    -----------------------------------------------------------------------------
    """

    try:
        inpcubes, nside, inpres
    except NameError:
        raise NameError('Inputs inpcubes, nside and inpres must be specified')

    assert isinstance(inpcubes, NP.ndarray), 'Input cube must be a numpy array'
    assert inpcubes.ndim==4, 'Input cubes must be specified as a 4D numpy array (3 spatial and 1 spectral/redshift)'

    assert isinstance(nside, int), 'Parameter nside must be a scalar'
    assert HP.isnsideok(nside), 'Invalid nside parameter specified'

    assert isinstance(method, str), 'Method of interpolation must be a string'

    assert isinstance(los_axis, int), 'Input los_axis must be an integer'
    assert inpcubes.ndim > los_axis+1, 'Input los_axis exceeds the dimensions of the input cubes'

    if (freqs is None) and (redshifts is None):
        raise ValueError('One and only one of redshifts or freqs must be specified')
    elif (freqs is not None) and (redshifts is not None):
        raise ValueError('One and only one of redshifts or freqs must be specified')
    else:
        if freqs is not None:
            assert isinstance(freqs, (int,float,NP.ndarray)), 'Input freqs must be a scalar or a numpy array'
            freqs = NP.asarray(freqs).reshape(-1)
            redshifts = rest_freq / freqs - 1
            list_redshifts = [None for i in xrange(redshifts.size)]
            list_freqs = freqs.tolist()
        else:
            redshifts = NP.asarray(redshifts).reshape(-1)
            list_redshifts = redshifts.tolist()
            freqs = rest_freq / (1 + redshifts)
            list_freqs = [None for i in xrange(freqs.size)]
        assert isinstance(redshifts, (int,float,NP.ndarray)), 'Redshifts must be a scalar or a numpy array'
        if NP.any(redshifts < 0.0):
            raise ValueError('Redshift must be positive')

    assert inpcubes.shape[axis]==redshift.size, 'Dimension along los_axis of inpcubes is mismatched with number of redshifts'

    if isinstance(inpres, (int,float)):
        inpres = inpres + NP.zeros(redshifts.size)
        inpres = inpres.tolist()
    elif isinstance(inpres, list):
        assert len(inpres)==redshifts.size, 'Number of elements in inpres must match the number of redshifts'
    else:
        raise TypeError('Input resolution must be a scalar or a list')

    assert isinstance(parallel, bool), 'Input parallel must be a boolean'
    
    hpxsurfaces = []
    if parallel:
        try:
            list_inpcubes = [NP.take(inpcubes, ind, axis=los_axis) for ind in xrange(redshifts.size)]
            list_nsides = [nside i in xrange(redshifts.size)]
            list_methods = [method for i in xrange(redshifts.size)]
            list_rest_freqs = [rest_freq for i in xrange(redshifts.size)]
            list_cosmo = [cosmo for i in xrange(redshifts.size)]
        
            if nproc is None:
                nproc = MP.cpu_count()
            assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
            nproc = min([nproc, redshifts.size])
            pool = MP.Pool(processes=nproc)
            hpxsurfaces = pool.map(convert_coevalcube_to_healpix_arg_splitter, IT.izip(list_inpcubes, inpres, list_nsides, list_freqs, list_redshifts, list_methods, list_rest_freqs, list_cosmo))
            pool.close()
            pool.join()
        except MemoryError:
            parallel = False
            del list_inpcubes
            del pool
            hpxsurfaces = []
            warnings.warn('Memory requirements too high. Downgrading to serial processing.')
    if not parallel:
        for ind in range(redshifts.size):
            hpxsurfaces += [convert_coevalcube_to_healpix(NP.take(inpcubes, ind, axis=los_axis), inpres[ind], nside, freq=list_freqs[ind], redshift=list_redshifts[ind], method=method, rest_freq=rest_freq, cosmo=cosmo)]

    hpxsurfaces = NP.asarray(hpxsurfaces)
    return hpxsurfaces

#################################################################################

