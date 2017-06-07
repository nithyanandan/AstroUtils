import os, sys
from os.path import basename
import numpy as NP
import healpy as HP
import astropy.cosmology as cosmology
import multiprocessing as MP
import warnings
import constants as CNST
import mathops as OPS

#################################################################################

def read_21cmfast_cube(cubefile, dtype=NP.float32):
    
    """
    -----------------------------------------------------------------------------
    Read 21cmfast cosmological cubes -- coeval or lightcone versions

    Inputs:

    cubefile    [string] Filename including full path for reading in the input 
                21cmfast cosmological cube. 21cmfast cubes are stored in binary 
                files. 

    Output:

    Numpy array containing the cosmological coeval/lightcone 21cmfast cube. It 
    is of shape nx x ny x nz where this is determined by cube dimensions parsed
    from the cubefile string
    -----------------------------------------------------------------------------
    """

    if basename(cubefile)[-11:]=='lighttravel':
        dim = int('' + cubefile.split('_')[-3])
        label = str('' + cubefile.split('_')[-2])
    else:
        dim = int('' + cubefile.split('_')[-2])
        label = str('' + cubefile.split('_')[-1])

    with open(filename, 'rb') as fileobj:
        data = fileobj.read()
    data = NP.fromstring(data, dtype)
    if sys.byteorder == 'big':
        data = data.byteswap()
    data = NP.asarray(data.reshape((dim, dim, dim), order='F'), order='C') # convert to C-order from F-order
    return data

#################################################################################

def interp_coevalcubes_arg_splitter(args, **kwargs):
    return interp_coevalcubes(*args, **kwargs)

def interp_coevalcubes_inpdict(inpdict):

    """
    -----------------------------------------------------------------------------
    Interpolate between coeval cosmological cubes at specified parameter values
    (usually redshift or frequency) to get the coeval cubes at required parameter
    values

    Inputs:

    inpdict     [dictionary] Dictionary of interpolation parameters for coeval
                cosmological cubes. It consists of the following keys and 
                values:
                invals      [list or numpy array] Locations using which the 
                            interpolation function is determined. It must be of 
                            size equal to the dimension of input array along 
                            which interpolation is to be determined specified 
                            by axis keyword input. It must be a list or numpy 
                            array. This key is mandatory
                outvals     [list or numpy array] Locations at which 
                            interpolated array is to be determined along the 
                            specified axis. It must be a scalar, list or numpy 
                            array. If any of outloc is outside the range of 
                            inploc, the first and the last cubes from the 
                            inparray will be used as boundary values. This key 
                            is mandatory
                inpcubes    [list of numpy arrays] List of cosmological coeval 
                            cubes in which each element has is a 3D numpy array 
                            with three dimensions of comoving distance. If set 
                            to None (default), cubefiles which contain the input 
                            cubes must be specified. If set to not None, length 
                            of the list must match the number of elements in 
                            invals. Only one of inpcubes or cubefiles can be set
                cubefiles   [list of strings] Filenames for reading in the input 
                            coeval cubes. If set to None (default), inpcubes 
                            must be specified. If set to not None, it must 
                            contain a list of filenames and length of list must 
                            match the number of elements in invals. Only one of 
                            inpcubes or cubefiles can be set
                cubedims    [integer, list, tuple or numpy array] Dimensions of 
                            the input cubes, will be used when input cubes are 
                            read from cubefiles especially when these are binary 
                            files such as from 21cmfast simulations. If 
                            specified as integer, it will be applied to all 
                            input cubes read from cubefiles, otherwise if 
                            specified as a list, tuple or numpy array, it must 
                            contain three elements one along each axis and this 
                            will be applied to all input cubes read from 
                            cubefiles. It is not applicable when input cubes are 
                            given directly in inpcubes
                cube_source [string] Source of input cubes. At the moment, the 
                            accepted values are '21cmfast'
                method      [string] Method of interpolation across coeval cubes 
                            along axis for which invals are provided. Usually it 
                            is the spectral, redshift or line of sight distance. 
                            Accepted values are 'linear', 'nearest', 'zero', 
                            'slinear', 'quadratic', 'cubic' where 'slinear', 
                            'quadratic' and 'cubic' refer to a spline 
                            interpolation of first, second or third order or as 
                            an integer specifying the order of the spline 
                            interpolator to use. Default='linear'.
                outfiles    [list of strings] Filenames for writing interpolated 
                            coeval cubes. If set to None (default), interpolated 
                            coeval cubes are not written out. If set to not 
                            None, it must be a list of strings where each 
                            element in the list corresponds to filename of an 
                            interpolated coeval cubes. The number of elements in 
                            this  list must match the number of elements in 
                            outvals. 
                returncubes [boolean] If set to True (default), the interpolated 
                            coeval cubes are returned as a list of coeval cubes. 
                            Thus each element in the list is a coeval cube and 
                            corresponds to the value in outvals. If set to 
                            False, the interpolated coeval cubes are not 
                            returned.

    Output:

    If input returncubes is set to True, the interpolated coeval cubes are 
    returned as a list of coeval cubes. Thus each element in the list is a 
    coeval cube and corresponds to the value in outvals. 
    -----------------------------------------------------------------------------
    """

    try:
        inpdict
    except NameError:
        raise NameError('Input inpdict must be specified')

    if not isinstance(inpdict, dict):
        raise TypeError('Input inpdict must be a dictionary')

    for key,val in inpdict.iteritems():
        eval(key + '=val')

    try:
        invals, outvals
    except NameError:
        raise NameError('Inputs invals and outvals must be specified')

    try:
        inpcubes
    except NameError:
        inpcubes = None

    try:
        cubefiles
    except NameError:
        cubefiles = None

    try:
        cubedims
    except NameError:
        cubedims = None

    try:
        cube_source
    except NameError:
        cube_source = None

    try:
        interp_method
    except NameError:
        interp_method = 'linear'
        
    try:
        outfiles
    except NameError:
        outfiles = None

    try:
        returncubes
    except NameError:
        returncubes = True

    return interp_coevalcubes(invals, outvals, inpcubes=inpcubes, cubefiles=cubefiles, cubedims=cubedims, cube_source=cube_source, interp_method=interp_method, outfiles=outfile, returncubes=returncubes)

#################################################################################

def interp_coevalcubes(invals, outvals, inpcubes=None, cubefiles=None,
                       cubedims=None, cube_source=None, interp_method='linear',
                       outfiles=None, returncubes=True):

    """
    -----------------------------------------------------------------------------
    Interpolate between coeval cosmological cubes at specified parameter values
    (usually redshift or frequency) to get the coeval cubes at required parameter
    values

    Inputs:

    invals      [list or numpy array] Locations using which the interpolation 
                function is determined. It must be of size equal to the 
                dimension of input array along which interpolation is to be 
                determined specified by axis keyword input. It must be a list or 
                numpy array

    outvals     [list or numpy array] Locations at which interpolated array is
                to be determined along the specified axis. It must be a scalar, 
                list or numpy array. If any of outloc is outside the range of
                inploc, the first and the last cubes from the inparray will
                be used as boundary values

    inpcubes    [list of numpy arrays] List of cosmological coeval cubes in 
                which each element has is a 3D numpy array with three dimensions 
                of comoving distance. If set to None (default), cubefiles which 
                contain the input cubes must be specified. If set to not None, 
                length of the list must match the number of elements in invals. 
                Only one of inpcubes or cubefiles can be set

    cubefiles   [list of strings] Filenames for reading in the input coeval 
                cubes. If set to None (default), inpcubes must be specified. If
                set to not None, it must contain a list of filenames and length 
                of list must match the number of elements in invals. Only one of 
                inpcubes or cubefiles can be set

    cubedims    [integer, list, tuple or numpy array] Dimensions of the input
                cubes, will be used when input cubes are read from cubefiles
                especially when these are binary files such as from 21cmfast 
                simulations. If specified as integer, it will be applied to all
                input cubes read from cubefiles, otherwise if specified as a 
                list, tuple or numpy array, it must contain three elements one
                along each axis and this will be applied to all input cubes read
                from cubefiles. It is not applicable when input cubes are given
                directly in inpcubes

    cube_source [string] Source of input cubes. At the moment, the accepted 
                values are '21cmfast'

    method      [string] Method of interpolation across coeval cubes along axis
                for which invals are provided. Usually it is the spectral, 
                redshift or line of sight distance. Accepted values are 
                'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' 
                where 'slinear', 'quadratic' and 'cubic' refer to a spline 
                interpolation of first, second or third order or as an integer 
                specifying the order of the spline interpolator to use. 
                Default='linear'.

    outfiles    [list of strings] Filenames for writing interpolated coeval 
                cubes. If set to None (default), interpolated coeval cubes are 
                not written out. If set to not None, it must be a list of 
                strings where each element in the list corresponds to filename 
                of an interpolated coeval cubes. The number of elements in this 
                list must match the number of elements in outvals. 

    returncubes [boolean] If set to True (default), the interpolated coeval
                cubes are returned as a list of coeval cubes. Thus each element
                in the list is a coeval cube and corresponds to the value in
                outvals. If set to False, the interpolated coeval cubes are not
                returned.

    Output:

    If input returncubes is set to True, the interpolated coeval cubes are 
    returned as a list of coeval cubes. Thus each element in the list is a 
    coeval cube and corresponds to the value in outvals. 
    -----------------------------------------------------------------------------
    """

    try:
        invals, outvals
    except NameError:
        raise NameError('Inputs invals and outvals must be specified')

    assert isinstance(outvals, (int, float, list, NP.ndarray)), 'Output values of interpolated variable must be a scalar, list or numpy array'
    outvals = NP.asarray(outvals).reshape(-1)

    if outfiles is not None:
        if isinstance(outfiles, str):
            if outvals.size != 1:
                raise TypeError('Number of outfiles must match the number of output interpolated values')
            outfiles = [outfiles]
        elif isinstance(outfiles, list):
            if len(outfiles) != outvals.size:
                raise TypeError('Number of outfiles must match the number of output interpolated values')
        else:
            raise TypeError('outfiles must be a string or list of strings')

    if (cubefiles is None) and (inpcubes is None):
        raise ValueError('One of the inputs cubefiles and inpcubes must be specified')
    elif (cubefiles is not None) and (inpcubes is not None):
        raise ValueError('One and only one of the inputs cubefiles and inpcubes must be specified')
    elif cubefiles is not None:
        if not isinstance(cube_source, str):
            raise TypeError('Input cube_source must be a string')
        if cube_source.lower() not in ['21cmfast']:
            raise ValueError('Processing of {0} cubes not supported currently',format(cube_source))
        if cube_source.lower() == '21cmfast':
            if not isinstance(cubedims, (int,list,tuple,NP.ndarray)):
                raise TypeError('Input cubedims must be specified as an integer, list, tuple or numpy array')
            if isinstance(cubedims, int):
                cubedims = NP.asarray([cubedims, cubedims, cubedims])
            else:
                cubedims = NP.asarray(cubedims).reshape(-1)
            if cubedims.size != 3:
                raise ValueError('Input cubedims must be a three element iterable')
            
            inpcubes = [read_21cmfast_cube(cubefile) for cubefile in cubefiles]

    if NP.allclose(invals, outvals): # no interpolation required, just return outcube=inpcubes
        outcubes = inpcubes
    else:
        inpcubes = NP.asarray(inpcubes)
        outcubes = OPS.interpolate_array(inpcubes, invals, outvals, axis=0, kind=interp_method)
        outcubes = [NP.take(outcubes, i, axis=0) for i in range(outvals.size)]

    if outfiles is not None:
        for fi,outfile in enumerate(outfiles):
            write_coeval_cube(outfile, outcubes[fi]):

    if returncubes:
        return outcubes

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

    inpcubes    [numpy array] Array of cosmological coeval cubes in which each 
                element has is a 3D numpy array with three dimensions of comoving 
                distance 

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

