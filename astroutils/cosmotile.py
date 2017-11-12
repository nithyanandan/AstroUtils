import os, sys
from os.path import basename
import numpy as NP
from scipy import interpolate
import scipy as SP
import healpy as HP
import h5py
import astropy.cosmology as cosmology
import multiprocessing as MP
import itertools as IT
import time, warnings
import constants as CNST
import catalog as SM
import mathops as OPS
import DSP_modules as DSP

#################################################################################

def read_21cmfast_cube(cubefile, dtype=NP.float32):
    
    """
    -----------------------------------------------------------------------------
    Read 21cmfast cosmological cubes -- coeval or lightcone versions. Use faster
    version fastread_21cmfast_cube()

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

    with open(cubefile, 'rb') as fileobj:
        data = fileobj.read()
    data = NP.fromstring(data, dtype)
    if sys.byteorder == 'big':
        data = data.byteswap()
    data = NP.asarray(data.reshape((dim, dim, dim), order='F'), order='C') # convert to C-order from F-order (it has to be read in F-order first)
    return data

#################################################################################

def fastread_21cmfast_cube(cubefile, dtype=NP.float32):
    
    """
    -----------------------------------------------------------------------------
    Read 21cmfast cosmological cubes -- coeval or lightcone versions. This is 
    faster than read_21cmfast_cube() because it uses numpy.fromfile()

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

    data = NP.fromfile(cubefile, dtype=dtype, count=-1)
    if sys.byteorder == 'big':
        data = data.byteswap()
    data = data.reshape((dim, dim, dim), order='F') # it has to be read in F-order
    return data

#################################################################################

def cube_smooth_downsample_save(inpdict):
    
    """
    -----------------------------------------------------------------------------
    Read, smooth, downsample and save a cube

    Inputs:

    inpdict     [dictionary] It should contain the following keys and values:
                'infile'    [string] Filename containing the raw coeval cube in
                            binary format
                'process_stage'
                            [string] Indicates whether the input file is 'raw'
                            (default) or 'processed' 
                'smooth_axes'
                            [int or list/array of ints] Axes to be smoothed as a
                            list. 
                'smooth_scale'
                            [int, float or list] Smoothing scales (in units of 
                            pixels). If it is a scalar, it will be applied to all
                            axes specified in 'smooth_axes' otherwise if given as
                            a list, its length must match the number of elements 
                            in 'smooth_axes'. If set to None, no smoothing is 
                            done.
                'downsample_axes'
                            [int or list/array of ints] Axes to be downsampled 
                            as a list. Value under 'indata' must contain these 
                            axes
                'downsample_factor'
                            [int, float or list] Downsampling factor. If it is a 
                            scalar, it will be applied to all axes specified in 
                            'downsample_axes' otherwise if given as
                            a list, its length must match the number of elements 
                            in 'downsample_axes'. If set to None, no downsampling 
                            is done.
                'outfile'   [string] Filename to save the smoothed and optionally 
                            downsampled cube in HDF5 format. No extension should 
                            be provided as it will be determined internally
    -----------------------------------------------------------------------------
    """

    process_stage = 'raw'
    if 'process_stage' in inpdict:
        if inpdict['process_stage'].lower() not in ['raw', 'processed']:
            raise ValueError('Process stage of input data must be set to "raw" or "processed"')
        process_stage = inpdict['process_stage'].lower()
            
    if process_stage == 'raw':
        data = fastread_21cmfast_cube(inpdict['infile'])
    else:
        data, hdrinfo = read_coeval_cube(inpdict['infile'])
    indata_dims = NP.asarray(data.shape)
    smooth_downsample_dict = {'indata': data, 'smooth_axes': inpdict['smooth_axes'], 'downsample_axes': inpdict['downsample_axes'], 'smooth_scale': inpdict['smooth_scale'], 'downsample_factor': inpdict['downsample_factor']}
    data = smooth_downsample_cube(smooth_downsample_dict)
    if process_stage == 'raw':
        inpres = inpdict['inpres']
    else:
        inpres = hdrinfo['pixres']
    outres = inpres * indata_dims / NP.asarray(data.shape)
    hdrinfo = {'pixres': outres}
    write_coeval_cube(data, inpdict['outfile'], hdrinfo=hdrinfo)

#################################################################################

def smooth_downsample_cube(inpdict):
    
    """
    -----------------------------------------------------------------------------
    Smooth a cube and optionally downsample

    Inputs:

    inpdict     [dictionary] contains info for smoothing and downsampling. It has
                the following keys and values:
                'indata'    [numpy array] Coeval cube. Usually it is in 3D
                'smooth_axes'
                            [int or list/array of ints] Axes to be smoothed as a
                            list. Value under 'indata' must contain these axes
                'smooth_scale'
                            [int, float or list] Smoothing scales (in units of 
                            pixels). If it is a scalar, it will be applied to all
                            axes specified in 'smooth_axes' otherwise if given as
                            a list, its length must match the number of elements 
                            in 'smooth_axes'. If set to None, no smoothing is 
                            done.
                'downsample_axes'
                            [int or list/array of ints] Axes to be downsampled 
                            as a list. Value under 'indata' must contain these 
                            axes
                'downsample_factor'
                            [int, float or list] Downsampling factor. If it is a 
                            scalar, it will be applied to all axes specified in 
                            'downsample_axes' otherwise if given as
                            a list, its length must match the number of elements 
                            in 'downsample_axes'. If set to None, no downsampling 
                            is done.

    Output:

    Numpy array containing the smoothed (and optionally downsampled) cube
    -----------------------------------------------------------------------------
    """

    data = inpdict['indata']
    if not isinstance(data, NP.ndarray):
        raise TypeError('Value in key "indata" must be a numpy array')

    if 'smooth_axes' in inpdict:
        smooth_axes = inpdict['smooth_axes']
        if smooth_axes is not None:
            if isinstance(smooth_axes, int):
                smooth_axes = NP.asarray(smooth_axes).reshape(-1)
            elif isinstance(smooth_axes, (list, NP.ndarray)):
                smooth_axes = NP.asarray(smooth_axes).reshape(-1)
            else:
                raise TypeError('Value under key "smooth_axes" must be an integer or numpy array')
    
            smooth_axes = NP.unique(smooth_axes)
            if NP.any(smooth_axes >= data.ndim):
                raise ValueError('One or more of smoothing axis not found in input array')
    else:
        smooth_axes = None

    if 'downsample_axes' in inpdict:
        downsample_axes = inpdict['downsample_axes']
        if downsample_axes is not None:
            if isinstance(downsample_axes, int):
                downsample_axes = NP.asarray(downsample_axes).reshape(-1)
            elif isinstance(downsample_axes, (list, NP.ndarray)):
                downsample_axes = NP.asarray(downsample_axes).reshape(-1)
            else:
                raise TypeError('Value under key "downsample_axes" must be an integer or numpy array')
        
            downsample_axes = NP.unique(downsample_axes)
            if NP.any(downsample_axes >= data.ndim):
                raise ValueError('One or more of downsample axis not found in input array')
    else:
        downsample_axes = None
    
    if 'smooth_scale' in inpdict:
        smooth_scale = inpdict['smooth_scale']
        if smooth_scale is not None:
            if isinstance(smooth_scale, (int,float,list,NP.ndarray)):
                smooth_scale = NP.asarray(smooth_scale).reshape(-1)
            else:
                raise TypeError('Value under key "smooth_scale" is not of the correct type')
            if smooth_axes is None:
                smooth_axes = NP.arange(data.ndim)
            if smooth_scale.size == 1:
                smooth_scale = smooth_scale + NP.zeros(smooth_axes.size)
            elif smooth_scale.size != smooth_axes.size:
                raise ValueError('Smooth scales and axes not compatible')
    else:
        smooth_scale = None

    if 'downsample_factor' in inpdict:
        downsample_factor = inpdict['downsample_factor']
        if downsample_factor is not None:
            if isinstance(downsample_factor, (int,float,list,NP.ndarray)):
                downsample_factor = NP.asarray(downsample_factor).reshape(-1)
            else:
                raise TypeError('Value under key "downsample_factor" is not of the correct type')
            if downsample_axes is None:
                downsample_axes = NP.arange(data.ndim)
            if downsample_factor.size == 1:
                downsample_factor = downsample_factor + NP.zeros(downsample_axes.size)
            elif downsample_factor.size != downsample_axes.size:
                raise ValueError('Downsample factors and axes not compatible')
    else:
        downsample_factor = None

    if smooth_scale is not None:
        if smooth_scale.size == data.ndim:
            data = SP.ndimage.filters.gaussian_filter(data, smooth_scale)
        else:
            for si, smax in enumerate(smooth_axes):
                data = SP.ndimage.filters.gaussian_filter1d(data, smooth_scale[si], axis=smax)

    if downsample_factor is not None:
        for di, dsax in enumerate(downsample_axes):
            data = DSP.downsampler(data, downsample_factor[di], axis=dsax)

    return data

#################################################################################

def interp_coevalcubes_arg_splitter(args, **kwargs):
    return interp_coevalcubes(*args, **kwargs)

def interp_coevalcubes_inpdict(inpdict):

    """
    -----------------------------------------------------------------------------
    Interpolate between coeval cosmological cubes at specified parameter values
    (usually redshift or frequency) to get the coeval cubes at required parameter
    values. Wrapper for interp_coevalcubes()

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
        exec(key + '=val')

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
        process_stage
    except NameError:
        process_stage = 'raw'
        
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

    return interp_coevalcubes(invals, outvals, inpcubes=inpcubes, cubefiles=cubefiles, cubedims=cubedims, cube_source=cube_source, process_stage=process_stage, interp_method=interp_method, outfiles=outfiles, returncubes=returncubes)

#################################################################################

def interp_coevalcubes(invals, outvals, inpcubes=None, cubefiles=None,
                       cubedims=None, cube_source=None, process_stage='raw', 
                       interp_method='linear', outfiles=None, returncubes=True):

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

    process_stage
                [string] Indicates whether the input file is 'raw' (default) or 
                'processed' 

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

    assert isinstance(invals, (int, float, list, NP.ndarray)), 'Input values of interpolated variable must be a scalar, list or numpy array'
    invals = NP.asarray(invals).reshape(-1)

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

    assert isinstance(process_stage, str), 'Input process_stage must be a string'
    if process_stage.lower() not in ['raw', 'processed']:
        raise ValueError('Input process_stage must be set to "raw" or "processed"')

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
            if cubedims is not None:
                if not isinstance(cubedims, (int,list,tuple,NP.ndarray)):
                    raise TypeError('Input cubedims must be specified as an integer, list, tuple or numpy array')
                if isinstance(cubedims, int):
                    cubedims = NP.asarray([cubedims, cubedims, cubedims])
                else:
                    cubedims = NP.asarray(cubedims).reshape(-1)
                if cubedims.size != 3:
                    raise ValueError('Input cubedims must be a three element iterable')
            
            print '\nReading in 21cmfast cubes...'
            ts = time.time()

            if process_stage.lower() == 'raw':
                inpcubes = [fastread_21cmfast_cube(cubefile) for cubefile in cubefiles]
            else:
                inpcubes = [read_coeval_cube(cubefile)[0] for cubefile in cubefiles]

            te = time.time()
            print 'Reading 21cmfast cubes took {0:.1f} seconds'.format(te-ts)

    interp_required = True
    if invals.size == outvals.size:
        if NP.allclose(invals, outvals): # no interpolation required, just return outcube=inpcubes
            outcubes = inpcubes
            interp_required = False
    if interp_required:
        inpcubes = NP.asarray(inpcubes)
        print 'Interpolating 21cmfast cube at desired parameter value(s)...'
        ts = time.time()
        
        outcubes = OPS.interpolate_array(inpcubes, invals, outvals, axis=0, kind=interp_method)

        te = time.time()
        print 'Interpolating 21cmfast cube at desired parameter value(s) took {0:.1f} seconds'.format(te-ts)

        outcubes = NP.array_split(outcubes, outcubes.shape[0], axis=0)
        outcubes = [NP.squeeze(ocube) for ocube in outcubes]
        # outcubes = [NP.take(outcubes, i, axis=0) for i in range(outvals.size)]

    if outfiles is not None:
        for fi,outfile in enumerate(outfiles):
            write_coeval_cube(outcubes[fi], outfile)

    if returncubes:
        return outcubes

#################################################################################

def convert_coevalcube_to_sphere_surface_arg_splitter(args, **kwargs):
    return convert_coevalcube_to_sphere_surface(*args, **kwargs)

def convert_coevalcube_to_sphere_surface_inpdict(inpdict):

    """
    -----------------------------------------------------------------------------
    Covert a cosmological coeval cube at a given resolution (in physical comoving 
    distance) to HEALPIX coordinates of a specified nside covering the whole sky
    or coordinates covering a spherical patch. Wrapper for 
    convert_coevalcube_to_sphere_surface() 

    Inputs:

    inpdict     [dictionary] Dictionary of parameters for tiling cosmological 
                coeval cubes to healpix lightcone cubes. It consists of the 
                following keys and values:
                inpcube     [numpy array] Cosmological cube in three dimensions 
                            of comoving distance 
                inpres      [scalar or tuple or list or numpy array] Input cube 
                            pixel resolution (in comoving Mpc). If specified as 
                            scalar, it is applied to all three dimensions. 
                            Otherwise a three-element tuple, list or numpy array 
                            must be specified one for each dimension
                nside       [scalar] HEALPIX nside parameter for output HEALPIX 
                            map. If set theta_phi will be ignored. 
                theta_phi   [numpy array] nsrc x 2 numpy array of theta and phi 
                            (in degrees) at which the lightcone surface should 
                            be evaluated. One and only one of nside or theta_phi 
                            must be specified.
                freq        [scalar] Frequency (in Hz) to be processed. One and 
                            only one of inputs freq or z (see below) must be set 
                            in order to determined the redshift at which this 
                            processing is to take place. Redshift is necessary 
                            to determine the cosmology. If set to None, redshift 
                            must be specified (see below)
                redshift    [scalar] Redshift to be processed. One and only one 
                            of inputs freq (see above) or redshift must be 
                            specified. If set to None, freq must be specified 
                            (see above)
                method      [string] Method of interpolation from cube to 
                            spherical surface pixels. Accepted values are 
                            'nearest_rounded' (fastest but not accurate), and 
                            those accepted by the input keyword method in 
                            scipy.interpolate.interpn(), namely, 'linear' and 
                            'nearest', and 'splinef2d'. 'splinef2d' is only 
                            supported for 2-dimensional data. Default='linear'
                rest_freq   [scalar] Rest frame frequency (in Hz) to be used in 
                            determination of redshift. Will be used only if 
                            freq is set and redshift is set to None. 
                            Default=1420405751.77 Hz (the rest frame frequency 
                            of neutral Hydrogen spin flip transition)
                cosmo       [instance of class astropy.cosmology] Instance of 
                            class astropy.cosmology to determine comoving 
                            distance for a given redshift. By default (None) it 
                            is set to WMAP9

    Output:

    Stacked lightcone surfaces covering spherical patch (whole sky using HEALPIX
    if nside is specified) or just at specified theta and phi coordinates. It is 
    of shape npix
    -----------------------------------------------------------------------------
    """

    try:
        inpdict
    except NameError:
        raise NameError('Input inpdict must be provided')

    if not isinstance(inpdict, dict):
        raise TypeError('Input inpdict must be a dictionary')

    for key,val in inpdict.iteritems():
        exec(key + '=val')

    try:
        inpcube, inpres
    except NameError:
        raise NameError('Inputs inpcube and inpres must be specified in inpdict')

    try:
        nside
    except NameError:
        nside = None

    try:
        theta_phi
    except NameError:
        theta_phi = None

    try:
        freq
    except NameError:
        freq = None

    try:
        redshift
    except NameError:
        redshift = None
    
    try:
        cosmo
    except NameError:
        cosmo = None

    try:
        method
    except NameError:
        method = 'linear'

    try:
        rest_freq
    except NameError:
        rest_freq = CNST.rest_freq_HI

    return convert_coevalcube_to_sphere_surface(inpcube, inpres, nside=nside, theta_phi=theta_phi, freq=freq, redshift=redshift, method=method, rest_freq=rest_freq, cosmo=cosmo)

#################################################################################

def convert_coevalcube_to_sphere_surface(inpcube, inpres, nside=None, 
                                         theta_phi=None, freq=None, 
                                         redshift=None, method='linear',
                                         rest_freq=CNST.rest_freq_HI, cosmo=None):

    """
    -----------------------------------------------------------------------------
    Covert a cosmological coeval cube at a given resolution (in physical comoving 
    distance) to specified coordinates on sphereical surface or whole sphere in 
    HEALPIX coordinates 

    Inputs:

    inpcube     [numpy array] Cosmological cube in three dimensions of comoving
                distance 

    inpres      [scalar or tuple or list or numpy array] Input cube pixel 
                resolution (in comoving Mpc). If specified as scalar, it is 
                applied to all three dimensions. Otherwise a three-element tuple, 
                list or numpy array must be specified one for each dimension

    nside       [scalar] HEALPIX nside parameter for output HEALPIX map. If set
                theta_phi will be ignored. 

    theta_phi   [numpy array] nsrc x 2 numpy array of theta and phi (in degrees)
                at which the lightcone surface should be evaluated. One and only
                one of nside or theta_phi must be specified.

    freq        [scalar] Frequency (in Hz) to be processed. One and only one of
                inputs freq or z (see below) must be set in order to determined
                the redshift at which this processing is to take place. Redshift
                is necessary to determine the cosmology. If set to None, 
                redshift must be specified (see below)

    redshift    [scalar] Redshift to be processed. One and only one of inputs
                freq (see above) or redshift must be specified. If set to None, 
                freq must be specified (see above)

    method      [string] Method of interpolation from cube to sphere pixels. 
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

    lightcone surface of specified patch on sphere or whole sphere in HEALPIX
    coordinates. It is of shape nsrc
    -----------------------------------------------------------------------------
    """

    try:
        inpcube, inpres
    except NameError:
        raise NameError('Inputs inpcube and inpres must be specified')

    assert isinstance(inpcube, NP.ndarray), 'Input cube must be a numpy array'
    assert inpcube.ndim==3, 'Input cube must be a 3D numpy array'

    if (nside is None) and (theta_phi is None):
        raise TypeError('One of the inputs nside or theta_phi must not be None')
    elif (nside is not None) and (theta_phi is not None):
        raise TypeError('One and only one of the inputs nside or theta_phi must not be None')
    elif nside is not None:
        assert isinstance(nside, int), 'Parameter nside must be a scalar'
        assert HP.isnsideok(nside), 'Invalid nside parameter specified'
    else:
        assert isinstance(theta_phi, NP.ndarray), 'Input theta_phi must be a numpy array'
        assert theta_phi.ndim==2, 'Input theta_phi must be a 2D numpy array'
        assert theta_phi.shape[1]==2, 'Input theta_phi must be a nsrc x 2 array'

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
    if nside is not None:
        x, y, z = HP.pix2vec(nside, NP.arange(HP.nside2npix(nside)))
    else:
        theta_phi = NP.radians(theta_phi)
        z = NP.cos(theta_phi[:,0])
        x = NP.sin(theta_phi[:,0]) * NP.cos(theta_phi[:,1])
        y = NP.sin(theta_phi[:,0]) * NP.sin(theta_phi[:,1])
    xmod = NP.mod(x*comoving_distance, inpres[0]*inpcube.shape[0])
    ymod = NP.mod(y*comoving_distance, inpres[1]*inpcube.shape[1])
    zmod = NP.mod(z*comoving_distance, inpres[2]*inpcube.shape[2])

    print 'Interpolating to spherical surface...'
    if method == 'nearest_rounded':
        xi = xmod / inpres[0]
        yi = ymod / inpres[1]
        zi = zmod / inpres[2]
        patch = inpcube[xi.astype(int), yi.astype(int), zi.astype(int)]
    else:
        xyz_mod = NP.hstack((xmod.reshape(-1,1), ymod.reshape(-1,1), zmod.reshape(-1,1)))
        patch = interpolate.interpn((inpres[0]*NP.arange(inpcube.shape[0]), inpres[1]*NP.arange(inpcube.shape[1]), inpres[2]*NP.arange(inpcube.shape[2])), inpcube, xyz_mod, method=method, bounds_error=False, fill_value=None)
    print 'Interpolated to spherical surface'

    return patch

#################################################################################

def convert_coevalcubes_to_sphere_surfaces(inpcubes, inpres, nside=None,
                                           theta_phi=None, redshifts=None,
                                           freqs=None, los_axis=-1, method='linear',
                                           rest_freq=CNST.rest_freq_HI, cosmo=None,
                                           parallel=False, nproc=None):

    """
    -----------------------------------------------------------------------------
    Convert array of comoving coeval cosmological cubes at a given resolution 
    (in physical comoving distance) to HEALPIX coordinates of a specified nside 
    covering the whole sky or just a spherical patch using given theta and phi 
    coordinates with output as stacked lightcone surfaces

    Inputs:

    inpcubes    [numpy array] Array of cosmological coeval cubes in which each 
                element has is a 3D numpy array with three dimensions of comoving 
                distance 

    inpres      [scalar or tuple or list or numpy array] Input cube pixel 
                resolution (in comoving Mpc). If specified as scalar, it is 
                applied to all three dimensions. Otherwise a three-element tuple, 
                list or numpy array must be specified one for each dimension

    nside       [scalar] HEALPIX nside parameter for output HEALPIX map. If set 
                theta_phi will be ignored. 

    theta_phi   [numpy array] nsrc x 2 numpy array of theta and phi (in degrees) 
                at which the lightcone surface should be evaluated. One and only 
                one of nside or theta_phi must be specified.

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

    method      [string] Method of interpolation from cube to spherical surface 
                pixels. Accepted values are 'nearest_rounded' (fastest but not 
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

    Stacked spherical surfaces either covering whole sky (using nside and 
    HEALPIX) or a patch at specified theta and phi for each of the redshifts or 
    frequencies. It will be a numpy array of shape nchan x npix
    -----------------------------------------------------------------------------
    """

    try:
        inpcubes, inpres
    except NameError:
        raise NameError('Inputs inpcubes and inpres must be specified')

    assert isinstance(inpcubes, NP.ndarray), 'Input cube must be a numpy array'
    assert inpcubes.ndim==4, 'Input cubes must be specified as a 4D numpy array (3 spatial and 1 spectral/redshift)'

    if (nside is None) and (theta_phi is None):
        raise TypeError('One of the inputs nside or theta_phi must not be None')
    elif (nside is not None) and (theta_phi is not None):
        raise TypeError('One and only one of the inputs nside or theta_phi must not be None')
    elif nside is not None:
        assert isinstance(nside, int), 'Parameter nside must be a scalar'
        assert HP.isnsideok(nside), 'Invalid nside parameter specified'
    else:
        assert isinstance(theta_phi, NP.ndarray), 'Input theta_phi must be a numpy array'
        assert theta_phi.ndim==2, 'Input theta_phi must be a 2D numpy array'
        assert theta_phi.shape[1]==2, 'Input theta_phi must be a nsrc x 2 array'

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
    
    sphsurfaces = []
    if parallel:
        try:
            list_inpcubes = [NP.take(inpcubes, ind, axis=los_axis) for ind in xrange(redshifts.size)]
            list_nsides = [nside] * redshifts.size
            list_theta_phi = [theta_phi] * redshifts.size
            list_methods = [method] * redshifts.size
            list_rest_freqs = [rest_freq] * redshifts.size
            list_cosmo = [cosmo] * redshifts.size
        
            if nproc is None:
                nproc = MP.cpu_count()
            assert isinstance(nproc, int), 'Number of parallel processes must be an integer'
            nproc = min([nproc, redshifts.size])
            pool = MP.Pool(processes=nproc)
            sphsurfaces = pool.map(convert_coevalcube_to_sphere_surface_arg_splitter, IT.izip(list_inpcubes, inpres, list_nsides, list_theta_phi, list_freqs, list_redshifts, list_methods, list_rest_freqs, list_cosmo))
            pool.close()
            pool.join()
        except MemoryError:
            parallel = False
            del list_inpcubes
            del pool
            sphsurfaces = []
            warnings.warn('Memory requirements too high. Downgrading to serial processing.')
    if not parallel:
        for ind in range(redshifts.size):
            sphsurfaces += [convert_coevalcube_to_sphere_surface(NP.take(inpcubes, ind, axis=los_axis), inpres[ind], nside=nside, theta_phi=theta_phi, freq=list_freqs[ind], redshift=list_redshifts[ind], method=method, rest_freq=rest_freq, cosmo=cosmo)]

    sphsurfaces = NP.asarray(sphsurfaces)
    return sphsurfaces

#################################################################################

def coeval_interp_cube_to_sphere_surface_wrapper_arg_splitter(args, **kwargs):
    return coeval_interp_cube_to_sphere_surface_wrapper(*args, **kwargs)

def coeval_interp_cube_to_sphere_surface_wrapper(interpdict, tiledict):

    """
    -----------------------------------------------------------------------------
    Interpolate cosmological coeval cubes and transform to produce healpix 
    lightcone cube

    Inputs:

    interpdict  [dictionary] See docstring of interp_coevalcubes_inpdict()
                input, namely, inpdict

    tiledict    [dictionary] See docstring of 
                convert_coevalcube_to_sphere_surface_inpdict() input, namely, 
                inpdict

    Output:

    Stacked lightcone surfaces of specified sphereical patch (whole sphere if 
    nside parameter given). It is of shape npix
    -----------------------------------------------------------------------------
    """
    
    try:
        interpdict, tiledict
    except NameError:
        raise NameError('Inputs interpdict and tiledict must be specified')

    interpcube = interp_coevalcubes_inpdict(interpdict)[0] # List should contain only one element
    tiledict['inpcube'] = interpcube
    return convert_coevalcube_to_sphere_surface_inpdict(tiledict)
    
#################################################################################

def write_lightcone_surfaces(light_cone_surfaces, units, outfile, freqs,
                             cosmo=None, is_healpix=False):

    """
    -----------------------------------------------------------------------------
    Write light cone surfaces to HDF5 file

    Inputs:

    light_cone_surfaces
                [numpy array] Light cone surfaces. Must be of shape nchan x npix

    units       [string] Units of the values in light_cone_surfaces 

    outfile     [string] Filename to write the output to

    freqs       [numpy array] The frequencies corresponding to the surfaces. 
                It is of size nchan and units in 'Hz'

    cosmo       [dictionary or instance of class astropy.cosmology.FLRW] 
                Cosmological parameters. If specified as dictionary, it must 
                contain the following keys and values (no defaults):
                'Om0'   [float] Matter density parameter at z=0
                'Ode0'  [float] Dark energy density parameter at z=0
                'Ob0'   [float] Baryon energy density parameter at z=0
                'h'     [float] Hubble constant factor in units of km/s/Mpc
                'w0'    [float] Dark energy equation of state parameter at z=0

    is_healpix  [boolean] The axis=1 of light_cone_surfaces represents a
                HEALPIX surface if set to True. If False (default), it may not 
                denote a HEALPIX surface. 
    -----------------------------------------------------------------------------
    """

    try:
        light_cone_surfaces, units, outfile, freqs
    except NameError:
        raise NameError('Inputs light_cone_surfaces, units, outfile, freqs must be provided')

    if not isinstance(light_cone_surfaces, NP.ndarray):
        raise TypeError('Input light_cone_surfaces must be a numpy array')
    if light_cone_surfaces.ndim != 2:
        raise ValueError('Input light_cone_surfaces must be a two-dimensional numpy array')

    if not isinstance(units, str):
        raise TypeError('Input units must be specified')
    
    if not isinstance(outfile, str):
        raise TypeError('Output file must be specified as a string')

    if not isinstance(freqs, NP.ndarray):
        raise TypeError('Input freqs must be a numpy array')
    freqs = freqs.ravel()
    if freqs.size != light_cone_surfaces.shape[0]:
        raise ValueError('Size of input freqs must be same as that in light_cone_surfaces')
    if NP.any(freqs <= 0.0):
        raise ValueError('Input freqs must be negative')

    if not isinstance(is_healpix, bool):
        raise TypeError('Input is_healpix must be a boolean')

    cosmoinfo = {'Om0': None, 'Ode0': None, 'h': None, 'Ob0': None, 'w0': None}
    req_keys = cosmoinfo.keys()
    if cosmo is None:
        cosmo = cosmology.WMAP9
        cosmoinfo = {'Om0': cosmo.Om0, 'Ode0': cosmo.Ode0, 'h': cosmo.h, 'Ob0': cosmo.Ob0, 'w0': cosmo.w(0.0)}
    elif isinstance(cosmo, dict):
        for key in cosmoinfo:
            if key not in cosmo:
                raise KeyError('Input cosmo is missing "{0}" value'.format(key))
            if cosmo[key] is None:
                raise ValueError('Cosmological parameter values cannot be set to None. No defaults can be assumed')
            cosmoinfo[key] = cosmo[key]
    elif isinstance(cosmo, cosmology.FLRW):
        cosmoinfo = {'Om0': cosmo.Om0, 'Ode0': cosmo.Ode0, 'h': cosmo.h, 'Ob0': cosmo.Ob0, 'w0': cosmo.w(0.0)}
    else:
        raise TypeError('Input cosmology must be an instance of class astropy.cosmology.FLRW')

    with h5py.File(outfile, 'w') as fileobj:
        hdr_grp = fileobj.create_group('header')
        hdr_grp['units'] = units
        hdr_grp['is_healpix'] = int(is_healpix)
        spec_grp = fileobj.create_group('specinfo')
        # spec_grp['freqs'] = freqs
        freq_dset = spec_grp.create_dataset('freqs', (freqs.size,), maxshape=(None,), data=freqs.ravel())
        freq_dset.attrs['units'] = 'Hz'
        cosmo_grp = fileobj.create_group('cosmology')
        for key in cosmoinfo:
            cosmo_grp[key] = cosmoinfo[key]
        surfaces_grp = fileobj.create_group('skyinfo')
        dset = surfaces_grp.create_dataset('surfaces', light_cone_surfaces.shape, maxshape=(None, None), data=light_cone_surfaces, chunks=(1,light_cone_surfaces.shape[1]), compression='gzip', compression_opts=9)

#################################################################################

def write_lightcone_catalog(init_parms, outfile=None, action='return'):

    """
    -----------------------------------------------------------------------------
    Write light cone surfaces to HDF5 file that can be read in as an instance of
    class SkyModel

    Inputs:

    init_parms  [dictionary] Dictionary containing parameters used to create
                an instance of class SkyModel. Sky model Initialization 
                parameters are specified using the following keys and values
                (identical to those used in initializing an instance of class
                SkyModel):
                'name'      [scalar or vector] Name of the catalog. If 
                            scalar, will be used for all sources in the sky 
                            model. If vector, will be used for corresponding 
                            object. If vector, size must equal the number of 
                            objects.
                'frequency' [scalar or vector] Frequency range for which sky 
                            model is applicable. Units in Hz.
                'location'  [numpy array or list of lists] Positions of the 
                            sources in sky model. Each position is specified 
                            as a row (numpy array) or a 2-element list which 
                            is input as a list of lists for all the sources 
                            in the sky model
                'spec_type' [string] specifies the flux variation along the 
                            spectral axis. Allowed values are 'func' and 
                            'spectrum'. It must be set to 'spectrum' and values
                            for key 'spectrum' (see below) must be specified.
                'spectrum'  [numpy array] Spectrum of the catalog. Will be 
                            applicable if attribute spec_type is set to 
                            'spectrum'. It must be of shape nsrc x nchan
                'src_shape' [3-column numpy array or list of 3-element 
                            lists] source shape specified by major axis 
                            FWHM (first column), minor axis FWHM (second 
                            column), and position angle (third column). The 
                            major and minor axes and position angle are 
                            stored in degrees. The number of rows must match 
                            the number of sources. Position angle is in 
                            degrees east of north (same convention as local 
                            azimuth)
                'epoch'     [string] Epoch appropriate for the coordinate 
                            system. Default is 'J2000'
                'coords'    [string] Coordinate system used for the source 
                            positions in the sky model. Currently accepted 
                            values are 'radec' (RA-Dec)
                'src_shape_units' 
                            [3-element list or tuple of strings] Specifies 
                            the units of major axis FWHM, minor axis FWHM, 
                            and position angle. Accepted values for major 
                            and minor axes units are 'arcsec', 'arcmin', 
                            'degree', or 'radian'. Accepted values for 
                            position angle units are 'degree' or 'radian'

    outfile     [string] Output filename including full path omitting the
                extension (.hdf5) which will be appended automatically. This will
                occur only if action is set to 'store'

    action      [string] Specifies if the instance of class SkyModel is to be 
                returned if action='return' (default) or save to file specified 
                in outfile if action='store'
    -----------------------------------------------------------------------------
    """

    if not isinstance(action, str):
        raise TypeError('Input action must be a string')
    if action.lower() not in ['store', 'return']:
        raise ValueError('Input action must be set to "store" or "return"')
    if action.lower() == 'store':
        if not isinstance(outfile, str):
            raise TypeError('Output filename must be a string')

    skymod = SM.SkyModel(init_file=None, init_parms=init_parms)
    if (skymod.spec_type != 'spectrum') and (skymod.spectrum is not None):
        raise ValueError('Input data must be specified in the form of a spectrum')

    if action.lower() == 'store':
        skymod.save(outfile, fileformat='hdf5')
    else:
        return skymod

#################################################################################

def write_coeval_cube(data, outfile, hdrinfo=None):

    """
    -----------------------------------------------------------------------------
    Write cosmological coeval cubes to HDF5 file

    Inputs:

    cube        [numpy array] Coeval cube. Usually 3D.

    outfile     [string] Filename including full path where the data is to be
                saved in HDF5 format. It should not include the extension as it
                will be determined internally

    hdrinfo     [dictionary] Any header information in the form of a dictionary
    -----------------------------------------------------------------------------
    """

    if not isinstance(data, NP.ndarray):
        raise TypeError('Input data must be a numpy array')

    if not isinstance(outfile, str):
        raise TypeError('outfile must be a string')

    with h5py.File(outfile, 'w') as fileobj:
        if hdrinfo is not None:
            if not isinstance(hdrinfo, dict):
                raise TypeError('Input hdrinfo must be a dictionary')
            hdr_grp = fileobj.create_group('header')
            for key in hdrinfo:
                hdr_grp[key] = hdrinfo[key]

        dset = fileobj.create_dataset('data', data=data, compression='gzip', compression_opts=9)
        
#################################################################################

def read_coeval_cube(infile):

    """
    -----------------------------------------------------------------------------
    Read processed cosmological coeval cubes from HDF5 file

    Inputs:

    infile      [string] Filename including full path where the processed data 
                is to be read in HDF5 format. It should not include the extension 
                as it will be determined internally

    Output:

    Tuple containing processed 21cmfast coeval cube as a 3D numpy array and a 
    dictionary that contains header information (set to None if no header info 
    found)
    -----------------------------------------------------------------------------
    """

    if not isinstance(infile, str):
        raise TypeError('infile must be a string')

    hdrinfo = None
    with h5py.File(infile, 'r') as fileobj:
        if 'header' in fileobj:
            hdrinfo = {key: fileobj['header'][key].value for key in fileobj['header']}
        if 'data' not in fileobj:
            raise KeyError('Input HDF5 file does not contain data')
        data = fileobj['data'].value

    return (data, hdrinfo)

#################################################################################

