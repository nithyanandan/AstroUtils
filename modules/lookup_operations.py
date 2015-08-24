import numpy as NP
from astropy.io import ascii
import multiprocessing as MP
import itertools as IT
try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

#################################################################################

def find_NN_arg_splitter(args, **kwargs):
    return find_NN_pp(*args, **kwargs)

def find_NN_pp(ngbrof, ngbrin, distance_ULIM):
    kdtself = KDT(ngbrof)
    kdtother = KDT(ngbrin)
    indNN_list = kdtself.query_ball_tree(kdtother, distance_ULIM, p=2.0)
    return indNN_list

#################################################################################

def gen_lookup(x, y, data, file):

    """
    -----------------------------------------------------------------------------
    Generate a lookup table given two-dimensional coordinates and values at these
    coordinates.

    Inputs:

    x          [numpy vector] x-coordinates for the lookup values.

    y          [numpy vector] y-coordinates for the lookup values

    data       [numpy vector] data values at the coordinates given by x and y. x,
               y, and data must all have the same size

    file       [string] Name of the output file to write the lookup table to. It 
               will be created as an ascii table with 'x', 'y', 'real_value' and 
               'imag_value' (if imagianry value present) being the column headers

    -----------------------------------------------------------------------------
    """

    try:
        x, y, data, file
    except NameError:
        raise NameError('To generate a lookup table, input parameters x, y, data, and file should be specified.')

    if (x.size != y.size) or (x.size != data.size):
        raise ValueError('x, y, and data must be of same size.')

    if not isinstance(file, str):
        raise TypeError('Input parameter file must be of string data type')

    if NP.iscomplexobj(data):
        try:
            ascii.write([x.ravel(), y.ravel(), data.ravel().real, data.ravel().imag], file, names=['x', 'y', 'real_value', 'imag_value'])
        except IOError:
            raise IOError('Could not write to specified file: '+file)
    else:
        try:
            ascii.write([x.ravel(), y.ravel(), data.ravel()], file, names=['x', 'y', 'real_value'])
        except IOError:
            raise IOError('Could not write to specified file: '+file)
    
#################################################################################

def read_lookup(file):

    """
    -----------------------------------------------------------------------------
    Read data from a lookup database.

    Inputs:

    file      [string] Input file containing the lookup data base.

    Outputs:

    [tuple] each element of the tuple is a numpy array. The elements in order are
            x-coordinates, y-coordinates, data value at those coordiantes. The
            data values are real or complex depending on whether the lookup table
            has an 'imag_value' column
    -----------------------------------------------------------------------------
    """

    if not isinstance(file, str):
        raise TypeError('Input parameter file must be of string data type')

    try:
        cols = ascii.read(file, data_start=1, comment='#')
    except IOError:
        raise IOError('Could not read the specified file: '+file)

    if 'imag_value' in cols.colnames:
        return cols['x'].data, cols['y'].data, cols['real_value'].data+1j*cols['imag_value'].data
    else:
        return cols['x'].data, cols['y'].data, cols['real_value'].data

#################################################################################

def lookup(x, y, val, xin, yin, distance_ULIM=NP.inf, oob_value=0.0,
           remove_oob=True, tol=None, maxmatch=None):

    """
    -----------------------------------------------------------------------------
    Perform a lookup operation based on nearest neighbour algorithm using
    KD-Trees when the lookup database and required coordinate locations are
    specified.

    Inputs:
    
    x       [numpy array] x-coordinates in the lookup table

    y       [numpy array] y-coordinates in the lookup table

    val     [numpy array] values at the x and y locations. x, y, and val must be
            of same size

    xin     [numpy array] x-coordinates at which values are required

    yin     [numpy array] y-coordinates at which values are required. xin and yin
            must be of same size

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    oob_value
            [scalar] Value to be returned at the location if the nearest
            neighbour for the location is not found inside distance_ULIM.
            Default = 0

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    maxmatch
            [scalar] A positive value indicating maximum number of input 
            locations (xin, yin) to be assigned. Default = None. If set to None, 
            all the input locations specified are assigned values. For instance,
            to have only one input location to be populated per antenna, use
            maxmatch=1. 

    tol     [scalar] If set, only lookup data with abs(val) > tol will be 
            considered for nearest neighbour lookup. Default = None implies all
            lookup values will be considered for nearest neighbour determination.
            tol is to be interpreted as a minimum value considered as significant
            in the lookup table.

    Outputs: 

    Returns a tuple of nearest neighbour lookup value, the index of input
    location and nearest neighbour distance in the lookup table. The tuple
    consists of the following elements in this order. The number of pairs
    matched will be the lesser of those determined by distNN and maxmatch

    ibind   [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of ibind is equal to size of xin or yin, 
            otherwise size of ibind is less than or equal to size of xin or yin. 
            Values of ibind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    nnval   [numpy vector] nearest neighbour lookup values for the given input
            locations. If remove_oob is not set to True, size of nnval is equal
            to size of xin or yin, otherwise size of nnval is less than or equal
            to size of xin or yin. In such a case, out of bound locations are
            filled with oob_value. Corresponding values in distNN are returned as
            inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 
    
    -----------------------------------------------------------------------------
    """

    try:
        x, y, val, xin, yin
    except NameError:
        raise NameError('x, y, val, xin, and yin must be specified for lookup operation.')

    if (x.size != y.size) or (x.size != val.size):
        raise ValueError('x, y, and val must be of same size.')

    if xin.size != yin.size:
        raise ValueError('Input parameters xin and yin must have same size.')

    if tol is not None:
        if isinstance(tol, (int, float)):
            if tol < 0.0:
                raise ValueError('tol value must be non-negative')
            x = x[NP.abs(val) >= tol]
            y = y[NP.abs(val) >= tol]
            val = val[NP.abs(val) >= tol]
        else:
            raise TypeError('tol must be a scalar integer or float')

    kdt = KDT(zip(x,y))
    dist, ind = kdt.query(NP.hstack((xin.reshape(-1,1), yin.reshape(-1,1))), k=1, distance_upper_bound=distance_ULIM)
    nnval = NP.zeros(ind.size, dtype=val.dtype)
    nnval[dist < distance_ULIM] = val[ind[dist < distance_ULIM]]
    nnval[dist >= distance_ULIM] = oob_value
    ibind = NP.arange(ind.size) # in-bound indices
    distNN = dist
    if remove_oob:
        nnval = nnval[dist <= distance_ULIM]
        ibind = ibind[dist <= distance_ULIM]
        distNN = dist[dist <= distance_ULIM]

    if maxmatch is not None:
        if not isinstance(maxmatch, int):
            raise TypeError('maxmatch must be an integer.')

        if maxmatch < distNN.size:
            if maxmatch > 0:
                sortind = NP.argsort(distNN)[:maxmatch] # Indices of maxmatch nearest matches
                nnval = nnval[sortind]
                ibind = ibind[sortind]
                distNN = distNN[sortind]
            else:
                raise ValueError('Maximum number of nearest neighbours must be positive.')

    return ibind, nnval, distNN

#################################################################################

def lookup_1NN_old(ref, val, inp, distance_ULIM=NP.inf, oob_value=0.0,
                   remove_oob=True, tol=None, maxmatch=None):

    """
    -----------------------------------------------------------------------------
    Perform a lookup operation based on the first nearest neighbour algorithm 
    using KD-Trees when the lookup database and required coordinate locations are
    specified.

    Inputs:
    
    ref     [numpy array] Reference locations to be looked up to. NxK numpy 
            array which represents N points in K-dimensional coordinates

    val     [numpy array] values at the x and y locations. x, y, and val must be
            of same size

    inp     [numpy array] Input locations for which nearest neighbours will be
            searched in reference locations specified in ref. MxK numpy array
            representing M points in K-dimensional coordinates. Must have same
            number of columns as input parameter ref.

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    maxmatch
            [scalar] A positive value indicating maximum number of input 
            locations (xin, yin) to be assigned. Default = None. If set to None, 
            all the input locations specified are assigned values. For instance,
            to have only one input location to be populated per antenna, use
            maxmatch=1. 

    tol     [scalar] If set, only lookup data with abs(val) > tol will be 
            considered for nearest neighbour lookup. Default = None implies all
            lookup values will be considered for nearest neighbour determination.
            tol is to be interpreted as a minimum value considered as significant
            in the lookup table.

    oob_value
            [scalar] Value to be returned at the location if the nearest
            neighbour for the location is not found inside distance_ULIM.
            Default = 0

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a tuple in the following order:
    index of input location, nearest neighbour lookup
    value, nearest neighbour distance from the lookup table, and the index of the
    nearest neighbour in the lookup table. 
    The number of pairs matched will be the lesser of those determined by distNN
    and maxmatch

    ibind   [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of ibind is equal to size of xin or yin, 
            otherwise size of ibind is less than or equal to size of xin or yin. 
            Values of ibind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    nnval   [numpy vector] nearest neighbour lookup values for the given input
            locations. If remove_oob is not set to True, size of nnval is equal
            to size of xin or yin, otherwise size of nnval is less than or equal
            to size of xin or yin. In such a case, out of bound locations are
            filled with oob_value. Corresponding values in distNN are returned as
            inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 
    
    refind  [numpy vector] indices of the nearest neighbours in the reference 
            lookup table. i.e., ref[refind,:] are the nearest neighbours of 
            inp[inpind] with distances distNN.

    -----------------------------------------------------------------------------
    """

    try:
        ref, val, inp
    except NameError:
        raise NameError('ref, val, and inp must be specified for lookup operation.')

    if (ref.shape[0] != val.size):
        raise ValueError('ref and val must contain same number of entries')

    if tol is not None:
        if isinstance(tol, (int, float)):
            if tol < 0.0:
                raise ValueError('tol value must be non-negative')
            ref = ref[NP.abs(val) >= tol, :]
            val = val[NP.abs(val) >= tol]
        else:
            raise TypeError('tol must be a scalar integer or float')

    inpind, distNN, refind = find_1NN_old(ref, inp, distance_ULIM=distance_ULIM, remove_oob=remove_oob, maxmatch=maxmatch)
    nnval = NP.zeros(refind.size, dtype=val.dtype)
    nnval[distNN < distance_ULIM] = val[refind[distNN < distance_ULIM]]
    nnval[distNN >= distance_ULIM] = oob_value

    if maxmatch is not None:
        if not isinstance(maxmatch, int):
            raise TypeError('maxmatch must be an integer.')

        if maxmatch < distNN.size:
            if maxmatch > 0:
                sortind = NP.argsort(distNN)[:maxmatch] # Indices of maxmatch nearest matches
                nnval = nnval[sortind]
                inpind = inpind[sortind]
                distNN = distNN[sortind]
                refind = refind[sortind]
            else:
                raise ValueError('Maximum number of nearest neighbours must be positive.')

    return inpind, nnval, distNN, refind

#################################################################################

def find_1NN_old(ref, inp, distance_ULIM=NP.inf, remove_oob=True, maxmatch=None):

    """
    -----------------------------------------------------------------------------
    Find the first nearest neighbour of input locations to a set of reference 
    locations using KD-Trees nearest neighbour algorithm.

    Inputs:
    
    ref     [numpy array] Reference locations to be looked up to. NxK numpy 
            array which represents N points in K-dimensional coordinates

    inp     [numpy array] Input locations for which nearest neighbours will be
            searched in reference locations specified in ref. MxK numpy array
            representing M points in K-dimensional coordinates. Must have same
            number of columns as input parameter ref.

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a tuple in the following order:
    index of input location (inpind), nearest neighbour distance in the lookup 
    table (distNN), and the index of the nearest neighbour in the lookup table 
    (refind).
    The number of pairs matched will be the lesser of those determined by distNN
    and maxmatch

    inpind  [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of inpind is equal to size of xin or yin, 
            otherwise size of inpind is less than or equal to size of xin or yin. 
            Values of inpind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 

    refind  [numpy vector] indices of the nearest neighbours in the reference 
            lookup table. i.e., ref[refind,:] are the nearest neighbours of 
            inp[inpind] with distances distNN.

    -----------------------------------------------------------------------------
    """

    try:
        ref, inp
    except NameError:
        raise NameError('ref and inp must be specified for lookup operation.')

    if (ref.shape[1] != inp.shape[1]):
        raise ValueError('ref and inp must contain same number of columns')

    kdt = KDT(ref)
    distNN, refind = kdt.query(inp, k=1, distance_upper_bound=distance_ULIM)
    inpind = NP.arange(refind.size) # in-bound indices
    if remove_oob:
        inpind = inpind[distNN <= distance_ULIM]
        refind = refind[distNN <= distance_ULIM]
        distNN = distNN[distNN <= distance_ULIM]

    return inpind, distNN, refind

#################################################################################

def lookup_1NN(ref, val, inp, distance_ULIM=NP.inf, oob_value=0.0,
                   remove_oob=True, tol=None, maxmatch=None):

    """
    -----------------------------------------------------------------------------
    Perform a lookup operation based on the first nearest neighbour algorithm 
    using KD-Trees when the lookup database and required coordinate locations are
    specified.

    Inputs:
    
    ref     [numpy array] Reference locations to be looked up to. NxK numpy 
            array which represents N points in K-dimensional coordinates

    val     [numpy array] values at the x and y locations. x, y, and val must be
            of same size

    inp     [numpy array] Input locations for which nearest neighbours will be
            searched in reference locations specified in ref. MxK numpy array
            representing M points in K-dimensional coordinates. Must have same
            number of columns as input parameter ref.

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    maxmatch
            [scalar] A positive value indicating maximum number of input 
            locations (xin, yin) to be assigned. Default = None. If set to None, 
            all the input locations specified are assigned values. For instance,
            to have only one input location to be populated per antenna, use
            maxmatch=1. 

    tol     [scalar] If set, only lookup data with abs(val) > tol will be 
            considered for nearest neighbour lookup. Default = None implies all
            lookup values will be considered for nearest neighbour determination.
            tol is to be interpreted as a minimum value considered as significant
            in the lookup table.

    oob_value
            [scalar] Value to be returned at the location if the nearest
            neighbour for the location is not found inside distance_ULIM.
            Default = 0

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a tuple in the following order:
    index of input location, nearest neighbour lookup
    value, nearest neighbour distance from the lookup table, and the index of the
    nearest neighbour in the lookup table. 
    The number of pairs matched will be the lesser of those determined by distNN
    and maxmatch

    ibind   [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of ibind is equal to size of xin or yin, 
            otherwise size of ibind is less than or equal to size of xin or yin. 
            Values of ibind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    nnval   [numpy vector] nearest neighbour lookup values for the given input
            locations. If remove_oob is not set to True, size of nnval is equal
            to size of xin or yin, otherwise size of nnval is less than or equal
            to size of xin or yin. In such a case, out of bound locations are
            filled with oob_value. Corresponding values in distNN are returned as
            inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 
    
    refind  [numpy vector] indices of the nearest neighbours in the reference 
            lookup table. i.e., ref[refind,:] are the nearest neighbours of 
            inp[inpind] with distances distNN.

    -----------------------------------------------------------------------------
    """

    try:
        ref, val, inp
    except NameError:
        raise NameError('ref, val, and inp must be specified for lookup operation.')

    if (ref.shape[0] != val.size):
        raise ValueError('ref and val must contain same number of entries')

    if tol is not None:
        if isinstance(tol, (int, float)):
            if tol < 0.0:
                raise ValueError('tol value must be non-negative')
            ref = ref[NP.abs(val) >= tol, :]
            val = val[NP.abs(val) >= tol]
        else:
            raise TypeError('tol must be a scalar integer or float')

    inpind, distNN, refind = find_1NN(ref, inp, distance_ULIM=distance_ULIM, remove_oob=remove_oob, maxmatch=maxmatch)
    nnval = NP.zeros(refind.size, dtype=val.dtype)
    nnval[distNN < distance_ULIM] = val[refind[distNN < distance_ULIM]]
    nnval[distNN >= distance_ULIM] = oob_value

    if maxmatch is not None:
        if not isinstance(maxmatch, int):
            raise TypeError('maxmatch must be an integer.')

        if maxmatch < distNN.size:
            if maxmatch > 0:
                sortind = NP.argsort(distNN)[:maxmatch] # Indices of maxmatch nearest matches
                nnval = nnval[sortind]
                inpind = inpind[sortind]
                distNN = distNN[sortind]
                refind = refind[sortind]
            else:
                raise ValueError('Maximum number of nearest neighbours must be positive.')

    return inpind, nnval, distNN, refind

#################################################################################

def find_1NN(ref, inp, distance_ULIM=NP.inf, remove_oob=True):

    """
    -----------------------------------------------------------------------------
    Find the first nearest neighbour of input locations to a set of reference 
    locations using KD-Trees nearest neighbour algorithm.

    Inputs:
    
    ref     [numpy array] Reference locations to be looked up to. NxK numpy 
            array which represents N points in K-dimensional coordinates

    inp     [numpy array] Input locations for which nearest neighbours will be
            searched in reference locations specified in ref. MxK numpy array
            representing M points in K-dimensional coordinates. Must have same
            number of columns as input parameter ref.

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a tuple in the following order:
    index of input location (inpind), nearest neighbour distance in the lookup 
    table (distNN), and the index of the nearest neighbour in the lookup table 
    (refind).
    The number of pairs matched will be the lesser of those determined by distNN
    and maxmatch

    inpind  [numpy vector] indices of the input locations for which nearest 
            neighbours were found from the lookup table. If remove_oob is 
            not set to True, size of inpind is equal to size of xin or yin, 
            otherwise size of inpind is less than or equal to size of xin or yin. 
            Values of inpind equal to size of lookup table values indicates the 
            nearest neighbours for the corresponding input location could not be 
            found within the distance upper bound from the lookup table. The
            corresponding values in distNN are returned as inf or NP.inf

    distNN  [numpy vector] distance to the nearest neighbour in the lookup table
            for the given input locations. If remove_oob is not set to True, size 
            of distNN is equal to size of xin or yin, otherwise size of distNN is 
            less than or equal to size of xin or yin. In such a case, out of 
            bound locations are filled with inf or NP.inf. 

    refind  [numpy vector] indices of the nearest neighbours in the reference 
            lookup table. i.e., ref[refind,:] are the nearest neighbours of 
            inp[inpind] with distances distNN.

    -----------------------------------------------------------------------------
    """

    try:
        ref, inp
    except NameError:
        raise NameError('ref and inp must be specified for lookup operation.')

    if (ref.shape[1] != inp.shape[1]):
        raise ValueError('ref and inp must contain same number of columns')

    kdt = KDT(ref)
    distNN, refind = kdt.query(inp, k=1, distance_upper_bound=distance_ULIM)
    inpind = NP.arange(refind.size) # in-bound indices
    if remove_oob:
        inpind = inpind[distNN <= distance_ULIM]
        refind = refind[distNN <= distance_ULIM]
        distNN = distNN[distNN <= distance_ULIM]

    # kdt = KDT(inp)
    # distNN, inpind = kdt.query(ref, k=1, distance_upper_bound=distance_ULIM)
    # refind = NP.arange(inpind.size) # in-bound indices
    # if remove_oob:
    #     inpind = inpind[distNN <= distance_ULIM]
    #     refind = refind[distNN <= distance_ULIM]
    #     distNN = distNN[distNN <= distance_ULIM]

    return inpind, refind, distNN

#################################################################################

def find_1NN_pp(ref, inp, pid, outq, distance_ULIM=NP.inf, remove_oob=True):

    """
    -----------------------------------------------------------------------------
    Find the first nearest neighbour of input locations to a set of reference 
    locations using KD-Trees nearest neighbour algorithm. Identical to 
    find_1NN_pp() but should be used only in case of parallel programming
    applications

    Inputs:
    
    ref     [numpy array] Reference locations to be looked up to. NxK numpy 
            array which represents N points in K-dimensional coordinates

    inp     [numpy array] Input locations for which nearest neighbours will be
            searched in reference locations specified in ref. MxK numpy array
            representing M points in K-dimensional coordinates. Must have same
            number of columns as input parameter ref.

    pid     [any scalar] process id which will be used as a key in the returned 
            dictionary

    outq    [instance of multiprocessing.Queue] Output dictionary will be 
            returned in outq

    distance_ULIM
            [scalar] A positive number for the upper bound on distance while
            searching for nearest neighbours. Neighbours outside of this upper
            bound are not searched for. Default = NP.inf (infinite distance upper 
            bound means nearest neighbours will be searched all the way out to 
            infinite distance). Should be in the same units as x, y, xin and yin

    remove_oob
            [boolean] If set to True, results of nearest neighbour search and 
            lookup are returned only for those input locations which have a 
            neighbour within the distance upper bound. Locations with no 
            neighbours inside the distance upper bound will not be in the 
            returned results. Default = True. If set to False, even if no 
            nearest neighbours are found within the distance upper bound, an
            infinite distance and out of bound index from the lookup table are
            returned.

    Outputs: 

    Returns a dictionary with key given by pid and contains a nested dictionary
    with the following keys and values: 
    'inpind'  [numpy vector] indices of the input locations for which nearest 
              neighbours were found from the lookup table. If remove_oob is 
              not set to True, size of inpind is equal to size of xin or yin, 
              otherwise size of inpind is less than or equal to size of xin or 
              yin. Values of inpind equal to size of lookup table values 
              indicates the nearest neighbours for the corresponding input 
              location could not be found within the distance upper bound from 
              the lookup table. The corresponding values in distNN are returned 
              as inf or NP.inf

    'distNN'  [numpy vector] distance to the nearest neighbour in the lookup 
              table for the given input locations. If remove_oob is not set to 
              True, size of distNN is equal to size of xin or yin, otherwise 
              size of distNN is less than or equal to size of xin or yin. In 
              such a case, out of bound locations are filled with inf or NP.inf. 

    'refind'  [numpy vector] indices of the nearest neighbours in the reference 
              lookup table. i.e., ref[refind,:] are the nearest neighbours of 
              inp[inpind] with distances distNN.


    The number of pairs matched will be the lesser of those determined by distNN
    and maxmatch
    -----------------------------------------------------------------------------
    """

    try:
        ref, inp
    except NameError:
        raise NameError('ref and inp must be specified for lookup operation.')

    if (ref.shape[1] != inp.shape[1]):
        raise ValueError('ref and inp must contain same number of columns')

    kdt = KDT(ref)
    distNN, refind = kdt.query(inp, k=1, distance_upper_bound=distance_ULIM)
    inpind = NP.arange(refind.size) # in-bound indices
    if remove_oob:
        inpind = inpind[distNN <= distance_ULIM]
        refind = refind[distNN <= distance_ULIM]
        distNN = distNN[distNN <= distance_ULIM]

    # kdt = KDT(inp)
    # distNN, inpind = kdt.query(ref, k=1, distance_upper_bound=distance_ULIM)
    # refind = NP.arange(inpind.size) # in-bound indices
    # if remove_oob:
    #     inpind = inpind[distNN <= distance_ULIM]
    #     refind = refind[distNN <= distance_ULIM]
    #     distNN = distNN[distNN <= distance_ULIM]

    outdict = {}
    outdict[pid] = {}
    outdict[pid]['inpind'] = inpind
    outdict[pid]['distNN'] = distNN
    outdict[pid]['refind'] = refind

    # print MP.current_process().name
    outq.put(outdict)

#################################################################################

def find_NN(ngbrof, ngbrin, distance_ULIM=NP.inf, flatten=False, parallel=False,
            nproc=None):

    """
    -----------------------------------------------------------------------------
    Find all nearest neighbours of one set of locations in another set of 
    locations within a specified distance. 

    Inputs:

    ngbrof    [numpy array] Locations for nearest neighbours are to be 
              determined. Has dimensions MxK where M is the number of locations.

    ngbrin    [numpy array] Locations from which nearest neighbours are to be 
              chosen for the locations in ngbrof. Has dimensions NxK.

    distance_ULIM
              [scalar] Maximum search radius to look for neighbours. 
              Default=NP.inf

    flatten   [boolean] If set to True, flattens the output of the nearest
              neighbour search algorithm to yield two separate sets of matching
              indices - one for ngbrof and the other for ngbrin. Default=False

    parallel  [boolean] specifies if parallelization is to be invoked. False 
              (default) means only serial processing. Parallelization is done
              over ngbrof

    nproc     [scalar] specifies number of independent processes to spawn.
              Default=None, means automatically determines the number of 
              process cores in the system and use one less than that to 
              avoid locking the system for other processes. Applies only 
              if input parameter 'parallel' (see above) is set to True. 
              If nproc is set to a value more than the number of process
              cores in the system, it will be reset to number of process 
              cores in the system minus one to avoid locking the system out 
              for other processes

    Outputs:

    List containing three items. The first item is a list of M lists where each 
    of the M inner lists corresponds to one entry in ngbrof and the elements in 
    the inner list contains indices to ngbrin that are the nearest neighbours of 
    that specific ngbrof (same as output of cKDTree.query_ball_tree()). The 
    second item in the output list is a numpy array of indices to ngbrof 
    (obtained from the first item if input keyword flatten is set to True) or 
    None (if input keyword flatten is set to False). The third item in the output 
    list is a numpy array of indices to ngbrin that is a valid neighbour of 
    ngbrof (obtained from the first item if input keyword flatten is set to 
    True) or None (if input keyword flatten is set to False). 
    -----------------------------------------------------------------------------
    """

    try:
        ngbrof, ngbrin
    except NameError:
        raise NameError('ngbrof and ngbrin must be specified for finding nearest neighbours.')

    if (ngbrof.shape[1] != ngbrin.shape[1]):
        raise ValueError('ngbrof and ngbrin must contain same number of columns')

    if parallel or (nproc is not None):
        if nproc is None:
            nproc = max(MP.cpu_count()-1, 1) 
        else:
            nproc = min(nproc, max(MP.cpu_count()-1, 1))

        split_ind = NP.arange(ngbrof.shape[0]/nproc, ngbrof.shape[0], ngbrof.shape[0]/nproc)
        split_ngbrof_list = NP.split(ngbrof, split_ind, axis=0)
        ngbrin_list = [ngbrin] * len(split_ngbrof_list)
        distance_ULIM_list = [distance_ULIM] * len(split_ngbrof_list)

        pool = MP.Pool(processes=nproc)
        lolol = pool.map(find_NN_arg_splitter, IT.izip(split_ngbrof_list, ngbrin_list, distance_ULIM_list))
        pool.close()
        pool.join()

        indNN_list = [subitem for item in lolol for subitem in item]
    else:
        kdtself = KDT(ngbrof)
        kdtother = KDT(ngbrin)
        indNN_list = kdtself.query_ball_tree(kdtother, distance_ULIM, p=2.0)

    ind_ngbrof = None
    ind_ngbrin = None
    if flatten:
        list_of_ind_tuples = [(i,ind) for i,item in enumerate(indNN_list) for ind in item]
        ind_ngbrof, ind_ngbrin = zip(*list_of_ind_tuples)

    return [indNN_list, NP.asarray(ind_ngbrof), NP.asarray(ind_ngbrin)]
    
#################################################################################

