import sys
import numpy as NP

def reverse(inp, axis=0, ind_range=[-1,-1]):

    """
    ---------------------------------------------------------------
    A generic function to reverse a specific axis or a subset of 
    indices of this specific axis of a multi-dimensional array. This 
    works on data up to 8 dimensions.

    Input:

    inp        Multi-dimensional array (up to 8 dimensions)

    Keyword Inputs:

    axis      [scalar, default = 0] The axis along which the array 
              is to be reversed while preserving the order of the other 
              axes. 0 <= axis <= 7

    ind_range [2-element list] The lower and upper limits of indices
              of the axis over which the data is to be reversed. 

    Output:

    The array with its data reversed over a subset or the entirety 
    of the specified axis.
    ---------------------------------------------------------------
    """

    inp = NP.asarray(inp)

    try:
        isinstance(inp, NP.ndarray)
        # type(inp) is numpy.ndarray
    except TypeError: 
        print 'Unable to convert to Numpy array data type'
        sys.exit(1) # Abort execution

    shp = NP.shape(inp)
    ndim = len(shp)
    
    if ndim > 8:
        print "Input data with more than 8 dimensions not supported."
        print "Aborted execution in my_operations.reverse()"
        sys.exit(1)

    if (axis < 0) or (axis >= ndim):
        print "Input data does not contain the axis specified."
        print "Aborted execution in my_operations.reverse()"
        sys.exit(1) 

    if (ind_range[0] <= -1):
        ind_range[0] = 0 # set default to starting index

    if (ind_range[1] == -1) or (ind_range[1] >= shp[axis]):
        ind_range[1] = shp[axis]-1 # set default to ending index

    if shp[axis] == 1:
        return inp

    revinds = range(ind_range[1],ind_range[0]-1,-1)

    if ndim == 1:
        return inp[revinds]
    elif ndim == 2:
        if axis == 0:
            return inp[revinds,:]
        else:
            return inp[:,revinds]
    elif ndim == 3:
        if axis == 0:
            return inp[revinds,:,:]
        elif axis == 1:
            return inp[:,revinds,:]
        else:
            return inp[:,:,revinds]
    elif ndim == 4:
        if axis == 0:
            return inp[revinds,:,:,:]
        elif axis == 1:
            return inp[:,revinds,:,:]
        elif axis == 2:
            return inp[:,:,revinds,:]
        else:
            return inp[:,:,:,revinds]
    elif ndim == 5:
        if axis == 0:
            return inp[revinds,:,:,:,:]
        elif axis == 1:
            return inp[:,revinds,:,:,:]
        elif axis == 2:
            return inp[:,:,revinds,:,:]
        elif axis == 3:
            return inp[:,:,:,revinds,:]
        else:
            return inp[:,:,:,:,revinds]
    elif ndim == 6:
        if axis == 0:
            return inp[revinds,:,:,:,:,:]
        elif axis == 1:
            return inp[:,revinds,:,:,:,:]
        elif axis == 2:
            return inp[:,:,revinds,:,:,:]
        elif axis == 3:
            return inp[:,:,:,revinds,:,:]
        elif axis == 4:
            return inp[:,:,:,:,revinds,:]
        else:
            return inp[:,:,:,:,:,revinds]
    elif ndim == 7:
        if axis == 0:
            return inp[revinds,:,:,:,:,:,:]
        elif axis == 1:
            return inp[:,revinds,:,:,:,:,:]
        elif axis == 2:
            return inp[:,:,revinds,:,:,:,:]
        elif axis == 3:
            return inp[:,:,:,revinds,:,:,:]
        elif axis == 4:
            return inp[:,:,:,:,revinds,:,:]
        elif axis == 5:
            return inp[:,:,:,:,:,revinds,:]
        else:
            return inp[:,:,:,:,:,:,revinds]
    elif ndim == 8:
        if axis == 0:
            return inp[revinds,:,:,:,:,:,:,:]
        elif axis == 1:
            return inp[:,revinds,:,:,:,:,:,:]
        elif axis == 2:
            return inp[:,:,revinds,:,:,:,:,:]
        elif axis == 3:
            return inp[:,:,:,revinds,:,:,:,:]
        elif axis == 4:
            return inp[:,:,:,:,revinds,:,:,:]
        elif axis == 5:
            return inp[:,:,:,:,:,revinds,:,:]
        elif axis == 6:
            return inp[:,:,:,:,:,:,revinds,:]
        else:
            return inp[:,:,:,:,:,:,:,revinds]
