import numpy as NP
import scipy as SP

################################################################################

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
        raise TypeError('Unable to convert to Numpy array data type')

    shp = NP.shape(inp)
    ndim = len(shp)
    
    if ndim > 8:
        raise ValueError("Input data with more than 8 dimensions not supported. Aborted execution in my_operations.reverse()")

    if (axis < 0) or (axis >= ndim):
        raise ValueError("Input data does not contain the axis specified. Aborted execution in my_operations.reverse()")

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

#################################################################################

def binned_statistic(x, values=None, statistic='mean', bins=10, range=None):

    """
    -----------------------------------------------------------------------------
    Same functionality as binned_statistic() under scipy.stats module but can 
    return reverse indices such as in IDL version of histogram. Read the 
    documentation on the two aforementioned functions.

    Inputs:

    x          [numpy vector] A sequence of values to be binned.

    values     [numpy vector] The values on which the statistic will be computed.
               This must be the same shape as x.

    statistic  [string or callable, optional] The statistic to compute (default
               is 'mean'). The following statistics are available:

               'mean'    : compute the mean of values for points within each bin.
                          Empty bins will be represented by NaN.
               'median'  : compute the median of values for points within each
                          bin. Empty bins will be represented by NaN.
               'count'   : compute the count of points within each bin. This is
                          identical to an unweighted histogram. values array is
                          not referenced.
               'sum'     : compute the sum of values for points within each bin.
                          This is identical to a weighted histogram.
               function : a user-defined function which takes a 1D array of
                          values, and outputs a single numerical statistic. This
                          function will be called on the values in each bin.
                          Empty bins will be represented by function([]), or NaN
                          if this returns an error.

    bins       [int or sequence of scalars, optional] If bins is an int, it
               defines the number of equal-width bins in the given range (10, by
               default). If bins is a sequence, it defines the bin edges,
               including the rightmost edge, allowing for non-uniform bin widths.

    range      [2-element tuple in list, optional] The lower and upper range of
               the bins. If not provided, range is simply (x.min(), x.max()).
               Values outside the range are ignored.

    reverse_indices
               [boolean] If set to True (default), returns the reverse indices
               in revind

    Outputs:

    statistic  [numpy vector] The values of the selected statistic in each bin.

    bin_edges  [numpy vector] Return the bin edges (length(statistic)+1).

    binnumber  [numpy vector] This assigns to each observation an integer that
               represents the bin in which this observation falls. Array has the
               same length as values.

    revind     [numpy vector] list of reverse indices like the IDL counterpart.
               Vector whose number of elements is the sum of the number of
               elements in the histogram, N, and the number of array elements
               included in the histogram, plus one. The subscripts of the
               original array elements falling in the ith bin, 0 <= i < N, are
               given by the expression: R(R[i] : R[i+1]), where R is the
               reverse index list. If R[i] is equal to R[i+1], no elements are
               present in the i-th bin. 

    -----------------------------------------------------------------------------
    """

    sortind = NP.argsort(x, kind='heapsort')
    if values is not None:
        stat, bin_edges, binnum = SP.stats.binned_statistic(x[sortind], values[sortind], statistic=statistic, bins=bins, range=range)
    else:
        stat, bin_edges, binnum = SP.stats.binned_statistic(x[sortind], x[sortind], statistic=statistic, bins=bins, range=range)
        
    revind = NP.hstack((bin_edges.size, bin_edges.size+NP.cumsum(stat.astype(NP.int)), sortind))

    return stat, bin_edges, binnum, revind

#################################################################################

