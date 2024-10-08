from __future__ import print_function, division, unicode_literals
from builtins import zip, range
from typing import Optional, Union, List, Tuple
import warnings

import numpy as NP
import numpy.ma as MA
import numpy.linalg as NPLA
from numpy.typing import NDArray
import scipy as SP
import scipy.linalg as SPLA
from scipy import interpolate
from skimage import img_as_float
import skimage.morphology as morphology
from skimage.filters import median
from skimage.filters.rank import mean
from skimage.restoration import unwrap_phase
import astropy.convolution as CONV
import healpy as HP

from . import lookup_operations as LKP

#################################################################################

def reverse(inp, axis=0, ind_range=None):

    """
    -----------------------------------------------------------------------------
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
              of the axis over which the data is to be reversed. Default = None
              selects all indices for reversal.

    Output:

    The array with its data reversed over a subset or the entirety of the
    specified axis.
    -----------------------------------------------------------------------------
    """

    inp = NP.asarray(inp)
    if not isinstance(inp, NP.ndarray):
        raise TypeError('Input inp must beb a numpy array')

    shp = inp.shape
    ndim = inp.ndim
    
    if ndim > 8:
        raise ValueError("Input data with more than 8 dimensions not supported. Aborted execution in my_operations.reverse()")

    if (axis < 0) or (axis >= ndim):
        raise ValueError("Input data does not contain the axis specified. Aborted execution in my_operations.reverse()")

    if ind_range is None:
        ind_range = [0, shp[axis]-1] # Select all indices
    elif isinstance(ind_range, list):
        ind_range = NP.asarray(ind_range).ravel()
        if ind_range.size == 2:
            if (ind_range[0] <= -1):
                ind_range[0] = 0 # set default to starting index
            if (ind_range[1] == -1) or (ind_range[1] >= shp[axis]):
                ind_range[1] = shp[axis]-1 # set default to ending index
        else:
            raise ValueError('ind_range must be a two-element list or numpy array')
    else:
        raise TypeError('ind_range must be a two-element list or numpy array')

    if shp[axis] == 1:
        return inp

    revinds = NP.arange(ind_range[1],ind_range[0]-1,-1)

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

def is_broadcastable(shp1, shp2):

    """
    -----------------------------------------------------------------------------
    Check if two shapes are compatible for numpy braodcasting

    Inputs:

    shp1    [tuple] Shape tuple of first array

    shp2    [tuple] Shape tuple of second array

    Output:

    True if shapes are compatible for numpy broadcasting, False otherwise
    -----------------------------------------------------------------------------
    """

    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shp1[::-1], shp2[::-1]))

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

    if statistic.lower() != 'count':
        counts, _, _ = SP.stats.binned_statistic(x[sortind], x[sortind], statistic='count', bins=bins, range=range)    
        revind = NP.hstack((bin_edges.size, bin_edges.size+NP.cumsum(counts.astype(int)), sortind))
    else:
        revind = NP.hstack((bin_edges.size, bin_edges.size+NP.cumsum(stat.astype(int)), sortind))

    return stat, bin_edges, binnum, revind

#################################################################################

def rms(inp, axis=None, filter_dict=None, mask=None, verbose=True):

    """
    -----------------------------------------------------------------------------
    Estimate the rms of multi-dimensional (complex) input data along an axis (if
    specified). Optionally, fourier frequency filtering and masks can be used to
    refine the data before estimating rms.

    Inputs:

    inp         [Numpy array] input data for which RMS has to be estimated.

    Keyword Inputs:

    axis        [scalar integer] Axis over which FFT is performed. Default = None
                (last axis). Any negative value or values exceeding the number of
                axes in the input data will be reset to use the last axis.

    filter_dict [dictionary] Filter parameters in the Fourier (frequency) domain.
                Default is None (no filtering to be applied). If set, the
                filtering will be applied along the specified axis. If axis is
                not specified, no frequency domain filtering will be applied. 
                This is a dictionary consisting of the following keys and info:
                'freqwts'    [Numpy array] frequency window of weights. Should 
                             either have same shape as inp or have number of
                             elements equal to the number of elements in input
                             data along specified axis. Default = None.
                             If not set, then it will be set to a rectangular
                             window of width specified by key 'width' (see below) 
                             and will be applied as a filter identically to the
                             entire data along the specified axis.
                'width'      [scalar] Width of the frequency window as a fraction 
                             of the bandwidth. Has to be positive. Default is
                             None. If width is None, wts should be set. One and
                             only one among wts and width should be set.  
                'passband'   [string scalar] String specifying the passband
                             ('low' or 'high') to be used. Default = 'low'.

    mask        [Numpy array] Numpy array with same dimensions as the input 
                data. The values can be Boolean or can be integers which in turn
                will be converted to Boolean values. Mask values with True will 
                be masked and ignored in the rms estimates while mask values with
                False will only be considered in obtaining the rms estimates. 
                Default = None (no masking to be applied)

    verbose     [boolean] If set to True (default), print messages indicating
                progress

    Output:
    
    RMS estimate of the input data. If the input data is complex, the output 
    consists of rms estimate of the real and imaginary parts of the data after
    applying the specified filtering and/or masking.
    -----------------------------------------------------------------------------
    """

    from . import DSP_modules as DSP
    try:
        inp
    except NameError:
        raise NameError('No input data specified.')

    if isinstance(inp, list):
        inp = NP.asarray(inp)
    elif not isinstance(inp, NP.ndarray):
        raise TypeError('inp must be a list or numpy array')

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')
        else:
            if (axis < 0) or (axis >= len(inp.shape)):
                axis = len(inp.shape) - 1        
                if verbose:
                    print('\tSetting axis to be the last dimension of data')
    
        tempinp = NP.copy(inp)
        if filter_dict is not None:
            if not isinstance(filter_dict, dict):
                raise TypeError('filter_dict must be a dictionary')
            else:
                freqwts = None
                if 'freqwts' in filter_dict:
                    freqwts = filter_dict['freqwts']
            
                width = None
                if 'width' in filter_dict:
                    width = filter_dict['width']

                passband = 'low'
                if 'passband' in filter_dict:
                    passband = filter_dict['passband']

                if verbose:
                    print('\tInvoking fft_filter() in the DSP module...')

                tempinp = DSP.fft_filter(inp, axis=axis, wts=freqwts, width=width, passband=passband, verbose=verbose)

        if mask is not None:
            # Check for broadcastability
            if mask.shape != tempinp.shape:
                if mask.size != tempinp.shape[axis]:
                    raise ValueError('mask and inp cannot be broadcast as numpy arrays')

            try:
                msk = NP.ones_like(inp) * mask.astype(bool)
            except ValueError:
                raise ValueError('mask and inp cannot be broadcast as compatible numpy arrays in order to create the mask')

            tempinp = NP.ma.masked_array(tempinp, mask=msk) # create a masked array

        if NP.iscomplexobj(inp):
            rms = NP.std(tempinp.real, axis=axis, keepdims=True) + 1j * NP.std(tempinp.imag, axis=axis, keepdims=True)
        else:
            rms = NP.std(tempinp.real, axis=axis, keepdims=True)

        if len(rms.shape) < len(tempinp.shape):
            rms = NP.expand_dims(rms, axis)
        
    else:
        tempinp = NP.copy(inp)
        if mask is not None:
            if mask.shape != inp.shape:
                raise ValueError('mask and inp cannot be broadcasted as numpy arrays')

            try:
                msk = NP.ones_like(inp) * mask.astype(bool)
            except ValueError:
                raise ValueError('mask and inp cannot be broadcast as compatible numpy arrays in order to create the mask')

            tempinp = NP.ma.masked_array(tempinp, mask=msk)

        if NP.iscomplexobj(inp):
            rms = NP.std(tempinp.real) + 1j * NP.std(tempinp.imag)
        else:
            rms = NP.std(tempinp.real)

    return rms
            
#################################################################################

def healpix_interp_along_axis(indata, theta_phi=None, inloc_axis=None,
                              outloc_axis=None, axis=-1, kind='linear',
                              bounds_error=True, fill_value=NP.nan,
                              assume_sorted=False, nest=False):

    """
    -----------------------------------------------------------------------------
    Interpolate healpix data to specified angular locations (HEALPIX 
    interpolation) and along one other specified axis (usually frequency axis, 
    for instance) via SciPy interpolation. Wraps HEALPIX and SciPy interpolations
    into one routine.

    Inputs:

    indata      [numpy array] input data to be interpolated. Must be of shape 
                (nhpy x nax1 x nax2 x ...). Currently works only for 
                (nhpy x nax1). nhpy is a HEALPIX compatible npix

    theta_phi   [numpy array] spherical angle locations (in radians) at which
                the healpix data is to be interpolated to at each of the other 
                given axes. It must be of size nang x 2 where nang is the number 
                of spherical angle locations, 2 denotes theta and phi. If set to
                None (default), no healpix interpolation is performed

    inloc_axis  [numpy array] locations along the axis specified in axis (to be 
                interpolated with SciPy) in which indata is specified. It 
                should be of size nax1, nax2, ... or naxm. Currently it works 
                only if set to nax1

    outloc_axis [numpy array] locations along the axis specified in axis to be 
                interpolated to with SciPy. The axis over which this 
                interpolation is to be done is specified in axis. It must be of
                size nout. If this is set exactly equal to inloc_axis, no 
                interpolation along this axis is performed

    axis        [integer] axis along which SciPy interpolation is to be done. 
                If set to -1 (default), the interpolation happens over the last
                axis. Since the first axis of indata is reserved for the healpix
                pixels, axis must be set to 1 or above (upto indata.ndim-1).

    kind        [str or int] Specifies the kind of interpolation as a 
                string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 
                'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a 
                spline interpolation of first, second or third order) or as an 
                integer specifying the order of the spline interpolator to use. 
                Default is 'linear'.

    bounds_error 
                [bool, optional] If True, a ValueError is raised any time 
                interpolation is attempted on a value outside of the range of x 
                (where extrapolation is necessary). If False, out of bounds 
                values are assigned fill_value. By default, an error is raised.

    fill_value  [float] If provided, then this value will be used to fill in 
                for requested points outside of the data range. If not provided, 
                then the default is NaN.

    assume_sorted 
                [bool] If False, values of inloc_axis can be in any order and 
                they are sorted first. If True, inloc_axis has to be an array 
                of monotonically increasing values.
    
    nest        [bool] if True, the is assumed to be in NESTED ordering.

    Outputs:

    HEALPIX interpolated and SciPy interpolated output. Will be of size
    nang x ... x nout x ... x naxm. Currently returns an array of shape 
    nang x nout
    -----------------------------------------------------------------------------
    """

    from . import DSP_modules as DSP
    try:
        indata
    except NameError:
        raise NameError('input data not specified')

    if not isinstance(indata, NP.ndarray):
        raise TypeError('input data must be a numpy array')

    if theta_phi is not None:
        if not isinstance(theta_phi, NP.ndarray):
            raise TypeError('output locations must be a numpy array')

        if theta_phi.ndim != 2:
            raise ValueError('Output locations must be a 2D array')

    if axis == -1:
        axis = indata.ndim - 1

    if (axis < 1) or (axis >= indata.ndim):
        raise ValueError('input axis out of range')

    if theta_phi is not None:
        intermediate_data_shape = list(indata.shape)
        intermediate_data_shape[0] = theta_phi.shape[0]
        intermediate_data_shape = tuple(intermediate_data_shape)
        
        intermediate_data = NP.zeros(intermediate_data_shape, dtype=NP.float64)
        for ax in range(1,indata.ndim):
            for i in range(indata.shape[ax]):
                intermediate_data[:,i] = HP.get_interp_val(indata[:,i], theta_phi[:,0], theta_phi[:,1], nest=nest)
    else:
        intermediate_data = NP.copy(indata)

    if outloc_axis is not None:
        if inloc_axis is not None:
            outloc_axis = outloc_axis.flatten()
            inloc_axis = inloc_axis.flatten()
            eps = 1e-8
            if (outloc_axis.size == inloc_axis.size) and (NP.abs(inloc_axis-outloc_axis).max() <= eps):
                outdata = intermediate_data
            else:
                if kind == 'fft':
                    df_inp = NP.mean(NP.diff(inloc_axis))
                    df_out = NP.mean(NP.diff(outloc_axis))
                    ntau = df_inp / df_out * inloc_axis.size
                    ntau = NP.round(ntau).astype(int)
                    tau_inp = DSP.spectral_axis(inloc_axis.size, delx=df_inp, shift=True)
                    fftinp = NP.fft.fft(intermediate_data, axis=axis)
                    fftinp_shifted = NP.fft.fftshift(fftinp, axes=axis)
                    if fftinp.size % 2 == 0:
                        fftinp_shifted[:,0] = 0.0 # Blank out the N/2 element (0 element when FFT-shifted) for conjugate symmetry
                    npad = ntau - inloc_axis.size
                    if npad % 2 == 0:
                        npad_before = npad/2
                        npad_after = npad/2
                    else:
                        npad_before = npad/2 + 1
                        npad_after = npad/2

                    fftinp_shifted_padded = NP.pad(fftinp_shifted, [(0,0), (npad_before, npad_after)], mode='constant')
                    fftinp_padded = NP.fft.ifftshift(fftinp_shifted_padded, axes=axis)
                    ifftout = NP.fft.ifft(fftinp_padded, axis=axis) * (1.0 * ntau / inloc_axis.size)
                    eps_imag = 1e-10
                    if NP.any(NP.abs(ifftout.imag) > eps_imag):
                        raise ValueError('Significant imaginary component has been introduced unintentionally during the FFT based interpolation. Debug the code.')
                    else:
                        ifftout = ifftout.real
                    fout = DSP.spectral_axis(ntau, delx=tau_inp[1]-tau_inp[0], shift=True)
                    fout -= fout.min()
                    fout += inloc_axis.min() 
                    ind_outloc, ind_fout, dfreq = LKP.find_1NN(fout.reshape(-1,1), outloc_axis.reshape(-1,1), distance_ULIM=0.5*(fout[1]-fout[0]), remove_oob=True)
                    outdata = ifftout[:,ind_fout]
                    
                    # npad = 2 * (outloc_axis.size - inloc_axis.size)
                    # dt_inp = DSP.spectral_axis(2*inloc_axis.size, delx=inloc_axis[1]-inloc_axis[0], shift=True)
                    # dt_out = DSP.spectral_axis(2*outloc_axis.size, delx=outloc_axis[1]-outloc_axis[0], shift=True)
                    # fftinp = NP.fft.fft(NP.pad(intermediate_data, [(0,0), (0,inloc_axis.size)], mode='constant'), axis=axis) * (1.0 * outloc_axis.size / inloc_axis.size)
                    # fftinp = NP.fft.fftshift(fftinp, axes=axis)
                    # fftinp[0,0] = 0.0  # Blank out the N/2 element for conjugate symmetry
                    # fftout = NP.pad(fftinp, [(0,0), (npad/2, npad/2)], mode='constant')
                    # fftout = NP.fft.ifftshift(fftout, axes=axis)
                    # outdata = NP.fft.ifft(fftout, axis=axis)
                    # outdata = outdata[0,:outloc_axis.size]
                else:
                    interp_func = interpolate.interp1d(inloc_axis, intermediate_data, axis=axis, kind=kind, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
                    outdata = interp_func(outloc_axis)
        else:
            raise ValueError('input inloc_axis not specified')
    else:
        outdata = intermediate_data

    return outdata

#################################################################################

def interpolate_array(inparray, inploc, outloc, axis=-1, kind='linear'):

    """
    -----------------------------------------------------------------------------
    Interpolate a multi-dimensional array along one of its dimensions. It acts 
    as a wrapper to scipy.interpolate.interp1d but applies boundary conditions 
    differently

    Inputs:

    inparray    [numpy array] Multi-dimensional input array which will be used 
                in determining the interpolation function

    inploc      [numpy array] Locations using which the interpolation function
                is determined. It must be of size equal to the dimension of 
                input array along which interpolation is to be determined 
                specified by axis keyword input. It must be a list or numpy 
                array

    outloc      [list or numpy array] Locations at which interpolated array is
                to be determined along the specified axis. It must be a scalar, 
                list or numpy array. If any of outloc is outside the range of
                inploc, the first and the last cubes from the inparray will
                be used as boundary values

    axis        [scalar] Axis along which interpolation is to be performed. 
                Default=-1 (last axis)

    kind        [string or integer] Specifies the kind of interpolation as a 
                string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 
                'cubic' where 'slinear', 'quadratic' and 'cubic' refer to a 
                spline interpolation of first, second or third order) or as an 
                integer specifying the order of the spline interpolator to use. 
                Default is 'linear'.

    Output:

    outarray    [numpy array] Output array after interpolation 
    -----------------------------------------------------------------------------
    """

    assert isinstance(inparray, NP.ndarray), 'Input array inparray must be a numpy array'
    assert isinstance(inploc, (list, NP.ndarray)), 'Input locations must be a list or numpy array'
    assert isinstance(outloc, (int, float, list, NP.ndarray)), 'Output locations must be a scalar, list or numpy array'
    assert isinstance(axis, int), 'Interpolation axis must be an integer'
    assert isinstance(kind, str), 'Kind of interpolation must be a string'

    inploc = NP.asarray(inploc).reshape(-1)
    outloc = NP.asarray(outloc).reshape(-1)
    assert axis < inparray.ndim, 'Insufficient dimensions in inparray for interpolation'
    assert inparray.shape[axis]==inploc.size, 'Dimension of interpolation axis of inparray is mismatched with number of locations at which interpolation is requested'

    interp_required = True
    if inploc.size == outloc.size:
        if NP.allclose(inploc, outloc):
            interp_required = False
            return inparray # no interpolation required, just return outarray=inparray
    if interp_required:
        inbound_ind = NP.where(NP.logical_and(outloc >= inploc.min(), outloc <= inploc.max()))[0]
        outbound_low_ind = NP.where(outloc < inploc.min())[0]
        outbound_high_ind = NP.where(outloc > inploc.max())[0]
    
        outarray = None
        if inbound_ind.size > 0:
            interpfunc = interpolate.interp1d(inploc, inparray, kind=kind, axis=axis, copy=False, assume_sorted=False)
            outarray = interpfunc(outloc[inbound_ind])
        if outbound_low_ind.size > 0:
            if outarray is None:
                outarray = NP.repeat(NP.take(inparray, [NP.argmin(inploc)], axis=axis), outbound_low_ind.size, axis=axis)
            else:
                outarray = NP.concatenate((NP.repeat(NP.take(inparray, [NP.argmin(inploc)], axis=axis), outbound_low_ind.size, axis=axis), outarray), axis=axis)
        if outbound_high_ind.size > 0:
            if outarray is None:
                outarray = NP.repeat(NP.take(inparray, [NP.argmax(inploc)], axis=axis), outbound_high_ind.size, axis=axis)
            else:
                outarray = NP.concatenate((outarray, NP.repeat(NP.take(inparray, [NP.argmax(inploc)], axis=axis), outbound_high_ind.size, axis=axis)), axis=axis)
    
        return outarray

#################################################################################

def percentiles_to_2D_contour_levels(pdf, percentiles):

    """
    -----------------------------------------------------------------------------
    Determine 2D contour levels given percentiles

    Inputs:

    pdf         [numpy array] 2D array denoting the probability density function
                from which contour levels for the given percentiles are to be
                determined

    percentiles [list or numpy array] The percentiles (in percent) for which 
                contour levels are to be determined. All elements in this
                array must lie between 0 and 100

    Output:     [numpy array] Contour levels corresponding to the input 
                percentiles. Size of returned array will be equal to that of 
                the input percentiles.
    -----------------------------------------------------------------------------
    """

    try:
        pdf, percentiles
    except NameError:
        raise NameError('Inputs pdf and percentiles must be specified')

    assert isinstance(pdf, NP.ndarray), 'Input pdf must be a numpy array'
    assert pdf.ndim==2, 'Input pdf must be a 2D numpy array'
    assert isinstance(percentiles, (list,NP.ndarray)), 'Input percentiles must be a list or numpy array'
    if NP.any(pdf < 0.0):
        raise ValueError('Input pdf must be non-negative')
    percentiles = NP.asarray(percentiles).ravel()
    eps = 1e-12
    percentiles = NP.clip(percentiles, eps, 100.0-eps)
    if NP.any(percentiles < 0.0) or NP.any(percentiles > 100.0):
        raise ValueError('Percentiles must lie between 0 and 100')
    n = 10000
    adjustment_factor = NP.sum(pdf)
    pdf = pdf / adjustment_factor
    t = NP.linspace(0.0, pdf.max(), n)
    mask = (pdf >= t[:, NP.newaxis, NP.newaxis]+eps)
    integral = (mask * pdf).sum(axis=(1,2))
    interpfunc = interpolate.interp1d(integral, t, kind='linear')
    cntr_levels = interpfunc(percentiles/100.0) * adjustment_factor
    return cntr_levels

#################################################################################

def smooth_masked_array_1D(maskarr, smooth_parms, axis=-1, boundary='fill',
                           fill_value=0., nan_treatment='interpolate',
                           normalize_kernel=True, preserve_nan=False,
                           normalization_zero_tol=1e-8):

    """
    -----------------------------------------------------------------------------
    Smooth a multi-dimensional array including masked arrays or arrays with 
    missing (NaN) values along a specified axis. It can work on complex valued 
    arrays as well. It is a wrapper for astropy.convolution module but along a 
    single specified axis.

    Inputs:

    maskarr     [numpy array or masked array] Numpy array (possibly including 
                NaN) or masked array to be smoothed.

    smooth_parms
                [dictionary] Dictionary specifying smoothing parameters. It has
                the following keys and values:
                'op_type'       [string] Specifies the smoothing kernel.
                                Must be specified (no default). Accepted values are
                                'tophat' (astropy.convolution) and 'gaussian' 
                                (astropy.convolution)
                'window_size'   [integer (optional)] Specifies the size of the
                                interpolating/smoothing kernel. The kernel is a 
                                tophat function when 'op_type' is set to 'tophat'
                                and refers to FWHM when 'op_type' is set to 
                                'gaussian'
    
    axis        [integer] Axis along which smoothing is to be done. Must be an 
                integer. Default=-1 (last axis)

    boundary    [str (optional)] A flag indicating how to handle boundaries. 
                Accepted values are:
                * `None`
                    Set the ``result`` values to zero where the kernel
                    extends beyond the edge of the array.
                * 'fill'
                    Set values outside the array boundary to ``fill_value`` 
                    (default).
                * 'wrap'
                    Periodic boundary that wrap to the other side of ``array``.
                * 'extend'
                    Set values outside the array to the nearest ``array``
                    value.

    fill_value  [float (optional)] The value to use outside the array when 
                using ``boundary='fill'``

    normalize_kernel 
                [bool (optional)] Whether to normalize the kernel to have a sum 
                of one prior to convolving
    nan_treatment 
                [string (optional)] Accepted values are 'interpolate', 'fill'. 
                'interpolate' will result in renormalization of the kernel at 
                each position ignoring (pixels that are NaN in the image) in 
                both the image and the kernel. 'fill' will replace the NaN 
                pixels with a fixed numerical value (default zero, see 
                ``fill_value``) prior to convolution. Note that if the kernel 
                has a sum equal to zero, NaN interpolation is not possible and 
                will raise an exception

    preserve_nan 
                [bool (optional)] After performing convolution, should pixels 
                that were originally NaN again become NaN?

    normalization_zero_tol 
                [float (optional)] The absolute tolerance on whether the 
                kernel is different than zero. If the kernel sums to zero to 
                within this precision, it cannot be normalized. Default is 
                "1e-8".

    Output:

    Masked array containing smoothed array. Same shape as input maskarr.
    -----------------------------------------------------------------------------
    """

    if not isinstance(maskarr, (NP.ndarray,MA.MaskedArray)):
        raise TypeError('Input maskarr must be a numpy array or an instance of class Masked Array')
    if not isinstance(axis, int):
        raise TypeError('Input axis must be a scalar')
    if not isinstance(smooth_parms, dict):
        raise TypeError('Input smooth_parms must be a dictionary')
    if 'op_type' not in smooth_parms:
        raise KeyError('Key "op_type" not found in input smooth_parms')
    if smooth_parms['op_type'].lower() not in ['gaussian', 'tophat']:
        raise ValueError('op_type specified in smooth_parms currently not supported')
    if smooth_parms['op_type'].lower() in ['gaussian', 'tophat']:
        if 'window_size' not in smooth_parms:
            raise KeyError('Input "window_size" not found in smooth_parms')
        if smooth_parms['window_size'] <= 0:
            raise ValueError('Filter window size must be positive')
    if isinstance(maskarr, MA.MaskedArray):
        maskarr_filled = MA.filled(maskarr, NP.nan)
    else:
        maskarr_filled = NP.copy(maskarr)
    if NP.iscomplexobj(maskarr):
        maskarr_filled[NP.isnan(maskarr)] += 1j * NP.nan # Make imaginary part NaN
    maskarr_reshaped = NP.moveaxis(maskarr_filled, axis, maskarr.ndim-1)
    reshaped_shape = maskarr_reshaped.shape
    if smooth_parms['op_type'].lower() == 'gaussian':
        fwhm = smooth_parms['window_size']
        x_sigma = fwhm / (2.0 * NP.sqrt(2.0 * NP.log(2.0)))
        kernel1D = CONV.Gaussian1DKernel(x_sigma)
    elif smooth_parms['op_type'].lower() == 'tophat':
        if smooth_parms['window_size'] % 2 == 0:
            smooth_parms['window_size'] += 1
        kernel1D = CONV.Box1DKernel(smooth_parms['window_size'])
    kernel = CONV.CustomKernel(kernel1D.array[NP.newaxis,:]) # Make a 2D kernel from the 1D kernel where it spans only one element in the new axis
    if NP.iscomplexobj(maskarr):
        maskarr_smoothed = CONV.convolve(maskarr_reshaped.real.reshape(-1,reshaped_shape[-1]), kernel, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate') + 1j * CONV.convolve(maskarr_reshaped.imag.reshape(-1,reshaped_shape[-1]), kernel, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate')
    else:
        maskarr_smoothed = CONV.convolve(maskarr_reshaped.reshape(-1,reshaped_shape[-1]), kernel, boundary='extend', fill_value=NP.nan, nan_treatment='interpolate')
    maskarr_smoothed = maskarr_smoothed.reshape(reshaped_shape)
    maskarr_smoothed = NP.moveaxis(maskarr_smoothed, maskarr.ndim-1, axis)

    if isinstance(maskarr, MA.MaskedArray):
        mask = NP.isnan(maskarr_smoothed)
        return MA.array(maskarr_smoothed, mask=mask, fill_value=NP.nan)
    else:
        return maskarr_smoothed

#################################################################################

def phase_unwrap_1D(phase, axis=-1, wrap_around=False, seed=None, tol=NP.pi):

    """
    -----------------------------------------------------------------------------
    Unwrap the phases of a multi-dimensional array along the specified axis. 
    Wrapper to skimage.restoration.unwrap_phase() to take care of certain 
    inconsistencies (large jumps) and also to numpy.unwrap while handling masked 
    arrays. 

    Inputs:

    phase       [numpy array or masked array] Multi-dimensional numpy or masked 
                array containing phases (in radians).

    axis        [integer] Axis along which unwrapping is to be done. Must be an 
                integer. Default=-1 (last axis)

    tol         [scalar] Tolerance (in radians) to check for smoothness. If the
                the unwrapped phases are not smooth to this level of tolerance,
                those pixels are masked in the output. Default=pi.

    wrap_around [bool or sequence of bool (optional)] When an element of the 
                sequence is  `True`, the unwrapping process will regard the 
                edges along the corresponding axis of the image to be connected 
                and use this connectivity to guide the phase unwrapping process. 
                If only a single boolean is given, it will apply to all axes. 
                Wrap around is not supported for 1D arrays.

    seed        [int (optional)] Unwrapping 2D or 3D images uses random 
                initialization. This sets the seed of the PRNG to achieve 
                deterministic behavior.

    Output:

    Masked array containing the unwrapped phase array, with same shape as input
    array phase. Besides the original mask (if present), any unsmooth behavior 
    as dictated by input tol will also be masked. See documentation of 
    skimage.restoration.unwrap_phase() and numpy.unwrap() for more details. 
    -----------------------------------------------------------------------------
    """

    if not isinstance(phase, (NP.ndarray,MA.MaskedArray)):
        raise TypeError('Input phase data type is invalid')
    if tol is None:
        tol = NP.pi
    elif not isinstance(tol, (int,float)):
        raise TypeError('Input tol must be a scalar')
    else:
        if tol <= 0.0:
            raise ValueError('Input tol must be positive')

    phase_reshaped = NP.moveaxis(phase, axis, phase.ndim-1)
    reshaped_shape = phase_reshaped.shape
    phase_reshaped = phase_reshaped.reshape(-1, reshaped_shape[-1])

    if isinstance(phase, MA.MaskedArray):
        phase_unwrapped = unwrap_phase(phase_reshaped,
                                       wrap_around=wrap_around, seed=seed)
    else:
        phase_unwrapped = NP.unwrap(phase, discont=NP.pi, axis=axis)
        # phase_unwrapped = NP.apply_along_axis(NP.unwrap, axis, phase)
        
    smooth_phase_unwrapped = smooth_masked_array_1D(phase_unwrapped,
                                                    {'op_type': 'tophat',
                                                     'window_size': 7},
                                                    axis=-1, boundary='extend',
                                                    fill_value=NP.nan,
                                                    nan_treatment='interpolate')

    if isinstance(phase, MA.MaskedArray):
        diff_phase_unwrapped = smooth_phase_unwrapped.data - phase_unwrapped.data
    else:
        diff_phase_unwrapped = smooth_phase_unwrapped.data - phase_unwrapped
    largediff_loc = NP.abs(diff_phase_unwrapped) >= tol
    if isinstance(phase, MA.MaskedArray):
        if NP.any(largediff_loc):
            phase_unwrapped.mask = NP.logical_or(phase_unwrapped.mask,
                                                 largediff_loc)
    else:
        mask = NP.zeros(phase.shape, dtype=bool)
        mask[NP.isnan(phase)] = True
        if NP.any(largediff_loc):
            mask[largediff_loc] = True
        phase_unwrapped = MA.MaskedArray(phase_unwrapped, mask=mask)

    phase_unwrapped = phase_unwrapped.reshape(reshaped_shape)
    phase_unwrapped = NP.moveaxis(phase_unwrapped, phase.ndim-1, axis)

    return phase_unwrapped
        
#################################################################################

def interpolate_phase_1D(phase, wts, axis, interp_parms, collapse_axes=None,
                         collapse_stat='median'):

    """
    ----------------------------------------------------------------------------
    Interpolate phase array (spectrum) and fill values where values are not 
    available. 

    Inputs:

    phase   [Masked array] Masked array containing closure phases (in radians). 
            Usually, it has shape (nlst, ndays, ntriads, nchan)

    wts     [Masked array] Masked array containing weights corresponding to
            number of measurements. It has same shape as input phase, which is 
            usually (nlst, ndays, ntriads, nchan)

    axis    [integer] Axis along which interpolation occurs. It corresponds to 
            nchan. Must be an integer.

    interp_parms
            [dictionary] Dictionary specifying interpolation parameters. It has
            the following keys and values:
            'op_type'       [string] Specifies the interpolating operation.
                            Must be specified (no default). Accepted values are
                            'interp1d' (scipy.interpolate), 'median' 
                            (skimage.filters), 'tophat' (astropy.convolution) 
                            and 'gaussian' (astropy.convolution)
            'interp_kind'   [string (optional)] Specifies the interpolation 
                            kind (if 'op_type' is set to 'interp1d'). For
                            accepted values, see scipy.interpolate.interp1d()
            'window_size'   [integer (optional)] Specifies the size of the
                            interpolating/smoothing kernel. Only applies when
                            'op_type' is set to 'median', 'tophat' or 'gaussian'
                            The kernel is a tophat function when 'op_type' is 
                            set to 'median' or 'tophat'. If refers to FWHM when
                            'op_type' is set to 'gaussian'
    collapse_axes
            [Nonetype, int, tuple, list, or numpy array] Axes to collapse the
            data along before interpolation. If set to None (default), no
            collapse is performed. Otherwise, the axes specified here will be
            collapsed in the data using statistic specified in input
            collapse_stat. Usually, these axes are those along which closure 
            phases can be assumed to be coherent.

    collapse_stat
            [string (optional)] Statistic used to collapse the input data along
            the axes specified in input collapse_axes. Only applies if input
            collapse_axes is not set to None. Accepted values are 'mean' and
            'median' (default)

    Outputs:

    Tuple consisting of two elements. First element is a masked array of 
    interpolated closure phases (in radians) of shape (nlst, ndays, 
    ntriads, nchan) except along the axes which were collapsed. Second element
    is a masked array of interpolated weights of same shape as the interpolated
    closure phases.
    ----------------------------------------------------------------------------
    """

    if not isinstance(phase, MA.MaskedArray):
        raise TypeError('Input phase must be a numpy masked array')
    if not isinstance(wts, MA.MaskedArray):
        raise TypeError('Input wts must be a numpy masked array')
    if phase.shape != wts.shape:
        raise ValueError('Inputs phase and wts must have the same shape')
    if not isinstance(axis, int):
        raise TypeError('Input axis must be an integer')
    if axis >= phase.ndim:
        raise ValueError('Input axis out of bounds')
    nchan = phase.shape[axis]
    if collapse_axes is not None:
        if not isinstance(collapse_axes, (int,tuple,list,NP.ndarray)):
            raise TypeError('Input collapse_axes must be an integer, list, tuple, or numpy array')
        collapse_axes = NP.asarray(collapse_axes).ravel()
        if NP.sum(NP.in1d(axis, collapse_axes)) > 0:
            raise ValueError('axis must not be included in collapse_axes')
        if not isinstance(collapse_stat, str):
            raise TypeError('Input collapse_stat must be a string')
        if collapse_stat.lower() not in ['mean', 'median']:
            raise ValueError('Invalid input for collapse_stat') 
        ceip = NP.exp(1j*phase)
        if collapse_stat.lower() == 'mean':
            ceip_collapsed = MA.sum(ceip*wts, axis=tuple(collapse_axes), keepdims=True) / MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
        elif collapse_stat.lower() == 'median':
            ceip_collapsed = MA.median(ceip.real, axis=tuple(collapse_axes), keepdims=True) +1j * MA.median(ceip.imag, axis=tuple(collapse_axes), keepdims=True)
        wts_collapsed = MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
        ceip_collapsed /= NP.abs(ceip_collapsed) # Renormalize to unit amplitude
        phase_collapsed = NP.angle(ceip_collapsed) 
    else:
        phase_collapsed = MA.copy(phase)
        wts_collapsed = MA.copy(wts)

    # Perform phase unwrapping
    
    phase_unwrapped = phase_unwrap_1D(phase_collapsed, axis=axis, seed=200,
                                      tol=NP.pi)
    wts_collapsed.mask = phase_unwrapped.mask

    if not isinstance(interp_parms, dict):
        raise TypeError('Input interp_parms must be a dictionary')
    if 'op_type' not in interp_parms:
        raise KeyError('Key "op_type" not found in input interp_parms')
    if interp_parms['op_type'].lower() not in ['median', 'gaussian', 'tophat', 'interp1d']:
        raise ValueError('op_type specified in interp_parms currently not supported')
    if interp_parms['op_type'].lower() in ['median', 'gaussian', 'tophat']:
        if 'window_size' not in interp_parms:
            raise KeyError('Input "window_size" not found in interp_parms')
        if interp_parms['window_size'] <= 0:
            raise ValueError('Spectral filter window size must be positive')
    if interp_parms['op_type'].lower() == 'interp1d':
        if 'interp_kind' not in interp_parms:
            interp_parms['interp_kind'] = 'linear'

    mask_in = phase_unwrapped.mask
    phase_filled = MA.filled(phase_unwrapped, fill_value=NP.nan) # Fill with NaN for missing values
    wts_filled = MA.filled(wts_collapsed, fill_value=0.0) # Fill with 0.0 for missing values

    if interp_parms['op_type'].lower() == 'interp1d':
        other_axes = NP.where(NP.arange(phase_unwrapped.ndim) != axis)[0]
        axis_mask = NP.sum(mask_in, axis=tuple(other_axes)) # shape=(nchan,)
        if NP.sum(axis_mask.astype(bool)) > 1.0/3 * axis_mask.size:
            raise ValueError('More than 1/3 of channels are flagged at some point or another. This will lead to failure of interp1d method. Try other interpolation options.')
        masked_chans = NP.arange(phase_unwrapped.shape[axis])[axis_mask.astype(bool)]
        unmasked_chans = NP.arange(phase_unwrapped.shape[axis])[NP.logical_not(axis_mask.astype(bool))]
        unmasked_phase = NP.take(phase_filled, unmasked_chans, axis=axis, mode='clip')
        unmasked_wts = NP.take(wts_filled, unmasked_chans, axis=axis, mode='clip')
        phase_interpfunc = interpolate.interp1d(unmasked_chans, unmasked_phase, kind=interp_parms['interp_kind'], axis=axis, bounds_error=False, fill_value=NP.nan)
        wts_interpfunc = interpolate.interp1d(unmasked_chans, unmasked_wts, kind=interp_parms['interp_kind'], axis=axis, bounds_error=False, fill_value=0.0)
        wts_interped = wts_interpfunc(NP.arange(phase_unwrapped.shape[axis]))
        phase_interped = phase_interpfunc_real(NP.arange(phase_unwrapped.shape[axis]))
    else:
        wts_reshaped = NP.moveaxis(wts_filled, axis, wts_collapsed.ndim-1) # axis is the last dimension
        wts_reshaped_shape = wts_reshaped.shape
        mask_reshaped = NP.moveaxis(wts_collapsed.mask, axis, wts_collapsed.ndim-1) # axis is the last dimension
        phase_reshaped = NP.moveaxis(phase_filled, axis, phase_unwrapped.ndim-1) # axis is the last dimension

        if interp_parms['op_type'].lower() == 'median': # Always typecasts to int which is a problem!!! Needs to be fixed.
            kernel = morphology.rectangle(1, interp_parms['window_size'], dtype=NP.float64)
            maxval = NP.nanmax(NP.abs(wts_reshaped)) 
            wts_interped = maxval * mean(img_as_float(wts_reshaped.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) # shape=(-1,nchan), use mean not median for weights, array must be normalized to lie inside [-1,1]
            maxval = NP.nanmax(NP.abs(ceip_reshaped))
            phase_interped = maxval * median(img_as_float(phase_reshaped.real.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) # array must be normalized to lie inside [-1,1]
        else:
            wts_filled = MA.filled(wts_collapsed, fill_value=NP.nan)
            wts_reshaped = NP.moveaxis(wts_filled, axis, wts_collapsed.ndim-1) # axis is the last dimension
            if interp_parms['op_type'].lower() == 'gaussian':
                fwhm = interp_parms['window_size']
                x_sigma = fwhm / (2.0 * NP.sqrt(2.0 * NP.log(2.0)))
                kernel1D = CONV.Gaussian1DKernel(x_sigma)
            elif interp_parms['op_type'].lower() == 'tophat':
                if interp_parms['window_size'] % 2 == 0:
                    interp_parms['window_size'] += 1
                kernel1D = CONV.Box1DKernel(interp_parms['window_size'])
            kernel = CONV.CustomKernel(kernel1D.array[NP.newaxis,:]) # Make a 2D kernel from the 1D kernel where it spans only one element in the new axis
            wts_interped = CONV.interpolate_replace_nans(wts_reshaped.reshape(-1,wts_reshaped_shape[-1]), kernel)
            phase_interped = CONV.interpolate_replace_nans(phase_reshaped.reshape(-1,wts_reshaped_shape[-1]), kernel)
        
        wts_interped = wts_interped.reshape(wts_reshaped_shape) # back to intermediate shape with axis as the last dimension
        wts_interped = NP.moveaxis(wts_interped, wts_collapsed.ndim-1, axis) # Original shape
        phase_interped = phase_interped.reshape(wts_reshaped_shape) # back to intermediate shape with axis as the last dimension
        phase_interped = NP.moveaxis(phase_interped, phase_collapsed.ndim-1, axis) # Original shape

    eps = 1e-10
    mask_out = NP.logical_or(wts_interped < eps, NP.isnan(wts_interped))
    wts_interped = MA.array(wts_interped, mask=mask_out)
    phase_interped = MA.array(phase_interped, mask=mask_out)

    return (phase_interped, wts_interped)

################################################################################

def interpolate_masked_array_1D(arr, wts, axis, interp_parms, inploc=None,
                                outloc=None, fix_ampl=None, collapse_axes=None,
                                collapse_stat='median'):

    """
    ----------------------------------------------------------------------------
    Interpolate masked array (can handle complex arrays) along a specified axis 
    and fill values where values are not available. It can also be used to work 
    with fixed amplitude such as complex exponentials where amplitude is 
    constrained to be unity. This is a masked (complex) array wrapper to 
    scipy.interpolate.interp1d(), astropy.convolution.convolve() but 
    interpolates along the specified dimension. It can also use the interpolated
    array to further predict the array at arbitrary locations.

    CAUTION: Except for op_type='interp1d', it assumes the locations along the 
    axis are monotonic and uniformly spaced

    Inputs:

    arr     [Masked array] Complex masked array

    wts     [Masked array] Masked array containing weights corresponding to
            number of measurements. It has same shape as input arr

    axis    [integer] Axis along which input array is to be interpolated. Must 
            be an integer.

    interp_parms
            [dictionary] Dictionary specifying interpolation parameters. It has
            the following keys and values:
            'op_type'       [string] Specifies the interpolating operation.
                            Must be specified (no default). Accepted values are
                            'interp1d' (scipy.interpolate), 'median' 
                            (skimage.filters), 'tophat' (astropy.convolution) 
                            and 'gaussian' (astropy.convolution)
            'interp_kind'   [string (optional)] Specifies the interpolation 
                            kind (if 'op_type' is set to 'interp1d' or if 
                            'outloc' is not None). For accepted values, see 
                            scipy.interpolate.interp1d()
            'window_size'   [integer (optional)] Specifies the size of the
                            interpolating/smoothing kernel. Only applies when
                            'op_type' is set to 'median', 'tophat' or 'gaussian'
                            The kernel is a tophat function when 'op_type' is 
                            set to 'median' or 'tophat'. If refers to FWHM when
                            'op_type' is set to 'gaussian'

    inploc  [NoneType or numpy array] Input locations along interpolation axis 
            where the input array is provided. If outloc is not set to None, 
            then this parameter also has to be set. If set to a numpy array, 
            interp_kind in interp_parms must be set. 

    outloc  [NoneType or numpy array] Output locations along interpolation axis 
            where the interpolated values are to be resampled. If set to None, 
            the interpolation is used to predict only the missing values. If set 
            to a numpy array, it must be an array of locations along axis to be
            interpolated, and inploc as well as interp_kind in interp_parms must 
            be set. 

    fix_ampl
            [NoneType or scalar] If set to None (default), amplitudes are not
            constrained in any of the operations. If not set to None, it must
            be a positive scalar, indicating amplitudes of all complex numbers
            must be constrained to this value. This is useful if working with
            complex exponentials of phases whose amplitudes are always 
            constrained to be unity. Thus, fix_ampl = 1 for complex 
            exponentials constructed from phases.

    collapse_axes
            [Nonetype, int, tuple, list, or numpy array] Axes to collapse the
            data along before interpolation. If set to None (default), no
            collapse is performed. Otherwise, the axes specified here will be
            collapsed in the data using statistic specified in input
            collapse_stat. Usually, these axes are those along which the 
            complex array can be assumed to phase coherent.

    collapse_stat
            [string (optional)] Statistic used to collapse the input data along
            the axes specified in input collapse_axes. Only applies if input
            collapse_axes is not set to None. Accepted values are 'mean' and
            'median' (default)

    Outputs:

    Tuple consisting of two elements. First element is a masked array of 
    interpolated complex array. It has same shape as input arr except 
    along the axes which were collapsed. Second element is a masked array of 
    interpolated weights of same shape as the interpolated complex array output
    ----------------------------------------------------------------------------
    """

    if not isinstance(arr, MA.MaskedArray):
        raise TypeError('Input arr must be a numpy masked array')
    if not isinstance(wts, MA.MaskedArray):
        raise TypeError('Input wts must be a numpy masked array')
    if arr.shape != wts.shape:
        raise ValueError('Inputs arr and wts must have the same shape')
    if not isinstance(axis, int):
        raise TypeError('Input axis must be an integer')
    if axis >= arr.ndim:
        raise ValueError('Input axis out of bounds')
    nchan = arr.shape[axis]
    if fix_ampl is not None:
        if not isinstance(fix_ampl, (int,float)):
            raise TypeError('Input fix_ampl must be a scalar')
        if fix_ampl <= 0.0:
            raise ValueError('Input fix_ampl must be positive')
    if collapse_axes is not None:
        if not isinstance(collapse_axes, (int,tuple,list,NP.ndarray)):
            raise TypeError('Input collapse_axes must be an integer, list, tuple, or numpy array')
        collapse_axes = NP.asarray(collapse_axes).ravel()
        if NP.sum(NP.in1d(axis, collapse_axes)) > 0:
            raise ValueError('axis must not be included in collapse_axes')
        if not isinstance(collapse_stat, str):
            raise TypeError('Input collapse_stat must be a string')
        if collapse_stat.lower() not in ['mean', 'median']:
            raise ValueError('Invalid input for collapse_stat') 
        if collapse_stat.lower() == 'mean':
            arr_collapsed = MA.sum(arr*wts, axis=tuple(collapse_axes), keepdims=True) / MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
        elif collapse_stat.lower() == 'median':
            arr_collapsed = MA.median(arr.real, axis=tuple(collapse_axes), keepdims=True) +1j * MA.median(arr.imag, axis=tuple(collapse_axes), keepdims=True)
        wts_collapsed = MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
    else:
        wts_collapsed = MA.copy(wts)
        arr_collapsed = MA.copy(arr)
    if fix_ampl is not None:
        if NP.iscomplexobj(arr):
            arr_collapsed *= fix_ampl / NP.abs(arr_collapsed) # Renormalize to fix_ampl
        
    if not isinstance(interp_parms, dict):
        raise TypeError('Input interp_parms must be a dictionary')
    if 'op_type' not in interp_parms:
        raise KeyError('Key "op_type" not found in input interp_parms')
    if interp_parms['op_type'].lower() not in ['median', 'gaussian', 'tophat', 'interp1d']:
        raise ValueError('op_type specified in interp_parms currently not supported')
    if interp_parms['op_type'].lower() in ['median', 'gaussian', 'tophat']:
        if 'window_size' not in interp_parms:
            raise KeyError('Input "window_size" not found in interp_parms')
        if interp_parms['window_size'] <= 0:
            raise ValueError('Spectral filter window size must be positive')
    if inploc is not None:
        if not isinstance(inploc, NP.ndarray):
            raise TypeError('Input inploc must be a numpy array')
        inploc = inploc.ravel()
        if inploc.size != arr.shape[axis]:
            raise ValueError('Input inploc incompatible with dimensions of input arr along the specified axis')
    if outloc is not None:
        if not isinstance(outloc, NP.ndarray):
            raise TypeError('Input outloc must be a numpy array')
        if inploc is None:
            raise ValueError('For outloc to be used, inploc must be set')
        outloc = outloc.ravel()
    if (interp_parms['op_type'].lower() == 'interp1d') or (outloc is not None):
        if 'interp_kind' not in interp_parms:
            interp_parms['interp_kind'] = 'linear'
    mask_in = arr.mask
    if NP.iscomplexobj(arr):
        arr_filled = MA.filled(arr_collapsed.real, fill_value=NP.nan) + 1j * MA.filled(arr_collapsed.imag, fill_value=NP.nan) # Both real and imaginary parts need to contain NaN for interpolation to work later separately on these parts
    else:
        arr_filled = MA.filled(arr_collapsed, fill_value=NP.nan)
    wts_filled = MA.filled(wts_collapsed, fill_value=0.0)
    if interp_parms['op_type'].lower() == 'interp1d':
        wts_reshaped = NP.moveaxis(wts_collapsed, axis, wts_collapsed.ndim-1) # axis is the last dimension
        wts_reshaped_shape = wts_reshaped.shape
        wts_reshaped = wts_reshaped.reshape(-1,wts_reshaped_shape[-1]) # 2D array with axis to be interpolated as last dimension
        mask_reshaped = NP.moveaxis(wts_collapsed.mask, axis, wts_collapsed.ndim-1).reshape(-1,wts_reshaped_shape[-1]) # axis is the last dimension
        arr_reshaped = NP.moveaxis(arr_collapsed, axis, arr_collapsed.ndim-1).reshape(-1,wts_reshaped_shape[-1]) # 2D array with axis to be interpolated as the last dimension
        if outloc is None:
            wts_interped = MA.copy(wts_reshaped)
            arr_interped = MA.copy(arr_reshaped)
        else:
            wts_interped = MA.array(NP.full((wts_reshaped.shape[0],outloc.size), dtype=float, fill_value=NP.nan), mask=NP.full((wts_reshaped.shape[0],outloc.size), dtype=bool, fill_value=False))
            arr_interped = MA.array(NP.full((arr_reshaped.shape[0],outloc.size), dtype=float, fill_value=NP.nan), mask=NP.full((arr_reshaped.shape[0],outloc.size), dtype=bool, fill_value=False))
            if NP.iscomplexobj(arr):
                arr_interped = arr_interped.astype(NP.complex)
                arr_interped += 1j * NP.full((wts_reshaped.shape[0],outloc.size), dtype=float, fill_value=NP.nan)
        for ax0ind in NP.arange(arr_reshaped.shape[0]):
            inpind = NP.where(NP.logical_not(mask_reshaped[ax0ind,:]))[0]
            if outloc is None:
                outind = NP.where(mask_reshaped[ax0ind,:])[0]
                inlocs = NP.copy(inpind)
                outlocs = NP.copy(outind)
            else:
                inlocs = inploc[inpind]
                outlocs = NP.copy(outloc)
            if (inpind.size > 1) and (outlocs.size > 0):
                inpwts = wts_reshaped[ax0ind,inpind]
                inparr = arr_reshaped[ax0ind,inpind]
                interpfunc_wts = interpolate.interp1d(inlocs, inpwts, kind=interp_parms['interp_kind'], axis=-1, bounds_error=False, fill_value=NP.nan)
                outwts = interpfunc_wts(outlocs)
                if NP.iscomplexobj(arr):
                    interpfunc_arr_real = interpolate.interp1d(inlocs, inparr.real, kind=interp_parms['interp_kind'], axis=-1, bounds_error=False, fill_value=NP.nan)
                    interpfunc_arr_imag = interpolate.interp1d(inlocs, inparr.imag, kind=interp_parms['interp_kind'], axis=-1, bounds_error=False, fill_value=NP.nan)
                    outarr = interpfunc_arr_real(outlocs) + 1j * interpfunc_arr_imag(outlocs)
                else:
                    interpfunc_arr = interpolate.interp1d(inlocs, inparr, kind=interp_parms['interp_kind'], axis=-1, bounds_error=False, fill_value=NP.nan)
                    outarr = interpfunc_arr(outlocs)
                if outloc is None:
                    wts_interped[ax0ind,outind] = NP.copy(outwts)
                    arr_interped[ax0ind,outind] = NP.copy(outarr)
                else:
                    wts_interped[ax0ind,:] = NP.copy(outwts)
                    arr_interped[ax0ind,:] = NP.copy(outarr)
    else:
        wts_reshaped = NP.moveaxis(wts_filled, axis, wts_collapsed.ndim-1) # axis is the last dimension
        wts_reshaped_shape = wts_reshaped.shape
        mask_reshaped = NP.moveaxis(wts_collapsed.mask, axis, wts_collapsed.ndim-1) # axis is the last dimension
        arr_reshaped = NP.moveaxis(arr_filled, axis, arr_collapsed.ndim-1) # axis is the last dimension
        
        if interp_parms['op_type'].lower() == 'median': # Always typecasts to int which is a problem!!! Needs to be fixed.
            kernel = morphology.rectangle(1, interp_parms['window_size'], dtype=NP.float64)
            maxval = NP.nanmax(NP.abs(wts_reshaped)) 
            wts_interped = maxval * mean(img_as_float(wts_reshaped.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) # shape=(-1,nchan), use mean not median for weights, array must be normalized to lie inside [-1,1]
            maxval = NP.nanmax(NP.abs(arr_reshaped))
            if NP.iscomplexobj(arr):
                arr_interped = maxval * (median(img_as_float(arr_reshaped.real.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) + 1j * median(img_as_float(arr_reshaped.imag.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1]))) # input array must be normalized to lie inside [-1,1]
            else:
                arr_interped = maxval * median(img_as_float(arr_reshaped.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) # input array must be normalized to lie inside [-1,1]
        else:
            wts_filled = MA.filled(wts_collapsed, fill_value=NP.nan)
            wts_reshaped = NP.moveaxis(wts_filled, axis, wts_collapsed.ndim-1) # axis is the last dimension
            if interp_parms['op_type'].lower() == 'gaussian':
                fwhm = interp_parms['window_size']
                x_sigma = fwhm / (2.0 * NP.sqrt(2.0 * NP.log(2.0)))
                kernel1D = CONV.Gaussian1DKernel(x_sigma)
            elif interp_parms['op_type'].lower() == 'tophat':
                if interp_parms['window_size'] % 2 == 0:
                    interp_parms['window_size'] += 1
                kernel1D = CONV.Box1DKernel(interp_parms['window_size'])
            kernel = CONV.CustomKernel(kernel1D.array[NP.newaxis,:]) # Make a 2D kernel from the 1D kernel where it spans only one element in the new axis
            wts_interped = CONV.interpolate_replace_nans(wts_reshaped.reshape(-1,wts_reshaped_shape[-1]), kernel)
            if NP.iscomplexobj(arr):
                arr_interped = CONV.interpolate_replace_nans(arr_reshaped.real.reshape(-1,wts_reshaped_shape[-1]), kernel) + 1j * CONV.interpolate_replace_nans(arr_reshaped.imag.reshape(-1,wts_reshaped_shape[-1]), kernel)
            else:
                arr_interped = CONV.interpolate_replace_nans(arr_reshaped.reshape(-1,wts_reshaped_shape[-1]), kernel)
        
    if outloc is None:
        wts_interped = wts_interped.reshape(wts_reshaped_shape) # back to intermediate shape with axis as the last dimension
        arr_interped = arr_interped.reshape(wts_reshaped_shape) # back to intermediate shape with axis as the last dimension
    else:
        wts_interped = wts_interped.reshape(tuple(wts_reshaped_shape[:-1])+(wts_interped.shape[-1],)) # back to intermediate shape with axis as the last dimension
        arr_interped = arr_interped.reshape(tuple(wts_reshaped_shape[:-1])+(arr_interped.shape[-1],)) # back to intermediate shape with axis as the last dimension
    wts_interped = NP.moveaxis(wts_interped, wts_collapsed.ndim-1, axis) # Original shape
    arr_interped = NP.moveaxis(arr_interped, arr_collapsed.ndim-1, axis) # Original shape

    if fix_ampl is not None:
        if NP.iscomplexobj(arr):
            arr_interped *= fix_ampl / NP.abs(arr_interped)

    eps = 1e-10
    if isinstance(wts_interped, MA.MaskedArray):
        mask_out = NP.logical_or(wts_interped.data < eps, NP.isnan(wts_interped.data)) # Mask small, negative, and NaN weights
    else:
        mask_out = NP.logical_or(wts_interped < eps, NP.isnan(wts_interped)) # Mask small, negative, and NaN weights
    wts_interped = MA.array(wts_interped, mask=mask_out)
    arr_interped = MA.array(arr_interped, mask=mask_out)

    return (arr_interped, wts_interped)

################################################################################

def array_trace(inparr, offsets=None, axis1=0, axis2=1, outaxis='axis1',
                dtype=None):

    """
    ----------------------------------------------------------------------------
    A generalized array trace estimator which is a wrapper around numpy.trace()
    but produces multiple traces at once. Read documentation of numpy.trace()
    to fully comprehend this docstring

    Inputs:

    inparr      [numpy array] Input array (possibly multi-dimensional)

    offsets     [NoneType or numpy array] If set to None (default), traces along
                all possible offsets are computed. If set to numpy array, it 
                must be an array of integers, and traces along these offsets
                are only computed.

    axis1       [integer] Axes to be used as the first axis of the 2-D 
                sub-arrays from which the diagonals should be taken. Default=0 
                (first axis of inparr)

    axis2       [integer] Axes to be used as the second axis of the 2-D 
                sub-arrays from which the diagonals should be taken. Default=0 
                (second axis of inparr)

    outaxis     [string] Output axis specification where the traces will be
                placed. Accepted values are 'axis1' (default) and 'axis2' and
                that corresponding axis will serve as the primary axis based
                on which offsets are calculated. For example, if 
                outaxis='axis1', then offsets can range from -naxis1+1 to 
                naxis2+1. if outaxis='axis2', then offsets can range from 
                -naxis2+1 to naxis1+1. In the output, naxis1=len(offsets) 
                and axis2 is removed if outaxis='axis1', and vice versa.

    dtype       [NoneType or numpy datatype] Determines the data-type of the 
                returned array and of the accumulator where the elements are 
                summed. If dtype has the value None and a is of integer type of 
                precision less than the default integer precision, then the 
                default integer precision is used. Otherwise, the precision is 
                the same as that of inparr

    Outputs:

    A three-element tuple. The first element contains the trace(s). If inparr 
    is of shape (...,naxis1,...,naxis2,...), the output is of shape 
    (...,len(offsets),...,[],...) or (...,[],...,len(offsets),...). The second 
    element is a numpy array of offsets to indicate the signed offsets of the
    traces computed. The third element is a numpy array of weights indicating
    the number of elements that went into the trace along the specified 
    diagonal offset
    ----------------------------------------------------------------------------
    """

    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input inparr must be a numpy array')

    if not isinstance(axis1, int):
        raise TypeError('Input axis1 must be an integer')
    if not isinstance(axis2, int):
        raise TypeError('Input axis2 must be an integer')

    if not isinstance(outaxis, str):
        raise TypeError('Input outaxis must be a string')
    if outaxis.lower() not in ['axis1', 'axis2']:
        raise ValueError('Invalid input specified in outaxis')

    inpshape = NP.asarray(inparr.shape)
    if (axis1 >= inparr.ndim) or (axis2 >= inparr.ndim):
        raise ValueError('Input axes exceed the dimensions of the input array')

    if offsets is None:
        offsets = NP.arange(-inpshape[axis1]+1, inpshape[axis2])
        if outaxis.lower() == 'axis2':
            offsets = offsets[::-1]
    elif isinstance(offsets, int):
        offsets = NP.asarray(offsets).ravel()
    else:
        raise TypeError('Input offsets must be NoneType or integer')

    if outaxis.lower() == 'axis1':
        if NP.any(NP.logical_or(offsets < -inpshape[axis1]+1, offsets > inpshape[axis2]-1)):
            raise ValueError('One or more diagonal offsets exceed the dimensions of the array')
    else:
        if NP.any(NP.logical_or(offsets < -inpshape[axis2]+1, offsets > inpshape[axis1]-1)):
            raise ValueError('One or more diagonal offsets exceed the dimensions of the array')

    outarr = []
    weights = []

    for oind,offset in enumerate(offsets):
        if offset >= 0:
            weights += [min([inparr.shape[axis1], inparr.shape[axis2]-NP.abs(offset)])]
        else:
            weights += [min([inparr.shape[axis1]-NP.abs(offset), inparr.shape[axis2]])]
        outarr += [NP.trace(inparr, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=None)]

    weights = NP.asarray(weights)
    outarr = NP.asarray(outarr)
    if outaxis.lower() == 'axis1':
        outarr = NP.moveaxis(outarr, 0, axis1)
    else:
        outarr = NP.moveaxis(outarr, 0, axis2)

    return (outarr, offsets, weights)

################################################################################

def tile(inparr, reps, mirror_axes=None):

    """
    ----------------------------------------------------------------------------
    A generalized version of numpy.tile. It allows for non-integer values for
    repetitions. It also allows for tiling of selected axes in a mirrored
    fashion rather than periodic fashion thus providing smooth transitions 
    at the boundaries.

    Inputs:

    inparr  [numpy array] An N-dimensional numpy array which is to be tiled

    reps    [list or numpy array] Number of repeats along various axis. It can 
            contain non-integers but they have to be positive. Check numpy.tile
            behavior for when reps.size < inparr.ndim and 
            reps.size > inparr.ndim

    mirror_axes
            [list or numpy array] The axes which are to be mirrored first 
            and then repeated. The axes specified here should be present in
            inparr. If set to None (default), no mirroring is performed and
            the axes to be repeated are repeated periodically

    Output:

    Numpy array after tiling (and after possibly mirroring)
    ----------------------------------------------------------------------------
    """

    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input inparr must be a numpy array')
    if not isinstance(reps, (list, tuple, NP.ndarray)):
        raise TypeError('Input reps must be a list, tuple, or numpy array')
    else:
        reps = NP.asarray(reps).ravel().astype(float)
    if mirror_axes is not None:
        if not isinstance(mirror_axes, (list, tuple, NP.ndarray)):
            raise TypeError('Input mirror_axes must be a list, tuple, or numpy array')
        else:
            mirror_axes = NP.asarray(mirror_axes).ravel()
            if mirror_axes.dtype != int:
                raise TypeError('Input mirror_axes must consist of integers')
            mirror_axes = NP.unique(mirror_axes)

    inpshape = NP.asarray(inparr.shape)
    inp_ndim = inparr.ndim

    if NP.any(reps <= 0.0):
        raise ValueError('Input reps must be positive')

    if mirror_axes is not None:
        if NP.any(mirror_axes < 0) or NP.any(mirror_axes >= inp_ndim):
            raise ValueError('mirror_axes contains axes not present in input inparr')

    if inp_ndim < reps.size:
        for i in range(reps.size - inp_ndim):
            inparr = inparr[NP.newaxis,...]
            if mirror_axes is not None:
                mirror_axes += 1
        inp_ndim = inparr.ndim
        inpshape = NP.asarray(inparr.shape)

    if inp_ndim > reps.size:
        reps = NP.insert(reps, 0, NP.ones(inp_ndim-reps.size), axis=None)

    if mirror_axes is not None:
        for mirrorax in mirror_axes:
            inparr = NP.concatenate((inparr, NP.flip(inparr, mirrorax)), axis=mirrorax)
            inpshape[mirrorax] *= 2
            reps[mirrorax] /= 2
        
    outshape = NP.floor(inpshape * reps).astype(int)
    outarr = NP.tile(inparr, NP.ceil(reps).astype(int))

    idx_grid = NP.ix_(*[NP.arange(n) for n in outshape])

    outarr = outarr[idx_grid]

    return outarr

################################################################################

def minmax_scaler(inp, low=0.0, high=1.0, axis=None):
    """
    ----------------------------------------------------------------------------
    Scale an input between minimum and maximum values linearly

    Inputs:

    inp  [numpy array] An N-dimensional numpy array which is to be scaled

    low  [scalar] Minimum value of inp will be scaled to this value

    high [scalar] Maximum value of inp will be scaled to this value

    axis [scalar or None] Select axis along which min and max will be 
         determined. Otherwise (default=None), a flattened array will be used.

    Output:

    Numpy array after scaling
    ----------------------------------------------------------------------------
    """
    inp_std = (inp - NP.min(inp,axis=axis,keepdims=True)) / (NP.max(inp,axis=axis,keepdims=True) - NP.min(inp,axis=axis,keepdims=True))
    inp_scaled = inp_std * (high - low) + low
    return inp_scaled

################################################################################

def hermitian(inparr: NDArray[NP.complex128], axes: Optional[Union[List[int],Tuple[int,int],NDArray[int]]]=(-2,-1)) -> NDArray[NP.complex128]:
    """
    Return the Hermitian of the input array along specified axes.

    Parameters
    ----------
    inparr : numpy.ndarray
        Input array to be Hermitian-transposed.
    axes : {tuple, list, numpy.ndarray}, optional
        Two-element sequence denoting which two axes are to be Hermitian-transposed.
        Default is (-2, -1).

    Returns
    -------
    numpy.ndarray
        Hermitian-transposed array.

    Raises
    ------
    TypeError
        If inparr is not a numpy array or if axes is not a list, tuple, or numpy array.
    ValueError
        If axes is not a two-element list, tuple, or numpy array or if the two entries in axes are the same.

    Notes
    -----
    The Hermitian of an array is obtained by swapping the specified axes and taking the complex conjugate.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[1, 2 + 1j], [3 - 2j, 4]])
    >>> hermitian(arr, axes=(0, 1))
    array([[1.-0.j, 3.+2.j],
           [2.-1.j, 4.-0.j]])
    """
    # axes denotes which two axes are to be Hermitian-transposed
    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input array inparr must be a numpy array')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1, -1)

    if axes is None:
        axes = NP.asarray([-2, -1])
    if not isinstance(axes, (list, tuple, NP.ndarray)):
        raise TypeError('Input axes must be a list, tuple, or numpy array')
    axes = NP.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError('Input axes must be a two-element list, tuple, or numpy array')
    negind = NP.where(axes < 0)[0]
    if negind.size > 0:
        axes[negind] += inparr.ndim  # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    return NP.swapaxes(inparr, axes[0], axes[1]).conj()

################################################################################

def hat(inparr: NDArray[NP.complex128], axes: Optional[Union[List[int],Tuple[int,int],NP.ndarray]]=None) -> NDArray[NP.complex128]:
    """
    Compute the inverse of the Hermitian operation along specified axes (hat operation).

    Parameters
    ----------
    inparr : numpy.ndarray
        Input array to be inversed with the Hermitian operation.
    axes : Union[list, tuple, numpy.ndarray], optional
        Two-element sequence denoting which two axes are to be Hermitian-transposed.
        If None, defaults to np.asarray([-2, -1]).

    Returns
    -------
    numpy.ndarray
        Result of the hat operation, the inverse of the Hermitian-transposed array.

    Raises
    ------
    TypeError
        If inparr is not a numpy array.
    ValueError
        If axes is not a two-element list, tuple, or numpy array or if the two entries in axes are the same.

    Notes
    -----
    The hat operation is the inverse of the Hermitian operation. It involves computing the Hermitian-transposed array,
    taking the inverse along the last two axes using numpy.linalg.inv(), and optionally rearranging the axes.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[1, 2 + 1j], [3 - 2j, 4]])
    >>> hat_result = hat(arr, axes=(0, 1))
    """

    if not isinstance(inparr, NP.ndarray):
        raise TypeError('Input array inparr must be a numpy array')
    if inparr.ndim == 1:
        inparr = inparr.reshape(1, -1)

    if axes is None:
        axes = NP.asarray([-2, -1])
    if not isinstance(axes, (list, tuple, NP.ndarray)):
        raise TypeError('Input axes must be a list, tuple, or numpy array')
    axes = NP.asarray(axes).ravel()
    if axes.size != 2:
        raise ValueError('Input axes must be a two-element list, tuple, or numpy array')
    negind = NP.where(axes < 0)[0]
    if negind.size > 0:
        axes[negind] += inparr.ndim  # Convert negative axis numbers to positive
    if axes[0] == axes[1]:
        raise ValueError('The two entries in axes cannot be the same')

    inparr_H = hermitian(inparr, axes=axes)
    invaxes = inparr.ndim - 2 + NP.arange(2)  # For inverse, they must be the last two axes because of numpy.linalg.inv() requirements
    if not NP.array_equal(NP.sort(axes), invaxes):  # the axes are not at the end, so move them for taking inverse
        inparr_H = NP.moveaxis(inparr_H, NP.sort(axes), invaxes)
    if inparr_H.shape[-1] != inparr_H.shape[-2]:
        raise ValueError('The axes of inversion must be square in shape')
    inparr_IH = NP.linalg.inv(inparr_H)
    if not NP.array_equal(NP.sort(axes), invaxes):  # the axes were moved to the end, so move them back
        inparr_IH = NP.moveaxis(inparr_H, invaxes, NP.sort(axes))

    return inparr_IH

################################################################################

def sqrt_matrix_factorization(in_matrix: NDArray[NP.complex128]) -> NDArray[NP.complex128]:
    """
    Perform matrix square root factorization on a Hermitian matrix.

    The function first checks if the input matrix is Hermitian along the last two axes. 
    It then attempts to factorize the matrix using the Cholesky decomposition. If Cholesky 
    decomposition fails, Singular Value Decomposition (SVD) is used instead.

    Parameters
    ----------
    in_matrix : NDArray[NP.complex128]
        The input Hermitian matrix of shape (..., M, M).

    Returns
    -------
    NDArray[NP.complex128]
        The square root matrix with the same shape as the input matrix (..., M, M), 
        factorized either by Cholesky or SVD.

    Raises
    ------
    ValueError
        If the input matrix is not Hermitian along the last two axes.
    """

    # Check if the matrix is Hermitian along the last two axes
    if not NP.allclose(in_matrix, hermitian(in_matrix, axes=(-2,-1))):
        raise ValueError("Input matrix is not Hermitian along the last two axes.")

    # Number of leading dimensions in in_matrix
    num_leading_dims = in_matrix.ndim - 2  

    try:
        # Attempt Cholesky decomposition
        sqrt_matrix = NPLA.cholesky(in_matrix)
    except NPLA.LinAlgError:
        # Issue a warning if Cholesky fails
        warnings.warn(
            "The input matrix is not positive semi-definite. It indicates something may be wrong with the input matrix. SVD will be used, but results may be in error.",
            category=UserWarning
        )

        # If Cholesky fails, perform SVD
        U, S, Vh = NPLA.svd(in_matrix)

        # Get the size of the last dimension (dimsize)
        dimsize = S.shape[-1]

        # Create an identity matrix of shape (dimsize, dimsize) and expand it to broadcast across the leading dimensions
        # Multiply the identity matrix by S to create diagonal matrix with broadcastable dimensions
        sqrtS = NP.sqrt(S)[...,NP.newaxis] * NP.eye(dimsize).reshape((1,) * num_leading_dims + (dimsize,dimsize)) # Shape = (...,dimsize,dimsize)

        sqrt_matrix = U @ sqrtS

    return sqrt_matrix

################################################################################

def multivariate_gaussian(
    covariance: NDArray[NP.complex128], mean: NDArray[NP.complex128] = 0, 
    size: tuple = ()) -> NDArray[NP.complex128]:
    """
    Generate complex random variables from a multivariate Gaussian distribution.

    This function generates random variables that follow a multivariate complex Gaussian distribution
    with the specified mean and covariance. If `mean` is a scalar, it will apply the same value across 
    all dimensions. If `size` is not provided, it will be determined based on the first N-2 dimensions 
    of `covariance`. If both `size` and a multi-dimensional `covariance` are provided, they must be 
    broadcast-compatible.

    Parameters
    ----------
    covariance : NDArray[NP.complex128]
        Positive semi-definite covariance matrix, of shape (..., n, n) or (n, n). If batched, `...` 
        represents the batch dimensions.
    mean : NDArray[NP.complex128], optional
        Mean vector or scalar for the distribution, of shape (..., n), (n,), or scalar. Default is `0`. 
        If provided as a scalar, it will apply to all dimensions.
    size : tuple, optional
        Shape of the number of instances to generate, e.g., `(m,)` to generate `m` instances. If not 
        provided, the size is determined based on the first N-2 dimensions of `covariance`.

    Returns
    -------
    NDArray[NP.complex128]
        Generated complex random variables, of shape `size + (n,)`. The random variables follow 
        the specified multivariate Gaussian distribution.

    Raises
    ------
    ValueError
        If `size` is not broadcast-compatible with the first N-2 dimensions of `covariance`.

    Notes
    -----
    The function computes the square root of the covariance matrix using `sqrt_matrix_factorization()`.
    If the covariance matrix is not positive semi-definite, SVD is used to compute the square root.
    
    Examples
    --------
    >>> covariance = NP.array([[2+0j, 1+1j], [1-1j, 2+0j]], dtype=NP.complex128)
    >>> multivariate_gaussian(covariance).shape
    (2,)
    
    >>> covariance = NP.array([[[2+0j, 1+1j], [1-1j, 2+0j]], [[1+0j, 0+1j], [0-1j, 1+0j]]], dtype=NP.complex128)
    >>> multivariate_gaussian(covariance, size=(5,)).shape
    (5, 2, 2)
    """
    
    # Determine n from the last dimension of the covariance matrix
    n = covariance.shape[-1]
    
    # If mean is a scalar, convert it to a broadcastable shape of (..., n)
    mean = NP.array(mean)
    mean_shape = mean.shape

    # If size is not provided, infer size from covariance shape
    if not size:
        size = covariance.shape[:-2]
    
    # Check for broadcast compatibility between size and the first N-2 dimensions of covariance
    try:
        NP.broadcast_shapes(size, covariance.shape[:-2])
    except ValueError:
        raise ValueError("The provided `size` is not broadcast-compatible with the first N-2 dimensions of `covariance`.")
    
    mean = mean.reshape((1,)*(len(size)+1-len(mean_shape))+mean_shape) # Make it broadcastable to the output array

    # Compute the square root of the covariance matrix using sqrt_matrix_factorization
    sqrt_cov = sqrt_matrix_factorization(covariance)  # Shape (..., n, n)
    
    ## Generate standard normal random variables with mean 0 and variance 1
    rvs = (NP.random.normal(size=size + (n,)) + 1j * NP.random.normal(size=size + (n,))) / NP.sqrt(2.0)

    ## Matrix multiplication of sqrt_cov with the random variables + mean added
    modified_rvs = (sqrt_cov @ rvs[...,NP.newaxis])[...,0] + mean
   
    return modified_rvs

################################################################################

def unscented_transform(mean: NDArray[NP.float64], 
                        covariance: NDArray[NP.float64], 
                        func, 
                        alpha: float = 1.0, 
                        beta: float = 2.0, 
                        kappa: float = 0.0) -> tuple[NDArray[NP.float64], NDArray[NP.float64]]:
    """
    Perform the unscented transform on a given random variable defined by mean and covariance.

    Parameters
    ----------
    mean : NDArray[NP.float64]
        The mean of the random variable (shape (..., n)).
    covariance : NDArray[NP.float64]
        The covariance matrix of the random variable (shape (..., n, n)).
    func : callable
        A function that transforms the sigma points.
    alpha : float, optional
        Scaling parameter (default is 1).
    beta : float, optional
        Parameter for incorporating prior knowledge about the distribution (default is 2.0 for Gaussian).
    kappa : float, optional
        Secondary scaling parameter (default is 0.0).

    Returns
    -------
    transformed_mean : NDArray[NP.float64]
        The transformed mean.
    transformed_covariance : NDArray[NP.float64]
        The transformed covariance.
    """

    n = covariance.shape[-1]
    lambda_ = alpha ** 2 * (n + kappa) - n
    sigma_points_shape = mean.shape[:-1] + (2 * n + 1, n)
    sigma_points = NP.zeros(sigma_points_shape, dtype=NP.complex64)
    weights_mean = NP.zeros(2 * n + 1)
    weights_cov = NP.zeros(2 * n + 1)

    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta)
    weights_mean[1:] = weights_cov[1:] = 1 / (2 * (n + lambda_))

    try:
        if covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1]:
            sqrt_cov = SPLA.sqrtm((n + lambda_) * covariance)
        else:
            raise ValueError("Covariance matrix must be 2D square for sqrtm.")
    except (SPLA.LinAlgError, ValueError):
        sqrt_cov = sqrt_matrix_factorization((n + lambda_) * covariance)

    sigma_points[..., 0, :] = mean
    # Assign positive deviations: mean + sqrt_cov
    sigma_points[..., 1:n+1, :] = mean[..., NP.newaxis, :] + sqrt_cov

    # Assign negative deviations: mean - sqrt_cov
    sigma_points[..., n+1:, :] = mean[..., NP.newaxis, :] - sqrt_cov

    # sqrt_cov_expanded = sqrt_cov[..., NP.newaxis]
    # sigma_points[..., 1:n+1, :] = mean[..., NP.newaxis, :] + sqrt_cov_expanded.swapaxes(-1, -2)
    # sigma_points[..., n+1:, :] = mean[..., NP.newaxis, :] - sqrt_cov_expanded.swapaxes(-1, -2)

    # # Expand sqrt_cov for proper broadcasting with the mean
    # sqrt_cov_expanded = sqrt_cov[..., NP.newaxis, :, :]
    
    # # Assign positive deviations: mean + sqrt_cov
    # sigma_points[..., 1:n+1, :] = mean[..., NP.newaxis, :] + NP.swapaxes(sqrt_cov_expanded, -2, -1)
    
    # # Assign negative deviations: mean - sqrt_cov
    # sigma_points[..., n+1:, :] = mean[..., NP.newaxis, :] - NP.swapaxes(sqrt_cov_expanded, -2, -1)

    # for i in range(n):
    #     sigma_points[..., i + 1, :] = mean + sqrt_cov_expanded[..., i].squeeze()
    #     sigma_points[..., i + n + 1, :] = mean - sqrt_cov_expanded[..., i].squeeze()

    transformed_points = NP.array([func(point) for point in sigma_points.reshape(-1, n)]).reshape(sigma_points.shape)

    transformed_mean = NP.sum(weights_mean[..., NP.newaxis] * transformed_points, axis=-2)
    # transformed_mean = NP.sum(weights_mean[..., NP.newaxis] * transformed_points, axis=-2)

    devns = transformed_points - transformed_mean[..., NP.newaxis, :] 
    transformed_covariance = NP.einsum('...ki,...kj,k->...ij', devns, devns, weights_cov)
    # sqr_devns = devns[...,NP.newaxis] * devns[...,NP.newaxis,:].conj()
    # transformed_covariance = NP.mean()
    # transformed_covariance = NP.zeros(transformed_points.shape[-2:])
    # for i in range(2 * n + 1):
    #     diff = transformed_points[..., i, :] - transformed_mean
    #     transformed_covariance += weights_cov[i] * NP.einsum('...i,...j->...ij', diff, diff)

    return (transformed_mean, transformed_covariance)
