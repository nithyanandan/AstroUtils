import numpy as NP
import scipy as SP
from scipy import interpolate
import my_operations as OPS

################################################################################

def FT1D(inp, ax=-1, use_real=False, shift=False, verbose=True):

    """
    ---------------------------------------------------------------------
    Compute FFT from Numpy. 
    Inputs:

    inp:    Input data (vector or array) to be Fourier transformed

    Keyword Inputs:

    ax:         Axis (scalar integer) over which FFT is performed. 
                Default = -1 (last axis)

    use_real:   [Boolean scalar] If True, compute only the positive
                frequency components using the real part of the data

    oututs:    
    
    fftout: FFT of input data over the specified axes
    -------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('inp not defined. Aborting FT1D().')

    if not isinstance(inp, NP.ndarray):   # type(inp) is numpy.ndarray
        raise TypeError('Input array should be Numpy array data type')

    if use_real:
        inp = NP.real(inp)
        if verbose:
            print "Opted for FFT of real data. Hence performing numpy.rfft()."
            print "numpy.rfft() returns only positive frequencies."
        fftout = NP.fft.rfft(inp, axis=ax)
    else:
        fftout = NP.fft.fft(inp, axis=ax)

    if shift:
        fftout = NP.fft.fftshift(fftout, axes=ax)
    return fftout

################################################################################

def spectral_axis(length, delx=1.0, shift=False, use_real=False):

    """
    ----------------------------------------------------------------
    Compute spectral axis in the FFT

    Inputs:

    length:    Length of vector to be Fourier transformed

    Keyword Inputs:

    delx:        x-axis interval, used only in case of 1D inp.
                 Default = 1.0

    shift:       [Boolean scalar] True => Shift to center of frequencies

    use_real:    [Boolean scalar] True => Compute only positive 
                 frequencies using numpy.fft.rfftfreq() 

    Output:    
    
    spaxis: Discrete spectral axis in the output FFT
    ---------------------------------------------------------------
    """
    
    # try: 
    #     size(length) == 1 and isinstance(length, int)
    #     print type(length)
    # except: 
    #     print "length has to be a scalar positive integer."
    #     print "Aborted execution in my_DSP_modules.frequencies()"
    #     SYS.exit(1) # Abort execution

    if use_real:
        spaxis = NP.fft.rfftfreq(length, d=delx)
    else: 
        spaxis = NP.fft.fftfreq(length, d=delx)
        if shift:
            spaxis = NP.fft.fftshift(spaxis)

    return spaxis

################################################################################

def rfft_append(inp, axis=0):

    """
    ------------------------------------------------------------------
    Compute the negative frequency left out by numpy.rfft()
    and append in the right order to the output from numpy.rfft().

    Input:

    inp       Input data of any dimensions to which negative frequency 
              components have to be appended.

    Keyword Input: 

    axis      [scalar] Axis along which negative frequency components
              are to be appended. It has to be a scalar in the range
              0 to Ndim-1 where Ndim is the number of axes in the data.

    Output:

    Appended data along the axis specified. 
    -------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('inp undefined. Aborting rfft_append()')

    if not isinstance(inp, NP.ndarray):
        raise TypeError('inp should be Numpy array data type.')

    if isinstance(axis, (list, tuple, str)):
        raise TypeError('axis should be a scalar integer in the range 0 to Ndim-1')
    axis = int(axis)
    shp = NP.shape(inp)
    ndim = len(shp)

    if (axis < 0) or (axis >= ndim):
        raise ValueError("Input data does not contain the axis specified. Aborted execution in reverse()")

    if shp[axis] == 1:
        return inp

    return NP.append(inp, NP.conj(OPS.reverse(inp, axis=axis, ind_range=[1,shp[axis]-2])), axis=axis)

################################################################################

def rfftfreq_append(rfft_freqs):

    """
    ------------------------------------------------------------
    Compute the negative frequencies for the output of 
    numpy.rfftfreq() and rearrange the frequencies in the correct
    order. 

    Input: 

    rfft_freqs      [Vector] Positive frequencies

    Output:

    Positive and negative frequencies computed from numpy.rfftfreq()
    made equal to the output of numpy.fftfreq()
    ------------------------------------------------------------
    """

    try:
        rfft_freqs
    except NameError:
        raise NameError('Input rfft_freqs not specified. Aborting rfftfreq_append()')

    if not isinstance(rfft_freqs, (list, NP.ndarray)):
        raise TypeError('Input rfft_freqs should be a list or a 1D Numpy array')

    rfft_freqs = NP.asarray(rfft_freqs)

    return NP.append(rfft_freqs[:-1],-rfft_freqs[-1:0:-1],axis=0)

################################################################################

def shaping(N_samples, fraction=1.0, shape='rect', area_normalize=False,
            peak=None, verbose=True):
    
    """
    ----------------------------------------------------------------------------
    Routine to produce sequences which can be used as shaping windows for other
    sequences. 

    Inputs:

    N_samples    [Integer] Number of samples in the sequence. Should be positive

    Keyword inputs:

    fraction     [Float] Fraction of N_samples outside of which the sequence is 
                 zero-padded. The fraction of N_samples is where the shaped 
                 sequence is actually generated. For fraction less than unity,
                 the zero padding comes about symmetrically at the edges.

    shape        [string] Shape type. Currently allowed values are 'rect' and
                 'bnw' for rectangular and Blackman-Nuttall windows respectively

    area_normalize
                 [Boolean] True mean re-normalize the sequence to have unit
                 area. False means no re-normalization is performed. Cannot be
                 set simulataneously if peak is set.

    peak         [Float] If set, rescale the sequence so the peak is set to the
                 specified value. 

    verbose      [Boolean] If set, print progress and/or diagnostic messages.

    Output:

    samples      [Numpy array] Sequence containing the required shape and zero
                 padding if fraction < 1.0
    
    ----------------------------------------------------------------------------
    """

    try:
        N_samples
    except NameError:
        raise NameError('Number of samples undefined. Aborting shaping().')

    if (area_normalize) and (peak is not None):
        raise ValueError('Both area_normalize and peak cannot be set at the same time in shaping().')

    if not isinstance(area_normalize, bool):
        raise TypeError('area_normalize should be a boolean value. Aborting shaping().')

    if peak is not None:
        if not isinstance(peak, (int, float)):
            raise ValueError('Peak should be a scalar value. Aborting shaping().')

    if not isinstance(N_samples, (int, float)):
        raise TypeError('N_samples should be a positive integer. Aborting shaping().')
    else:
        if N_samples < 1.0:
            raise ValueError('N_samples should be a positive integer. Aborting shaping().')
        N_samples = int(N_samples)

    if fraction <= 0.0:
        raise ValueError('fraction should be in the range 0.0 < fraction <= 1.0. Aborting shaping().')
    
    if fraction > 1.0:
        fraction = 1.0
        if verbose:
            print 'fraction was found to exceed 1.0. Resetting fraction to 1.0 in shaping().'
    
    center = int(0.5 * N_samples)
    # PDB.set_trace()
    N_window = N_samples * fraction

    if (N_window % 2) == 0.0:
        if (shape == 'bnw'):
            N_window = int(N_window - 1)
        if (N_window < N_samples) and (shape == 'rect'):
            N_window = int(N_window - 1)
    elif (N_window % 2.0) < 1.0:
        N_window = NP.ceil(N_window)
    elif (N_window % 2.0) >= 1.0:
        N_window = NP.floor(N_window)

    if shape == 'rect':
        window = NP.ones(N_window)
    elif shape == 'bnw':
        a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
        window = a[0]*NP.ones(N_window) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window)/(N_window-1)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window)/(N_window-1)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window)/(N_window-1))

    N_zeros = N_samples - N_window

    if N_zeros > 0:
        N_zeros_pfx = NP.ceil(0.5*N_zeros)
        samples = NP.concatenate((NP.zeros(N_zeros_pfx), window))
        N_zeros_sfx = N_zeros - N_zeros_pfx
        if N_zeros_sfx > 0:
            samples = NP.concatenate((samples, NP.zeros(N_zeros_sfx)))
    else:
        samples = window

    if peak is not None:
        samples *= peak/NP.amax(NP.abs(samples))
        if verbose:
            print 'Rescaled the shaping window to peak value.'
    elif area_normalize:
        area = NP.trapz(samples) # Beware that NP.trapz could differ from NP.cumsum due to edge effects. Sufficient padding will make them converge
        samples /= area
        if verbose:
            print 'Renormalized the shaping window to unit area.'

    return samples

################################################################################

def downsampler(inp, factor, axis=-1, verbose=True, kind='linear',
                fill_value=NP.nan):

    """
    ----------------------------------------------------------------------------
    Routine to downsample a given input sequence along a specific dimension 
    where the input could be multi-dimensional (up to 8 dimensions)

    Inputs:

    inp           [Numpy array] array which has to be downsampled. Cannot have
                  more than 8 dimensions

    factor        [scalar] downsampling factor. positive integer or floating
                  point number greater than or equal to unity. If an integer, 
                  output is simply a sampled subset of the input. If not an 
                  integer, downsampling is obtained by interpolation.

    Keyword Inputs:

    axis          [scalar] Integer specifying the axis along which the array is
                  to be downsampled. Default = -1, the last axis.

    verbose       [Boolean] If set to True, will print progress and/or
                  diagnostic messages. If False, will suppress printing such
                  messages. Default = True

    kind          [string] Spcifies the kind of interpolation. This is used only
                  if factor is not an integer thus requiring interpolation. 
                  Accepted values are 'linear', 'quadratic' and 'cubic'.
                  Default = 'linear'

    fill_value    [scalar] Value to fill locations outside the index range of 
                  input array. Default = NaN
    ----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('No input specified. Aborting downsampler().')

    try:
        factor
    except NameError:
        if verbose:
            print 'No downsampling factor specified. No downsampling performed on input.'
        return input

    if not isinstance(inp, NP.ndarray):
        raise TypeError('Input should be a numpy array. Aborting downsampler().')

    if not isinstance(factor, (int, float)):
        raise TypeError('Downsampling factor must be a scalar value.')

    if factor < 1.0:
        raise ValueError('Downsampling factor must be greater than 1.')

    if len(inp.shape) < 2:
        inp = inp.reshape(1,-1)

    if (axis <= -len(inp.shape)) or (axis > len(inp.shape)):
        raise IndexError('The axis specified does not exist in the input. Aborting downsampler().')

    if len(inp.shape) > 8:
        raise ValueError('The routine cannot handle inputs with more than 8 dimensions. Aborting downsampler().')

    # PDB.set_trace()
    axis = range(len(inp.shape))[axis]
    if (factor % 1) == 0:
        factor = int(factor)
        if len(inp.shape) == 2:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor]
        elif len(inp.shape) == 3:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:]
            elif (axis + 8) % 8 == 2:
                return inp[:,:,::factor]
        elif len(inp.shape) == 4:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:]
            elif (axis + 8) % 8 == 2:
                return inp[:,:,::factor,:]
            elif (axis + 8) % 8 == 3:
                return inp[:,:,:,::factor]
        elif len(inp.shape) == 5:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor]
        elif len(inp.shape) == 6:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor]
        elif len(inp.shape) == 7:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 6:      
                return inp[:,:,:,:,:,:,::factor]
        elif len(inp.shape) == 8:
            if (axis + 8) % 8 == 0:
                return inp[::factor,:,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 1:
                return inp[:,::factor,:,:,:,:,:,:]
            elif (axis + 8) % 8 == 2:      
                return inp[:,:,::factor,:,:,:,:,:]
            elif (axis + 8) % 8 == 3:      
                return inp[:,:,:,::factor,:,:,:,:]
            elif (axis + 8) % 8 == 4:      
                return inp[:,:,:,:,::factor,:,:,:]
            elif (axis + 8) % 8 == 5:      
                return inp[:,:,:,:,:,::factor,:,:]
            elif (axis + 8) % 8 == 6:      
                return inp[:,:,:,:,:,:,::factor,:]
            elif (axis + 8) % 8 == 7:      
                return inp[:,:,:,:,:,:,:,::factor]
    else:
        if verbose:
            print 'Determining the interpolating function for downsampling.'
        intpfunc = interpolate.interp1d(NP.arange(inp.shape[axis]), inp,
                                        kind=kind, fill_value=fill_value,
                                        axis=axis) 
        reqd_inds = NP.arange(0, inp.shape[axis], factor)
        if verbose:
            print 'Returning the downsampled data.'
        return intpfunc(reqd_inds)

################################################################################

def upsampler(inp, factor, axis=-1, verbose=True, kind='linear',
              fill_value=NP.nan):

    """
    ----------------------------------------------------------------------------
    Routine to upsample a given input sequence along a specific dimension 
    where the input could be multi-dimensional (up to 8 dimensions)

    Inputs:

    inp           [Numpy array] array which has to be upsampled. Cannot have
                  more than 8 dimensions

    factor        [scalar] upsampling factor. positive integer or floating
                  point number greater than or equal to unity. Upsampling is
                  obtained by interpolation.

    Keyword Inputs:

    axis          [scalar] Integer specifying the axis along which the array is
                  to be upsampled. Default = -1, the last axis.

    verbose       [Boolean] If set to True, will print progress and/or
                  diagnostic messages. If False, will suppress printing such
                  messages. Default = True

    kind          [string] Spcifies the kind of interpolation. Accepted values
                  are 'linear', 'quadratic' and 'cubic'. Default = 'linear'

    fill_value    [scalar] Value to fill locations outside the index range of 
                  input array. Default = NaN
    ----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('No input specified. Aborting upsampler().')

    try:
        factor
    except NameError:
        if verbose:
            print 'No upsampling factor specified. No upsampling performed on input.'
        return input

    if not isinstance(inp, NP.ndarray):
        raise TypeError('Input should be a numpy array. Aborting upsampler().')

    if not isinstance(factor, (int, float)):
        raise TypeError('Upsampling factor must be a scalar value.')

    if factor < 1.0:
        raise ValueError('Upsampling factor must be greater than 1.')

    if len(inp.shape) < 2:
        inp = inp.reshape(1,-1)

    if (axis <= -len(inp.shape)) or (axis > len(inp.shape)):
        raise IndexError('The axis specified does not exist in the input. Aborting upsampler().')

    if len(inp.shape) > 8:
        raise ValueError('The routine cannot handle inputs with more than 8 dimensions. Aborting upsampler().')

    if factor == 1:
        if verbose:
            print 'Upsampling factor is 1. No upsampling performed. Returning the original array.'
        return inp
    else:
        if verbose:
            print 'Determing the interpolating function for upsampling.'
        intpfunc = interpolate.interp1d(NP.arange(inp.shape[axis]), inp,
                                        kind=kind, fill_value=fill_value,
                                        axis=axis) 
        reqd_inds = NP.arange(0, inp.shape[axis], 1/factor)
        if verbose:
            print 'Returning the upsampled data.'
        return intpfunc(reqd_inds)
        
################################################################################
    
