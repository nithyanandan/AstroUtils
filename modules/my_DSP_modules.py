import numpy as NP
from scipy import signal
from scipy import interpolate
import my_operations as OPS
import lookup_operations as LKP
import ipdb as PDB

#################################################################################

def FT1D(inp, ax=-1, use_real=False, shift=False, inverse=False, verbose=True):

    """
    -----------------------------------------------------------------------------
    Compute FFT/IFFT using Numpy. 

    Inputs:

    inp:    Input data (vector or array) to be Fourier transformed

    Keyword Inputs:

    ax:         Axis (scalar integer) over which FFT is performed. Default = -1
                (last axis)

    use_real:   [Boolean scalar] If True, compute only the positive frequency
                components using the real part of the data

    shift:      [Boolean] If True, shift the result to make the zeroth component
                move to the center. Default=False

    inverse:    [Boolean] If True, compute the inverse FFT. If False, compute the
                FFT. Default=False

    oututs:    
    
    fftout: FFT/IFFT of input data over the specified axes
    -----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('inp not defined. Aborting FT1D().')

    if not isinstance(inp, NP.ndarray):   # type(inp) is numpy.ndarray
        raise TypeError('Input array should be Numpy array data type')

    if inverse:
        if use_real:
            inp = NP.real(inp)
            if verbose:
                print "Opted for IFFT of real data. Hence performing numpy.irfft()."
                print "numpy.irfft() returns only positive frequencies."
            fftout = NP.fft.irfft(inp, axis=ax)
        else:
            fftout = NP.fft.ifft(inp, axis=ax)
    
        if shift:
            fftout = NP.fft.ifftshift(fftout, axes=ax)
    else:
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

#################################################################################

def spectral_axis(length, delx=1.0, shift=False, use_real=False):

    """
    -----------------------------------------------------------------------------
    Compute spectral axis in the FFT

    Inputs:

    length:    Length of vector to be Fourier transformed

    Keyword Inputs:

    delx:        x-axis interval, used only in case of 1D inp. Default = 1.0

    shift:       [Boolean scalar] True => Shift to center of frequencies

    use_real:    [Boolean scalar] True => Compute only positive frequencies using
                 numpy.fft.rfftfreq() 

    Output:    
    
    spaxis: Discrete spectral axis in the output FFT
    -----------------------------------------------------------------------------
    """
        
    if use_real:
        spaxis = NP.fft.rfftfreq(length, d=delx)
    else: 
        spaxis = NP.fft.fftfreq(length, d=delx)
        if shift:
            spaxis = NP.fft.fftshift(spaxis)

    return spaxis

#################################################################################

def rfft_append(inp, axis=0):

    """
    -----------------------------------------------------------------------------
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
    -----------------------------------------------------------------------------
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

#################################################################################

def rfftfreq_append(rfft_freqs):

    """
    -----------------------------------------------------------------------------
    Compute the negative frequencies for the output of numpy.rfftfreq() and
    rearrange the frequencies in the correct order. 

    Input: 

    rfft_freqs      [Vector] Positive frequencies

    Output:

    Positive and negative frequencies computed from numpy.rfftfreq() made equal
    to the output of numpy.fftfreq()
    -----------------------------------------------------------------------------
    """

    try:
        rfft_freqs
    except NameError:
        raise NameError('Input rfft_freqs not specified. Aborting rfftfreq_append()')

    if not isinstance(rfft_freqs, (list, NP.ndarray)):
        raise TypeError('Input rfft_freqs should be a list or a 1D Numpy array')

    rfft_freqs = NP.asarray(rfft_freqs)

    return NP.append(rfft_freqs[:-1],-rfft_freqs[-1:0:-1],axis=0)

#################################################################################

def shaping(N_samples, fraction=1.0, shape='rect', area_normalize=False,
            peak=None, verbose=True, centering=False):
    
    """
    -----------------------------------------------------------------------------
    Routine to produce sequences which can be used as shaping windows for other
    sequences. 

    Inputs:

    N_samples    [Integer] Number of samples in the sequence. Should be positive

    Keyword inputs:

    fraction     [Float] Fraction of N_samples outside of which the sequence is 
                 zero-padded. The fraction of N_samples is where the shaped 
                 sequence is actually generated. For fraction less than unity,
                 the zero padding comes about symmetrically at the edges.

    shape        [string] Shape type. Currently allowed values are 'rect', 'bhw'
                 and 'bnw' for rectangular, Blackman-Harris and Blackman-Nuttall
                 windows respectively

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
    
    -----------------------------------------------------------------------------
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
    N_window = N_samples * fraction

    if (N_window % 2) == 0.0:
        if (shape == 'bnw') or (shape == 'BNW') or (shape == 'bhw') or (shape == 'BHW'):
            N_window = int(N_window - 1)
        if (N_window < N_samples) and ((shape == 'rect') or (shape == 'RECT')):
            N_window = int(N_window - 1)
    elif (N_window % 2.0) < 1.0:
        N_window = NP.ceil(N_window)
    elif (N_window % 2.0) >= 1.0:
        N_window = NP.floor(N_window)

    if (shape == 'rect') or (shape == 'RECT'):
        window = NP.ones(N_window)
    elif (shape == 'bnw') or (shape == 'BNW'):
        a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
        window = a[0]*NP.ones(N_window) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window)/(N_window-1)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window)/(N_window-1)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window)/(N_window-1))
    elif (shape == 'bhw') or (shape == 'BHW'):
        a = [0.35875, -0.48829, 0.14128, -0.01168]
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

#################################################################################

def windowing(N_window, shape='rect', pad_width=0, pad_value=0.0, 
              area_normalize=False, peak=None, verbose=True, centering=True):
    
    """
    -----------------------------------------------------------------------------
    Routine to produce sequences which can be used as shaping windows for other
    sequences. 

    Inputs:

    N_window     [Integer] Number of samples in the actual window. Should be
                 positive

    Keyword inputs:

    shape        [string] Shape type. Currently allowed values are 'rect',
                 'bnw' and 'bhw' for rectangular, Blackman-Nuttall and 
                 Blackman-Harris windows respectively

    pad_width    [scalar integer] Number of padding samples. it has to be 
                 non-negative. Padding values are provided in pad_values.

    area_normalize
                 [Boolean] True mean re-normalize the window to have unit
                 area. False means no re-normalization is performed. Cannot be
                 set simulataneously if peak is set.

    peak         [Float] If set, rescale the window so the peak is set to the
                 specified value. 

    verbose      [Boolean] If set, print progress and/or diagnostic messages.

    centering    [Boolean] If set to True, centers the window with close to 
                 symmetric on either side. If False, padding is done on the 
                 right side. Default = True

    Output:

    window       [Numpy array] window containing the required shape and padding
                 if pad_width > 0
    
    -----------------------------------------------------------------------------
    """

    try:
        N_window
    except NameError:
        raise NameError('Window size undefined. Aborting windowing().')

    if (area_normalize) and (peak is not None):
        raise ValueError('Both area_normalize and peak cannot be set at the same time in windowing().')

    if not isinstance(area_normalize, bool):
        raise TypeError('area_normalize should be a boolean value. Aborting windowing().')

    if peak is not None:
        if not isinstance(peak, (int, float)):
            raise ValueError('Peak should be a scalar value. Aborting windowing().')

    if not isinstance(N_window, (int, float)):
        raise TypeError('N_window should be a positive integer. Aborting windowing().')
    else:
        N_window = int(N_window)
        if N_window < 1:
            raise ValueError('N_window should be a positive integer. Aborting windowing().')

    if not isinstance(pad_width, (int, float)):
        raise TypeError('pad_width must be an integer.')
    else:
        pad_width = int(pad_width)

    if pad_width < 0:
        raise ValueError('pad_width should be non-negative. Aborting windowing().')
    
    if (shape == 'rect') or (shape == 'RECT'):
        if not centering:
            window = NP.pad(NP.ones(N_window), (0, pad_width), mode='constant', constant_values=(pad_value, pad_value))
        else:
            window = NP.pad(NP.ones(N_window), (NP.ceil(0.5*pad_width), NP.floor(0.5*pad_width)), mode='constant', constant_values=(pad_value, pad_value))
    elif (shape == 'bnw') or (shape == 'BNW'):
        a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
        if (N_window % 2 == 1):
            win = a[0]*NP.ones(N_window) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window)/(N_window-1)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window)/(N_window-1)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window)/(N_window-1))
            if not centering:
                if pad_width >= 1:
                    window = NP.pad(win, (1, pad_width-1), mode='constant', constant_values=(pad_value, pad_value))
                else:
                    window = win
            else:
                window = NP.pad(win, (NP.ceil(0.5*pad_width), NP.floor(0.5*pad_width)), mode='constant', constant_values=(pad_value, pad_value))
        else:
            win = a[0]*NP.ones(N_window-1) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window-1)/(N_window-2)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window-1)/(N_window-2)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window-1)/(N_window-2))
            if not centering:
                window = NP.pad(win, (1, pad_width), mode='constant', constant_values=(pad_value, pad_value))
            else:
                window = NP.pad(win, (NP.ceil(0.5*(pad_width+1)), NP.floor(0.5*(pad_width+1))), mode='constant', constant_values=(pad_value, pad_value))
    elif (shape == 'bhw') or (shape == 'BHW'):
        a = [0.35875, -0.48829, 0.14128, -0.01168]
        if (N_window % 2 == 1):
            win = a[0]*NP.ones(N_window) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window)/(N_window-1)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window)/(N_window-1)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window)/(N_window-1))
            if not centering:
                if pad_width >= 1:
                    window = NP.pad(win, (1, pad_width-1), mode='constant', constant_values=(pad_value, pad_value))
                else:
                    window = win
            else:
                window = NP.pad(win, (NP.ceil(0.5*pad_width), NP.floor(0.5*pad_width)), mode='constant', constant_values=(pad_value, pad_value))
        else:
            win = a[0]*NP.ones(N_window-1) + a[1]*NP.cos(2*NP.pi*NP.arange(N_window-1)/(N_window-2)) + a[2]*NP.cos(4*NP.pi*NP.arange(N_window-1)/(N_window-2)) + a[3]*NP.cos(6*NP.pi*NP.arange(N_window-1)/(N_window-2))
            if not centering:
                window = NP.pad(win, (1, pad_width), mode='constant', constant_values=(pad_value, pad_value))
            else:
                window = NP.pad(win, (NP.ceil(0.5*(pad_width+1)), NP.floor(0.5*(pad_width+1))), mode='constant', constant_values=(pad_value, pad_value))

    if peak is not None:
        window *= peak/NP.amax(NP.abs(window))
        if verbose:
            print '\tRescaled the shaping window to peak value.'
    elif area_normalize:
        # area = NP.trapz(window) # Beware that NP.trapz could differ from NP.cumsum due to edge effects. Sufficient padding will make them converge
        area = NP.sum(window) # Using sum is preferable to using trapz although less accurate especially when FFT is going to be involved later on.
        window /= area
        if verbose:
            print '\tRenormalized the shaping window to unit area.'

    return window

#################################################################################

def downsampler(inp, factor, axis=-1, verbose=True, kind='linear',
                fill_value=NP.nan):

    """
    -----------------------------------------------------------------------------
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
    -----------------------------------------------------------------------------
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

#################################################################################

def upsampler(inp, factor, axis=-1, verbose=True, kind='linear',
              fill_value=NP.nan):

    """
    -----------------------------------------------------------------------------
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
    -----------------------------------------------------------------------------
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
        
#################################################################################
    
def XC(inp1, inp2=None, pow2=False, shift=True):
    """
    -----------------------------------------------------------------------------
    Cross-correlate two sequences.

    Inputs:

    inp1:    [list or numpy array] First sequence.

    inp2:    [list or numpy array] If not given, auto-correlation of inp1 is
             returned.

    pow2     [boolean] If set to True, will pad the results of the correlation 
             with zeros so the length of the correlated sequence is equal to the
             next power of 2. If set to False, the correlated sequence is just 
             padded with one sample of value 0. Default = False

    shift:   [Boolean] If True, shift the correlated product such that it is 
             represented in FFT format. i.e., the first sample corresponds to
             zero lag followed by positive lags. The second half corresponds to
             negative lags. Default = True

    Output:  The correlation of input sequences inp1 and inp2. The output is of 
             length len(inp1)+len(inp2)-1 zero padded to the nearest power of 2 
             (if pow2 is True) or zero padded by one sample (if pow2 is False)
             and shifted to be identical to a Fourier transform based estimate.
    
    -----------------------------------------------------------------------------
    """

    try:
        inp1
    except NameError:
        raise NameError('inp1 not defined. Aborting XC().')

    if not isinstance(inp1, (list, tuple, NP.ndarray, int, float, complex)):
        raise TypeError('inp1 is of the wrong data type. Check inputs again. Aborting XC().')

    inp1 = NP.asarray(inp1)

    if inp2 is None:
        inp2 = NP.copy(inp1)
    elif not isinstance(inp2, (list, tuple, int, float, complex, NP.ndarray)):
        raise TypeError('inp2 has incompatible data type. Verify inputs. Aborting XC().')

    inp2 = NP.asarray(inp2)

    if pow2:
        zero_pad_length = 2**NP.ceil(NP.log2(len(inp1)+len(inp2)-1))-(len(inp1)+len(inp2)-1)
    else:
        zero_pad_length = 1

    xc = NP.pad(NP.correlate(inp1, inp2, mode='full'), (zero_pad_length,0), mode='constant', constant_values=(0.0,0.0))
    xc = NP.roll(xc, -int(NP.floor(0.5*zero_pad_length)))
    # xc = NP.append(NP.correlate(inp1, inp2, mode='full'), NP.zeros(zero_pad_length))
    if shift:
        # xc = NP.roll(xc, -(inp2.size-1))
        xc = NP.fft.ifftshift(xc)

    return xc

#################################################################################  

def spectax(length, resolution=1.0, shift=True, use_real=False):
    """
    -----------------------------------------------------------------------------
    Determine the spectral axis after a Fourier Transform

    Inputs:

    length     [Scalar] Positive integer specifying the length of sequence which is
               to be Fourier transformed

    resolution [Scalar] Positive value for resolution in the sequence before
               Fourier Transform

    Keyword Inputs:

    use_real   [Boolean] If true, the input sequence is assumed to consist only
               of real values and the spectral axis is computed accordingly. 
               Default = False

    shift      [Boolean] If true, the spectral axis values are shifted 
               cooresponding to a fftshift. Default = True

    Output:

    Spectral axis for an input sequence of given length and resolution.
    -----------------------------------------------------------------------------
    """
    
    try:
        length
    except NameError:
        raise NameError('Input length not defined. Aborting spectax().')
        
    if not isinstance(resolution, (int, float)):
        raise TypeError('Input resolution must be a positive scalar integer or floating point number. Aborting spectax().')
    elif resolution < 0.0:
        raise ValueError('Input resolution must be positive. Aborting spectax().')

    return spectral_axis(length, resolution, shift, use_real)

#################################################################################

def smooth(inp, wts=None, width=None, stat='mean', verbose=True):

    """
    -----------------------------------------------------------------------------
    Smoothen the input data using a moving average or median window along an
    axis

    Inputs:

    inp         [Numpy vector or array] M x N numpy array which has to be 
                smoothed across columns. 

    Keyword Inputs:

    wts         [Numpy vector] 1 x P array which will be used as the window of
                weights in case of a moving average. Will not be used if a 
                median is used in place of mean. P <= N. Sum of the weights
                should equal unity, otherwise the weights will be accordingly
                scaled. Default = None. If not set, then it will be set to a 
                rectangular window of width specified in width (see below)

    width       [scalar] Width of the moving window. Has to be positive. Default
                is None. If width is None, wts should be set. One and only one
                among wts and width should be set. 

    stat        [string scalar] String specifying the statistic ('mean' or
                'median') to be used. Default = 'mean'

    verbose     [boolean] If set to True (default), print messages indicating
                progress

    Output:
    
    Smoothed output (M x N numpy array)
    -----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('No input sequence specified.')

    if stat is None:
        stat = 'mean'

    if (stat != 'mean') and (stat != 'median'):
        raise ValueError('stat can only be either mean or median.')

    if (wts is None) and (width is None):
        raise NameError('Neither weights nor window width specified.')

    if wts is not None:
        wts = NP.asarray(wts)
        wts_shape = wts.shape
        if len(wts_shape) == 1:
            wts /= NP.sum(wts)
        elif (wts_shape[0] == 1) or (wts_shape[1] == 1):
            wts = NP.ravel(wts)
        else:
            raise TypeError('wts must be a vector.')
        width = wts.size
    else:
        width = int(width)
        if width <= 0:
            raise ValueError('Window width has to be positive.')
        wts = NP.ones(width)/width

    if width == 1:
        if verbose:
            print '\tWindow width is one. Input will be returned without smoothing.'
            return inp

    if stat == 'mean':
        out = NP.convolve(inp, wts, mode='same')
    else:
        if width % 2 == 0:
            if verbose:
                raise ValueError('\tWindow width must be odd for median filtering.')
        else:
            out = signal.medfilt(inp, width) 

    return out

#################################################################################  

def filter_(inp, wts=None, width=None, passband='low', verbose=True):    
    
    """
    -----------------------------------------------------------------------------
    Filter the input data using a low or high pass filter in frequency domain 
    along an axis

    Inputs:

    inp         [Numpy vector or array] M x N numpy array which has to be 
                filtered across columns. 

    Keyword Inputs:

    wts         [Numpy vector] 1 x P or M x P array which will be used as the 
                frequency window of weights. P <= N. Zeroth frequency of the 
                weights should equal unity, otherwise the weights will be 
                scaled accordingly. Default = None. If not set, then it will be 
                set to a rectangular window of width specified in width
                (see below) and will be applied as a filter identically to all
                rows

    width       [scalar] Width of the frequency window as a fraction of the 
                bandwidth (or equivalently N). Has to be positive. Default
                is None. If width is None, wts should be set. One and only one
                among wts and width should be set. 

    passband    [string scalar] String specifying the passband ('low' or 'high')
                to be used. Default = 'low'

    verbose     [boolean] If set to True (default), print messages indicating
                progress

    Output:
    
    Filtered output (M x N numpy array)
    -----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('No input specified for filtering.')

    if isinstance(inp, list):
        inp = NP.asarray(inp)
    elif not isinstance(inp, NP.ndarray):
        raise TypeError('Input should be of type list or numpy array.')

    if len(inp.shape) == 1:
        inp = inp.reshape(1,-1)
    elif (inp.shape[0] == 1) or (inp.shape[1] == 1):
        inp = inp.reshape(1,-1)

    if (passband != 'low') and (passband != 'high'):
        raise ValueError('Invalid passband specified. Valid passbands are low or high.')

    if (wts is None) and (width is None):
        raise NameError('Neither frequency weights nor filter width specified.')

    if wts is None:
        if not isinstance(width, (int,float)):
            raise TypeError('Filter width should be a scalar.')

        if width <= 0.0:
            raise ValueError('Filter width should be positive.')
        elif width >= 1.0:
            if verbose:
                print '\tFilter width exceeds 1.0. Returning input without filtering.'
            return inp

        filter_width = inp.shape[1] * width

        # Even samples in input or low passband, keep the filter width odd
        # Odd samples in input and high passband, keep the filter width even
        # to have no imaginary parts after filtering

        if (inp.shape[1] % 2 == 0) or (passband == 'low'): 
            if NP.floor(filter_width) % 2 == 0:            
                filter_width = NP.floor(filter_width) + 1
                if filter_width > inp.shape[1]:
                    filter_width = inp.shape[1]
            else:
                filter_width = NP.floor(filter_width)

            wts = NP.ones(filter_width).reshape(1,-1) # Simple rectangular filter
            pads = inp.shape[1] - filter_width

            if pads > 0:
                wts = NP.hstack((wts, NP.zeros(pads).reshape(1,-1)))

            wts = NP.repeat(wts, inp.shape[0], axis=0)

            if passband == 'low':
                wts = NP.roll(wts, -int(0.5*filter_width), axis=1)
            else:
                wts = NP.fft.fftshift(NP.roll(wts, -int(0.5*filter_width), axis=1), axes=1)
        else:
            if NP.floor(filter_width) % 2 != 0:
                filter_width = NP.floor(filter_width) + 1
                if filter_width > inp.shape[1]:
                    filter_width = inp.shape[1]
            else:
                filter_width = NP.floor(filter_width)

            wts = NP.ones(filter_width).reshape(1,-1) # Simple rectangular filter
            pads = inp.shape[1] - filter_width

            if pads > 0:
                wts = NP.hstack((wts, NP.zeros(pads).reshape(1,-1)))

            wts = NP.repeat(wts, inp.shape[0], axis=0)
            wts = NP.fft.fftshift(NP.roll(wts, -int(filter_width/2 - 1), axis=1), axes=1)
            
    else:
        if isinstance(wts, list):
            wts = NP.asarray(list)
        elif not isinstance(wts, NP.ndarray):
            raise TypeError('Frequency weights should be a numpy array.')

        if len(wts.shape) > 2:
            raise IndexError('Dimensions of frequency weights exceed dimensions of input.')
        elif len(wts.shape) == 1:
            wts = wts.reshape(1,-1)
        elif (wts.shape[0] == 1) or (wts.shape[1] == 1):
            wts = wts.reshape(1,-1)
        elif (wts.shape[0] > inp.shape[0]) or (wts.shape[1] > inp.shape[1]):
            raise IndexError('Dimensions of frequency weights exceed dimensions of input.')

        wshape = wts.shape

        if (wts.shape[0] != 1) and (wts.shape[0] != inp.shape[0]):
            raise IndexError('Dimensions of frequency weights exceed dimensions of input.')
        
        pads = inp.shape[1] - wts.shape[1]
        if pads > 0:
            if (wts.shape[0] == 1):
                wts = NP.hstack((wts, NP.zeros(pads).reshape(-1,1)))
                wts = NP.repeat(wts, inp.shape[0], axis=0)
            else:
                wts = NP.hstack(wts, NP.zeros((inp.shape[0],pads)))
        else:
            if (wts.shape[0] == 1):
                wts = NP.repeat(wts, inp.shape[0], axis=0)

        if passband == 'low':
            wts = NP.roll(wts, -int(0.5*wshape[1]), axis=1)
        else:
            wts = NP.fft.fftshift(NP.roll(wts, -int(0.5*wshape[1]), axis=1), axes=1)        

        wts = wts/wts[0,0] # Scale the weights to have zeroth frequency to have weight of unity

    return NP.fft.ifft(NP.fft.fft(inp, axis=1) * wts, axis=1)
        
#################################################################################  
        
def PFB_empirical(nchan, bank_nchan, rise_frac_half_max, fall_frac_half_max,
                  log_fall_rise=True, verbose=True):        

    """
    -----------------------------------------------------------------------------
    Create a polyphase filter bank from empirical models. 

    Inputs:

    nchan         [scalar] total number of samples in the output
                  
    bank_nchan    [scalar] number of channels in each filter bank. bank_nchan
                  must be less than nchan

    rise_frac_half_max
                  [scalar] fraction of bank_nchan in which the function rises 
                  from half-maximum to maximum value

    fall_frac_half_max
                  [scalar] fraction of bank_nchan in which the function falls 
                  from maximum to half-maximum value

    log_fall_rise [boolean] flag indicating if the rise and fall between maximum
                  and half-maximum values occur in logarithmic or linear
                  intervals. if True, log scale is used (default), otherwise a 
                  linear scale is used.

    verbose       [boolean] If True (default), prints progress and diagnostic 
                  messages. If False, suppress printing such messages.

    Output:

    window        [numpy vector] Frequency window function of size nchan.
    -----------------------------------------------------------------------------
    """

    try:
        nchan, bank_nchan, rise_frac_half_max, fall_frac_half_max
    except NameError:
        raise NameError('nchan, bank_nchan, rise_frac_half_max, fall_frac_half_max must be specified.')

    if not isinstance(nchan, (int, float)):
        raise TypeError('nchan must be a scalar integer.')

    if not isinstance(bank_nchan, (int, float)):
        raise TypeError('bank_nchan must be a scalar integer.')

    if not isinstance(rise_frac_half_max, (int, float)):
        raise TypeError('rise_frac_half_max must be a scalar value.')

    if not isinstance(fall_frac_half_max, (int, float)):
        raise TypeError('fall_frac_half_max must be a scalar value.')

    if nchan <= 0:
        raise ValueError('nchan must be a positive integer.')

    if bank_nchan <= 0:
        raise ValueError('bank_nchan must be a positive integer.')

    if (rise_frac_half_max < 0.0) or (rise_frac_half_max > 1.0):
        raise ValueError('rise_frac_half_max must lie in the range 0 and 1.')

    if (fall_frac_half_max < 0.0) or (fall_frac_half_max > 1.0):
        raise ValueError('fall_frac_half_max must lie in the range 0 and 1.')

    if bank_nchan > nchan:
        raise TypeError('Number of channels, bank_nchan, in the filter bank must be less than the total number of channels, nchan.')

    nfall = int(rise_frac_half_max * bank_nchan)
    nrise = int(fall_frac_half_max * bank_nchan)

    bank_window = NP.ones(bank_nchan)
    if log_fall_rise:
        fall = 10.0 ** NP.linspace(NP.log10(1.0), NP.log10(0.5), nfall)
        rise = 10.0 ** NP.linspace(NP.log10(0.5), NP.log10(1.0), nrise)
    else:
        fall = NP.linspace(1.0, 0.5, nfall)
        rise = NP.linspace(0.5, 1.0, nrise)
    bank_window[:nrise] = rise
    bank_window[bank_nchan-nfall:] = fall

    window = NP.repeat(bank_window.reshape(1,-1), NP.ceil(1.0*nchan/bank_nchan), axis=0)
    window = window.ravel()
    window = window[:nchan]
    
    return window
    
#################################################################################  

def PFB_FIR_coeff(nside, ntap, cutoff, nbits, window=('kaiser',5)):
    b = signal.firwin(nside*ntap, cutoff, window=window)
    b = (1-2**(-(nbits-1))) * b/b.max()
    return NP.round(b * 2**(nbits-1))

#################################################################################  

def apply_PFB_filter(inp, nside, ntap, coeff=None, cutoff=None, nbits=None,
                     window=('kaiser',5)):
    if coeff is None:
        if cutoff is None:
            cutoff = 1.0/nside
        if nbits is None:
            nbits = 12
        coeff = PFB_FIR_coeff(nside, ntap, cutoff, nbits, window=window)
    coeff = coeff/8192/32 # Normalize the amplitude
    arr = (inp[:nside*ntap]*coeff).reshape(ntap, nside)
    wsum = NP.sum(arr, axis=0)
    wf = NP.fft.ifft(wsum) * wsum.size
    return wf[:nside/2]

#################################################################################  

def PFB_shape(coeffs):
    pass

#################################################################################  

def sinewave(freq, tlen=8192, clock_freq=655.36):
    dt = 1./clock_freq
    t = NP.arange(tlen) * dt
    return NP.sin(2*NP.pi*freq.reshape(-1,1)*t.reshape(1,-1))

#################################################################################  

def pfbshape(freq):
    coeff = PFB_FIR_coeff(512, 8, 1.0/512, 12, window=('kaiser',5))
    twaves = sinewave(freq, tlen=4096)
    filter_twaves = NP.empty((freq.size, 512/2))
    for i in range(freq.size):
        filter_twaves[i,:] = apply_PFB_filter(twaves[i,:], 512, 8, coeff=coeff)
    return NP.abs(filter_twaves)**2
    
#################################################################################  

def fft_filter(inp, axis=None, wts=None, width=None, passband='low', verbose=True):    
    
    """
    -----------------------------------------------------------------------------
    Filter the input data using a low or high pass filter in frequency domain 
    along an axis

    Inputs:

    inp         [Numpy array] input data which has to be filtered across a given 
                axis 

    Keyword Inputs:

    axis        Axis (scalar integer) over which FFT is performed. Default = None
                (last axis). Any negative value or values exceeding the number of
                axes in the input data will be reset to use the last axis.

    wts         [Numpy array] frequency window of weights. Should either have 
                same shape as inp or have number of elements equal to the number
                of elements in input data along specified axis. Default = None.
                If not set, then it will be set to a rectangular window of width
                specified in width (see below) and will be applied as a filter
                identically to the entire data along the specified axis. The wts
                should correspond to frequency domain wts that would have been 
                obtained after fftshift. For instance, a low pass filter weights
                should be dominant around the middle of the sequence while a high
                pass filter weights would be concentrated at the beginning and 
                end of the sequence.

    width       [scalar] Width of the frequency window as a fraction of the 
                bandwidth. Has to be positive. Default is None. If width is None,
                wts should be set. One and only one among wts and width should be 
                set. 

    passband    [string scalar] String specifying the passband ('low' or 'high')
                to be used. Default = 'low'

    verbose     [boolean] If set to True (default), print messages indicating
                progress

    Output:
    
    Filtered output with same shape as inp
    -----------------------------------------------------------------------------
    """

    if verbose:
        print 'Entering fft_filtering()...'
        print '\tChecking inputs for compatibility...'

    try:
        inp
    except NameError:
        raise NameError('No input specified for filtering.')

    if isinstance(inp, list):
        inp = NP.asarray(inp)
    elif not isinstance(inp, NP.ndarray):
        raise TypeError('Input should be of type list or numpy array.')

    if (passband != 'low') and (passband != 'high'):
        raise ValueError('Invalid passband specified. Valid passbands are low or high.')

    if (wts is None) and (width is None):
        raise NameError('Neither frequency weights nor filter width specified.')

    if axis is None:
        axis = len(inp.shape) - 1
    elif not isinstance(axis, int):
        raise TypeError('axis must be an integer')
    else:
        if (axis < 0) or (axis >= len(inp.shape)):
            axis = len(inp.shape) - 1
            if verbose:
                print '\tSetting axis to be the last dimension of data'

    if wts is None:
        if not isinstance(width, (int,float)):
            raise TypeError('Filter width should be a scalar.')

        if verbose:
            print '\tFrequency weights not provided. Using filter width instead.'

        if width <= 0.0:
            raise ValueError('Filter width should be positive.')
        elif width >= 1.0:
            if verbose:
                print '\tFilter width exceeds 1.0. Returning input without filtering.'
            return inp

        filter_width = inp.shape[axis] * width

        # Even samples in input or low passband, keep the filter width odd
        # Odd samples in input and high passband, keep the filter width even
        # to have no imaginary parts after filtering

        shape = NP.asarray(inp.shape)
        shape[NP.arange(len(shape)) != axis] = 1
        shape = tuple(shape)
        if (inp.shape[axis] % 2 == 0) or (passband == 'low'): 
            if NP.floor(filter_width) % 2 == 0:            
                filter_width = NP.floor(filter_width) + 1
                if filter_width > inp.shape[axis]:
                    filter_width = inp.shape[axis]
            else:
                filter_width = NP.floor(filter_width)
            
            if verbose:
                print '\tAdjusting filter width to integer and to have FFT symmetry properties.'
            
            wts = NP.ones(filter_width) # Simple rectangular filter
            if verbose:
                print '\tFrequency weights have been set to be a rectangular filter of unit height'

            pads = shape[axis] - filter_width
            if pads > 0:
                wts = NP.pad(wts, (0,pads), mode='constant', constant_values=(0,))

            wts = wts.reshape(shape)

            if passband == 'low':
                wts = NP.roll(wts, -int(0.5*filter_width), axis=axis)
            else:
                wts = NP.fft.fftshift(NP.roll(wts, -int(0.5*filter_width), axis=axis), axes=axis)
        else:
            if NP.floor(filter_width) % 2 != 0:
                filter_width = NP.floor(filter_width) + 1
                if filter_width > inp.shape[1]:
                    filter_width = inp.shape[1]
            else:
                filter_width = NP.floor(filter_width)

            wts = NP.ones(filter_width) # Simple rectangular filter
            pads = shape[axis] - filter_width
            if pads > 0:
                wts = NP.pad(wts, (0,pads), mode='constant', constant_values=(0,))
            wts = wts.reshape(shape)

            wts = NP.fft.fftshift(NP.roll(wts, -int(filter_width/2 - 1), axis=axis), axes=axis)
            
    else:
        if isinstance(wts, list):
            wts = NP.asarray(list)
        elif not isinstance(wts, NP.ndarray):
            raise TypeError('Frequency weights should be a numpy array.')

        shape = NP.asarray(inp.shape)
        shape[NP.arange(len(shape)) != axis] = 1
        if wts.size == inp.shape[axis]:
            wts = wts.reshape(tuple(shape))
        elif wts.shape != inp.shape:
            raise IndexError('Dimensions of frequency weights do not match dimensions of input.')

        wshape = wts.shape

        if passband == 'low':
            wts = NP.roll(wts, -int(0.5*wshape[axis]), axis=axis)
        else:
            wts = NP.fft.fftshift(NP.roll(wts, -int(0.5*wshape[axis]), axis=axis), axes=axis)        

        # wts = wts/wts.ravel()[0] # Scale the weights to have zeroth frequency to have weight of unity

    out = NP.fft.ifft(NP.fft.fft(inp, axis=axis) * wts, axis=axis)
    out = out.reshape(inp.shape)

    if verbose:
        print '\tInput data successfully filtered in the frequency domain.'

    return out
        
#################################################################################  

def discretize(inp, nbits=None, nlevels=None, inprange=None, mode='floor', 
               discrete_out=True, verbose=True):

    """
    -----------------------------------------------------------------------------
    Discretize the input sequence either through truncation or rounding to the
    nearest levels

    Inputs:

    inp         [numpy array] Input sequence as a vector or array, could also be
                complex

    nbits       [scalar integer] Number of bits. Must be positive. Number of 
                levels will be 2**nbits. This takes precedence over nlevels if
                both are specified.

    nlevels     [scalar integer] Number of levels. Must be positive. Will be used
                only when nbits is not set.

    inprange    [2-element list] Consists of min and max bounds for the data. The
                data will be clipped outside this range. If set to None
                (default), min and max of the data will be used

    mode        [string] determines if the nearest neighbour is determined by 
                truncation to the next lower level or by round to the nearest 
                level. mode can be set to 'floor' or 'truncate' for truncation.
                It must be set to 'round' or 'nearest' for rounding to the 
                nearest level. Default = None (applies truncation).

    discrete_out
                [boolean] If set to True, return the output in discrete levels. 
                If False, output is scaled back to the input scale. Default is
                True.

    verbose     [boolean] If set to True, print progress and diagnostic messages

    Outputs:

    inpmod      [numpy array] Discretized array same shape as input array. If 
                discrete_out is set to True, inpmod is given in discrete 
                integer levels. If discrete_out is False, inpmod is given in
                the scale of the input corresponding to discrete integer levels.
                If input is complex, real and imaginary parts are separately 
                discretized.

    inpmin      [scalar] Lower bound on the data range used in discretization

    interval    [scalar] Resolution between levels

    Tuple consisting of discretized sequence (inpmod), minimum value determined
    from original sequence or specified range (inpmin), and resolution between
    levels (interval). If discrete_out is True, the rescaled output is given by 
    inpmin + interval * inpmod when the input sequence is real and 
    inpmin * (1+1j) + interval * inpmod when it is complex. When discrete_out is
    False, the discretized values scaled to integer values are given by 
    (inpmod - inpmin*(1+1j))/interval when the input is complex and 
    (inpmod - inpmin)/interval when it is real.

    -----------------------------------------------------------------------------
    """

    try:
        inp
    except NameError:
        raise NameError('inp must be provided')

    if isinstance(inp, list):
        inp = NP.asarray(inp)
    elif not isinstance(inp, NP.ndarray):
        raise TypeError('inp must be a list or numpy array')

    if (nbits is None) and (nlevels is None):
        if verbose:
            print '\tnbits or nlevels must be specified for discretization. Returning input unmodified'
        return inp
    elif (nbits is not None) and (nlevels is not None):
        nlevels = None

    if nlevels is not None:
        if not isinstance(nlevels, (int,float)):
            raise TypeError('nlevels must be an integer')
        nlevels = int(nlevels)
        if nlevels <= 1:
            raise ValueError('nlevels must be greater than 1')

    if nbits is not None:
        if not isinstance(nbits, (int,float)):
            raise TypeError('nbits must be an integer')
        nbits = int(nbits)
        if nbits <= 0:
            raise ValueError('nbits must be greater than 0')
        nlevels = 2**nbits

    if inprange is None:
        if NP.iscomplexobj(inp):
            inpmin = min(inp.real.min(), inp.imag.min())
            inpmax = max(inp.real.max(), inp.imag.max())
        else:
            inpmin = inp.min()
            inpmax = inp.max()
    else:
        inprange = NP.asarray(inprange).ravel()
        inpmin = inprange[0]
        inpmax = inprange[1]

    interval = (inpmax - inpmin)/nlevels
    levels = NP.arange(nlevels, dtype=NP.float)

    if NP.iscomplexobj(inp):
        scaled_inp = (inp - inpmin*(1+1j))/interval
    else:
        scaled_inp = (inp - inpmin)/interval

    if (mode == 'floor') or (mode == 'truncate') or (mode is None):
        if NP.iscomplexobj(scaled_inp):
            inpmod = NP.clip(NP.floor(scaled_inp.real), levels.min(), levels.max()) + 1j * NP.clip(NP.floor(scaled_inp.imag), levels.min(), levels.max())
        else:
            inpmod = NP.clip(NP.floor(scaled_inp), levels.min(), levels.max())
    elif (mode == 'round') or (mode == 'nearest'):
        inpshape = scaled_inp.shape
        scaled_inp = scaled_inp.reshape(-1,1)
        if NP.iscomplexobj(scaled_inp):
            inpind, distNN, refind = LKP.find_1NN(levels.reshape(-1,1), scaled_inp.real)
            inpmod = levels[refind].astype(NP.complex64)
            inpind, distNN, refind = LKP.find_1NN(levels.reshape(-1,1), scaled_inp.imag)
            inpmod += 1j * levels[refind]
        else:
            inpind, distNN, refind = LKP.find_1NN(levels.reshape(-1,1), scaled_inp)
            inpmod = levels[refind]
        inpmod = inpmod.reshape(inpshape)
            
    else:
        raise ValueError('Invalid mode specified for discretization.')
            
    if not discrete_out:
        inpmod *= interval

    return inpmod, inpmin, interval

#################################################################################

