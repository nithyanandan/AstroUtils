from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import zip
from builtins import range
import numpy as NP
import scipy as SP
from scipy import interpolate

#################################################################################

def grid(rangelist, pad=None, spacing=None, pow2=True, verbose=True):

    """
    -----------------------------------------------------------------------------
    Produce a multi-dimensional grid.
    
    Inputs:
    rangelist [list of tuples] Each tuple is made of two elements, the min and
              max with min < max. One tuple for each axis.
             
    pad       [Optional. Scalar or list] The padding (in same units as range) to
              be applied along the axes. Default=None implies no padding.
              
    spacing   [Optional. Scalar or list] The spacing for the grid along each of
              the axes. If not supplied, a default of sqrt(max-min) is used. If a
              scalar is supplied, it applies for all axes. A list applies for
              each of the axes.
              
    pow2      [Optional, default=True] If set, the grid produced is a power of 2
              in all axes, which is useful for FFT.
              
    verbose   [Default=True]

    Outputs:

    tupleout  [List of tuples] A 4-element tuple for each axis. The elements in
              each tuple are min, max, lengths, and spacing (which could have
              been modified relative to input). The precise grid points can be
              generated by using numpy's linspace using min, max and lengths.
    -----------------------------------------------------------------------------
    """

    for item in rangelist:
        if item[0] >= item[1]:
            raise ValueError('Data ranges provided not compatible with min < max. Exiting from grid().')

    if pad is None:
        pad = NP.zeros(len(rangelist))
    elif isinstance(pad, (int,float)):
        pad = [pad]
    elif isinstance(pad, NP.ndarray):
        pad = pad.tolist()
    elif not isinstance(pad, list):
        raise TypeError('pad must be set to None, scalar, list or numpy array')
        
    if len(pad) == 1:
        pad = [pad] * len(rangelist)
    elif len(pad) > len(rangelist):
        pad = NP.asarray(pad[:len(rangelist)])
    elif (len(pad) > 1) and (len(pad) < len(rangelist)):
        if verbose is True:
            print('Insufficient paddings provided compared to the number of data ranges.')
            print('Assuming the remaining paddings to be zero.')
        pad += [0.0 for ranges in rangelist[len(pad):]]
    # pad = NP.reshape(NP.asarray(pad).reshape(1,-1),len(pad)) # Force it to be row vector
    pad = NP.asarray(pad).flatten()
    pad.clip(min(NP.abs(pad))) # Remove any negative values for pad

    if spacing is None:
        if verbose is True:
            print('No spacing provided. Setting defaults to sqrt(max-min)')
            print('Final spacings could be different from defaults assumed.')
        spacing = [NP.sqrt(rangelist[i][1]-rangelist[i][0]+2*pad[i]) for i in range(len(rangelist))]
    elif isinstance(spacing, (int, float)):
        if verbose is True:
            print('Scalar value for spacing provided. Assuming spacing is identical along all axes.')
        spacing = [spacing] * len(rangelist)
    elif len(spacing) > len(rangelist):
        if verbose is True:
            print('Too many values of spacing provided. Ignoring values for indices beyond the length of data ranges.')
        spacing = NP.asarray(spacing[:len(rangelist)])
    elif (len(spacing) > 1) and (len(spacing) < len(rangelist)):
        if verbose is True:
            print('Insufficient spacings provided compared to the number of data ranges.')
            print('Assuming the remaining spacings to be default spacing.')
            print('Final spacings could be different from defaults assumed.')
        # spacing += [NP.sqrt(ranges[1]-ranges[0]) for ranges in rangelist[len(spacing):]]
        spacing += [NP.sqrt(rangelist[i][1]-rangelist[i][0]+2*pad[i]) for i in range(len(spacing),len(rangelist))]
    # spacing = NP.asarray(spacing).reshape(1,-1) 
    spacing = NP.asarray(spacing).flatten()
    spacing.clip(min(NP.abs(spacing)))

    rangelist = NP.asarray(rangelist)
    lengths = NP.ceil((rangelist[:,1]-rangelist[:,0]+2*pad)/spacing)+1
    lengths = lengths.astype(int)

    for i in range(len(lengths)): 
        if (lengths[i] % 2) == 0: lengths[i] += 1
        # make it odd number of bin edges enclsoing first
        # and last intervals so that the mid-point is one
        # of the bin edges.

    if pow2 is True:
        lengths = 2**NP.ceil(NP.log2(lengths))
        lengths = lengths.astype(int)
        newspacing = (rangelist[:,1]-rangelist[:,0]+2*pad)/(lengths-2)
        offsets = rangelist[:,0]-pad+(lengths-2)*newspacing - (rangelist[:,1]+pad)
        tupleout = list(zip(rangelist[:,0]-pad-0.5*offsets-newspacing, rangelist[:,1]+pad+0.5*offsets, lengths, newspacing)) # converts numpy arrays into a list of tuples
        # tupleout = tuple(map(tuple, NP.column_stack((rangelist[:,0]-pad-0.5*offsets-newspacing, rangelist[:,1]+pad+0.5*offsets, lengths, newspacing)))) # converts numpy arrays into a list of tuples
    else:
        offsets = rangelist[:,0]-pad+(lengths-1)*spacing - (rangelist[:,1]+pad)
        tupleout = list(zip(rangelist[:,0]-pad-0.5*offsets, rangelist[:,1]+pad+0.5*offsets, lengths, spacing)) # converts numpy arrays into a list of tuples
        # tupleout = tuple(map(tuple, NP.column_stack((rangelist[:,0]-pad-0.5*offsets, rangelist[:,1]+pad+0.5*offsets, lengths, spacing)))) # converts numpy arrays into a list of tuples
    
    return tupleout

#################################################################################

def grid_1d(ranges, pad=None, spacing=None, pow2=True, verbose=True):

    """
    -----------------------------------------------------------------------------
    1D wrapper for grid()
    
    Inputs:
    ranges   [2-element list or a tuple with 2 elements] min and max with
             min < max.
             
    pad      [Optional or Scalar] The padding (in same units as ranges) to be
             applied along the axis. Default=None implies no padding.

    spacing  [Optional or Scalar] The spacing for the grid along the axis. If not
             supplied, a default of sqrt(max-min) is used. 

    pow2     [Optional, default=True] If set, the grid produced is a power of 2
             in its axis, which is useful for FFT.

    verbose  [Default=True]

    Outputs:

    A numpy array with x-values on the grid.
    -----------------------------------------------------------------------------
    """

    if ranges is None:
        raise NameError('No range provided. Exiting from grid_1d()')
    if not isinstance(ranges, (list,tuple)):
        raise TypeError('A 2-element list or tuple containing range with min < max should be provided.')

    if isinstance(ranges, tuple):
        grid_info = grid([ranges], pad=[pad], spacing=[spacing], pow2=pow2, verbose=verbose)
    if isinstance(ranges, list):
        grid_info = grid([tuple(ranges)], pad=[pad], spacing=[spacing], pow2=pow2, verbose=verbose)
    return NP.linspace(grid_info[0][0], grid_info[0][1], int(grid_info[0][2]))

#################################################################################

def grid_2d(rangelist, pad=None, spacing=None, pow2=True, verbose=True):

    """
    -----------------------------------------------------------------------------
    2D wrapper for grid()
    
    Inputs:
    rangelist   
             [2-element list of tuples] Each tuple is made of two elements, the
             min and max with min < max.
             
    pad      [Optional. Scalar or list] The padding (in same units as rangelist) to
             be applied along the two axes. Default=None implies no padding.

    spacing  [Optional. Scalar or list] The spacing for the grid along each of
             the axes. If not supplied, a default of sqrt(max-min) is used. If a
             scalar is supplied, it applies for all axes. A list applies for each
             of the axes.

    pow2     [Optional, default=True] If set, the grid produced is a power of 2
             in all axes, which is useful for FFT.

    verbose  [Default=True]

    Outputs:

    Two 2D numpy arrays. The first array with x-values on the grid, and the
    second with y-values on the grid.
    -----------------------------------------------------------------------------
    """

    if rangelist is None:
        raise NameError('No ranges provided. Exiting from grid_2d()')
    if not isinstance(rangelist, list):
        raise TypeError('A 2-element list of tuples specifying ranges with min < max should be provided. Exiting from grid_2d()')
    else:
        if not isinstance(rangelist[0], tuple):
            raise TypeError('A 2-element list of tuples specifying ranges with min < max should be provided. Exiting from grid_2d()')

    grid_info = grid(rangelist, pad=pad, spacing=spacing, pow2=pow2, verbose=verbose)
    return NP.meshgrid(NP.linspace(grid_info[0][0], grid_info[0][1], int(grid_info[0][2])), NP.linspace(grid_info[1][0], grid_info[1][1], int(grid_info[1][2])))

#################################################################################

def grid_3d(rangelist, pad=None, spacing=None, pow2=True, verbose=True):

    """
    -----------------------------------------------------------------------------
    3D wrapper for grid()
    
    Inputs:
    rangelist   
             [3-element list of tuples] Each tuple is made of two elements, the
             min and max with min < max.
             
    pad      [Optional. Scalar or list] The padding (in same units as rangelist) to
             be applied along the two axes. Default=None implies no padding.

    spacing  [Optional. Scalar or list] The spacing for the grid along each of
             the axes. If not supplied, a default of sqrt(max-min) is used. If a
             scalar is supplied, it applies for all axes. A list applies for each
             of the axes.

    pow2     [Optional, default=True] If set, the grid produced is a power of 2
             in all axes, which is useful for FFT.

    verbose  [Default=True]

    Outputs:

    Three 3D numpy arrays. The first array with x-values, the second with 
    y-values, and the third with z-values on the grid.
    -----------------------------------------------------------------------------
    """

    if rangelist is None:
        raise NameError('No ranges provided. Exiting from grid_3d()')
    if not isinstance(rangelist, list):
        raise TypeError('A 3-element list of tuples specifying ranges with min < max should be provided. Exiting from grid_2d()')
    else:
        if not isinstance(rangelist[0], tuple):
            raise TypeError('A 3-element list of tuples specifying ranges with min < max should be provided. Exiting from grid_3d()')

    grid_info = grid(rangelist, pad=pad, spacing=spacing, pow2=pow2, verbose=verbose)
    return NP.meshgrid(NP.linspace(grid_info[0][0], grid_info[0][1], int(grid_info[0][2])), NP.linspace(grid_info[1][0], grid_info[1][1], int(grid_info[1][2])), NP.linspace(grid_info[2][0], grid_info[2][1], int(grid_info[2][2])))

#################################################################################

def conv_grid1d(xc, xkern, kernel, xgrid, method='NN'):
    """
    -----------------------------------------------------------------------------
    Perform 1D gridding convolution.
    Inputs:

    xc:      [vector as list or numpy array] x-coordinates of center of gridding
             function 

    xkern:   [vector as list or numpy array] x-coordinates of gridding function

    kernel:  [vector as list or numpy array] gridding function (or kernel). If
             kernel is a scalar, a nearest neighbour interpolation is used
             overriding the method requested for. This kernel will be
             interpolated at the requested locations in xgrid

    xgrid:   [vector as list or numpy array] x-coordinates of grid locations at
             which interpolates are required

    Keyword Inputs:

    method:  String indicating interpolation method. [Default = 'NN']
             'NN' => Nearest Neighbour 
             'SL' => Single linear
             'CS' => Cubic Spline

    Output:

    outdata: [vector] Gridded values at values of xgrid

    -----------------------------------------------------------------------------
    """
    try:
        xc
    except NameError:
        raise NameError("Argument 'xc' not defined. Aborting conv_grid1d().")

    try:
        xkern
    except NameError:
        raise NameError("Argument 'xkern' not defined. Aborting conv_grid1d().")

    try:
        xgrid
    except NameError:
        raise NameError("Argument 'xgrid' not defined. Aborting conv_grid1d().")

    try:
        method
    except NameError:
        method='NN'

    if (method != 'SL') and (method != 'CS'):
        method = 'NN'

    try:
        kernel
    except NameError:
        print("Argument 'kernel' not defined. ")
        if method == 'NN':
            print("Since method is Nearest Neighbor interpolation, ")
            print("proceeding with kernel=1. ")
            kernel = 1.0
        else:
            raise ValueError("Aborting conv_grid1d().")

    if not isinstance(kernel, (list, NP.ndarray)):
        print('Kernel seems to be a scalar. Proceeding with Nearest Neighbour \n method of interpolation.')
        method = 'NN'

    if isinstance(xc, (int, float)):
        xc = NP.asarray(xc).reshape(1)

    if isinstance(xc, list):
        xc = NP.asarray(xc)
    elif isinstance(xc, NP.ndarray):
        xc = xc.flatten()

    if isinstance(xkern, (int, float)):
        xkern = NP.asarray(xkern).reshape(1)

    if isinstance(xkern, list):
        xkern = NP.asarray(xkern)
    elif isinstance(xkern, NP.ndarray):
        xkern = xkern.flatten()

    if isinstance(kernel, (int, float)):
        kernel = NP.asarray(kernel).reshape(1)

    if isinstance(kernel, list):
        kernel = NP.asarray(kernel)
    elif isinstance(kernel, NP.ndarray):
        kernel = kernel.flatten()

    kernel_real = NP.real(kernel)
    kernel_imag = 0.0
    if NP.iscomplexobj(kernel):
        kernel_imag = NP.imag(kernel)

    if xkern.shape != kernel.shape:
        raise ValueError(' Incompatible kernel coordinates. Verify their lengths are equal.\n Aborting conv_grid1d().')

    if isinstance(xgrid, (int, float)):
        xgrid = NP.asarray(xgrid).reshape(1)

    if isinstance(xgrid, list):
        xgrid = NP.asarray(xgrid)
    elif isinstance(xgrid, NP.ndarray):
        xgrid = xgrid.flatten()

    if NP.iscomplexobj(kernel):
        outdata = NP.zeros(len(xgrid), dtype=NP.complex_)
    else:
        outdata = NP.zeros(len(xgrid))

    xckern = 0.5*(max(xkern)+min(xkern))
    xshift = xc - xckern
    
    for npoints in range(0,len(xc)):
        xkern_shifted = xkern + NP.repeat(xshift[npoints],xkern.shape[0])
        interp_func_imag = None
        if method == 'SL':
            interp_func_real = interpolate.interp1d(xkern_shifted, kernel_real, kind='slinear', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                interp_func_imag = interpolate.interp1d(xkern_shifted, kernel_imag, kind='slinear', fill_value=0.0)
        elif method == 'CS':
            interp_func_real = interpolate.interp1d(xkern_shifted, kernel_real, kind='cubic', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                interp_func_imag = interpolate.interp1d(xkern_shifted, kernel_imag, kind='cubic', fill_value=0.0)
        else:
            interp_func_real = interpolate.interp1d(xkern_shifted, kernel_real, kind='nearest', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                interp_func_imag = interpolate.interp1d(xkern_shifted, kernel_imag, kind='nearest', fill_value=0.0)
        outdata += interp_func_real(xgrid)
        if NP.iscomplexobj(kernel):
            outdata += 1j * interp_func_imag(xgrid)

    return outdata
        
#################################################################################

def conv_grid2d(xc, yc, xkern, ykern, kernel, xgrid, ygrid, method='NN'):
    """
    -----------------------------------------------------------------------------
    Perform gridding convolution.
    Inputs:

    xc:      [vector as list of numpy array] x-coordinates of center of gridding
             function 

    yc:      [vector as list or numpy array] y-coordinates of center of gridding
             function 

    xkern:   [vector as list or numpy array] x-coordinates of gridding function

    ykern:   [vector as list or numpy array] y-coordinates of gridding function

    kernel:  [vector as list or numpy array] gridding function (or kernel). If
             kernel is a scalar, a nearest neighbour interpolation is used
             overriding the method requested for. This kernel will be
             interpolated at the requested locations in (xgrid, ygrid)

    xgrid:   [vector as list or numpy array] x-coordinates of grid locations at
             which interpolates are required

    ygrid:   [vector as list or numpy array] y-coordinates of grid locations at
             which interpolates are required
    
    Keyword Inputs:

    method:  String indicating interpolation method. [Default = 'NN']
             'NN' => Nearest Neighbour 
             'BL' => Bilinear
             'CS' => Cubic Spline

    Output:

    outdata: [vector] Gridded values at values of (xgrid, ygrid) 

    ----------------------------------------------------------------------
    """
    try:
        xc
    except NameError:
        raise NameError("Argument 'xc' not defined. Aborting conv_grid2d().")

    try:
        yc
    except NameError:
        raise NameError("Argument 'yc' not defined. Aborting conv_grid().")

    try:
        xkern
    except NameError:
        raise NameError("Argument 'xkern' not defined. Aborting conv_grid2d().")

    try:
        ykern
    except NameError:
        raise NameError("Argument 'ykern' not defined. Aborting conv_grid2d().")

    try:
        xgrid
    except NameError:
        raise NameError("Argument 'xgrid' not defined. Aborting conv_grid2d().")

    try:
        ygrid
    except NameError:
        raise NameError("Argument 'ygrid' not defined. Aborting conv_grid2d().")

    try:
        method
    except NameError:
        method='NN'

    if (method != 'BL') and (method != 'CS'):
        method = 'NN'

    try:
        kernel
    except NameError:
        print("Argument 'kernel' not defined. ")
        if method == 'NN':
            print("Since method is Nearest Neighbor interpolation, ")
            print("proceeding with kernel=1. ")
            kernel = 1.0
        else:
            raise ValueError("Aborting conv_grid2d().")

    if not isinstance(kernel, (list, NP.ndarray)):
        print("Kernel is a scalar. Proceeding with Nearest Neighbour")
        print("method of interpolation.")
        method = 'NN'

    if isinstance(xc, (int, float)):
        xc = NP.asarray(xc).reshape(1)
    if isinstance(yc, (int, float)):
        yc = NP.asarray(yc).reshape(1)

    xyc = None
    if isinstance(xc, list) and isinstance(yc, list):
        if (len(xc) != len(yc)):
            raise ValueError(" Incompatible input location coordinates.\n Verify their lengths are equal. Aborting conv_grid2d().")
        else:
            xyc = NP.hstack((NP.asarray(xc).reshape(-1,1), NP.asarray(yc).reshape(-1,1)))
    elif isinstance(xc, NP.ndarray) and isinstance(yc, NP.ndarray):
        if xc.shape != yc.shape:
            raise ValueError('Incompatible grid lattice coordinates. Verify their shapes are equal.\n Aborting conv_grid2d().')
        else:
            xyc = NP.hstack((xc.reshape(-1,1), yc.reshape(-1,1)))
    else:
        raise TypeError('xc and yc should be of identical data types. Allowed types are lists and numpy arrays.')

    if xyc is None:
        raise NameError('** Diagnostic Message ** xyc, an internal variable in conv_grid2d() should have been created by now. Data type incompatibility in xc and/or yc has to be resolved.')

    xykern = None
    if isinstance(xkern, list) and isinstance(ykern, list):
        if (len(xkern) != len(ykern)):
            raise IndexError(" Incompatible grid lattice coordinates. Verify their lengths are equal.\n Aborting conv_grid2d().")
        else:
            xykern = NP.hstack((NP.asarray(xkern).reshape(-1,1), NP.asarray(ykern).reshape(-1,1)))
    elif isinstance(xkern, NP.ndarray) and isinstance(ykern, NP.ndarray):
        if xkern.shape != ykern.shape:
            raise IndexError('Incompatible grid lattice coordinates. Verify their shapes are equal.\n Aborting conv_grid2d().')
        else:
            xykern = NP.hstack((xkern.reshape(-1,1), ykern.reshape(-1,1)))
    else:
        raise TypeError('xkern and ykern should be of identical data types. Allowed types are lists and numpy arrays.')

    if xykern is None:
        raise NameError('** Diagnostic Message ** xykern, an internal variable in conv_grid2d() should have been created by now. Data type incompatibility in xkern and/or ykern has to be resolved.')

    if isinstance(kernel, list):
        if len(kernel) != xykern.shape[0]:
            raise IndexError(" Incompatible grid lattice coordinates. Verify their lengths are equal.\n Aborting conv_grid2d().")
        else:
            kernel = NP.asarray(kernel).reshape(-1,1)
    elif isinstance(kernel, NP.ndarray):
        kernel = kernel.reshape(-1,1)
        if xykern.shape[0] != kernel.shape[0]:
            raise IndexError('Incompatible grid lattice coordinates. Verify their shapes are equal.\n Aborting conv_grid2d().')
    else:
        raise TypeError('kernel data type should be a list or numpy array.')

    kernel_real = NP.real(kernel)
    kernel_imag = 0.0
    if NP.iscomplexobj(kernel):
        kernel_imag = NP.imag(kernel)

    xygrid = None
    if isinstance(xgrid, (int, float)):
        xgrid = NP.asarray(xgrid).reshape(1)
    if isinstance(ygrid, (int, float)):
        ygrid = NP.asarray(ygrid).reshape(1)

    if isinstance(xgrid, list) and isinstance(ygrid, list):
        if (len(xgrid) != len(ygrid)):
            raise IndexError(" Incompatible grid lattice coordinates. Verify their lengths are equal.\n Aborting conv_grid2d().")
        else:
            xygrid = NP.hstack((NP.asarray(xgrid).reshape(-1,1), NP.asarray(ygrid).reshape(-1,1)))
    elif isinstance(xgrid, NP.ndarray) and isinstance(ygrid, NP.ndarray):
        if xgrid.shape != ygrid.shape:
            raise IndexError('Incompatible grid lattice coordinates. Verify their shapes are equal.\n Aborting conv_grid2d().')
        else:
            xygrid = NP.hstack((xgrid.reshape(-1,1), ygrid.reshape(-1,1)))
    else:
        raise TypeError('xgrid and ygrid should be of identical data types. Allowed types are lists and numpy arrays.')

    if xygrid is None:
        raise NameError('** Diagnostic Message ** xygrid, an internal variable in conv_grid2d() should have been created by now. Data type incompatibility in xgrid and/or ygrid has to be resolved.')

    if NP.iscomplexobj(kernel):
        outdata = NP.zeros((xygrid.shape[0],1), dtype=NP.complex_)
    else:
        outdata = NP.zeros((xygrid.shape[0],1))

    xckern = 0.5*(NP.amax(xkern)+NP.amin(xkern))
    yckern = 0.5*(NP.amax(ykern)+NP.amin(ykern))
    xshift = xc - xckern
    yshift = yc - yckern
    for npoints in range(len(xc)):
        xykern_shifted = xykern + NP.hstack((NP.repeat(xshift[npoints],xykern.shape[0]).reshape(xykern.shape[0],1), NP.repeat(yshift[npoints],xykern.shape[0]).reshape(xykern.shape[0],1)))
        if method == 'BL':
            outdata += interpolate.griddata(xykern_shifted, kernel_real, xygrid, method='linear', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                outdata += 1j * interpolate.griddata(xykern_shifted, kernel_imag, xygrid, method='linear', fill_value=0.0)
        elif method == 'CS':
            outdata += interpolate.griddata(xykern_shifted, kernel_real, xygrid, method='cubic', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                outdata += 1j * interpolate.griddata(xykern_shifted, kernel_imag, xygrid, method='cubic', fill_value=0.0)
        else:
            outdata += interpolate.griddata(xykern_shifted, kernel_real, xygrid, method='nearest', fill_value=0.0)
            if NP.iscomplexobj(kernel):
                outdata += 1j * interpolate.griddata(xykern_shifted, kernel_imag, xygrid, method='nearest', fill_value=0.0)

    return outdata.reshape(xgrid.shape)
        
#################################################################################

