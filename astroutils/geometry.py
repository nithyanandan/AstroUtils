from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import zip, range, object
import sys
import numpy as NP
import numpy.linalg as LA
import healpy as HP
import warnings
import ipdb as PDB 
try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

epsilon = sys.float_info.epsilon # typical floating-point calculation error

#################################################################################

class Point(object):

    """
    ----------------------------------------------------------------------------
    Class to manage 3-dimensional locations in a Cartesian coordinate system.

    Attributes:

    x       [Scalar] x-coordinate 

    y       [Scalar] y-coordinate

    z       [Scalar] z-coordinate

    Member functions:

    __init__():   Initializes an instance of class Point

    __str__():    Prints the coordinates stored in the instance

    __add__():    Operator overloading to add two vectors which are represented
                  as instances of class Point

    __sub__():    Operator overloading to subtract one vector from another
                  which are both represented as instances of class Point

    __mul__():    Operator overloading to multiply two vectors represented as 
                  instances of Point class and produce a dot product. 

    __rmul__():   Operator overlaoding to multiply a vector represented as 
                  instance of Point class with a scalar value. 

    ----------------------------------------------------------------------------
    """

    def __init__(self, xyz=None):

        """
        -------------------------------------------------------------------------
        Initializes an instance of class Point

        Inputs:

        xyz    [Tuple, list or scalar] If not provided, the instance of class
               Point is initialized to (0.0, 0.0, 0.0). If scalar is provided, it
               is assigned to the x value. The y and z values are both set to 0.0.
               If a list or tuple is provided, the first element is assigned to x,
               the second to y, and the third to z. If the second or third
               elements are not provided, a value 0.0 is assigned to the 
               corresponding coordinates. If more than three elements are 
               provided, only the first three values are used and the others 
               ignored. Default = None which initializes to (0.0, 0.0, 0.0).

        Class attributes initialized are:
        x, y, z

        Read docstring of class Point for details on these attributes.
        -------------------------------------------------------------------------
        """
        
        if xyz is None:
            xyz = [0.0, 0.0, 0.0]

        if isinstance(xyz, Point):
            self.x, self.y, self.z = xyz.x, xyz.y, xyz.z
        elif isinstance(xyz, (tuple,list)): # input is list or tuple
            if len(xyz) == 0:
                self.x, self.y, self.z = 0.0, 0.0, 0.0
            elif len(xyz) == 1:
                self.x, self.y, self.z = float(xyz[0]), 0.0, 0.0
            elif len(xyz) == 2:
                self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), 0.0
            else:
                self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        elif isinstance(xyz, NP.ndarray):
            xyz = xyz.flatten()
            if len(xyz) == 0:
                self.x, self.y, self.z = 0.0, 0.0, 0.0
            elif len(xyz) == 1:
                self.x, self.y, self.z = float(xyz[0]), 0.0, 0.0
            elif len(xyz) == 2:
                self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), 0.0
            else:
                self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])            
        elif isinstance(xyz, (int, float)): 
            self.x, self.y, self.z = float(xyz), 0.0, 0.0
        else:
            raise TypeError('Data type mismatch. Check input again.')

    #############################################################################

    def __str__(self):
        return '({0}, {1}, {2})'.format(self.x, self.y, self.z)

    #############################################################################

    def __add__(self, other):

        """
        -------------------------------------------------------------------------
        Operator overloading to add two vectors which are represented as
        instances of class Point

        Input(s):

        other     [instance of class Point] The instance of class Point which is
                  to be added to the current instance. 

        -------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            warnings.warn('No object provided for addition.')
            return self
        
        if not isinstance(other, Point):
            raise TypeError('Object type is incompatible.')

        return Point((self.x+other.x, self.y+other.y, self.z+other.z))
 
    #############################################################################

    def __sub__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to subtract one vector from another which are both
        represented as instances of class Point

        Input(s):

        other     [instance of class Point] The instance of class Point which is
                  to be subtracted from the current instance. 

        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            warnings.warn('No object provided for subtraction.')
            return self
        
        if not isinstance(other, Point):
            raise TypeError('Object type is incompatible.')

        return Point((self.x-other.x, self.y-other.y, self.z-other.z))

    #############################################################################

    def __mul__(self, other): # Dot product

        """
        ------------------------------------------------------------------------
        Operator overloading to multiply two vectors represented as instances of
        class Point and produce a dot product

        Input(s):

        other     [instance of class Point] The instance of class Point which is
                  to be multiplied with the current instance to yield a dot 
                  product. 

        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            warnings.warn('No object provided for product.')
            return self
        
        if not isinstance(other, Point):
            raise TypeError('Object type is incompatible.')

        return self.x*other.x + self.y*other.y + self.z*other.z

    #############################################################################

    def __rmul__(self, value): # scalar product

        """
        ------------------------------------------------------------------------
        Operator overloading to multiply a vector represented as instance of
        class Point with a scalar value

        Input(s):

        other     [scalar int or float] The scalar value which scales the
                  current instance to yield a scaled vector

        ------------------------------------------------------------------------
        """

        try:
            value
        except NameError:
            warnings.warn('No scalar value provided for product.')
            return self
        
        if not isinstance(value, (int,float)):
            raise TypeError('Value is incompatible for product.')

        return Point((value*self.x, value*self.y, value*self.z))

    #############################################################################
    
    def __abs__(self):

        """
        ------------------------------------------------------------------------
        Operator overloading to give the magnitude of the  vector represented as 
        instance of class Point

        Output(s): [float] Magnitude of the vector represented as an instance of 
                   class Point
        ------------------------------------------------------------------------
        """

        return NP.sqrt(self.__mul__(self))

    #############################################################################

    def __lt__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to check if the magnitude of the vector represented 
        as instance of class Point is lesser than that of another vector given 
        as instance of Point class

        Input(s):

        other     [scalar int or float] Instance of Point class with which the 
                  current instance will be compared to

        Output(s): [Boolean] True if magnitude of current class is lesser than 
                   that of the other vector to be compared to, False otherwise.
        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            raise NameError('No object provided for comparison.')

        if not isinstance(other, Point):
            raise TypeError('The object provided for comparison must be an instance of class Point.')

        return abs(self) < abs(other)

    #############################################################################

    def __le__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to check if the magnitude of the vector represented 
        as instance of class Point is lesser than or equal to that of another 
        vector given as instance of Point class

        Input(s):

        other     [scalar int or float] Instance of Point class with which the 
                  current instance will be compared to

        Output(s): [Boolean] True if magnitude of current class is lesser than 
                   or equal to that of the other vector to be compared to, False
                   otherwise.
        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            raise NameError('No object provided for comparison.')

        if not isinstance(other, Point):
            raise TypeError('The object provided for comparison must be an instance of class Point.')

        return abs(self) <= abs(other)

    #############################################################################

    def __gt__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to check if the magnitude of the vector represented 
        as instance of class Point is greater than that of another vector given 
        as instance of Point class

        Input(s):

        other     [scalar int or float] Instance of Point class with which the 
                  current instance will be compared to

        Output(s): [Boolean] True if magnitude of current class is greater than 
                   that of the other vector to be compared to, False otherwise.
        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            raise NameError('No object provided for comparison.')

        if not isinstance(other, Point):
            raise TypeError('The object provided for comparison must be an instance of class Point.')

        return abs(self) > abs(other)

    #############################################################################

    def __ge__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to check if the magnitude of the vector represented 
        as instance of class Point is greater than or equal to that of another 
        vector given as instance of Point class

        Input(s):

        other     [scalar int or float] Instance of Point class with which the 
                  current instance will be compared to

        Output(s): [Boolean] True if magnitude of current class is greater than 
                   or equal to that of the other vector to be compared to, False
                   otherwise.
        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            raise NameError('No object provided for comparison.')

        if not isinstance(other, Point):
            raise TypeError('The object provided for comparison must be an instance of class Point.')

        return abs(self) >= abs(other)

    #############################################################################

    def __eq__(self, other):

        """
        ------------------------------------------------------------------------
        Operator overloading to check if the magnitude of the vector represented 
        as instance of class Point is equal to that of another vector given as
        instance of Point class

        Input(s):

        other     [scalar int or float] Instance of Point class with which the 
                  current instance will be compared to

        Output(s): [Boolean] True if magnitude of current class is equal to that
                   of the other vector to be compared to, False otherwise.
        ------------------------------------------------------------------------
        """

        try:
            other
        except NameError:
            raise NameError('No object provided for comparison.')

        if not isinstance(other, Point):
            raise TypeError('The object provided for comparison must be an instance of class Point.')

        return abs(self) == abs(other)

#################################################################################

def points_from_line2d_intersection(coeffs, dvect, ravel=True):
    """
    ----------------------------------------------------------------------------
    Find pairwise intersections between a system of equations denoting lines in 
    the Cartesian plane. If N equations are provided then N(N-1)/2 intersection 
    points (from all pairwise line intersections) are determined and returned. 
    The equations must be represented by coeffs(dot)xyvect = dvect
    
    Inputs:
    
    coeffs  [numpy array] NxM numpy array denoting N measurements using (M=2) 
            parameters in (M=2)-dimensional space
    
    dvect   [numpy array] Numpy array of shape (N,1) or (N,) denoting the 
            measured values

    ravel   [boolean] If set to True (default), ravel the N(N-1)/2 points of
            intersection into a (N(N-1)/2, 2) numpy array
    
    Output:
    
    (N,N,M=2) array containing intersection between all pairs chosen from the N 
    lines. The main diagonal and upper diagonal is set to NaN since that denotes 
    lines intersecting with themselves or transposes of the lower diagonal 
    respectively. If no intersection is found between any pair of lines, they 
    are also denoted by NaN. If ravel is set to True, the N(N-1)/2 points of
    intersection are raveled into (N(N-1)/2,2) numpy array. The order of these
    points is in the following order of intersection of lines: 2--1, 3--2, 3--1, 
    4--3, 4--2, 4--1, 5--4, 5--3, 5--2, 5--1, ...,  N--(N-1), N--(N-2), ... N--1.
    ----------------------------------------------------------------------------
    """
    
    if not isinstance(coeffs, NP.ndarray):
        raise TypeError('Input coeffs must be a numpy array')
    if not isinstance(dvect, NP.ndarray):
        raise TypeError('Input dvect must be a numpy array')
    if coeffs.ndim != 2:
        raise ValueError('Input coeffs must be a 2D numpy array')
    if coeffs.shape[1] != 2:
        raise ValueError('Input coeffs must be of shape (N,2)')
    dvect = dvect.reshape(-1)
    if dvect.size != coeffs.shape[0]:
        raise ValueError('Size of input dvect must match the first dimension of coeffs')
    
    if dvect.dtype == NP.complexfloating:
        outarr = NP.empty((dvect.size, dvect.size, 2), dtype=dvect.dtype)
        outarr.fill(NP.nan + 1j*NP.nan)
    else:
        outarr = NP.empty((dvect.size, dvect.size, 2), dtype=float)
        outarr.fill(NP.nan)
    
    for i in range(dvect.size):
        for j in range(i):
            coeff_2x2 = NP.vstack((coeffs[i,:], coeffs[j,:]))
            d2 = dvect[[i,j]]
            try:
                outarr[i,j] = LA.solve(coeff_2x2, d2)
            except LA.LinAlgError as LAerr:
                print('Encountered {0}. System of equations ({1:0d}) and ({2:0d}) are ill-conditioned. Proceeding...'.format(LAerr, i, j))
                
    if not ravel:
        return outarr
    else:
        raveled_outarr = NP.asarray([outarr[i,j,:] for i in range(outarr.shape[0]) for j in range(i-1,-1,-1)])
        return raveled_outarr

#################################################################################

def generate_line_from_point_and_slope(points, slopes):
    """
    ----------------------------------------------------------------------------
    Find equation of a line given a point on the line and the slope. This 
    function can return N lines at a time given the corresponding N points and N 
    slopes. The returned line equations will be represented by 
    coeffs(dot)xyvect = dvect
    
    Inputs:
    
    points  [numpy array] NpxM numpy array denoting Np points in 
            (M=2)-dimensional space. If Np=1, determine line equations for this
            point with each of the Ns slopes. Otherwise Np=Ns.
    
    slopes  [numpy array] Numpy array of shape (Ns,) denoting the Ns slopes to be 
            associated with the Ns lines whose Np points are given above. If 
            Ns=1, determine line equations for this slope with each of the Np 
            points. Otherwise Np=Ns.
    
    Output:
    
    (N,M+1) array (an augmented matrix) where the (N,M=2) array corresponds to 
    the coefficients and the last column corresponds to the dvect. N=max(Np, Ns).
    ----------------------------------------------------------------------------
    """
  
    if not isinstance(points, NP.ndarray):
        raise TypeError('Input points must be a numpy array')
    if not isinstance(slopes, NP.ndarray):
        raise TypeError('Input slopes must be a numpy array')
    if points.ndim != 2:
        raise ValueError('Input points must be a 2D numpy array')
    if points.shape[1] != 2:
        raise ValueError('Input points must be of shape (N,2)')
    slopes = slopes.reshape(-1,1)
    if (slopes.shape[0] != points.shape[0]):
        if slopes.shape[0] == 1:
            slopes = slopes * NP.ones((points.shape[0],1))
        elif points.shape[0] == 1:
            points = points + NP.zeros((slopes.size,2))
        else:
            raise ValueError('Size of input slopes must match the number of points or they must be broadcastable')

    coeffs = NP.hstack((-slopes, NP.ones(slopes.shape)))
    dvect = points[:,1] - slopes.ravel() * points[:,0]
    
    ind_infinite = NP.isinf(slopes.ravel())
    coeffs[ind_infinite,0] = 1.0
    coeffs[ind_infinite,1] = 0.0
    dvect[ind_infinite] = points[ind_infinite,0]
    
    return NP.hstack((coeffs, dvect.reshape(-1,1)))

#################################################################################

def generate_line_from_two_points(points1, points2):
    """
    -----------------------------------------------------------------------------
    Find equation of a line given two points on the line. This function can 
    return N lines at a time given the corresponding N pairs of points. The 
    returned line equations will be represented by coeffs(dot)xyvect = dvect

    Inputs:

    points1 [numpy array] N1 points specified as a numpy array of shape 
            (N1,M=2). N1=1 or N1=N2. If N1=1, this point is used with every point
            in points2 to give N2 line equations. Otherwise, N1=N2, which will 
            give N=N1=N2 line equations from the N=N1=N2 corresponding pairs of 
            points

    points2 [numpy array] N2 points specified as a numpy array of shape 
            (N2,M=2). N2=1 or N2=N1. If N2=1, this point is used with every point
            in points1 to give N1 line equations. Otherwise, N1=N2, which will 
            give N=N1=N2 line equations from the N=N1=N2 corresponding pairs of 
            points

    Output:

    (N,M+1) array (an augmented matrix) where the (N=max(N1,N2),M=2) array 
    corresponds to the coefficients and the last column corresponds to the 
    dvect. If N1=1, this point in points1 is used with every point in points2 to 
    give N1 line equations. If N2=1, this point in points2 is used with every 
    point in points1 to give N1 line equations. Otherwise, N1=N2, which will give 
    N=N1=N2 line equations from the N=N1=N2 corresponding pairs of points. 
    -----------------------------------------------------------------------------
    """
    
    if not isinstance(points1, NP.ndarray):
        raise TypeError('Input points1 must be a numpy array')
    if points1.ndim != 2:
        raise ValueError('Input points1 must be a 2D array')
    if points1.shape[1] != 2:
        raise ValueError('Input points1 must be a (N,2) array')
        
    if not isinstance(points2, NP.ndarray):
        raise TypeError('Input points2 must be a numpy array')
    if points2.ndim != 2:
        raise ValueError('Input points2 must be a 2D array')
    if points2.shape[1] != 2:
        raise ValueError('Input points2 must be a (N,2) array')
    if points2.shape[0] != points1.shape[0]:
        if (points2.shape[0] != 1) and (points1.shape[0] != 1):
            raise ValueError('Number of points in points1 and points2 are not broadcastable')
    slopes = (points2[:,1] - points1[:,1])/(points2[:,0] - points1[:,0])
    coeffs_dvects = generate_line_from_point_and_slope(points1, slopes)
    return coeffs_dvects

#################################################################################

def get_abscissa_from_ordinate_on_line(coeffs, dvect, ordinates):
    """
    ----------------------------------------------------------------------------
    Find abscissae for given ordinates on a specified line: 
    coeffs(dot)xyvect = dvect
    
    Inputs:
    
    coeffs      [numpy array] (M=2)-element numpy array denoting (M=2) 
                parameters in (M=2)-dimensional space
    
    dvect       [int or float] Measured value on the RHS
    
    ordinates   [numpy array] y-values on the line
    
    Output:
    
    Numpy array representing x-values (abscissae) on the line. The array size is 
    equal to that of the input ordinates. If the slope of the line is zero, the 
    corresponding abscissa is set to NaN.
    ----------------------------------------------------------------------------
    """
    if not isinstance(coeffs, NP.ndarray):
        raise TypeError('Input coeffs must be a numpy array')
    if not isinstance(dvect, (int,float,NP.ndarray)):
        raise TypeError('Input dvect must be a scalar or numpy array')
    elif isinstance(dvect, NP.ndarray):
        if dvect.size != 1:
            raise ValueError('Size of input dvect must be 1')
        dvect = dvect.ravel()[0]        
    coeffs = coeffs.reshape(-1)
    if coeffs.size != 2:
        raise ValueError('Input coeffs must be of shape (1,2) or (2,)')
    if not isinstance(ordinates, (int,float,NP.ndarray)):
        raise TypeError('Input ordinates must be a scalar or numpy array')
    ordinates = NP.asarray(ordinates).reshape(-1)
    
    slope = -coeffs[0]/coeffs[1] # Check for infinite slope
    ind_zero_slope = NP.isinf(1/slope) # Check for zero slope
    ind_infinite_slope = NP.isinf(slope) # Check for infinite slope
    abscissae = (dvect - coeffs[1]*ordinates) / coeffs[0]
    abscissae[ind_zero_slope] = NP.nan
    abscissae[ind_infinite_slope] = dvect / coeffs[0]
    return abscissae

#################################################################################

def get_ordinate_from_abscissa_on_line(coeffs, dvect, abscissae):
    """
    ----------------------------------------------------------------------------
    Find ordinates for given abscissae on a specified line: 
    coeffs(dot)xyvect = dvect
    
    Inputs:
    
    coeffs      [numpy array] (M=2)-element numpy array denoting (M=2) 
                parameters in (M=2)-dimensional space
    
    dvect       [int or float] Measured value on the RHS
    
    abscissae   [numpy array] x-values on the line
    
    Output:
    
    Numpy array representing y-values (ordinates) on the line. The array size is 
    equal to that of the input abscissae. If the slope of the line is infinite, 
    the corresponding ordinate is set to NaN.
    ----------------------------------------------------------------------------
    """
    if not isinstance(coeffs, NP.ndarray):
        raise TypeError('Input coeffs must be a numpy array')
    if not isinstance(dvect, (int,float,NP.ndarray)):
        raise TypeError('Input dvect must be a scalar or numpy array')
    elif isinstance(dvect, NP.ndarray):
        if dvect.size != 1:
            raise ValueError('Size of input dvect must be 1')
        dvect = dvect.ravel()[0]        
    coeffs = coeffs.reshape(-1)
    if coeffs.size != 2:
        raise ValueError('Input coeffs must be of shape (1,2) or (2,)')
    coeffs = coeffs.reshape(-1)
    if not isinstance(abscissae, (int,float,NP.ndarray)):
        raise TypeError('Input abscissae must be a scalar or numpy array')
    abscissae = NP.asarray(abscissae).reshape(-1)

    slope = -coeffs[0]/coeffs[1] # Check for infinite slope
    ind_infinite_slope = NP.isinf(slope) # Check for infinite slope
    ind_zero_slope = NP.isinf(1/slope) # Check for zero slope
    ordinates = (dvect - coeffs[0]*abscissae) / coeffs[1]
    ordinates[ind_infinite_slope] = NP.nan
    ordinates[ind_zero_slope] = dvect / coeffs[1]
    return ordinates

#################################################################################

def generate_parallel_lines_at_distance_from_line(coeffs, dvects, distances):
    """
    -------------------------------------------------------------------------
    Given a set of equations of lines using coeff (dot) xyvect = dvect, find
    new dvects such that the new lines are at the specified distances from 
    the original lines. For each input line, two values of output dvects are
    returned corresponding to the positive and negative distances.
    
    Inputs:
    
    coeffs      [numpy array] Array of shape (N,2) or (1,2) specifying the 
                x and y coefficients of the N (or 1) lines.
                
    dvects      [scalar or numpy array] Array of shape (N,) or (1,) or scalar
                specifying the dvect of the input lines corresponding to 
                input coeffs.
                
    distances   [scalar or numpy array] Array of shape (N,) or (1,) or scalar
                specifying the distances from the corresponding lines given
                by coeffs and dvect. 
                
    Output:
    
    Output dvect array of shape (N,2) where [:,0] corresponds to negative
    distance offsets, and [:,1] corresponds to positive distance offsets from
    the original input lines. The coeffs remain the same as the slope is 
    unchanged. If any of the inputs coeffs, dvects, or distances have only
    1 inputs, then they are assumed to apply to all the N inputs and are 
    broadcasted. 
    -------------------------------------------------------------------------
    """
    
    if not isinstance(coeffs, NP.ndarray):
        raise TypeError('Input coeffs must be a numpy array')
    if isinstance(dvects, (int,float)):
        dvects = NP.asarray(dvects).reshape(-1)
    if isinstance(distances, (int,float)):
        distanes = NP.asarray(distances).reshape(-1)
    if not isinstance(dvects, NP.ndarray):
        raise TypeError('Input dvects must be a numpy array')
    if not isinstance(distances, NP.ndarray):
        raise TypeError('Input distances must be a numpy array')
    if coeffs.size == 2:
        coeffs = coeffs.reshape(-1,2)
    if coeffs.ndim != 2:
        raise ValueError('Input coeffs must be a 2D numpy array')
    if coeffs.shape[1] != 2:
        raise ValueError('Input coeffs must be a Nx2 array')
        
    dvects = dvects.reshape(-1)
    distances = NP.abs(distances).reshape(-1)
    
    n_dvects = dvects.size
    n_distances = distances.size
    n_coeffs = coeffs.shape[0]
    
    if (n_dvects != n_distances) and ((n_dvects != 1) and (n_distances != 1)):
        raise ValueError('Inputs dvects and distances must be equal in size or broadcastable')
    if (n_dvects != n_coeffs) and ((n_dvects != 1) and (n_coeffs != 1)):
        raise ValueError('Inputs dvects and number of coeffs must be equal in size or broadcastable')
    if (n_distances != n_coeffs) and ((n_distances != 1) and (n_coeffs != 1)):
        raise ValueError('Inputs distances and number of coeffs must be equal in size or broadcastable')
        
    n_max = NP.max([n_coeffs, n_dvects, n_distances])
    coeffs = coeffs + NP.zeros((n_max,2), dtype=float)
    dvects = dvects + NP.zeros(n_max, dtype=float)
    distances = distances + NP.zeros(n_max, dtype=float)
    
    pm_distances = NP.asarray([-1.0, 1.0]).reshape(1,2) * distances.reshape(-1,1)
    
    dvects_out_pm = dvects.reshape(-1,1) + pm_distances * NP.sqrt(NP.sum(coeffs**2, axis=1, keepdims=True))
    
    return dvects_out_pm

#################################################################################

def polygonArea2D(vertices, absolute=False):
    """
    ----------------------------------------------------------------------------
    Find area of a polygon in 2D when its vertices are specified
    
    Inputs:

    vertices   [numpy array] An array of shape (N,2) where each row specifies a 
               vertex in 2-dimensions

    absolute   [boolean] If set to False (default), return the signed area, otherwise
               return the absolute value
    
    Output: 

    area       [float] Area of the polygon. Will return zero if N<=2.

    ----------------------------------------------------------------------------
    """

    if not isinstance(vertices, NP.ndarray):
        raise TypeError('Input vertices must be a numpy array')
    if vertices.ndim != 2:
        raise ValueError('Input vertices must be a 2D array')
    if vertices.shape[0] <= 2: # Need at least 3 vertices to get non-zero area
        return 0.0 
    if vertices.shape[1] > 2:
        raise ValueError('This function can only compute area in 2D coordinates')
    if vertices.shape[1] == 1:
        return 0.0
    if not isinstance(absolute, bool):
        raise TypeError('Input absolute must be a boolean value')
    area = 0.5 * (NP.sum(vertices[:,0]*NP.roll(vertices[:,1],-1) - vertices[:,1]*NP.roll(vertices[:,0],-1)))
    if absolute:
        area = NP.abs(area)
    return area

#################################################################################

def altaz2dircos(altaz, units=None):

    """
    -----------------------------------------------------------------------------
    Convert altitude and azimuth to direction cosines

    Inputs:
    
    altaz:       Altitude and Azimuth as a list of tuples or Nx2 Numpy array
    
    Keyword Inputs:

    units:       [Default = 'radians'] Units of altitude and azimuth. Could be
                 radians or degrees. 

    Output:
    
    dircos:      Direction cosines corresponding to altitude and azimuth. The
                 first axis corresponds to local East, second to local North and
                 the third corresponds to Up.
    -----------------------------------------------------------------------------
    """

    try:
        altaz
    except NameError:
        raise NameError('No altitude or azimuth specified. Check inputs.')

    if not isinstance(altaz, NP.ndarray):
        if not isinstance(altaz, list):
            if not isinstance(altaz, tuple):
                raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            elif len(altaz) > 2:
                altaz = (altaz[0], altaz[1])
            elif len(altaz) < 2:
                raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif isinstance(altaz[0], tuple):
            for elem in altaz:
                if not isinstance(elem, tuple):
                    raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
                if len(elem) < 2:
                    raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            altaz = [((elem[0],elem[1]) if len(elem) > 2 else elem) for elem in altaz]
        elif len(altaz) != 2:
            raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
    else:
        if altaz.shape[1] < 2:
            raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif altaz.shape[1] > 2:
            altaz = altaz[:,:2]

    altaz = NP.asarray(altaz).reshape(-1,2)

    if units is None: units = 'radians'
    if units == 'degrees':
        altaz = NP.radians(altaz)

    if NP.any(NP.abs(altaz[:,0]) > NP.pi/2):
        raise ValueError('Altitude(s) should lie between -90 and 90 degrees. Check inputs and units.')

    phi = NP.pi/2 - altaz[:,1] # Convert azimuth (measured from north towards east) to angle measured from east towards north
    l = NP.cos(altaz[:,0])*NP.cos(phi) # towards east
    m = NP.cos(altaz[:,0])*NP.sin(phi) # towards north
    n = NP.sin(altaz[:,0])             # towards zenith

    return NP.asarray(list(zip(l,m,n)))

#################################################################################

def dircos2altaz(dircos, units=None):

    """
    -----------------------------------------------------------------------------
    Convert direction cosines to altitude and azimuth

    Inputs:
    
    dircos:      directin cosines as a list of tuples or Nx3 Numpy array
    
    Keyword Inputs:

    units:       [Default = 'radians'] Units of altitude and azimuth. Could be
                 radians or degrees. 

    Output:
    
    altaz:       Altitude and azimuth corresponding to direction cosines. 
    -----------------------------------------------------------------------------
    """

    try:
        dircos
    except NameError:
        raise NameError('No direction cosines specified. Check inputs.')

    if not isinstance(dircos, NP.ndarray):
        if not isinstance(dircos, list):
            if not isinstance(dircos, tuple):
                raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            elif len(dircos) > 3:
                dircos = (dircos[0], dircos[1], dircos[2])
            elif len(dircos) < 3:
                raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif isinstance(dircos[0], tuple):
            for elem in dircos:
                if not isinstance(elem, tuple):
                    raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
                if len(elem) < 3:
                    raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            dircos = [((elem[0],elem[1],elem[2]) if len(elem) > 3 else elem) for elem in dircos]
        elif len(dircos) != 3:
            raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
    else:
        if dircos.shape[1] < 3:
            raise TypeError('dircos should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif dircos.shape[1] > 3:
            dircos = dircos[:,:3]

    dircos = NP.asarray(dircos)

    eps = 1.0e-10
    if ((NP.any(NP.abs(dircos[:,0]) > 1.0)) or
        (NP.any(NP.abs(dircos[:,1]) > 1.0)) or
        (NP.any(NP.abs(dircos[:,2]) > 1.0)) or
        (NP.any(NP.abs(NP.sqrt(NP.sum(dircos**2,axis=1))-1.0) > eps))):

        raise ValueError('Individual components of direction cosines should lie between 0 and 1 and the direction cosines should have unit magnitudes. Check inputs.')

    altaz = NP.empty((dircos.shape[0],2))
    altaz[:,0] = NP.pi/2 - NP.arccos(dircos[:,2]) # Altitude/elevation
    altaz[:,1] = NP.pi/2 - NP.arctan2(dircos[:,1],dircos[:,0]) # Azimuth (measured from North)
    altaz[:,1] = NP.mod(altaz[:,1], 2*NP.pi)

    if units is None: units = 'radians'
    if units == 'degrees':
        altaz = NP.degrees(altaz)

    return altaz

#################################################################################

def hadec2altaz(hadec, latitude, units=None):

    """
    -----------------------------------------------------------------------------
    Convert HA and declination to altitude and azimuth

    Inputs:
    
    hadec:       HA and declination as a list of tuples or Nx2 Numpy array
    
    latitude:    Latitude of the observatory. 

    Keyword Inputs:

    units:       [Default = 'radians'] Units of HA, dec and latitude. Could be
                 'radians' or 'degrees'. 

    Output:
    
    altaz:       Altitude and azimuth corresponding to HA and dec at the given
                 latitude. Units are identical to those in input.
    -----------------------------------------------------------------------------
    """

    try:
        hadec
    except NameError:
        raise NameError('No HA or declination specified. Check inputs.')

    try:
        latitude
    except NameError:
        raise NameError('No latitude specified. Check inputs.')

    if isinstance(latitude, (list, tuple, str)):
        raise TypeError('Latitude should be a scalar number.')

    if not isinstance(hadec, NP.ndarray):
        if not isinstance(hadec, list):
            if not isinstance(hadec, tuple):
                raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            elif len(hadec) > 2:
                hadec = (hadec[0], hadec[1])
            elif len(hadec) < 2:
                raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif isinstance(hadec[0], tuple):
            for elem in hadec:
                if not isinstance(elem, tuple):
                    raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
                if len(elem) < 2:
                    raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            hadec = [((elem[0],elem[1]) if len(elem) >= 2 else elem) for elem in hadec]
        elif len(hadec) != 2:
            raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
    else:
        if len(hadec.shape) < 2:
            hadec = NP.asarray(hadec).reshape(-1,hadec.size)

        if hadec.shape[1] < 2:
            raise TypeError('hadec should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif hadec.shape[1] > 2:
            hadec = hadec[:,0:2]

    hadec = NP.asarray(hadec)

    if units is None: units = 'radians'
    if units == 'degrees':
        hadec = NP.radians(hadec)
        latitude = NP.radians(latitude)

    if NP.any(NP.abs(hadec[:,1]) > NP.pi/2):
        raise ValueError('Declination(s) should lie between -90 and 90 degrees. Check inputs and units.')

    if NP.abs(latitude) > NP.pi/2:
        raise ValueError('Latitude should lie between -90 and 90 degrees. Check inputs and units.')

    eps = 1e-10
    sin_alt = NP.sin(hadec[:,1])*NP.sin(latitude) + NP.cos(hadec[:,1])*NP.cos(latitude)*NP.cos(hadec[:,0])
    valid_ind = NP.where(NP.abs(sin_alt) <= 1.0+eps)
    sin_alt[valid_ind] = NP.clip(sin_alt[valid_ind], -1.0, 1.0)
    altitude = NP.arcsin(sin_alt)
    zenith_ind = NP.abs(NP.abs(sin_alt)-1.0) < eps
    cos_az = NP.where(zenith_ind, NP.zeros_like(sin_alt), (NP.sin(hadec[:,1])-NP.sin(altitude)*NP.sin(latitude))/(NP.cos(altitude)*NP.cos(latitude)))
    eps = 1e-6
    valid_ind = NP.where(NP.abs(cos_az) <= 1.0+eps)
    cos_az[valid_ind] = NP.clip(cos_az[valid_ind], -1.0, 1.0)
    azimuth = NP.arccos(cos_az)

    # Need to make sure the values are in the conventional range
    azimuth = NP.where(NP.sin(hadec[:,0])<0.0, azimuth, 2.0*NP.pi-azimuth)
    if units == 'degrees':
        altitude = NP.degrees(altitude)
        azimuth = NP.degrees(azimuth)
    return NP.asarray(list(zip(altitude, azimuth)))

#################################################################################

def altaz2hadec(altaz, latitude, units=None):

    """
    -----------------------------------------------------------------------------
    Convert altitude and azimuth to HA and declination. Same transformation
    function as hadec2altaz, replace azimuth with HA and altitude with declination.

    Inputs:
    
    altaz:       Altitude and azimtuh as a list of tuples or Nx2 Numpy array
    
    latitude:    Latitude of the observatory. 

    Keyword Inputs:

    units:       [Default = 'radians'] Units of HA, dec and latitude. Could be
                 radians or degrees. 

    Output:
    
    hadec:       HA and declination corresponding to altitude and azimuth given
                 latitude. Units are identical to those in input.
    -----------------------------------------------------------------------------
    """

    try:
        altaz
    except NameError:
        raise NameError('No altitude or azimuth specified. Check inputs.')

    try:
        latitude
    except NameError:
        raise NameError('No latitude specified. Check inputs.')

    if isinstance(latitude, (list, tuple, str)):
        raise TypeError('Latitude should be a scalar number.')

    if not isinstance(altaz, NP.ndarray):
        if not isinstance(altaz, list):
            if not isinstance(altaz, tuple):
                raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            elif len(altaz) > 2:
                altaz = (altaz[0], altaz[1])
            elif len(altaz) < 2:
                raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif isinstance(altaz[0], tuple):
            for elem in altaz:
                if not isinstance(elem, tuple):
                    raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
                if len(elem) < 2:
                    raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
            altaz = [((elem[0],elem[1]) if len(elem) >= 2 else elem) for elem in altaz]
        elif len(altaz) != 2:
            raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
    else:
        if altaz.ndim == 1:
            altaz = altaz.reshape(1,-1)
        if altaz.shape[1] < 2:
            raise TypeError('altaz should be a 2-element tuple, a list of 2-element tuples, a 2-element list of scalars, or a Nx2 numpy array. Check inputs.')
        elif altaz.shape[1] > 2:
            altaz = altaz[:,0:2]

    altaz = NP.asarray(altaz)

    if units is None: units = 'radians'
    if units == 'degrees':
        altaz = NP.radians(altaz)
        latitude = NP.radians(latitude)

    if NP.any(NP.abs(altaz[:,0]) > NP.pi/2):
        raise ValueError('Altitude(s) magnitude should be <= 90 degrees. Check inputs and units.')
    
    if NP.abs(latitude) > NP.pi/2:
        raise ValueError('Latitude should lie between -90 and 90 degrees. Check inputs and units.')

    eps = 1e-10
    
    # arg = NP.sin(altaz[:,0])*NP.sin(latitude) + NP.cos(altaz[:,1])*NP.cos(latitude)*NP.cos(altaz[:,0])
    arg = NP.sin(altaz[:,0].astype(NP.longdouble))*NP.sin(latitude.astype(NP.longdouble)) + NP.cos(altaz[:,1].astype(NP.longdouble))*NP.cos(latitude.astype(NP.longdouble))*NP.cos(altaz[:,0].astype(NP.longdouble))
    if NP.abs(arg).max() > 1.0:
        if NP.abs(arg).max() - 1.0 > eps:
            raise ValueError('Non-physical angles found')
        else:
            arg = NP.clip(arg, -1.0, 1.0)
    dec = NP.arcsin(arg)

    # arg = (NP.sin(altaz[:,0])-NP.sin(dec)*NP.sin(latitude))/(NP.cos(dec)*NP.cos(latitude))
    arg = (NP.sin(altaz[:,0].astype(NP.longdouble))-NP.sin(dec.astype(NP.longdouble))*NP.sin(latitude.astype(NP.longdouble)))/(NP.cos(dec.astype(NP.longdouble))*NP.cos(latitude.astype(NP.longdouble)))
    if NP.abs(arg).max() > 1.0:
        if NP.abs(arg).max() - 1.0 > eps:
            raise ValueError('Non-physical angles found')
        else:
            arg = NP.clip(arg, -1.0, 1.0)

    pole_ind = NP.abs(NP.abs(NP.sin(dec)*NP.sin(latitude))-1.0) < eps # When the telescope is pointed at the poles while located at the poles
    arg[pole_ind] = NP.sign(dec[pole_ind]) + 0.0 # Assign HA = 0 or 180 deg
    
    ha = NP.arccos(arg)
 
    # Make sure the conventions are taken into account
    ha = NP.where(NP.sin(altaz[:,1])<0.0, ha, -ha)

    if units == 'degrees':
        ha *= 180.0/NP.pi
        dec *= 180.0/NP.pi
    return NP.asarray(list(zip(ha, dec)))

#################################################################################

def enu2xyz(enu, latitude, units='radians'):

    """
    -----------------------------------------------------------------------------
    Convert local ENU coordinates in local tangential plane to XYZ in equatorial
    coordinates.

    Inputs:
    
    enu:         local ENU coordinates in local tangential plane as a list of
                 tuples or Nx3 Numpy array. First column refers to local East,
                 second to local North and third to local Up.
    
    latitude:    Latitude of the observatory. 

    Keyword Inputs:

    units:       [Default = 'radians'] Units of latitude. Could be radians or
                 degrees. 

    Output:
    
    xyz:         Equatorial XYZ coordinates corresponding to local ENU 
                 coordinates given latitude. Units are identical to those in
                 input. First column refers to X (ha=0, dec=0), second column to
                 Y (ha=-6h, dec=0), and third column to north celestial pole 
                 Z (dec=90)
    -----------------------------------------------------------------------------
    """

    try:
        enu
    except NameError:
        raise NameError('No baselines specified. Check inputs.')

    try:
        latitude
    except NameError:
        raise NameError('No latitude specified. Check inputs.')

    if isinstance(latitude, (list, tuple, str)):
        raise TypeError('Latitude should be a scalar number.')

    if not isinstance(enu, NP.ndarray):
        if not isinstance(enu, list):
            if not isinstance(enu, tuple):
                raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            elif len(enu) > 3:
                enu = (enu[0], enu[1], enu[2])
            elif len(enu) < 3:
                raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif isinstance(enu[0], tuple):
            for elem in enu:
                if not isinstance(elem, tuple):
                    raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
                if len(elem) < 3:
                    raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            enu = [((elem[0],elem[1]) if len(elem) >= 3 else elem) for elem in enu]
        elif len(enu) != 3:
            raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
    else:
        if enu.shape[1] < 3:
            raise TypeError('enu should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif enu.shape[1] > 3:
            enu = enu[:,0:3]

    enu = NP.asarray(enu)

    if units == 'degrees':
        latitude = NP.radians(latitude)

    rotation_matrix = NP.asarray([[0.0, -NP.sin(latitude), NP.cos(latitude)],
                                  [1.0, 0.0,               0.0             ],
                                  [0.0, NP.cos(latitude),  NP.sin(latitude)]])

    xyz = NP.dot(enu, rotation_matrix.T)

    return xyz

#################################################################################

def xyz2enu(xyz, latitude, units='radians'):

    """
    -----------------------------------------------------------------------------
    Convert equatorial XYZ coordinates to local ENU coordinates in the local 
    tangential plane.

    Inputs:
    
    xyz:         equatorial XYZ coordinates as a list of tuples or Nx3 Numpy
                 array. First column refers to X (ha=0, dec=0), second column to
                 Y (ha=-6h, dec=0), and third column to north celestial pole 
                 Z (dec=90)
    
    latitude:    Latitude of the observatory. 

    Keyword Inputs:

    units:       [Default = 'radians'] Units of latitude. Could be radians or
                 degrees. 

    Output:
    
    enu:         local ENU coordinates in the local tangential plane
                 corresponding to equatorial XYZ coordinates given latitude.
                 Units are identical to those in input. First column refers to 
                 local East, second to local North and third to local Up.
    -----------------------------------------------------------------------------
    """

    try:
        xyz
    except NameError:
        raise NameError('No baselines specified. Check inputs.')

    try:
        latitude
    except NameError:
        raise NameError('No latitude specified. Check inputs.')

    if isinstance(latitude, (list, tuple, str)):
        raise TypeError('Latitude should be a scalar number.')

    if not isinstance(xyz, NP.ndarray):
        if not isinstance(xyz, list):
            if not isinstance(xyz, tuple):
                raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            elif len(xyz) > 3:
                xyz = (xyz[0], xyz[1], xyz[2])
            elif len(xyz) < 3:
                raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif isinstance(xyz[0], tuple):
            for elem in xyz:
                if not isinstance(elem, tuple):
                    raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
                if len(elem) < 3:
                    raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
            xyz = [((elem[0],elem[1]) if len(elem) >= 3 else elem) for elem in xyz]
        elif len(xyz) != 3:
            raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
    else:
        if xyz.shape[1] < 3:
            raise TypeError('xyz should be a 3-element tuple, a list of 3-element tuples, a 3-element list of scalars, or a Nx3 numpy array. Check inputs.')
        elif xyz.shape[1] > 3:
            xyz = xyz[:,0:3]

    xyz = NP.asarray(xyz)

    if units == 'degrees':
        latitude = NP.radians(latitude)

    rotation_matrix = NP.asarray([[0.0               , 1.0, 0.0],
                                  [NP.sin(-latitude), 0.0, NP.cos(-latitude)],
                                  [NP.cos(-latitude) , 0.0, -NP.sin(-latitude)]])

    enu = NP.dot(xyz, rotation_matrix.T)

    return enu

#################################################################################

def lla2ecef(lat, lon, alt=None, units='radians'):

    """
    -----------------------------------------------------------------------------
    Convert geodetic latitude, longitude and altitude to XYZ in ECEF coordinates
    based on WGS84 parameters

    Inputs:

    lat     [scalar or numpy array] Geodetic latitude in units specified by 
            input units. Same size as lon and alt

    lon     [scalar or numpy array] Geodetic longitude in units specified by 
            input units. Same size as lat and alt.

    alt     [scalar or numpy array] Geodetic altitude in meters. Same size as 
            lat and lon. If set to None, it is assumed to be zeros.

    units   [string] Specifies units of inputs lat and lon. Accepted values are
            'radians' (default) or 'degrees'

    Outputs:

    Tuple (x,y,z) where x, y and z in meters are the components in the ECEF 
    system. Each will be of same size as lat
    -----------------------------------------------------------------------------
    """

    try:
        lat, lon
    except NameError:
        raise NameError('Inputs lat and lon must be specified')

    if units not in ['degrees', 'radians']:
        raise ValueError('Invalid input specified for "units"')
    if not isinstance(lat, (int,float,NP.ndarray)):
        raise TypeError('Input lat must be a scalar or numpy array')
    lat = NP.asarray(lat).reshape(-1)
    if not isinstance(lon, (int,float,NP.ndarray)):
        raise TypeError('Input lon must be a scalar or numpy array')
    lon = NP.asarray(lon).reshape(-1)
    if lat.size != lon.size:
        raise ValueError('Inputs lat and lon must be of same size')
    if alt is not None:
        if not isinstance(alt, (int,float,NP.ndarray)):
            raise TypeError('Input alt must be a scalar or numpy array')
        alt = NP.asarray(alt).reshape(-1)
        if alt.size != lat.size:
            raise ValueError('Input alt must have same size as input lat')
    else:
        alt = NP.zeros_like(lat)

    lat = lat.ravel()
    lon = lon.ravel()
    alt = alt.ravel()
    if units != 'radians':
        lat = NP.radians(lat)
        lon = NP.radians(lon)

    gps_a = 6378137.0
    gps_f = 1.0 / 298.257223563 # flattening parameter
    gps_b = gps_a * (1 - gps_f)
    e_sqr = 1.0 - (gps_b/gps_a)**2 # first eccentricity
    eprime_sqr = (gps_a/gps_b)**2 - 1.0 # second eccentricity
    gps_N = gps_a / NP.sqrt(1.0 - e_sqr * NP.sin(lat)**2) # Radius of curvature
    x = (gps_N + alt) * NP.cos(lat) * NP.cos(lon)
    y = (gps_N + alt) * NP.cos(lat) * NP.sin(lon)
    z = (gps_b**2 / gps_a**2 * gps_N + alt) * NP.sin(lat)

    return (x, y, z)

#################################################################################

def ecef2lla(x, y, z, units='radians'):

    """
    -----------------------------------------------------------------------------
    Convert XYZ in ECEF to geodetic latitude, longitude and altitude coordinates
    based on WGS84 parameters

    Inputs:

    x       [scalar or numpy array] x-coordinate (in m) in ECEF system 

    y       [scalar or numpy array] y-coordinate (in m) in ECEF system 

    z       [scalar or numpy array] z-coordinate (in m) in ECEF system 

    units   [string] Specifies units of outputs lat and lon. Accepted values are
            'radians' (default) or 'degrees'

    Outputs:

    Tuple (lat, lon, alt) where lat (angle units), lon (angle units) and alt (m) 
    are the geodetic latitude, longitudes and altitudes in WGS 84 system
    -----------------------------------------------------------------------------
    """

    try:
        x, y, z
    except NameError:
        raise NameError('Inputs x, y, z must be specified')

    if units not in ['degrees', 'radians']:
        raise ValueError('Invalid input specified for "units"')

    if not isinstance(x, (int,float,NP.ndarray)):
        raise TypeError('Input x must be a scalar or numpy array')
    x = NP.asarray(x).reshape(-1)
    if not isinstance(y, (int,float,NP.ndarray)):
        raise TypeError('Input y must be a scalar or numpy array')
    y = NP.asarray(y).reshape(-1)
    if not isinstance(z, (int,float,NP.ndarray)):
        raise TypeError('Input z must be a scalar or numpy array')
    z = NP.asarray(z).reshape(-1)
    if (x.size != y.size) or (x.size != z.size):
        raise ValueError('Inputs x, y and z must be of same size')

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # checking for acceptable values for Earth's surface
    xyz_radius = NP.sqrt(x**2 + y**2 + z**2)
    if NP.any(NP.logical_or(xyz_radius < 6.35e6, xyz_radius > 6.39e6)):
        raise ValueError('xyz values should be topocentric ECEF coordinates in meters')

    gps_a = 6378137.0
    gps_f = 1.0 / 298.257223563 # flattening parameter
    gps_b = gps_a * (1 - gps_f)
    e_sqr = 1.0 - (gps_b/gps_a)**2 # first eccentricity
    eprime_sqr = (gps_a/gps_b)**2 - 1.0 # second eccentricity
    # gps_N = gps_a / NP.sqrt(1.0 - e_sqr * NP.sin(x)**2) # Radius of curvature (Wrong)
    gps_p = NP.sqrt(x**2 + y**2)
    gps_theta = NP.arctan2(z * gps_a, gps_p * gps_b)

    lon = NP.arctan2(y, x)
    lat = NP.arctan2(z + eprime_sqr * gps_b * NP.sin(gps_theta)**3, gps_p - e_sqr * gps_a * NP.cos(gps_theta)**3)
    gps_N = gps_a / NP.sqrt(1.0 - e_sqr * NP.sin(lat)**2) # Radius of curvature
    alt = gps_p / NP.cos(lat) - gps_N

    if units != 'radians':
        lat = NP.degrees(lat)
        lon = NP.degrees(lon)

    return (lat, lon, alt)

#################################################################################

def ecef2enu(xyz, ref_info):

    """
    -----------------------------------------------------------------------------
    Convert XYZ in ECEF based on WGS 84 parameters to local ENU coordinates
    (Refer to Wikipedia 
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion)

    Inputs:

    xyz     [numpy array] xyz-coordinates (in m) in ECEF system. It is of size
            nloc x 3.

    ref_info
            [dictionary] contains information about the reference point relative 
            to which all the ENU coordinates will be estimated. It consists of 
            the following keys and information:
            'xyz'   [3-element numpy array] ECEF XYZ location of the reference
                    point
            'lat'   [scalar] geodetic latitude (in radians or degrees as 
                    specified by key 'units') of reference point 
            'lon'   [scalar] geodetic longitude (in radians or degrees as 
                    specified by key 'units') of reference point 
            'units' [string] Specifies units of lat and lon of reference point. 
                    Accepted values are 'radians' (default) or 'degrees'

    Outputs:

    Numpy array containing converted local ENU locations of the input ECEF 
    locations. It will be of size nloc x 3
    -----------------------------------------------------------------------------
    """

    try:
        xyz, ref_info
    except NameError:
        raise NameError('Inputs xyz and ref_info must be specified')

    if not isinstance(xyz, NP.ndarray):
        raise TypeError('Input xyz must be a numpy array')
    if xyz.ndim == 1:
        if xyz.size != 3:
            raise ValueError('Input xyz must be a 3-column array')
        xyz = xyz.reshape(1,-1)
    elif xyz.ndim == 2:
        if xyz.shape[1] != 3:
            raise ValueError('Input xyz must be a 3-column array')
    else:
        raise ValueError('Input xyz must be a 2D 3-column array')

    if isinstance(ref_info, dict):
        if 'xyz' in ref_info:
            if isinstance(ref_info['xyz'], NP.ndarray):
                if ref_info['xyz'].size != 3:
                    raise ValueError('Value under key "xyz" in input ref_info must be a 3-element numpy array')
                ref_info['xyz'] = ref_info['xyz'].ravel()
            else:
                raise TypeError('Value under key "xyz" in input ref_info must be a numpy array')
        else:
            raise KeyError('Key "xyz" not found in input ref_info')

        if 'lat' in ref_info:
            if not isinstance(ref_info['lat'], (int,float)):
                raise TypeError('Value under key "lat" in input ref_info must be a scalar number')
            ref_info['lat'] = float(ref_info['lat'])
        else:
            raise KeyError('Key "lat" not found in input ref_info')
        if 'lon' in ref_info:
            if not isinstance(ref_info['lon'], (int,float)):
                raise TypeError('Value under key "lon" in input ref_info must be a scalar number')
            ref_info['lon'] = float(ref_info['lon'])
        else:
            raise KeyError('Key "lon" not found in input ref_info')
        if 'units' in ref_info:
            if ref_info['units'] not in ['degrees', 'radians']:
                raise ValueError('Invalid specification for value under key "units" in input ref_info')
        else:
            warnings.warn('Value under key "units" in input ref_info not specified. Assuming the units are in radians.')
    else:
        raise TypeError('Input ref_info must be a dictionary')

    ref_lat = ref_info['lat']
    ref_lon = ref_info['lon']
    if ref_info['units'] != 'radians':
        ref_lat = NP.radians(ref_lat)
        ref_lon = NP.radians(ref_lon)

    xyz_rel = xyz - ref_info['xyz'].reshape(1,-1)
    rot_matrix = NP.asarray([[-NP.sin(ref_lon), NP.cos(ref_lon), 0.0],
                             [-NP.sin(ref_lat)*NP.cos(ref_lon), -NP.sin(ref_lat)*NP.sin(ref_lon), NP.cos(ref_lat)],
                             [NP.cos(ref_lat)*NP.cos(ref_lon), NP.cos(ref_lat)*NP.sin(ref_lon), NP.sin(ref_lat)]])
    enu = NP.dot(xyz_rel, rot_matrix.T)
    return enu

#################################################################################

def enu2ecef(enu, ref_info):

    """
    -----------------------------------------------------------------------------
    Convert local ENU coordinates to XYZ in ECEF based on WGS 84 parameters
    (Refer to Wikipedia 
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion)

    Inputs:

    enu     [numpy array] local ENU-coordinates (in m). It is of size nloc x 3.

    ref_info
            [dictionary] contains information about the reference point relative 
            to which all the ECEF coordinates will be estimated. It consists of 
            the following keys and information:
            'lat'   [scalar] geodetic latitude (in radians or degrees as 
                    specified by key 'units') of reference point 
            'lon'   [scalar] geodetic longitude (in radians or degrees as 
                    specified by key 'units') of reference point 
            'alt'   [scalar] Altitude (in same units as ENU coordinates) of 
                    reference point. If none specified, it is assumed to be
                    0.0 (default)
            'units' [string] Specifies units of lat and lon of reference point. 
                    Accepted values are 'radians' (default) or 'degrees'

    Outputs:

    Numpy array containing converted ECEF XYZ locations of the input local ENU 
    locations. It will be of size nloc x 3
    -----------------------------------------------------------------------------
    """

    try:
        enu, ref_info
    except NameError:
        raise NameError('Inputs enu and ref_info must be specified')

    if not isinstance(enu, NP.ndarray):
        raise TypeError('Input enu must be a numpy array')
    if enu.ndim == 1:
        if enu.size != 3:
            raise ValueError('Input enu must be a 3-column array')
        enu = enu.reshape(1,-1)
    elif enu.ndim == 2:
        if enu.shape[1] != 3:
            raise ValueError('Input enu must be a 3-column array')
    else:
        raise ValueError('Input enu must be a 2D 3-column array')

    if isinstance(ref_info, dict):
        if 'lat' in ref_info:
            if not isinstance(ref_info['lat'], (int,float)):
                raise TypeError('Value under key "lat" in input ref_info must be a scalar number')
            ref_info['lat'] = float(ref_info['lat'])
        else:
            raise KeyError('Key "lat" not found in input ref_info')

        if 'lon' in ref_info:
            if not isinstance(ref_info['lon'], (int,float)):
                raise TypeError('Value under key "lon" in input ref_info must be a scalar number')
            ref_info['lon'] = float(ref_info['lon'])
        else:
            raise KeyError('Key "lon" not found in input ref_info')

        if 'alt' in ref_info:
            if not isinstance(ref_info['alt'], (int,float)):
                raise TypeError('Value under key "alt" in input ref_info must be a scalar number')
            ref_info['alt'] = float(ref_info['alt'])
        else:
            ref_info['alt'] = 0.0
            
        if 'units' in ref_info:
            if ref_info['units'] not in ['degrees', 'radians']:
                raise ValueError('Invalid specification for value under key "units" in input ref_info')
        else:
            warnings.warn('Value under key "units" in input ref_info not specified. Assuming the units are in radians.')
    else:
        raise TypeError('Input ref_info must be a dictionary')

    ref_lat = ref_info['lat']
    ref_lon = ref_info['lon']
    ref_alt = ref_info['alt']
    if ref_info['units'] != 'radians':
        ref_lat = NP.radians(ref_lat)
        ref_lon = NP.radians(ref_lon)
    ref_x, ref_y, ref_z = lla2ecef(ref_lat, ref_lon, alt=ref_alt, units='radians')
    ref_xyz = NP.hstack((ref_x.reshape(-1,1), ref_y.reshape(-1,1), ref_z.reshape(-1,1)))

    rot_matrix = NP.asarray([[-NP.sin(ref_lon), -NP.sin(ref_lat)*NP.cos(ref_lon), NP.cos(ref_lat)*NP.cos(ref_lon)],
                             [NP.cos(ref_lon), -NP.sin(ref_lat)*NP.sin(ref_lon), NP.cos(ref_lat)*NP.sin(ref_lon)],
                             [0.0, NP.cos(ref_lat), NP.sin(ref_lat)]])
    xyz = NP.dot(enu, rot_matrix.T) + ref_xyz
    return xyz

#################################################################################

# def angular_ring(skypos, angles, npoints=100, skyunits='radec', angleunits='degrees'):

#     skypos = NP.radians(skypos).reshape(-1,2)
#     angles = NP.asarray(angles).reshape(1,-1)
#     if angleunits == 'degrees':
#         angles = NP.radians(angles)

#     azangle = 2.0 * NP.pi * NP.arange(npoints)/npoints 

#     radec = NP.empty((npoints, 2, skypos.shape[0]))

#     for i in range(skypos.shape[0]):
#         local_enu = NP.hstack((NP.sin(angles[i])*NP.cos(azangle).reshape(-1,1), NP.sin(angles[i])*NP.sin(azangle).reshape(-1,1), NP.cos(angles[i])*NP.ones((npoints,1))))
#         rotation_matrix = NP.asarray([[0.0, -NP.sin(skypos[i,1]), NP.cos(skypos[i,1])],
#                                       [1.0, 0.0,                  0.0              ],
#                                       [0.0, NP.cos(skypos[i,1]),  NP.sin(skypos[i,1])]])
#         xyz = NP.dot(local_enu, rotation_matrix.T)
#         ring_dec = NP.degrees(NP.arcsin(xyz[:,2]))
#         ring_az = NP.degrees(NP.arctan(xyz[:,1], xyz[:,0]))
#         ring_ra = NP.degrees(skypos[:,0]) + ring_az

#         radec[:,0,i] = ring_ra
#         radec[:,1,i] = ring_dec

#         return radec

#     # local_east = NP.repeat(NP.cos(azangle).reshape(npoints,-1), len(angles), axis=1) * NP.repeat(NP.sin(angles).reshape(-1,len(angles)), npoints, axis=0)
#     # local_north = NP.repeat(NP.sin(azangle).reshape(npoints,-1), len(angles), axis=1) * NP.repeat(NP.sin(angles).reshape(-1,len(angles)), npoints, axis=0)
#     # local_up = NP.repeat(NP.cos(angles).reshape(-1,len(angles)), npoints, axis=0) 

#     # Rotate the "north" and "up" to equatorial coordinates by the declination provided

#################################################################################

def sph2xyz(lon, lat, rad=None):

    """
    -----------------------------------------------------------------------------
    Convert from spherical coordinates (radius, latitude, longigtude) to 
    Cartesian coordinates

    Inputs:

    lon [scalar or vector] longitude in degrees.  Longitude is equivalent to
        azimuth = 90 - CCW angle from X-axis

    lat [scalar or vector] latitude (=90 - ZA) in degrees. Same size as lon. 

    rad [Optional. scalar or vector] radius. Same size as lon and lat.
        Default = 1.0

    Outputs:

    x   [scalar or vector] x-coordinates. Same size as lon and lat

    y   [scalar or vector] y-coordinates. Same size as lon and lat

    z   [scalar or vector] z-coordinates. Same size as lon and lat
    -----------------------------------------------------------------------------
    """

    try:
        lon, lat
    except NameError:
        raise NameError('lon and/or lat not defined in sph2xyz().')

    lonr = NP.radians(lon)
    latr = NP.radians(lat)

    if lonr.size != latr.size:
        raise ValueError('lon and lat should have same size in sph2xyz().')

    if rad is None:
        rad = NP.ones(lonr.size)
    else:
        rad = NP.asarray(rad)

    if lonr.size != rad.size:
        raise ValueError('rad must have same size as lon and lat in sph2xyz().')

    x = rad * NP.cos(NP.pi/2-lonr) * NP.sin(NP.pi/2-latr)
    y = rad * NP.sin(NP.pi/2-lonr) * NP.sin(NP.pi/2-latr)
    z = rad * NP.cos(NP.pi/2-latr)

    return x, y, z

#################################################################################

def xyz2sph(x, y, z, units='radians'):

    """
    -----------------------------------------------------------------------------
    Convert from Cartesian coordinates to spherical coordinates (radius, latitude
    and longitude)

    Inputs:

    x       [scalar or vector] x-coordinates. Same size as y and z
            
    y       [scalar or vector] y-coordinates. Same size as x and z
            
    z       [scalar or vector] z-coordinates. Same size as x and y

    units   [string] Specifies units of output latitude and longitude. If set
            to 'degrees' it will be in degrees, otherwise in radians

    Outputs:

    r   [scalar or vector] radius. Same size and units as x, y and z 

    lon [scalar or vector] longitude in units specified by keyword input 'units'

    lat [scalar or vector] latitude in units specified by keyword input 'units'
    -----------------------------------------------------------------------------
    """

    try:
        x, y, z
    except NameError:
        raise NameError('x, y, and z must be defined in xyz2sph().')

    r = NP.sqrt(x**2 + y**2 + z**2)
    lat = NP.pi/2 - NP.arccos(z/r)
    lon = NP.pi/2 - NP.arctan2(y,x)
    if units == 'degrees':
        lat = NP.degrees(lat)
        lon = NP.degrees(lon)

    return r, lat, lon

#################################################################################

def sphdist(lon1, lat1, lon2, lat2):

    """
    -----------------------------------------------------------------------------
    Returns great circle distance.  

    Uses vicenty distance formula - a bit slower than others, but numerically
    stable.

    Inputs: 

    lon1 [scalar or vector] Longtitude in first list in degrees

    lat1 [scalar or vector] Latitude in first list in degrees. Must be of same
         size as lon1 if vector

    lon2 [scalar or vector] Longtitude in second list in degrees. Must be of same
         size as lon1 if vector

    lat2 [scalar or vector] Latitude in second list in degrees. Must be of same
         size as lon1 if vector

    Outputs:

    Angular distance (in degrees) subtended on the great circle between the given
    set of points. Same size as the inputs.
    -----------------------------------------------------------------------------
    """

    try:
        lon1, lat1, lon2, lat2
    except NameError:
        raise NameError('At least one of lon1, lat1, lon2, lat2 undefined in sphdist().')
        
    lon1 = NP.asarray(lon1)
    lat1 = NP.asarray(lat1)

    if lon1.size != lat1.size:
        raise ValueError('lon1 and lat1 must be of same size in sphdist().')

    lon2 = NP.asarray(lon2)
    lat2 = NP.asarray(lat2)

    if lon2.size != lat2.size:
        raise ValueError('lon2 and lat2 must be of same size in sphdist().')

    if lon1.size != lon2.size:
        if lon1.size == 1:
            lon1 = lon1 * NP.ones(lon2.size)
            lat1 = lat1 * NP.ones(lat2.size)
        elif lon2.size == 1:
            lon2 = lon2 * NP.ones(lon1.size)
            lat2 = lat2 * NP.ones(lat1.size)
        else:
            raise ValueError('first and second sets of coordinates must have same size.')

    # terminology from the Vicenty formula - lambda and phi and
    # "standpoint" and "forepoint"

    lambs = NP.radians(lon1)
    phis = NP.radians(lat1)
    lambf = NP.radians(lon2)
    phif = NP.radians(lat2)

    dlamb = lambf - lambs

    numera = NP.cos(phif) * NP.sin(dlamb)
    numerb = NP.cos(phis) * NP.sin(phif) - NP.sin(phis) * NP.cos(phif) * NP.cos(dlamb)
    numer = NP.hypot(numera, numerb)
    denom = NP.sin(phis) * NP.sin(phif) + NP.cos(phis) * NP.cos(phif) * NP.cos(dlamb)

    return NP.degrees(NP.arctan2(numer, denom)).ravel()

#################################################################################

def spherematch(lon1, lat1, lon2=None, lat2=None, matchrad=None, nnearest=0,
                maxmatches=-1):

    """
    -----------------------------------------------------------------------------
    Finds matches in one catalog to another. Matches for the first catalog are
    searched for in the second catalog.

    Parameters
    lon1 : array-like
         Longitude-like (RA, etc.) coordinate in degrees of the first catalog
    lat1 : array-like
        Latitude-like (Dec, etc.) coordinate in degrees of the first catalog
        (shape of array must match `lon1`)
    lon2 : array-like
        Latitude-like (RA, etc.) coordinate in degrees of the second catalog
    lat2 : array-like
        Latitude-like (Dec, etc.) in degrees of the second catalog (shape of
        array must match `ra2`)
    matchrad : float or None, optional
        How close (in degrees) a match has to be to count as a match.  If None,
        all nearest neighbors for the first catalog will be returned. 
    nnearest : int, optional
        The nth nearest neighbor to find. Default = 0 (if maxmatches >= 0,
        maxmatches is used, else nnearest is set to 1 thereby searching for the
        first nearest neighbour)
    maxmatches : int, optional
        Maximum number of matches to find. If maxmatches > 0, the code finds 
        all matches up to maxmatches satisfying matchrad. If maxmatches = 0, all
        matches upto matchrad are found. And nnearest is ignored. Default = -1 
        (nnearest, if positive, is used instead of maxmatches. if nnearest is 0, 
        then nnearest is set to 1 and the first nearest neighbour is searched.)

    Returns
    -------
    m1 : int array
        Indices into the first catalog of the matches. 
    m2 : int array
        Indices into the second catalog of the matches. 
    d12 : float array
        Distance (in degrees) between the matches 
    -----------------------------------------------------------------------------
    """

    try:
        lon1, lat1
    except NameError:
        raise NameError('lon1 and/or lat1 not defined. Aborting spherematch()')
    
    if not isinstance(lon1, (list, NP.ndarray)):
        lon1 = NP.asarray(lon1)
    if not isinstance(lat1, (list, NP.ndarray)):
        lat1 = NP.asarray(lat1)

    try:
        nnearest = int(nnearest)
    except TypeError:
        raise TypeError('nnearest should be a non-negative integer.')

    try:
        maxmatches = int(maxmatches)
    except TypeError:
        raise TypeError('maxmatches should be a non-negative integer.')

    if matchrad is None:
        if maxmatches > 0:
            nnearest = 0
            warnings.warn('No matchrad specified. Will determine all the {0} nearest neighbours.'.format(maxmatches))
        else:
            maxmatches = -1
            if nnearest <= 0:
                nnearest = 1
            warnings.warn('No matchrad specified. Will determine the nearest neighbour # {0}.'.format(nnearest))
    elif not isinstance(matchrad, (int,float)):
        raise TypeError('matchrad should be a scalar number.')
    elif matchrad > 0.0:
        matchrad_cartesian = 2.0*NP.sin(0.5*matchrad*NP.pi/180.0)
        if maxmatches >= 0:
            nnearest = 0
        else:
            if nnearest <= 0:
                nnearest = 1
            warnings.warn('maxmatches is negative. Will determine the nearest neighbour # {0}.'.format(nnearest))
    else:
        raise ValueError('matchrad is not positive.')

    self_match = False
    if (lon2 is None) and (lat2 is None):
        self_match = True
    elif lon2 is None:
        lon2 = lon1
    elif lat2 is None:
        lat2 = lat2

    if lon1.shape != lat1.shape:
        raise ValueError('lon1 and lat1 should be of same length')
    if not self_match:
        if lon2.shape != lat2.shape:
            raise ValueError('lon2 and lat2 should be of same length')
    
    if lon1.size == 1:
        lon1 = NP.asarray(lon1).reshape(1)
        lat1 = NP.asarray(lat1).reshape(1)

    if lon2.size == 1:
        lon2 = NP.asarray(lon2).reshape(1)
        lat2 = NP.asarray(lat2).reshape(1)

    x1, y1, z1 = sph2xyz(lon1.ravel(), lat1.ravel())

    # this is equivalent to, but faster than just doing NP.array([x1, y1, z1])
    coords1 = NP.empty((x1.size, 3))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
    coords1[:, 2] = z1

    if (lon2 is not None) and (lat2 is not None):
        x2, y2, z2 = sph2xyz(lon2.ravel(), lat2.ravel())

        # this is equivalent to, but faster than just doing NP.array([x1, y1, z1])
        coords2 = NP.empty((x2.size, 3))
        coords2[:, 0] = x2
        coords2[:, 1] = y2
        coords2[:, 2] = z2
    
    if maxmatches == 0:
        kdt1 = KDT(coords1)
        if not self_match:
            kdt2 = KDT(coords2)
            ngbr_of_first_in_second = kdt1.query_ball_tree(kdt2, matchrad_cartesian)
            m1 = [i for i in range(len(ngbr_of_first_in_second)) for j in ngbr_of_first_in_second[i] if ngbr_of_first_in_second[i] != []]
            m2 = [j for i in range(len(ngbr_of_first_in_second)) for j in ngbr_of_first_in_second[i] if ngbr_of_first_in_second[i] != []]
            d12 = sphdist(lon1[m1], lat1[m1], lon2[m2], lat2[m2])
        else:
            ngbr_of_first_in_itself = kdt1.query_ball_tree(kdt1, matchrad_cartesian)
            m1 = [i for i in range(len(ngbr_of_first_in_itself)) for j in ngbr_of_first_in_itself[i] if ngbr_of_first_in_itself[i] != [] and i != j]
            m2 = [j for i in range(len(ngbr_of_first_in_itself)) for j in ngbr_of_first_in_itself[i] if ngbr_of_first_in_itself[i] != [] and i != j]
            d12 = sphdist(lon1[m1], lat1[m1], lon1[m2], lat1[m2])
    else:
        if not self_match:
            kdt2 = KDT(coords2)
            if matchrad is None:
                m2 = kdt2.query(coords1, max(maxmatches,nnearest))[1]
            else:
                m2 = kdt2.query(coords1, max(maxmatches,nnearest), distance_upper_bound=matchrad_cartesian)[1]
        else:
            kdt1 = KDT(coords1)
            if matchrad is None:
                m2 = kdt1.query(coords1, max(maxmatches,nnearest)+1)[1]
            else:
                m2 = kdt2.query(coords1, max(maxmatches,nnearest)+1, distance_upper_bound=matchrad_cartesian)[1]

        if nnearest > 0:
            m1 = NP.arange(lon1.size)
            if nnearest > 1:
                m2 = m2[:,-1]
        else: 
            m1 = NP.repeat(NP.arange(lon1.size).reshape(lon1.size,1), maxmatches, axis=1).flatten()
            if maxmatches > 1:
                m2 = m2[:,-maxmatches:].flatten() # Extract the last maxmatches columns

        if not self_match:
            msk = m2 < lon2.size
        else:
            msk = m1 < lon1.size
        m1 = m1[msk]
        m2 = m2[msk]

        if not self_match:
            d12 = sphdist(lon1[m1], lat1[m1], lon2[m2], lat2[m2])
        else:
            d12 = sphdist(lon1[m1], lat1[m1], lon1[m2], lat1[m2])

        if matchrad is not None:
            msk = d12 <= matchrad
            m1 = m1[msk]
            m2 = m2[msk]
            d12 = d12[msk]

    return m1, m2, d12        

#################################################################################

def parabola_parameters(dia=None, f_to_dia_ratio=None, f=None, depth=None):

    """
    -----------------------------------------------------------------------------
    Compute parabola parameters given specific input parameters

    Inputs:

    Two and only two of the following inputs must be given

    dia         [scalar or numpy array] Width of the parabola 

    f_to_dia_ratio
                [scalar or numpy array] ratio of focal distance to width

    f           [scalar or numpy array] focal distance

    depth       [scalar or numpy array] depth of the parabola

    Output:

    Dictionary containing the following keys and values:
    'f'         [numpy array] focal distance
    'D'         [numpy array] width of the parabola
    'h'         [numpy array] depth of the parabola
    'f/D'       [numpy array] ratio of focal distance to diameter
    'angle'     [scalar or numpy array] Opening angle between the edge and the 
                axis of the parabola (in degrees)
    -----------------------------------------------------------------------------
    """

    dia_specified = dia is not None
    f_to_dia_ratio_specified = f_to_dia_ratio is not None
    f_specified = f is not None
    depth_specified = depth is not None
    dia_specified = NP.asarray(dia_specified, dtype=int)
    f_to_dia_ratio_specified = NP.asarray(f_to_dia_ratio_specified, dtype=int)
    f_specified = NP.asarray(f_specified, dtype=int)
    depth_specified = NP.asarray(depth_specified, dtype=int)
    num_keywords = f_to_dia_ratio_specified + f_specified + depth_specified + dia_specified
    if dia_specified:
        dia = NP.asarray(dia).reshape(-1)
        if NP.any(dia <= 0.0):
            raise ValueError('Parabola diameter must be positive')
    if f_to_dia_ratio_specified:
        f_to_dia_ratio = NP.asarray(f_to_dia_ratio).reshape(-1)
        if NP.any(f_to_dia_ratio <= 0.0):
            raise ValueError('Parabola f/D ratio must be positive')
    if f_specified:
        f = NP.asarray(f).reshape(-1)
        if NP.any(f <= 0.0):
            raise ValueError('Parabola focal length must be positive')
    if depth_specified:
        depth = NP.asarray(depth).reshape(-1)
        if NP.any(depth <= 0.0):
            raise ValueError('Parabola depth must be positive')
    
    parms = {'f': None, 'D': None, 'f/D': None, 'h': None, 'angle': None}
    if num_keywords == 2:
        if f_specified:
            if dia_specified:
                f_to_dia_ratio = f / dia
                depth = dia**2 / (16.0 * f)
            if f_to_dia_ratio_specified:
                dia = f / f_to_dia_ratio
                depth = dia**2 / (16.0 * f)
            if depth_specified:
                dia = NP.sqrt(16.0 * f * depth)
                f_to_dia_ratio = f / dia
        elif dia_specified:
            if f_to_dia_ratio_specified:
                f = dia * f_to_dia_ratio
                depth = dia**2 / (16.0 * f)
            if depth_specified:
                f = dia**2 / (16.0 * depth)
                f_to_dia_ratio = f / dia
        elif depth_specified:
            if f_to_dia_ratio_specified:
                f = 16.0 * depth * f_to_dia_ratio**2
                dia = NP.sqrt(16.0 * f * depth)
        else:
            raise NameError('Insufficient parameters specified')
        parms['f'] = f
        parms['D'] = dia
        parms['f/D'] = f_to_dia_ratio
        parms['h'] = depth
        parms['angle'] = 2 * NP.degrees(NP.arctan2(0.5*dia, f-depth))
    else:
        raise ValueError('Either too little or too many parameters specified for unique determination of parabola parameters')
    
    return parms

################################################################################

def sample_parabola(f, open_angle, wavelength=1.0, axis=90.0, angunits='degrees'):

    """
    -----------------------------------------------------------------------------
    Sample points on a parabola defined by focal length and opening angle

    Inputs:

    f       [scalar] focal length, must be positive

    open_angle
            [scalar] opening angle (in units specified by input angunits) which
            is defined as the angle the edge of the parabola measured from the
            vertex of the parabola

    axis    [scalar] Angle the principal axis makes with the horizon (x-axis)
            measured counter-clockwise towards the z-axis (zenith). Default=90 
            degrees implies it is along the zenith

    wavelength
            [scalar] Wavelength to calculate the sampling interval. At the 
            moment the sampling interval is fixed at one-tenth of whichever 
            is minimum between the wavelength and diameter of the parabola
            
    angunits
            [string] Units of the angles specified in open_angle and axis. By
            default, it is set to 'degrees'

    Output:

    Mx3 array of x, y, z positions in same units as input focal length. The 
    y-values are zeros.
    -----------------------------------------------------------------------------
    """

    try:
        f, open_angle
    except NameError:
        raise NameError('Inputs focal length and opening angle must be specified')

    if not isinstance(f, (int,float)):
        raise TypeError('Input f must be a scalar')
    if f <= 0.0:
        raise ValueError('Input focal length must be positive')

    if not isinstance(wavelength, (int,float)):
        raise TypeError('Input wavelength must be a scalar')
    if wavelength <= 0.0:
        raise ValueError('Input wavelength must be positive')

    if not isinstance(open_angle, (int,float)):
        raise TypeError('Input open_angle must be a scalar')
    if open_angle <= 0.0:
        raise ValueError('Input opening angle must be positive')

    if not isinstance(axis, (int,float)):
        raise TypeError('Input axis must be a scalar')

    if angunits == 'degrees':
        open_angle = NP.radians(open_angle)
        axis = NP.radians(axis)
    tilt = 0.5 * NP.pi - axis

    theta_min = NP.pi - open_angle
    theta_max = NP.pi + open_angle

    rmax = 2.0 * f / (1.0 - NP.cos(theta_min))
    dia = 2.0 * rmax * NP.sin(theta_min)
    dx = 0.1 * min([dia, wavelength])
    dtheta = dx / rmax
    numsamples = NP.ceil(2*open_angle/dtheta).astype(int)
    theta = NP.linspace(theta_min-tilt, theta_max-tilt, num=numsamples)
    r = 2.0 * f / (1.0 - NP.cos(theta+tilt))
    x = r * NP.cos(NP.pi/2+theta)
    z = r * NP.sin(NP.pi/2+theta)

    xyz = NP.hstack((x.reshape(-1,1), NP.zeros((x.size,1), dtype=float), z.reshape(-1,1)))
    return xyz
    
################################################################################

def sample_paraboloid(f, open_angle, wavelength=1.0, axis=[90.0,270.0], angunits='degrees'):

    """
    -----------------------------------------------------------------------------
    Sample points on a parabola defined by focal length and opening angle

    Inputs:

    f       [scalar] focal length, must be positive

    open_angle
            [scalar] opening angle (in units specified by input angunits) which
            is defined as the angle the edge of the parabola measured from the
            vertex of the parabola

    axis    [list or numpy array] Angles (alta, az) the principal axis makes 
            relative to zenith. Default=[90, 270] degrees implies it is along 
            the z-axis (zenith)

    wavelength
            [scalar] Wavelength to calculate the sampling interval. At the 
            moment the sampling interval is fixed at one-tenth of whichever 
            is minimum between the wavelength and diameter of the parabola
            
    angunits
            [string] Units of the angles specified in open_angle and axis. By
            default, it is set to 'degrees'

    Output:

    Mx3 array of x, y, z positions in same units as input focal length. The 
    y-values are zeros.
    -----------------------------------------------------------------------------
    """

    try:
        f, open_angle
    except NameError:
        raise NameError('Inputs focal length and opening angle must be specified')

    if not isinstance(f, (int,float)):
        raise TypeError('Input f must be a scalar')
    if f <= 0.0:
        raise ValueError('Input focal length must be positive')

    if not isinstance(wavelength, (int,float)):
        raise TypeError('Input wavelength must be a scalar')
    if wavelength <= 0.0:
        raise ValueError('Input wavelength must be positive')

    if not isinstance(open_angle, (int,float)):
        raise TypeError('Input open_angle must be a scalar')
    if open_angle <= 0.0:
        raise ValueError('Input opening angle must be positive')

    if not isinstance(axis, (list,NP.ndarray)):
        raise TypeError('Input axis must be a list or numpy array')
    axis = NP.asarray(axis).reshape(-1)
    if axis.size != 2:
        raise ValueError('Input axis must be a 2-element array')

    if angunits == 'degrees':
        open_angle = NP.radians(open_angle)
        axis = NP.radians(axis)
    if NP.abs(axis[0]) > NP.pi/2:
        raise ValueError('Absolute value of altitude must be less than 90 degrees')
    axis[1] = NP.fmod(axis[1], 2*NP.pi)
    if axis[1] < 0.0:
        axis[1] += 2*NP.pi

    if axis[1] <= NP.pi:
        tilt = 0.5 * NP.pi - axis[0]
    else:
        tilt = 0.5 * NP.pi + axis[0]

    theta_min = NP.pi - open_angle
    theta_max = NP.pi + open_angle

    rmax = 2.0 * f / (1.0 - NP.cos(theta_min))
    dia = 2.0 * rmax * NP.sin(theta_min)
    dx = 0.1 * min([dia, wavelength])
    dtheta = dx / rmax
    nside = 2
    hpx_angres = HP.nside2resol(nside)
    while hpx_angres > dtheta:
        nside *= 2
        hpx_angres = HP.nside2resol(nside)
    theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
    select_ind = (theta >= theta_min) & (theta <= theta_max)
    theta = theta[select_ind]
    phi = phi[select_ind]
    numsamples = theta.size
    r = 2.0 * f / (1.0 - NP.cos(theta+tilt))
    z = r * NP.cos(theta)
    rho = r * NP.sin(theta)
    x = rho * NP.cos(phi)
    y = rho * NP.sin(phi)
    xyz = NP.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

    return xyz
    
################################################################################

# Below are some triangle solving routines 
# Source: https://github.com/sbyrnes321/trianglesolver
# These allow numpy array operations

# Following the usual convention, lower-case letters are side lengths and
# capital letters are angles. Corresponding letters are opposite each other,
# e.g. side b is opposite angle B. All angles are assumed to be in radians

################################################################################

def aaas(D, E, F, f):

    """ 
    ---------------------------------------------------------------------------
    This function solves the triangle and returns (d,e,f,D,E,F) 
    ---------------------------------------------------------------------------
    """
    
    d = f * NP.sin(D) / NP.sin(F)
    e = f * NP.sin(E) / NP.sin(F)
    return (d,e,f,D,E,F)

###############################################################################

def sss(d,e,f):
    
    """ 
    ---------------------------------------------------------------------------
    This function solves the triangle and returns (d,e,f,D,E,F) 
    ---------------------------------------------------------------------------
    """

    # assert d + e > f and e + f > d and f + d > e
    if NP.any(d+e <= f) or NP.any(e+f <= d) or NP.any(f+d <= e):
        raise ValueError('Invalid sides specified for triangle')
    F = NP.arccos((d**2 + e**2 - f**2) / (2 * d * e))
    E = NP.arccos((d**2 + f**2 - e**2) / (2 * d * f))
    D = NP.pi - F - E
    return (d,e,f,D,E,F)

###############################################################################

def sas(d,e,F):

    """ 
    ---------------------------------------------------------------------------
    This function solves the triangle and returns (d,e,f,D,E,F) 
    ---------------------------------------------------------------------------
    """

    f = NP.sqrt(d**2 + e**2 - 2 * d * e * NP.cos(F))
    return sss(d,e,f)

###############################################################################

def ssa(d, e, D, ssa_flag):

    """ 
    ---------------------------------------------------------------------------
    This function solves the triangle and returns (d,e,f,D,E,F) 
    ---------------------------------------------------------------------------
    """

    sinE = NP.sin(D) * e / d
    if NP.abs(sinE - 1) < 100 * epsilon:
        # Right triangle, where the solution is unique
        E = NP.pi/2
    elif sinE > 1:
        raise ValueError('No such triangle')
    elif ssa_flag == 'forbid':
        raise ValueError('Two different triangles fit this description')
    else:
        E = NP.arcsin(sinE)
        if ssa_flag == 'obtuse':
            E = NP.pi - E
    F = NP.pi - D - E
    e,f,d,E,F,D = aaas(E,F,D,d)
    return (d,e,f,D,E,F)

################################################################################

def trisolve(a=None, b=None, c=None, A=None, B=None, C=None, ssa_flag='forbid'):

    """
    ---------------------------------------------------------------------------
    Solve to find all the information about a triangle, given partial
    information.
    
    a, b, c, A, B, and C are the three sides and angles. (e.g. A is the angle
    opposite the side of length a.) Out of these six possibilities, you need 
    to tell the program exactly three. Then the program will tell you all six.
    
    It returns a tuple (a, b, c, A, B, C).
    
    "ssa" is the situation when you give two sides and an angle which is not
    between them. This is usually not enough information to specify a unique
    triangle. (Except in one special case involving right triangles.) Instead
    there are usually two possibilities.
    
    Therefore there is an 'ssa_flag'. You can set it to'forbid' (raise an error
    if the answer is not unique - the default setting), or 'acute' (where the
    unknown angle across from the known side is chosen to be acute) or 'obtuse'
    (similarly).
    ---------------------------------------------------------------------------
    """

    if sum(x is not None for x in [a,b,c,A,B,C]) != 3:
        raise ValueError('Must provide exactly 3 inputs')
    if sum(x is None for x in [a,b,c]) == 3:
        raise ValueError('Must provide at least 1 side length')
    assert all(NP.all(x > 0) for x in [a,b,c,A,B,C] if x is not None)
    assert all(NP.all(x < NP.pi) for x in [A,B,C] if x is not None)
    assert ssa_flag in ['forbid', 'acute', 'obtuse']
    
    # If three sides are known...
    if sum(x is not None for x in [a,b,c]) == 3:
        a,b,c,A,B,C = sss(a,b,c)
        return (a,b,c,A,B,C)

    # If two sides and one angle are known...
    if sum(x is not None for x in [a,b,c]) == 2:
        # ssa case
        if all(x is not None for x in [a, A, b]):
            a,b,c,A,B,C = ssa(a, b, A, ssa_flag)
        elif all(x is not None for x in [a, A, c]):
            a,c,b,A,C,B = ssa(a, c, A, ssa_flag)
        elif all(x is not None for x in [b, B, a]):
            b,a,c,B,A,C = ssa(b, a, B, ssa_flag)
        elif all(x is not None for x in [b, B, c]):
            b,c,a,B,C,A = ssa(b, c, B, ssa_flag)
        elif all(x is not None for x in [c, C, a]):
            c,a,b,C,A,B = ssa(c, a, C, ssa_flag)
        elif all(x is not None for x in [c, C, b]):
            c,b,a,C,B,A = ssa(c, b, C, ssa_flag)
        
        # sas case
        elif all(x is not None for x in [a, b, C]):
            a,b,c,A,B,C = sas(a, b, C)
        elif all(x is not None for x in [b, c, A]):
            b,c,a,B,C,A = sas(b, c, A)
        elif all(x is not None for x in [c, a, B]):
            c,a,b,C,A,B = sas(c, a, B)
        else:
            raise ValueError('Oops, this code should never run')
        return (a,b,c,A,B,C)
    
    # If one side and two angles are known...
    if sum(x is not None for x in [a,b,c]) == 1:
        # Find the third angle...
        if A is None:
            A = NP.pi - B - C
        elif B is None:
            B = NP.pi - A - C
        else:
            C = NP.pi - A - B
        assert NP.all(A > 0) and NP.all(B > 0) and NP.all(C > 0)
        # Then solve the triangle...
        if c is not None:
            a,b,c,A,B,C = aaas(A,B,C,c)
        elif a is not None:
            b,c,a,B,C,A = aaas(B,C,A,a)
        else:
            c,a,b,C,A,B = aaas(C,A,B,b)
        return (a,b,c,A,B,C)
    raise ValueError('Oops, this code should never run')

################################################################################

# End of triangle solving routines

################################################################################
