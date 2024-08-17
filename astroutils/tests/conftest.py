from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import zip
from builtins import range
import pytest
import numpy as NP

############# Fixtures for test_geometry.py ##############

points1 = [([1.0, 1.0]),
           ([1.0, -2.0]),
           ([2.0, 3.0]),
           ([1.0, -2.0])]
points2 = [([-1.0, 1.0]),
           ([1.0, 1.0]),
           ([-1.0, 0.0]),
           ([[-1.0, 1.0],
             [1.0, 1.0],
             [-1.0, 0.0]])]
slopes = [0.0, NP.inf, 1.0, [-1.5, NP.inf, -1.0]]
eqns = [([0.0, 1.0, 1.0]),
        ([1.0, 0.0, 1.0]),
        ([-1.0, 1.0, 1.0]),
        ([[1.5, 1.0, -0.5],
          [1.0, 0.0, 1.0],
          [1.0, 1.0, -1.0]])]

@pytest.fixture(scope='module', params=list(zip(points1, slopes, eqns)))
def generate_line_from_point_and_slope(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(points1, points2, eqns)))
def generate_line_from_two_points(request):
    return request.param

coeffs = [([[0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0]])]
dvect = [([0.0,
           0.0,
           1.0,
           1.0])]
pts = [([[[NP.nan, NP.nan],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan]],
         [[0.0, 0.0],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan]],
         [[1.0, 0.0],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan],
          [NP.nan, NP.nan]],
         [[-1.0, 0.0],
          [0.0, 1.0],
          [1.0, 2.0],
          [NP.nan, NP.nan]]])]

displacements = [([1.0, 3.0, 2.0, 1.0])]

dvects_from_displacement = [([[-1.0, 1.0], [-3.0, 3.0], [-1.0, 3.0], [1.0-NP.sqrt(2.0), 1.0+NP.sqrt(2.0)]])]

@pytest.fixture(scope='module', params=list(zip(coeffs, dvect, pts)))
def points_from_line2d_intersection(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(coeffs, dvect, displacements, dvects_from_displacement)))
def generate_parallel_lines_at_distance_from_line(request):
    return request.param

coeffs2 = [[0.0, 1.0],
           [1.0, 0.0],
           [1.0, 0.0],
           [-1.0, 1.0]]
dvect2 = [0.0,
          0.0,
          1.0,
          1.0]
xvals = [[NP.nan, 0.0, 1.0],
         [0.0, NP.nan],
         [0.0, NP.nan],
         [-1.0, 0.0, 1.0]]
expected_yvals = [[0.0, 0.0, 0.0],
                  [NP.nan, NP.nan],
                  [NP.nan, NP.nan],
                  [0.0, 1.0, 2.0]]
yvals = [[0.0, NP.nan],
         [NP.nan, 0.0, 1.0],
         [NP.nan, 0.0, NP.nan],
         [0.0, 1.0, 2.0]]
expected_xvals = [[NP.nan, NP.nan],
                  [0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0],
                  [-1.0, 0.0, 1.0]]

@pytest.fixture(scope='module', params=list(zip(coeffs2, dvect2, xvals, expected_yvals)))
def get_ordinate_from_abscissa_on_line(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(coeffs2, dvect2, yvals, expected_xvals)))
def get_abscissa_from_ordinate_on_line(request):
    return request.param

polygon_vertices_2D = [([[0.0, 2.0], [85.0, -25.0], [40.0, 25.0]]),
                       ([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]),
                       ([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0], [3.0, 4.0]]),
                       ([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                       ([[0.0, 0.0], [1.0, 0.0]]),
                       ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
polygon_areas_2D = [1517.5, 6.0, 12.0, 0.0, 0.0, NP.nan]

@pytest.fixture(scope='module', params=list(zip(polygon_vertices_2D, polygon_areas_2D)))
def polygonArea2D(request):
    return request.param

altaz = [([90.0, 270.0]), ([0.0, 90.0]), ([-90.0, 270.0]), ([0.0, -90.0]), ([0.0, 180.0]), ([0.0, 0.0]), ([90.0, 270.0])]
dircos = [([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0]), ([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0]), ([0.0, -1.0, 0.0]), ([0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0])]
hadec = [(180.0, -90.0), (-90.0, 0.0), (180.0, 30.0), (90.0, 0.0), (0.0, -60.0), (180.0, 30.0), (0.0, 90.0)]

latitude = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]

@pytest.fixture(scope='module', params=list(zip(altaz, dircos)))
def altaz_to_dircos(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(dircos, altaz)))
def dircos_to_altaz(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(altaz, hadec, latitude)))
def altaz_to_hadec(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(hadec, altaz, latitude)))
def hadec_to_altaz(request):
    return request.param

enu = dircos
# x = dot(dircos, [0, -sin(lat), cos(lat)])
# y = dot(dircos, [1, 0, 0])
# z = dot(dircos, [0, cos(lat), sin(lat)])
x = [NP.dot(NP.asarray(dircos[i]), NP.asarray([0.0, -NP.sin(NP.radians(latitude[i])), NP.cos(NP.radians(latitude[i]))])) for i in range(len(latitude))]
y = [NP.dot(NP.asarray(dircos[i]), NP.asarray([1.0, 0.0, 0.0])) for i in range(len(latitude))]
z = [NP.dot(NP.asarray(dircos[i]), NP.asarray([0.0, NP.cos(NP.radians(latitude[i])), NP.sin(NP.radians(latitude[i]))])) for i in range(len(latitude))]
xyz = list(zip(x,y,z))

r = NP.sqrt(NP.asarray(x)**2 + NP.asarray(y)**2 + NP.asarray(z)**2)
latang = NP.pi/2 - NP.arccos(NP.asarray(z)/r)
lonang = NP.pi/2 - NP.arctan2(NP.asarray(y), NP.asarray(x))

r = r.tolist()
latang = latang.tolist()
lonang = lonang.tolist()

r_lat_lon = list(zip(r, latang, lonang))

@pytest.fixture(scope='module', params=list(zip(enu, xyz, latitude)))
def enu_to_xyz(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(xyz, enu, latitude)))
def xyz_to_enu(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(xyz, r_lat_lon)))
def xyz_to_sph(request):
    return request.param

@pytest.fixture(scope='module', params=list(zip(r_lat_lon, xyz)))
def sph_to_xyz(request):
    return request.param

lonlat1_lonlat2 = [(lonlat1, lonlat2) for i,lonlat1 in enumerate(zip(lonang,latang)) for j,lonlat2 in enumerate(zip(lonang,latang)) if j>=i]
lonlat1_lonlat2 = NP.degrees(NP.asarray(lonlat1_lonlat2).reshape(len(lonlat1_lonlat2), -1))
lonlat1 = lonlat1_lonlat2[:,:2]
lonlat2 = lonlat1_lonlat2[:,-2:]

xyz1_xyz2 = [(xyz1, xyz2) for i,xyz1 in enumerate(xyz) for j,xyz2 in enumerate(xyz) if j>=i]
xyz1_xyz2 = NP.asarray(xyz1_xyz2).reshape(len(xyz1_xyz2), -1)
xyz1 = xyz1_xyz2[:,:3]
xyz2 = xyz1_xyz2[:,-3:]

diffang_cartesian = NP.degrees(NP.arccos(NP.sum(xyz1*xyz2, axis=1)))

@pytest.fixture(scope='module', params=[(lonlat1, lonlat2, diffang_cartesian)])
def sph_dist(request):
    return request.param

parabolaparms = {'D': NP.asarray([20.0]), 'f/D': NP.asarray([0.25]), 'f': NP.asarray([5.0]), 'h': NP.asarray([5.0]), 'angle': NP.asarray([180.0])}

@pytest.fixture(scope='module', params=[({'f/D': parabolaparms['f/D'], 'D': parabolaparms['D']}, parabolaparms), ({'f/D': parabolaparms['f/D'], 'f': parabolaparms['f']}, parabolaparms), ({'f/D': parabolaparms['f/D'], 'h': parabolaparms['h']}, parabolaparms), ({'D': parabolaparms['D'], 'f': parabolaparms['f']}, parabolaparms), ({'D': parabolaparms['D'], 'h': parabolaparms['h']}, parabolaparms), ({'f': parabolaparms['f'], 'h': parabolaparms['h']}, parabolaparms)])
def parabola_parms(request):
    return request.param

############# Fixtures for test_mathops.py ##############

@pytest.fixture
def nruns_shape():
    return (13,7)

@pytest.fixture
def positive_definite_hermitian_matrix(nruns_shape):
    m = 3
    A = NP.random.normal(size=nruns_shape+(m,m)) + 1j * NP.random.normal(size=nruns_shape+(m,m))
    return A @ NP.swapaxes(A.conj(),-2,-1)

@pytest.fixture
def positive_semidefinite_hermitian_matrix(positive_definite_hermitian_matrix):
    evals, evecs = NP.linalg.eigh(positive_definite_hermitian_matrix)
    evals[...,0] = 0 # Make the first eigenvalue zero in all runs
    m = evals.shape[-1]
    # Number of leading dimensions in in_matrix
    num_leading_dims = positive_definite_hermitian_matrix.ndim - 2 
    diag_evals = evals[...,NP.newaxis] * NP.eye(m).reshape((1,) * num_leading_dims + (m,m)) # Shape = (...,m,m)
    return evecs @ diag_evals @ NP.swapaxes(evecs.conj(),-2,-1)

@pytest.fixture
def non_hermitian_matrix(nruns_shape):
    m = 3
    return NP.random.normal(size=nruns_shape+(m,m)) + 1j * NP.random.normal(size=nruns_shape+(m,m))
