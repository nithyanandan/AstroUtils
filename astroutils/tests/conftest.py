from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import zip
from builtins import range
import pytest
import numpy as NP

############# Fixtures for test_geometry.py ##############

# Inputs to Point.__init__() and the (x,y,z) tuple each should produce
point_init_inputs   = [None, 5, 5.0, [], [1.0], [1.0,2.0], [1.0,2.0,3.0], [1.0,2.0,3.0,4.0], (1.0,2.0,3.0), NP.asarray([1.0,2.0,3.0])]
point_init_expected = [(0.0,0.0,0.0), (5.0,0.0,0.0), (5.0,0.0,0.0), (0.0,0.0,0.0), (1.0,0.0,0.0), (1.0,2.0,0.0), (1.0,2.0,3.0), (1.0,2.0,3.0), (1.0,2.0,3.0), (1.0,2.0,3.0)]

@pytest.fixture(scope='module', params=list(zip(point_init_inputs, point_init_expected)))
def point_init_case(request):
    return request.param

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

# Orders and their known (deterministic) permutations returned by
# costas_array() -- n=2,4,6,10,12 are produced by the Welch construction
# (n+1 prime); n=0,1,3,5,7,8 fall back to the backtracking search.
costas_orders          = [0,  1,   2,      3,        4,          5,             6,               7,                  8,                     10,                       12]
costas_expected_perms  = [[], [0], [1,0], [0,2,1], [1,3,2,0], [0,2,3,1,4], [2,1,5,3,4,0], [0,1,5,3,6,2,4], [0,1,4,6,5,3,7,2], [1,3,7,4,9,8,6,2,5,0], [1,3,7,2,5,11,10,8,4,9,6,0]]

@pytest.fixture(scope='module', params=list(zip(costas_orders, costas_expected_perms)))
def costas_array_order(request):
    return request.param

# Orders n for which n+1 is prime, so the Welch construction applies
costas_welch_valid_orders = [2, 4, 6, 10, 12]

@pytest.fixture(scope='module', params=costas_welch_valid_orders)
def costas_welch_valid_order(request):
    return request.param

# Orders n for which n+1 is not prime, so the Welch construction must fail
costas_welch_invalid_orders = [5, 7, 8]

@pytest.fixture(scope='module', params=costas_welch_invalid_orders)
def costas_welch_invalid_order(request):
    return request.param

# Case 1: reference point at (lat=0, lon=0, alt=0). At this point the local
# ENU basis vectors align exactly with the ECEF axes: local Up is along ECEF
# X, local East along ECEF Y, and local North along ECEF Z. This gives a
# hand-derivable check independent of lla2ecef().
ecef_enu_ref_xyz_eq = NP.array([6378137.0, 0.0, 0.0]) # ECEF pos of lat=lon=alt=0 (WGS84 semi-major axis)
ecef_enu_ref_info_eq = {'xyz': ecef_enu_ref_xyz_eq.copy(), 'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
ecef_enu_offset_eq = NP.array([100.0, 200.0, 300.0])      # (dx,dy,dz) offset from reference in ECEF
ecef_enu_expected_enu_eq = NP.array([200.0, 300.0, 100.0]) # (east,north,up) = (dy,dz,dx) at lat=lon=0

@pytest.fixture(scope='module')
def ecef2enu_equatorial_case():
    return (ecef_enu_ref_xyz_eq.copy(), dict(ecef_enu_ref_info_eq), ecef_enu_offset_eq.copy(), ecef_enu_expected_enu_eq.copy())

# Case 2: a general mid-latitude reference point, used for round-trip checks
# between ecef2enu() and enu2ecef(). ref_xyz below is the ECEF position of
# ref_latlonalt as returned by lla2ecef() (WGS84).
ecef_enu_ref_latlonalt = (34.5, -118.2, 500.0) # degrees, degrees, metres
ecef_enu_ref_xyz_general = NP.array([-2486777.318714501, -4637820.0998305855, 3592575.143868436])
ecef_enu_offset_general = NP.array([50.0, -75.0, 120.0])

@pytest.fixture(scope='module')
def ecef2enu_general_case():
    return (ecef_enu_ref_latlonalt, ecef_enu_ref_xyz_general.copy(), ecef_enu_offset_general.copy())

# Known (lat,lon,alt) <-> (x,y,z) pairs for lla2ecef()/ecef2lla(), all in
# degrees/metres: equator/prime-meridian, north pole, south pole, equator at
# lon=90, and a general mid-latitude point (shared with the ECEF/ENU fixture
# above for consistency).
lla_known_lat = [0.0,        90.0,        -90.0,        0.0,        34.5]
lla_known_lon = [0.0,        0.0,         0.0,          90.0,       -118.2]
lla_known_alt = [0.0,        0.0,         0.0,          0.0,        500.0]
lla_known_xyz = [(6378137.0, 0.0, 0.0),
                 (0.0, 0.0, 6356752.314245179),
                 (0.0, 0.0, -6356752.314245179),
                 (0.0, 6378137.0, 0.0),
                 (-2486777.318714501, -4637820.0998305855, 3592575.143868436)]

@pytest.fixture(scope='module', params=list(zip(lla_known_lat, lla_known_lon, lla_known_alt, lla_known_xyz)))
def lla_ecef_known_point(request):
    return request.param

# A batch of (lat,lon,alt) points -- including both poles -- used to check
# lla2ecef()/ecef2lla() round-trip consistency
lla_roundtrip_lat = NP.array([-90.0, -60.0, -30.0, 0.0, 15.0, 45.0, 60.0, 80.0, 90.0])
lla_roundtrip_lon = NP.array([  0.0, -170.0, -45.0, 0.0, 30.0, 90.0, 135.0, 179.0, 45.0])
lla_roundtrip_alt = NP.array([  0.0,    0.0, 100.0, 500.0, -20.0, 1000.0, 2000.0, 50.0, 10.0])

@pytest.fixture(scope='module')
def lla_ecef_roundtrip_points():
    return (lla_roundtrip_lat.copy(), lla_roundtrip_lon.copy(), lla_roundtrip_alt.copy())

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
    evals[...,-1] = 0 # Make the largest eigenvalue zero in all runs
    m = evals.shape[-1]
    # Number of leading dimensions in in_matrix
    num_leading_dims = positive_definite_hermitian_matrix.ndim - 2 
    diag_evals = evals[...,NP.newaxis] * NP.eye(m).reshape((1,) * num_leading_dims + (m,m)) # Shape = (...,m,m)
    return evecs @ diag_evals @ NP.swapaxes(evecs.conj(),-2,-1)

@pytest.fixture
def non_hermitian_matrix(nruns_shape):
    m = 3
    return NP.random.normal(size=nruns_shape+(m,m)) + 1j * NP.random.normal(size=nruns_shape+(m,m))
