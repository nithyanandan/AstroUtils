import pytest
import numpy as NP

############# Fixtures for test_geometry.py ##############

altaz = [([90.0, 270.0]), ([0.0, 90.0]), ([-90.0, 270.0]), ([0.0, -90.0]), ([0.0, 180.0]), ([0.0, 0.0]), ([90.0, 270.0])]
dircos = [([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0]), ([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0]), ([0.0, -1.0, 0.0]), ([0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0])]
hadec = [(180.0, -90.0), (-90.0, 0.0), (180.0, 30.0), (90.0, 0.0), (0.0, -60.0), (180.0, 30.0), (0.0, 90.0)]

latitude = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]

@pytest.fixture(scope='module', params=zip(altaz, dircos))
def altaz_to_dircos(request):
    return request.param

@pytest.fixture(scope='module', params=zip(dircos, altaz))
def dircos_to_altaz(request):
    return request.param

@pytest.fixture(scope='module', params=zip(altaz, hadec, latitude))
def altaz_to_hadec(request):
    return request.param

@pytest.fixture(scope='module', params=zip(hadec, altaz, latitude))
def hadec_to_altaz(request):
    return request.param

enu = dircos
# x = dot(dircos, [0, -sin(lat), cos(lat)])
# y = dot(dircos, [1, 0, 0])
# z = dot(dircos, [0, cos(lat), sin(lat)])
x = [NP.dot(NP.asarray(dircos[i]), NP.asarray([0.0, -NP.sin(NP.radians(latitude[i])), NP.cos(NP.radians(latitude[i]))])) for i in range(len(latitude))]
y = [NP.dot(NP.asarray(dircos[i]), NP.asarray([1.0, 0.0, 0.0])) for i in range(len(latitude))]
z = [NP.dot(NP.asarray(dircos[i]), NP.asarray([0.0, NP.cos(NP.radians(latitude[i])), NP.sin(NP.radians(latitude[i]))])) for i in range(len(latitude))]
xyz = zip(x,y,z)

r = NP.sqrt(NP.asarray(x)**2 + NP.asarray(y)**2 + NP.asarray(z)**2)
latang = NP.pi/2 - NP.arccos(NP.asarray(z)/r)
lonang = NP.pi/2 - NP.arctan2(NP.asarray(y), NP.asarray(x))

r = r.tolist()
latang = latang.tolist()
lonang = lonang.tolist()

r_lat_lon = zip(r, latang, lonang)

@pytest.fixture(scope='module', params=zip(enu, xyz, latitude))
def enu_to_xyz(request):
    return request.param

@pytest.fixture(scope='module', params=zip(xyz, enu, latitude))
def xyz_to_enu(request):
    return request.param

@pytest.fixture(scope='module', params=zip(xyz, r_lat_lon))
def xyz_to_sph(request):
    return request.param

@pytest.fixture(scope='module', params=zip(r_lat_lon, xyz))
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

