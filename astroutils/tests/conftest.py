import pytest
import numpy as NP

altaz = [([90.0, 270.0]), ([0.0, 90.0]), ([-90.0, 270.0]), ([0.0, -90.0]), ([0.0, 180.0]), ([0.0, 0.0]), ([90.0, 270.0])]
dircos = [([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0]), ([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0]), ([0.0, -1.0, 0.0]), ([0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0])]
hadec = [(180.0, -90.0), (-90.0, 0.0), (180.0, 30.0), (90.0, 0.0), (0.0, -60.0), (180.0, 30.0), (0.0, 90.0)]

latitude = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]

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


