import pytest

altaz = [([90.0, 270.0]), ([0.0, 90.0]), ([-90.0, 270.0]), ([0.0, -90.0]), ([0.0, 180.0]), ([0.0, 0.0]), ([90.0, 270.0]), ]
dircos = [([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0]), ([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0]), ([0.0, -1.0, 0.0]), ([0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0])]
latitude = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
hadec = [(180.0, -90.0), (-90.0, 0.0), (180.0, 30.0), (90.0, 0.0), (0.0, -60.0), (180.0, 30.0), (0.0, 90.0)]

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


