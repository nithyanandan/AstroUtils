import pytest

altaz_vals = [([90.0, 270.0]), ([0.0, 90.0]), ([-90.0, 270.0]), ([0.0, -90.0]), ([0.0, 180.0]), ([0.0, 0.0])]
dircos_vals = [([0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0]), ([0.0, 0.0, -1.0]), ([-1.0, 0.0, 0.0]), ([0.0, -1.0, 0.0]), ([0.0, 1.0, 0.0])]

@pytest.fixture(scope='module', params=zip(altaz_vals, dircos_vals))
def altaz_to_dircos(request):
    return request.param

@pytest.fixture(scope='module', params=zip(dircos_vals, altaz_vals))
def dircos_to_altaz(request):
    return request.param


