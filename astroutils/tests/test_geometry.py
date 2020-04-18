import pytest
from astroutils import geometry as GEOM
import numpy as NP

# @pytest.mark.parametrize('altaz, lmn', [([90.0, 270.0], [0.0, 0.0, 1.0]),
#                                         ([0.0, 90.0], [1.0, 0.0, 0.0]),
#                                         ([-90.0, 270.0], [0.0, 0.0, -1.0]),
#                                         ([0.0, -90.0], [-1.0, 0.0, 0.0]),
#                                         ([0.0, 180.0], [0.0, -1.0, 0.0]),
#                                         ([0.0, 0.0], [0.0, 1.0, 0.0])])
# def test_altaz2dircos(altaz, lmn):
#     alt_az = NP.asarray(altaz).reshape(1,-1)
#     l_m_n = NP.asarray(lmn).reshape(1,-1)
#     expected_dircos = GEOM.altaz2dircos(alt_az, units='degrees')
#     NP.testing.assert_allclose(l_m_n, expected_dircos, atol=1e-12)

def test_altaz2dircos(altaz_to_dircos):
    altaz, expected_dircos = altaz_to_dircos # Read from fixture in conftest.py
    altaz = NP.asarray(altaz).reshape(1,-1)
    expected_dircos = NP.asarray(expected_dircos).reshape(-1)
    dircos = GEOM.altaz2dircos(altaz, units='degrees').ravel()
    NP.testing.assert_allclose(dircos, expected_dircos, atol=1e-12)

def test_dircos2altaz(dircos_to_altaz):
    dircos, altaz = dircos_to_altaz # Read from fixture in conftest.py
    altaz = NP.asarray(altaz).reshape(-1)
    dircos = NP.asarray(dircos).reshape(1,-1)
    expected_altaz = GEOM.dircos2altaz(dircos, units='degrees').ravel()
    if (NP.abs(expected_altaz[0]) - 90.0) <= 1e-10:
        NP.testing.assert_allclose(altaz[0], expected_altaz[0], atol=1e-12)
    else:
        NP.testing.assert_allclose(altaz, expected_altaz, atol=1e-12)

def test_altaz2hadec(altaz_to_hadec):
    altaz, hadec, latitude = altaz_to_hadec # Read from fixture in conftest.py
    hadec = NP.asarray(hadec).reshape(-1)
    altaz = NP.asarray(altaz).reshape(1,-1)
    expected_hadec = GEOM.altaz2hadec(altaz, latitude=latitude, units='degrees').ravel()
    if NP.abs(NP.abs(expected_hadec[0] - hadec[0]) - 360.0) <= 1e-10:
        expected_hadec = NP.asarray([hadec[0], expected_hadec[1]])
    NP.testing.assert_allclose(hadec, expected_hadec, atol=1e-12)

def test_hadec2altaz(hadec_to_altaz):
    hadec, altaz, latitude = hadec_to_altaz # Read from fixture in conftest.py
    hadec = NP.asarray(hadec).reshape(1,-1)
    altaz = NP.asarray(altaz).reshape(-1)
    expected_altaz = GEOM.hadec2altaz(hadec, latitude=latitude, units='degrees').ravel()
    if NP.abs(NP.abs(expected_altaz[1] - altaz[1]) - 360.0) <= 1e-10:
        expected_altaz = NP.asarray([expected_altaz[0], altaz[1]])
    NP.testing.assert_allclose(altaz, expected_altaz, atol=1e-12)

def test_enu2xyz(enu_to_xyz):
    enu, xyz, latitude = enu_to_xyz # Read from fixture in conftest.py
    xyz = NP.asarray(xyz).reshape(-1)
    enu = NP.asarray(enu).reshape(1,-1)
    expected_xyz = GEOM.enu2xyz(enu, latitude, units='degrees').ravel()
    NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)

def test_xyz2enu(xyz_to_enu):
    xyz, enu, latitude = xyz_to_enu # Read from fixture in conftest.py
    xyz = NP.asarray(xyz).reshape(1,-1)
    enu = NP.asarray(enu).reshape(-1)
    expected_enu = GEOM.xyz2enu(xyz, latitude, units='degrees').ravel()
    NP.testing.assert_allclose(enu, expected_enu, atol=1e-12)

# def test_sph2xyz():
#     lon = NP.asarray([0.0, 45.0])
#     lat = NP.asarray([30.0, 0.0])
#     expected_x = NP.cos(NP.radians(lat)) * NP.cos(NP.radians(lon))
#     expected_y = NP.cos(NP.radians(lat)) * NP.sin(NP.radians(lon))
#     expected_z = NP.sin(NP.radians(lat))
#     expected_xyz = NP.hstack((expected_x.reshape(-1,1), expected_y.reshape(-1,1), expected_z.reshape(-1,1)))
#     x, y, z = GEOM.sph2xyz(lon, lat)
#     xyz = NP.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
#     NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)

def test_xyz2sph(xyz_to_sph):
    xyz, r_lat_lon = xyz_to_sph # Read from fixture in conftest.py
    xyz = NP.asarray(xyz).reshape(-1)
    r_lat_lon = NP.asarray(r_lat_lon).reshape(-1)

    expected_r, expected_lat, expected_lon = GEOM.xyz2sph(xyz[0], xyz[1], xyz[2], units='radians')
    expected_r_lat_lon = NP.asarray([expected_r, expected_lat, expected_lon])
    NP.testing.assert_allclose(r_lat_lon, expected_r_lat_lon, atol=1e-12)

# def test_sph2xyz(sph_to_xyz):
#     r_lat_lon, xyz = sph_to_xyz # Read from fixture in conftest.py
#     xyz = NP.asarray(xyz).reshape(-1)
#     r_lat_lon = NP.asarray(r_lat_lon).reshape(-1)

#     expected_x, expected_y, expected_z = GEOM.sph2xyz(NP.degrees(r_lat_lon[2]), NP.degrees(r_lat_lon[1]), rad=r_lat_lon[0])
#     expected_xyz = NP.asarray([expected_x, expected_y, expected_z])
#     NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)
    
def test_sphdist():
    lon1 = NP.asarray([0.0, 45.0])
    lat1 = NP.asarray([30.0, 0.0])
    lon2 = NP.asarray([90.0, 0.0])
    lat2 = NP.asarray([90.0, 0.0])
    expected_dist = NP.asarray([60.0, 45.0])
    dist = GEOM.sphdist(lon1, lat1, lon2, lat2)
    NP.testing.assert_allclose(dist, expected_dist, atol=1e-12)

def test_spherematch_1match():
    lon1 = NP.asarray([0.0, 45.0])
    lat1 = NP.asarray([30.0, 0.0])
    lon2 = NP.asarray([5.0, 40.0, 0.0])
    lat2 = NP.asarray([35.0, 5.0, 21.0])
    matchrad = 10.0
    maxmatches = 1
    expected_m1 = NP.asarray([0, 1])
    expected_m2 = NP.asarray([0, 1])
    expected_d12 = NP.asarray([6.53868718, 7.06657439])
    m1, m2, d12 = GEOM.spherematch(lon1, lat1, lon2, lat2, matchrad=matchrad, maxmatches=maxmatches)
    m1 = NP.asarray(m1)
    m2 = NP.asarray(m2)
    d12 = NP.asarray(d12)
    NP.testing.assert_array_equal(m1, expected_m1)
    NP.testing.assert_array_equal(m2, expected_m2)
    NP.testing.assert_allclose(d12, expected_d12, atol=1e-12)

def test_spherematch_allmatch():
    lon1 = NP.asarray([0.0, 45.0])
    lat1 = NP.asarray([30.0, 0.0])
    lon2 = NP.asarray([5.0, 40.0, 0.0])
    lat2 = NP.asarray([35.0, 5.0, 21.0])
    matchrad = 10.0
    maxmatches = 0
    expected_m1 = NP.asarray([0, 0, 1])
    expected_m2 = NP.asarray([0, 2, 1])
    expected_d12 = NP.asarray([6.53868718, 9.0, 7.06657439])
    m1, m2, d12 = GEOM.spherematch(lon1, lat1, lon2, lat2, matchrad=matchrad, maxmatches=maxmatches)
    m1 = NP.asarray(m1)
    m2 = NP.asarray(m2)
    d12 = NP.asarray(d12)
    NP.testing.assert_array_equal(m1, expected_m1)
    NP.testing.assert_array_equal(m2, expected_m2)
    NP.testing.assert_allclose(d12, expected_d12, atol=1e-12)

    
