from generalutils import geometry as GEOM
import numpy as NP

def test_altaz2dircos():
    altaz = NP.asarray([[90.0, 270.0], [0.0, 90.0]]).reshape(-1,2)
    dircos = NP.asarray([[0.0, 0.0, 1.0],[1.0, 0.0, 0.0]]).reshape(-1,3)
    expected_dircos = GEOM.altaz2dircos(altaz, units='degrees')
    NP.testing.assert_allclose(dircos, expected_dircos, atol=1e-12)

def test_dircos2altaz():
    dircos = NP.asarray([[NP.cos(NP.radians(60.0)), 0.0, NP.cos(NP.radians(30.0))],[1.0, 0.0, 0.0]]).reshape(-1,3)
    altaz = NP.asarray([[60.0, 90.0], [0.0, 90.0]]).reshape(-1,2)
    expected_altaz = GEOM.dircos2altaz(dircos, units='degrees')
    NP.testing.assert_allclose(altaz, expected_altaz, atol=1e-12)

def test_hadec2altaz():
    hadec = NP.asarray([[30.0, 0.0], [-90.0, 0.0]]).reshape(-1,2)
    latitude = 0.0
    expected_altaz = NP.asarray([[60.0, 270.0], [0.0, 90.0]]).reshape(-1,2)
    altaz = GEOM.hadec2altaz(hadec, latitude, units='degrees')
    NP.testing.assert_allclose(altaz, expected_altaz, atol=1e-12)

def test_altaz2hadec():
    altaz = NP.asarray([[60.0, 270.0], [0.0, 90.0]]).reshape(-1,2)
    expected_hadec = NP.asarray([[30.0, 0.0], [270.0, 0.0]]).reshape(-1,2)
    latitude = 0.0
    hadec = GEOM.altaz2hadec(altaz, latitude, units='degrees')
    NP.testing.assert_allclose(hadec, expected_hadec, atol=1e-12)
    
def test_enu2xyz():
    enu = NP.asarray([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]).reshape(-1,3)
    latitude = 45.0
    expected_xyz = NP.asarray([[0.0, 1.0, 0.0],
                               [-1/NP.sqrt(2.0), 0.0, 1/NP.sqrt(2.0)],
                               [1/NP.sqrt(2.0), 0.0, 1/NP.sqrt(2.0)]]).reshape(-1,3)
    xyz = GEOM.enu2xyz(enu, latitude, units='degrees')
    NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)

def test_xyz2enu():
    xyz = NP.asarray([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]).reshape(-1,3)
    latitude = 45.0
    expected_enu = NP.asarray([[0.0, -1/NP.sqrt(2.0), 1/NP.sqrt(2.0)],
                               [1.0, 0.0, 0.0],
                               [0.0, 1/NP.sqrt(2.0), 1/NP.sqrt(2.0)]]).reshape(-1,3)
    enu = GEOM.xyz2enu(xyz, latitude, units='degrees')
    NP.testing.assert_allclose(enu, expected_enu, atol=1e-12)
    
def test_sph2xyz():
    lon = NP.asarray([0.0, 45.0])
    lat = NP.asarray([30.0, 0.0])
    expected_x = NP.cos(NP.radians(lat)) * NP.cos(NP.radians(lon))
    expected_y = NP.cos(NP.radians(lat)) * NP.sin(NP.radians(lon))
    expected_z = NP.sin(NP.radians(lat))
    expected_xyz = NP.hstack((expected_x.reshape(-1,1), expected_y.reshape(-1,1), expected_z.reshape(-1,1)))
    x, y, z = GEOM.sph2xyz(lon, lat)
    xyz = NP.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)

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

    
