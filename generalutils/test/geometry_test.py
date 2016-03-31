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
    
