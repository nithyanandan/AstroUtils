from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import range
import sys
import pytest
from .. import geometry as GEOM
import numpy as NP

if sys.version_info.major == 2:
    def assert_dict_equal(d1, d2):
        assert d1==d2
else:
    from unittest import TestCase
    tc = TestCase()
    assert_dict_equal = tc.assertDictEqual

def test_generate_line_from_point_and_slope(generate_line_from_point_and_slope):
    points, slopes, expected_eqns = generate_line_from_point_and_slope # Read from fixture in conftest.py
    points = NP.asarray(points).reshape(-1,2)
    slopes = NP.asarray(slopes).reshape(-1)
    expected_eqns = NP.asarray(expected_eqns).reshape(-1,3)
    eqns = GEOM.generate_line_from_point_and_slope(points, slopes)
    NP.testing.assert_allclose(eqns, expected_eqns, atol=1e-10, equal_nan=True, err_msg='Line equations do not match', verbose=True)    

def test_generate_line_from_two_points(generate_line_from_two_points):
    points1, points2, expected_eqns = generate_line_from_two_points # Read from fixture in conftest.py
    points1 = NP.asarray(points1).reshape(-1,2)
    points2 = NP.asarray(points2).reshape(-1,2)
    expected_eqns = NP.asarray(expected_eqns).reshape(-1,3)
    eqns = GEOM.generate_line_from_two_points(points1, points2)
    NP.testing.assert_allclose(eqns, expected_eqns, atol=1e-10, equal_nan=True, err_msg='Line equations do not match', verbose=True)    

def test_points_from_line2d_intersection(points_from_line2d_intersection):
    coeffs, dvect, expected_pts = points_from_line2d_intersection # Read from fixture in conftest.py
    coeffs = NP.asarray(coeffs)
    dvect = NP.asarray(dvect)
    expected_pts = NP.asarray(expected_pts)
    pts = GEOM.points_from_line2d_intersection(coeffs, dvect, ravel=False)
    expected_pts_raveled = NP.asarray([expected_pts[i,j,:] for i in range(expected_pts.shape[0]) for j in range(i-1,-1,-1)])
    pts_raveled = GEOM.points_from_line2d_intersection(coeffs, dvect, ravel=True)
    NP.testing.assert_allclose(pts, expected_pts, atol=1e-10, equal_nan=True, err_msg='Intersection points do not match', verbose=True)
    NP.testing.assert_allclose(pts_raveled,  expected_pts_raveled, atol=1e-10, equal_nan=True, err_msg='Intersection points do not match', verbose=True)

def test_get_abscissa_from_ordinate_on_line(get_abscissa_from_ordinate_on_line):
    coeffs, dvect, yvals, expected_xvals = get_abscissa_from_ordinate_on_line # Read from fixture in conftest.py
    coeffs = NP.asarray(coeffs)
    dvect = NP.asarray(dvect)
    yvals = NP.asarray(yvals)
    expected_xvals = NP.asarray(expected_xvals)
    xvals = GEOM.get_abscissa_from_ordinate_on_line(coeffs, dvect, yvals)
    NP.testing.assert_allclose(xvals, expected_xvals, atol=1e-10, equal_nan=True, err_msg='Abscissa values do not match', verbose=True)

def test_get_ordinate_from_abscissa_on_line(get_ordinate_from_abscissa_on_line):
    coeffs, dvect, xvals, expected_yvals = get_ordinate_from_abscissa_on_line # Read from fixture in conftest.py
    coeffs = NP.asarray(coeffs)
    dvect = NP.asarray(dvect)
    xvals = NP.asarray(xvals)
    expected_yvals = NP.asarray(expected_yvals)
    yvals = GEOM.get_ordinate_from_abscissa_on_line(coeffs, dvect, xvals)
    NP.testing.assert_allclose(yvals, expected_yvals, atol=1e-10, equal_nan=True, err_msg='Ordinate values do not match', verbose=True)

def test_generate_parallel_lines_at_distance_from_line(generate_parallel_lines_at_distance_from_line):
    coeffs, dvect, displacements, expected_output_dvects = generate_parallel_lines_at_distance_from_line # Read from fixture in conftest.py
    coeffs = NP.asarray(coeffs)
    dvect = NP.asarray(dvect)
    displacements = NP.asarray(displacements)
    expected_output_dvects = NP.asarray(expected_output_dvects)
    output_dvects = GEOM.generate_parallel_lines_at_distance_from_line(coeffs, dvect, displacements)
    NP.testing.assert_allclose(output_dvects, expected_output_dvects, atol=1e-10, equal_nan=True, err_msg='dvect values do not match', verbose=True)

def test_polygonArea2D(polygonArea2D):
    vertices2D, expected_area2D = polygonArea2D # Read from fixture in conftest.py
    vertices2D = NP.asarray(vertices2D)
    expected_area2D = NP.asarray(expected_area2D)
    try:
        area2D = GEOM.polygonArea2D(vertices2D, absolute=False)
        NP.testing.assert_allclose(area2D, expected_area2D, atol=1e-10, equal_nan=True, err_msg='Polygon areas do not match', verbose=True)
    except Exception as err:
        NP.testing.assert_equal(NP.nan, expected_area2D, err_msg='Exception must have been raised', verbose=True)
    
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
    NP.testing.assert_allclose(hadec, expected_hadec, atol=1e-7)

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

def test_xyz2sph(xyz_to_sph):
    xyz, r_lat_lon = xyz_to_sph # Read from fixture in conftest.py
    xyz = NP.asarray(xyz).reshape(-1)
    r_lat_lon = NP.asarray(r_lat_lon).reshape(-1)

    expected_r, expected_lat, expected_lon = GEOM.xyz2sph(xyz[0], xyz[1], xyz[2], units='radians')
    expected_r_lat_lon = NP.asarray([expected_r, expected_lat, expected_lon])
    NP.testing.assert_allclose(r_lat_lon, expected_r_lat_lon, atol=1e-12)

def test_sph2xyz(sph_to_xyz):
    r_lat_lon, xyz = sph_to_xyz # Read from fixture in conftest.py
    xyz = NP.asarray(xyz).reshape(-1)
    r_lat_lon = NP.asarray(r_lat_lon).reshape(-1)

    expected_x, expected_y, expected_z = GEOM.sph2xyz(NP.degrees(r_lat_lon[2]), NP.degrees(r_lat_lon[1]), rad=r_lat_lon[0])
    expected_xyz = NP.asarray([expected_x, expected_y, expected_z])
    NP.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)
    
def test_sphdist(sph_dist):
    lonlat1, lonlat2, diffang_cartesian = sph_dist # Read from fixture in conftest.py
    print(NP.hstack((lonlat1, lonlat2)))
    expected_diffang = GEOM.sphdist(lonlat1[:,0], lonlat1[:,1], lonlat2[:,0], lonlat2[:,1])
    NP.testing.assert_allclose(diffang_cartesian, expected_diffang, atol=1e-12)

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

def test_parabola_parameters(parabola_parms):
    indict, fulldict = parabola_parms
    if ('f/D' in indict) and ('D' in indict):
        outdict = GEOM.parabola_parameters(dia=indict['D'], f_to_dia_ratio=indict['f/D'])
    if ('f/D' in indict) and ('f' in indict):
        outdict = GEOM.parabola_parameters(f=indict['f'], f_to_dia_ratio=indict['f/D'])
    if ('f/D' in indict) and ('h' in indict):
        outdict = GEOM.parabola_parameters(depth=indict['h'], f_to_dia_ratio=indict['f/D'])
    if ('D' in indict) and ('f' in indict):
        outdict = GEOM.parabola_parameters(f=indict['f'], dia=indict['D'])
    if ('D' in indict) and ('h' in indict):
        outdict = GEOM.parabola_parameters(depth=indict['h'], dia=indict['D'])
    if ('f' in indict) and ('h' in indict):
        outdict = GEOM.parabola_parameters(depth=indict['h'], f=indict['f'])
    assert_dict_equal(fulldict, outdict)

