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

def test_point_init(point_init_case):
    xyz_input, expected_xyz = point_init_case # Read from fixture in conftest.py
    p = GEOM.Point(xyz_input)
    assert (p.x, p.y, p.z) == expected_xyz

def test_point_init_from_point_instance():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    p2 = GEOM.Point(p1)
    assert (p2.x, p2.y, p2.z) == (1.0, 2.0, 3.0)

def test_point_init_invalid_type():
    with pytest.raises(TypeError):
        GEOM.Point('abc')

def test_point_str():
    p = GEOM.Point([1.0, 2.0, 3.0])
    assert str(p) == '(1.0, 2.0, 3.0)'

def test_point_add():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    p2 = GEOM.Point([4.0, 5.0, 6.0])
    p3 = p1 + p2
    assert (p3.x, p3.y, p3.z) == (5.0, 7.0, 9.0)

def test_point_add_invalid_type():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        p1 + 5.0

def test_point_sub():
    p1 = GEOM.Point([4.0, 5.0, 6.0])
    p2 = GEOM.Point([1.0, 2.0, 3.0])
    p3 = p1 - p2
    assert (p3.x, p3.y, p3.z) == (3.0, 3.0, 3.0)

def test_point_sub_invalid_type():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        p1 - 5.0

def test_point_mul_dot_product():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    p2 = GEOM.Point([4.0, 5.0, 6.0])
    assert p1 * p2 == 32.0 # 1*4 + 2*5 + 3*6

def test_point_mul_invalid_type():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        p1 * 5.0

def test_point_rmul_scalar_product():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    p2 = 2.0 * p1
    assert (p2.x, p2.y, p2.z) == (2.0, 4.0, 6.0)

def test_point_rmul_invalid_type():
    p1 = GEOM.Point([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        p1.__rmul__('a')

def test_point_abs():
    p = GEOM.Point([3.0, 4.0, 0.0])
    assert abs(p) == 5.0

def test_point_comparisons():
    p_small = GEOM.Point([1.0, 0.0, 0.0]) # magnitude 1
    p_large = GEOM.Point([3.0, 4.0, 0.0]) # magnitude 5
    p_equal = GEOM.Point([0.0, 5.0, 0.0]) # magnitude 5, same as p_large
    assert p_small < p_large
    assert p_small <= p_large
    assert p_large > p_small
    assert p_large >= p_small
    assert p_large == p_equal
    assert p_large <= p_equal
    assert p_large >= p_equal
    assert not (p_large < p_equal)
    assert not (p_large > p_equal)

def test_point_comparison_invalid_type():
    p = GEOM.Point([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        p < 5.0
    with pytest.raises(TypeError):
        p <= 5.0
    with pytest.raises(TypeError):
        p > 5.0
    with pytest.raises(TypeError):
        p >= 5.0
    with pytest.raises(TypeError):
        p == 5.0

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

def test_sample_2D_plane_rectangle():
    nx, ny, dx, dy = 2, 2, 1.0, 1.0
    expected_locs = NP.asarray([[-0.5, -0.5],
                                 [ 0.5, -0.5],
                                 [-0.5,  0.5],
                                 [ 0.5,  0.5]])
    locs = GEOM.sample_2D_plane('rectangle', nx=nx, ny=ny, dx=dx, dy=dy)
    assert locs.shape == (nx*ny, 2)
    NP.testing.assert_allclose(locs, expected_locs, atol=1e-10)

def test_sample_2D_plane_rectangle_default_spacing():
    nx, ny = 3, 1
    locs = GEOM.sample_2D_plane('rectangle', nx=nx, ny=ny)
    expected_locs = NP.asarray([[-1.0, 0.0],
                                 [ 0.0, 0.0],
                                 [ 1.0, 0.0]])
    NP.testing.assert_allclose(locs, expected_locs, atol=1e-10)

def test_sample_2D_plane_hexagon():
    nedge, dx = 2, 1.0
    hexdy = dx * NP.sin(NP.pi/3)
    expected_locs = NP.asarray([[-0.5, -hexdy],
                                 [ 0.5, -hexdy],
                                 [-1.0,  0.0  ],
                                 [ 0.0,  0.0  ],
                                 [ 1.0,  0.0  ],
                                 [-0.5,  hexdy],
                                 [ 0.5,  hexdy]])
    locs = GEOM.sample_2D_plane('hexagon', nedge=nedge, dx=dx)
    assert locs.shape == (7, 2)
    NP.testing.assert_allclose(locs, expected_locs, atol=1e-10)

def test_sample_2D_plane_centred_at_origin():
    locs_rect = GEOM.sample_2D_plane('rectangle', nx=4, ny=3, dx=1.3, dy=0.7)
    NP.testing.assert_allclose(locs_rect.mean(axis=0), [0.0, 0.0], atol=1e-10)
    locs_hex = GEOM.sample_2D_plane('hexagon', nedge=3, dx=0.9)
    NP.testing.assert_allclose(locs_hex.mean(axis=0), [0.0, 0.0], atol=1e-10)

def test_sample_2D_plane_indices_subset():
    nx, ny, dx, dy = 2, 2, 1.0, 1.0
    full_locs = GEOM.sample_2D_plane('rectangle', nx=nx, ny=ny, dx=dx, dy=dy)
    indices = [1, 3]
    subset_locs = GEOM.sample_2D_plane('rectangle', indices=indices, nx=nx, ny=ny, dx=dx, dy=dy)
    assert subset_locs.shape == (len(indices), 2)
    NP.testing.assert_allclose(subset_locs, full_locs[indices,:], atol=1e-10)

def test_sample_2D_plane_indices_none_returns_all():
    full_locs = GEOM.sample_2D_plane('hexagon', nedge=2, dx=1.0)
    all_locs = GEOM.sample_2D_plane('hexagon', indices=None, nedge=2, dx=1.0)
    NP.testing.assert_allclose(all_locs, full_locs, atol=1e-10)

def test_sample_2D_plane_invalid_pattern():
    with pytest.raises(ValueError):
        GEOM.sample_2D_plane('triangle', nx=2, ny=2)

def test_sample_2D_plane_missing_required_kwarg():
    with pytest.raises(KeyError):
        GEOM.sample_2D_plane('rectangle', ny=2)

def test_ecef2enu_equatorial_reference(ecef2enu_equatorial_case):
    ref_xyz, ref_info, offset, expected_enu = ecef2enu_equatorial_case # Read from fixture in conftest.py
    xyz = (ref_xyz + offset).reshape(1,-1)
    enu = GEOM.ecef2enu(xyz, dict(ref_info))
    NP.testing.assert_allclose(enu.ravel(), expected_enu, atol=1e-8)

def test_ecef2enu_1d_input(ecef2enu_equatorial_case):
    ref_xyz, ref_info, offset, expected_enu = ecef2enu_equatorial_case # Read from fixture in conftest.py
    xyz_1d = ref_xyz + offset # shape (3,)
    enu = GEOM.ecef2enu(xyz_1d.copy(), dict(ref_info))
    NP.testing.assert_allclose(enu.ravel(), expected_enu, atol=1e-8)

def test_ecef2enu_at_reference_point(ecef2enu_equatorial_case):
    ref_xyz, ref_info, offset, expected_enu = ecef2enu_equatorial_case # Read from fixture in conftest.py
    enu = GEOM.ecef2enu(ref_xyz.reshape(1,-1).copy(), dict(ref_info))
    NP.testing.assert_allclose(enu.ravel(), [0.0, 0.0, 0.0], atol=1e-8)

def test_enu2ecef_at_reference_point(ecef2enu_general_case):
    (ref_lat, ref_lon, ref_alt), ref_xyz, offset = ecef2enu_general_case # Read from fixture in conftest.py
    ref_info = {'lat': ref_lat, 'lon': ref_lon, 'alt': ref_alt, 'units': 'degrees'}
    xyz = GEOM.enu2ecef(NP.zeros((1,3)), ref_info)
    NP.testing.assert_allclose(xyz.ravel(), ref_xyz, atol=1e-6)

def test_enu2ecef_default_alt():
    ref_info = {'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    xyz = GEOM.enu2ecef(NP.zeros((1,3)), ref_info)
    NP.testing.assert_allclose(xyz.ravel(), [6378137.0, 0.0, 0.0], atol=1e-6)

def test_ecef2enu_enu2ecef_roundtrip(ecef2enu_general_case):
    (ref_lat, ref_lon, ref_alt), ref_xyz, offset = ecef2enu_general_case # Read from fixture in conftest.py
    xyz = (ref_xyz + offset).reshape(1,-1)
    ref_info_fwd = {'xyz': ref_xyz.copy(), 'lat': ref_lat, 'lon': ref_lon, 'units': 'degrees'}
    enu = GEOM.ecef2enu(xyz.copy(), ref_info_fwd)
    ref_info_inv = {'lat': ref_lat, 'lon': ref_lon, 'alt': ref_alt, 'units': 'degrees'}
    xyz_back = GEOM.enu2ecef(enu, ref_info_inv)
    NP.testing.assert_allclose(xyz_back, xyz, atol=1e-5)

def test_enu2ecef_ecef2enu_roundtrip(ecef2enu_general_case):
    (ref_lat, ref_lon, ref_alt), ref_xyz, offset = ecef2enu_general_case # Read from fixture in conftest.py
    enu = offset.reshape(1,-1)
    ref_info_inv = {'lat': ref_lat, 'lon': ref_lon, 'alt': ref_alt, 'units': 'degrees'}
    xyz = GEOM.enu2ecef(enu.copy(), ref_info_inv)
    ref_info_fwd = {'xyz': ref_xyz.copy(), 'lat': ref_lat, 'lon': ref_lon, 'units': 'degrees'}
    enu_back = GEOM.ecef2enu(xyz, ref_info_fwd)
    NP.testing.assert_allclose(enu_back, enu, atol=1e-6)

def test_enu2ecef_1d_input(ecef2enu_general_case):
    (ref_lat, ref_lon, ref_alt), ref_xyz, offset = ecef2enu_general_case # Read from fixture in conftest.py
    ref_info = {'lat': ref_lat, 'lon': ref_lon, 'alt': ref_alt, 'units': 'degrees'}
    xyz = GEOM.enu2ecef(offset.copy(), ref_info) # 1D (3,) input
    ref_info_fwd = {'xyz': ref_xyz.copy(), 'lat': ref_lat, 'lon': ref_lon, 'units': 'degrees'}
    enu_back = GEOM.ecef2enu(xyz, ref_info_fwd)
    NP.testing.assert_allclose(enu_back.ravel(), offset, atol=1e-6)

def test_ecef2enu_invalid_xyz_type():
    ref_info = {'xyz': NP.zeros(3), 'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(TypeError):
        GEOM.ecef2enu([1.0, 2.0, 3.0], ref_info)

def test_ecef2enu_invalid_xyz_shape():
    ref_info = {'xyz': NP.zeros(3), 'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(ValueError):
        GEOM.ecef2enu(NP.zeros((2,2)), ref_info)

def test_ecef2enu_ref_info_not_dict():
    with pytest.raises(TypeError):
        GEOM.ecef2enu(NP.zeros((1,3)), [1.0, 2.0, 3.0])

def test_ecef2enu_missing_xyz_key():
    ref_info = {'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(KeyError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_ref_xyz_not_ndarray():
    ref_info = {'xyz': [0.0, 0.0, 0.0], 'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(TypeError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_ref_xyz_wrong_size():
    ref_info = {'xyz': NP.zeros(2), 'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(ValueError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_missing_lat_key():
    ref_info = {'xyz': NP.zeros(3), 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(KeyError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_missing_lon_key():
    ref_info = {'xyz': NP.zeros(3), 'lat': 0.0, 'units': 'degrees'}
    with pytest.raises(KeyError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_invalid_units_value():
    ref_info = {'xyz': NP.zeros(3), 'lat': 0.0, 'lon': 0.0, 'units': 'foo'}
    with pytest.raises(ValueError):
        GEOM.ecef2enu(NP.zeros((1,3)), ref_info)

def test_ecef2enu_missing_units_warns_and_defaults_to_radians():
    ref_info_norad = {'xyz': NP.zeros(3), 'lat': 0.0, 'lon': 0.0}
    with pytest.warns(UserWarning):
        enu_norad = GEOM.ecef2enu(NP.zeros((1,3)), ref_info_norad)
    ref_info_rad = {'xyz': NP.zeros(3), 'lat': 0.0, 'lon': 0.0, 'units': 'radians'}
    enu_rad = GEOM.ecef2enu(NP.zeros((1,3)), ref_info_rad)
    NP.testing.assert_allclose(enu_norad, enu_rad)

def test_enu2ecef_invalid_enu_type():
    ref_info = {'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(TypeError):
        GEOM.enu2ecef([1.0, 2.0, 3.0], ref_info)

def test_enu2ecef_invalid_enu_shape():
    ref_info = {'lat': 0.0, 'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(ValueError):
        GEOM.enu2ecef(NP.zeros((2,2)), ref_info)

def test_enu2ecef_ref_info_not_dict():
    with pytest.raises(TypeError):
        GEOM.enu2ecef(NP.zeros((1,3)), [1.0, 2.0, 3.0])

def test_enu2ecef_missing_lat_key():
    ref_info = {'lon': 0.0, 'units': 'degrees'}
    with pytest.raises(KeyError):
        GEOM.enu2ecef(NP.zeros((1,3)), ref_info)

def test_enu2ecef_missing_lon_key():
    ref_info = {'lat': 0.0, 'units': 'degrees'}
    with pytest.raises(KeyError):
        GEOM.enu2ecef(NP.zeros((1,3)), ref_info)

def test_enu2ecef_invalid_units_value():
    ref_info = {'lat': 0.0, 'lon': 0.0, 'units': 'foo'}
    with pytest.raises(ValueError):
        GEOM.enu2ecef(NP.zeros((1,3)), ref_info)

def test_enu2ecef_missing_units_warns_and_defaults_to_radians():
    ref_info_norad = {'lat': 0.0, 'lon': 0.0}
    with pytest.warns(UserWarning):
        xyz_norad = GEOM.enu2ecef(NP.zeros((1,3)), ref_info_norad)
    ref_info_rad = {'lat': 0.0, 'lon': 0.0, 'units': 'radians'}
    xyz_rad = GEOM.enu2ecef(NP.zeros((1,3)), ref_info_rad)
    NP.testing.assert_allclose(xyz_norad, xyz_rad)

def test_costas_array_valid(costas_array_order):
    n, expected_perm = costas_array_order # Read from fixture in conftest.py
    expected_perm = NP.asarray(expected_perm, dtype=int)
    perm = GEOM.costas_array(n)
    NP.testing.assert_array_equal(perm, expected_perm)
    assert GEOM.is_costas_array(perm)
    if n > 0:
        NP.testing.assert_array_equal(NP.sort(perm), NP.arange(n))

def test_costas_array_welch_method(costas_welch_valid_order):
    n = costas_welch_valid_order # Read from fixture in conftest.py
    perm_auto = GEOM.costas_array(n)
    perm_welch = GEOM.costas_array(n, method='welch')
    NP.testing.assert_array_equal(perm_auto, perm_welch)
    assert GEOM.is_costas_array(perm_welch)

def test_costas_array_welch_not_applicable(costas_welch_invalid_order):
    n = costas_welch_invalid_order # Read from fixture in conftest.py
    with pytest.raises(ValueError):
        GEOM.costas_array(n, method='welch')

def test_costas_array_search_method(costas_array_order):
    n, expected_perm = costas_array_order # Read from fixture in conftest.py
    perm = GEOM.costas_array(n, method='search')
    assert GEOM.is_costas_array(perm)
    if n > 0:
        NP.testing.assert_array_equal(NP.sort(perm), NP.arange(n))

def test_costas_array_invalid_method():
    with pytest.raises(ValueError):
        GEOM.costas_array(4, method='bogus')

def test_costas_array_negative_order():
    with pytest.raises(ValueError):
        GEOM.costas_array(-1)

def test_costas_array_noninteger_order():
    with pytest.raises(TypeError):
        GEOM.costas_array(3.5)

def test_is_costas_array_valid():
    assert GEOM.is_costas_array(NP.asarray([1, 3, 2, 0]))

def test_is_costas_array_not_a_permutation():
    assert not GEOM.is_costas_array(NP.asarray([0, 0, 1]))

def test_is_costas_array_repeated_difference():
    # An arithmetic-progression permutation has a constant lag-1 difference,
    # violating the Costas condition
    assert not GEOM.is_costas_array(NP.asarray([0, 1, 2, 3]))

