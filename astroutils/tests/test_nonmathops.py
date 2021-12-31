from __future__ import print_function, division, unicode_literals, absolute_import
import sys
from .. import nonmathops as NMO

if sys.version_info.major == 2:
    def assert_dict_equal(d1, d2):
        assert d1==d2
else:
    from unittest import TestCase
    tc = TestCase()
    assert_dict_equal = tc.assertDictEqual

def test_recursive_find_notNone_in_dict():
    inpdict = {'a': 1, 'b': None, 'c': [0,1], 'd': {'d1': None, 'd2': {'d21': None, 'd22': 22}, 'd3': 3}, 'e': {'e1': None, 'e2': None}}
    expdict = {'c': [0,1], 'a': 1, 'd': {'d3': 3, 'd2': {'d22': 22}}}
    outdict = NMO.recursive_find_notNone_in_dict(inpdict)
    assert_dict_equal(expdict, outdict)

def test_recursive_find_notNone_in_allNone_dict():
    inpdict = {'a': None, 'b': None, 'c': {'c1': None, 'c2': None}}
    expdict = {}
    outdict = NMO.recursive_find_notNone_in_dict(inpdict)
    assert_dict_equal(expdict, outdict)

def test_is_dict1_subset_of_dict2_case1():
    dict1 = {'a': None, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}}
    dict2 = {'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}, 'd': 'xyz'}
    assert NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=True)
    assert not NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=False)

def test_is_dict1_subset_of_dict2_case2():
    dict1 = {'a': None, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}}
    dict2 = {'a': 0, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}, 'd': 'xyz'}
    assert NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=True)
    assert not NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=False)
