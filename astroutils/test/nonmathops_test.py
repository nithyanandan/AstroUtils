from astroutils import nonmathops as NMO

def test_recursive_find_notNone_in_dict():
    inpdict = {'a': 1, 'b': None, 'c': [0,1], 'd': {'d1': None, 'd2': {'d21': None, 'd22': 22}, 'd3': 3}, 'e': {'e1': None, 'e2': None}}
    expdict = {'c': [0,1], 'a': 1, 'd': {'d3': 3, 'd2': {'d22': 22}}}
    outdict = NMO.recursive_find_notNone_in_dict(inpdict)
    assert cmp(expdict, outdict) == 0

def test_recursive_find_notNone_in_allNone_dict():
    inpdict = {'a': None, 'b': None, 'c': {'c1': None, 'c2': None}}
    expdict = {}
    outdict = NMO.recursive_find_notNone_in_dict(inpdict)
    assert cmp(expdict, outdict) == 0

def test_is_dict1_subset_of_dict1_case1():
    dict1 = {'a': None, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}}
    dict2 = {'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}, 'd': 'xyz'}
    assert NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=True)
    assert not NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=False)

def test_is_dict1_subset_of_dict1_case2():
    dict1 = {'a': None, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}}
    dict2 = {'a': 0, 'b': [0,1], 'c': {'c1': 1, 'c2': {'c21': 21, 'c22': 22}}, 'd': 'xyz'}
    assert NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=True)
    assert not NMO.is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=False)
