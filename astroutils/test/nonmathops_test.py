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
