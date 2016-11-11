import numpy as NP

def recursive_find_notNone_in_dict(inpdict):

    """
    ----------------------------------------------------------------------------
    Recursively walk through a dictionary and reduce it to only non-None values.

    Inputs:

    inpdict     [dictionary] Input dictionary to reduced to non-None values

    Outputs:

    outdict is an output dictionary which only contains keys and values 
    corresponding to non-None values
    ----------------------------------------------------------------------------
    """
    
    if not isinstance(inpdict, dict):
        raise TypeError('inpdict must be a dictionary')

    outdict = {}
    for k, v in inpdict.iteritems():
        if v is not None:
            if not isinstance(v, dict):
                outdict[k] = v
            else:
                temp = recursive_find_notNone_in_dict(v)
                if temp:
                    outdict[k] = temp
    return outdict

################################################################################

def is_dict1_subset_of_dict2(dict1, dict2, ignoreNone=True):

    """
    ----------------------------------------------------------------------------
    Check if keys and values of the first dictionary are a subset of the second.

    Inputs:

    dict1       [dictionary] First dictionary. It will be checked if both its
                keys and values are a subset of the second dictionary.

    dict2       [dictionary] Second dictionary. The values and keys of the first
                dictionary will be checked against this dictionary to check if
                the first is a subset of the second. 

    ignoreNone  [boolean] If set to True (default), the subset checking happens
                using the non-None values in both dictionaries. This is a 
                loose check. If set to False, a strict subset checking happens
                not ignoring the None values, if any.

    Output: 

    Boolean value True if dict1 is found to be a subset of dict2, False 
    otherwise
    ----------------------------------------------------------------------------
    """
    
    if not isinstance(dict1, dict):
        raise TypeError('Input dict1 must be a dictionary')

    if not isinstance(dict2, dict):
        raise TypeError('Input dict2 must be a dictionary')

    if ignoreNone:
        dict1 = recursive_find_notNone_in_dict(dict1)
        dict2 = recursive_find_notNone_in_dict(dict2)

    if cmp(dict1, dict2) == 0:
        return True
    else:
        dict2sub = {}
        for k, v in dict1.iteritems():
            if k in dict2:
                dict2sub[k] = dict2[k]
            else:
                return False
        if cmp(dict1, dict2sub) == 0:
            return True
        else:
            return False

################################################################################

def find_list_in_list(reference_array, inp):

    """
    ---------------------------------------------------------------------------
    Find occurrences of input list in a reference list and return indices 
    into the reference list

    Inputs:

    reference_array [list or numpy array] One-dimensional reference list or
                    numpy array in which occurrences of elements in the input 
                    list or array will be found

    inp             [list or numpy array] One-dimensional input list whose 
                    elements will be searched in the reference array and 
                    the indices into the reference array will be returned

    Output:

    ind             [numpy masked array] Indices of occurrences of elements 
                    of input array in the reference array. It will be of same 
                    size as input array. For example, 
                    inp = reference_array[ind]. Indices for elements which are 
                    not found in the reference array will be masked.
    ---------------------------------------------------------------------------
    """

    try:
        reference_array, inp
    except NameError:
        raise NameError('Inputs reference_array, inp must be specified')

    if not isinstance(reference_array, (list, NP.ndarray)):
        raise TypeError('Input reference_array must be a list or numpy array')
    reference_array = NP.asarray(reference_array).ravel()

    if not isinstance(inp, (list, NP.ndarray)):
        raise TypeError('Input inp must be a list or numpy array')
    inp = NP.asarray(inp).ravel()

    if (inp.size == 0) or (reference_array.size == 0):
        raise ValueError('One or both inputs contain no elements')

    sortind_ref = NP.argsort(reference_array)
    sorted_ref = reference_array[sortind_ref]
    ind_in_sorted_ref = NP.searchsorted(sorted_ref, inp)
    ii = NP.take(sortind_ref, ind_in_sorted_ref, mode='clip')
    mask = reference_array[ii] != inp
    ind = NP.ma.array(ii, mask=mask)

    return ind
    
################################################################################
