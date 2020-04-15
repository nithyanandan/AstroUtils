import numpy as NP
import h5py
import ast
import warnings

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

def find_all_occurrences_list1_in_list2(list1, list2):

    """
    ---------------------------------------------------------------------------
    Find all occurrences of input list1 (a reference list) in input list2

    Inputs:

    list1   [list or numpy array] List of elements which need to be searched 
            for in list2. Must be a flattened list or numpy array

    list2   [list or numpy array] List of elements in which elements in list1 
            are searched for. Must be a flattened list or numpy array

    Output:

    ind     [list of lists] Indices of occurrences of elements 
            of input list1 indexed into list2. For each element in list1, 
            there is an output list which contains all the indices of this 
            element occurring in list2. Hence, the output is a list of lists
            where the top level list contains equal number of items as list1.
            Each i-th item in this list is another list containing indices of 
            the element list1[i] in list2
    ---------------------------------------------------------------------------
    """

    if not isinstance(list1, (list, NP.ndarray)):
        raise TypeError('Input list1 must be a list or numpy array')
    if not isinstance(list2, (list, NP.ndarray)):
        raise TypeError('Input list2 must be a list or numpy array')

    list_of_list_of_inds = [[i for i, x in enumerate(list2) if x == e] for e in list1]
    return list_of_list_of_inds

################################################################################

def save_dict_to_hdf5(dic, filename, compressinfo=None):

    """
    ---------------------------------------------------------------------------
    Save a dictionary as a HDF5 structure under the given filename preserving 
    its structure

    Inputs:

    dic         [dictionary] Input dictionary which is to be stored in HDF5
                format

    filename    [string] string containing full path to the HDF5 file including 
                the file name

    compressinfo
                [dictionary] Dictionary containing compression options or 
                set as None (default) when no compression is to be applied. 
                When compression is to be applied, it contains keys of those 
                data that are to be compressed. Under each key is another 
                dictionary with the following keys and values:
                'compress_fmt'  [string] Compression format. Accepted values
                                are 'gzip' and 'lzf'
                'compress_opts' [int] Integer denoting level of compression. 
                                Only applies if compress_fmt is set to 'gzip'.
                                It must be an integer between 0 and 9
                'chunkshape'    [tuple] Shape of the chunks to be used in 
                                compression. It must be broadcastable to the
                                data shape inside input dic
                If at any point, any error is encountered, it will switch to 
                no compression
    ---------------------------------------------------------------------------
    """

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic, compressinfo=compressinfo)

################################################################################

def recursively_save_dict_contents_to_group(h5file, path, dic, compressinfo=None):

    """
    ---------------------------------------------------------------------------
    Recursively store contents of a dictionary in HDF5 groups

    Inputs:

    h5file      [Python File Object] An open file object under which the HDF5
                groups will be created

    path        [string] String containing the root group under the python file
                object h5file

    dic         [dictionary] dictionary whose keys and items will be stored 
                under the root group specified by path under the python file
                object h5file
    
    compressinfo
                [dictionary] Dictionary containing compression options or 
                set as None (default) when no compression is to be applied. 
                When compression is to be applied, it contains keys of those 
                data that are to be compressed. Under each key is another 
                dictionary with the following keys and values:
                'compress_fmt'  [string] Compression format. Accepted values
                                are 'gzip' and 'lzf'
                'compress_opts' [int] Integer denoting level of compression. 
                                Only applies if compress_fmt is set to 'gzip'.
                                It must be an integer between 0 and 9
                'chunkshape'    [tuple] Shape of the chunks to be used in 
                                compression. It must be broadcastable to the
                                data shape inside input dic
                If at any point, any error is encountered, it will switch to 
                no compression
    ---------------------------------------------------------------------------
    """

    for key, item in dic.iteritems():
        if not isinstance(key, str):
            warnings.warn('Key found not to be a string. Converting the key to string and proceeding...')
            key = str(key)
        if isinstance(item, (NP.ndarray, NP.int, NP.int32, NP.int64, NP.float, NP.float32, NP.float64, NP.complex, NP.complex64, NP.complex128, str, bytes)):
            if isinstance(item, NP.ndarray):
                if compressinfo is not None:
                    if isinstance(compressinfo, dict):
                        try:
                            compress_fmt = compressinfo[key]['compress_fmt'].lower()
                            compress_opts = NP.clip(compressinfo[key]['compress_opts'], 0, 9)
                            chunkshape = compressinfo[key]['chunkshape']
                        except:
                            h5file[path + key] = item
                        else:
                            dset = h5file.create_dataset(path+key, data=item, chunks=chunkshape, compression=compress_fmt, compression_opts=compress_opts)
                            
                        # if not isinstance(compressinfo[key]['compress_fmt'], str):
                        #     raise TypeError('Input parameter compress_fmt must be a string')
                        # compress_fmt = compressinfo[key]['compress_fmt'].lower()
                        # if compress_fmt not in ['gzip', 'lzf']:
                        #     raise ValueError('Input parameter compress_fmt invalid')
                        # if compress_fmt == 'gzip':
                        #     if not isinstance(compressinfo[key]['compress_opts'], int):
                        #         raise TypeError('Input parameter compress_opts must be an integer')
                        #     compress_opts = NP.clip(compressinfo[key]['compress_opts'], 0, 9)
                        # if 'chunkshape' not in compressinfo[key]:
                        #     raise KeyError('Key chunkshape not provided in cmagompressinfo parameter')
                        # elif not isinstance(compressinfo[key]['chunkshape'], tuple):
                        #     raise TypeError('Value under chunkshape key in compressinfo parameter must be a tuple')
                        # else:
                        #     dset = h5file.create_dataset(path+key, data=item, chunks=chunkshape, compression=compress_fmt, compression_opts=compress_opts)
                    else:
                        warnings.warn('Compression options not specified properly. Proceeding with no compression')
                        h5file[path + key] = item
                else:
                    h5file[path + key] = item
            else:
                h5file[path + key] = item
        elif item is None:
            h5file[path + key] = 'None'
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, compressinfo=compressinfo)
        else:
            raise ValueError('Cannot save %s type'%type(item))

################################################################################

def load_dict_from_hdf5(filename):

    """
    ---------------------------------------------------------------------------
    Load HDF5 contents into a python dictionary preserving the structure

    Input:

    filename    [string] Full path to the HDF5 file

    Output:

    Python dictionary containing the contents of the HDF5 file
    ---------------------------------------------------------------------------
    """

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

################################################################################

def recursively_load_dict_contents_from_group(h5file, path):

    """
    ---------------------------------------------------------------------------
    Recursively load HDF5 group contents into python dictionary structure

    Inputs:

    h5file      [Python File Object] An open file object under which the HDF5
                groups will be created

    path        [string] String containing the root group under the python file
                object h5file

    Output:

    Python structure that is copied from the HDF5 content at the level 
    specified by the path in the python object h5file
    ---------------------------------------------------------------------------
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if isinstance(item.value, str):
                try:
                    if ast.literal_eval(item.value) is None:
                        ans[key] = None
                except:
                    ans[key] = item.value
            else:
                ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

################################################################################
