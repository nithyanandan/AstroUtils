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
