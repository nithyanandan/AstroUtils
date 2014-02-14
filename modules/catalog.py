import numpy as NP
import geometry as GEOM

#################################################################################

class Catalog:

    """
    -----------------------------------------------------------------------------
    Class to manage catalog information.

    Attributes:

    frequency      [scalar] Frequency at which the catalog exists

    location       [numpy array or list of lists] Positions of the sources in 
                   catalog. Each position is specified as a row (numpy array)
                   or a 2-element list which is input as a list of lists for all
                   the sources in the catalog

    flux_density   [numpy vector or list] Flux densities of sources at the 
                   frequency of the catalog specified as a numpy vector or list,
                   one for each source

    spectral_index [numpy vector or list] Spectral index for each source
                   (flux ~ freq^spectral_index). If not specified, it is taken
                   as 0 for all sources. Same size as flux_density

    epoch          [string] Epoch appropriate for the coordinate system. Default
                   is 'J2000'

    coords         [string] Coordinate system used for the source positions in 
                   the catalog. Currently accepted values are 'radec' (RA-Dec)

    Member Functions:

    __init__()     Initialize an instance of class Catalog

    match()        Match the source positions in an instance of class Catalog 
                   with another instance of the same class to a specified
                   angular radius using spherematch() in the geometry module

    subset()       Provide a subset of the catalog using a list of indices onto
                   the existing catalog
    ------------------------------------------------------------------------------
    """

    ##############################################################################

    def __init__(self, frequency, location, flux_density, spectral_index=None,
                 epoch='J2000', coords='radec'):

        """
        --------------------------------------------------------------------------
        Initialize an instance of class Catalog

        Class attributes initialized are:
        frequency, location, flux_density, epoch, spectral_index, coords

        Read docstring of class Catalog for details on these attributes.
        --------------------------------------------------------------------------
        """

        self.frequency = frequency
        self.location = NP.asarray(location)
        self.flux_density = NP.asarray(flux_density).reshape(-1)
        self.epoch = epoch
        self.coords = coords

        if spectral_index is None:
            self.spectral_index = NP.zeros(self.flux_density.size).reshape(-1)
        else: 
            self.spectral_index = NP.asarray(spectral_index).reshape(-1)
        
        if (self.location.shape[0] != self.flux_density.size) or (self.flux_density.size != self.spectral_index.size) or (self.location.shape[0] != self.spectral_index.size):
            raise ValueError('location, flux_density, and spectral_index must be of equal size.')

    #############################################################################

    def match(self, other, matchrad=None, nnearest=0, maxmatches=-1):

        """
        -------------------------------------------------------------------------
        Match the source positions in an instance of class Catalog with another
        instance of the same class to a specified angular radius using
        spherematch() in the geometry module

        Inputs:

        other       [2-column numpy array instance of class Catalog] A numpy
                    array with two columns specifying the source positions in the
                    other catalog or the other instance of class Catalog with
                    which the current instance is to be matched with

        matchrad    [scalar] Angular radius (in degrees) inside which matching 
                    should occur. If not specified, if maxmatches is positive, 
                    all the nearest maxmatches neighbours are found, and if
                    maxmatches is not positive, the nnearest-th nearest neighbour
                    specified by nnearest is found.

        maxmatches  [scalar] The maximum number of matches (all of the maxmatches
                    nearest neighbours) that lie within matchrad are found. If
                    matchrad is not specified, all the maxmatches nearest
                    neighbours are found. If maxmatches < 0, and matchrad is not
                    set, then the nnearest-th nearest neighbour is found (which
                    defaults to the nearest neighbour if nnearest <= 0)

        nnearest    [scalar] nnearest-th neighbour to be found. Used only when
                    maxmatches is not positive. If matchrad is set, the 
                    specified neighbour is identified if found inside matchrad, 
                    otherwise the nnearest-th neighbour is identified regardless
                    of the angular distance.

        Outputs:

        m1          [list] List of indices of matches in the current instance of
                    class Catalog
        
        m2          [list] List of indices of matches in the other instance of
                    class Catalog

        d12         [list] List of angular distances between the matched subsets
                    of the two catalogs indexed by m1 and m2 respectively
        -------------------------------------------------------------------------
        """

        if not isinstance(other, (NP.ndarray, Catalog)):
            raise TypeError('"other" must be a Nx2 numpy array or an instance of class Catalog.')
        
        if isinstance(other, Catalog):
            if (self.epoch == other.epoch) and (self.coords == other.coords):
                return GEOM.spherematch(self.location[:,0], self.location[:,1],
                                        other.location[:,0],
                                        other.location[:,1], matchrad,
                                        nnearest, maxmatches)
            else:
                raise ValueError('epoch and/or sky coordinate type mismatch. Cannot match.')
        else:
            return GEOM.spherematch(self.location[:,0], self.location[:,1],
                                    other[:,0], other[:,1], matchrad,
                                    nnearest, maxmatches)
        
    #############################################################################

    def subset(self, indices=None):

        """
        -------------------------------------------------------------------------
        Provide a subset of the catalog using a list of indices onto the existing
        catalog
        
        Inputs:

        indices    [list] List of indices of sources in the current instance of 
        class Catalog

        Output:    [instance of class Catalog] An instance of class Catalog
                   holding a subset of the sources in the current instance of 
                   class Catalog 
        -------------------------------------------------------------------------
        """

        if (indices is None) or (len(indices) == 0):
            raise IndexError('No indices specified to select a catalog subset.')
        else:
            return Catalog(self.frequency, self.location[indices, :], self.flux_density[indices, :], self.spectral_index[indices], self.epoch, self.coords)

#################################################################################

class SkyModel:
    
    """
    -----------------------------------------------------------------------------
    Class to manage sky model information. Sky model can be specified in the form
    of a catalog, an image, or an image cube.

    Attributes:

    catalog       [instance of class Catalog] An instance of class Catalog that
                  contains information about sources on the sky such as flux 
                  densities, positions, spectral indices, etc.

    image         [numpy array] Image of a part of the sky represented as a
                  2-D numpy array 

    cube          [numpy array] Image cube where the third axis represents 
                  frequency usually

    Member functions:

    __init__()    Initialize an instance of class SkyModel
    -----------------------------------------------------------------------------
    """

    #############################################################################

    def __init__(self, catalog=None, image=None, cube=None):

        """
        -------------------------------------------------------------------------
        Initialize an instance of class SkyModel

        Attributes initialized are:
        catalog, image, cube

        Read the docstring of class SkyModel for more details on these attributes
        -------------------------------------------------------------------------
        """

        if isinstance(catalog, Catalog):
            self.catalog = catalog
        self.image = image
        self.cube = cube

    def add_sources(self, catalog):
        pass

#################################################################################
