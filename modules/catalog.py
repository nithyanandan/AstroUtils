import numpy as NP
import geometry as GEOM
import my_operations as OPS
import lookup_operations as LKP

#################################################################################

# class Catalog:

#     """
#     -----------------------------------------------------------------------------
#     Class to manage catalog information.

#     Attributes:

#     name           [scalar or vector] Name of the catalog. If scalar, will be 
#                    used for all sources in the catalog. If vector, will be used
#                    for corresponding object. If vector, size must equal the 
#                    number of objects.

#     frequency      [scalar or vector] Frequency at which the catalog exists. If 
#                    scalar, will be used for all sources in the catalog. If vector, 
#                    will be used for corresponding object. If vector, size must 
#                    equal the number of objects.

#     location       [numpy array or list of lists] Positions of the sources in 
#                    catalog. Each position is specified as a row (numpy array)
#                    or a 2-element list which is input as a list of lists for all
#                    the sources in the catalog

#     flux_density   [numpy vector or list] Flux densities of sources at the 
#                    frequency of the catalog specified as a numpy vector or list,
#                    one for each source

#     spectral_index [numpy vector or list] Spectral index for each source
#                    (flux ~ freq^spectral_index). If not specified, it is taken
#                    as 0 for all sources. Same size as flux_density

#     src_shape      [3-column numpy array or list of 3-element lists] source shape
#                    specified by major axis FWHM (first column), minor axis FWHM 
#                    (second column), and position angle (third column). The major
#                    and minor axes and position angle are stored in degrees. The
#                    number of rows must match the number of sources. Position 
#                    angle is in degrees east of north (same convention as local
#                    azimuth)

#     epoch          [string] Epoch appropriate for the coordinate system. Default
#                    is 'J2000'

#     coords         [string] Coordinate system used for the source positions in 
#                    the catalog. Currently accepted values are 'radec' (RA-Dec)

#     Member Functions:

#     __init__()     Initialize an instance of class Catalog

#     match()        Match the source positions in an instance of class Catalog 
#                    with another instance of the same class to a specified
#                    angular radius using spherematch() in the geometry module

#     subset()       Provide a subset of the catalog using a list of indices onto
#                    the existing catalog
#     ------------------------------------------------------------------------------
#     """

#     ##############################################################################

#     def __init__(self, name, frequency, location, flux_density, 
#                  spectral_index=None, src_shape=None, epoch='J2000', 
#                  coords='radec', src_shape_units=None):

#         """
#         --------------------------------------------------------------------------
#         Initialize an instance of class Catalog

#         Class attributes initialized are:
#         frequency, location, flux_density, epoch, spectral_index, coords, and 
#         src_shape

#         Other input(s):

#         src_shape_units  [3-element list or tuple of strings] Specifies the units 
#                          of major axis FWHM, minor axis FWHM, and position angle.
#                          Accepted values for major and minor axes units are 
#                          'arcsec', 'arcmin', 'degree', or 'radian'. Accepted
#                          values for position angle units are 'degree' or 'radian'

#         Read docstring of class Catalog for details on these attributes.
#         --------------------------------------------------------------------------
#         """

#         try:
#             name, frequency, location, flux_density
#         except NameError:
#             raise NameError('Catalog name, frequency, source location and flux density must be provided.')

#         self.location = NP.asarray(location)
#         self.flux_density = NP.asarray(flux_density).reshape(-1)
#         self.epoch = epoch
#         self.coords = coords
#         self.src_shape = None

#         if isinstance(name, (int, float, str)):
#             self.name = NP.repeat(NP.asarray(name).reshape(-1), flux_density.size)
#         elif isinstance(name, NP.ndarray):
#             if name.size == 1:
#                 self.name = NP.repeat(NP.asarray(name).reshape(-1), flux_density.size)
#             elif (name.size == flux_density.size):
#                 self.name = name.reshape(-1)
#             else:
#                 raise ValueError('Size of input "name" does not match number of objects')
#         else:
#             raise TypeError('Catalog name must be a integer, float, string or numpy array')

#         if isinstance(frequency, (int, float)):
#             self.frequency = NP.repeat(NP.asarray(frequency).reshape(-1), flux_density.size)
#         elif isinstance(frequency, NP.ndarray):
#             if frequency.size == 1:
#                 self.frequency = NP.repeat(NP.asarray(frequency).reshape(-1), flux_density.size)
#             elif (frequency.size == flux_density.size):
#                 self.frequency = frequency.reshape(-1)
#             else:
#                 raise ValueError('Size of input frequency does not match number of objects')
#         else:
#             raise TypeError('Catalog frequency must be a integer, float, or numpy array')

#         if spectral_index is None:
#             self.spectral_index = NP.zeros(self.flux_density.size).reshape(-1)
#         else: 
#             self.spectral_index = NP.asarray(spectral_index).reshape(-1)
        
#         if src_shape is not None:
#             self.src_shape = NP.asarray(src_shape)
#             if self.src_shape.shape[1] != 3:
#                 raise ValueError('Source shape must consist of three columns (major axis FWHM, minor axis FWHM, position angle) per source.')
#             if src_shape_units is not None:
#                 if not isinstance(src_shape_units, (list, tuple)):
#                     raise TypeError('Source shape units must be provided as a list or tuple')
#                 if len(src_shape_units) != 3:
#                     raise ValueError('Source shape units must contain three elements.')

#                 if src_shape_units[0] == 'arcsec':
#                     self.src_shape[:,0] = self.src_shape[:,0]/3.6e3
#                 elif src_shape_units[0] == 'arcmin':
#                     self.src_shape[:,0] = self.src_shape[:,0]/60.0
#                 elif src_shape_units[0] == 'radian':
#                     self.src_shape[:,0] = NP.degrees(self.src_shape[:,0])
#                 elif src_shape_units[0] != 'degree':
#                     raise ValueError('major axis FWHM must be specified as "arcsec", "arcmin", "degree" or "radian"')

#                 if src_shape_units[1] == 'arcsec':
#                     self.src_shape[:,1] = self.src_shape[:,1]/3.6e3
#                 elif src_shape_units[1] == 'arcmin':
#                     self.src_shape[:,1] = self.src_shape[:,1]/60.0
#                 elif src_shape_units[1] == 'radian':
#                     self.src_shape[:,1] = NP.degrees(self.src_shape[:,1])
#                 elif src_shape_units[0] != 'degree':
#                     raise ValueError('minor axis FWHM must be specified as "arcsec", "arcmin", "degree" or "radian"')

#                 if src_shape_units[2] == 'radian':
#                     self.src_shape[:,2] = NP.degrees(self.src_shape[:,2])
#                 elif src_shape_units[2] != 'degree':
#                     raise ValueError('position angle must be specified as "degree" or "radian" measured from north towards east.')

#         if src_shape is not None:
#             if (self.location.shape[0] != self.flux_density.size) or (self.flux_density.size != self.spectral_index.size) or (self.location.shape[0] != self.spectral_index.size) or (self.src_shape.shape[0] != self.flux_density.size):
#                 raise ValueError('location, flux_density, spectral_index, and src_shape must be provided for each source.')
#         else:
#             if (self.location.shape[0] != self.flux_density.size) or (self.flux_density.size != self.spectral_index.size) or (self.location.shape[0] != self.spectral_index.size):
#                 raise ValueError('location, flux_density, and spectral_index must be provided for each source.')            

#     #############################################################################

#     def match(self, other, matchrad=None, nnearest=0, maxmatches=-1):

#         """
#         -------------------------------------------------------------------------
#         Match the source positions in an instance of class Catalog with another
#         instance of the same class to a specified angular radius using
#         spherematch() in the geometry module

#         Inputs:

#         other       [2-column numpy array instance of class Catalog] A numpy
#                     array with two columns specifying the source positions in the
#                     other catalog or the other instance of class Catalog with
#                     which the current instance is to be matched with

#         matchrad    [scalar] Angular radius (in degrees) inside which matching 
#                     should occur. If not specified, if maxmatches is positive, 
#                     all the nearest maxmatches neighbours are found, and if
#                     maxmatches is not positive, the nnearest-th nearest neighbour
#                     specified by nnearest is found.

#         maxmatches  [scalar] The maximum number of matches (all of the maxmatches
#                     nearest neighbours) that lie within matchrad are found. If
#                     matchrad is not specified, all the maxmatches nearest
#                     neighbours are found. If maxmatches < 0, and matchrad is not
#                     set, then the nnearest-th nearest neighbour is found (which
#                     defaults to the nearest neighbour if nnearest <= 0)

#         nnearest    [scalar] nnearest-th neighbour to be found. Used only when
#                     maxmatches is not positive. If matchrad is set, the 
#                     specified neighbour is identified if found inside matchrad, 
#                     otherwise the nnearest-th neighbour is identified regardless
#                     of the angular distance.

#         Outputs:

#         m1          [list] List of indices of matches in the current instance of
#                     class Catalog
        
#         m2          [list] List of indices of matches in the other instance of
#                     class Catalog

#         d12         [list] List of angular distances between the matched subsets
#                     of the two catalogs indexed by m1 and m2 respectively
#         -------------------------------------------------------------------------
#         """

#         if not isinstance(other, (NP.ndarray, Catalog)):
#             raise TypeError('"other" must be a Nx2 numpy array or an instance of class Catalog.')
        
#         if isinstance(other, Catalog):
#             if (self.epoch == other.epoch) and (self.coords == other.coords):
#                 return GEOM.spherematch(self.location[:,0], self.location[:,1],
#                                         other.location[:,0],
#                                         other.location[:,1], matchrad,
#                                         nnearest, maxmatches)
#             else:
#                 raise ValueError('epoch and/or sky coordinate type mismatch. Cannot match.')
#         else:
#             return GEOM.spherematch(self.location[:,0], self.location[:,1],
#                                     other[:,0], other[:,1], matchrad,
#                                     nnearest, maxmatches)
        
#     #############################################################################

#     def subset(self, indices=None):

#         """
#         -------------------------------------------------------------------------
#         Provide a subset of the catalog using a list of indices onto the existing
#         catalog
        
#         Inputs:

#         indices    [list] List of indices of sources in the current instance of 
#                    class Catalog

#         Output:    [instance of class Catalog] An instance of class Catalog
#                    holding a subset of the sources in the current instance of 
#                    class Catalog 
#         -------------------------------------------------------------------------
#         """

#         if (indices is None) or (len(indices) == 0):
#             raise IndexError('No indices specified to select a catalog subset.')
#         else:
#             # return Catalog(self.frequency, self.location[indices, :], self.flux_density[indices, :], self.spectral_index[indices], self.epoch, self.coords)
#             return Catalog(NP.take(self.name, indices), NP.take(self.frequency, indices), NP.take(self.location, indices, axis=0), NP.take(self.flux_density, indices, axis=0), NP.take(self.spectral_index, indices), NP.take(self.src_shape, indices, axis=0), self.epoch, self.coords)

# #################################################################################

# class SkyModel:
    
#     """
#     -----------------------------------------------------------------------------
#     Class to manage sky model information. Sky model can be specified in the form
#     of a catalog, an image, or an image cube.

#     Attributes:

#     catalog       [instance of class Catalog] An instance of class Catalog that
#                   contains information about sources on the sky such as flux 
#                   densities, positions, spectral indices, etc.

#     image         [numpy array] Image of a part of the sky represented as a
#                   2-D numpy array 

#     cube          [numpy array] Image cube where the third axis represents 
#                   frequency usually

#     Member functions:

#     __init__()    Initialize an instance of class SkyModel
#     -----------------------------------------------------------------------------
#     """

#     #############################################################################

#     def __init__(self, catalog=None, image=None, cube=None):

#         """
#         -------------------------------------------------------------------------
#         Initialize an instance of class SkyModel

#         Attributes initialized are:
#         catalog, image, cube

#         Read the docstring of class SkyModel for more details on these attributes
#         -------------------------------------------------------------------------
#         """

#         if isinstance(catalog, Catalog):
#             self.catalog = catalog
#         self.image = image
#         self.cube = cube

#     def add_sources(self, catalog):
#         pass

#################################################################################

class SkyModel(object):

    """
    -----------------------------------------------------------------------------
    Class to manage sky model information.

    Attributes:

    name           [scalar or vector] Name of the catalog. If scalar, will be 
                   used for all sources in the sky model. If vector, will be 
                   used for corresponding object. If vector, size must equal 
                   the number of objects.

    frequency      [scalar or vector] Frequency range for which sky model is
                   applicable. Units in Hz.

    location       [numpy array or list of lists] Positions of the sources in 
                   sky model. Each position is specified as a row (numpy array)
                   or a 2-element list which is input as a list of lists for all
                   the sources in the sky model

    spec_type      [string] specifies the flux variation along the spectral 
                   axis. Allowed values are 'func' and 'spectrum'

    spec_parms     [dictionary] specifies spectral parameters applicable for 
                   different spectral types. It contains values in the following
                   keys:
                   'name'   [string] Specifies name of the functional variation
                            of spectrum. Applicable when spec_type is set to 
                            'func'. Allowed values are 'random', 'monotone', 
                            'power-law', and 'tanh'. Default='power-law' 
                            (with power law index set to 0). See member 
                            functions for these function definitions.
                   'power-law-index' 
                            [scalar numpy vector or list] Power law index for 
                            each object (flux ~ freq^power_law_index). Will be 
                            specified and applicable when value in key 'name' 
                            is set to 'power-law'. Same size as the number of 
                            object locations.
                   'freq-ref'
                            [scalar or numpy array or list] Reference or pivot
                            frequency as applicable. If a scalar, it is 
                            identical at all object locations. If a list or 
                            numpy array it must of size equal to the number of
                            objects, one value at each location. If
                            value under key 'name' is set to 'power-law', this
                            specifies the reference frequency at which the flux
                            density is specified under key 'flux-scale'. If 
                            value under key 'name' is 'monotone', this specifies
                            the frequency at which the spectrum of the object 
                            contains a spike and zero elsewhere. If value under
                            key 'name' is 'tanh', this specifies the frequency 
                            at which the spectrum is mid-way between min and 
                            max of the tanh function. This is not applicable 
                            when value under key 'name' is set to 'random' or 
                            'flat'. 
                   'flux-scale' 
                            [scalar or numpy array] Flux scale of the flux 
                            densities at object locations. If a scalar, it is
                            common for all object locations. If it is a vector, 
                            it has a size equal to the number of object 
                            locations, one value for each object location. If 
                            value in 'name' is set to 'power-law', this refers 
                            to the flux density scale at the reference frequency
                            specified under key 'freq-ref'. If value under key 
                            'name' is 'tanh', the flux density scale is half of
                            the value specified under this key.
                   'flux-offset'
                            [numpy vector] Flux density offset applicable after
                            applying the flux scale. Same units as the flux 
                            scale. If a scalar, it is common for all object 
                            locations. If it is a vector, it has a size equal 
                            to the number of object locations, one value 
                            for each object location. When value under the key
                            'name' is set to 'random', this amounts to setting
                            a mean flux density along the spectral axis.
                   'freq-width'
                            [numpy vector] Characteristic frequency full-width 
                            of the spectrum. Used if value under key 'name' is 
                            set to 'tanh' and it denotes the width of the inner 
                            50% of transition from max to min. Same units and 
                            size as value under key 'freq-ref'.

    src_shape      [3-column numpy array or list of 3-element lists] source 
                   shape specified by major axis FWHM (first column), minor axis 
                   FWHM (second column), and position angle (third column). The 
                   major and minor axes and position angle are stored in degrees. 
                   The number of rows must match the number of sources. Position 
                   angle is in degrees east of north (same convention as local
                   azimuth)

    epoch          [string] Epoch appropriate for the coordinate system. Default
                   is 'J2000'

    coords         [string] Coordinate system used for the source positions in 
                   the sky model. Currently accepted values are 'radec' (RA-Dec)

    Member Functions:

    __init__()     Initialize an instance of class SkyModel

    match()        Match the source positions in an instance of class 
                   SkyModel with another instance of the same class to a 
                   specified angular radius using spherematch() in the geometry 
                   module

    subset()       Provide a subset of the sky model using a list of indices onto
                   the existing sky model

    generate_spectrum()
                   Generate and return a spectrum from functional spectral 
                   parameters
    ------------------------------------------------------------------------------
    """

    ##############################################################################

    def __init__(self, name, frequency, location, spec_type, spec_parms=None, 
                 spectrum=None, src_shape=None, epoch='J2000', coords='radec', 
                 src_shape_units=None):

        """
        --------------------------------------------------------------------------
        Initialize an instance of class SkyModel

        Class attributes initialized are:
        frequency, location, flux_density, epoch, spectral_index, coords, and 
        src_shape

        Other input(s):

        src_shape_units  [3-element list or tuple of strings] Specifies the units 
                         of major axis FWHM, minor axis FWHM, and position angle.
                         Accepted values for major and minor axes units are 
                         'arcsec', 'arcmin', 'degree', or 'radian'. Accepted
                         values for position angle units are 'degree' or 'radian'

        Read docstring of class SkyModel for details on these attributes.
        --------------------------------------------------------------------------
        """

        try:
            name, frequency, location, spec_type
        except NameError:
            raise NameError('Catalog name, frequency, location, and spectral type must be provided.')

        self.location = NP.asarray(location)
        self.epoch = epoch
        self.coords = coords
        self.src_shape = None
        self.spec_parms = None

        if isinstance(name, (int, float, str)):
            self.name = NP.repeat(NP.asarray(name).reshape(-1), location.shape[0])
        elif isinstance(name, NP.ndarray):
            if name.size == 1:
                self.name = NP.repeat(NP.asarray(name).reshape(-1), location.shape[0])
            elif (name.size == location.shape[0]):
                self.name = name.reshape(-1)
            else:
                raise ValueError('Size of input "name" does not match number of objects')
        else:
            raise TypeError('Catalog name must be a integer, float, string or numpy array')

        if isinstance(spec_type, str):
            if spec_type in ['func', 'spectrum']:
                self.spec_type = spec_type
            else:
                raise ValueError('Spectrum specification in spec_type must be "func" or "spectrum"')
        else:
            raise TypeError('Spectrum specification in spec_type must be a string')

        if isinstance(frequency, (int, float, NP.ndarray)):
            self.frequency = NP.asarray(frequency).reshape(1,-1)
        else:
            raise TypeError('Sky model frequency must be a integer, float, or numpy array')

        if self.spec_type == 'spectrum':
            if spectrum is None:
                raise ValueError('Sky model spectrum not provided.')
            if not isinstance(spectrum, NP.ndarray):
                raise TypeError('Sky model spectrum must be a numpy array')
            if spectrum.shape != (self.location.shape[0], self.frequency.size):
                raise ValueError('Sky model spectrum must have same number of rows as number of object locations and same number of columns as number of frequency channels')
            self.spectrum = spectrum
        else:
            if spec_parms is None:
                spec_parms = {}
                spec_parms['name'] = NP.repeat('power-law', self.location.shape[0])
                spec_parms['power-law-index'] = NP.zeros(self.location.shape[0])
                spec_parms['freq-ref'] = NP.mean(self.frequency) + NP.zeros(self.location.shape[0])
                spec_parms['flux-scale'] = NP.ones(self.location.shape[0])
                spec_parms['flux-offset'] = NP.zeros(self.location.shape[0])
                spec_parms['freq-width'] = NP.zeros(self.location.shape[0])

            elif not isinstance(spec_parms, dict):
                raise TypeError('Spectral parameters in spec_parms must be specified as a dictionary')

            if not 'name' in spec_parms:
                spec_parms['name'] = NP.repeat('power-law', self.location.shape[0])

            if isinstance(spec_parms['name'], (list, NP.ndarray)):
                spec_parms['name'] = NP.asarray(spec_parms['name'])
                if spec_parms['name'].size != self.location.shape[0]:
                    raise ValueError('Number of spectral functional names should match the number of object locations.')
                uniq_names = NP.unique(spec_parms['name'])
                for name in uniq_names:
                    if name not in ['random', 'monotone', 'power-law', 'tanh']:
                        raise ValueError('Spectral functional names must be from "random", "monotone", "power-law" and "tanh".')

            else:
                raise TypeError('Values under key "name" in dictionary spec_parms must be a list or numpy array of strings')
            
            if 'flux-scale' not in spec_parms:
                spec_parms['flux-scale'] = NP.ones(self.location.shape[0])
            else:
                if not isinstance(spec_parms['flux-scale'], (int,float,NP.ndarray)):
                    raise TypeError('Flux scale must be a scalar or numpy array')
                spec_parms['flux-scale'] = NP.asarray(spec_parms['flux-scale'])
                if spec_parms['flux-scale'].size == 1:
                    spec_parms['flux-scale'] = spec_parms['flux-scale'] + NP.zeros(self.location.shape[0])
                elif spec_parms['flux-scale'].size == self.location.shape[0]:
                    spec_parms['flux-scale'] = spec_parms['flux-scale'].ravel()
                else:
                    raise ValueError('Size of flux scale must be equal to the number of sky locations')
                
                if NP.any(spec_parms['flux-scale'] <= 0.0):
                    raise ValueError('Flux scale values must be positive')

            if 'flux-offset' not in spec_parms:
                spec_parms['flux-offset'] = NP.zeros(self.location.shape[0])

            if 'freq-ref' not in spec_parms:
                spec_parms['freq-ref'] = NP.mean(self.frequency) + NP.zeros(self.location.shape[0])
            elif NP.any(spec_parms['freq-ref'] <= 0.0):
                raise ValueError('Reference frequency values must be positive')

            if 'power-law-index' not in spec_parms:
                spec_parms['power-law-index'] = NP.zeros(self.location.shape[0])

            if 'freq-width' not in spec_parms:
                spec_parms['freq-width'] = NP.zeros(self.location.shape[0])                
            elif NP.any(spec_parms['freq-width'] < 0.0):
                raise ValueError('Characteristic frequency widths must not be negative')

            self.spec_parms = spec_parms

            if src_shape is not None:
                self.src_shape = NP.asarray(src_shape)
                if self.src_shape.shape[1] != 3:
                    raise ValueError('Source shape must consist of three columns (major axis FWHM, minor axis FWHM, position angle) per source.')
                if src_shape_units is not None:
                    if not isinstance(src_shape_units, (list, tuple)):
                        raise TypeError('Source shape units must be provided as a list or tuple')
                    if len(src_shape_units) != 3:
                        raise ValueError('Source shape units must contain three elements.')
    
                    if src_shape_units[0] == 'arcsec':
                        self.src_shape[:,0] = self.src_shape[:,0]/3.6e3
                    elif src_shape_units[0] == 'arcmin':
                        self.src_shape[:,0] = self.src_shape[:,0]/60.0
                    elif src_shape_units[0] == 'radian':
                        self.src_shape[:,0] = NP.degrees(self.src_shape[:,0])
                    elif src_shape_units[0] != 'degree':
                        raise ValueError('major axis FWHM must be specified as "arcsec", "arcmin", "degree" or "radian"')
    
                    if src_shape_units[1] == 'arcsec':
                        self.src_shape[:,1] = self.src_shape[:,1]/3.6e3
                    elif src_shape_units[1] == 'arcmin':
                        self.src_shape[:,1] = self.src_shape[:,1]/60.0
                    elif src_shape_units[1] == 'radian':
                        self.src_shape[:,1] = NP.degrees(self.src_shape[:,1])
                    elif src_shape_units[0] != 'degree':
                        raise ValueError('minor axis FWHM must be specified as "arcsec", "arcmin", "degree" or "radian"')
    
                    if src_shape_units[2] == 'radian':
                        self.src_shape[:,2] = NP.degrees(self.src_shape[:,2])
                    elif src_shape_units[2] != 'degree':
                        raise ValueError('position angle must be specified as "degree" or "radian" measured from north towards east.')

                if src_shape.shape[0] != self.location.shape[0]:
                    raise ValueError('Number of source shapes in src_shape must match the number of object lcoations')

    #############################################################################

    def match(self, other, matchrad=None, nnearest=0, maxmatches=-1):

        """
        -------------------------------------------------------------------------
        Match the source positions in an instance of class SkyModel with 
        another instance of the same class to a specified angular radius using
        spherematch() in the geometry module

        Inputs:

        other       [2-column numpy array instance of class SkyModel] Numpy
                    array with two columns specifying the source positions in 
                    the other sky model or the other instance of class 
                    SkyModel with which the current instance is to be 
                    matched with

        matchrad    [scalar] Angular radius (in degrees) inside which matching 
                    should occur. If not specified, if maxmatches is positive, 
                    all the nearest maxmatches neighbours are found, and if
                    maxmatches is not positive, the nnearest-th nearest 
                    neighbour specified by nnearest is found.

        maxmatches  [scalar] The maximum number of matches (all of the 
                    maxmatches nearest neighbours) that lie within matchrad are 
                    found. If matchrad is not specified, all the maxmatches 
                    nearest neighbours are found. If maxmatches < 0, and matchrad 
                    is not set, then the nnearest-th nearest neighbour is found 
                    (which defaults to the nearest neighbour if nnearest <= 0)

        nnearest    [scalar] nnearest-th neighbour to be found. Used only when
                    maxmatches is not positive. If matchrad is set, the 
                    specified neighbour is identified if found inside matchrad, 
                    otherwise the nnearest-th neighbour is identified regardless
                    of the angular distance.

        Outputs:

        m1          [list] List of indices of matches in the current instance of
                    class SkyModel
        
        m2          [list] List of indices of matches in the other instance of
                    class SkyModel

        d12         [list] List of angular distances between the matched subsets
                    of the two sky models indexed by m1 and m2 respectively
        -------------------------------------------------------------------------
        """

        if not isinstance(other, (NP.ndarray, SkyModel)):
            raise TypeError('"other" must be a Nx2 numpy array or an instance of class SkyModel.')
        
        if isinstance(other, SkyModel):
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

    def subset(self, indices):

        """
        -------------------------------------------------------------------------
        Provide a subset of the sky model using a list of indices onto the 
        existing sky model
        
        Inputs:

        indices    [list or numpy array] Flattened list or numpy array of 
                   indices of sources in the current instance of class SkyModel

        Output:    [instance of class SkyModel] An instance of class 
                   SkyModel holding a subset of the sources in the current 
                   instance of class SkyModel 
        -------------------------------------------------------------------------
        """

        try:
            indices
        except NameError:
            return self

        if (indices is None) or (len(indices) == 0):
            return self
        else:
            indices = NP.asarray(indices).ravel()
            if self.spec_type == 'spectrum':
                if self.src_shape is not None:
                    return SkyModel(NP.take(self.name, indices), self.frequency, NP.take(self.location, indices, axis=0), self.spec_type, spectrum=NP.take(self.spectrum, indices, axis=0), src_shape=NP.take(self.src_shape, indices, axis=0), epoch=self.epoch, coords=self.coords)
                else:
                    return SkyModel(NP.take(self.name, indices), self.frequency, NP.take(self.location, indices, axis=0), self.spec_type, spectrum=NP.take(self.spectrum, indices, axis=0), epoch=self.epoch, coords=self.coords)
            else:
                spec_parms = {}
                spec_parms['name'] = NP.take(self.spec_parms['name'], indices)
                spec_parms['power-law-index'] = NP.take(self.spec_parms['power-law-index'], indices)
                spec_parms['freq-ref'] = NP.take(self.spec_parms['freq-ref'], indices)
                spec_parms['flux-scale'] = NP.take(self.spec_parms['flux-scale'], indices)
                spec_parms['flux-offset'] = NP.take(self.spec_parms['flux-offset'], indices)
                spec_parms['freq-width'] = NP.take(self.spec_parms['freq-width'], indices)
                if self.src_shape is not None:
                    return SkyModel(NP.take(self.name, indices), self.frequency, NP.take(self.location, indices, axis=0), self.spec_type, spec_parms=spec_parms, src_shape=NP.take(self.src_shape, indices, axis=0), epoch=self.epoch, coords=self.coords)
                else:
                    return SkyModel(NP.take(self.name, indices), self.frequency, NP.take(self.location, indices, axis=0), self.spec_type, spec_parms=spec_parms, epoch=self.epoch, coords=self.coords)                    

    #############################################################################

    def generate_spectrum(self, frequency=None):

        """
        -------------------------------------------------------------------------
        Generate and return a spectrum from functional spectral parameters

        Inputs:

        frequency  [scalar or numpy array] Frequencies at which the spectrum at
                   all object lcoations is to be created. Must be in same units
                   as the attribute frequency and values under keys 'freq-ref' 
                   and 'freq-width' of attribute spec_parms. If not provided 
                   (default=None), a spectrum is generated for all the 
                   frequencies specified in the attribute frequency and values 
                   under keys 'freq-ref' and 'freq-width' of attribute 
                   spec_parms. 

        Outputs:

        spectrum   [numpy array] Spectrum of the sky model at the respective
                   sky locations. Has shape nobj x nfreq.

        Power law calculation uses the convention, 
        spectrum = flux_offset + flux_scale * (freq/freq0)**spindex

        Monotone places a delta function at the frequency channel closest to the
        reference frequency if it lies inside the frequency range, otherwise
        a zero spectrum is assigned. 
        Thus spectrum = flux_scale * delta(freq-freq0)

        Random (currently only gaussian) places random fluxes in the spectrum
        spectrum = flux_offset + flux_scale * random_normal(freq.size)

        tanh spectrum is defined as 
        spectrum=flux_offset+flux_scale*(exp(a*x)-exp(-a*x))/(exp(a*x)+exp(-a*x))
        where, x = freq/freq0, and a = log((1+b)/(1-b)) / dx, where 
        dx = df/freq0, and df = two-sided frequency width at which the curve 
        transitions by a fraction b relative to height difference between max 
        and min, centered on the origin. 

        If the attribute spec_type is 'spectrum' the attribute spectrum is 
        returned without any modifications.
        -------------------------------------------------------------------------
        """

        if self.spec_type == 'func':
            if frequency is not None:
                if isisntance(frequency, (int,float,NP.ndarray)):
                    frequency = NP.asarray(frequency)
                else:
                    raise ValueError('Input parameter frequency must be a scalar or a numpy array')

                if NP.any(frequency <= 0.0):
                    raise ValueError('Input parameter frequency must contain positive values')
            else:
                frequency = NP.copy(self.frequency)

            spectrum = NP.empty((self.location.shape[0], frequency.size))
            spectrum.fill(NP.nan)

            uniq_names, invind = NP.unique(self.spec_parms['name'], return_inverse=True)
            if len(uniq_names) > 1:
                counts, edges, bnum, ri = OPS.binned_statistic(invind, statistic='count', bins=range(len(uniq_names)))
            else:
                counts = len(invind)
                ri = range(counts)

            for i, name in enumerate(uniq_names):
                if len(uniq_names) > 1:
                    indices = ri[ri[i]:ri[i+1]]
                else:
                    indices = ri

                if name == 'random':
                    spectrum[indices,:] = self.spec_parms['flux-offset'][indices].reshape(-1,1) + self.spec_parms['flux-scale'][indices].reshape(-1,1) * NP.random.randn(counts[i], frequency.size)
                if name == 'monotone':  # Needs serious testing
                    spectrum[indices,:] = 0.0
                    inpind, refind, dNN = LKP.find_1NN(frequency, self.spec_parms['freq-ref'][indices], distance=frequency[1]-frequency[0], remove_oob=True) 
                    ind = indices[inpind]
                    ind2d = zip(ind, refind)
                    spectrum[zip(*ind2d)] = self.spec_parms['flux-scale'][ind]
                if name == 'power-law':
                    spectrum[indices,:] = self.spec_parms['flux-offset'][indices].reshape(-1,1) + self.spec_parms['flux-scale'][indices].reshape(-1,1) * (frequency.reshape(1,-1)/self.spec_parms['freq-ref'][indices].reshape(-1,1))**self.spec_parms['power-law-index'][indices].reshape(-1,1)
                if name == 'tanh':
                    b = 0.5
                    x = 1 - frequency.reshape(1,-1)/self.spec_parms['freq-ref'][indices].reshape(-1,1)
                    df = self.spec_parms['freq-width'][indices]
                    dx = df / self.spec_parms['freq-ref'][indices]
                    a = NP.log((1+b)/(1-b)) / dx
                    a = a.reshape(-1,1)
                    spectrum[indices,:] = self.spec_parms['flux-offset'][indices].reshape(-1,1) + 0.5*self.spec_parms['flux-scale'][indices].reshape(-1,1) * NP.tanh(a*x)
                    
            return spectrum
        else:
            return self.spectrum

    #############################################################################
