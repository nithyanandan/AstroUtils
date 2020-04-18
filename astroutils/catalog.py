import multiprocessing as MP
import itertools as IT
import progressbar as PGB
import os, warnings
import numpy as NP
import healpy as HP
from astropy.table import Table
from astropy.io import fits, ascii
import h5py
import warnings
from astropy.coordinates import Angle, SkyCoord
from astropy import units
import scipy.constants as FCNST
from scipy.interpolate import interp1d
import geometry as GEOM
import mathops as OPS
import lookup_operations as LKP
import constants as CNST
import foregrounds as FG
try:
    from pygsm import GlobalSkyModel, GlobalSkyModel2016
except ImportError:
    pygsm_found = False
else:
    pygsm_found = True

#################################################################################

def healpix_smooth_and_udgrade_arg_splitter(args, **kwargs):
    return healpix_smooth_and_udgrade(*args, **kwargs)

def healpix_smooth_and_udgrade(input_map, fwhm, nside, order_in='RING', verbose=True):
    smooth_map = HP.smoothing(input_map.ravel(), fwhm, verbose=verbose)
    return HP.ud_grade(smooth_map, nside, order_in=order_in)

#################################################################################

def retrieve_external_spectrum(spec_extfile, ind=None):

    """
    -------------------------------------------------------------------------
    Retrieve an externally stored spectrum (may include recursive calls)

    Inputs:

    spec_extfile
               [string] full path to filename which contains externally
               stored spectrum. Must be specified (no default)

    ind        [scalar, list or numpy array] Indices to select objects in
               the externally stored spectrum of the catalog or sky model. 
               If set to None (default), all objects will be selected.

    Outputs:

    spectrum   [numpy array] Spectrum of the sky model at the specified
               sky locations. Has shape nobj x nfreq.
    -------------------------------------------------------------------------
    """

    try:
        spec_extfile
    except NameError:
        raise NameError('The externally stored spectrum file must be specified')

    if spec_extfile is None:
        raise TypeError('Input spec_extfile must be a string')
    if not isinstance(spec_extfile, str):
        raise TypeError('External filename spec_extfile must be a string')

    if ind is not None:
        if not isinstance(ind, (int,list,NP.ndarray)):
            raise TypeError('Input ind must be an integer, list or numpy array')
        else:
            ind = NP.asarray(ind).astype(NP.int)
            if NP.any(ind < 0):
                raise IndexError('Out of bound indices found in input ind')

    with h5py.File(spec_extfile, 'r') as fileobj:
        nobj = fileobj['object/name'].value.size
        if ind is None:
            ind = NP.arange(nobj, dtype=NP.int)
        else:
            if NP.any(ind >= nobj):
                raise ValueError('Specified indices exceed maximum number of objects in the external file')
        spec_type = fileobj['header/spec_type'].value
        if spec_type != 'spectrum':
            raise ValueError('Attribute spec_type not set to "spectrum" in external file {0}'.format(spec_extfile))
        if 'spectral_info/spectrum' in fileobj:
            return fileobj['spectral_info/spectrum'].value[ind,:]
        elif 'spectral_info/spec_extfile' in fileobj:
            next_spec_extfile = fileobj['spectral_info/spec_extfile'].value
            return retrieve_external_spectrum(next_spec_extfile, ind=ind) # Recursive call
        else:
            raise KeyError('Externally stored spectrum not found in {0}'.format(spec_extfile))

#################################################################################

def append_SkyModel_file(skymodfile, skymod, appendaxis, filemode='a'):

    """
    -----------------------------------------------------------------------------
    Append an instance of class SkyModel to an already existing file or create a
    new file.

    Inputs:

    skymodfile  [string] Pull path to HDF5 file (without .hdf5 extension) that
                contains saved information of an instance of class SkyModel. If
                it does not exist, it will be created. If it already exists, 
                skymod -- an instance of class SkyModel will be appended to it.

    skymod      [instance of class SkyModel] Instance of class SkyModel that will
                be appended on to the skymodfile if it exists or will be saved
                to skymodfile if the file does not exist already

    appendaxis  [string] Axis along which the specified skymod data has to be 
                appended. Accepted values are 'freq' (append along frequency
                axis) or 'src' (append along source location axis). All other 
                axes and attributes must match.

    filemode    [string] Mode in which the HDF5 must be opened. Accepted values
                are 'a' (Read/write if exists, create otherwise (default)) and 
                'w' (Create file, truncate if exists).
    -----------------------------------------------------------------------------
    """

    try:
        skymodfile, skymod, appendaxis
    except NameError:
        raise NameError('Inputs skymodfile, skymod, and appendaxis must be specified')

    if not isinstance(skymodfile, str):
        raise TypeError('Input skymodfile must be a string')

    if not isinstance(filemode, str):
        raise TypeError('Input filemode must be a string')
    else:
        filemode = filemode.lower()
        if filemode not in ['a', 'w']:
            raise ValueError('Invalid value specified for filemore')

    if not isinstance(skymod, SkyModel):
        raise TypeError('Input skymod must be an instance of class SkyModel')

    if not isinstance(appendaxis, str):
        raise TypeError('Input appendaxis must be a string')
    else:
        appendaxis = appendaxis.lower()
        if appendaxis not in ['src', 'freq']:
            raise ValueError('Invalid value specified for input appendaxis')

    if not os.path.isfile(skymodfile+'.hdf5'):
        skymod.save(skymodfile, fileformat='hdf5')
    else:
        with h5py.File(skymodfile+'.hdf5', filemode) as fileobj:
            hdr_group = fileobj['header']
            if hdr_group['spec_type'].value != skymod.spec_type:
                raise ValueError('The spectral type in the SkyModel instance and the skymodfile do not match')
            object_group = fileobj['object']
            if object_group.attrs['epoch'] != skymod.epoch:
                raise ValueError('The epochs in the SkyModel instance and the skymodfile do not match')
            if object_group.attrs['coords'] != skymod.coords:
                raise ValueError('The coordinate system in the SkyModel instance and the skymodfile do not match')
            src_shape_in_file = 'shape' in object_group
            src_shape_in_skymod = skymod.src_shape is not None
            if src_shape_in_file != src_shape_in_skymod:
                raise KeyError('src_shape is not consistent between the SkyModel instance and the skymodfile')
            if skymod.coords == 'radec':
                lon = 'RA'
                lat = 'Dec'
            else:
                lon = 'Az'
                lat = 'Alt'
            spec_group = fileobj['spectral_info']
            if appendaxis == 'src':
                if skymod.frequency.size != spec_group['freq'].size:
                    raise IndexError('The frequencies in the skymodefile and the SkyModel instance do not match')
                if NP.any(NP.abs(skymod.frequency - spec_group['freq'].value) > 1e-14):
                    raise ValueError('The frequencies in the skymodefile and the SkyModel instance do not match')
                object_group['name'].resize(object_group['name'].size+skymod.name.size, axis=0)
                object_group['name'][-skymod.name.size:] = skymod.name
                object_group[lon].resize(object_group[lon].size+skymod.location.shape[0], axis=0)
                object_group[lat].resize(object_group[lat].size+skymod.location.shape[0], axis=0)
                if skymod.coords == 'radec':
                    object_group[lon][-skymod.locations.shape[0]:] = skymod.location[:,0]
                    object_group[lat][-skymod.locations.shape[0]:] = skymod.location[:,1]
                else:
                    object_group[lon][-skymod.locations.shape[0]:] = skymod.location[:,1]
                    object_group[lat][-skymod.locations.shape[0]:] = skymod.location[:,0]
                if src_shape_in_file:
                    object_group['shape'].resize(object_group['shape'].shape[0]+skymod.src_shape.shape[0], axis=0)
                    object_group['shape'][-skymod.src_shape.shape[0]:,:] = skymod.src_shape
                if skymod.spec_type == 'func':
                    spec_group['func-name'].resize(spec_group['func-name'].size+skymod.spec_parms['name'].size, axis=0)
                    spec_group['func-name'][-skymod.name.size:] = skymod.spec_parms['name']
                    spec_group['freq'].resize(spec_group['freq'].size+skymod.spec_parms['freq-ref'].size, axis=0)
                    spec_group['freq'][-skymod.name.size:] = skymod.spec_parms['freq-ref']
                    spec_group['flux_density'].resize(spec_group['flux_density'].size+skymod.spec_parms['flux-scale'].size, axis=0)
                    spec_group['flux_density'][-skymod.name.size:] = skymod.spec_parms['flux-scale']
                    if ('spindex' in spec_group) and ('power-law-index' in skymod.spec_parms):
                        spec_group['spindex'].resize(spec_group['spindex'].size+skymod.spec_parms['power-law-index'].size, axis=0)
                        spec_group['spindex'][-skymod.name.size:] = skymod.spec_parms['power-law-index']
                else:
                    spec_group['spectrum'].resize(spec_group['spectrum'].shape[0]+skymod.spectrum.shape[0], axis=0)
                    spec_group['spectrum'][-skymod.name.size:,:] = skymod.spectrum
            else:
                if skymod.name.size != object_group['name'].size:
                    raise IndexError('The objects in the skymodefile and the SkyModel instance do not match')
                if NP.any(skymod.name != object_group['name'].value):
                    raise ValueError('The objects in the skymodefile and the SkyModel instance do not match')
                if skymod.coords == 'radec':
                    if NP.any(NP.abs(skymod.location - NP.hstack((object_group['RA'].value.reshape(-1,1), object_group['Dec'].value.reshape(-1,1)))) > 1e-14):
                        raise ValueError('The locations in the skymodefile and the SkyModel instance do not match')
                else:
                    if NP.any(NP.abs(skymod.location - NP.hstack((object_group['Alt'].value.reshape(-1,1), object_group['Az'].value.reshape(-1,1)))) > 1e-14):
                        raise ValueError('The locations in the skymodefile and the SkyModel instance do not match')
                if src_shape_in_file:
                    if NP.any(NP.abs(skymod.src_shape - object_group['shape'].value) > 1e-14):
                        raise ValueError('The locations in the skymodefile and the SkyModel instance do not match')
                spec_group['freq'].resize(spec_group['freq'].size+skymod.frequency.size, axis=0)
                spec_group['freq'][-skymod.frequency.size:] = skymod.frequency.ravel()
                if skymod.spec_type == 'spectrum':
                    spec_group['spectrum'].resize(spec_group['spectrum'].shape[1]+skymod.spectrum.shape[1], axis=1)
                    spec_group['spectrum'][:,-skymod.name.size:] = skymod.spectrum

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

    is_healpix     [boolean] If True, it is a healpix map with ordering as 
                   specified in attribute healpix_ordering. By default it is 
                   set to False

    healpix_ordering
                   [string] specifies the ordering of healpix pixels if 
                   is_healpix is set to True in which case it is set to 'nest'
                   or 'ring'. If is_healpix is set to False, it is set to 'na'
                   (default) indicating 'not applicable'

    spec_type      [string] specifies the flux variation along the spectral 
                   axis. Allowed values are 'func' and 'spectrum'. If set to 
                   'func', values under spec_parms are applicable. If set to 
                   'spectrum', values under key 'spectrum' are applicable.

    spec_parms     [dictionary] specifies spectral parameters applicable for 
                   different spectral types. Only applicable if spec_type is 
                   set to 'func'. It contains values in the following
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
                   'z-width'
                            [numpy vector] Characteristic redshift full-wdith
                            in the definition of tanh expression applicable to
                            global EoR signal. 

    spectrum       [numpy array] Spectrum of the catalog. Will be applicable 
                   if attribute spec_type is set to 'spectrum' or if spectrum
                   was computed using the member function. It will be of shape
                   nsrc x nchan. If not specified or set to None, an external 
                   file must be specified under attribute 'spec_extfile'

    spec_extfile   [string] full path filename of external file containing saved 
                   version of instance of class SkyModel which contains an 
                   offline version of full spectrum data. This will be 
                   applicable only if spec_type is set to 'spectrum'. If set to
                   None, full spectrum will be provided under attribute 
                   'spectrum'

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
                   the existing sky model. Subset can be either in position or 
                   frequency channels

    generate_spectrum()
                   Generate and return a spectrum from functional spectral 
                   parameters

    load_external_spectrum()
                   Load full spectrum from external file

    to_healpix()   Convert catalog to a healpix format of given nside at 
                   specified frequencies.

    save()         Save sky model to the specified output file
    ------------------------------------------------------------------------------
    """

    ##############################################################################

    def __init__(self, init_file=None, init_parms=None, load_spectrum=False):

        """
        --------------------------------------------------------------------------
        Initialize an instance of class SkyModel

        Class attributes initialized are:
        frequency, location, flux_density, epoch, spectral_index, coords, 
        spectrum, src_shape

        Inputs:

        init_file   [string] Full path to the file containing saved 
                    information of an instance of class SkyModel. If set to
                    None (default), parameters in the input init_parms 
                    will be used to initialize an instance of this class. 
                    If set to a filename, the instance will be initialized from 
                    this file, and init_parms and its parameters will be 
                    ignored.

        init_parms  [dictionary] Dictionary containing parameters used to create
                    an instance of class SkyModel. Used only if init_file is set
                    to None. Initialization parameters are specified using the
                    following keys and values:
                    'name'      [scalar or vector] Name of the catalog. If 
                                scalar, will be used for all sources in the sky 
                                model. If vector, will be used for corresponding 
                                object. If vector, size must equal the number of 
                                objects.
                    'frequency' [scalar or vector] Frequency range for which sky 
                                model is applicable. Units in Hz.
                    'location'  [numpy array or list of lists] Positions of the 
                                sources in sky model. Each position is specified 
                                as a row (numpy array) or a 2-element list which 
                                is input as a list of lists for all the sources 
                                in the sky model
                    'is_healpix'
                                [boolean] If True, it is a healpix map with 
                                ordering as specified in attribute 
                                healpix_ordering. If not specified, it will be 
                                assumed to be False
                    'healpix_ordering'
                                [string] specifies the ordering of healpix 
                                pixels if is_healpix is set to True in which 
                                case it is set to 'nest' or 'ring'. If 
                                is_healpix is set to False, it will be assumed 
                                to be set to 'na' (default) indicating 'not 
                                applicable'
                    'spec_type' [string] specifies the flux variation along the 
                                spectral axis. Allowed values are 'func' and 
                                'spectrum'. If set to 'func', values under 
                                spec_parms are applicable. If set to 'spectrum', 
                                values under key 'spectrum' are applicable.
                    'spec_parms'
                                [dictionary] specifies spectral parameters 
                                applicable for different spectral types. Only 
                                applicable if spec_type is set to 'func'. It 
                                contains values in the following keys:
                                'name'   [string] Specifies name of the 
                                         functional variation of spectrum. 
                                         Applicable when spec_type is set to 
                                         'func'. Allowed values are 'random', 
                                         'monotone', 'power-law', and 'tanh'. 
                                         Default='power-law' (with power law 
                                         index set to 0). See member functions 
                                         for these function definitions.
                                'power-law-index' 
                                         [scalar numpy vector or list] Power 
                                         law index for each object 
                                         (flux ~ freq^power_law_index). Will be 
                                         specified and applicable when value in 
                                         key 'name' is set to 'power-law'. Same 
                                         size as the number of object locations.
                                'freq-ref'
                                         [scalar or numpy array or list] 
                                         Reference or pivot frequency as 
                                         applicable. If a scalar, it is 
                                         identical at all object locations. If a 
                                         list or numpy array it must of size 
                                         equal to the number of objects, one 
                                         value at each location. If value under 
                                         key 'name' is set to 'power-law', this 
                                         specifies the reference frequency at 
                                         which the flux density is specified 
                                         under key 'flux-scale'. If value under 
                                         key 'name' is 'monotone', this 
                                         specifies the frequency at which the 
                                         spectrum of the object contains a spike 
                                         and zero elsewhere. If value under key 
                                         'name' is 'tanh', this specifies the 
                                         frequency at which the spectrum is 
                                         mid-way between min and max of the tanh 
                                         function. This is not applicable when 
                                         value under key 'name' is set to 
                                         'random' or 'flat'. 
                                'flux-scale' 
                                         [scalar or numpy array] Flux scale of 
                                         the flux densities at object locations. 
                                         If a scalar, it is common for all 
                                         object locations. If it is a vector, it 
                                         has a size equal to the number of 
                                         object locations, one value for each 
                                         object location. If value in 'name' is 
                                         set to 'power-law', this refers to the 
                                         flux density scale at the reference 
                                         frequency specified under key 
                                         'freq-ref'. If value under key 'name' 
                                         is 'tanh', the flux density scale is 
                                         half of the value specified under this 
                                         key.
                                'flux-offset'
                                         [numpy vector] Flux density offset 
                                         applicable after applying the flux 
                                         scale. Same units as the flux scale. If 
                                         a scalar, it is common for all object 
                                         locations. If it is a vector, it has a 
                                         size equal to the number of object 
                                         locations, one value for each object 
                                         location. When value under the key 
                                         'name' is set to 'random', this amounts 
                                         to setting a mean flux density along 
                                         the spectral axis.
                                'z-width'
                                         [numpy vector] Characteristic redshift 
                                         full-wdith in the definition of tanh 
                                         expression applicable to global EoR 
                                         signal. 
                    'spectrum'  [numpy array] Spectrum of the catalog. Will be 
                                applicable if attribute spec_type is set to 
                                'spectrum'. It must be of shape nsrc x nchan.
                                If not specified or set to None, an external 
                                file must be specified under 'spec_extfile'
                                and the spectrum can be determined from offline
                                data

                    'spec_extfile'
                                [string] full path filename of external file 
                                containing saved version of instance of class
                                SkyModel which contains an offline version of 
                                full spectrum data. This will be applicable 
                                only if spec_type is set to 'spectrum'. If set
                                to None, full spectrum will be provided under
                                key 'spectrum'

                    'src_shape' [3-column numpy array or list of 3-element 
                                lists] source shape specified by major axis 
                                FWHM (first column), minor axis FWHM (second 
                                column), and position angle (third column). The 
                                major and minor axes and position angle are 
                                stored in degrees. The number of rows must match 
                                the number of sources. Position angle is in 
                                degrees east of north (same convention as local 
                                azimuth)
                    
                    'epoch'     [string] Epoch appropriate for the coordinate 
                                system. Default is 'J2000'

                    'coords'    [string] Coordinate system used for the source 
                                positions in the sky model. Currently accepted 
                                values are 'radec' (RA-Dec)

                    'src_shape_units' 
                                [3-element list or tuple of strings] Specifies 
                                the units of major axis FWHM, minor axis FWHM, 
                                and position angle. Accepted values for major 
                                and minor axes units are 'arcsec', 'arcmin', 
                                'degree', or 'radian'. Accepted values for 
                                position angle units are 'degree' or 'radian'

        load_spectrum
                    [boolean] It is applicable only if initialization happens from
                    init_file and if full spectrum is available in the group
                    'spectral_info/spectrum'. If set to True, it will load the
                    the full spectrum into the instance attribute spectrum. If
                    set to False (default), it will only store the path to the 
                    file containing the full spectrum under the attribute 
                    spec_extfile and not load the full spectrum into the instance
                    attribute.
        --------------------------------------------------------------------------
        """

        if init_file is not None:
            with h5py.File(init_file, 'r') as fileobj:
                # for key in fileobj.keys():
                for key in ['header', 'object', 'spectral_info']:
                    grp = fileobj[key]
                    if key == 'header':
                        self.spec_type = grp['spec_type'].value
                        self.is_healpix = False
                        self.healpix_ordering = 'na'
                        if 'is_healpix' in grp:
                            self.is_healpix = bool(grp['is_healpix'].value)
                            if self.is_healpix:
                                if 'healpix_ordering' in grp:
                                    if grp['healpix_ordering'].value.lower() in ['nest', 'ring']:
                                        self.healpix_ordering = grp['healpix_ordering'].value.lower()
                    if key == 'object':
                        self.epoch = grp.attrs['epoch']
                        self.coords = grp.attrs['coords']
                        self.name = grp['name'].value
                        if self.coords == 'radec':
                            self.location = NP.hstack((grp['RA'].value.reshape(-1,1), grp['Dec'].value.reshape(-1,1)))
                        elif self.coords == 'altaz':
                            self.location = NP.hstack((grp['Alt'].value.reshape(-1,1), grp['Az'].value.reshape(-1,1)))
                        if 'shape' in grp:
                            self.src_shape = grp['shape'].value
                        else:
                            self.src_shape = None
                    if key == 'spectral_info':
                        self.spec_extfile = None
                        self.spectrum = None
                        self.frequency = grp['freq'].value.reshape(1,-1)
                        self.spec_parms = {}
                        if self.spec_type == 'func':
                            self.spec_parms['name'] = grp['func-name'].value
                            self.spec_parms['freq-ref'] = grp['freq-ref'].value
                            self.spec_parms['flux-scale'] = grp['flux_density'].value
                            if 'spindex' in grp:
                                self.spec_parms['power-law-index'] = grp['spindex'].value
                        else:
                            self.frequency = grp['freq'].value.reshape(1,-1)
                            if 'spec_extfile' in grp:
                                self.spec_extfile = grp['spec_extfile'].value
                            else:
                                if not isinstance(load_spectrum, bool):
                                    load_spectrum = False
                                if load_spectrum:
                                    self.spectrum = grp['spectrum'].value
                                else:
                                    self.spec_extfile = init_file
                            # self.frequency = grp['freq'].value.reshape(1,-1)
                            self.spectrum = grp['spectrum'].value
        elif (init_parms is None):
            raise ValueError('In the absence of init_file, init_parms must be provided for initialization')
        else:
            if not isinstance(init_parms, dict):
                raise TypeError('Input init_parms must be a dictionary')

            try:
                name = init_parms['name']
                frequency = init_parms['frequency']
                location = init_parms['location']
                spec_type = init_parms['spec_type']
            except KeyError:
                raise KeyError('Catalog name, frequency, location, and spectral type must be provided.')

            self.is_healpix = False
            self.healpix_ordering = 'na'
            if 'is_healpix' in init_parms:
                if isinstance(init_parms['is_healpix'], bool):
                    self.is_healpix = init_parms['is_healpix']
                    if 'healpix_ordering'in init_parms:
                        if not isinstance(init_parms['healpix_ordering'], str):
                            raise TypeError('alue under key "healpix_ordering" must be a string')
                        if init_parms['healpix_ordering'].lower() not in ['nest', 'ring']:
                            raise ValueError('Invalid specification for value under key "healpix_ordering"')
                        self.healpix_ordering = init_parms['healpix_ordering']

            if 'spec_parms' in init_parms:
                spec_parms = init_parms['spec_parms']
            else:
                spec_parms = None

            if 'src_shape' in init_parms:
                src_shape = init_parms['src_shape']
            else:
                src_shape = None

            if 'src_shape_units' in init_parms:
                src_shape_units = init_parms['src_shape_units']
            else:
                src_shape_units = None
                
            if 'coords' in init_parms:
                self.coords = init_parms['coords']
            else:
                self.coords = 'radec'

            if 'epoch' in init_parms:
                self.epoch = init_parms['epoch']
            else:
                self.epoch = 'J2000'
                
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
    
            self.location = location
            self.spectrum = None
            self.spec_extfile= None
            if self.spec_type == 'spectrum':
                check_spectrum = False
                check_extfile = False
                if 'spec_extfile' in init_parms:
                    if init_parms['spec_extfile'] is None:
                        check_spectrum = True
                        warnings.warn('No value specified under key "spec_extfile". Will check for value under skey "spectrum"')
                    elif isinstance(init_parms['spec_extfile'], str):
                        self.spec_extfile = init_parms['spec_extfile']
                        spectrum = retrieve_external_spectrum(init_parms['spec_extfile'], ind=None)
                        if not isinstance(spectrum, NP.ndarray):
                            raise TypeError('Spectrum in external file is not a numpy array')
                        if spectrum.shape != (self.location.shape[0], self.frequency.size):
                            raise ValueError('Spectrum data in external file does not have compatible dimensions with number of objects and number of frequency channels')
                        del spectrum

                        # with h5py.File(self.spec_extfile, 'r') as fileobj:
                        #     try:
                        #         assert fileobj['spectral_info/spectrum'].value is not None, 'Data in external file must not be None'
                        #         assert fileobj['spectral_info/spectrum'].value.shape == (self.location.shape[0], self.frequency.size), 'Data in external file must not be None'
                        #     except KeyError:
                        #         raise KeyError('Key "spectral_info/spectrum" not found in external file')
                        #     check_extfile = True
                    else:
                        raise TypeError('Value under spec_extfile must be a string')
                else:
                    check_spectrum = True
                if check_spectrum:
                    if 'spectrum' not in init_parms:
                        raise KeyError('Sky model spectrum not provided.')
                    spectrum = init_parms['spectrum']
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
                    spec_parms['z-width'] = NP.zeros(self.location.shape[0])
                elif not isinstance(spec_parms, dict):
                    raise TypeError('Spectral parameters in spec_parms must be specified as a dictionary')
    
                if 'name' not in spec_parms:
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
    
                if 'z-width' not in spec_parms:
                    spec_parms['z-width'] = NP.zeros(self.location.shape[0])
                elif NP.any(spec_parms['z-width'] < 0.0):
                    raise ValueError('Characteristic redshift widths must not be negative')
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

    def subset(self, indices, axis='position'):

        """
        -------------------------------------------------------------------------
        Provide a subset of the sky model using a list of indices onto the 
        existing sky model. Subset can be either in position or frequency 
        channels
        
        Inputs:

        indices    [list or numpy array] Flattened list or numpy array of 
                   indices of sources in the current instance of class SkyModel

        axis       [string] the axis to take the subset along. Currently 
                   accepted values are 'position' (default) and 'spectrum' 
                   which indicates the indices are to be used along the 
                   position or spectrum axis respectively to obtain the subset.
                   When spectral axis is specified by spec_type='func', then 
                   there will be no slicing along the spectral axis and will 
                   return the original instance. 

        Output:    [instance of class SkyModel] An instance of class 
                   SkyModel holding a subset of the sources in the current 
                   instance of class SkyModel 
        -------------------------------------------------------------------------
        """

        try:
            indices
        except NameError:
            return self

        if axis not in ['position', 'spectrum']:
            raise ValueError('input axis must be along position or spectrum')

        if (indices is None) or (len(indices) == 0):
            return self
        else:
            init_parms = {}
            if axis == 'position':
                indices = NP.asarray(indices).ravel()
                init_parms = {'name': NP.take(self.name, indices), 'frequency': self.frequency, 'location': NP.take(self.location, indices, axis=0), 'spec_type': self.spec_type, 'epoch': self.epoch, 'coords': self.coords}
                if self.spec_type == 'spectrum':
                    if self.spectrum is not None:
                        init_parms['spectrum'] = NP.take(self.spectrum, indices, axis=0)
                    elif self.spec_extfile is not None:
                        init_parms['spectrum'] = retrieve_external_spectrum(self.spec_extfile, ind=indices)
                    else:
                        raise AttributeError('Neither attribute "spectrum" nor "spec_extfile" found in the instance')

                    if self.src_shape is not None:
                        init_parms['src_shape'] = NP.take(self.src_shape, indices, axis=0)
                else:
                    spec_parms = {}
                    spec_parms['name'] = NP.take(self.spec_parms['name'], indices)
                    spec_parms['power-law-index'] = NP.take(self.spec_parms['power-law-index'], indices)
                    spec_parms['freq-ref'] = NP.take(self.spec_parms['freq-ref'], indices)
                    spec_parms['flux-scale'] = NP.take(self.spec_parms['flux-scale'], indices)
                    spec_parms['flux-offset'] = NP.take(self.spec_parms['flux-offset'], indices)
                    spec_parms['z-width'] = NP.take(self.spec_parms['z-width'], indices)                
                    init_parms['spec_parms'] = spec_parms
                    if self.src_shape is not None:
                        init_parms['src_shape'] = NP.take(self.src_shape, indices, axis=0)
            else:
                indices = NP.asarray(indices).ravel()
                init_parms = {'name': self.name, 'frequency': NP.take(self.frequency, indices, axis=1), 'location': self.location, 'spec_type': self.spec_type, 'epoch': self.epoch, 'coords': self.coords}
                if self.src_shape is not None:
                    init_parms['src_shape'] = self.src_shape
                if self.spec_type == 'func':
                    init_parms['spec_parms'] = self.spec_parms
                else:
                    if self.spectrum is not None:
                        init_parms['spectrum'] = NP.take(self.spectrum, indices, axis=1)
                    elif self.spec_extfile is not None:
                        spectrum = retrieve_external_spectrum(self.spec_extfile, ind=None)
                        init_parms['spectrum'] = NP.take(spectrum, indices, axis=1)
                    else:
                        raise AttributeError('Neither attribute "spectrum" nor "spec_extfile" found in the instance')

            return SkyModel(init_parms=init_parms, init_file=None)

    #############################################################################

    def generate_spectrum(self, ind=None, frequency=None, interp_method='linear'):

        """
        -------------------------------------------------------------------------
        Generate and return a spectrum from functional spectral parameters

        Inputs:

        ind        [scalar, list or numpy array] Indices to select objects in
                   the catalog or sky model. If set to None (default), all 
                   objects will be selected.

        frequency  [scalar or numpy array] Frequencies at which the spectrum at
                   all object locations is to be created. Must be in same units
                   as the attribute frequency and values under key 'freq-ref' 
                   of attribute spec_parms. If not provided (default=None), a 
                   spectrum is generated for all the frequencies specified in 
                   the attribute frequency and values under keys 'freq-ref' and
                   'z-width' of attribute spec_parms. 

        interp_method 
                   [string] Specified kind of interpolation to be used if 
                   self.spec_type is set to 'spectrum'. Default='linear'. 
                   Accepted values are described in docstring of 
                   scipy.interpolate.interp1d() or as a power law index 
                   specified by 'power-law'

        Outputs:

        spectrum   [numpy array] Spectrum of the sky model at the specified
                   sky locations. Has shape nobj x nfreq.

        Power law calculation uses the convention, 
        spectrum = flux_offset + flux_scale * (freq/freq0)**spindex

        Monotone places a delta function at the frequency channel closest to the
        reference frequency if it lies inside the frequency range, otherwise
        a zero spectrum is assigned. 
        Thus spectrum = flux_scale * delta(freq-freq0)

        Random (currently only gaussian) places random fluxes in the spectrum
        spectrum = flux_offset + flux_scale * random_normal(freq.size)

        tanh spectrum is defined as (Bowman & Rogers 2010):
        spectrum = flux_scale * sqrt((1+z)/10) * 0.5 * [tanh((z-zr)/dz) + 1]
        where, flux_scale is typically 0.027 K, zr = reionization redshift 
        when x_i = 0.5, and dz = redshift width (dz ~ 1)

        If the attribute spec_type is 'spectrum' the attribute spectrum is 
        returned on the selected indices and requested spectral interpolation
        method
        -------------------------------------------------------------------------
        """

        if ind is None:
            ind = NP.arange(self.location.shape[0], dtype=NP.int)
        elif not isinstance(ind, (int,list,NP.ndarray)):
            raise TypeError('Input ind must be an integer, list or numpy array')
        else:
            ind = NP.asarray(ind).astype(NP.int)
            if NP.any(NP.logical_or(ind < 0, ind >= self.location.shape[0])):
                raise IndexError('Out of bound indices found in input ind')

        if frequency is not None:
            if isinstance(frequency, (int,float,NP.ndarray)):
                frequency = NP.asarray(frequency).ravel()
            else:
                raise ValueError('Input parameter frequency must be a scalar or a numpy array')

            if NP.any(frequency <= 0.0):
                raise ValueError('Input parameter frequency must contain positive values')
        else:
            frequency = NP.copy(self.frequency)

        if self.spec_type == 'func':
            spectrum = NP.empty((ind.size, frequency.size))
            spectrum.fill(NP.nan)

            uniq_names, invind = NP.unique(self.spec_parms['name'][ind], return_inverse=True)
            if len(uniq_names) > 1:
                counts, edges, bnum, ri = OPS.binned_statistic(invind, statistic='count', bins=range(len(uniq_names)))
            else:
                counts = len(invind)
                ri = range(counts)

            for i, name in enumerate(uniq_names):
                if len(uniq_names) > 1:
                    ind_to_be_filled = NP.asarray(ri[ri[i]:ri[i+1]])
                else:
                    ind_to_be_filled = NP.asarray(ri)
                ind_to_be_used = ind[ind_to_be_filled]

                if name == 'random':
                    spectrum[ind_to_be_filled,:] = self.spec_parms['flux-offset'][ind_to_be_used].reshape(-1,1) + self.spec_parms['flux-scale'][ind_to_be_used].reshape(-1,1) * NP.random.randn(counts[i], frequency.size)
                if name == 'monotone':  # Needs serious testing
                    spectrum[ind_to_be_filled,:] = 0.0
                    inpind, refind, dNN = LKP.find_1NN(frequency, self.spec_parms['freq-ref'][ind_to_be_used], distance=frequency[1]-frequency[0], remove_oob=True) 
                    ind_match = ind_to_be_used[inpind]
                    ind2d = zip(ind_match, refind)
                    spectrum[zip(*ind2d)] = self.spec_parms['flux-scale'][ind_match]
                if name == 'power-law':
                    spectrum[ind_to_be_filled,:] = self.spec_parms['flux-offset'][ind_to_be_used].reshape(-1,1) + self.spec_parms['flux-scale'][ind_to_be_used].reshape(-1,1) * (frequency.reshape(1,-1)/self.spec_parms['freq-ref'][ind_to_be_used].reshape(-1,1))**self.spec_parms['power-law-index'][ind_to_be_used].reshape(-1,1)
                if name == 'tanh':
                    z = CNST.rest_freq_HI/frequency.reshape(1,-1) - 1
                    zr = CNST.rest_freq_HI/self.spec_parms['freq-ref'][ind_to_be_used].reshape(-1,1) - 1
                    dz = self.spec_parms['z-width'][ind_to_be_used].reshape(-1,1)

                    amp = self.spec_parms['flux-scale'][ind_to_be_used].reshape(-1,1) * NP.sqrt((1+z)/10)
                    xh = 0.5 * (NP.tanh((z-zr)/dz) + 1)
                    spectrum[ind_to_be_filled,:] = amp * xh
                    
            return spectrum
        else:
            if not isinstance(interp_method, str):
                raise TypeError('Input interp_method must be a string')
            if NP.any(NP.logical_or(frequency < self.frequency.min(), frequency > self.frequency.max())):
                raise ValueError('Frequencies requested in output lie out of range of sky model frequencies and hence cannot be interpolated')
            if self.spectrum is not None:
                spectrum = NP.take(self.spectrum, ind, axis=0)
            elif self.spec_extfile is not None:
                spectrum = retrieve_external_spectrum(self.spec_extfile, ind=ind)
            else:
                raise AttributeError('Neither attribute "spectrum" nor "spec_extfile" found in the instance')
            
            if self.frequency.size == frequency.size:
                if NP.all(NP.abs(self.frequency - frequency) < 1e-10):
                    return spectrum

            if interp_method.lower() == 'power-law':
                spindex = FG.power_law_spectral_index(self.frequency.ravel(), spectrum)
                return spectrum[:,int(self.frequency.size/2)].reshape(-1,1) * (frequency.ravel()/self.frequency.ravel()[int(self.frequency.size/2)]).reshape(1,-1)**spindex.reshape(-1,1)
            else:
                interp_func = interp1d(self.frequency.ravel(), spectrum, axis=1, kind=interp_method)
                return interp_func(frequency)

    #############################################################################

    def load_external_spectrum(self):

        """
        -------------------------------------------------------------------------
        Load full spectrum from external file
        -------------------------------------------------------------------------
        """

        if self.spec_type == 'spectrum':
            if self.spectrum is not None:
                if self.spec_extfile is not None:
                    raise ValueError('Both attributes spectrum and spec_extfile are set. This will overwrite the existing values in attribute spectrum')
                else:
                    warnings.warn('Attribute spec_extfile is not set. Continuing without any action.')
            elif self.spec_extfile is not None:
                self.spectrum = retrieve_external_spectrum(self.spec_extfile, ind=None)
                if self.spectrum.shape != (self.location.shape[0], self.frequency.size):
                    raise ValueError('Dimensions of external spectrum incompatible with expected number of sources and spectral channels')
                self.spec_extfile = None
            else:
                raise AttributeError('Neither attribute spectrum not spec_extfile is set')
        else:
            warnings.warn('Attribute spec_type is not spectrum. Continuing without any action.')

    #############################################################################

    def to_healpix(self, freq, nside, in_units='Jy', out_coords='equatorial',
                   out_units='K', outfile=None, outfmt='fits'):

        """
        -------------------------------------------------------------------------
        Convert catalog to a healpix format of given nside at specified 
        frequencies.

        Inputs:

        freq         [scalar or numpy array] Frequencies at which HEALPIX output
                     maps are to be generated

        nside        [integer] HEALPIX nside parameter for the output map(s)

        in_units     [string] Units of input map or catalog. Accepted values are
                     'K' for Temperature of 'Jy' for flux density. Default='Jy'

        out_coords   [string] Output coordinate system. Accepted values are 
                     'galactic' and 'equatorial' (default)

        out_units    [string] Units of output map. Accepted values are
                     'K' for Temperature of 'Jy' for flux density. Default='K'

        outfile      [string] Filename with full path to save the output HEALPIX
                     map(s) to. Default=None

        outfmt       [string] File format for output file. Accepted values are
                     'fits' (default) and 'ascii'

        Output(s):

        A dictionary with the following keys and values:

        'filename'   Pull path to the output file. Set only if input parameter 
                     outfile is set. Default=None.

        'spectrum'   A numpy array of size nsrc x nchan where nsrc is the number 
                     sky locations depending on input parameter out_nside and 
                     nchan is the number of frequencies in input parameter freq
        -------------------------------------------------------------------------
        """

        try:
            freq
        except NameError:
            freq = self.frequency.ravel()[self.frequency.size/2]
        else:
            if not isinstance(freq, (int,float,list,NP.ndarray)):
                raise TypeError('Input parameter freq must be a scalar or numpy array')
            else:
                freq = NP.asarray(freq).reshape(-1)

        try:
            nside
        except NameError:
            raise NameError('Input parameter nside not specified')
        else:
            if not isinstance(nside, int):
                raise TypeError('Input parameter nside must be an integer')

        if not isinstance(out_coords, str):
            raise TypeError('Input parameter out_coords must be a string')
        elif out_coords not in ['equatorial', 'galactic']:
            raise ValueError('Input parameter out_coords must be set to "equatorial" or "galactic"')

        if not isinstance(in_units, str):
            raise TypeError('in_units must be a string')
        elif in_units not in ['K', 'Jy']:
            raise ValueError('in_units must be "K" or "Jy"')

        if not isinstance(out_units, str):
            raise TypeError('out_units must be a string')
        elif out_units not in ['K', 'Jy']:
            raise ValueError('out_units must be "K" or "Jy"')

        if outfile is not None:
            if not isinstance(outfile, str):
                raise TypeError('outfile must be a string')

            if not isinstance(outfmt, str):
                raise TypeError('outfile format must be specified by a string')
            elif outfmt not in ['ascii', 'fits']:
                raise ValueError('outfile format must be "ascii" or "fits"')

        ec = SkyCoord(ra=self.location[:,0], dec=self.location[:,1], unit='deg', frame='icrs')
        gc = ec.transform_to('galactic')
        if out_coords == 'galactic':
            phi = gc.l.radian
            theta = NP.pi/2 - gc.b.radian
        else:
            phi = ec.ra.radian
            theta = NP.pi/2 - ec.dec.radian

        outmap = NP.zeros((HP.nside2npix(nside), freq.size))
        pixarea = HP.nside2pixarea(nside)
        pix = HP.ang2pix(nside, theta, phi)
        spectrum = self.generate_spectrum(frequency=freq)
        if in_units != out_units:
            if out_units == 'K':
                spectrum = (FCNST.c / freq.reshape(1,-1))**2 / (2*FCNST.k*pixarea) * spectrum * CNST.Jy # Convert into temperature
            else:
                spectrum = (freq.reshape(1,-1) / FCNST.c)**2 * (2*FCNST.k*pixarea) * spectrum / CNST.Jy # Convert into Jy

        uniq_pix, uniq_pix_ind, pix_invind = NP.unique(pix, return_index=True, return_inverse=True)
        counts, binedges, binnums, ri = OPS.binned_statistic(pix_invind, statistic='count', bins=NP.arange(uniq_pix.size+1))
        ind_count_gt1, = NP.where(counts > 1)
        ind_count_eq1, = NP.where(counts == 1)        
        upix_1count = []
        spec_ind = []
        for i in ind_count_eq1:
            ind = ri[ri[i]:ri[i+1]]
            upix_1count += [uniq_pix[i]]
            spec_ind += [ind]

        upix_1count = NP.asarray(upix_1count)
        spec_ind = NP.asarray(spec_ind).ravel()
        outmap[upix_1count,:] = spectrum[spec_ind,:]

        for i in ind_count_gt1:
            upix = uniq_pix[i]
            ind = ri[ri[i]:ri[i+1]]
            outmap[upix,:] = NP.sum(spectrum[ind,:], axis=0)

        # Save the healpix spectrum to file
        if outfile is not None:
            if outfmt == 'fits':
                hdulist = []
    
                hdulist += [fits.PrimaryHDU()]
                hdulist[0].header['EXTNAME'] = 'PRIMARY'
                hdulist[0].header['NSIDE'] = nside
                hdulist[0].header['UNTIS'] = out_units
                hdulist[0].header['NFREQ'] = freq.size
        
                for chan, f in enumerate(freq):
                    hdulist += [fits.ImageHDU(outmap[:,chan], name='{0:.1f} MHz'.format(f/1e6))]
        
                hdu = fits.HDUList(hdulist)
                hdu.writeto(outfile+'.fits', clobber=True)

                return {'hlpxfile': outfile+outfmt, 'hlpxspec': outmap}
            else:
                out_dict = {}
                colnames = []
                colfrmts = {}
                for chan, f in enumerate(freq):
                    out_dict['{0:.1f}_MHz'.format(f/1e6)] = outmap[:,chan]
                    colnames += ['{0:.1f}_MHz'.format(f/1e6)]
                    colfrmts['{0:.1f}_MHz'.format(f/1e6)] = '%0.5f'

                tbdata = Table(out_dict, names=colnames)
                ascii.write(tbdata, output=outfile+'.txt', format='fixed_width_two_line', formats=colfrmts, bookend=False, delimiter='|', delimiter_pad=' ')

                return {'filename': outfile+outfmt, 'spectrum': outmap}
        else:
            return {'filename': outfile, 'spectrum': outmap}

    ############################################################################

    def save(self, outfile, fileformat='hdf5', extspec_action=None):

        """
        -------------------------------------------------------------------------
        Save sky model to the specified output file

        Inputs:

        outfile     [string] Output filename including full path omitting the
                    extension which will be appended automatically

        fileformat  [string] format for the output. Accepted values are 'ascii'
                    and 'hdf5' (default). 

        extspec_action
                    [string] Specifies if full spectrum in attribute spectrum
                    is unloaded on to the external file and set the spectrum 
                    attribute to None while simultaneously setting spec_extfile 
                    attribute to outfile. If this input is set to None 
                    (default), then the attribute spectrum is not unloaded and 
                    data is carried in the instance in the attribute. This only 
                    applies if attribute 'spectrum' is present and not None.
        -------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('outfile not specified')

        if fileformat not in ['hdf5', 'ascii']:
            raise ValueError('Output fileformat must be set to "hdf5" or "ascii"')

        if fileformat == 'hdf5':
            outfile = outfile + '.hdf5'
            with h5py.File(outfile, 'w') as fileobj:
                hdr_group = fileobj.create_group('header')
                hdr_group['spec_type'] = self.spec_type
                hdr_group['is_healpix'] = int(self.is_healpix)
                hdr_group['healpix_ordering'] = self.healpix_ordering

                object_group = fileobj.create_group('object')
                object_group.attrs['epoch'] = self.epoch
                object_group.attrs['coords'] = self.coords
                name_dset = object_group.create_dataset('name', (self.name.size,), maxshape=(None,), data=self.name, compression='gzip', compression_opts=9)
                if self.coords == 'radec':
                    ra_dset = object_group.create_dataset('RA', (self.location.shape[0],), maxshape=(None,), data=self.location[:,0], compression='gzip', compression_opts=9)
                    ra_dset.attrs['units'] = 'degrees'
                    dec_dset = object_group.create_dataset('Dec', (self.location.shape[0],), maxshape=(None,), data=self.location[:,1], compression='gzip', compression_opts=9)
                    dec_dset.attrs['units'] = 'degrees'
                elif self.coords == 'altaz':
                    alt_dset = object_group.create_dataset('Alt', (self.location.shape[0],), maxshape=(None,), data=self.location[:,0], compression='gzip', compression_opts=9)
                    alt_dset.attrs['units'] = 'degrees'
                    az_dset = object_group.create_dataset('Az', (self.location.shape[0],), maxshape=(None,), data=self.location[:,1], compression='gzip', compression_opts=9)
                    az_dset.attrs['units'] = 'degrees'
                else:
                    raise ValueError('This coordinate system is not currently supported')
                if self.src_shape is not None:
                    src_shape_dset = object_group.create_dataset('shape', self.src_shape.shape, maxshape=(None,self.src_shape.shape[1]), data=self.src_shape, compression='gzip', compression_opts=9)
                    src_shape_dset.attrs['units'] = 'degrees'

                spec_group = fileobj.create_group('spectral_info')
                freq_range_dset = spec_group.create_dataset('freq', (self.frequency.size,), maxshape=(None,), data=self.frequency.ravel(), compression='gzip', compression_opts=9)
                freq_range_dset.attrs['units'] = 'Hz'
                if self.spec_type == 'func':
                    # spec_group['func-name'] = self.spec_parms['name']
                    func_name_dset = spec_group.create_dataset('func-name', self.spec_parms['name'].shape, maxshape=(None,), data=self.spec_parms['name'])
                    freq_ref_dset = spec_group.create_dataset('freq-ref', self.spec_parms['freq-ref'].shape, maxshape=(None,), data=self.spec_parms['freq-ref'], compression='gzip', compression_opts=9)
                    freq_ref_dset.attrs['units'] = 'Hz'
                    flux_dset = spec_group.create_dataset('flux_density', self.spec_parms['flux-scale'].shape, maxshape=(None,), data=self.spec_parms['flux-scale'], compression='gzip', compression_opts=9)
                    flux_dset.attrs['units'] = 'Jy'
                    if NP.all(self.spec_parms['name'] == 'power-law'):
                        spindex_dset = spec_group.create_dataset('spindex', self.spec_parms['power-law-index'].shape, maxshape=(None,), data=self.spec_parms['power-law-index'], compression='gzip', compression_opts=9)
                else:
                    freq_dset = spec_group.create_dataset('freq', data=self.frequency.ravel(), compression='gzip', compression_opts=9)
                    freq_dset.attrs['units'] = 'Hz'
                    if self.spectrum is not None:
                        spectrum_dset = spec_group.create_dataset('spectrum', data=self.spectrum, chunks=(1,self.frequency.size), compression='gzip', compression_opts=9)
                        if extspec_action is not None:
                            if not isinstance(extspec_action, str):
                                raise TypeError('Input extspec_action must be a string')
                            if extspec_action.lower() not in ['unload']:
                                raise ValueError('Value specified for input extspec_action invalid')
                            if extspec_action.lower() == 'unload':
                                self.spectrum = None
                                self.spec_extfile = outfile
                    elif self.spec_extfile is not None:
                        spec_group['spec_extfile'] = self.spec_extfile
                    else:
                        raise AttributeError('Neither attribute "spectrum" nor "spec_extfile" found in the instance')
        else:
            outfile = outfile + '.txt'
            frmts = {}
            frmts['ID'] = '%s19'
            frmts['RA (deg)'] = '%10.6f'
            frmts['DEC (deg)'] = '%+10.6f'
            frmts['S (Jy)'] = '%8.3f'
            frmts['FREQ (MHz)'] = '%12.7f'
    
            data_dict = {}
            if self.coords == 'radec':
                if self.epoch == 'B1950':
                    data_dict['ID'] = NP.char.replace('B'+NP.char.array(Angle(self.location[:,0],unit=units.degree).to_string(unit=units.hour,sep=':',alwayssign=False,pad=True,precision=2))+NP.char.array(Angle(self.location[:,1],unit=units.degree).to_string(unit=units.degree,sep=':',alwayssign=True,pad=True,precision=1)), ':', '')
                else:
                    data_dict['ID'] = NP.char.replace('J'+NP.char.array(Angle(self.location[:,0],unit=units.degree).to_string(unit=units.hour,sep=':',alwayssign=False,pad=True,precision=2))+NP.char.array(Angle(self.location[:,1],unit=units.degree).to_string(unit=units.degree,sep=':',alwayssign=True,pad=True,precision=1)), ':', '')                
            data_dict['RA (deg)'] = self.location[:,0]
            data_dict['DEC (deg)'] = self.location[:,1]
            if self.spec_type == 'func':
                data_dict['S (Jy)'] = self.spec_parms['flux-scale']
                data_dict['FREQ (MHz)'] = self.spec_parms['freq-ref']/1e6 # in MHz
                if NP.all(self.spec_parms['name'] == 'power-law'):
                    data_dict['SPINDEX'] = self.spec_parms['power-law-index']
                    frmts['SPINDEX'] = '%0.2f'
                    field_names = ['ID', 'RA (deg)', 'DEC (deg)', 'S (Jy)', 'FREQ (MHz)', 'SPINDEX']
                else:
                    field_names = ['ID', 'RA (deg)', 'DEC (deg)', 'S (Jy)', 'FREQ (MHz)']
            else:
                data_dict['FREQ (MHz)'] = self.frequency.flatten()[self.frequency.size/2] / 1e6 + NP.zeros(self.location.shape[0]) # MHz
                field_names = ['ID', 'RA (deg)', 'DEC (deg)', 'FREQ (MHz)']
    
            tbdata = Table(data_dict, names=field_names)
            ascii.write(tbdata, output=outfile, format='fixed_width_two_line', formats=frmts, delimiter=' ', delimiter_pad=' ', bookend=False)

################################################################################

def diffuse_radio_sky_model(outfreqs, gsmversion='gsm2008', nside=512, ind=None, outfile=None, parallel=False):

    """
    ---------------------------------------------------------------------------
    Generate a Global Sky Model (GSM) using PyGSM package, GSM2008, 
    (Oliveira-Costa et. al.,2008) for frequencies between 10 MHz to 5 THz

    Inputs:

    outfreqs    [list or numpy array] array containing the list of output 
                frequencies in MHz for which GSMs are generated. 
                Must be specified, no default.

    gsmversion  [string] string specifying the verion of PyGSM to use which
                can be either GSM2008 or GSM2016. Default = 'gsm2008'

    nside       [scalar] positive integer specifying the required number of 
                pixels per side (according to HEALPIX format) for the GSM. 
                If not specified, the output GSM will have NSIDE = 512. 

    ind         [numpy array] array containing the list of sky pixel indices. 
                If set to None (default), all indices are returned otherwise
                indices denoting specific locations are returned

    outfile     [string] Full path to filename (without '.hdf5' extension 
                which will be appended automatically) to which the sky model
                will be written to. It will be in a format compatible for
                initializing an instance of class SkyModel. If set to None,
                no output file is written

    parallel    [boolean] If set to False (default), the healpix smoothing
                and up/down grading will be performed in series, and if set
                to True these steps will be parallelized across frequency
                axis

    Outputs:

    skymod      [instance of class SkyModel] Instance of class SkyModel. If 
                outfile was provided in the inputs, it will contain the 
                spectrum in attribute spectrum. If outfile was provided, the
                spectrum would have been written to external file and unloaded
                from the spectrum attribute 
    ---------------------------------------------------------------------------
    """
    
    try:
        outfreqs
    except NameError:
        raise NameError('outfreqs must be specified')

    if outfreqs is None:
        raise ValueError('outfreqs cannot be NoneType')

    if not isinstance(outfreqs, (list, NP.ndarray)):
        raise TypeError('outfreqs must be a list or a numpy array')
    outfreqs = NP.asarray(outfreqs).reshape(-1)
    if NP.any(NP.logical_or(outfreqs < 10e6, outfreqs > 5e12)):
        raise ValueError('outfreqs must lie in the range [10MHz, 5THz]')

    if not isinstance(nside, int):
        raise TypeError('nside must be an integer')
    if not HP.isnsideok(nside):
        raise ValueError('nside must be valid')

    if ind is not None:
        if not isinstance(ind, NP.ndarray):
            raise TypeError('ind must be a numpy array')

    if outfile is not None:
        if not isinstance(outfile, str):
            raise TypeError('outfile must be a string')

    if gsmversion == 'gsm2008':
        gsm = GlobalSkyModel()
    elif gsmversion == 'gsm2016':
        gsm = GlobalSkyModel2016()

    map_cube = gsm.generate(outfreqs/1e6)
    if HP.npix2nside(map_cube.shape[1]) > nside:
        fwhm = HP.nside2resol(nside)
        if not isinstance(parallel, bool):
            parallel = False
        print '\tComputing diffuse radio sky model...'
        if parallel:
            nproc = outfreqs.size
            list_split_maps = NP.array_split(map_cube, nproc, axis=0)
            list_fwhm = [fwhm] * nproc
            list_nside = [nside] * nproc
            list_ordering = ['RING'] * nproc
            list_verbose = [False] * nproc
            pool = MP.Pool(processes=nproc)
            list_outmaps = pool.map(healpix_smooth_and_udgrade_arg_splitter, IT.izip(list_split_maps, list_fwhm, list_nside, list_ordering, list_verbose))
            outmaps = NP.asarray(list_outmaps)
        else:
            outmaps = None
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} channels '.format(outfreqs.size), PGB.ETA()], maxval=outfreqs.size).start()
            for freqi, freq in enumerate(outfreqs):
                smooth_map = HP.smoothing(map_cube[freqi,:], fwhm=fwhm, verbose=False)
                outmap = HP.ud_grade(smooth_map, nside, order_in='RING')
                if outmaps is None:
                    outmaps = outmap.reshape(1,-1)
                else:
                    outmaps = NP.concatenate((outmaps, outmap.reshape(1,-1)))
                progress.update(freqi+1)
            progress.finish()
        print '\tCompleted estimating diffuse radio sky model.'
    elif HP.npix2nside(map_cube.shape[1]) < nside:
        outmaps = None
        print '\tComputing diffuse radio sky model...'
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} channels'.format(outfreqs.size), PGB.ETA()], maxval=outfreqs.size).start()
        for freqi, freq in enumerate(outfreqs):
            outmap = HP.ud_grade(map_cube[freqi,:], nside, order_in='RING')
            if outmaps is None:
                outmaps = outmap.reshape(1,-1)
            else:
                outmaps = NP.concatenate((outmaps, outmap.reshape(1,-1)), axis=0)
            progress.update(freqi+1)
        progress.finish()
        print '\tCompleted estimating diffuse radio sky model.'
    else:
        outmaps = map_cube
        
    outmaps = outmaps.T
    pixarea = HP.nside2pixarea(nside) # Steradians
    if gsmversion == 'gsm2016':
        outmaps *= 1e6 * pixarea # 1e6 * (MJy/Sr) * Sr = Jy
    else:
        outmaps = outmaps * (2.0 * FCNST.k * outfreqs.reshape(1,-1)**2 / FCNST.c**2) * pixarea / CNST.Jy

    theta, phi = HP.pix2ang(nside, NP.arange(outmaps.shape[0]), nest=False)
    gc = SkyCoord(l=NP.degrees(phi)*units.degree, b=(90.0-NP.degrees(theta))*units.degree, frame='galactic')
    radec = gc.fk5
    ra = radec.ra.degree
    dec = radec.dec.degree

    if ind is not None:
        if NP.any(NP.logical_or(ind < 0, ind >= HP.nside2npix(nside))):
            raise IndexError('Specified indices outside allowed range')
        outmaps = outmaps[ind,:]
    npix = outmaps.shape[0]
    is_healpix = HP.isnpixok(npix)
    if is_healpix:
        healpix_ordering = 'ring'
    else:
        healpix_ordering = 'na'
    flux_unit = 'Jy'
    catlabel = NP.asarray([gsmversion]*npix)
    majax = NP.degrees(HP.nside2resol(nside)) * NP.ones(npix)
    minax = NP.degrees(HP.nside2resol(nside)) * NP.ones(npix)
    spec_type = 'spectrum'
    spec_parms = {}
    skymod_init_parms = {'name': catlabel, 'frequency': outfreqs, 'location': NP.hstack((ra.reshape(-1,1), dec.reshape(-1,1))), 'is_healpix': is_healpix, 'healpix_ordering': healpix_ordering, 'spec_type': spec_type, 'spec_parms': spec_parms, 'spectrum': outmaps, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(npix).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}
    skymod = SkyModel(init_parms=skymod_init_parms, init_file=None)
    if outfile is not None:
        skymod.save(outfile, fileformat='hdf5', extspec_action='unload')
    return skymod
        
################################################################################
