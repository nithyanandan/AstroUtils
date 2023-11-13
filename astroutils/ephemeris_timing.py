from __future__ import print_function, division, unicode_literals
from builtins import str
import warnings, copy
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, AltAz, ICRS, FK5, EarthLocation, Longitude
from astropy import units as U
import numpy as NP
import constants as CNST

# Perform some IERS adjustments

from astropy.utils import iers

tnow = Time.now()
try:
    print('Checking if some IERS related adjustments are required...')
    tnow_ut1 = tnow.ut1
except iers.IERSRangeError as exception:
    default_iers_auto_url = 'http://maia.usno.navy.mil/ser7/finals2000A.all'
    secondary_iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
    tertiary_iers_auto_url = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
    
    try:    
    #     iers.conf.iers_auto_url = default_iers_auto_url
        iers.conf.remote_timeout = 120.0
        iers.IERS_A.open(iers.IERS_A_URL)
    except Exception as err:
        if ('url' in str(err).lower()) or (('connection' in str(err).lower())):
            print(err)
            print('Original source URL for IERS_A: {0} FAILED!'.format(iers.conf.iers_auto_url))
            print('Original IERS Configuration:')
            print(iers.conf.__class__.__dict__)
            print('Modifying the source URL for IERS_A table to {0}'.format(secondary_iers_auto_url))
    #         iers.IERS_A_URL = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
            iers.conf.auto_download = True
            iers.conf.iers_auto_url = secondary_iers_auto_url
    #         iers.conf.iers_auto_url = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
            try:
                print('Now testing {0}'.format(secondary_iers_auto_url))
                iers_a = iers.IERS_A.open(secondary_iers_auto_url)
            except Exception as newerr:
                if ('url' in str(err).lower()):
                    print(newerr)
                    print('Modified URL also did not work. Computation of LST may be affected or will completely fail.')
    #                 raise newerr
            else:
                print('Updated source URL {0} worked!'.format(secondary_iers_auto_url))
                print('Modified IERS Configuration:')
                print(iers.conf.__class__.__dict__)
                try:
                    tnow_ut1 = tnow.ut1
                except iers.IERARangeError as exception:
                    print(exception)
                    warnings.warn('Ephemeris predictions will be unreliable despite a successful download of IERS tables')
            
################################################################################

def equation_of_equinoxes(jd):
    """
    ----------------------------------------------------------------------------
    Estimate the equation of the equinoxes
    Inputs:

    jd      [scalar or numpy array] Julian date at which nutation is to be 
            estimated and the equation of equinoxes is returned. 

    Output:

    Equation of the equinoxes (in hours) that should be used to correct the 
    Greenwich Mean Sidereal Time to obtain the Greenwich Apparent Sidereal Time

    Notes: Adopted from https://aa.usno.navy.mil/faq/docs/GAST.php
    ----------------------------------------------------------------------------
    """

    if not isinstance(jd, (int, float, NP.ndarray)):
        raise TypeError('Input julian date(s) must be a scalar or numpy array')

    d = jd - 2451545.0 # Days since 2000 January 1, 12h UT, Julian date 2451545.0
    omega = 125.04 - 0.052954 * d # Longitude of the ascending node of the Moon in degrees
    l = 280.47 + 0.98565 * d # Mean Longitude of the Sun in degrees
    obliquity = 23.4393 - 0.0000004 * d # in degrees

    nutation = -0.000319 * NP.sin(NP.radians(omega)) - 0.000024 * NP.sin(NP.radians(2*l))  # in hours
    eqeq = nutation * NP.cos(NP.radians(obliquity)) # Equation of the equinoxes in hours
    # t = d / 36525 # number of centuries since 2000 January 1, 12h UT, Julian date 2451545.0

    return eqeq

################################################################################

def julian_date_from_LAST(last, jd0, longitude, tol=1e-6):
    """
    ----------------------------------------------------------------------------
    Inputs:

    last        [scalar or numpy array] Local Apparent Sidereal Time (in hours)

    jd0         [scalar or numpy array] Julian date at previous midnight. Same
                size as input 'last' or numpy broadcastable to that shape

    longitude   [scalar or numpy array] Longitude of observing site (in hours). 
                Same size as input 'last' or numpy broadcastable to that shape

    tol         [scalar] Tolerance for convergence since these calculations
                are iteratively solved

    Output:

    Julian date(s) as a numpy array correspoding to the input apparent sidereal
    time and longitude on given starting Julian dates.

    Notes: Adopted from https://aa.usno.navy.mil/faq/docs/GAST.php
    ----------------------------------------------------------------------------
    """

    if not isinstance(jd0, (int, float, NP.ndarray)):
        raise TypeError('Input starting julian date(s) must be a scalar or numpy array')
    jd0 = NP.asarray(jd0).ravel()

    if not isinstance(last, (int, float, NP.ndarray)):
        raise TypeError('Input local apparent sidereal time(s) must be a scalar or numpy array')
    last = NP.asarray(last).ravel()

    if not isinstance(longitude, (int, float, NP.ndarray)):
        raise TypeError('Input longitude(s) must be a scalar or numpy array')
    longitude = NP.asarray(longitude).ravel()

    jd = NP.copy(jd0)
    gast = last - longitude
    d0 = jd0 - 2451545.0

    if gast < 0.0:
        gast += 24.0
    if gast >= 24.0:
        gast -= 24.0

    gmst0 = 18.697374558 + 24.06570982441908 * d0 # Accurate to 0.1s per century
    gmst0 = gmst0 % 24
    if gmst0 < 0.0:
        gmst0 += 24.0
    if gmst0 >= 24.0:
        gmst0 -= 24.0
        
    dev = 100 * tol
    iternum = 0
    hr = None
    while (iternum < 1000) and (NP.abs(dev) > tol):
        eqeq = equation_of_equinoxes(jd)
        gmst = gast - eqeq
        gmst = gmst % 24
        if gmst < 0.0:
            gmst += 24.0
        if gmst >= 24.0:
            gmst -= 24.0
        
        newhr = gmst - gmst0
        newhr *= CNST.sday
    
        if hr is not None:
            dev = (newhr - hr) / hr
        hr = NP.copy(newhr)
        jd = jd0 + hr/24.0
        iternum += 1

    return jd

################################################################################

def gmst2gps(day, GMST, type='mean', iterations=10, precision=1e-14):

    """
    ----------------------------------------------------------------------------
    gps=gmst2gps(day, GMST, type='mean', iterations=10, precision=1e-14)
    returns the GPS time associated with the GMST/GAST on the given day
    type can be 'mean' or 'apparent'
    uses a root-find, so not super fast
    ----------------------------------------------------------------------------
    """

    assert type in ['mean','apparent']
    gmst=Longitude(GMST,unit='h')
    t=Time(day,scale='utc')
    iteration=0
 
    siderealday_to_solarday=0.99726958
 
    while iteration < iterations:
        error=t.sidereal_time(type,'greenwich')-gmst
        if NP.abs(error/gmst).value <= precision:
            return t.gps
        t=t-TimeDelta((error).hour*U.hour)*siderealday_to_solarday
        iteration+=1
 
    return None

################################################################################

def hadec2radec(hadec, lst, obstime=None, epoch_RA=2000.0, time_type=None):

    """
    ----------------------------------------------------------------------------
    Convert HA-Dec to RA-Dec with accurate ephemeris

    Inputs:

    hadec   [numpy array] HA and Dec as a Nx2 numpy array. All units in degrees

    lst     [scalar or numpy array] Local Sidereal time (in degrees). If a 
            scalar is specified, it will be applied to all the entries of input
            hadec. If an array is given it should be numpy array broadcastable
            with hadec. That is, shape must be (1,) or (N,). 

    obstime [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to input hadec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. If set to None (default), it 
            will be set equal to epoch_RA. 

    epoch_RA
            [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to output radec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. It must be in the same format 
            as the one obstime is specified in. If set to 2000.0 (default), it 
            is assumed to be in 'jyear' format. If set to None, will be set 
            equal to 2000.0 in 'jyear' format. 

    time_type
            [string] Specifies the format in which obstime and epoch_RA are
            provided. Accepted values are 'jd' (Julian Day), 'jyear' (Julian 
            year), 'iso' or 'isot'. If set to None (default) and if obstime 
            and/or epoch_RA is a scalar, the corresponding scalar entries are
            assumed to be in Julian Year. 

    Output:

    The output radec as a numpy array of shape (N,2) in units of degrees is 
    returned at the epoch specified in epoch_RA. 
    ----------------------------------------------------------------------------
    """

    if not isinstance(hadec, NP.ndarray):
        raise TypeError('Input hadec must be a numpy array')
    if hadec.size == 2:
        hadec = hadec.reshape(1,-1)
    if hadec.ndim != 2:
        raise ValueError('Input hadec must be a 2D numpy array of shape(nsrc,2)')
    if hadec.shape[1] != 2:
        raise ValueError('Input hadec must be a 2D numpy array of shape(nsrc,2)')
    if isinstance(lst, (int,float)):
        lst = NP.asarray(lst).astype(float).reshape(-1)
    if (lst.size != 1) and (lst.size != hadec.shape[0]):
        raise ValueError('Input LST must match the shape of input hadec')

    if epoch_RA is not None:
        if isinstance(epoch_RA, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_RA = Time(epoch_RA, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_RA = Time(epoch_RA, scale='utc', format='jd')
        elif isinstance(epoch_RA, str):
            if time_type.lower() == 'jyear':
                equinox_RA = Time('J{0:.9f}'.format(epoch_RA), scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_RA = Time(epoch_RA, scale='utc', format=time_type.lower())
        elif isinstance(epoch_RA, Time):
            equinox_RA = copy.copy(epoch_RA)
        else:
            raise TypeError('Input epoch_RA is invalid or currently not accepted')
    else:
        equinox_RA = Time(2000.0, format='jyear', scale='utc')
        warnings.warn('No epoch_RA provided. Setting epoch to {0}'.format(equinox_RA.jyear_str))

    if obstime is not None:
        if isinstance(obstime, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_HA = Time(obstime, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_HA = Time(obstime, scale='utc', format='jd')
        elif isinstance(obstime, str):
            if time_type.lower() == 'jyear':
                equinox_HA = Time(obstime, scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_HA = Time(obstime, scale='utc', format=time_type.lower())
        elif isinstance(obstime, Time):
            equinox_HA = copy.copy(obstime)
        else:
            raise TypeError('Input obstime is invalid or currently not accepted')
        
    if obstime is None:
        equinox_HA = copy.copy(equinox_RA)
        warnings.warn('No obstime provided. Setting obstime to {0}'.format(equinox_HA.jyear_str))

    # if (obstime is None) and (epoch_RA is not None):
    #     equinox_HA = copy.copy(equinox_RA)
    # elif (obstime is not None) and (epoch_RA is None):
    #     equinox_RA = copy.copy(equinox_HA)
    # elif (obstime is None) and (epoch_RA is None):
    #     equinox_HA = None
    #     equinox_RA = None

    radec_obstime = NP.hstack(((lst-hadec[:,0]).reshape(-1,1), hadec[:,1].reshape(-1,1)))
    if (equinox_HA is None) and (equinox_RA is None):
        return radec_obstime
    else:
        skycoords = SkyCoord(ra=radec_obstime[:,0]*U.deg, dec=radec_obstime[:,1]*U.deg, frame='fk5', equinox=equinox_HA).transform_to(FK5(equinox=equinox_RA))
        radec = NP.hstack((skycoords.ra.deg.reshape(-1,1), skycoords.dec.deg.reshape(-1,1)))
        return radec

#################################################################################

def radec2hadec(radec, lst, obstime=None, epoch_RA=None, time_type=None):

    """
    ----------------------------------------------------------------------------
    Convert RA-Dec to HA-Dec with accurate ephemeris

    Inputs:

    radec   [numpy array] RA and Dec as a Nx2 numpy array. All units in degrees

    lst     [scalar or numpy array] Local Sidereal time (in degrees). If a 
            scalar is specified, it will be applied to all the entries of input
            radec. If an array is given it should be numpy array broadcastable
            with radec. That is, shape must be (1,) or (N,). 

    obstime [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to output hadec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. If set to None (default), it 
            will be set equal to epoch_RA. 

    epoch_RA
            [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to input radec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. It must be in the same format 
            as the one obstime is specified in. If set to 2000.0 (default), it 
            is assumed to be in 'jyear' format. If set to None, it will be set 
            equal to default of 2000.0 in 'jyear' format. 

    time_type
            [string] Specifies the format in which obstime and epoch_RA are
            provided. Accepted values are 'jd' (Julian Day), 'jyear' (Julian 
            year), 'iso' or 'isot'. If set to None (default) and if obstime 
            and/or epoch_RA is a scalar, the corresponding scalar entries are
            assumed to be in Julian Year. 

    Output:

    The output hadec as a numpy array of shape (N,2) in units of degrees 
    is returned at the epoch specified in obstime.
    ----------------------------------------------------------------------------
    """

    if not isinstance(radec, NP.ndarray):
        raise TypeError('Input radec must be a numpy array')
    if radec.size == 2:
        radec = radec.reshape(1,-1)
    if radec.ndim != 2:
        raise ValueError('Input radec must be a 2D numpy array of shape(nsrc,2)')
    if radec.shape[1] != 2:
        raise ValueError('Input radec must be a 2D numpy array of shape(nsrc,2)')
    if isinstance(lst, (int,float)):
        lst = NP.asarray(lst).astype(NP.float).reshape(-1)
    if (lst.size != 1) and (lst.size != radec.shape[0]):
        raise ValueError('Input LST must match the shape of input radec')

    if epoch_RA is not None:
        if isinstance(epoch_RA, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_RA = Time(epoch_RA, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_RA = Time(epoch_RA, scale='utc', format='jd')
        elif isinstance(epoch_RA, str):
            if time_type.lower() == 'jyear':
                equinox_RA = Time('J{0:.9f}'.format(epoch_RA), scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_RA = Time(epoch_RA, scale='utc', format=time_type.lower())
        elif isinstance(epoch_RA, Time):
            equinox_RA = copy.copy(epoch_RA)
        else:
            raise TypeError('Input epoch_RA is invalid or currently not accepted')
    else:
        equinox_RA = Time(2000.0, format='jyear', scale='utc')
        warnings.warn('No epoch_RA provided. Setting epoch to {0}'.format(equinox_RA.jyear_str))

    if obstime is not None:
        if isinstance(obstime, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_HA = Time(obstime, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_HA = Time(obstime, scale='utc', format='jd')
        elif isinstance(obstime, str):
            if time_type.lower() == 'jyear':
                equinox_HA = Time(obstime, scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_HA = Time(obstime, scale='utc', format=time_type.lower())
        elif isinstance(obstime, Time):
            equinox_HA = copy.copy(obstime)
        else:
            raise TypeError('Input obstime is invalid or currently not accepted')
        
    if obstime is None:
        equinox_HA = copy.copy(equinox_RA)
        warnings.warn('No obstime provided. Setting obstime to {0}'.format(equinox_HA.jyear_str))
        
    # if (obstime is None) and (epoch_RA is not None):
    #     equinox_HA = copy.copy(equinox_RA)
    # elif (obstime is not None) and (epoch_RA is None):
    #     equinox_RA = copy.copy(equinox_HA)
    # elif (obstime is None) and (epoch_RA is None):
    #     equinox_HA = None
    #     equinox_RA = None

    # if (equinox_HA is None) and (equinox_RA is None):
    #     return NP.hstack(((lst - radec[:,0]).reshape(-1,1), radec[:,1].reshape(-1,1)))
    # else:
    skycoords = SkyCoord(ra=radec[:,0]*U.deg, dec=radec[:,1]*U.deg, equinox=equinox_RA, frame='fk5').transform_to(FK5(equinox=equinox_HA))
    hadec = NP.hstack(((lst-skycoords.ra.deg).reshape(-1,1), skycoords.dec.deg.reshape(-1,1)))
    return hadec
        
#################################################################################

def altaz2radec(altaz, location, obstime=None, epoch_RA=2000.0, time_type=None):

    """
    ----------------------------------------------------------------------------
    Convert Alt-Az to RA-Dec with accurate ephemeris

    Inputs:

    altaz   [numpy array] Altitude and Azimuth as a Nx2 numpy array. All units 
            in degrees

    location
            [instance of class astropy.coordinates.EarthLocation] Location of
            the observer provided as an instance of class 
            astropy.coordinates.EarthLocation

    obstime [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to input altaz. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. If set to None (default), it 
            will be set equal to epoch_RA. 

    epoch_RA
            [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to output radec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. It must be in the same format 
            as the one obstime is specified in. If set to 2000.0 (default), it 
            is assumed to be in 'jyear' format. If set to None, it will be set 
            equal to default of 2000.0 in 'jyear' format. 

    time_type
            [string] Specifies the format in which obstime and epoch_RA are
            provided. Accepted values are 'jd' (Julian Day), 'jyear' (Julian 
            year), 'iso' or 'isot'. If set to None (default) and if obstime 
            and/or epoch_RA is a scalar, the corresponding scalar entries are
            assumed to be in Julian Year. 

    Output:

    The output radec as a numpy array of shape (N,2) in units of degrees is 
    returned at the epoch specified in epoch_RA.
    ----------------------------------------------------------------------------
    """

    if isinstance(altaz, NP.ndarray):
        if altaz.size == 2:
            altaz = altaz.reshape(1,-1)
        if altaz.ndim != 2:
            raise ValueError('Input altaz must be a numpy array of shape (N,2)')
        if altaz.shape[1] != 2:
            raise ValueError('Input altaz must be a numpy array of shape (N,2)')
    elif not isinstance(altaz, AltAz):
        raise TypeError('Input altaz must be a numpy array or an instance of class astropy.coordinates.AltAz')

    if not isinstance(location, EarthLocation):
        raise TypeError('Input location must be an instance of class astropy.coordinates.EarthLocation')

    if epoch_RA is not None:
        if isinstance(epoch_RA, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_RA = Time(epoch_RA, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_RA = Time(epoch_RA, scale='utc', format='jd')
        elif isinstance(epoch_RA, str):
            if time_type.lower() == 'jyear':
                equinox_RA = Time('J{0:.9f}'.format(epoch_RA), scale='utc', format='jyear_str')
            elif (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_RA = Time(epoch_RA, scale='utc', format=time_type.lower())
        elif isinstance(epoch_RA, Time):
            equinox_RA = copy.copy(epoch_RA)
        else:
            raise TypeError('Input epoch_RA is invalid or currently not accepted')
    else:
        equinox_RA = Time(2000.0, format='jyear', scale='utc')
        warnings.warn('No epoch_RA provided. Setting epoch to {0}'.format(equinox_RA.jyear_str))
        
    if obstime is not None:
        if isinstance(obstime, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_altaz = Time(obstime, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_altaz = Time(obstime, scale='utc', format='jd')
        elif isinstance(obstime, str):
            if time_type.lower() == 'jyear':
                equinox_altaz = Time('J{0:.9f}'.format(obstime), scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_altaz = Time(obstime, scale='utc', format=time_type.lower())
        elif isinstance(obstime, Time):
            equinox_altaz = copy.copy(obstime)
        else:
            raise TypeError('Input obstime is invalid or currently not accepted')
    else:
        if isinstance(altaz, AltAz):
            equinox_altaz = copy.deepcopy(altaz.obstime)
        else:
            equinox_altaz = copy.copy(equinox_RA)
        warnings.warn('No obstime provided. Setting obstime to {0}'.format(equinox_altaz.jyear_str))
        
    if isinstance(altaz, AltAz):
        elaz = copy.deepcopy(altaz)
    else:
        elaz = AltAz(alt=altaz[:,0]*U.deg, az=altaz[:,1]*U.deg, obstime=equinox_altaz, location=location)
    coords_radec = elaz.transform_to(FK5(equinox=equinox_RA))
    radec = NP.hstack((coords_radec.ra.deg.reshape(-1,1), coords_radec.dec.deg.reshape(-1,1)))
    return radec

#################################################################################

def radec2altaz(radec, location, obstime=None, epoch_RA=2000.0, time_type=None):

    """
    ----------------------------------------------------------------------------
    Convert RA-Dec to Alt-Az with accurate ephemeris

    Inputs:

    radec   [numpy array] Altitude and Azimuth as a Nx2 numpy array. All units 
            in degrees

    location
            [instance of class astropy.coordinates.EarthLocation] Location of
            the observer provided as an instance of class 
            astropy.coordinates.EarthLocation

    obstime [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to output altaz. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. If set to None (default), it 
            will be set equal to epoch_RA. 

    epoch_RA
            [scalar, string, or instance of class astropy.time.Time] The time 
            or epoch which applies to input radec. It can be a scalar (in JD 
            or JYear), string (JYear string prefixed with 'J' or ISO or ISOT), 
            or an instance of class astropy.time.Time. The appropriate format 
            must be specified in input time_type. It must be in the same format 
            as the one obstime is specified in. If set to 2000.0 (default), it 
            is assumed to be in 'jyear' format. If set to None, it will be set 
            equal to default of 2000.0 in 'jyear' format. 

    time_type
            [string] Specifies the format in which obstime and epoch_RA are
            provided. Accepted values are 'jd' (Julian Day), 'jyear' (Julian 
            year), 'iso' or 'isot'. If set to None (default) and if obstime 
            and/or epoch_RA is a scalar, the corresponding scalar entries are
            assumed to be in Julian Year. 

    Output:

    The output altaz as a numpy array of shape (N,2) in units of degrees is 
    returned at the epoch specified in epoch_RA.
    ----------------------------------------------------------------------------
    """

    if isinstance(radec, NP.ndarray):
        if radec.size == 2:
            radec = radec.reshape(1,-1)
        if radec.ndim != 2:
            raise ValueError('Input radec must be a numpy array of shape (N,2)')
        if radec.shape[1] != 2:
            raise ValueError('Input radec must be a numpy array of shape (N,2)')
    elif not isinstance(radec, SkyCoord):
        raise TypeError('Input radec must be a numpy array or an instance of class astropy.coordinates.SkyCoord')

    if not isinstance(location, EarthLocation):
        raise TypeError('Input location must be an instance of class astropy.coordinates.EarthLocation')

    if epoch_RA is not None:
        if isinstance(epoch_RA, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_RA = Time(epoch_RA, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_RA = Time(epoch_RA, scale='utc', format='jd')
        elif isinstance(epoch_RA, str):
            if time_type.lower() == 'jyear':
                equinox_RA = Time('J{0:.9f}'.format(epoch_RA), scale='utc', format='jyear_str')
            elif (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_RA = Time(epoch_RA, scale='utc', format=time_type.lower())
        elif isinstance(epoch_RA, Time):
            equinox_RA = copy.copy(epoch_RA)
        else:
            raise TypeError('Input epoch_RA is invalid or currently not accepted')
    else:
        equinox_RA = Time(2000.0, format='jyear', scale='utc')
        warnings.warn('No epoch_RA provided. Setting epoch to {0}'.format(equinox_RA.jyear_str))
        
    if obstime is not None:
        if isinstance(obstime, (int,float)):
            if (time_type is None) or (time_type.lower() == 'jyear'):
                equinox_altaz = Time(obstime, scale='utc', format='jyear')
            elif time_type.lower() == 'jd':
                equinox_altaz = Time(obstime, scale='utc', format='jd')
        elif isinstance(obstime, str):
            if time_type.lower() == 'jyear':
                equinox_altaz = Time('J{0:.9f}'.format(obstime), scale='utc', format='jyear_str')
            if (time_type.lower() == 'iso') or (time_type.lower() == 'isot'):
                equinox_altaz = Time(obstime, scale='utc', format=time_type.lower())
        elif isinstance(obstime, Time):
            equinox_altaz = copy.copy(obstime)
        else:
            raise TypeError('Input obstime is invalid or currently not accepted')
    else:
        equinox_altaz = copy.copy(equinox_RA)
        warnings.warn('No obstime provided. Setting obstime to {0}'.format(equinox_altaz.jyear_str))
        
    if isinstance(radec, SkyCoord):
        coords_radec = copy.deepcopy(radec)
    else:
        coords_radec = SkyCoord(ra=radec[:,0]*U.deg, dec=radec[:,1]*U.deg, equinox=equinox_RA, frame='fk5')
    elaz = coords_radec.transform_to(AltAz(obstime=equinox_altaz, location=location))
    altaz = NP.hstack((elaz.alt.deg.reshape(-1,1), elaz.az.deg.reshape(-1,1)))
    return altaz

#################################################################################

