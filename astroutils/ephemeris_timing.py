from astropy.time import Time,TimeDelta
from astropy.coordinates import Longitude
import numpy as NP
from astropy import units as u
import ephem as EP

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
    gast = last + longitude

    if gast < 0.0:
        gast += 24.0
    if gast >= 24.0:
        gast -= 24.0

    dev = 100 * tol
    while (iter < 1000) and (NP.abs(dev) > tol):
        d = jd - 2451545.0
        t = d / 36525.0
        eqeq = equation_of_equinoxes(jd)
        gmst = gast - eqeq
        if gmst < 0.0:
            gmst += 24.0
        if gmst >= 24.0:
            gmst -= 24.0
        
        newhr = (gmst - 6.697374558 - 0.06570982441908 - 0.000026 * t**2) / 1.00273790935
        dev = (newhr - hr) / hr
        hr = NP.copy(newhr)
        jd = jd0 + hr/24.0

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
        t=t-TimeDelta((error).hour*u.hour)*siderealday_to_solarday
        iteration+=1
 
    return None

################################################################################

if __name__ == "__main__":

    # day='2013-08-23'
    # gmst='00:00:10'
    # t=gmst2gps(day, gmst)
    # if t is not None:
    #     print 'GMST {0} on {1} = {2} GPS seconds'.format(gmst, day, t)
    # else:
    #     print 'Failure to converge in Ephemeris/Timing.'

    obsdate = '2013/08/23'
    LST_given = 21.18331  # in hours
    latitude = -26.701  # in degrees
    longitude = 116.670815 # in degrees East
    
    lstobj = EP.FixedBody()
    lstobj._ra = NP.radians(LST_given * 15.0)
    lstobj._epoch = obsdate
    
    obsrvr = EP.Observer()
    obsrvr.lat = NP.radians(latitude)
    obsrvr.lon = NP.radians(longitude)
    obsrvr.date = obsdate
    
    lstobj.compute(obsrvr)
    
    transitJD = EP.julian_date(obsrvr.next_transit(lstobj))
    transitDT = str(obsrvr.next_transit(lstobj))
        
    transit_instant = Time(transitDT.replace('/', '-'), scale='utc', format='iso')
    transit_day, transit_time = str(transit_instant).split(' ')

    gmst = LST_given - longitude / 15.0
    t = gmst2gps(transit_day, gmst)
    if t is not None:
        print 'GMST {0} on {1} = {2} GPS seconds'.format(gmst, obsdate, t)
    else:
        print 'Failure to converge in Ephemeris/Timing.'
    
    # print transitDT
