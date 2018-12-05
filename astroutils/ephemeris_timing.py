from astropy.time import Time,TimeDelta
from astropy.coordinates import Longitude
import numpy as NP
from astropy import units as u
import ephem as EP

################################################################################

def equation_of_equinoxes(jd):
    """
    ----------------------------------------------------------------------------
    Inputs:

    jd      [numpy array] Julian date at which nutation is to be estimated and
            the equation of equinoxes returned. 

    Output:

    Equation of the equinoxes (in hours) that should be used to correct the 
    Greenwich Mean Sidereal Time to obtain the Greenwich Apparent Sidereal Time
    ----------------------------------------------------------------------------
    """

    d = jd - 2451545.0 # Days since 2000 January 1, 12h UT, Julian date 2451545.0
    omega = 125.04 - 0.052954 * d # Longitude of the ascending node of the Moon in degrees
    l = 280.47 + 0.98565 * d # Mean Longitude of the Sun in degrees
    obliquity = 23.4393-- 0.0000004 * d # in degrees

    nutation = -0.000319 * NP.sin(NP.radians(omega)) - 0.000024 * NP.sin(NP.radians(2*l))  # in hours
    eqeq = nutation * NP.cos(NP.radians(obliquity)) # Equation of the equinoxes in hours
    # t = d / 36525 # number of centuries since 2000 January 1, 12h UT, Julian date 2451545.0

    return eqeq

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
