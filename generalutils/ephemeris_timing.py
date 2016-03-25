from astropy.time import Time,TimeDelta
from astropy.coordinates import Longitude
import numpy as NP
from astropy import units as u
import ephem as EP

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
