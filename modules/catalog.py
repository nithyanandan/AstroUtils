import numpy as NP
import geometry as GEOM

#########################################################################

class Catalog:

    def __init__(self, frequency, location, flux_density, spectral_index=None,
                 epoch='J2000', coords='radec'):

        self.frequency = frequency
        self.location = NP.asarray(location)
        self.flux_density = NP.asarray(flux_density)
        self.epoch = epoch
        self.coords = coords

        if spectral_index is None:
            self.spectral_index = NP.zeros(self.flux_density.size)
        else: 
            self.spectral_index = NP.asarray(spectral_index)
        
        if (self.location.shape[0] != self.flux_density.size) or (self.flux_density.size != self.spectral_index.size) or (self.location.shape[0] != self.spectral_index.size):
            raise ValueError('location, flux_density, and spectral_index must be of equal size.')

    def match(self, other, matchrad=None, nnearest=0, maxmatches=-1):
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
        
    def subset(self, indices=None):
        if (indices is None) or (len(indices) == 0):
            return []
        else:
            return Catalog(self.frequency, self.location[indices, :], self.flux_density[indices, :], self.spectral_index[indices], self.epoch, self.coords)


#########################################################################

class SkyModel:
    
    def __init__(self, catalog=None, image=None, cube=None):
        self.catalog = catalog
        self.image = image
        self.cube = cube

    def add_sources(self, catalog):
        pass

#########################################################################
