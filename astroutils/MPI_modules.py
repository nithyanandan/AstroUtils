from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import object
from mpi4py import MPI
from array import array as _array
import struct as _struct

class Counter(object):

    def __init__(self, comm):
        #
        size = comm.Get_size()
        rank = comm.Get_rank()
        #
        itemsize = MPI.INT.Get_size()
        if rank == 0:
            mem = MPI.Alloc_mem(itemsize*size, MPI.INFO_NULL)
            mem[:] = _struct.pack('i', 0) * size
        else:
            mem = MPI.BOTTOM
        self.win = MPI.Win.Create(mem, itemsize, MPI.INFO_NULL, comm)
        #
        blens = [rank, size-rank-1]
        disps = [0, rank+1]
        self.dt_get = MPI.INT.Create_indexed(blens, disps).Commit()
        #
        self.myval = 0

    def free(self):
        self.dt_get.Free()
        mem = self.win.memory
        self.win.Free()
        if mem: MPI.Free_mem(mem)

    def __next__(self):
        #
        group  = self.win.Get_group()
        size = group.Get_size()
        rank = group.Get_rank()
        group.Free()
        #
        incr = _array('i', [1])
        vals = _array('i', [0])*size
        self.win.Lock(MPI.LOCK_EXCLUSIVE, 0, 0)
        self.win.Accumulate([incr, 1, MPI.INT], 0,
                            [rank, 1, MPI.INT], MPI.SUM)
        self.win.Get([vals, 1, self.dt_get], 0,
                     [   0, 1, self.dt_get])
        self.win.Unlock(0)
        #
        vals[rank] = self.myval
        self.myval += 1
        nxtval = sum(vals)
        #
        return nxtval
