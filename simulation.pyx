# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:27:51 2014

@author: koher
"""

cimport cython
import numpy as np
import numpy.random as rn
cimport numpy as np
from libc.math cimport log
cdef extern from "stdlib.h":
    int c_rand "rand" ()
    int RAND_MAX

#%% --------------------------------------------------------------------------- Random numbers
@cython.cdivision(True)
cdef inline double crand():
    return c_rand() / (RAND_MAX + 0.0)

#%% --------------------------------------------------------------------------- Random Integer
@cython.cdivision(True)
cdef inline int crandint(unsigned int MAX):
    return <int>(c_rand() / (RAND_MAX + 0.0) * MAX)

#%% --------------------------------------------------------------------------- Exponential Distribution
@cython.cdivision(True)
cdef inline double cexp(double scale):
    return -log(1-crand())*scale

#%% --------------------------------------------------------------------------- Update Bond
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_bond(long[:,:,:] bond, double[:,:,:] update_time,
                double system_time, double p, double mu,
                np.intp_t ii, np.intp_t jj, np.intp_t kk):
    while update_time[ii,jj,kk] < system_time:
        if bond[ii,jj,kk] == 1:
            bond[ii,jj,kk] = <int>(crand() > p)
        else:
            bond[ii,jj,kk] = <int>(crand() < 1-p)
        update_time[ii,jj,kk] += cexp(1.0/mu)
    return

#%% --------------------------------------------------------------------------- Update Lattice
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_lattice(long[:,:,:] bonds, double[:,:,:] times,
                   double system_time, 
                   double p, double mu, int dim):
    
    cdef np.intp_t ii,jj,kk
    for ii in range(2):
        for jj in range(dim):
            for kk in range(dim):
                if times[ii,jj,kk] < system_time:
                    update_bond(bonds,times,system_time, p, mu,ii,jj,kk)
    return

#%% --------------------------------------------------------------------------- Run Simulation
@cython.boundscheck(False)
@cython.wraparound(False)
def runsimulation(long N = 100, long dim = 400, double p = 0.3, double mu = 1./100.):
    
    cdef double [::1]     dist  = np.empty((N,),dtype=float)
    cdef long   [:,:,::1] bonds = (rn.rand(2,dim,dim) < p).astype(int)
    cdef double [:,:,::1] times = rn.exponential(scale=1.0/mu, size=(2,dim,dim))
    cdef unsigned int x, y, x0, y0, choice, system_time
    x = int(dim/2)
    y = int(dim/2)
    x0 = int(dim/2)
    y0 = int(dim/2)
    system_time = 0
    cdef np.intp_t run
    #cdef long [:,::1] neighbours = np.empty((4,2),dtype=long)
    
    for run in xrange(N):
        
        choice = crandint(4)
        if   choice == 0:
            x = <unsigned int>(x + bonds[0,x,y])
        elif choice == 1:
            x = <unsigned int>(x-bonds[0,x-1,y])
        elif choice == 2:
            y = <unsigned int>(y-bonds[1,x,y-1])
        elif choice == 3:
            y = <unsigned int>(y+bonds[1,x,y])
        
        update_lattice(bonds,times,system_time,p,mu,dim)
        system_time += 1
        dist[run] = (x-x0)*(x-x0) + (y-y0)*(y-y0)

    return np.asarray(dist)