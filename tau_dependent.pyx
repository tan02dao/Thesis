cimport cython
import numpy as np
cimport numpy as np

from numpy cimport ndarray as ar
from libc.math cimport sqrt
from libc.math cimport exp
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.math cimport isnan
from libc.math cimport sin, cos, tan, asin, acos, atan
cdef extern from "limits.h":
    int RAND_MAX

cdef extern from "complex.h":
    long double complex clog(long double complex)

cdef extern from "complex.h":
    long double complex I

cdef extern from "complex.h":
    long double cabs(long double complex)

cdef extern from "complex.h":
    long double complex conjl(long double complex)

cdef extern from "complex.h":
    long double cimagl(long double complex)

cdef extern from "complex.h":
    long double creal(long double complex)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def mc_update(ar[double, ndim=2] lattice, double beta, double B, double D, ar[long, ndim=2] nb):
    cdef int N = lattice.shape[0]
    cdef long double dE
    cdef double x,y,z,k,xi,xt,yt
    cdef double Ei, Ef
    cdef double DMi, DMf
    cdef double J_x,J_y,J_z
    cdef double neighbor
    cdef int i,l,r,u,d

    for i in range(N):
        #get random spin on a unit sphere
        xi = 2.0
        while xi >= 1.0:
            xt = 1.0-float(rand())*2.0/RAND_MAX
            yt = 1.0-float(rand())*2.0/RAND_MAX
            xi = xt*xt + yt*yt

        k = (sqrt(1.0 - float(xi)))
        x = 2.0*xt*k
        y = 2.0*yt*k
        z = 1.0 - 2.0*float(xi)

        l = nb[i,0] #list of neighbors
        r = nb[i,1]
        u = nb[i,2]
        d = nb[i,3]

        #nearest neighbors exchange interaction
        J_x = lattice[l, 0] + lattice[r, 0] + lattice[u, 0] + lattice[d, 0]
        J_y = lattice[l, 1] + lattice[r, 1] + lattice[u, 1] + lattice[d, 1]
        J_z = lattice[l, 2] + lattice[r, 2] + lattice[u, 2] + lattice[d, 2]

        #DM interaction
        DMi = D*(lattice[i,1]*lattice[r,2] - lattice[i,2]*lattice[r,1]  +
                 lattice[i,2]*lattice[u,0] - lattice[i,0]*lattice[u,2] +
                 lattice[i,2]*lattice[l,1] - lattice[i,1]*lattice[l,2] +
                 lattice[i,0]*lattice[d,2] - lattice[i,2]*lattice[d,0])

        DMf =  D*(y*lattice[r,2] - z*lattice[r,1]  +
                  z*lattice[u,0] - x*lattice[u,2] +
                  z*lattice[l,1] - y*lattice[l,2] +
                  x*lattice[d,2] - z*lattice[d,0])

        Ei = -1.0*(lattice[i,2]*J_z + lattice[i,1]*J_y + lattice[i,0]*J_x + B*lattice[i,2]) + DMi

        Ef = -1.0*(z*J_z + y*J_y + x*J_x+ B*z) + DMf

        dE = Ef - Ei

        if dE < 0.0:
            lattice[i,0] = x
            lattice[i,1] = y
            lattice[i,2] = z
        elif exp(-dE * beta)*RAND_MAX > rand():
            lattice[i,0] = x
            lattice[i,1] = y
            lattice[i,2] = z

    return lattice


@cython.wraparound(False)
@cython.boundscheck(False)
def calcTC(ar[double, ndim = 2] lattice, ar[long, ndim = 2] nb): #calculate topological charge and scalar chirality
    cdef int V = lattice.shape[0]
    cdef int i,u,d,l,r
    cdef long double tc = 0.0
    cdef long double chi = 0.0
    cdef double complex solid_angle_1, solid_angle_2
    cdef double n1x,n1y,n1z
    cdef double n2x,n2y,n2z
    cdef double n3x,n3y,n3z
    cdef double n4x,n4y,n4z

    for i in range(V):
        l = nb[i,0]
        r = nb[i,1]
        u = nb[i,2]
        d = nb[i,3]

        x = lattice[i,0]
        y = lattice[i,1]
        z = lattice[i,2]

        n1x = lattice[u,0] #up
        n1y = lattice[u,1]
        n1z = lattice[u,2]

        n2x = lattice[r,0] #right
        n2y = lattice[r,1]
        n2z = lattice[r,2]

        n3x = lattice[d,0] #down
        n3y = lattice[d,1]
        n3z = lattice[d,2]

        n4x = lattice[l,0] #left
        n4y = lattice[l,1]
        n4z = lattice[l,2]

        solid_angle_1 = 1.0 + triple_dot(x, y, z, n1x, n1y, n1z, n2x, n2y, n2z) + I*triple_product(x, y, z, n1x, n1y, n1z, n2x, n2y, n2z)
        solid_angle_2 = 1.0 + triple_dot(x, y, z, n3x, n3y, n3z, n4x, n4y, n4z) + I*triple_product(x, y, z, n3x, n3y, n3z, n4x, n4y, n4z)

        tc = tc + 2.0*cimagl(clog(solid_angle_1) + clog(solid_angle_2))/float(4.0*3.141592653589793) #(2*I*clog(solid_angle)/(4pi)
        chi = chi + triple_product(x, y, z, n1x, n1y, n1z, n2x, n2y, n2z) + triple_product(x, y, z, n3x, n3y, n3z, n4x, n4y, n4z)

    return tc, chi/(8.0*3.141592653589793)

#compute S1*(S2xS3)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double triple_product(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3):
    cdef double scalar_chirality = x1*(y2*z3 - y3*z2) - y1*(x2*z3 - x3*z2) + z1*(x2*y3 - y2*x3)
    return scalar_chirality

#compute S1*S2 + S1*S3 + S2*S3
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double triple_dot(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3):
    cdef double scalar = x1*x2 + x1*x3 + x2*x3 + y1*y2 + y1*y3 + y2*y3 + z1*z2 + z1*z3 + z2*z3
    return scalar


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef long double probability(ar[double, ndim=1] eigenvalues, long x, double temperature_e, double mu): #Fermi-Dirac distribution
    cdef long double probability
    probability = 1.0/(exp((eigenvalues[x] - mu)/temperature_e)+1.0)
    return probability

@cython.wraparound(False)
@cython.boundscheck(False)
def hall_conduct(ar[double, ndim=1] eigenvalues, ar[double complex, ndim=2] eigenstates, ar[double complex, ndim=2] eigenstates_conj, ar[long, ndim=1] Jx_sparse, ar[long, ndim=1] Jy_sparse, double temperature_e, double mu, ar[double, ndim=1] eta):
    cdef int T = eta.shape[0]
    cdef long double en_delta = 7
    cdef int mm,nn,site, upper, lower, t
    cdef int int_temp, int_temp1
    cdef long double en_diff, summ, summxx
    cdef long double complex jxmn, jymn
    cdef long double current_operators
    cdef long double current_operators_xx

    cdef np.ndarray sigmaxy = np.zeros(T,dtype = float)
    cdef np.ndarray sigmaxx = np.zeros(T,dtype = float)
    cdef np.ndarray exact = np.zeros(T,dtype = complex)
    cdef np.ndarray exact_xx = np.zeros(T,dtype = complex)

    cdef int dim_ham = eigenstates.shape[0]
    lower = np.argmin(abs(eigenvalues+en_delta-mu))
    upper = np.argmin(abs(eigenvalues-en_delta-mu))
    for mm in range(dim_ham): #full spectrum calculation
         for nn in range(mm+1,dim_ham):
    #for mm in range(lower,upper-1):
    #    for nn in range(mm+1,upper):
            jxmn = 0.0
            jymn = 0.0
            for site in range(dim_ham):
                int_temp = Jx_sparse[site]
                jxmn = jxmn + eigenstates_conj[site,mm]*eigenstates[int_temp,nn]-eigenstates_conj[int_temp,mm]*eigenstates[site,nn]
                int_temp1 = Jy_sparse[site]
                jymn = jymn + eigenstates_conj[site,mm]*eigenstates[int_temp1,nn]-eigenstates_conj[int_temp1,mm]*eigenstates[site,nn]

            en_diff = (probability(eigenvalues,mm,temperature_e,mu)-probability(eigenvalues,nn,temperature_e,mu))
            summ = -en_diff*cimagl(jxmn*(conjl(jymn)))
            summxx = -en_diff*abs(conjl(jxmn)*jxmn)

            for t in range(T): #calculate tau dependency
                exact[t] = 1.0/((eigenvalues[mm]-eigenvalues[nn])*((eigenvalues[mm]-eigenvalues[nn]) + I*eta[t]))
                exact_xx[t] = eta[t]*exact[t]/(eigenvalues[nn]-eigenvalues[mm])

                current_operators = creal(summ*exact[t])

                current_operators_xx = creal(summxx*exact_xx[t])

                sigmaxy[t] = sigmaxy[t] + current_operators

                sigmaxx[t] = sigmaxx[t] + current_operators_xx

    sigmaxy[:] *= 25.132/dim_ham #L^2
    sigmaxx[:] *= 25.132/dim_ham

    return sigmaxy, sigmaxx


#cython hamiltonian
@cython.wraparound(False)
@cython.boundscheck(False)
def hamiltonian_cy(ar[double, ndim=2] lattice, ar[long, ndim=2] nb, ar[double complex, ndim=2] ham, double Jh): #Hund's coupling
    cdef int site
    cdef int V = lattice.shape[0]

    for site in range(V):
        ham[2*site+1,2*site+1] =  Jh*lattice[site,2]
        ham[2*site,2*site]     = -Jh*lattice[site,2]
        ham[2*site+1,2*site]   =  Jh*(lattice[site,0] - I*lattice[site,1])
        ham[2*site,2*site+1]   =  Jh*(lattice[site,0] + I*lattice[site,1])

    return ham
