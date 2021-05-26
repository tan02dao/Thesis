import numpy as np

def initialize_lattice(V):
    lattice = np.zeros((V,3))
    for i in range(V):
        xi = 2.0
        while (xi > 1.0):
            xt = 1.0 - 2.0*np.random.rand()
            yt = 1.0 - 2.0*np.random.rand()
            xi = xt*xt + yt*yt

        k = np.sqrt(1.0 - xi)
        lattice[i][0] = 2.0*xt*k
        lattice[i][1] = 2.0*yt*k
        lattice[i][2] = 1.0 - 2.0*xi

    return lattice

def get_neighbors(N):
    nb = np.zeros((N*N,4), dtype = int)
    for i in range(N):
        for j in range(N):
            u = (j+1)%N
            d = (j-1)%N
            r = (i+1)%N
            l = (i-1)%N

            nb[i+j*N][0] = l+j*N #left
            nb[i+j*N][1] = r+j*N #right
            nb[i+j*N][2] = i+u*N #up
            nb[i+j*N][3] = i+d*N #down

    return nb

def hopping_hamiltonian(nb,V,dim_ham,hopping):
    ham = np.zeros((dim_ham,dim_ham), dtype=np.complex128)
    for site in range(V):
        nbs = nb[site]
        for i in range(4):
            ham[2*site+1,2*nbs[i]+1] = hopping
            ham[2*site,2*nbs[i]] = hopping

    return ham


def current_ops_sparse(V,nb):
    Jx_sparse = np.zeros(2*V, dtype = np.long)
    Jy_sparse = np.zeros(2*V, dtype = np.long)
    for site in range(V):
        nbs = nb[site]
        Jx_sparse[2*site+1] = 2*nbs[0]+1
        Jx_sparse[2*site] = 2*nbs[0]
        Jy_sparse[2*site+1] = 2*nbs[2]+1
        Jy_sparse[2*site] = 2*nbs[2]

    return Jx_sparse, Jy_sparse

def current_ops(V,nb):
    Jx = np.zeros((dim_ham, dim_ham), dtype = np.complex128)
    Jy = np.zeros((dim_ham, dim_ham), dtype = np.complex128)
    for site in range(V):
        nbs = nb[site,:]
        Jx[2*site+1,2*nbs[0]+1] = 1j
        Jx[2*site,2*nbs[0]] = 1j
        Jx[2*site+1,2*nbs[1]+1] = -1j
        Jx[2*site,2*nbs[1]] = -1j
        Jy[2*site+1,2*nbs[2]+1] = 1j
        Jy[2*site,2*nbs[2]] = 1j
        Jy[2*site+1,2*nbs[3]+1] = -1j
        Jy[2*site,2*nbs[3]] = -1j

    return Jx, Jy
