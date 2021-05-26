import numpy as np
import matplotlib.pyplot as plt
import timeit

from simulation import *
import initialize as init

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

def sub_hamiltonian(nb,V,dim_ham):
    ham = np.zeros((dim_ham,dim_ham), dtype=np.complex128)
    for site in range(V):
        nbs = nb[site]
        for i in range(4):
            ham[2*site+1,2*nbs[i]+1] = -1.0
            ham[2*site,2*nbs[i]] = -1.0

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


#~~~~~~~~~~********~~~~~~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~~***********~~~~~~~~~~~~~~~
#---------------------------------------------------------------------------------------
N = 18
D = 0.9
B = 0.45
Jh = -8
mu = -10
num_ave = 1
tau = np.arange(1,100,1)
eta = 1.0/tau
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
V = N*N
dim_ham = 2*V

lattice = initialize_lattice(V)
# dataset = np.loadtxt("spins-{}_{}x{}.txt".format(1,N,N)) #load spin configuration
# for k in range(V):
#     lattice[k][0] = dataset[k][0]
#     lattice[k][1] = dataset[k][1]
#     lattice[k][2] = dataset[k][2]

nb = get_neighbors(N)
Jx_sparse, Jy_sparse = current_ops_sparse(V,nb)
Jx, Jy = current_ops(V,nb)
#------------------------------------------------------------------------------------
t_init = np.linspace(2.0,0.02,30)
#
for j in range(len(t_init)): #Annealing step/ comment this step when loading in a spin config.
    beta = 1/t_init[j]
    for i in range(10000):
       lattice = mc_update(lattice, beta, B, D, nb)


# lat = open("spins-{}_{}x{}_{}.txt".format(t_init[-1],N,N,i),"+w") #save the spin configuration
# for ii in range(V):
#     lat.write(str(lattice[ii,0]) + " " + str(lattice[ii,1]) + " " + str(lattice[ii,2]) + "\n" )
#     lat.flush()

#---------------------------------------------------------------------------------------
#f = open("tau_dependent_T{}B{}J{}mu{}_skx_{}ave_full.txt".format(t_init[-1],B,Jh,mu,num_ave),"w+")
#f = open("tau_dependent_T{}B{}J{}mu{}_flc_{}ave_full.txt".format(t_init[-1],B,Jh,mu,num_ave),"w+")
#g = open("Sigmaxx_T{}J{}mu{}_pol_flc.txt".format(t_init[-1],Jh,mu),"w+")
#f = open("Sigmaxy_T{}eta{}.txt".format(t_init[-1],eta),"w+")

ham0 = sub_hamiltonian(nb,V,dim_ham).copy() #hopping hamiltonian


beta = 1.0/t_init[-1]
temperature = t_init[-1]

Sigmaxy = np.zeros(len(eta))
Sigmaxx = np.zeros(len(eta))

Sigxy = 0.0
Sigxx = 0.0

start_time = timeit.default_timer()
for i in range(num_ave):
    for l in range(200): #update a few hundred times before taking data
        lattice = mc_update(lattice, beta, B, D, nb)

    ham = hamiltonian_cy(lattice, nb, ham0, Jh)
    eigenvalues, eigenstates = np.linalg.eigh(ham)
    eigenstates_conj = eigenstates.conj()
    sigxy, sigxx = hall_conduct(eigenvalues, eigenstates, eigenstates_conj, Jx_sparse, Jy_sparse, temperature, mu, eta)
    Sigxy += sigxy
    Sigxx += sigxx

Sigmaxy = Sigxy/(num_ave)
Sigmaxx = Sigxx/(num_ave)

# for i in range(len(eta)):
#     f.write(str(tau[i]) + "      " + str(Sigmaxy[i]) + "      " + str(Sigmaxx[i]) +  "\n")
#     f.flush()


#g.write(str(Sigmaxx) + "      " + str(Sigmaxx1[j]) + "      " + str(Sigmaxx2[j]) + "      " + str(Sigmaxx3[j]) + "      " + str(Sigmaxx4[j]) + "      " + str(Sigmaxx5[j]) + "      " + str(Sigmaxx6[j]) + "\n")
#g.flush()
# err.write(str(Sigmaxy0_err[j]) + "      " + str(Sigmaxy1_err[j]) + "      " + str(Sigmaxy2_err[j]) + "      " + str(Sigmaxy3_err[j]) + "      " + str(Sigmaxy4_err[j]) + "      " + str(Sigmaxy5_err[j]) + "      " + str(Sigmaxy6_err[j]) + "      " + str(TC_err[j]) + "      "  + str(Chirality_err[j]) + "\n")
# err.flush()
#err.write(str(Sigmaxy0_err[j]) + "      " + str(Sigmaxy1_err[j]) + "      " + str(Sigmaxy2_err[j]) + "      " + str(Sigmaxy3_err[j]) + "      " + str(Sigmaxy4_err[j]) + "      " + str(Sigmaxy5_err[j]) + "      " + str(Sigmaxy6_err[j]) + "\n")
#err.flush()
# xtc.write(str(t[j]) + "      " + str(TC[j]) + "       " + str(Chirality[j]) + "     " + str(TC_err[j]) + "      "  + str(Chirality_err[j]) + "\n")
# xtc.flush()

elapsed = timeit.default_timer() - start_time
print(elapsed)


x,y = np.arange(0,N,1),np.arange(0,N,1)
x,y = np.meshgrid(x,y)
plt.figure(1, figsize = (7,7))
u,v,s = lattice[x+y*N,0],lattice[x+y*N,1],lattice[x+y*N,2] #xy
fig = plt.quiver(x,y,u,v,s,cmap=plt.cm.jet, clim=[-1,1])
plt.show()

#
plt.figure(1, figsize = (7,7))
plt.plot(tau,Sigmaxy,'r.')
plt.show()

plt.figure(2, figsize = (7,7))
plt.plot(tau,Sigmaxx,'r.')
plt.show()

plt.figure(2, figsize = (7,7))
plt.plot(tau,Sigmaxy/(Sigmaxx**2 + Sigmaxy**2),'r.')
plt.show()
