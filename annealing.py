import numpy as np
import matplotlib.pyplot as plt
import timeit
from math import sqrt
#import matplotlib as mpl

from Hall_Conduct_main import *

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
N = 16
t = np.arange(2.0,0.1-0.025,-0.025)#np.linspace(1.6,0.2,50)
D = 0.9
B = 0.7
num_ave = 10000
mu = -0.1
epsilon = np.asarray([0.05,0.1,0.2,0.3,0.4,0.5,0.6])#np.asarray([0.01,0.02,0.05,0.1,0.2,0.5,1.0])
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
V = N*N
dim_ham = 2*V

lattice = initialize_lattice(V)
nb = get_neighbors(N)
Jx_sparse, Jy_sparse = current_ops_sparse(V,nb)
Jx, Jy = current_ops(V,nb)
#------------------------------------------------------------------------------------
t_init = np.linspace(10.0,2.0,50)

for j in range(len(t_init)): #Annealing step
    beta = 1/t_init[j]
    for i in range(10000):
        lattice = mc_update(lattice, beta, B, D, nb)
#---------------------------------------------------------------------------------------
Sigmaxy0 = np.zeros(len(t))
Sigmaxy1 = np.zeros(len(t))
Sigmaxy2 = np.zeros(len(t))
Sigmaxy3 = np.zeros(len(t))
Sigmaxy4 = np.zeros(len(t))
Sigmaxy5 = np.zeros(len(t))
Sigmaxy6 = np.zeros(len(t))

Sigmaxx0 = np.zeros(len(t))
Sigmaxx1 = np.zeros(len(t))
Sigmaxx2 = np.zeros(len(t))
Sigmaxx3 = np.zeros(len(t))
Sigmaxx4 = np.zeros(len(t))
Sigmaxx5 = np.zeros(len(t))
Sigmaxx6 = np.zeros(len(t))

TC = np.zeros(len(t))
Chirality = np.zeros(len(t))

Sigmaxy0_err = np.zeros(len(t))
Sigmaxy1_err = np.zeros(len(t))
Sigmaxy2_err = np.zeros(len(t))
Sigmaxy3_err = np.zeros(len(t))
Sigmaxy4_err = np.zeros(len(t))
Sigmaxy5_err = np.zeros(len(t))
Sigmaxy6_err = np.zeros(len(t))

TC_err = np.zeros(len(t))
Chirality_err = np.zeros(len(t))
Sigmaxx_err = np.zeros(len(t))
#snapshot = np.zeros((len(t),V,3))


#f = open("D{}H{}mu{}e{}_{}x{}_{}ave_ecut3.txt".format(D,B,mu,epsilon,N,N,num_ave),"w+")
#g = open("D{}H{}mu{}e{}_{}x{}_Sxx_{}ave_ecut3.txt".format(D,B,mu,epsilon,N,N,num_ave),"w+")
# err = open("D{}H{}mu{}e{}_{}x{}_{}ave_error_4.0.txt".format(D,B,mu,epsilon,N,N,num_ave),"w+")

#err = open("D{}H{}mu{}e{}_{}x{}_Sxxerror.txt".format(D,B,mu,epsilon,N,N,num_ave),"w+")
#xtc = open("D{}H{}_{}x{}_{}ave_TC.txt".format(D,B,N,N,num_ave),"w+")


ham0 = sub_hamiltonian(nb,V,dim_ham).copy()

start_time = timeit.default_timer()

#Annealing step + taking data
for j in range(len(t)):
    beta = 1.0/t[j]
    temperature_e = t[j]
    for k in range(20000): #annealing
        lattice = mc_update(lattice, beta, B, D, nb)

    Tc = 0.0
    Chiral = 0.0
    Sigxy0 = 0.0
    Sigxy1 = 0.0
    Sigxy2 = 0.0
    Sigxy3 = 0.0
    Sigxy4 = 0.0
    Sigxy5 = 0.0
    Sigxy6 = 0.0

    Sigxx0 = 0.0
    Sigxx1 = 0.0
    Sigxx2 = 0.0
    Sigxx3 = 0.0
    Sigxx4 = 0.0
    Sigxx5 = 0.0
    Sigxx6 = 0.0

    Tc_2 = 0.0
    Chiral_2 = 0.0
    Sigxy0_2 = 0.0
    Sigxy1_2 = 0.0
    Sigxy2_2 = 0.0
    Sigxy3_2 = 0.0
    Sigxy4_2 = 0.0
    Sigxy5_2 = 0.0
    Sigxy6_2 = 0.0
    Sigxx_2 = 0.0

    for i in range(num_ave): #taking data
        for l in range(20):
            lattice = mc_update(lattice, beta, B, D, nb)

        tc = calcTC(lattice,nb)
        Tc += tc
        Tc_2 += tc*tc

        chiral = calcChirality(lattice,nb)
        Chiral += chiral
        Chiral_2 += chiral*chiral

    #     ham = hamiltonian_cy(lattice, nb, ham0)
    #     eigenvalues, eigenstates = np.linalg.eigh(ham)
    #     eigenstates_conj = eigenstates.conj()
    #     sigxy0, sigxy1, sigxy2, sigxy3, sigxy4, sigxy5, sigxy6, sigxx0, sigxx1, sigxx2, sigxx3, sigxx4, sigxx5, sigxx6 = hall_conduct_cy(eigenvalues, eigenstates, eigenstates_conj, Jx_sparse, Jy_sparse, temperature_e, mu, epsilon)
    #     Sigxy0 += sigxy0
    #     Sigxy1 += sigxy1
    #     Sigxy2 += sigxy2
    #     Sigxy3 += sigxy3
    #     Sigxy4 += sigxy4
    #     Sigxy5 += sigxy5
    #     Sigxy6 += sigxy6
    #
    #     Sigxx0 += sigxx0
    #     Sigxx1 += sigxx1
    #     Sigxx2 += sigxx2
    #     Sigxx3 += sigxx3
    #     Sigxx4 += sigxx4
    #     Sigxx5 += sigxx5
    #     Sigxx6 += sigxx6
    #
    #     Sigxy0_2 += sigxy0*sigxy0
    #     Sigxy1_2 += sigxy1*sigxy1
    #     Sigxy2_2 += sigxy2*sigxy2
    #     Sigxy3_2 += sigxy3*sigxy3
    #     Sigxy4_2 += sigxy4*sigxy4
    #     Sigxy5_2 += sigxy5*sigxy5
    #     Sigxy6_2 += sigxy6*sigxy6
    #
    #
    # Sigmaxy0[j] = Sigxy0/(num_ave)
    # Sigmaxy1[j] = Sigxy1/(num_ave)
    # Sigmaxy2[j] = Sigxy2/(num_ave)
    # Sigmaxy3[j] = Sigxy3/(num_ave)
    # Sigmaxy4[j] = Sigxy4/(num_ave)
    # Sigmaxy5[j] = Sigxy5/(num_ave)
    # Sigmaxy6[j] = Sigxy6/(num_ave)
    #
    # Sigmaxx0[j] = Sigxx0/(num_ave)
    # Sigmaxx1[j] = Sigxx1/(num_ave)
    # Sigmaxx2[j] = Sigxx2/(num_ave)
    # Sigmaxx3[j] = Sigxx3/(num_ave)
    # Sigmaxx4[j] = Sigxx4/(num_ave)
    # Sigmaxx5[j] = Sigxx5/(num_ave)
    # Sigmaxx6[j] = Sigxx6/(num_ave)
    #
    # Sigmaxy0_err[j] = sqrt((Sigxy0_2/num_ave - Sigxy0*Sigxy0/(num_ave*num_ave))/num_ave)
    # Sigmaxy1_err[j] = sqrt((Sigxy1_2/num_ave - Sigxy1*Sigxy1/(num_ave*num_ave))/num_ave)
    # Sigmaxy2_err[j] = sqrt((Sigxy2_2/num_ave - Sigxy2*Sigxy2/(num_ave*num_ave))/num_ave)
    # Sigmaxy3_err[j] = sqrt((Sigxy3_2/num_ave - Sigxy3*Sigxy3/(num_ave*num_ave))/num_ave)
    # Sigmaxy4_err[j] = sqrt((Sigxy4_2/num_ave - Sigxy4*Sigxy4/(num_ave*num_ave))/num_ave)
    # Sigmaxy5_err[j] = sqrt((Sigxy5_2/num_ave - Sigxy5*Sigxy5/(num_ave*num_ave))/num_ave)
    # Sigmaxy6_err[j] = sqrt((Sigxy6_2/num_ave - Sigxy6*Sigxy6/(num_ave*num_ave))/num_ave)



    TC[j] = Tc/(num_ave)*1000/V
    #TC_err[j] = sqrt((Tc_2/num_ave - Tc*Tc/(num_ave*num_ave))/num_ave)*1000/V
    Chirality[j] = Chiral/(num_ave)*1000/V
    #Chirality_err[j] = sqrt((Chiral_2/num_ave - Chiral*Chiral/(num_ave*num_ave))/num_ave)*1000/V
    #print(Sigmaxy0[j],Sigmaxy2[j],Sigmaxy6[j])
    # f.write(str(t[j]) + "      " + str(Sigmaxy0[j]) + "      " + str(Sigmaxy1[j]) + "      " + str(Sigmaxy2[j]) + "      " + str(Sigmaxy3[j]) + "      " + str(Sigmaxy4[j]) + "      " + str(Sigmaxy5[j]) + "      " + str(Sigmaxy6[j]) +  "      " + str(TC[j]) + "       " + str(Chirality[j]) + "\n")
    # f.flush()
    # g.write(str(Sigmaxx0[j]) + "      " + str(Sigmaxx1[j]) + "      " + str(Sigmaxx2[j]) + "      " + str(Sigmaxx3[j]) + "      " + str(Sigmaxx4[j]) + "      " + str(Sigmaxx5[j]) + "      " + str(Sigmaxx6[j]) + "\n")
    # g.flush()
    # err.write(str(Sigmaxy0_err[j]) + "      " + str(Sigmaxy1_err[j]) + "      " + str(Sigmaxy2_err[j]) + "      " + str(Sigmaxy3_err[j]) + "      " + str(Sigmaxy4_err[j]) + "      " + str(Sigmaxy5_err[j]) + "      " + str(Sigmaxy6_err[j]) + "      " + str(TC_err[j]) + "      "  + str(Chirality_err[j]) + "\n")
    # err.flush()
    #err.write(str(Sigmaxy0_err[j]) + "      " + str(Sigmaxy1_err[j]) + "      " + str(Sigmaxy2_err[j]) + "      " + str(Sigmaxy3_err[j]) + "      " + str(Sigmaxy4_err[j]) + "      " + str(Sigmaxy5_err[j]) + "      " + str(Sigmaxy6_err[j]) + "\n")
    #err.flush()
    # xtc.write(str(t[j]) + "      " + str(TC[j]) + "       " + str(Chirality[j]) + "     " + str(TC_err[j]) + "      "  + str(Chirality_err[j]) + "\n")
    # xtc.flush()

elapsed = timeit.default_timer() - start_time
print(elapsed)

# lat = open("spins-{}_{}x{}.txt".format(1,N,N),"w+")
# for ii in range(V):
#     lat.write(str(lattice[ii,0]) + " " + str(lattice[ii,1]) + " " + str(lattice[ii,2]) + "\n" )
#     lat.flush()


# x,y = np.arange(0,N,1),np.arange(0,N,1)
# x,y = np.meshgrid(x,y)
#
# r = 0
# up = 0
#
# plt.figure(1, figsize = (7,7))
# u,v,s = lattice[(x-r)%N+((y-up)%N)*N,0],lattice[(x-r)%N+((y-up)%N)*N,1],lattice[(x-r)%N+((y-up)%N)*N,2] #xy
# #u,v,s = field[x,y,0],field[x,y,1],field[x,y,2] #xy
# fig = plt.quiver(x,y,u,v,s,cmap=plt.cm.jet, clim=[-1,1])
# plt.show()


#
# plt.figure(1, figsize = (7,7))
# plt.plot(t,-Sigmaxy0,'r.')
# plt.plot(t,-Sigmaxy1,'g.')
# plt.plot(t,-Sigmaxy2,'b.')
# plt.plot(t,-Sigmaxy3,'c.')
# plt.plot(t,-Sigmaxy4,'m.')
# plt.plot(t,-Sigmaxy5,'k.')
# plt.show()
# #
#
# plt.figure(2, figsize = (7,7))
# plt.plot(t,Sigmaxx0,'r.')
# plt.plot(t,Sigmaxx1,'g.')
# plt.plot(t,Sigmaxx2,'b.')
# plt.plot(t,Sigmaxx3,'c.')
# plt.plot(t,Sigmaxx4,'m.')
# plt.plot(t,Sigmaxx5,'k.')
# plt.plot(t,Sigmaxx6,'y.')
# plt.show()

#
plt.figure(3, figsize = (7,7))
#plt.plot(t,-Sigmaxy1/Sigmaxx1**2,'g.',label = "sigmaxy")
#plt.plot(t,-Sigmaxy2/Sigmaxx2**2,'r.',label = "sigmaxy")
plt.plot(t,TC,'b.')
plt.plot(t,Chirality*min(TC)/min(Chirality),'r.')
plt.legend()
plt.show()
