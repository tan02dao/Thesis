import numpy as np
import matplotlib.pyplot as plt
import timeit
from math import sqrt
#import matplotlib as mpl

from simulation import *
import initialize as init


#~~~~~~~~~~********~~~~~~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~~***********~~~~~~~~~~~~~~~
#---------------------------------------------------------------------------------------
N = 16
t = np.arange(1.5,0.04-0.02,-0.02)
D = 0.9
B = 0.45
num_ave = 1000
Jh = -8
mu = -10
tau = np.arange(10,100,10)
eta = 1.0/tau
hopping = -1.0
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
V = N*N
dim_ham = 2*V

lattice = init.initialize_lattice(V)
nb = init.get_neighbors(N)
Jx_sparse, Jy_sparse = init.current_ops_sparse(V,nb)
ham0 = init.hopping_hamiltonian(nb,V,dim_ham,hopping).copy()
#------------------------------------------------------------------------------------
temp = np.linspace(5.0,2.0,30) #Temperature for annealing step
for j in range(len(temp)):
    beta = 1/temp[j]
    for i in range(10000):
        lattice = mc_update(lattice, beta, B, D, nb)
#---------------------------------------------------------------------------------------
Sigmaxy = np.zeros((len(t),len(tau)))
Sigmaxx = np.zeros((len(t),len(tau)))

TC = np.zeros(len(t))
Chirality = np.zeros(len(t))

Sigmaxy_err = np.zeros((len(t),len(tau)))
Sigmaxx_err = np.zeros((len(t),len(tau)))

TC_err = np.zeros(len(t))
Chirality_err = np.zeros(len(t))
#snapshot = np.zeros((len(t),V,3))


#f = open("D{}H{}mu{}e{}_{}x{}_{}ave_ecut3.txt".format(D,B,mu,eta,N,N,num_ave),"w+")
#g = open("D{}H{}mu{}e{}_{}x{}_Sxx_{}ave_ecut3.txt".format(D,B,mu,eta,N,N,num_ave),"w+")
# err = open("D{}H{}mu{}e{}_{}x{}_{}ave_error_4.0.txt".format(D,B,mu,eta,N,N,num_ave),"w+")

#err = open("D{}H{}mu{}e{}_{}x{}_Sxxerror.txt".format(D,B,mu,eta,N,N,num_ave),"w+")
#xtc = open("D{}H{}_{}x{}_{}ave_TC.txt".format(D,B,N,N,num_ave),"w+")


start_time = timeit.default_timer()
#Annealing step + taking data
for j in range(len(t)):
    beta = 1.0/t[j]
    temperature = t[j]
    for k in range(10000): #annealing
        lattice = mc_update(lattice, beta, B, D, nb)

    Tc = 0.0
    Chi = 0.0
    Sigxy = 0.0
    Sigxx = 0.0

    for i in range(num_ave): #taking data
        for l in range(20):
            lattice = mc_update(lattice, beta, B, D, nb)

        tc, chi = calcTC(lattice,nb)
        Tc += tc
        Chi += chi
        #Tc_2 += tc*tc

    #     ham = hamiltonian_cy(lattice, nb, ham0, Jh)
    #     eigenvalues, eigenstates = np.linalg.eigh(ham)
    #     eigenstates_conj = eigenstates.conj()
    #     sigxy, sigxx = hall_conduct(eigenvalues, eigenstates, eigenstates_conj, Jx_sparse, Jy_sparse, temperature, mu, eta)
    #
    #     Sigxy += sigxy
    #     Sigxx += sigxx
    #
    # Sigmaxy[j] = Sigxy/num_ave
    # Sigmaxx[j] = Sigxx/num_ave
    TC[j] = Tc/(num_ave)*1000/V
    #TC_err[j] = sqrt((Tc_2/num_ave - Tc*Tc/(num_ave*num_ave))/num_ave)*1000/V
    Chirality[j] = Chi/(num_ave)*1000/V
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


x,y = np.arange(0,N,1),np.arange(0,N,1)
x,y = np.meshgrid(x,y)
plt.figure(1, figsize = (7,7))
u,v,s = lattice[x+y*N,0],lattice[x+y*N,1],lattice[x+y*N,2] #xy
fig = plt.quiver(x,y,u,v,s,cmap=plt.cm.jet, clim=[-1,1])
plt.show()


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
# plt.figure(1, figsize = (7,7))
# plt.plot(t,Sigmaxy[:,0],'r.')
# plt.plot(t,Sigmaxy[:,2],'g.')
# plt.plot(t,Sigmaxy[:,4],'b.')
# plt.show()
#
# plt.figure(1, figsize = (7,7))
# plt.plot(t,Sigmaxx[:,0],'r.')
# plt.plot(t,Sigmaxx[:,2],'g.')
# plt.plot(t,Sigmaxx[:,4],'b.')
# plt.show()
#
# plt.figure(2, figsize = (7,7))
# plt.plot(t,Sigmaxy[:,0]/(Sigmaxx[:,0]**2+Sigmaxy[:,0]**2),'r.')
# plt.plot(t,Sigmaxy[:,2]/(Sigmaxx[:,2]**2+Sigmaxy[:,2]**2),'g.')
# plt.plot(t,Sigmaxy[:,4]/(Sigmaxx[:,4]**2+Sigmaxy[:,4]**2),'b.')
# plt.show()

#
plt.figure(3, figsize = (7,7))
plt.plot(t,TC,'b.')
plt.plot(t,Chirality,'r.')
plt.show()
