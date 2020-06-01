import nbinteract as nbi

import numpy as np
import matplotlib.pyplot as plt

#from dashboard import *

Species = ['non fixer', 'facultative fixer', 'obligate fixer']

def payoff_matrix_param_low(alpha, beta, gamma):
    Al = np.array([[0, -alpha, -beta],
                   [alpha, 0, -gamma],
                   [beta, gamma, 0]])
    return Al

def payoff_matrix_param_high(Alpha, Beta, Gamma):
    Ah = np.array([[0, Alpha, Beta],
                   [-Alpha, 0, Gamma],
                   [-Beta, -Gamma, 0]])
    return Ah


def payoff_matrix_param(alpha, beta, gamma, Alpha, Beta, Gamma):
    A = np.zeros((2,3,3))
    A[0] = payoff_matrix_param_low(alpha, beta, gamma)
    A[1] = payoff_matrix_param_high(Alpha, Beta, Gamma)
    return A    


def F(X, An):
    X.dot(An)
    phi = X.dot(An).dot(X)
#    phi = X.dot(An.dot(X))
    return X*(An.dot(X) - phi)


"""
An = payoff_matrix_param_low(1, 2, 3)
plot_determinist_level(An)
"""


"""
def plot_determinist(A, show=True):
    nbre_d_eq = 3
#    T = np.linspace(0, 100000, 1000001)
    T = np.linspace(1, 10000, 100001)
    dt = T[1] - T[0]
    XX = np.zeros((len(T), nbre_d_eq))
    X0 = np.zeros(nbre_d_eq) + 1./nbre_d_eq
    
    #plt.subplots(1,2,figsize = (20, 6))
    XX = np.zeros((2, len(T), nbre_d_eq))
    for ind_n in range(2):
    #    plt.figure(figsize = (10, 5))
        XX[ind_n, 0] = X0
        for i, t in enumerate(T[:-1:]):
            An = A[ind_n]
            XX[ind_n, i+1] = XX[ind_n, i] + dt*F(XX[ind_n, i], An)
    
#    plt.figure(figsize = (20, 5))
    for ind_n, n in enumerate(["low", "high"]):   
        if(show):
            plt.figure(figsize = (10, 7))
        else:
            plt.subplot(1,2,ind_n+1)
        for i in range(3):
            plt.semilogx(T, XX[ind_n,:,i], label=Species[i])
       # plt.plot(T, XX[:,3], label="x_4")
        #plt.plot(T, np.sum(XX, axis = 1), "--", color = "grey", label="sum")
        plt.legend(fontsize = 15)
        plt.title("Nitrogen "+n, fontsize=25)
        plt.xlabel("Time (log scale)", fontsize = 20)
        plt.ylabel("biomass proportion", fontsize = 20)
        plt.xticks(fontsize = 15)
        #plt.savefig("arbitrary_payoff_proportion_n="+str(n)+".png")
        if(show):
            plt.show()
    return

"""


def plot_determinist_level(An, T, final_time=10**4, show=True):
    nbre_d_eq = 3
#    T = np.linspace(1, 10000, 100001)
#    T = np.linspace(1, 10, 11)
#    T = np.linspace(1, final_time, final_time+1)
    dt = T[1] - T[0]
    XX = np.zeros((len(T), nbre_d_eq))
    X0 = np.zeros(nbre_d_eq) + 1./nbre_d_eq
    
    #plt.subplots(1,2,figsize = (20, 6))
#    XX = np.zeros((len(T), nbre_d_eq))
  #  for ind_n in range(2):
    #    plt.figure(figsize = (10, 5))
    XX[0] = X0
    for i, t in enumerate(T[:-1:]):
        XX[i+1] = XX[i] + dt*F(XX[i], An)
    
    if(show):
        plt.figure(figsize = (10, 7))
    for i in range(3):
        plt.semilogx(T, XX[:,i], label=Species[i])
    plt.legend(fontsize = 15)
    plt.xlabel("Time (log scale)", fontsize = 20)
    plt.ylabel("biomass proportion", fontsize = 20)
    plt.xticks(fontsize = 15)
    #plt.savefig("arbitrary_payoff_proportion_n="+str(n)+".png")
    if(show):
        plt.show()
    return



def solve(T, A, plh, phl, X0 = np.zeros(3) + 1./3):
    dt = T[1] - T[0]
    XX = np.zeros((len(T), 3))
    
    XX[0] = X0
    
    l = False
    h = True
    NN = np.zeros_like(T)
    
    ph = plh/(plh+phl)
    NN[0] = (np.random.binomial(n=1, p=ph)==1)
    for i, t in enumerate(T[:-1:]):
        if(NN[i]==l):
            if(np.random.rand()<plh*dt):
                NN[i+1] = h
                An = A[1]
            else:
                NN[i+1] = l
                An = A[0]
        elif(NN[i]==h):
            if(np.random.rand()<phl*dt):
                NN[i+1] = l
                An = A[0]
            else:
                NN[i+1] = h
                An = A[1]
        XX[i+1] = XX[i] + dt*F(XX[i], An)
    return XX, NN



def plot_nitrogen(T, XX, NN, forest_type):
#    plt.figure(figsize=(10,5))
    plt.plot(T, NN)
    plt.ylabel("Time", fontsize = 25)
    plt.ylabel("Nitrogen", fontsize = 25)
    plt.yticks([0,1], ["low", "high"])
    plt.title(forest_type, fontsize=20)
    #plt.savefig("arbitrary_payoff_"+forest_type+"_nitrogen.png")
    return


def plot_biomass(T, XX, NN, forest_type):
 #   plt.figure(figsize=(10,5))
    for i in range(3):
        plt.plot(T, XX[:,i], label=Species[i])
    #plt.plot(T, XX[:,3], label="x_4")
    #plt.plot(T, np.sum(XX, axis = 1), "--", color = "grey", label="sum")
    #plt.legend(fontsize = 20, loc="lower right")
    plt.xlabel("Time", fontsize = 25)
    plt.ylabel("Biomass proportion", fontsize = 25)
    plt.title(forest_type, fontsize=20)
    #plt.savefig("arbitrary_payoff_"+forest_type+".png")
#    plt.show()

