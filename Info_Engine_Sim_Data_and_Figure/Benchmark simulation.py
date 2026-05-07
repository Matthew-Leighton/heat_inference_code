#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 07:39:34 2024

@author: johandubuisson
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import math

#Draws the increment \Delta x for a time increment dt from its exact distribution as given in Eq. 4 in the paper, given the relevant system parameters and
# the current value x of the bead and l of the trap. Since the trap position is updated only every 20 \mu s, the distribution Eq. 4 for the bead's increment 
# is exact for dt = 20 \mu s. std represents the standard deviation of the normal distrubution in Eq. 4 and is calculated once off and then saved
# later on since it involves exponential factors. A similar thing is done with avgfactor which saves exponential factors involved in the mean of the increment
# distribution
def update_delta_x(x,l,klink, grav, dt, avgfactor, std):
    return np.random.normal(avgfactor*(l - x - grav/klink), std)

#Define relevant physical parameters for the simulation. Quantities definined with the dimensions stated in the caption for Fig.2 
gamma = 2.6e-8
klink = 3.52e-5
grav = (1.4137e-14)*9.8
sigma = np.sqrt((298*1.38e-23)/klink)
#define fundamental timestep dt = 20 \mu s and relevant factors featuring exponentials and appearing in update_delta_x
dt = 0.00002
std = np.sqrt((sigma**2)*(1 - np.exp(-2*dt/(gamma/klink))))
avgfactor = (1 - np.exp(-dt/(gamma/klink)))

alpha = 1.8 #alpha featuring in the feedback rule for the trap update. this value implements an approximately zero-work condition

#define 3 sampling intervals representing different experimental capabailities in sampling frequency dt_sample1 corresponds to sampling at the experimentally possible 20 \mu s while dt_sample2 and dt_sample3 represents less frequent sampling
dt_sample1 = 0.00002
dt_sample2 = 5*dt_sample1
dt_sample3 = 10*dt_sample1

totaltime = 1. #total time = 1 second for each trajectory
N = 500 #total number of trajectories

#prepare arrays to calculate exact heat for a given trajectory and average velocity for a given trajectory
Q = np.zeros(N)
vel = np.zeros(N)
dx21 = [] #this array will contain the average MSD for each trajectory, where dx is measured in time increments of dt_sample1
Qaverage1 = [] #the ith entry of this array will be the result of our heat estimator applied on the MSDs estimated from the first 10+i trajectories. as i increases these entries converge to the true heat estimated  
Qerror1 = [] #the ith entry of this array will contain the standard error of the mean of the quantity Qaverage1[i]
dx22 = [] #this array will contain the average MSD for each trajectory sampled at intervals dt_sample2
#the definitions of the following quantities should make sense given the preceding comments
Qaverage2 = []
Qerror2 = []
dx23 = []
Qaverage3 = []
Qerror3 = []

#now we start simulating the dynamics for the information engine mechanism. first we let the system evolve for duration T so that the dynamics have reached a NEQ steady state
xold = 0.
lold = 0.
cond = 0.
for i in range(int(totaltime/dt)-1):
    xnew = xold + update_delta_x(xold,lold,klink,grav,dt,avgfactor,std)
    #the following if implements the feedback rule for the information engine mechanism
    if cond == 1.:
        lnew = lold + alpha*(xold - lold)
    else:
        lnew = lold
    if xnew > lnew:
        cond = 1.
    else:
        cond = 0.
    xold = xnew
    lold = lnew
#now we generate N trajectories which will be used to calculate the true heat and the results of the heat estimator for various sampling intervals and number of trajectories
for k in range(N):
    dx21.append(0.)
    dx22.append(0.)
    dx23.append(0.)
    x = np.zeros(int(totaltime/dt))
    x[0] = xold

    linit = lold
    #actual trajectory
    for i in range(int(totaltime/dt)-1):
        x[i+1] = x[i] + update_delta_x(x[i],lold,klink,grav,dt,avgfactor,std)
        if cond == 1.:
            lnew = lold + alpha*(x[i] - lold)
        else:
            lnew = lold
        if x[i+1] > lnew:
            cond = 1.
        else:
            cond = 0.
        dx21[k] = dx21[k] + (x[i+1] - x[i])**2 #calculates sum of dx^2 for the trajectory.
        Q[k] = Q[k] + (klink/2)*(x[i+1] - lold)**2 + grav*x[i+1] - ((klink/2)*(x[i] - lold)**2 + grav*x[i]) #calculates total heat for the trajectory
        lold = lnew
    xold = x[-1]
    lend = lold
    vel[k] = (lend - linit)/totaltime
    Q[k] = Q[k]/((298*1.38e-23)*totaltime) #scales the heat by k_B T and the toral time T to obtain the heat flow rate per k_B T
    dx21[k] = dx21[k]/(totaltime/dt - 1) #finds trajectory average of dx^2
    x2 = x[::5]
    x3 = x[::10]
    for j in range(len(x2)-1):
        dx22[k] = dx22[k] + (x2[j+1]-x2[j])**2 #finds sum of dx^2 for total trajectory for sampling interval dt_sample2
    for j in range(len(x3)-1):
        dx23[k] = dx23[k] + (x3[j+1]-x3[j])**2 #finds sum of dx^2 for total trajectory for sampling interval dt_sample3
    dx22[k] = dx22[k]/(totaltime/dt_sample2 - 1) #finds trajectory average of dx^2 for sampling interval dt_sample2
    dx23[k] = dx23[k]/(totaltime/dt_sample3 - 1)
    if k >= 9: #here we calculate the heat flow using the heat estimator and its standard deviation for the first k trajectories given different sampling intervals 
        Qaverage1.append((2/dt)*(1 - np.mean(dx21)/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink)))))) #note that an analytical expression is used for <dx^2>_eq. this reflects the fact that we can determine this easily for a given bead and linker
        Qerror1.append(sem((2/dt)*(1 - dx21/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink)))))))
        Qaverage2.append((2/dt_sample2)*(1 - np.mean(dx22)/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink))))))
        Qerror2.append(sem((2/dt_sample2)*(1 - dx22/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink)))))))
        Qaverage3.append((2/dt_sample3)*(1 - np.mean(dx23)/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink))))))
        Qerror3.append(sem((2/dt_sample3)*(1 - dx23/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink)))))))

#in principle the arrays Qaverage1, Qerror1 and so forth can now be exported easily via np.save(Qaverage1)

#we print some values to see how the neq MSD and eq MSD (calculated analytically) compare. We also print velocities and true heats for easy comparison and consistency checks
print(np.mean(dx21),sem(dx21),(2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))
print(np.mean(Q),sem(Q))
print(np.mean(vel),sem(vel))

#at this point we simply repeat what has come before but for the conventional engine mechanism. new arrays are created and the trajectories are generated, as before
Qnew = np.zeros(N)
velnew = np.zeros(N)
dx21 = []
Qaveragenew1 = []
Qerrornew1 = []
dx22 = []
Qaveragenew2 = []
Qerrornew2 = []
dx23 = []
Qaveragenew3 = []
Qerrornew3 = []

#equilibrate
xold = 0.
lold = 0.
cond = 0.
for i in range(int(totaltime/dt)-1):
    xnew = xold + update_delta_x(xold,lold,klink,grav,dt,avgfactor,std)
    lnew = lold + np.mean(vel)*dt
    xold = xnew
    lold = lnew

for k in range(N):
    dx21.append(0.)
    dx22.append(0.)
    dx23.append(0.)
    x = np.zeros(int(totaltime/dt))
    x[0] = xold

    linit = lold
    #actual trajectory
    for i in range(int(totaltime/dt)-1):
        x[i+1] = x[i] + update_delta_x(x[i],lold,klink,grav,dt,avgfactor,std)
        lnew = lold + np.mean(vel)*dt
        dx21[k] = dx21[k] + (x[i+1] - x[i])**2
        Qnew[k] = Qnew[k] + (klink/2)*(x[i+1] - lold)**2 + grav*x[i+1] - ((klink/2)*(x[i] - lold)**2 + grav*x[i])
        lold = lnew
    xold = x[-1]
    lend = lold
    velnew[k] = (lend - linit)/totaltime
    Qnew[k] = Qnew[k]/((298*1.38e-23)*totaltime)
    dx21[k] = dx21[k]/(totaltime/dt - 1)
    x2 = x[::5]
    x3 = x[::10]
    for j in range(len(x2)-1):
        dx22[k] = dx22[k] + (x2[j+1]-x2[j])**2
    for j in range(len(x3)-1):
        dx23[k] = dx23[k] + (x3[j+1]-x3[j])**2
    dx22[k] = dx22[k]/(totaltime/dt_sample2 - 1)
    dx23[k] = dx23[k]/(totaltime/dt_sample3 - 1)
    if k >= 9:
        Qaveragenew1.append((2/dt)*(1 - np.mean(dx21)/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))))
        Qerrornew1.append(sem((2/dt)*(1 - dx21/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink)))))))
        Qaveragenew2.append((2/dt_sample2)*(1 - np.mean(dx22)/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink))))))
        Qerrornew2.append(sem((2/dt_sample2)*(1 - dx22/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink)))))))
        Qaveragenew3.append((2/dt_sample3)*(1 - np.mean(dx23)/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink))))))
        Qerrornew3.append(sem((2/dt_sample3)*(1 - dx23/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink)))))))

#here we produce plots similar to Fig.2 but with a different color scheme and labels. to get exact figures like Fig2 export the data files and use plotdata.py 
plt.axhline(y = 0. ,color='black', linestyle='-',zorder=7)
plt.axhline(y=np.mean(Q), color='black', linestyle='--',zorder=7,label = 'True heat flow')
plt.scatter(np.linspace(10,N,N-10+1),Qaverage1, color = 'blue', s = 1,zorder=5)
plt.errorbar(np.linspace(10,N,N-10+1),Qaverage1, yerr = Qerror1, color = 'blue', fmt = '.',elinewidth=1,zorder=6,alpha = 0.3)
plt.scatter(np.linspace(10,N,N-10+1),Qaverage2, color = 'red', s = 1,zorder=3)
plt.errorbar(np.linspace(10,N,N-10+1),Qaverage2, yerr = Qerror2, color = 'red', fmt = '.',elinewidth=1,zorder=4,alpha = 0.3)
plt.scatter(np.linspace(10,N,N-10+1),Qaverage3, color = 'forestgreen', s = 1,zorder=2)
plt.errorbar(np.linspace(10,N,N-10+1),Qaverage3, yerr = Qerror3, color = 'forestgreen', fmt = '.',elinewidth=1,zorder=1,alpha = 0.3)
plt.axhline(y=np.mean(Q), color='black', linestyle='--',zorder=7)
plt.xscale('log')
plt.xlabel('Number of trajectories of length 1s')
plt.ylabel(r'$\dot{Q}_X (k_{B} T/s)$')
plt.savefig("Benchmark simulation information.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(np.mean(dx21),sem(dx21),(2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))
print(np.mean(Qnew),sem(Qnew))
print(np.mean(velnew),sem(velnew))
plt.axhline(y = 0. ,color='black', linestyle='-',zorder=7)
plt.axhline(y=np.mean(Qnew), color='black', linestyle='--',zorder=7,label = 'True heat flow')
plt.scatter(np.linspace(10,N,N-10+1),Qaveragenew1, color = 'blue', s = 1,zorder=5)
plt.errorbar(np.linspace(10,N,N-10+1),Qaveragenew1, yerr = Qerrornew1, color = 'blue', fmt = '.',elinewidth=1,zorder=6,label = r'$\Delta t = 0.00002$s',alpha = 0.3)
plt.scatter(np.linspace(10,N,N-10+1),Qaveragenew2, color = 'red', s = 1,zorder=3)
plt.errorbar(np.linspace(10,N,N-10+1),Qaveragenew2, yerr = Qerrornew2, color = 'red', fmt = '.',elinewidth=1,zorder=4,label = r'$\Delta t = 0.0001$s',alpha = 0.3)
plt.scatter(np.linspace(10,N,N-10+1),Qaveragenew3, color = 'forestgreen', s = 1,zorder=2)
plt.errorbar(np.linspace(10,N,N-10+1),Qaveragenew3, yerr = Qerrornew3, color = 'forestgreen', fmt = '.',elinewidth=1,zorder=1,label = r'$\Delta t = 0.0002$s',alpha = 0.3)
plt.axhline(y=np.mean(Qnew), color='black', linestyle='--',zorder=7)
plt.xscale('log')
plt.legend(loc='center right')
plt.xlabel('Number of trajectories of length 1s')
plt.ylabel(r'$\dot{Q}_X (k_{B} T/s)$')
plt.savefig("Benchmark simulation ratchet.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.show()

