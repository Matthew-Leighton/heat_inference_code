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

#factor1 and sqrtfactor are constants that involve the exponentials that properly 
# time evolve x. I calculate them as constants once off so the exponentials don't
# have to be calculated each time. probably doesn't matter much though
def update_delta_x(x,l,klink,gamma, grav, dt, std):
    return np.random.normal((1 - np.exp(-dt/(gamma/klink)))*(l - x - grav/klink),std)


deq = 16.*(82e-9)**2
gamma = (298*1.38e-23)/deq
klink = 2*math.pi*170.*gamma
grav = (1000*(4/3)*math.pi*(1.5e-6)**3)*9.8
sigma = np.sqrt((298*1.38e-23)/klink)
dt = 0.00002
dt_sample1 = 0.00002
dt_sample2 = 5*dt_sample1
dt_sample3 = 10*dt_sample1
T = 1.
alpha = 1.8
std = np.sqrt((sigma**2)*(1 - np.exp(-2*dt/(gamma/klink))))

N = 10000
Q = np.zeros(N)
vel = np.zeros(N)
dx21 = []
Qaverage1 = []
Qerror1 = []
dx22 = []
Qaverage2 = []
Qerror2 = []
dx23 = []
Qaverage3 = []
Qerror3 = []

#equilibrate
xold = 0.
lold = 0.
cond = 0.
for i in range(int(T/dt)-1):
    xnew = xold + update_delta_x(xold,lold,klink,gamma,grav,dt,std)
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

for k in range(N):
    dx21.append(0.)
    dx22.append(0.)
    dx23.append(0.)
    x = np.zeros(int(T/dt))
    x[0] = xold

    linit = lold
    #actual trajectory
    for i in range(int(T/dt)-1):
        x[i+1] = x[i] + update_delta_x(x[i],lold,klink,gamma,grav,dt,std)
        if cond == 1.:
            lnew = lold + alpha*(x[i] - lold)
        else:
            lnew = lold
        if x[i+1] > lnew:
            cond = 1.
        else:
            cond = 0.
        dx21[k] = dx21[k] + (x[i+1] - x[i])**2
        Q[k] = Q[k] + (klink/2)*(x[i+1] - lold)**2 + grav*x[i+1] - ((klink/2)*(x[i] - lold)**2 + grav*x[i])
        lold = lnew
    xold = x[-1]
    lend = lold
    vel[k] = (lend - linit)/T
    Q[k] = Q[k]/((298*1.38e-23)*T)
    dx21[k] = dx21[k]/(T/dt - 1)
    x2 = x[::5]
    x3 = x[::10]
    for j in range(len(x2)-1):
        dx22[k] = dx22[k] + (x2[j+1]-x2[j])**2
    for j in range(len(x3)-1):
        dx23[k] = dx23[k] + (x3[j+1]-x3[j])**2
    dx22[k] = dx22[k]/(T/dt_sample2 - 1)
    dx23[k] = dx23[k]/(T/dt_sample3 - 1)
    if k >= 9:
        Qaverage1.append((2/dt)*(1 - np.mean(dx21)/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))))
        Qerror1.append(sem((2/dt)*(1 - dx21/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink)))))))
        Qaverage2.append((2/dt_sample2)*(1 - np.mean(dx22)/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink))))))
        Qerror2.append(sem((2/dt_sample2)*(1 - dx22/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink)))))))
        Qaverage3.append((2/dt_sample3)*(1 - np.mean(dx23)/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink))))))
        Qerror3.append(sem((2/dt_sample3)*(1 - dx23/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink)))))))



print(np.mean(dx21),sem(dx21),(2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))
print(np.mean(Q),sem(Q))
print(np.mean(vel),sem(vel))
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
for i in range(int(T/dt)-1):
    xnew = xold + update_delta_x(xold,lold,klink,gamma,grav,dt,std)
    lnew = lold + np.mean(vel)*dt
    xold = xnew
    lold = lnew

for k in range(N):
    dx21.append(0.)
    dx22.append(0.)
    dx23.append(0.)
    x = np.zeros(int(T/dt))
    x[0] = xold

    linit = lold
    #actual trajectory
    for i in range(int(T/dt)-1):
        x[i+1] = x[i] + update_delta_x(x[i],lold,klink,gamma,grav,dt,std)
        lnew = lold + np.mean(vel)*dt
        dx21[k] = dx21[k] + (x[i+1] - x[i])**2
        Qnew[k] = Qnew[k] + (klink/2)*(x[i+1] - lold)**2 + grav*x[i+1] - ((klink/2)*(x[i] - lold)**2 + grav*x[i])
        lold = lnew
    xold = x[-1]
    lend = lold
    velnew[k] = (lend - linit)/T
    Qnew[k] = Qnew[k]/((298*1.38e-23)*T)
    dx21[k] = dx21[k]/(T/dt - 1)
    x2 = x[::5]
    x3 = x[::10]
    for j in range(len(x2)-1):
        dx22[k] = dx22[k] + (x2[j+1]-x2[j])**2
    for j in range(len(x3)-1):
        dx23[k] = dx23[k] + (x3[j+1]-x3[j])**2
    dx22[k] = dx22[k]/(T/dt_sample2 - 1)
    dx23[k] = dx23[k]/(T/dt_sample3 - 1)
    if k >= 9:
        Qaveragenew1.append((2/dt)*(1 - np.mean(dx21)/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink))))))
        Qerrornew1.append(sem((2/dt)*(1 - dx21/((2*sigma**2)*(1 - np.exp(-dt/(gamma/klink)))))))
        Qaveragenew2.append((2/dt_sample2)*(1 - np.mean(dx22)/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink))))))
        Qerrornew2.append(sem((2/dt_sample2)*(1 - dx22/((2*sigma**2)*(1 - np.exp(-dt_sample2/(gamma/klink)))))))
        Qaveragenew3.append((2/dt_sample3)*(1 - np.mean(dx23)/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink))))))
        Qerrornew3.append(sem((2/dt_sample3)*(1 - dx23/((2*sigma**2)*(1 - np.exp(-dt_sample3/(gamma/klink)))))))



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

