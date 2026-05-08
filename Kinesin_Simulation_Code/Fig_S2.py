#Load all the packages I might need
from __future__ import division # must be first
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import matplotlib
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


plt.style.use('Leighton_Style_Colors2')


##### Kinesin Model Parameters

kf0 = 1002 #/s
kb0 = 27.9 #/s
kc = 102 #/s
df = 3.61 #nm
db = 1.14 #nm

d = 8 #nm

D = 126680 #nm^2/s
F = 0 # pN
kappa = 0.116 #pN/nm
beta = 1/4.12 #(1/pNnm)
gamma = 3.25 * 10**(-5) #pN s/nm

Dmu = 15

tau_r = gamma/kappa
sigma = np.sqrt(1/(beta*kappa))


def MSDeq_analytic(delta_t,D,kappa):
    gamma = 1/(beta*D)
    tau = gamma/kappa
    
    return 2*sigma**2 * (1-np.exp(-delta_t/tau))


N = 10000

Heat_true_list = np.loadtxt('Heat_true_list.csv')
MSD_mean_dt1 = np.loadtxt('MSD_mean_dt1.csv')
MSD_std_dt1 = np.loadtxt('MSD_std_dt1.csv')
MSD_mean_dt10 = np.loadtxt('MSD_mean_dt10.csv')
MSD_std_dt10 = np.loadtxt('MSD_std_dt10.csv')
MSD_mean_dt100 = np.loadtxt('MSD_mean_dt100.csv')
MSD_std_dt100 = np.loadtxt('MSD_std_dt100.csv')

Heat_est_indirect_list_dt1 = np.zeros(N)
est_indirect_err_list_dt1 = np.zeros(N)
Heat_est_indirect_list_dt10 = np.zeros(N)
est_indirect_err_list_dt10 = np.zeros(N)
Heat_est_indirect_list_dt100 = np.zeros(N)
est_indirect_err_list_dt100 = np.zeros(N)


dt = 1/20000
MSDeq = MSDeq_analytic(dt, D, kappa)

for j in range(N):
    Heat_est_indirect_list_dt1[j] = (2/dt) * (1 - np.mean(MSD_mean_dt1[:j+1])/MSDeq)
    est_indirect_err_list_dt1[j] = ((2/dt) * np.std(MSD_mean_dt1[:j+1])/MSDeq)/np.sqrt(j+1)
    
print(((Heat_est_indirect_list_dt1[-1] - Heat_true_list[-1])/Heat_true_list[-1])/(dt / (2*tau_r)))

dt = dt*10
MSDeq = MSDeq_analytic(dt, D, kappa)

for j in range(N):
    Heat_est_indirect_list_dt10[j] = (2/dt) * (1 - np.mean(MSD_mean_dt10[:j+1])/MSDeq)
    est_indirect_err_list_dt10[j] = ((2/dt) * np.std(MSD_mean_dt10[:j+1])/MSDeq)/np.sqrt(j+1)
    
print(((Heat_est_indirect_list_dt10[-1] - Heat_true_list[-1])/Heat_true_list[-1])/(dt / (2*tau_r)))
    
dt = dt*10
MSDeq = MSDeq_analytic(dt, D, kappa)

for j in range(N):
    Heat_est_indirect_list_dt100[j] = (2/dt) * (1 - np.mean(MSD_mean_dt100[:j+1])/MSDeq)
    est_indirect_err_list_dt100[j] = ((2/dt) * np.std(MSD_mean_dt100[:j+1])/MSDeq)/np.sqrt(j+1)


print(((Heat_est_indirect_list_dt100[-1] - Heat_true_list[-1])/Heat_true_list[-1])/(dt / (2*tau_r)))



width = 8.6
height = 5

fig=plt.figure(figsize=(width/2.54,height/2.54))

plt.hlines(Heat_true_list[-1],1,N,ls=':',color='black',label='True Heat')
plt.hlines(0,1,N,ls='-',color='black',lw=1)


plt.errorbar(np.arange(1,N+1),Heat_est_indirect_list_dt1,yerr=est_indirect_err_list_dt1,alpha=0.3,label=r'$\Delta t = 0.00005$s',elinewidth=1,ls='-')
plt.errorbar(np.arange(1,N+1),Heat_est_indirect_list_dt10,yerr=est_indirect_err_list_dt10,alpha=0.3,label=r'$\Delta t = 0.0005$s',elinewidth=1,ls='-')
plt.errorbar(np.arange(1,N+1),Heat_est_indirect_list_dt100,yerr=est_indirect_err_list_dt100,alpha=0.3,label=r'$\Delta t = 0.005$s',elinewidth=1,ls='-')
    

plt.ylabel(r'Heat Flow $\beta\dot{Q}_X$ (/s)',fontsize=10)
plt.xlabel(r'Number $n$ of Trajectories',fontsize=10)
plt.xscale('log')
#ax1.set_yscale('log')
plt.ylim(-250,250)
plt.xlim(10,N)
plt.legend(loc='upper right',frameon=False,fontsize=8)

plt.tight_layout(pad=0)


plt.savefig('Figure_S2')


plt.show()