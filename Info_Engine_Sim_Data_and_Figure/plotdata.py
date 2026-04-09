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


width = 8.6
height = 5

fig=plt.figure(figsize=(width/2.54,2*height/2.54))

gs=gridspec.GridSpec(2,1)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])


N=10000

Q = np.load('Qnew.npy')
Qaverage1 = np.load('Qaveragenew1.npy')
Qerror1 = np.load('Qerrornew1.npy')
Qaverage2 = np.load('Qaveragenew2.npy')
Qerror2 = np.load('Qerrornew2.npy')
Qaverage3 = np.load('Qaveragenew3.npy')
Qerror3 = np.load('Qerrornew3.npy')

ax1.hlines(np.mean(Q),1,N,ls=':',color='black',label='True Heat')
#ax1.hlines(0,1,N,ls='-',color='black',lw=1)


ax1.errorbar(np.arange(10,N+1),Qaverage1,yerr=Qerror1,alpha=0.3,label=r'$\Delta t = 0.00002$s',elinewidth=1,ls='-')
ax1.errorbar(np.arange(10,N+1),Qaverage2,yerr=Qerror2,alpha=0.3,label=r'$\Delta t = 0.0001$s',elinewidth=1,ls='-')
ax1.errorbar(np.arange(10,N+1),Qaverage3,yerr=Qerror3,alpha=0.3,label=r'$\Delta t = 0.0002$s',elinewidth=1,ls='-')
    

ax1.set_ylabel(r'Heat Flow $\beta\dot{Q}_X$ (/s)',fontsize=10)
#ax1.set_xlabel(r'Number of Trajectories $n$',fontsize=10)
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_ylim(-450,0)
ax1.set_xlim(10,N)
ax1.legend(loc='upper right',frameon=False,fontsize=8)

Q = np.load('Q.npy')
Qaverage1 = np.load('Qaverage1.npy')
Qerror1 = np.load('Qerror1.npy')
Qaverage2 = np.load('Qaverage2.npy')
Qerror2 = np.load('Qerror2.npy')
Qaverage3 = np.load('Qaverage3.npy')
Qerror3 = np.load('Qerror3.npy')


ax2.hlines(np.mean(Q),1,N,ls=':',color='black',label='True Heat')
ax2.hlines(0,1,N,ls='-',color='black',lw=1)


ax2.errorbar(np.arange(10,N+1),Qaverage1,yerr=Qerror1,alpha=0.3,label=r'$\Delta t = 0.00005$s',elinewidth=1,ls='-')
ax2.errorbar(np.arange(10,N+1),Qaverage2,yerr=Qerror2,alpha=0.3,label=r'$\Delta t = 0.0005$s',elinewidth=1,ls='-')
ax2.errorbar(np.arange(10,N+1),Qaverage3,yerr=Qerror3,alpha=0.3,label=r'$\Delta t = 0.005$s',elinewidth=1,ls='-')
    

ax2.set_ylabel(r'Heat Flow $\beta\dot{Q}_X$ (/s)',fontsize=10)
ax2.set_xlabel(r'Number of Trajectories $n$',fontsize=10)
ax2.set_xscale('log')
#ax1.set_yscale('log')
ax2.set_ylim(-100,500)
ax2.set_xlim(10,N)
#ax2.legend(loc='upper right',frameon=False,fontsize=8)



fig.text(0.02, 0.97, r'$\mathbf{a)}$', ha='center', fontsize=14)
fig.text(0.02, 0.47, r'$\mathbf{b)}$', ha='center', fontsize=14)


plt.tight_layout(pad=0)
#plt.subplots_adjust(hspace=0.4)



plt.savefig('Figure_2_Draft')


plt.show()