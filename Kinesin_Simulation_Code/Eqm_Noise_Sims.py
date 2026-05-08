### Simulations for kinesin with only equilibrium noise, used to produce the data for Figure S2.

import numpy as np
import matplotlib.pyplot as plt
#from numba import jit

# Model Parameters
kf0 = 1002  # /s
kb0 = 27.9  # /s
kc = 102  # /s
df = 3.61  # nm
db = 1.14  # nm
d = 8  # nm
D = 126680  # nm^2/s
F = 0  # pN
kappa = 0.116  # pN/nm
beta = 1/4.12  # (1/pNnm)
gamma = 1/(beta*D)  # pN s/nm

tau = gamma/kappa
sigma = np.sqrt(1/(beta*kappa))

def sim_trajectory_neq(dt, T, x0, y0, state0, D, F, kappa):
    steps = int(T/dt)
    x_traj = np.zeros(steps + 1)
    y_traj = np.zeros(steps + 1)
    state_traj = np.zeros(steps + 1)
    
    x_traj[0] = x0
    y_traj[0] = y0
    state_traj[0] = state0
    
    Q = 0
    x = x0
    y = y0
    state = state0
    
    exp_dt_tau = np.exp(-dt/tau)
    exp_2dt_tau = np.exp(-2*dt/tau)
    sqrt_sigma2 = np.sqrt(sigma**2 * (1-exp_2dt_tau))
    
    rng = np.random.default_rng()  # Use numpy's random number generator
    
    for i in range(steps):
        E0 = 0.5 * kappa * (x-y)**2 - F*x
        
        # Cargo update
        Dx_mean = (y - x + F/kappa) * (1 - exp_dt_tau)
        x = x + rng.normal(loc=Dx_mean, scale=sqrt_sigma2)
        
        E1 = 0.5 * kappa * (x-y)**2 - F*x
        
        Q += E1 - E0
        
        x_traj[i+1] = x
        
        # Motor update
        F_m = kappa * (x-y)
        kf = kf0 * np.exp(beta*df*F_m)
        kb = kb0 * np.exp(beta*db*F_m)

        r = rng.random()
        
        if state == 0:
            if r < kc*dt:
                state = 1
        elif state == 1:
            if r < (kf)*dt:
                y = y + d
                state = 0
            elif r < (kf+kb)*dt:
                y = y-d
                state = 0
        
        y_traj[i+1] = y
        state_traj[i+1] = state
    
    return x_traj, y_traj, state_traj, beta*Q/T

def MSDeq_analytic(dt, D, kappa):
    return 2*sigma**2 * (1-np.exp(-dt/tau))

# Simulation parameters
dt = 1/100000  # s
T = 1  # s
N = 10000


MSD_mean_dt1 = np.zeros(N)
MSD_std_dt1 = np.zeros(N)
Heat_true_list = np.zeros(N)
MSD_mean_dt2 = np.zeros(N)
MSD_std_dt2 = np.zeros(N)
MSD_mean_dt4 = np.zeros(N)
MSD_std_dt4 = np.zeros(N)

for j in range(N):
    # Equilibrate
    x_traj, y_traj, state_traj, _ = sim_trajectory_neq(dt, 1, 0, 0, 0, D, F, kappa)
    
    # Simulate
    x_traj, y_traj, state_traj, Qdot = sim_trajectory_neq(dt, T, x_traj[-1], y_traj[-1], state_traj[-1], D, F, kappa)
    
    Dx = x_traj[1:] - x_traj[:-1]
    
    n_complete = len(Dx) // 5
    Dx = Dx[:n_complete*5].reshape(-1, 5).sum(axis=1)
        
    MSD_mean_dt1[j] = np.mean(Dx**2)
    MSD_std_dt1[j] = np.std(Dx**2)
    
    Heat_true_list[j] = Qdot
    
    n_complete = len(Dx) // 10
    Dx_coarse = Dx[:n_complete*10].reshape(-1, 10).sum(axis=1)
    MSD_mean_dt2[j] = np.mean(Dx_coarse**2)
    MSD_std_dt2[j] = np.std(Dx_coarse**2)
    
    n_complete = len(Dx_coarse) // 10
    Dx_coarser = Dx_coarse[:n_complete*10].reshape(-1, 10).sum(axis=1)
    MSD_mean_dt4[j] = np.mean(Dx_coarser**2)
    MSD_std_dt4[j] = np.std(Dx_coarser**2)
    
np.savetxt('Heat_true_list.csv',Heat_true_list)
np.savetxt('MSD_mean_dt1.csv',MSD_mean_dt1)
np.savetxt('MSD_std_dt1.csv',MSD_std_dt1)
np.savetxt('MSD_mean_dt10.csv',MSD_mean_dt2)
np.savetxt('MSD_std_dt10.csv',MSD_std_dt2)
np.savetxt('MSD_mean_dt100.csv',MSD_mean_dt4)
np.savetxt('MSD_std_dt100.csv',MSD_std_dt4)