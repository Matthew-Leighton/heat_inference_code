from __future__ import division

import argparse
from pathlib import Path

import numpy as np
from numba import njit, prange


##### Kinesin Model Parameters

kf0 = 1002.0  # /s
kb0 = 27.9  # /s
kc = 102.0  # /s
df = 3.61  # nm
db = 1.14  # nm

d = 8.0  # nm

D = 126680.0  # nm^2/s
kappa = 0.116  # pN/nm
beta = 1.0 / 4.12  # (1/pNnm)
gamma = 1.0 / (beta * D)  # pN s/nm

tau = gamma / kappa


##### Simulation Parameters

# Use a 10 us integration step, then sample the cargo
# displacement every 50 us for the heat estimator.
dt_sim = 1e-5  # s, integration timestep used in the paper
dt = 5e-5  # s, sampling timestep for the heat estimator
sample_stride = int(round(dt / dt_sim))

T = 1.0  # s
T_equil = 2.0  # s
N = 4000

DN = 10
Dneq_list = np.logspace(-2, 1, num=DN) * D
f_list = np.array([0.0, -2.0, -4.0])


@njit(parallel=True, fastmath=True)
def sim_condition(F, Dneq, N, T, T_equil, seed):
    ##### Simulate one force/noise condition

    steps = int(round(T / dt_sim))
    equil_steps = int(round(T_equil / dt_sim))

    # Nonequilibrium noise is included as an effective cargo temperature.
    beta_eff = beta / (1.0 + Dneq / D)
    sigma = np.sqrt(1.0 / (beta_eff * kappa))
    exp_dt_tau = np.exp(-dt_sim / tau)
    exp_2dt_tau = np.exp(-2.0 * dt_sim / tau)
    Dx_SD = np.sqrt(sigma**2 * (1.0 - exp_2dt_tau))

    MSD_mean_dt1 = np.zeros(N)
    MSD_std_dt1 = np.zeros(N)
    Heat_true_list = np.zeros(N)
    vel_true_list = np.zeros(N)

    for j in prange(N):
        ##### Initial condition

        np.random.seed(seed + j)

        x = 0.0
        y = 0.0
        state = 0

        ##### Equilibrate before collecting statistics

        for _ in range(equil_steps):
            Dx_mean = (y - x + F / kappa) * (1.0 - exp_dt_tau)
            x = x + np.random.normal(Dx_mean, Dx_SD)

            F_m = kappa * (x - y)
            kf = kf0 * np.exp(beta * df * F_m)
            kb = kb0 * np.exp(beta * db * F_m)
            r = np.random.random()

            if state == 0:
                if r < kc * dt_sim:
                    state = 1
            elif state == 1:
                if r < kf * dt_sim:
                    y = y + d
                    state = 0
                elif r < (kf + kb) * dt_sim:
                    y = y - d
                    state = 0

        x_sample_old = x
        Q = 0.0
        Dx_sum = 0.0
        Dx2_sum = 0.0
        Dx4_sum = 0.0
        n_sample = 0

        ##### Main trajectory

        for i in range(steps):
            # True heat from the cargo energy change during the cargo update.
            E0 = 0.5 * kappa * (x - y) ** 2 - F * x

            Dx_mean = (y - x + F / kappa) * (1.0 - exp_dt_tau)
            x = x + np.random.normal(Dx_mean, Dx_SD)

            E1 = 0.5 * kappa * (x - y) ** 2 - F * x
            Q = Q + E1 - E0

            if (i + 1) % sample_stride == 0:
                Dx = x - x_sample_old
                Dx2 = Dx**2
                Dx_sum = Dx_sum + Dx
                Dx2_sum = Dx2_sum + Dx2
                Dx4_sum = Dx4_sum + Dx2**2
                n_sample = n_sample + 1
                x_sample_old = x

            # Two-state kinesin stepping model.
            F_m = kappa * (x - y)
            kf = kf0 * np.exp(beta * df * F_m)
            kb = kb0 * np.exp(beta * db * F_m)
            r = np.random.random()

            if state == 0:
                if r < kc * dt_sim:
                    state = 1
            elif state == 1:
                if r < kf * dt_sim:
                    y = y + d
                    state = 0
                elif r < (kf + kb) * dt_sim:
                    y = y - d
                    state = 0

        MSD_mean_dt1[j] = Dx2_sum / n_sample
        MSD_var_dt1 = Dx4_sum / n_sample - MSD_mean_dt1[j] ** 2
        if MSD_var_dt1 < 0.0:
            MSD_var_dt1 = 0.0
        MSD_std_dt1[j] = np.sqrt(MSD_var_dt1)
        Heat_true_list[j] = beta * Q / T
        vel_true_list[j] = (Dx_sum / n_sample) / dt

    return Heat_true_list, vel_true_list, MSD_mean_dt1, MSD_std_dt1


def main(N=N, T=T, T_equil=T_equil, out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ##### Loop over forces and nonequilibrium noise strengths

    for k in range(len(f_list)):
        F = f_list[k]
        print(k)

        for i in range(len(Dneq_list)):
            Dneq = Dneq_list[i]
            print(i)

            seed = 1729 + 100000 * k + 1000 * i
            Heat_true_list, vel_true_list, MSD_mean_dt1, MSD_std_dt1 = sim_condition(
                F, Dneq, N, T, T_equil, seed
            )

            np.savetxt(out_dir / ("Heat_true_list_f_" + str(k) + "_Dneq_" + str(i) + ".csv"), Heat_true_list)
            np.savetxt(out_dir / ("vel_true_list_f_" + str(k) + "_Dneq_" + str(i) + ".csv"), vel_true_list)
            np.savetxt(out_dir / ("MSD_mean_dt1_f_" + str(k) + "_Dneq_" + str(i) + ".csv"), MSD_mean_dt1)
            np.savetxt(out_dir / ("MSD_std_dt1_f_" + str(k) + "_Dneq_" + str(i) + ".csv"), MSD_std_dt1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--T", type=float, default=T)
    parser.add_argument("--T-equil", type=float, default=T_equil)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    main(N=args.N, T=args.T, T_equil=args.T_equil, out_dir=args.out_dir)
