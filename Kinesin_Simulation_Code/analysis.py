from __future__ import division

from pathlib import Path

import numpy as np


##### Kinesin Model Parameters

D = 126680.0  # nm^2/s
kappa = 0.116  # pN/nm
beta = 1.0 / 4.12  # (1/pNnm)
gamma = 1.0 / (beta * D)  # pN s/nm

tau = gamma / kappa


##### Analysis Parameters

dt = 5e-5

DN = 10
Dneq_list = np.logspace(-2, 1, num=DN) * D
F_label_list = np.array([0, 2, 4])


def MSDeq_analytic(delta_t, D, Dneq, kappa):
    # Equilibrium baseline with the same nonequilibrium noise strength.
    beta_eff = beta / (1.0 + Dneq / D)
    sigma = np.sqrt(1.0 / (beta_eff * kappa))

    return 2.0 * sigma**2 * (1.0 - np.exp(-delta_t / tau))


def main(data_dir="."):
    data_dir = Path(data_dir)

    ##### Convert per-trajectory CSVs into the curves used in Fig. 3

    for i in range(len(F_label_list)):
        Q_true = np.zeros(DN)
        est_mean = np.zeros(DN)
        est_err = np.zeros(DN)
        v_list = np.zeros(DN)

        for j in range(len(Dneq_list)):
            Heat_True_list = np.atleast_1d(
                np.loadtxt(data_dir / ("Heat_true_list_f_" + str(i) + "_Dneq_" + str(j) + ".csv"))
            )
            vel_true_list = np.atleast_1d(
                np.loadtxt(data_dir / ("vel_true_list_f_" + str(i) + "_Dneq_" + str(j) + ".csv"))
            )
            MSD_Mean_list = np.atleast_1d(
                np.loadtxt(data_dir / ("MSD_mean_dt1_f_" + str(i) + "_Dneq_" + str(j) + ".csv"))
            )

            n = len(MSD_Mean_list)
            MSDeq = MSDeq_analytic(dt, D, Dneq_list[j], kappa)
            prefactor = (2.0 / dt) * (1.0 + Dneq_list[j] / D)

            # True heat and velocity use the direct simulation averages.
            Q_true[j] = np.mean(Heat_True_list)
            v_list[j] = np.mean(vel_true_list)

            # Heat estimator from the ratio of nonequilibrium and baseline MSDs.
            est_mean[j] = prefactor * (1.0 - np.mean(MSD_Mean_list) / MSDeq)
            est_err[j] = prefactor * np.std(MSD_Mean_list, ddof=1) / (np.sqrt(n) * MSDeq)

        ##### Save the short summary files used by Fig3.py

        np.savetxt(data_dir / ("Q_true_f_" + str(i) + ".csv"), Q_true)
        np.savetxt(data_dir / ("est_mean_f_" + str(i) + ".csv"), est_mean)
        np.savetxt(data_dir / ("est_err_f_" + str(i) + ".csv"), est_err)
        np.savetxt(data_dir / ("v_Dneq_F" + str(F_label_list[i]) + ".csv"), v_list)
        np.savetxt(data_dir / ("Qdot_Dneq_F" + str(F_label_list[i]) + ".csv"), Q_true)


if __name__ == "__main__":
    main()
