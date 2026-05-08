from __future__ import division

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


this_dir = Path(__file__).resolve().parent
repo_dir = this_dir.parent

style_path = repo_dir / "Leighton_Style_Colors2"
if style_path.exists():
    plt.style.use(str(style_path))

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})


##### Kinesin Model Parameters

D = 126680.0  # nm^2/s
kappa = 0.116  # pN/nm
beta = 1.0 / 4.12  # (1/pNnm)
gamma = 1.0 / (beta * D)  # pN s/nm


##### Experimental Data From Ariga et al.

Expt_noise_SD_F2 = np.array(
    [0, 0.20723684210526327, 0.3991228070175441, 0.7006578947368425, 0.9747807017543861, 1.2580409356725148, 1.4225146198830414]
)
Expt_v_mean_F2 = np.array(
    [405.9027777777777, 419.79166666666663, 376.38888888888886, 397.22222222222223, 405.9027777777777, 452.77777777777777, 496.1805555555555]
)

Expt_noise_SD_F4 = np.array(
    [0, 0.3991228070175441, 0.8011695906432754, 1.4133771929824568, 1.9433479532163744, 2.528143274853802, 2.8662280701754392]
)
Expt_v_mean_F4 = np.array(
    [100.34722222222206, 88.19444444444434, 98.61111111111109, 102.08333333333326, 112.5, 157.6388888888888, 161.1111111111111]
)

gamma_expt = gamma / 10.0
Expt_Dneq_F2 = (1.0 / 20000.0) / 2.0 * (kappa * (6.85 / 11.6) * Expt_noise_SD_F2 / gamma_expt) ** 2
Expt_Dneq_F4 = (1.0 / 20000.0) / 2.0 * (kappa * Expt_noise_SD_F4 / gamma_expt) ** 2


##### Figure

# Three stacked panels: schematic, velocity, and heat estimator.
width = 8.6
height = 5

fig = plt.figure(figsize=(width / 2.54, 3 * height / 2.54))

gs = gridspec.GridSpec(3, 1)
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])
ax4 = plt.subplot(gs[2])

##### Panel a: schematic

ax1.set_axis_off()

im = image.imread(repo_dir / "Diagram.png")
h, w, _ = im.shape if len(im.shape) == 3 else (*im.shape, 1)
image_aspect = w / h

width_display = 2
height_display = width_display / image_aspect
x_center, y_center = 0.35, 0.45
extent = [
    x_center - width_display / 2,
    x_center + width_display / 2,
    y_center - height_display / 2,
    y_center + height_display / 2,
]

ax1.imshow(im, extent=extent, aspect="equal", zorder=1, clip_on=False)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

DN = 10
Dneq_list = np.logspace(-2, 1, num=DN) * D
Dneq_ratio_list = Dneq_list / D

color_list = ["red", "purple", "blue"]
F_label_list = [0, 2, 4]

##### Panel b: velocity vs nonequilibrium noise

for i in range(len(F_label_list)):
    v_list = np.loadtxt(this_dir / ("v_Dneq_F" + str(F_label_list[i]) + ".csv"))
    ax3.plot(Dneq_ratio_list, v_list, marker="D", alpha=0.5, ls="none", color=color_list[i])

ax3.scatter(Expt_Dneq_F2 / D, Expt_v_mean_F2, color="purple", marker="s")
ax3.scatter(Expt_Dneq_F4 / D, Expt_v_mean_F4, color="blue", marker="s")

ax3.scatter(0, 10000, marker="s", color="grey", alpha=1, label="Experiment [Ariga et al., PRL, 2021]", s=70)
ax3.scatter(0, 10000, marker="D", color="grey", alpha=0.5, label="Simulations", s=70)

ax3.set_ylabel(r"Velocity $\langle v\rangle$ (nm/s)", fontsize=10)
ax3.set_xscale("log")
ax3.legend(loc="upper left", fontsize=8, frameon=False)
ax3.set_ylim(0, 1100)
ax3.set_xlim(min(Dneq_ratio_list), max(Dneq_ratio_list))

##### Panel c: true heat and estimator

ax4.scatter(0, 10000, marker="D", color="grey", alpha=0.5, label="True Heat", s=70)
ax4.errorbar(0, 10000, yerr=1, marker="o", color="grey", alpha=0.5, label="Estimator", ls="-")
ax4.legend(loc="upper left", frameon=False)

lw_list = [3.5, 2.5, 1.5]

for i in range(len(F_label_list)):
    Q_true = np.loadtxt(this_dir / ("Q_true_f_" + str(i) + ".csv"))
    est_mean = np.loadtxt(this_dir / ("est_mean_f_" + str(i) + ".csv"))
    est_err = np.loadtxt(this_dir / ("est_err_f_" + str(i) + ".csv"))

    ax4.errorbar(Dneq_ratio_list, est_mean, yerr=est_err, marker="o", color=color_list[i], alpha=0.5, ls="none", elinewidth=lw_list[i])
    ax4.scatter(Dneq_ratio_list, Q_true, marker="D", alpha=0.5, s=70, color=color_list[i])

ax4.set_xlabel(r"Nonequilibrium Noise Strength $D_\mathrm{neq}/D_\mathrm{eq}$", fontsize=10)
ax4.set_ylabel(r"Heat Flow $\beta\dot{Q}_X$ (/s)", fontsize=10)
ax4.set_xscale("log")
ax4.set_yscale("symlog", linthresh=1)
ax4.set_xlim(min(Dneq_ratio_list), max(Dneq_ratio_list))
ax4.hlines(0, min(Dneq_ratio_list), max(Dneq_ratio_list), ls="-", color="black", lw=1)
ax4.set_ylim(-100, 1000)

fig.text(0.02, 0.97, r"$\mathbf{a)}$", ha="center", fontsize=14)
fig.text(0.02, 0.67, r"$\mathbf{b)}$", ha="center", fontsize=14)
fig.text(0.02, 0.34, r"$\mathbf{c)}$", ha="center", fontsize=14)

##### Save

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.25)

plt.savefig(this_dir / "Figure_3_250317.pdf", dpi=300)
plt.savefig(this_dir / "Figure_3_250317.png", dpi=300)
plt.show()
