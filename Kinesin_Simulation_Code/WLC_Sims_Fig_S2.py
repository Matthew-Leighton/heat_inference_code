"""Run the full-WLC kinesin simulations and reproduce the Figure S2 plot.
"""

from __future__ import division

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from numba import njit, prange


this_dir = Path(__file__).resolve().parent

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
style_path = this_dir / "Leighton_Style_Colors2"
if style_path.exists():
    plt.style.use(str(style_path))


##### Kinesin Model Parameters

kf0 = 1002.0  # /s
kb0 = 27.9  # /s
kc = 102.0  # /s
df = 3.61  # nm
db = 1.14  # nm
d = 8.0  # nm

D = 126680.0  # nm^2/s
F = 0.0  # pN
beta = 1.0 / 4.12  # 1/(pN nm)
kBT = 1.0 / beta  # pN nm
gamma = 1.0 / (beta * D)  # pN s/nm


##### Brown--Sivak WLC Parameters

lp = 0.1 * d  # persistence length = 0.8 nm
lc = 10.0 * d  # contour length = 80 nm
extension_limit = 0.98 * lc  # numerical guard below the WLC singularity


@njit(inline="always")
def WLC_force(extension):
    """Marko--Siggia force exerted by the linker on the cargo, in pN."""

    z = extension / lc
    return (kBT / lp) * (0.25 / (1.0 - z) ** 2 - 0.25 + z)


@njit(inline="always")
def WLC_potential(extension):
    """WLC potential U(extension), with U(0)=0, in pN nm."""

    z = extension / lc
    return (kBT / lp) * (
        0.25 * lc * (1.0 / (1.0 - z) - 1.0)
        - 0.25 * extension
        + 0.5 * extension**2 / lc
    )


@njit(inline="always")
def enforce_WLC_domain(x_proposed, y):
    """Reflect a rare Euler proposal that crosses the finite-extension wall."""

    extension = y - x_proposed
    if extension < extension_limit:
        return x_proposed, 0

    extension = 2.0 * extension_limit - extension
    if extension >= extension_limit:
        extension = extension_limit - 1e-9
    return y - extension, 1


##### Simulation


@njit(parallel=True, fastmath=True)
def simulate_driven(N, T, T_equil, dt, sample_strides, seed):
    """Simulate kinesin plus cargo with the full WLC linker."""

    steps = int(round(T / dt))
    equil_steps = int(round(T_equil / dt))
    noise_SD = np.sqrt(2.0 * D * dt)
    n_dt = len(sample_strides)

    Heat_true_list = np.zeros(N)
    MSD_mean = np.zeros((n_dt, N))
    vel_true_list = np.zeros(N)
    boundary_hits = np.zeros(N, dtype=np.int64)

    for j in prange(N):
        np.random.seed(seed + j)
        x = 0.0
        y = 0.0
        state = 0
        hits = 0

        ##### Equilibrate the driven system

        for _ in range(equil_steps):
            extension = y - x
            x_new = x + (F + WLC_force(extension)) / gamma * dt
            x_new += noise_SD * np.random.normal()
            x, hit = enforce_WLC_domain(x_new, y)
            hits += hit

            F_m = -WLC_force(y - x)
            kf = kf0 * np.exp(beta * df * F_m)
            kb = kb0 * np.exp(beta * db * F_m)

            if state == 0:
                if np.random.random() < 1.0 - np.exp(-kc * dt):
                    state = 1
            else:
                total_rate = kf + kb
                if np.random.random() < 1.0 - np.exp(-total_rate * dt):
                    if np.random.random() < kf / total_rate:
                        y += d
                    else:
                        y -= d
                    state = 0

        ##### Main trajectory

        x_start = x
        x_sample_old = np.full(n_dt, x)
        Dx2_sum = np.zeros(n_dt)
        sample_count = np.zeros(n_dt, dtype=np.int64)
        Q = 0.0

        for i in range(steps):
            extension = y - x
            E0 = WLC_potential(extension) - F * x

            x_new = x + (F + WLC_force(extension)) / gamma * dt
            x_new += noise_SD * np.random.normal()
            x, hit = enforce_WLC_domain(x_new, y)
            hits += hit

            extension = y - x
            E1 = WLC_potential(extension) - F * x
            Q += E1 - E0

            for m in range(n_dt):
                if (i + 1) % sample_strides[m] == 0:
                    Dx = x - x_sample_old[m]
                    Dx2_sum[m] += Dx * Dx
                    sample_count[m] += 1
                    x_sample_old[m] = x

            F_m = -WLC_force(extension)
            kf = kf0 * np.exp(beta * df * F_m)
            kb = kb0 * np.exp(beta * db * F_m)

            if state == 0:
                if np.random.random() < 1.0 - np.exp(-kc * dt):
                    state = 1
            else:
                total_rate = kf + kb
                if np.random.random() < 1.0 - np.exp(-total_rate * dt):
                    if np.random.random() < kf / total_rate:
                        y += d
                    else:
                        y -= d
                    state = 0

        Heat_true_list[j] = beta * Q / T
        vel_true_list[j] = (x - x_start) / T
        for m in range(n_dt):
            MSD_mean[m, j] = Dx2_sum[m] / sample_count[m]
        boundary_hits[j] = hits

    return Heat_true_list, MSD_mean, vel_true_list, boundary_hits


@njit(parallel=True, fastmath=True)
def simulate_equilibrium(N, T, T_equil, dt, sample_strides, seed):
    """Simulate the equilibrium WLC reference with the motor held fixed."""

    steps = int(round(T / dt))
    equil_steps = int(round(T_equil / dt))
    noise_SD = np.sqrt(2.0 * D * dt)
    n_dt = len(sample_strides)
    MSD_eq_mean = np.zeros((n_dt, N))
    boundary_hits = np.zeros(N, dtype=np.int64)

    for j in prange(N):
        np.random.seed(seed + j)
        x = 0.0
        y = 0.0
        hits = 0

        for _ in range(equil_steps):
            extension = y - x
            x_new = x + (F + WLC_force(extension)) / gamma * dt
            x_new += noise_SD * np.random.normal()
            x, hit = enforce_WLC_domain(x_new, y)
            hits += hit

        x_sample_old = np.full(n_dt, x)
        Dx2_sum = np.zeros(n_dt)
        sample_count = np.zeros(n_dt, dtype=np.int64)

        for i in range(steps):
            extension = y - x
            x_new = x + (F + WLC_force(extension)) / gamma * dt
            x_new += noise_SD * np.random.normal()
            x, hit = enforce_WLC_domain(x_new, y)
            hits += hit

            for m in range(n_dt):
                if (i + 1) % sample_strides[m] == 0:
                    Dx = x - x_sample_old[m]
                    Dx2_sum[m] += Dx * Dx
                    sample_count[m] += 1
                    x_sample_old[m] = x

        for m in range(n_dt):
            MSD_eq_mean[m, j] = Dx2_sum[m] / sample_count[m]
        boundary_hits[j] = hits

    return MSD_eq_mean, boundary_hits


##### Analysis


def cumulative_mean_and_variance(values):
    """Return cumulative means and variances of those means."""

    values = np.asarray(values, dtype=float)
    n = np.arange(1, len(values) + 1, dtype=float)
    cumulative_sum = np.cumsum(values)
    cumulative_sum2 = np.cumsum(values**2)
    mean = cumulative_sum / n

    mean_variance = np.full(len(values), np.nan)
    sample_variance = (
        cumulative_sum2[1:] - cumulative_sum[1:] ** 2 / n[1:]
    ) / (n[1:] - 1.0)
    mean_variance[1:] = np.maximum(sample_variance, 0.0) / n[1:]
    return mean, mean_variance


def cumulative_heat_estimator(MSD_driven, MSD_eq, delta_t):
    """Apply the MSD heat estimator, including uncertainty in both MSDs."""

    MSD_driven_mean, MSD_driven_var = cumulative_mean_and_variance(MSD_driven)
    MSD_eq_mean, MSD_eq_var = cumulative_mean_and_variance(MSD_eq)

    prefactor = 2.0 / delta_t
    estimate = prefactor * (1.0 - MSD_driven_mean / MSD_eq_mean)
    estimate_variance = prefactor**2 * (
        MSD_driven_var / MSD_eq_mean**2
        + MSD_driven_mean**2 * MSD_eq_var / MSD_eq_mean**4
    )
    return estimate, np.sqrt(estimate_variance)


def save_data(data_dir, heat, MSD_driven, MSD_eq, dt_sim, dt_samples):
    """Save simple CSV files analogous to those used by the original code."""

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_dir / "Heat_true_list_WLC.csv", heat)

    suffixes = ["dt1", "dt10", "dt100"]
    for m, suffix in enumerate(suffixes):
        np.savetxt(data_dir / f"MSD_mean_{suffix}_WLC.csv", MSD_driven[m])
        np.savetxt(data_dir / f"MSDeq_mean_{suffix}_WLC.csv", MSD_eq[m])

    with (data_dir / "WLC_parameters.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "value", "units"])
        writer.writerow(["persistence_length", lp, "nm"])
        writer.writerow(["contour_length", lc, "nm"])
        writer.writerow(["kBT", kBT, "pN nm"])
        writer.writerow(["integration_timestep", dt_sim, "s"])
        for m, delta_t in enumerate(dt_samples):
            writer.writerow([f"sampling_interval_{m + 1}", delta_t, "s"])


def make_figure(output_pdf, output_png, heat, estimates, estimate_errors):
    """Make the full-WLC equivalent of the original Figure S2."""

    N = estimates.shape[1]
    n_list = np.arange(1, N + 1)

    width = 8.6
    height = 5.0
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))

    plt.hlines(
        np.mean(heat), 1, N, ls=":", color="black", label="True Heat"
    )
    plt.hlines(0, 1, N, ls="-", color="black", lw=1)

    labels = [
        r"$\Delta t = 0.00005$s",
        r"$\Delta t = 0.0005$s",
        r"$\Delta t = 0.005$s",
    ]
    for m in range(3):
        plt.errorbar(
            n_list,
            estimates[m],
            yerr=estimate_errors[m],
            alpha=0.3,
            label=labels[m],
            elinewidth=1,
            ls="-",
        )

    plt.ylabel(r"Heat Flow $\beta\dot{Q}_X$ (/s)", fontsize=10)
    plt.xlabel(r"Number $n$ of Trajectories", fontsize=10)
    plt.xscale("log")
    plt.ylim(-250, 250)
    plt.xlim(min(10, N), N)
    plt.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout(pad=0)

    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=400)
    plt.close(fig)


##### Run simulations, analyze, and plot


def run(args):
    dt_samples = np.asarray(args.dt_samples, dtype=float)
    sample_strides = np.rint(dt_samples / args.dt_sim).astype(np.int64)
    if np.any(sample_strides < 1):
        raise ValueError("Sampling intervals must be at least dt-sim")
    if not np.allclose(sample_strides * args.dt_sim, dt_samples):
        raise ValueError("Each sampling interval must be a multiple of dt-sim")

    print(f"Simulating {args.N} driven WLC trajectories...", flush=True)
    heat, MSD_driven, velocity, driven_hits = simulate_driven(
        args.N,
        args.T,
        args.T_equil,
        args.dt_sim,
        sample_strides,
        args.seed,
    )

    print(f"Simulating {args.N} equilibrium WLC trajectories...", flush=True)
    MSD_eq, equilibrium_hits = simulate_equilibrium(
        args.N,
        args.T,
        args.T_equil,
        args.dt_sim,
        sample_strides,
        args.seed + 500000,
    )

    estimates = np.zeros_like(MSD_driven)
    estimate_errors = np.zeros_like(MSD_driven)
    for m, delta_t in enumerate(dt_samples):
        estimates[m], estimate_errors[m] = cumulative_heat_estimator(
            MSD_driven[m], MSD_eq[m], delta_t
        )
        print(
            f"dt={delta_t:g} s: true heat={np.mean(heat):.3f}; "
            f"estimate={estimates[m, -1]:.3f} +/- "
            f"{estimate_errors[m, -1]:.3f}",
            flush=True,
        )

    total_hits = int(np.sum(driven_hits) + np.sum(equilibrium_hits))
    print(f"WLC boundary reflections: {total_hits}", flush=True)
    print(f"Mean velocity: {np.mean(velocity):.3f} nm/s", flush=True)

    data_dir = Path(args.data_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    save_data(data_dir, heat, MSD_driven, MSD_eq, args.dt_sim, dt_samples)
    make_figure(output_pdf, output_png, heat, estimates, estimate_errors)
    print(f"Saved {output_pdf}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--T-equil", type=float, default=2.0)
    parser.add_argument("--dt-sim", type=float, default=1e-5)
    parser.add_argument(
        "--dt-samples",
        type=float,
        nargs=3,
        default=[5e-5, 5e-4, 5e-3],
    )
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument(
        "--data-dir", default=str(this_dir / "WLC_Fig_S2_Data")
    )
    parser.add_argument(
        "--output-pdf", default=str(this_dir / "Figure_S2_WLC.pdf")
    )
    parser.add_argument(
        "--output-png", default=str(this_dir / "Figure_S2_WLC.png")
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
