"""Thermodynamic flows for kinesin versus nonequilibrium cargo noise.

This reproduces the zero-load, linear-linker model used for Fig. 3 and
computes three steady-state currents:

* chemical work into the motor, beta*Delta_mu times its net step rate;
* interaction-energy flow from motor to cargo, from linker-energy changes at
  motor jumps; and
* information flow from motor to cargo, from extension-resolved motor-jump
  currents and stationary conditional extension distributions.

This reproduces the plot shown in the inset of Figure 3.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


##### Kinesin model parameters 

kf0 = 1002.0  # /s
kb0 = 27.9  # /s
kc = 102.0  # /s
df = 3.61  # nm
db = 1.14  # nm
d = 8.0  # nm

D = 126680.0  # nm^2/s
kappa = 0.116  # pN/nm
kBT = 4.12  # pN nm
beta = 1.0 / kBT  # 1/(pN nm)
gamma = 1.0 / (beta * D)  # pN s/nm
tau = gamma / kappa

beta_delta_mu = 15.0


##### Default numerical parameters

dt_sim_default = 1e-5  # s
duration_default = 4.0  # s per trajectory
equil_duration_default = 2.0  # s
n_trajectories_default = 1000
noise_ratios_default = np.logspace(-2.0, 1.0, 10)

# The fine histogram is rebinned to test information-flow convergence.
hist_min_default = -120.0  # nm, extension e=x-y
hist_max_default = 120.0  # nm
hist_bin_width_default = 0.25  # nm
hist_sample_stride_default = 5  # sample p(e,state) every 50 us
reported_bin_width_default = 0.5  # nm
block_count_default = 20


@njit(parallel=True, fastmath=True)
def simulate_condition(
    noise_ratio,
    n_trajectories,
    duration,
    equil_duration,
    dt_sim,
    hist_min,
    hist_bin_width,
    n_bins,
    hist_sample_stride,
    seed,
):
    """Simulate one noise condition and collect trajectory and flux data."""

    steps = int(round(duration / dt_sim))
    equil_steps = int(round(equil_duration / dt_sim))

    beta_eff = beta / (1.0 + noise_ratio)
    sigma = np.sqrt(1.0 / (beta_eff * kappa))
    exp_dt_tau = np.exp(-dt_sim / tau)
    exp_2dt_tau = np.exp(-2.0 * dt_sim / tau)
    dx_sd = np.sqrt(sigma * sigma * (1.0 - exp_2dt_tau))

    cargo_heat_beta_per_s = np.zeros(n_trajectories)
    motor_energy_beta_per_s = np.zeros(n_trajectories)
    velocity_nm_per_s = np.zeros(n_trajectories)
    forward_rate_per_s = np.zeros(n_trajectories)
    backward_rate_per_s = np.zeros(n_trajectories)
    activation_rate_per_s = np.zeros(n_trajectories)

    # Per-trajectory counts permit blockwise uncertainty estimates for the
    # nonlinear plug-in estimator of information flow.
    occupancy_counts = np.zeros(
        (n_trajectories, 2, n_bins), dtype=np.int64
    )
    transition_counts = np.zeros(
        (n_trajectories, 3, n_bins), dtype=np.int64
    )
    histogram_outside = np.zeros(n_trajectories, dtype=np.int64)
    transition_outside = np.zeros(n_trajectories, dtype=np.int64)

    for trajectory in prange(n_trajectories):
        np.random.seed(seed + trajectory)
        x = 0.0
        y = 0.0
        state = 0

        for _ in range(equil_steps):
            dx_mean = (y - x) * (1.0 - exp_dt_tau)
            x += np.random.normal(dx_mean, dx_sd)

            force_on_motor = kappa * (x - y)
            forward_rate = kf0 * np.exp(beta * df * force_on_motor)
            backward_rate = kb0 * np.exp(beta * db * force_on_motor)
            random_number = np.random.random()

            if state == 0:
                if random_number < kc * dt_sim:
                    state = 1
            else:
                if random_number < forward_rate * dt_sim:
                    y += d
                    state = 0
                elif random_number < (forward_rate + backward_rate) * dt_sim:
                    y -= d
                    state = 0

        x_start = x
        cargo_heat = 0.0
        motor_energy = 0.0
        forward_steps = 0
        backward_steps = 0
        activations = 0

        for step in range(steps):
            extension = x - y
            energy_before = 0.5 * kappa * extension * extension

            dx_mean = (y - x) * (1.0 - exp_dt_tau)
            x += np.random.normal(dx_mean, dx_sd)

            extension = x - y
            energy_after = 0.5 * kappa * extension * extension
            cargo_heat += energy_after - energy_before

            bin_index = int((extension - hist_min) / hist_bin_width)
            in_histogram = 0 <= bin_index < n_bins
            if (step + 1) % hist_sample_stride == 0:
                if in_histogram:
                    occupancy_counts[trajectory, state, bin_index] += 1
                else:
                    histogram_outside[trajectory] += 1

            force_on_motor = kappa * extension
            forward_rate = kf0 * np.exp(beta * df * force_on_motor)
            backward_rate = kb0 * np.exp(beta * db * force_on_motor)
            random_number = np.random.random()

            if state == 0:
                if random_number < kc * dt_sim:
                    if in_histogram:
                        transition_counts[trajectory, 0, bin_index] += 1
                    else:
                        transition_outside[trajectory] += 1
                    activations += 1
                    state = 1
            else:
                if random_number < forward_rate * dt_sim:
                    if in_histogram:
                        transition_counts[trajectory, 1, bin_index] += 1
                    else:
                        transition_outside[trajectory] += 1
                    energy_before = 0.5 * kappa * extension * extension
                    y += d
                    extension_after = x - y
                    energy_after = (
                        0.5 * kappa * extension_after * extension_after
                    )
                    motor_energy += energy_after - energy_before
                    forward_steps += 1
                    state = 0
                elif random_number < (forward_rate + backward_rate) * dt_sim:
                    if in_histogram:
                        transition_counts[trajectory, 2, bin_index] += 1
                    else:
                        transition_outside[trajectory] += 1
                    energy_before = 0.5 * kappa * extension * extension
                    y -= d
                    extension_after = x - y
                    energy_after = (
                        0.5 * kappa * extension_after * extension_after
                    )
                    motor_energy += energy_after - energy_before
                    backward_steps += 1
                    state = 0

        cargo_heat_beta_per_s[trajectory] = beta * cargo_heat / duration
        motor_energy_beta_per_s[trajectory] = beta * motor_energy / duration
        velocity_nm_per_s[trajectory] = (x - x_start) / duration
        forward_rate_per_s[trajectory] = forward_steps / duration
        backward_rate_per_s[trajectory] = backward_steps / duration
        activation_rate_per_s[trajectory] = activations / duration

    return (
        cargo_heat_beta_per_s,
        motor_energy_beta_per_s,
        velocity_nm_per_s,
        forward_rate_per_s,
        backward_rate_per_s,
        activation_rate_per_s,
        occupancy_counts,
        transition_counts,
        histogram_outside,
        transition_outside,
    )


def sem(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def rebin_last_axis(counts, factor):
    """Rebin integer histogram counts by an integer factor."""

    counts = np.asarray(counts)
    if factor == 1:
        return counts
    if counts.shape[-1] % factor:
        raise ValueError("histogram bin count must be divisible by rebin factor")
    shape = counts.shape[:-1] + (counts.shape[-1] // factor, factor)
    return counts.reshape(shape).sum(axis=-1)


def information_rate_from_counts(
    occupancy_counts,
    transition_counts,
    total_time,
    bin_width,
    pseudocount=0.5,
):
    """Motor-generated information current from binned stationary fluxes.

    The extension is e=x-y.  The three transition types are activation
    (0,e)->(1,e), forward step (1,e)->(0,e-d), and backward step
    (1,e)->(0,e+d).  The event contribution is the change in log conditional
    density, log[p(x|Y_after)/p(x|Y_before)].
    """

    occupancy_counts = np.asarray(occupancy_counts, dtype=float)
    transition_counts = np.asarray(transition_counts, dtype=float)
    if occupancy_counts.shape[0] != 2 or transition_counts.shape[0] != 3:
        raise ValueError("unexpected state or transition-count shape")

    n_bins = occupancy_counts.shape[1]
    shift = int(round(d / bin_width))
    if not np.isclose(shift * bin_width, d):
        raise ValueError("bin width must divide the kinesin step size")

    conditional_probability = occupancy_counts + pseudocount
    conditional_probability /= conditional_probability.sum(
        axis=1, keepdims=True
    )
    log_probability = np.log(conditional_probability)

    activation = np.sum(
        transition_counts[0]
        * (log_probability[1] - log_probability[0])
    )
    forward = np.sum(
        transition_counts[1, shift:]
        * (
            log_probability[0, :-shift]
            - log_probability[1, shift:]
        )
    )
    backward = np.sum(
        transition_counts[2, :-shift]
        * (
            log_probability[0, shift:]
            - log_probability[1, :-shift]
        )
    )
    return float((activation + forward + backward) / total_time)


def estimate_information_flow(
    occupancy_by_trajectory,
    transitions_by_trajectory,
    duration,
    base_bin_width,
    reported_bin_width,
    block_count,
):
    """Return pooled information current, block SEM, and bin convergence."""

    n_trajectories = occupancy_by_trajectory.shape[0]
    if n_trajectories % block_count:
        raise ValueError("N must be divisible by block-count")

    convergence_widths = np.array(
        [base_bin_width, reported_bin_width, 2.0 * reported_bin_width]
    )
    convergence_rates = np.empty(convergence_widths.size)
    for index, bin_width in enumerate(convergence_widths):
        factor = int(round(bin_width / base_bin_width))
        occupancy = rebin_last_axis(
            occupancy_by_trajectory.sum(axis=0), factor
        )
        transitions = rebin_last_axis(
            transitions_by_trajectory.sum(axis=0), factor
        )
        convergence_rates[index] = information_rate_from_counts(
            occupancy,
            transitions,
            n_trajectories * duration,
            bin_width,
        )

    reported_index = 1
    block_size = n_trajectories // block_count
    block_rates = np.empty(block_count)
    factor = int(round(reported_bin_width / base_bin_width))
    for block_index in range(block_count):
        start = block_index * block_size
        stop = start + block_size
        occupancy = rebin_last_axis(
            occupancy_by_trajectory[start:stop].sum(axis=0), factor
        )
        transitions = rebin_last_axis(
            transitions_by_trajectory[start:stop].sum(axis=0), factor
        )
        block_rates[block_index] = information_rate_from_counts(
            occupancy,
            transitions,
            block_size * duration,
            reported_bin_width,
        )

    return (
        float(convergence_rates[reported_index]),
        sem(block_rates),
        block_rates,
        convergence_widths,
        convergence_rates,
    )


def configure_style(style_path):
    plt.style.use(style_path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_flows(rows, output_pdf, output_png):
    """Create a manuscript-style thermodynamic-flow figure."""

    script_dir = Path(__file__).resolve().parent
    configure_style(script_dir / "Leighton_Style_Colors2")

    noise_ratio = np.array([row["noise_ratio"] for row in rows])
    chemical_work = np.array(
        [row["chemical_work_net_beta_per_s"] for row in rows]
    )
    chemical_work_sem = np.array(
        [row["chemical_work_net_sem"] for row in rows]
    )
    energy_flow = np.array(
        [row["energy_y_to_x_beta_per_s"] for row in rows]
    )
    energy_flow_sem = np.array(
        [row["energy_y_to_x_sem"] for row in rows]
    )
    information_flow = np.array(
        [row["information_y_to_x_nats_per_s"] for row in rows]
    )
    information_flow_sem = np.array(
        [row["information_y_to_x_sem"] for row in rows]
    )

    width_cm = 8.6
    height_cm = 5.4
    figure, axis = plt.subplots(
        figsize=(width_cm / 2.54, height_cm / 2.54)
    )

    series = [
        (
            chemical_work,
            chemical_work_sem,
            "tab:orange",
            "D",
            r"$\beta\dot{W}_{\rm chem}$",
        ),
        (
            energy_flow,
            energy_flow_sem,
            "tab:green",
            "o",
            r"$\beta\dot{E}_{Y\to X}$",
        ),
        (
            information_flow,
            information_flow_sem,
            "gray",
            "s",
            r"$\dot{I}^{Y}$",
        ),
    ]
    for values, errors, color, marker, label in series:
        axis.errorbar(
            noise_ratio,
            values,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            elinewidth=1.0,
            capsize=1.5,
            alpha=0.65,
            label=label,
        )

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlim(noise_ratio[0], noise_ratio[-1])
    axis.set_ylim(-500.0, 3000.0)
    axis.set_xlabel(
        r"Nonequilibrium Noise Strength $D_{\rm neq}/D_{\rm eq}$",
        fontsize=10,
    )
    axis.set_ylabel(r"Thermodynamic Flow (/s)", fontsize=10)
    axis.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    figure.tight_layout(pad=0.15)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf)
    figure.savefig(output_png, dpi=400)
    plt.close(figure)


def validate_args(args):
    if args.N < 2 or args.T <= 0.0 or args.T_equil < 0.0:
        raise ValueError("N and T must be positive; T-equil cannot be negative")
    if args.N % args.block_count:
        raise ValueError("N must be divisible by block-count")
    if args.dt_sim <= 0.0:
        raise ValueError("dt-sim must be positive")
    if args.hist_sample_stride < 1:
        raise ValueError("hist-sample-stride must be at least one")
    if args.hist_bin_width <= 0.0 or args.reported_bin_width <= 0.0:
        raise ValueError("histogram bin widths must be positive")
    n_bins_float = (
        (args.hist_max - args.hist_min) / args.hist_bin_width
    )
    n_bins = int(round(n_bins_float))
    if n_bins < 1 or not np.isclose(n_bins, n_bins_float):
        raise ValueError("histogram range must contain an integer number of bins")
    if n_bins % int(round(2.0 * args.reported_bin_width / args.hist_bin_width)):
        raise ValueError("histogram bins are incompatible with convergence rebinning")
    for width in (
        args.hist_bin_width,
        args.reported_bin_width,
        2.0 * args.reported_bin_width,
    ):
        if not np.isclose(round(d / width) * width, d):
            raise ValueError("each information-flow bin width must divide 8 nm")
    return n_bins


def run(args):
    n_bins = validate_args(args)
    noise_ratios = np.asarray(args.noise_ratios, dtype=float)
    if np.any(noise_ratios < 0.0):
        raise ValueError("noise ratios cannot be negative")

    results_dir = Path(args.results_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve()
    output_png = Path(args.output_png).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    n_conditions = noise_ratios.size
    shape = (n_conditions, args.N)
    raw_cargo_heat = np.empty(shape)
    raw_energy_flow = np.empty(shape)
    raw_velocity = np.empty(shape)
    raw_forward_rate = np.empty(shape)
    raw_backward_rate = np.empty(shape)
    raw_activation_rate = np.empty(shape)
    information_blocks = np.empty((n_conditions, args.block_count))
    information_convergence = np.empty((n_conditions, 3))
    aggregate_occupancy = np.empty((n_conditions, 2, n_bins), dtype=np.int64)
    aggregate_transitions = np.empty((n_conditions, 3, n_bins), dtype=np.int64)
    rows = []

    for condition_index, noise_ratio in enumerate(noise_ratios):
        print(
            f"[{condition_index + 1}/{n_conditions}] "
            f"Dneq/Deq={noise_ratio:g}",
            flush=True,
        )
        seed = args.seed + 100_000 * condition_index
        (
            cargo_heat,
            energy_flow,
            velocity,
            forward_rate,
            backward_rate,
            activation_rate,
            occupancy_counts,
            transition_counts,
            histogram_outside,
            transition_outside,
        ) = simulate_condition(
            noise_ratio,
            args.N,
            args.T,
            args.T_equil,
            args.dt_sim,
            args.hist_min,
            args.hist_bin_width,
            n_bins,
            args.hist_sample_stride,
            seed,
        )

        (
            information_flow,
            information_sem,
            block_rates,
            convergence_widths,
            convergence_rates,
        ) = estimate_information_flow(
            occupancy_counts,
            transition_counts,
            args.T,
            args.hist_bin_width,
            args.reported_bin_width,
            args.block_count,
        )

        net_step_rate = forward_rate - backward_rate
        total_step_rate = forward_rate + backward_rate
        chemical_work_net = beta_delta_mu * net_step_rate
        chemical_work_turnover = beta_delta_mu * total_step_rate
        energy_balance = energy_flow + cargo_heat
        beta_eff_over_beta = 1.0 / (1.0 + noise_ratio)
        sigma_x = beta_eff_over_beta * np.mean(energy_flow) + information_flow
        sigma_y = (
            np.mean(chemical_work_net)
            - np.mean(energy_flow)
            - information_flow
        )

        occupancy_total = occupancy_counts.sum(axis=0)
        transitions_total = transition_counts.sum(axis=0)
        raw_cargo_heat[condition_index] = cargo_heat
        raw_energy_flow[condition_index] = energy_flow
        raw_velocity[condition_index] = velocity
        raw_forward_rate[condition_index] = forward_rate
        raw_backward_rate[condition_index] = backward_rate
        raw_activation_rate[condition_index] = activation_rate
        aggregate_occupancy[condition_index] = occupancy_total
        aggregate_transitions[condition_index] = transitions_total
        information_blocks[condition_index] = block_rates
        information_convergence[condition_index] = convergence_rates

        row = {
            "noise_ratio": float(noise_ratio),
            "velocity_nm_per_s": float(np.mean(velocity)),
            "velocity_sem": sem(velocity),
            "net_step_rate_per_s": float(np.mean(net_step_rate)),
            "net_step_rate_sem": sem(net_step_rate),
            "total_step_rate_per_s": float(np.mean(total_step_rate)),
            "activation_rate_per_s": float(np.mean(activation_rate)),
            "chemical_work_net_beta_per_s": float(
                np.mean(chemical_work_net)
            ),
            "chemical_work_net_sem": sem(chemical_work_net),
            "chemical_work_turnover_beta_per_s": float(
                np.mean(chemical_work_turnover)
            ),
            "energy_y_to_x_beta_per_s": float(np.mean(energy_flow)),
            "energy_y_to_x_sem": sem(energy_flow),
            "cargo_heat_beta_per_s": float(np.mean(cargo_heat)),
            "cargo_heat_sem": sem(cargo_heat),
            "energy_balance_beta_per_s": float(np.mean(energy_balance)),
            "energy_balance_sem": sem(energy_balance),
            "information_y_to_x_nats_per_s": information_flow,
            "information_y_to_x_sem": information_sem,
            "information_bin_0p25_nats_per_s": float(convergence_rates[0]),
            "information_bin_0p5_nats_per_s": float(convergence_rates[1]),
            "information_bin_1p0_nats_per_s": float(convergence_rates[2]),
            "cargo_entropy_production_per_s": float(sigma_x),
            "motor_entropy_production_per_s": float(sigma_y),
            "histogram_outside_count": int(np.sum(histogram_outside)),
            "transition_outside_count": int(np.sum(transition_outside)),
        }
        rows.append(row)

        print(
            f"  beta Wchem={row['chemical_work_net_beta_per_s']:.3f}, "
            f"beta E_YX={row['energy_y_to_x_beta_per_s']:.3f}, "
            f"IY={information_flow:.3f} +/- {information_sem:.3f}, "
            f"balance={row['energy_balance_beta_per_s']:.3f}",
            flush=True,
        )

        del occupancy_counts, transition_counts

    summary_path = results_dir / "thermodynamic_flows_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        results_dir / "thermodynamic_flows_data.npz",
        noise_ratios=noise_ratios,
        cargo_heat_beta_per_s=raw_cargo_heat,
        energy_y_to_x_beta_per_s=raw_energy_flow,
        velocity_nm_per_s=raw_velocity,
        forward_rate_per_s=raw_forward_rate,
        backward_rate_per_s=raw_backward_rate,
        activation_rate_per_s=raw_activation_rate,
        information_block_estimates=information_blocks,
        information_convergence_bin_widths_nm=convergence_widths,
        information_convergence_nats_per_s=information_convergence,
        extension_occupancy_counts=aggregate_occupancy,
        extension_transition_counts=aggregate_transitions,
    )

    metadata = {
        "N": args.N,
        "T_s": args.T,
        "T_equil_s": args.T_equil,
        "dt_sim_s": args.dt_sim,
        "force_pN": 0.0,
        "noise_ratios": noise_ratios.tolist(),
        "beta_delta_mu": beta_delta_mu,
        "chemical_work_definition": "beta*Delta_mu*(forward-backward step rate)",
        "chemical_work_turnover_diagnostic": (
            "beta*Delta_mu*(forward+backward step rate)"
        ),
        "energy_flow_definition": (
            "linker-energy change caused by motor jumps; positive Y to X"
        ),
        "information_flow_definition": (
            "sum over motor jumps of log[p(x|Y_after)/p(x|Y_before)] per time"
        ),
        "information_units": "nats/s",
        "motor_state_definition": "chemical state plus lattice position",
        "extension_definition": "e=x-y",
        "histogram_range_nm": [args.hist_min, args.hist_max],
        "fine_histogram_bin_width_nm": args.hist_bin_width,
        "reported_information_bin_width_nm": args.reported_bin_width,
        "histogram_sample_stride": args.hist_sample_stride,
        "block_count": args.block_count,
        "information_sem": "SEM across equal trajectory blocks",
        "source_model": "zero-load linear-coupling model used for Fig. 3",
    }
    with (results_dir / "thermodynamic_flows_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    plot_flows(rows, output_pdf, output_png)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {output_pdf}", flush=True)
    print(f"Saved {output_png}", flush=True)
    return rows


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=n_trajectories_default)
    parser.add_argument("--T", type=float, default=duration_default)
    parser.add_argument("--T-equil", type=float, default=equil_duration_default)
    parser.add_argument("--dt-sim", type=float, default=dt_sim_default)
    parser.add_argument(
        "--noise-ratios",
        type=float,
        nargs="+",
        default=noise_ratios_default.tolist(),
    )
    parser.add_argument("--hist-min", type=float, default=hist_min_default)
    parser.add_argument("--hist-max", type=float, default=hist_max_default)
    parser.add_argument(
        "--hist-bin-width", type=float, default=hist_bin_width_default
    )
    parser.add_argument(
        "--hist-sample-stride",
        type=int,
        default=hist_sample_stride_default,
    )
    parser.add_argument(
        "--reported-bin-width",
        type=float,
        default=reported_bin_width_default,
    )
    parser.add_argument("--block-count", type=int, default=block_count_default)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--results-dir",
        default=str(script_dir / "thermodynamic_flow_results"),
    )
    parser.add_argument(
        "--output-pdf",
        default=str(
            repository_root
            / "output"
            / "pdf"
            / "Figure_Thermodynamic_Flows_Noneq.pdf"
        ),
    )
    parser.add_argument(
        "--output-png",
        default=str(
            repository_root
            / "output"
            / "pdf"
            / "Figure_Thermodynamic_Flows_Noneq.png"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
