import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data_time_latency_pf_kf_ekf_num_humans = """{


test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.1



}"""



PF_PATTERN = re.compile(r'PF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
KF_PATTERN = re.compile(r'KF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
EKF_PATTERN = re.compile(r'EKF inference time step:\s*([0-9.eE+-]+|nan)', re.IGNORECASE)
FAT_PATTERN = re.compile(r'Fat_coe_accu:([0-9.eE+-]+)', re.IGNORECASE)
REC_PATTERN = re.compile(r'Rec_coe_accu:([0-9.eE+-]+)', re.IGNORECASE)


def _to_float(value: str) -> float:
    try:
        if value is None:
            return np.nan
        value = value.strip()
        if value.lower() == "nan":
            return np.nan
        return float(value)
    except (AttributeError, ValueError):
        return np.nan


def _extract_with_pattern(pattern, text: str) -> float:
    match = pattern.search(text)
    return _to_float(match.group(1)) if match else np.nan


def _nanmean(values):
    if not values:
        return np.nan
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else np.nan


def aggregate_latency_by_humans(raw_text: str):
    """
    New format: each line has PF/KF/EKF inference time step: (value, count)
    in order for human 1,2,3... We keep per-filter values and their cumsums.
    """
    values = {'PF': [], 'KF': [], 'EKF': []}
    for line in raw_text.splitlines():
        if 'inference time step' not in line:
            continue
        for key in values.keys():
            m = re.search(rf'{key}\s+inference time step:\s*\(([^,]+),', line)
            if m:
                values[key].append(_to_float(m.group(1)))

    humans = list(range(1, len(values['PF']) + 1))
    cumsums = {k: list(np.cumsum(v)) for k, v in values.items()}
    return humans, values, cumsums


def plot_filter_latency_vs_humans(raw_text: str, save_path: str | None = None):
    humans, values, cumsums = aggregate_latency_by_humans(raw_text)
    if not humans:
        print("No latency data found for human comparison.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    style_map = {
        'PF': ('#1f77b4', 'o'),
        'KF': ('#ff7f0e', 's'),
        'EKF': ('#2ca02c', '^'),
    }
    for key, (color, marker) in style_map.items():
        ax.plot(
            humans,
            np.array(cumsums[key]) * 1e6,
            marker=marker,
            color=color,
            linewidth=2,
            label=f'{key} latency cumsum (µs)',
        )

    ax.set_xlabel('Number of humans', fontsize=12)
    ax.set_ylabel('Cumulative latency (µs)', fontsize=12)
    ax.set_title('Filter latency cumsum vs. humans', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(humans)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Latency vs. humans plot saved to: {save_path}")

    return fig


def aggregate_pf_particle_metrics(raw_text: str):
    """
    New format per particle block:
      PF inference time step: (value, count)
      fatigue_coe_accu: {'pf': v, ...}
      recovery_coe_accu: {'pf': v, ...}
    We compute weighted mean latency using the counts, and plain mean for accuracies.
    """
    particle_stats = {}
    current_particle = None
    latency_weighted_sum = 0.0
    latency_weight = 0.0
    fat_values = []
    rec_values = []

    def flush_particle():
        if current_particle is None:
            return
        if latency_weight > 0:
            latency_mean = latency_weighted_sum / latency_weight
        else:
            latency_mean = np.nan
        particle_stats[current_particle] = {
            'latency': latency_mean,
            'fat': _nanmean(fat_values),
            'rec': _nanmean(rec_values),
        }

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line in {'{', '}'}:
            continue

        # start of a new particle block
        if re.fullmatch(r'\d+', line):
            flush_particle()
            current_particle = int(line)
            latency_weighted_sum = 0.0
            latency_weight = 0.0
            fat_values = []
            rec_values = []
            continue

        # latency with count
        if 'PF inference time step' in line and current_particle is not None:
            m = re.search(r'PF inference time step:\s*\(([^,]+),\s*([^)]+)\)', line)
            if m:
                val = _to_float(m.group(1))
                cnt = _to_float(m.group(2))
                if not np.isnan(val) and not np.isnan(cnt):
                    latency_weighted_sum += val * cnt
                    latency_weight += cnt
            continue

        # accuracies
        if line.startswith('fatigue_coe_accu') and current_particle is not None:
            m = re.search(r"'pf':\s*([0-9.eE+-]+)", line)
            if m:
                fat_values.append(_to_float(m.group(1)))
            continue
        if line.startswith('recovery_coe_accu') and current_particle is not None:
            m = re.search(r"'pf':\s*([0-9.eE+-]+)", line)
            if m:
                rec_values.append(_to_float(m.group(1)))
            continue

    # flush last
    flush_particle()
    return particle_stats


def plot_pf_particles_metrics(raw_text: str, save_path: str | None = None):
    particle_stats = aggregate_pf_particle_metrics(raw_text)
    if not particle_stats:
        print("No PF particle data found.")
        return None

    particles = sorted(particle_stats.keys())
    latency_us = [particle_stats[p]['latency'] * 1e6 for p in particles]
    fat_values = [particle_stats[p]['fat'] for p in particles]
    rec_values = [particle_stats[p]['rec'] for p in particles]

    fig, ax_latency = plt.subplots(figsize=(7, 4))
    ax_latency.plot(
        particles,
        latency_us,
        color='#1f77b4',
        marker='o',
        linewidth=2,
        label='Latency (µs)',
    )
    ax_latency.set_xlabel('Number of particles', fontsize=14)
    ax_latency.set_ylabel('PF latency (µs)', fontsize=14, color='#1f77b4')
    ax_latency.tick_params(axis='y', labelcolor='#1f77b4')
    ax_latency.grid(True, linestyle='--', alpha=0.3)

    ax_acc = ax_latency.twinx()
    ax_acc.plot(
        particles,
        fat_values,
        color='#ff7f0e',
        marker='s',
        linewidth=2,
        label='Fatigue coeff. accuracy',
    )
    ax_acc.plot(
        particles,
        rec_values,
        color='#2ca02c',
        marker='^',
        linewidth=2,
        label='Recovery coeff. accuracy',
    )
    ax_acc.set_ylabel('Accuracy', fontsize=14)

    lines, labels = ax_latency.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_latency.legend(lines + lines2, labels + labels2, fontsize=11, loc='best')
    ax_latency.set_title('PF latency & accuracy vs. particle count', fontsize=16)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"PF particle analysis plot saved to: {save_path}")

    return fig


def create_combined_figure(save_path: str | None = None):
    """Create a single figure with (left) filter latency cumsum vs humans and (right) PF latency & accuracy vs particles."""
    humans, values, cumsums = aggregate_latency_by_humans(data_time_latency_pf_kf_ekf_num_humans)
    particle_stats = aggregate_pf_particle_metrics(data_num_particles_100_to_1000_pf)

    if not humans or not particle_stats:
        print("Insufficient data to create combined figure.")
        return None

    particles = sorted(particle_stats.keys())
    latency_us = [particle_stats[p]['latency'] * 1e6 for p in particles]
    fat_values = [particle_stats[p]['fat'] for p in particles]
    rec_values = [particle_stats[p]['rec'] for p in particles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: cumsum for PF/KF/EKF
    style_map = {
        'PF': ('#1f77b4', 'o'),
        'KF': ('#ff7f0e', 's'),
        'EKF': ('#2ca02c', '^'),
    }
    for key, (color, marker) in style_map.items():
        ax1.plot(
            humans,
            np.array(cumsums[key]) * 1e6,
            label=f'{key} latency cumsum (µs)',
            color=color,
            marker=marker,
            linewidth=2,
        )
    ax1.set_xlabel('Number of humans', fontsize=12)
    ax1.set_ylabel('Cumulative latency (µs)', fontsize=12)
    ax1.set_title('Filter latency cumsum vs. humans', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xticks(humans)
    ax1.legend(fontsize=10)

    # Right subplot: PF latency + accuracy vs particles (latency on left y, accuracies on right y)
    ax2_lat = ax2
    ax2_lat.plot(
        particles,
        latency_us,
        color='#1f77b4',
        marker='o',
        linewidth=2,
        label='Latency (µs)',
    )
    ax2_lat.set_xlabel('Number of particles', fontsize=12)
    ax2_lat.set_ylabel('PF latency (µs)', fontsize=12, color='#1f77b4')
    ax2_lat.tick_params(axis='y', labelcolor='#1f77b4')
    ax2_lat.grid(True, linestyle='--', alpha=0.3)

    ax2_acc = ax2_lat.twinx()
    ax2_acc.plot(
        particles,
        fat_values,
        color='#ff7f0e',
        marker='s',
        linewidth=2,
        label='Fatigue coeff. accuracy ↓',
    )
    ax2_acc.plot(
        particles,
        rec_values,
        color='#2ca02c',
        marker='^',
        linewidth=2,
        label='Recovery coeff. accuracy ↓',
    )
    ax2_acc.set_ylabel('Accuracy', fontsize=12)

    lines, labels = ax2_lat.get_legend_handles_labels()
    lines2, labels2 = ax2_acc.get_legend_handles_labels()
    ax2_lat.legend(
        lines + lines2,
        labels + labels2,
        fontsize=9,
        loc='center right',
        bbox_to_anchor=(0.9, 0.5),
    )
    ax2_lat.set_title('PF latency & accuracy vs. particles', fontsize=14)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Combined latency figure saved to: {save_path}")

    return fig


if __name__ == '__main__':
    figs_dir = os.path.dirname(__file__)
    combined_path = os.path.join(figs_dir, "filter_latency_combined.pdf")
    create_combined_figure(combined_path)
    plt.show()
