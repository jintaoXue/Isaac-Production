import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data_time_latency_pf_kf_ekf_num_humans = """{

2025-12-05 03:32:47
PF inference time step: (6.514491367649943e-05, 582013),               KF inference time step: (5.124174237464911e-05, 582013),               EKF inference time step: (3.2492090065396005e-05, 582013)
2025-12-05 03:32:47
PF inference time step: (4.7620238415270695e-05, 348354),               KF inference time step: (3.750177360196556e-05, 348354),               EKF inference time step: (2.755197045835607e-05, 348354)
2025-12-05 03:32:47
PF inference time step: (4.587385852114171e-05, 166766),               KF inference time step: (3.553885359514938e-05, 166766),               EKF inference time step: (2.6716263900996013e-05, 166766)
}"""


data_num_particles_100_to_1000_pf = """{

100


2025-12-04 20:32:51
PF inference time step: (6.005774621859394e-05, 114781),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:32:51
PF inference time step: (4.4601114907876876e-05, 67658),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:32:51
PF inference time step: (4.196626961333406e-05, 32085),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:33:06
fatigue_coe_accu: {'pf': 0.06831443411194615, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 20:33:06
recovery_coe_accu: {'pf': 0.051553178413046734, 'kf': 0.0, 'ekf': 0.0}



200

2025-12-04 20:44:40
PF inference time step: (6.048958723916017e-05, 114041),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:44:40
PF inference time step: (4.3943238737443e-05, 66935),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:44:40
PF inference time step: (4.154216378325389e-05, 32088),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:44:55
fatigue_coe_accu: {'pf': 0.0655684802474247, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 20:44:55
recovery_coe_accu: {'pf': 0.05125251428948508, 'kf': 0.0, 'ekf': 0.0}

300

2025-12-04 20:56:39
PF inference time step: (6.294189849174423e-05, 115987),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:56:39
PF inference time step: (4.6104691152361726e-05, 69002),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:56:39
PF inference time step: (4.3416725466923404e-05, 32153),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 20:56:56
fatigue_coe_accu: {'pf': 0.0670745681764351, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 20:56:56
recovery_coe_accu: {'pf': 0.050527682145022686, 'kf': 0.0, 'ekf': 0.0}

400

2025-12-04 21:08:32
PF inference time step: (6.223486848855039e-05, 115183),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:08:32
PF inference time step: (4.558689628417522e-05, 67793),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:08:32
PF inference time step: (4.3170029434553775e-05, 32597),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:08:46
fatigue_coe_accu: {'pf': 0.06445193845364783, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 21:08:46
recovery_coe_accu: {'pf': 0.049039851480888, 'kf': 0.0, 'ekf': 0.0}

500

2025-12-04 21:20:40
PF inference time step: (6.417253910580829e-05, 115705),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:20:40
PF inference time step: (4.6502945820839024e-05, 68412),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:20:40
PF inference time step: (4.373799130814796e-05, 32531),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:20:56
fatigue_coe_accu: {'pf': 0.06590557437804011, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 21:20:56
recovery_coe_accu: {'pf': 0.0537248507142067, 'kf': 0.0, 'ekf': 0.0}

600

2025-12-04 21:32:51
PF inference time step: (6.479634540589076e-05, 116344),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:32:51
PF inference time step: (4.7369505915341555e-05, 68819),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:32:51
PF inference time step: (4.452533501868015e-05, 32070),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:33:07
fatigue_coe_accu: {'pf': 0.065864790789783, 'kf': 0.0, 'ekf': 0.0}
2025-12-04 21:33:07
recovery_coe_accu: {'pf': 0.052597812935709955, 'kf': 0.0, 'ekf': 0.0}

700
2025-12-04 21:44:50
PF inference time step: (6.58875993781306e-05, 115234),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:44:50
PF inference time step: (4.743248570741028e-05, 68037),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:44:50
PF inference time step: (4.435070472812572e-05, 32107),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:45:06
fatigue_coe_accu: {'pf': 0.06667526772038804, 'kf': None, 'ekf': None}
2025-12-04 21:45:06
recovery_coe_accu: {'pf': 0.05238780075063308, 'kf': None, 'ekf': None}


800
2025-12-04 21:56:59
PF inference time step: (6.568692426550237e-05, 117392),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:56:59
PF inference time step: (4.759441176727837e-05, 70369),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:56:59
PF inference time step: (4.433218463805747e-05, 32647),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 21:57:12
fatigue_coe_accu: {'pf': 0.06753600860635439, 'kf': None, 'ekf': None}
2025-12-04 21:57:12
recovery_coe_accu: {'pf': 0.05056972676474187, 'kf': None, 'ekf': None}

900

2025-12-04 22:08:54
PF inference time step: (6.574297629504202e-05, 115446),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:08:54
PF inference time step: (4.782846561193198e-05, 68579),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:08:54
PF inference time step: (4.476031031088455e-05, 32142),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:09:11
fatigue_coe_accu: {'pf': 0.06622521090838644, 'kf': None, 'ekf': None}
2025-12-04 22:09:11
recovery_coe_accu: {'pf': 0.0520526344784432, 'kf': None, 'ekf': None}


1000

2025-12-04 22:20:46
PF inference time step: (6.69714805286886e-05, 114438),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:20:46
PF inference time step: (4.8518600222761764e-05, 67084),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:20:46
PF inference time step: (4.570796744795441e-05, 32340),               KF inference time step: (nan, 0),               EKF inference time step: (nan, 0)
2025-12-04 22:21:01
fatigue_coe_accu: {'pf': 0.06687692407932547, 'kf': None, 'ekf': None}
2025-12-04 22:21:01
recovery_coe_accu: {'pf': 0.051132304697400995, 'kf': None, 'ekf': None}

}
"""

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
        label='Fatigue coeff. accuracy',
    )
    ax2_acc.plot(
        particles,
        rec_values,
        color='#2ca02c',
        marker='^',
        linewidth=2,
        label='Recovery coeff. accuracy',
    )
    ax2_acc.set_ylabel('Accuracy (↓)', fontsize=12)

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
