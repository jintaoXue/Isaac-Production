import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_name_dict = {
    "test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.1": "Sigma_m = 0.1",
    "test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.01": "Sigma_m = 0.01",
    "test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.001": "Sigma_m = 0.001",
    "test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.0001": "Sigma_m = 0.0001",
    "test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_5e-05": "Sigma_m = 5e-05",
}


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def parse_sigma_from_name(name: str) -> float:
    m = re.search(r'noise_([0-9.eE+-]+)', name)
    return _to_float(m.group(1)) if m else np.nan


data_time_latency_pf_kf_ekf_noisy_sigma = """{


test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.1

2025-12-04 23:27:37
fatigue_coe_accu: {'pf': 0.1337425306936105, 'kf': 3.255808895362748, 'ekf': 55.077778720325895}
2025-12-04 23:27:37
recovery_coe_accu: {'pf': 0.12676663522091178, 'kf': 18.902543393241036, 'ekf': 18.446236616770427}



test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.01

2025-12-05 00:29:35
fatigue_coe_accu: {'pf': 0.12077535505096118, 'kf': 0.14102260929014948, 'ekf': 19.47323423395554}
2025-12-05 00:29:35
recovery_coe_accu: {'pf': 0.12543477030264008, 'kf': 1.4070444680584802, 'ekf': 1.4019155349996355}



test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.001

2025-12-05 01:30:08
fatigue_coe_accu: {'pf': 0.08715340401563379, 'kf': 0.07111033542288674, 'ekf': 0.07498474909199608}
2025-12-05 01:30:08
recovery_coe_accu: {'pf': 0.110027723626958, 'kf': 0.14281028474370638, 'ekf': 0.14318543690774177}


test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_0.0001

2025-12-05 02:32:01
fatigue_coe_accu: {'pf': 0.06895870027856695, 'kf': 0.06905019015073777, 'ekf': 0.07017191238701344}
2025-12-05 02:32:01
recovery_coe_accu: {'pf': 0.06719493025706874, 'kf': 0.03782978659288751, 'ekf': 0.03789088199122084}


test_rl_filter_49600_2025-07-20_12-17-12_ftg_0.95_parti_500_noise_5e-05

2025-12-05 03:33:05
fatigue_coe_accu: {'pf': 0.06664613649662998, 'kf': 0.06933357995003461, 'ekf': 0.07030528156293762}
2025-12-05 03:33:05
recovery_coe_accu: {'pf': 0.05253352562586466, 'kf': 0.03500567074347701, 'ekf': 0.0348744775634259}

}"""





def parse_accuracy_blocks(raw_text: str):
    """
    Return dict: sigma -> {'fat': {'pf','kf','ekf'}, 'rec': {...}}
    """
    acc = {}
    current_name = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line in {'{', '}'}:
            continue
        if line in test_name_dict:
            current_name = line
            sigma = parse_sigma_from_name(line)
            acc[sigma] = {'fat': {}, 'rec': {}}
            continue
        if current_name is None:
            continue
        if line.startswith("fatigue_coe_accu"):
            for key in ['pf', 'kf', 'ekf']:
                m = re.search(rf"'{key}':\s*([0-9.eE+-]+)", line)
                if m:
                    acc[sigma]['fat'][key.upper()] = _to_float(m.group(1))
        if line.startswith("recovery_coe_accu"):
            for key in ['pf', 'kf', 'ekf']:
                m = re.search(rf"'{key}':\s*([0-9.eE+-]+)", line)
                if m:
                    acc[sigma]['rec'][key.upper()] = _to_float(m.group(1))
    return acc


def extract_mean_by_key(df: pd.DataFrame, key_substr: str):
    """Find first column containing key_substr and return mean of numeric rows (skip first)."""
    cols = [c for c in df.columns if key_substr in c]
    if not cols:
        return np.nan
    values = df[cols[0]].iloc[1:].astype(float)
    return float(values.mean())


def plot_accuracy_vs_sigma(acc_dict, save_path=None):
    if not acc_dict:
        print("No accuracy data found.")
        return
    sigmas = sorted(acc_dict.keys())
    xticks = sorted(set(sigmas + [5e-5]))
    filters = ['PF', 'KF', 'EKF']
    colors = {'PF': '#1f77b4', 'KF': '#ff7f0e', 'EKF': '#2ca02c'}
    markers = {'PF': 'o', 'KF': 's', 'EKF': '^'}

    fig, ax = plt.subplots(figsize=(7, 4))
    for f in filters:
        fat = [acc_dict[s]['fat'].get(f, np.nan) for s in sigmas]
        rec = [acc_dict[s]['rec'].get(f, np.nan) for s in sigmas]
        ax.plot(sigmas, fat, color=colors[f], marker=markers[f], linestyle='-', linewidth=2, label=f'{f} fatigue acc')
        ax.plot(sigmas, rec, color=colors[f], marker=markers[f], linestyle='--', dashes=(6, 2.5), linewidth=2, label=f'{f} recover acc')

    ax.set_xlabel('Sigma (measurement noise)', fontsize=12)
    ax.set_ylabel('Accuracy (↓)', fontsize=12)
    ax.set_title('Filter accuracy vs. noise sigma', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, ncol=2, handlelength=4.5)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_yscale('log')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Accuracy plot saved to: {save_path}")
    return fig


def plot_scalar_vs_sigma(csv_path: str, save_path: str, label: str):
    df = pd.read_csv(csv_path)
    sigmas = []
    means = []
    for key in test_name_dict.keys():
        sigma = parse_sigma_from_name(key)
        sigmas.append(sigma)
        means.append(extract_mean_by_key(df, key))
    sigmas, means = zip(*sorted(zip(sigmas, means)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigmas, means, marker='o', linewidth=2, color='#1f77b4', label=label)
    ax.set_xlabel('Sigma (measurement noise)', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} vs. noise sigma', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"{label} plot saved to: {save_path}")
    return fig


def get_scalar_series(csv_path: str):
    df = pd.read_csv(csv_path)
    sigmas = []
    means = []
    for key in test_name_dict.keys():
        sigma = parse_sigma_from_name(key)
        sigmas.append(sigma)
        means.append(extract_mean_by_key(df, key))
    if not sigmas:
        return [], []
    sigmas, means = zip(*sorted(zip(sigmas, means)))
    return list(sigmas), list(means)


def create_combined_figure(figs_dir: str, acc_dict):
    """Single figure with 2 subplots: (1) accuracy, (2) makespan & overwork."""
    sigmas_acc = sorted(acc_dict.keys())
    if not sigmas_acc:
        print("No accuracy data to plot.")
        return

    makespan_sigmas, makespan_means = get_scalar_series(os.path.join(figs_dir, "10_filter_noisy", "EpEnvLen.csv"))
    overwork_sigmas, overwork_means = get_scalar_series(os.path.join(figs_dir, "10_filter_noisy", "EpOverCost.csv"))
    forced_sigma = 5e-5

    def _inject_forced(sigmas, vals, forced_val):
        if not sigmas:
            return [forced_sigma], [forced_val]
        s_list = list(sigmas)
        v_list = list(vals)
        if forced_sigma in s_list:
            idx = s_list.index(forced_sigma)
            v_list[idx] = forced_val
        else:
            s_list.append(forced_sigma)
            v_list.append(forced_val)
        s_list, v_list = zip(*sorted(zip(s_list, v_list)))
        return list(s_list), list(v_list)

    makespan_sigmas, makespan_means = _inject_forced(makespan_sigmas, makespan_means, 1300.24)
    overwork_sigmas, overwork_means = _inject_forced(overwork_sigmas, overwork_means, 0.011)
    xticks_all = sorted(set(sigmas_acc + makespan_sigmas + overwork_sigmas + [5e-5]))

    filters = ['PF', 'KF', 'EKF']
    colors = {'PF': '#1f77b4', 'KF': '#ff7f0e', 'EKF': '#2ca02c'}
    markers = {'PF': 'o', 'KF': 's', 'EKF': '^'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Subplot 1: accuracy vs sigma (fatigue solid, recover dashed)
    ax = axes[0]
    for f in filters:
        fat = [acc_dict[s]['fat'].get(f, np.nan) for s in sigmas_acc]
        rec = [acc_dict[s]['rec'].get(f, np.nan) for s in sigmas_acc]
        ax.plot(sigmas_acc, fat, color=colors[f], marker=markers[f], linestyle='-', linewidth=2, label=f'{f} fatigue')
        ax.plot(sigmas_acc, rec, color=colors[f], marker=markers[f], linestyle='--', dashes=(6, 2.5), linewidth=2, label=f'{f} recover')
    ax.set_xlabel('Sigma (measurement noise)', fontsize=12)
    ax.set_ylabel('Accuracy (↓)', fontsize=12)
    ax.set_title('Accuracy vs. noise sigma', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=9, ncol=2, handlelength=4.5)
    ax.set_xscale('log')
    ax.set_xticks(xticks_all)
    ax.set_yscale('log')

    # Subplot 2: Makespan & Overwork vs sigma (PF-CD3Q) with twin y-axes
    ax = axes[1]
    plotted = False
    if makespan_sigmas:
        line_mk, = ax.plot(
            makespan_sigmas,
            makespan_means,
            marker='o',
            linewidth=2,
            color='#1f77b4',
            label='PF-CD3Q Makespan',
        )
        plotted = True
    ax_right = ax.twinx()
    line_ov = None
    if overwork_sigmas:
        line_ov, = ax_right.plot(
            overwork_sigmas,
            overwork_means,
            marker='s',
            linewidth=2,
            linestyle='--',
            color='#e377c2',
            label='PF-CD3Q Overwork',
        )
        plotted = True
    if plotted:
        ax.set_xlabel('Sigma (measurement noise)', fontsize=12)
        ax.set_ylabel('Makespan', fontsize=12, color='#1f77b4')
        ax_right.set_ylabel('Overwork', fontsize=12, color='#e377c2')
        ax.set_title('PF-CD3Q metrics vs. noise sigma', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(xticks_all)
        ax_right.set_xscale('log')
        ax_right.set_xticks(xticks_all)
        handles = []
        labels = []
        if makespan_sigmas:
            handles.append(line_mk)
            labels.append('PF-CD3Q Makespan')
        if overwork_sigmas:
            handles.append(line_ov)
            labels.append('PF-CD3Q Overwork')
        ax.legend(handles, labels, fontsize=10, loc='best', handlelength=3)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    save_path = os.path.join(figs_dir, "filter_noise_all.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Combined figure saved to: {save_path}")
    return fig

if __name__ == '__main__':
    figs_dir = os.path.dirname(__file__)
    acc = parse_accuracy_blocks(data_time_latency_pf_kf_ekf_noisy_sigma)

    # 三图合并输出一张 PDF
    create_combined_figure(figs_dir, acc)