"""
Produce the main simulation figures in Section 5 of the SMART paper.

Reads per-seed result files written by:
    - run_RRR.R, run_SRRR.R, run_RSSVD.R, run_SOFAR.R   (baselines)
    - run_SMART.py, run_SMARTCV.py                      (SMART fixed-rank and CV)

and emits a 2x2 panel for the specified model_id into fig/simulation_<model>.pdf.

Usage:
    python visualize_simulation.py [model_id: 0|1|2]
    default model_id = 2 (Model-III, p=300 q=200 in the paper)
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
model_list = ["model1", "model2", "model3"]
exp_list = ["exp1", "exp2", "exp3", "exp4"]
rd_seed_ids = range(100)

model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
assert model_id in (0, 1, 2), "model_id must be 0, 1, or 2"
model_name = model_list[model_id]
results_base = f"result/{model_name}"

model_to_n = {
    0: [200, 400, 600, 800, 1000],
    1: [300, 500, 700, 1000, 1200],
    2: [500, 700, 1000, 1200, 1500],
}

x_axis_values = {
    "exp1": model_to_n[model_id],
    "exp2": [1, 3, 5, 7, 9, 11],
    "exp3": [0, 3, 5, 7, 10, 15, 20],
    "exp4": [0.0, 0.01, 0.02, 0.05, 0.1, 0.5],
}

x_axis_labels = {
    "exp1": r"$n$",
    "exp2": r"$r$",
    "exp3": r"$r_s$",
    "exp4": r"$\sigma_0$",
}

model_id_to_dimension = {0: (100, 50), 1: (150, 100), 2: (300, 200)}
roman = {1: "I", 2: "II", 3: "III"}
p, q = model_id_to_dimension[model_id]
model_title = f"Model-{roman[int(model_name[-1])]} (p={p} q={q})"

# "SMART" here uses the script run_SMART.py (fixed structural ranks); "SMARTCV" uses
# run_SMARTCV.py (rank + (r_u, r_v) selection via CV).
methods = {
    "RRR":     {"ext": "json", "loader": lambda f: json.load(f)},
    "SRRR":    {"ext": "json", "loader": lambda f: json.load(f)},
    "SOFAR":   {"ext": "json", "loader": lambda f: json.load(f)},
    "RSSVD":   {"ext": "json", "loader": lambda f: json.load(f)},
    "SMART":   {"ext": "pkl",  "loader": lambda f: pickle.load(f)},
    "SMARTCV": {"ext": "pkl",  "loader": lambda f: pickle.load(f)},
}

colors = {
    "RRR": "blue",
    "SMART": "gray",
    "SMARTCV": "green",
    "SOFAR": "orange",
    "SRRR": "red",
    "RSSVD": "purple",
}

markers = {
    "RRR": "o",
    "SMART": "*",
    "SMARTCV": "P",
    "SOFAR": "^",
    "SRRR": "D",
    "RSSVD": "v",
}

label_map = {"SMART": "SMART (fixed ranks)", "SMARTCV": "SMART"}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_path(method, exp, var_name, var_value, seed):
    ext = methods[method]["ext"]
    return (f"{results_base}/{exp}/{method}_result_{model_name}_{exp}"
            f"_{var_name}={var_value}_rd_seed_id={seed}.{ext}")


def load_errors(method, exp, var_name, var_value):
    loader = methods[method]["loader"]
    errors = []
    for seed in rd_seed_ids:
        path = build_path(method, exp, var_name, var_value, seed)
        if os.path.exists(path):
            with open(path, "rb") as f:
                result = loader(f)
                errors.append(result["avg_err"])
    return errors


def aggregate(errors):
    if errors:
        arr = np.array(errors)
        return arr.mean(), arr.std(ddof=1) / np.sqrt(len(errors))
    return None, None


def horizontal_stats(errors, length):
    mean, stderr = aggregate(errors)
    return [mean] * length, [stderr] * length


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

handles_legend, labels_legend = [], []
current_methods = list(methods.keys())

for exp_id, exp in enumerate(exp_list):
    ax = axs[exp_id]
    ax.set_title(f"Experiment {exp_id + 1}", fontsize=20)
    x_vals = x_axis_values[exp]
    stats = {m: ([], []) for m in current_methods}

    if exp == "exp1":
        var_name = "n"
        for x in x_vals:
            for method in current_methods:
                mean, stderr = aggregate(load_errors(method, exp, var_name, x))
                stats[method][0].append(mean)
                stats[method][1].append(stderr)

    elif exp == "exp2":
        var_name = "r"
        for method in ["RRR", "SOFAR", "SRRR", "RSSVD", "SMART"]:
            for x in x_vals:
                mean, stderr = aggregate(load_errors(method, exp, var_name, x))
                stats[method][0].append(mean)
                stats[method][1].append(stderr)
        stats["SMARTCV"] = horizontal_stats(
            load_errors("SMARTCV", "exp1", "n", model_to_n[model_id][0]), len(x_vals))

    elif exp == "exp3":
        var_name = "rs"
        for method in ["RRR", "SOFAR", "SRRR", "RSSVD"]:
            fixed_errors = load_errors(method, "exp1", "n", model_to_n[model_id][0])
            stats[method] = horizontal_stats(fixed_errors, len(x_vals))
        for x in x_vals:
            mean, stderr = aggregate(load_errors("SMART", exp, var_name, x))
            stats["SMART"][0].append(mean)
            stats["SMART"][1].append(stderr)
        stats["SMARTCV"] = horizontal_stats(
            load_errors("SMARTCV", "exp1", "n", model_to_n[model_id][0]), len(x_vals))

    elif exp == "exp4":
        var_name = "sigma0"
        for method in ["RRR", "SOFAR", "SRRR", "RSSVD"]:
            fixed_errors = load_errors(method, "exp1", "n", model_to_n[model_id][0])
            stats[method] = horizontal_stats(fixed_errors, len(x_vals))
        for method in ["SMART", "SMARTCV"]:
            for x in x_vals:
                mean, stderr = aggregate(load_errors(method, exp, var_name, x))
                stats[method][0].append(mean)
                stats[method][1].append(stderr)

    for method in current_methods:
        means, stderrs = stats[method]
        line = ax.errorbar(
            x_vals, means, yerr=stderrs,
            fmt=markers[method], label=label_map.get(method, method),
            capsize=4, color=colors[method], linestyle='-',
            markersize=6, linewidth=1.5,
        )
        if exp_id == 0:
            handles_legend.append(line)
            labels_legend.append(label_map.get(method, method))

    ax.set_xlabel(x_axis_labels[exp], fontsize=16)
    ax.set_ylabel("Average Frobenius Norm Error", fontsize=16)

    if exp_id == 0:
        ax.set_xticks(x_vals)
    elif exp_id == 1:
        ax.set_xticks(np.arange(min(x_vals), max(x_vals) + 1, 1))
    elif exp_id == 2:
        ax.set_xticks(np.arange(min(x_vals), max(x_vals) + 1, 2))
    ax.tick_params(axis='x', labelsize=16)

    scale_factor = 1e2
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick * scale_factor:.2f}" for tick in yticks])
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.tick_params(axis='y', labelsize=16)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=ymax)
    ax.annotate(r"$\times 10^{-2}$", xy=(0, 1.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=16)

fig.suptitle(model_title, fontsize=24)
fig.legend(handles_legend, labels_legend, loc="upper center",
           bbox_to_anchor=(0.5, 0.07), ncol=6, frameon=False, fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

os.makedirs("fig", exist_ok=True)
plt.savefig(f"fig/simulation_{model_name}.pdf")
plt.close()
print(f"Saved fig/simulation_{model_name}.pdf")
