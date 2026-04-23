import os
import pickle
import matplotlib.pyplot as plt

# Directories
result_dir = "./result"
fig_dir = "./fig"
os.makedirs(fig_dir, exist_ok=True)

# Initialize containers
errors_target_only_rrr = []
errors_source_only_rrr = []
errors_smartcv = {
    "RRR": [],
    "SRRR": [],
    "RSSVD": [],
    "SOFAR": [],
    "RIDGE": [],
    "LASSO": [],
    "OLS": []
}

# Parse result files
for filename in os.listdir(result_dir):
    if not filename.endswith(".pkl"):
        continue

    filepath = os.path.join(result_dir, filename)
    with open(filepath, "rb") as f:
        result = pickle.load(f)

    method = result.get("method_name", "").upper()
    fname_lower = filename.lower()

    # Target-only RRR
    if fname_lower.startswith("baseline_target_only_rrr"):
        errors_target_only_rrr.append(result["frob_error"])

    # Source-only RRR
    elif method == "SOURCE_ONLY" and "source_only_rrr" in fname_lower:
        errors_source_only_rrr.append(result["frob_error"])

    # SMARTCV methods
    elif method in errors_smartcv:
        errors_smartcv[method].append(result["frob_error"])

# Combine data in specified order
labels = [
    "Target Only (RRR)",
    "Source Only (RRR)",
    "SMART (RRR)",
    "SMART (SRRR)",
    "SMART (RSSVD)",
    "SMART (SOFAR)",
    "SMART (Ridge)",
    "SMART (Lasso)",
    "SMART (OLS)"
]
data = [
    errors_target_only_rrr,
    errors_source_only_rrr,
    errors_smartcv["RRR"],
    errors_smartcv["SRRR"],
    errors_smartcv["RSSVD"],
    errors_smartcv["SOFAR"],
    errors_smartcv["RIDGE"],
    errors_smartcv["LASSO"],
    errors_smartcv["OLS"]
]

# Create boxplot
plt.figure(figsize=(12, 6))
plt.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
plt.ylabel("Average Frobenius Prediction Error")
plt.title("SMART vs. Baselines")
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(fig_dir, "smart_vs_baselines_boxplot.pdf"), dpi=300)
plt.show()
