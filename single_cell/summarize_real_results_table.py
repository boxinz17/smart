import os
import pickle
import numpy as np
from prettytable import PrettyTable

# Directories
result_dir = "./result"
fig_dir = "./fig"
os.makedirs(fig_dir, exist_ok=True)
table_path = os.path.join(fig_dir, "summary_statistics_table.txt")

# Initialize result storage
results_dict = {
    # Target-only methods
    "Target Only (RRR)": [],
    "Target Only (SRRR)": [],
    "Target Only (RSSVD)": [],
    "Target Only (SOFAR)": [],
    "Target Only (Lasso)": [],
    "Target Only (OLS)": [],
    "Target Only (Ridge)": [],
    # Source-only methods
    "Source Only (RRR)": [],
    "Source Only (SRRR)": [],
    "Source Only (RSSVD)": [],
    "Source Only (SOFAR)": [],
    "Source Only (Lasso)": [],
    "Source Only (OLS)": [],
    "Source Only (Ridge)": [],
    # SMART methods
    "SMART (RRR)": [],
    "SMART (SRRR)": [],
    "SMART (RSSVD)": [],
    "SMART (SOFAR)": [],
    "SMART (Ridge)": [],
    "SMART (Lasso)": [],
    "SMART (OLS)": []
}

# Mapping for SMART methods
method_name_map = {
    "RRR": "SMART (RRR)",
    "SRRR": "SMART (SRRR)",
    "RSSVD": "SMART (RSSVD)",
    "SOFAR": "SMART (SOFAR)",
    "RIDGE": "SMART (Ridge)",
    "LASSO": "SMART (Lasso)",
    "OLS": "SMART (OLS)"
}

# Mapping from source matrix type to display name
source_only_display_map = {
    "RRR": "Source Only (RRR)",
    "SRRR": "Source Only (SRRR)",
    "RSSVD": "Source Only (RSSVD)",
    "SOFAR": "Source Only (SOFAR)",
    "LASSO": "Source Only (Lasso)",
    "OLS": "Source Only (OLS)",
    "RIDGE": "Source Only (Ridge)"
}

# Mapping for target-only filename prefixes
target_only_filename_map = {
    "baseline_target_only_rrr": "Target Only (RRR)",
    "baseline_target_only_srrr": "Target Only (SRRR)",
    "baseline_target_only_rssvd": "Target Only (RSSVD)",
    "baseline_target_only_sofar": "Target Only (SOFAR)",
    "baseline_target_only_lasso": "Target Only (Lasso)",
    "baseline_target_only_ols": "Target Only (OLS)",
    "baseline_target_only_ridge": "Target Only (Ridge)"
}

# Parse results
for filename in os.listdir(result_dir):
    if not filename.endswith(".pkl"):
        continue

    with open(os.path.join(result_dir, filename), "rb") as f:
        result = pickle.load(f)
        method_key = result.get("method_name", "").upper().strip()

        # SMART methods
        if method_key in method_name_map:
            display_name = method_name_map[method_key]
            results_dict[display_name].append(result["frob_error"])

        # Source-only methods
        elif method_key == "SOURCE_ONLY":
            sm_type = result.get("source_matrix_type", "").upper().strip()
            if not sm_type:
                print(f"Skipping file due to missing source_matrix_type: {filename}")
                continue

            display_name = source_only_display_map.get(sm_type)
            if display_name and display_name in results_dict:
                results_dict[display_name].append(result["frob_error"])
            else:
                print(f"Skipping file with unknown source matrix type: {sm_type} (from {filename})")

        # Target-only methods: infer from filename
        else:
            fname_lower = filename.lower()
            matched = False
            for prefix, display_name in target_only_filename_map.items():
                if fname_lower.startswith(prefix):
                    results_dict[display_name].append(result["frob_error"])
                    matched = True
                    break
            if not matched:
                print(f"Skipping unrecognized file: {filename}")

# Define display order
display_order = [
    "Target Only (RRR)",
    "Target Only (SRRR)",
    "Target Only (RSSVD)",
    "Target Only (SOFAR)",
    "Target Only (Lasso)",
    "Target Only (OLS)",
    "Target Only (Ridge)",
    "Source Only (RRR)",
    "Source Only (SRRR)",
    "Source Only (RSSVD)",
    "Source Only (SOFAR)",
    "Source Only (Lasso)",
    "Source Only (OLS)",
    "Source Only (Ridge)",
    "SMART (RRR)",
    "SMART (SRRR)",
    "SMART (RSSVD)",
    "SMART (SOFAR)",
    "SMART (Ridge)",
    "SMART (Lasso)",
    "SMART (OLS)"
]

# Create summary table
table = PrettyTable()
table.field_names = ["Method", "Mean", "SE", "Median", "Q1", "Q3"]

for method in display_order:
    values = results_dict.get(method, [])
    if len(values) == 0:
        continue

    values = np.array(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    median = np.median(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    table.add_row([
        method,
        f"{mean:.4f}",
        f"{se:.4f}",
        f"{median:.4f}",
        f"{q1:.4f}",
        f"{q3:.4f}"
    ])

# Print and save the table
print(table)
with open(table_path, "w") as f:
    f.write(str(table))

print(f"\nSummary table saved to: {table_path}")
