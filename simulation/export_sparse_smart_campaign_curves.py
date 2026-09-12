"""Export audited seed-level campaign results and descriptive V1 comparisons.

This reads only the compact campaign summary and the saved paper reference.
It never reads raw fitting shards, fits models, or chooses a cap by test error.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from paper_reference import DEFAULT_REFERENCE, METRIC, read_reference
from summarize_sparse_smart_campaign import _configuration, _digest, _read_json


AXES = {"exp1": ("n", "Training sample size"),
        "exp2": ("target_rank", "Fitted rank"),
        "exp3": ("source_rank", "Source rank"),
        "exp4": ("sigma0", "Source noise SD")}
POPULATIONS = ("available_results", "complete_grid", "excluding_permitted_missing")
METRICS = ("coefficient_error", "validation_mse", "selected_iteration")
CAVEAT = ("Descriptive comparison with digitized V1 means and standard errors; no paired seed data. "
          "SparseSMART uses 100 additional independent validation observations to tune parameters and iteration. "
          "V1 SMART uses BIC and includes adaptive-rank references. Legacy Experiment 2 R baseline scripts use "
          "n=200, whereas SparseSMART uses n=300 for Model II and n=500 for Model III. The saved figures alone "
          "do not establish all sample sizes/protocols for Model II/III experiments 2–4. These overlays do not "
          "establish a matched-protocol comparison or a fair ranking.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value, label):
    require(type(value) is bool, f"Invalid {label}")
    return value


def nonnegative(value, label, *, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), f"Invalid {label}")
    require(math.isfinite(value) and value >= 0, f"Invalid {label}")
    return value


def statistics_for(values):
    values = list(values)
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else None
    return dict(n=n, mean=statistics.fmean(values) if n else None, sample_sd=sd,
                mcse=sd / math.sqrt(n) if sd is not None else None)


def reference_key(row):
    return int(row["model_id"]), row["experiment"], float(row["x"])


def _case_records(row, schema):
    """Validate the compact envelope before exposing any selected measurement."""
    require(row["model"] in ("model1", "model2", "model3") and row["experiment"] in AXES,
            "Invalid model/experiment")
    model = int(row["model"][-1]) - 1
    seed = nonnegative(row["seed_id"], "seed ID", integer=True)
    random_seed = nonnegative(row["random_seed"], "random seed", integer=True)
    config = _configuration(row["configuration"])
    require(_digest(config) == row["configuration_fingerprint"], "Configuration fingerprint mismatch")
    setting = row["simulation_setting"]
    require(isinstance(setting, dict) and setting.get("suffix") == row["setting"], "Setting identity mismatch")
    x = nonnegative(setting.get(AXES[row["experiment"]][0]), "x coordinate")
    state = row["record_state"]
    require(state in ("published", "missing", "invalid"), "Invalid record state")
    applicable = boolean(row["applicable"], "applicability")
    usable = boolean(row["execution_usable"], "execution usability")
    policy_usable, missing_count = usable, 0
    if schema == 3:
        policy_usable = boolean(row["policy_execution_usable"], "policy execution usability")
        has_missing = boolean(row["has_permitted_missing_tasks"], "permitted missing flag")
        missing = row.get("missing_tuning")
        if state == "published":
            require(isinstance(missing, dict), "Published policy case omits missing-tuning metadata")
            unavailable = missing.get("unavailable_tasks")
            allowed = missing.get("allowed_tasks")
            require(isinstance(unavailable, list) and isinstance(allowed, list), "Invalid missing-task lists")
            require(all(item in allowed for item in unavailable), "Unavailable task was not permitted")
            ids = sorted(i for task in unavailable for i in task["grid_candidate_ids"])
            require(ids == missing.get("unavailable_grid_candidate_ids") and len(ids) == len(set(ids)),
                    "Invalid unavailable candidate IDs")
            planned = math.prod(len(config[key]) for key in ("init_penalties", "penalties_u", "penalties_v"))
            require(missing.get("planned_candidate_count") == planned and
                    missing.get("available_candidate_count") == planned - len(ids), "Invalid candidate counts")
            require(has_missing == bool(unavailable), "Missing-task flag contradicts metadata")
            missing_count = len(ids)
        else:
            require(not has_missing and not policy_usable, "Unpublished case cannot be policy usable")
    else:
        require(not row.get("has_permitted_missing_tasks") and not row.get("missing_tuning"),
                "Missing-task exception requires schema 3")
    scientific, summary, execution = row.get("scientific"), row.get("summary"), row.get("execution")
    audited = False
    status = state
    if state == "published":
        require(isinstance(scientific, dict) and isinstance(execution, dict), "Incomplete published envelope")
        audited = boolean(scientific["audit_passed"], "scientific audit flag")
        status = scientific["status"]
        require(status in ("complete", "partial", "inapplicable", "all_candidates_failed", "invalid"),
                "Unknown scientific status")
        fit = boolean(execution["fit_success"], "fit success")
        launch = boolean(execution["launch_success"], "launch success")
        require(usable == (fit and launch and not execution["issues"]), "Contradictory execution usability")
        if schema == 3:
            require(policy_usable == execution.get("available_tasks_success"), "Contradictory policy usability")
            require(not policy_usable or usable or missing_count > 0,
                    "Policy cannot excuse execution failure without permitted missing tasks")
        if audited:
            require(not scientific["issues"] and isinstance(summary, dict), "Failed audit cannot expose summary")
            for key in ("model", "experiment", "setting", "seed_id", "applicable"):
                require(summary.get(key) == row[key], f"Summary {key} mismatch")
            require(summary["status"] == status and _configuration(summary["configuration"]) == config,
                    "Summary configuration/status mismatch")
            require(applicable == (status != "inapplicable"), "Applicability/status mismatch")
            if not applicable:
                require(not summary["caps"], "Inapplicable case has candidate results")
            else:
                require([cap["iteration_budget"] for cap in summary["caps"]] == config["iteration_budgets"],
                        "Cap scope differs from configured budgets")
                final = summary["caps"][-1]
                expected = "complete" if final["coverage_complete"] else "partial" if final["success"] else "all_candidates_failed"
                require(status == expected, "Status contradicts final budget coverage")
                require(scientific["budget_coverage_complete"] == final["coverage_complete"],
                        "Scientific coverage contradicts final cap")
        else:
            require(summary is None, "Failed scientific audit exposes selected measurements")
    else:
        require(not usable and not policy_usable and summary is None, "Unpublished case exposes selected results")
    grid_size = math.prod(len(config[key]) for key in ("init_penalties", "penalties_u", "penalties_v"))
    base = dict(case_key=row["case_key"], model_id=model, model=row["model"], experiment=row["experiment"],
                setting=row["setting"], x=x, seed_id=seed, random_seed=random_seed,
                n_train=setting.get("n"), n_validation=config.get("n_validation"), p=setting.get("p"),
                q=setting.get("q"), applicable=applicable, record_state=state, scientific_status=status,
                audit_passed=audited, execution_usable=usable, policy_execution_usable=policy_usable,
                permitted_missing_candidates=missing_count, planned_candidate_count=grid_size,
                available_candidate_count=grid_size - missing_count if audited and applicable else None,
                configuration_fingerprint=row["configuration_fingerprint"],
                configured_max_budget=config["iteration_budgets"][-1],
                implementation_fingerprint=summary.get("implementation_fingerprint") if summary else None,
                audit_sha256=(row.get("provenance") or {}).get("audit_sha256"),
                receipt_sha256=(row.get("provenance") or {}).get("receipt_sha256"),
                source_fingerprint=row.get("source_fingerprint"), plan_fingerprint=row.get("plan_fingerprint"),
                inapplicability_reason=row.get("inapplicability_reason"))
    caps = {cap["iteration_budget"]: cap for cap in summary["caps"]} if audited and applicable else {}
    records = []
    for index, budget in enumerate(config["iteration_budgets"]):
        cap = caps.get(budget)
        result = dict(base, iteration_budget=budget, is_configured_max=budget == base["configured_max_budget"],
                      selected_available=False, coverage_complete=False, eligible=False,
                      init_penalty=None, penalty_u=None, penalty_v=None, grid_candidate_id=None,
                      winner_origin_budget=None, selected_checkpoint=None, optimizer_converged=None,
                      selected_converged=None, **{name: None for name in METRICS})
        if cap is not None:
            success = boolean(cap["success"], "cap success")
            complete = boolean(cap["coverage_complete"], "cap coverage")
            unresolved = cap.get("unresolved_candidate_count", len(cap.get("unresolved_candidate_ids", [])))
            require(type(unresolved) is int and 0 <= unresolved <= grid_size and
                    complete == (unresolved == 0) and cap["reached_candidates"] == grid_size - unresolved,
                    "Invalid cap coverage counts")
            require(not complete or success, "Complete cap has no selected estimate")
            require(not missing_count or not complete, "Permitted missing tasks cannot certify complete coverage")
            result.update(selected_available=success, coverage_complete=complete,
                          eligible=success and policy_usable, unresolved_candidate_count=unresolved)
            if success:
                for name in METRICS:
                    result[name] = nonnegative(cap[name], name, integer=name == "selected_iteration")
                checkpoint = nonnegative(cap["selected_checkpoint"], "selected checkpoint", integer=True)
                require(result["selected_iteration"] <= checkpoint <= budget, "Selected checkpoint exceeds budget")
                candidate = nonnegative(cap["winner_candidate_id"], "winner candidate", integer=True)
                require(candidate < grid_size * (index + 1), "Winner outside tuning grid or available budgets")
                origin, grid_id = divmod(candidate, grid_size)
                if missing_count:
                    require(grid_id not in row["missing_tuning"]["unavailable_grid_candidate_ids"],
                            "Missing candidate cannot win")
                init, remainder = divmod(grid_id, len(config["penalties_u"]) * len(config["penalties_v"]))
                u, v = divmod(remainder, len(config["penalties_v"]))
                result.update(init_penalty=config["init_penalties"][init], penalty_u=config["penalties_u"][u],
                              penalty_v=config["penalties_v"][v], grid_candidate_id=grid_id,
                              winner_origin_budget=config["iteration_budgets"][origin], selected_checkpoint=checkpoint,
                              optimizer_converged=boolean(cap["optimizer_converged"], "optimizer convergence"),
                              selected_converged=boolean(cap["selected_converged"], "selected convergence"))
                require(not result["selected_converged"] or result["optimizer_converged"],
                        "Selected convergence contradicts optimizer status")
            else:
                require(all(cap.get(name) is None for name in (*METRICS, "winner_candidate_id", "selected_checkpoint")),
                        "Unsuccessful cap contains selected measurements")
        records.append(result)
    return records


def build_tables(report, references):
    """Compute seed-weighted statistics without pooling different configurations."""
    require(report.get("schema_version") in (2, 3) and report.get("method") == "SparseSMARTCampaignSummary",
            "Unsupported campaign summary")
    cases = report.get("cases")
    require(isinstance(cases, list) and cases, "Campaign summary has no cases")
    require(report.get("expected_cases") == len(cases), "Case count mismatch")
    schema, rows, identities, random_ids, keys = report["schema_version"], [], set(), set(), set()
    ref_dimensions = {reference_key(ref): (int(ref["p"]), int(ref["q"])) for ref in references}
    grouped, configurations, implementations = defaultdict(list), {}, defaultdict(set)
    for case in cases:
        records = _case_records(case, schema)
        first = records[0]
        key = first["model_id"], first["experiment"], float(first["x"])
        require(key in ref_dimensions, f"No exact paper reference x match: {key}")
        require((first["p"], first["q"]) == ref_dimensions[key], "Model dimensions differ from paper reference")
        require(case["case_key"] not in keys, "Duplicate case key")
        keys.add(case["case_key"])
        identity, random_identity = (*key, first["seed_id"]), (*key, first["random_seed"])
        require(identity not in identities and random_identity not in random_ids, "Duplicate seed at a setting")
        identities.add(identity)
        random_ids.add(random_identity)
        config_key = (_digest(case["simulation_setting"]), first["configuration_fingerprint"])
        require(key not in configurations or configurations[key] == config_key,
                "Inconsistent setting/configuration among seeds at the same x")
        configurations[key] = config_key
        if first["implementation_fingerprint"]:
            implementations[key].add(first["implementation_fingerprint"])
        for row in records:
            grouped[(*key, row["iteration_budget"])].append(row)
        rows.extend(records)
    require(all(len(values) <= 1 for values in implementations.values()), "Inconsistent implementations among seeds")
    statistics_rows = []
    for key, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda row: row["seed_id"])
        first = candidates[0]
        for population in POPULATIONS:
            selected = [row for row in candidates if row["eligible"] and
                        (population != "complete_grid" or row["coverage_complete"]) and
                        (population != "excluding_permitted_missing" or not row["permitted_missing_candidates"])]
            out = {name: first[name] for name in ("model_id", "model", "experiment", "setting", "x", "iteration_budget",
                         "configured_max_budget", "is_configured_max", "configuration_fingerprint", "n_train", "n_validation")}
            out.update(population=population, n=len(selected), expected=len(candidates),
                       expected_applicable=sum(row["applicable"] for row in candidates),
                       inapplicable=sum(not row["applicable"] for row in candidates),
                       unavailable=sum(row["applicable"] and not row["eligible"] for row in candidates),
                       partial_count=sum(not row["coverage_complete"] for row in selected),
                       permitted_missing_seed_count=sum(bool(row["permitted_missing_candidates"]) for row in selected),
                       seed_ids=[row["seed_id"] for row in selected],
                       optimizer_converged_count=sum(row["optimizer_converged"] for row in selected),
                       selected_converged_count=sum(row["selected_converged"] for row in selected))
            for metric in METRICS:
                out.update({f"{metric}_{stat}": value for stat, value in statistics_for(row[metric] for row in selected).items()
                            if stat != "n"})
            statistics_rows.append(out)
    selected_caps = [row for row in statistics_rows if row["is_configured_max"]]
    selected_lookup = {(row["model_id"], row["experiment"], float(row["x"])): row for row in selected_caps
                       if row["population"] == "available_results"}
    models = sorted({row["model_id"] for row in rows})
    comparison = []
    for ref in references:
        key = reference_key(ref)
        if key[0] not in models:
            continue
        selected = selected_lookup.get(key)
        comparison.append(dict(model_id=key[0], experiment=key[1], x=key[2], method=ref["method"],
            method_label=ref["method_label"], paper_mean=float(ref["mean"]), paper_se=float(ref["se"]),
            paper_repetitions=int(ref["paper_repetitions"]), paper_source=ref["source_pdf"],
            paper_horizontal_reference=ref["is_horizontal_reference"],
            sparse_smart_mean=selected["coefficient_error_mean"] if selected else None,
            sparse_smart_mcse=selected["coefficient_error_mcse"] if selected else None,
            sparse_smart_n=selected["n"] if selected else 0,
            sparse_smart_partial=selected["partial_count"] if selected else 0,
            configured_max_budget=selected["configured_max_budget"] if selected else None,
            comparison_status="available_descriptive" if selected and selected["n"] else
                              "inapplicable" if selected and not selected["expected_applicable"] else "unavailable"))
    return dict(per_seed=sorted(rows, key=lambda row: (row["model_id"], row["experiment"], row["x"], row["seed_id"],
                                                     row["iteration_budget"])),
                budget_statistics=statistics_rows, performance_curves=selected_caps, paper_comparison=comparison)


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _plot_comparison_caveat(model):
    text = "Descriptive overlay; tuning differs. SparseSMART uses 100 additional validation observations."
    if model in (1, 2):
        text += (f"\nExperiment 2 R baselines use n=200 in legacy scripts; "
                 f"SparseSMART uses n={300 if model == 1 else 500}.")
    return text


def _missing_task_caption(rows):
    missing = sorted((row for row in rows if row["is_configured_max"] and row["eligible"]
                      and row["permitted_missing_candidates"]), key=lambda row: (row["x"], row["seed_id"]))
    if not missing:
        return ""
    if len(missing) > 2:
        return f"Permitted missing fits affect {len(missing)} seeds; see per-seed table."
    return "Permitted missing fits: " + "; ".join(
        f"x={row['x']:g}, seed {row['seed_id']} retains "
        f"{row['available_candidate_count']}/{row['planned_candidate_count']} candidates" for row in missing) + "."


def plot_curves(tables, references, output_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({row["model_id"] for row in tables["per_seed"]})
    paths = []
    styles = {"RRR": ("#999999", "o"), "SRRR": ("#b8860b", "v"), "SOFAR": ("#8c63a6", "s"),
              "RSSVD": ("#51a580", "^"), "SMART_fixed": ("#4799bc", "D"), "SMART": ("#d87840", "P")}
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), layout="constrained")
        selected = [row for row in tables["performance_curves"] if row["model_id"] == model
                    and row["population"] == "available_results"]
        default_cap = Counter(row["configured_max_budget"] for row in selected).most_common(1)[0][0]
        exceptions = [f"{row['experiment']} x={row['x']:g}: {row['configured_max_budget']:,}"
                      for row in selected if row["configured_max_budget"] != default_cap]
        cap_label = f"Configured caps: {default_cap:,}" + ("; " + ", ".join(exceptions) if exceptions else "")
        for experiment, axis in zip(AXES, axes.flat):
            refs = [row for row in references if int(row["model_id"]) == model and row["experiment"] == experiment]
            xs = sorted({float(row["x"]) for row in refs})
            for method, (color, marker) in styles.items():
                points = sorted((r for r in refs if r["method"] == method), key=lambda r: float(r["x"]))
                axis.errorbar([float(row["x"]) for row in points], [float(row["mean"]) for row in points],
                              yerr=[float(row["se"]) for row in points], label=f"V1 {points[0]['method_label']}",
                              color=color, marker=marker, markersize=3.5, linewidth=1.1, alpha=.75, capsize=2)
            own = [row for row in tables["performance_curves"] if row["model_id"] == model and row["experiment"] == experiment]
            for population, label, color, marker, line in (
                    ("available_results", "SparseSMART available", "#142b67", "o", "-"),
                    ("complete_grid", "SparseSMART complete grid", "#111111", "x", "--")):
                by_x = {float(row["x"]): row for row in own if row["population"] == population}
                values = [by_x.get(x, {}) for x in xs]
                axis.errorbar(xs, [row.get("coefficient_error_mean") if row.get("n", 0) else math.nan for row in values],
                              yerr=[row.get("coefficient_error_mcse") if row.get("coefficient_error_mcse") is not None
                                    else math.nan for row in values], color=color,
                              marker=marker, linestyle=line, label=label, linewidth=2, capsize=3, markersize=5)
            available = {float(row["x"]): row for row in own if row["population"] == "available_results"}
            if experiment != "exp4":
                axis.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=9)
            counts = sorted({row["n"] for row in available.values() if row["n"]})
            partial = [f"{x:g}: {row['partial_count']}" for x, row in sorted(available.items()) if row["partial_count"]]
            gap = [f"{x:g}" for x in xs if not available.get(x, {}).get("n")]
            subtitle = "Available seeds/point: " + (", ".join(map(str, counts)) if counts else "none")
            if partial:
                subtitle += "; partial (x: n): " + ", ".join(partial)
            if gap:
                subtitle += "\nNo SparseSMART estimate at x=" + ", ".join(gap)
            missing_caption = _missing_task_caption(
                row for row in tables["per_seed"] if row["model_id"] == model and row["experiment"] == experiment)
            if missing_caption:
                subtitle += "\n" + missing_caption
            axis.set_xlabel(AXES[experiment][1])
            axis.set_title(f"Experiment {experiment[-1]}\n{subtitle}", loc="left", fontsize=10)
            axis.set_ylabel(r"$\|\widehat C-C^*\|_F/\sqrt{pq}$")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
            axis.grid(axis="y", color=".9")
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
        fig.suptitle(f"Model {('I', 'II', 'III')[model]} — validation-selected SparseSMART and published V1 curves\n"
                     f"{cap_label}. Error bars: mean ± MCSE (new) / SE (V1).\n"
                     "Partial = incomplete grid budget coverage. Complete-grid line excludes those seeds (sensitivity only).\n"
                     + _plot_comparison_caveat(model),
                     fontsize=11)
        for suffix in ("png", "pdf"):
            path = output_root / f"model-{model+1}-performance.{suffix}"
            fig.savefig(path, dpi=180 if suffix == "png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def export_campaign(summary_path, output_root, *, reference_path=DEFAULT_REFERENCE, manuscript_root=None, plots=True):
    summary_path, output_root = Path(summary_path).absolute(), Path(output_root).absolute()
    report, sha = _read_json(summary_path)
    references, verification = read_reference(reference_path, manuscript_root=manuscript_root)
    tables = build_tables(report, references)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, rows in tables.items():
        path = output_root / f"{name}.csv"
        _csv(path, rows)
        outputs.append(path)
    if plots:
        outputs.extend(plot_curves(tables, references, output_root))
    final_seeds = [row for row in tables["per_seed"] if row["is_configured_max"]]
    manifest = dict(schema_version=1, method="SparseSMARTCampaignCurves", created_utc=datetime.now(timezone.utc).isoformat(),
        input_summary=dict(path=str(summary_path), sha256=sha, schema_version=report["schema_version"],
                           index_fingerprint=report["index_fingerprint"], index_sha256=report["index_sha256"]),
        exporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric=METRIC, metric_transform="none", paper_reference_verification=verification,
        selection="For each seed and setting, use the validation-selected checkpoint at the configured maximum cap. "
                  "No selection of cap or tuning parameters by coefficient error.",
        eligibility="Published, scientifically audited, execution-usable successful selected estimates. Schema 3 applies "
                    "the explicitly permitted missing-task policy. Valid partial-budget estimates are included and flagged.",
        populations=dict(available_results="Union of all eligible complete and partial grid results, pooled across individual seeds.",
                         complete_grid="Sensitivity restricted to eligible seeds with complete grid coverage at that cap.",
                         excluding_permitted_missing="Sensitivity excluding cases with actually missing, explicitly permitted tasks."),
        uncertainty="Sample SD uses ddof=1; MCSE = sample SD / sqrt(n). Both are unavailable at n<2. "
                    "Published error bars are digitized standard errors, not SD or independent new replicates.",
        comparison_limitations=CAVEAT,
        validation_mse_interpretation="Reused for tuning; not independent test error.",
        final_status_counts=dict(Counter(row["scientific_status"] for row in final_seeds)),
        eligible_final_cases=sum(row["eligible"] for row in final_seeds),
        excluded_final_cases=[dict(case_key=row["case_key"], applicable=row["applicable"], record_state=row["record_state"],
                                  scientific_status=row["scientific_status"]) for row in final_seeds if not row["eligible"]],
        outputs=[dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in outputs])
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_root / "README.md").write_text(
        "# SparseSMART campaign performance curves\n\n" + CAVEAT + "\n\n"
        "The main available-results curve combines audited, execution-usable complete and partial cases at each "
        "setting's configured maximum iteration cap. It selects checkpoints by validation MSE, not coefficient error. "
        "The dashed complete-grid curve is a sensitivity analysis; changing seed counts can change its mean. "
        "Grid completion is separate from optimizer convergence. Missing/inapplicable points are gaps, never zeros.\n\n"
        "`per_seed.csv` preserves all budgets, selected penalties/checkpoints and source/audit identities. "
        "`budget_statistics.csv` reports each cap; `performance_curves.csv` retains the configured maximum cap. "
        "`paper_comparison.csv` supplies the digitized V1 means/SE and aligned available SparseSMART means/MCSE. "
        "The tables also include a sensitivity excluding explicitly permitted missing-task cases. "
        "CSV blanks mean unavailable; n=0 is an empty population. Sample SD and MCSE require at least two seeds.\n\n"
        "The metric is the saved unsquared coefficient error, ||C_hat-C*||_F/sqrt(p*q), with no rescaling. "
        "Plots use numeric x coordinates. Extra digits in digitized references do not imply extra accuracy. "
        "Provenance and output hashes are recorded in `manifest.json`. Raw fitting data remain on Discovery.\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--manuscript-root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_campaign(args.summary, args.output_root, reference_path=args.reference,
                                 manuscript_root=args.manuscript_root, plots=not args.no_plots)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Curve export failed: {error}\n")
    print(json.dumps(dict(output_root=str(args.output_root.absolute()), eligible_final_cases=result["eligible_final_cases"],
                         final_status_counts=result["final_status_counts"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
