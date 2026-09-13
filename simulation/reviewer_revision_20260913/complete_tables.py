"""Make full-condition TeX risk tables from complete, independently audited CSV.

No fitting or array access. Every planned condition appears, including unavailable
methods. This supplements the figures, which necessarily show fewer comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


FAMILY = {
    "reference": "Reference", "source_noise": "Source noise",
    "containment": "Two-sided angle", "containment_side": "One-sided angle",
    "target_specific": "New directions", "diffuse_alignment": "Diffusion",
    "source_internal_gap": "Source internal gap", "target_internal_gap": "Target internal gap",
    "source_boundary_gap": "Source boundary", "target_boundary_gap": "Target boundary",
    "source_truncation": "Source dimension", "fitted_source": "Fitted source",
    "coefficient_close": "Coefficient-close", "spectral_shift": "Spectral shift",
    "fitted_relationship": "Fitted relationship", "sparse_stress": "Sparse stress",
}
GROUPS = (
    ("Source-subspace and initialization controls", (
        ("v2", "v2"), ("initializer_only", "Initializer"),
        ("source_subspace_rrr", "Source RRR"),
        ("source_subspace_ridge_rrr", "Source ridge"),
        ("oracle_subspace_rrr", "Clean oracle"))),
    ("Target-only and coefficient-transfer controls", (
        ("v2", "v2"), ("target_rrr", "Target RRR"),
        ("target_ridge_rrr", "Target ridge"),
        ("ridge_to_source", "Source-centered ridge"),
        ("source_target_mixture", "Mixture"),
        ("nuclear_contrast", "Nuclear contrast"))),
    ("Published raw-source comparison", (
        ("v2", "v2"), ("initializer_only", "Initializer"),
        ("source_subspace_ridge_rrr", "Source ridge"),
        ("nuclear_contrast", "Nuclear contrast"),
        ("park_two_stage_nr_external_validation", "Park holdout"))),
    ("Matched active-cap experiment", (
        ("v2", "Main v2"), ("v2_active_caps", "Active caps"),
        ("v2_cap_control", "Matched full caps"),
        ("initializer_only_active_caps", "Active-grid initializer"),
        ("initializer_only_cap_control", "Control initializer"))),
)


def tex(value):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%",
               "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(character, character) for character in str(value))


def compact(value):
    if value == 0:
        return "0"
    if abs(value) >= .001:
        return f"{value:.3g}"
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10**exponent:.2g}\\!\\times\\!10^{{{exponent}}}"


def summary(rows):
    values = [float(row["population_prediction_excess"]) for row in rows if row["status"] == "success"]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid risk in a successful audited row")
    return dict(n_planned=len(rows), status_counts=dict(Counter(row["status"] for row in rows)),
                n_success=len(values), mean=statistics.mean(values) if values else None,
                mcse=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)


def cell(value):
    if value is None:
        return r"\textemdash"
    if not value["n_success"]:
        return f"---\\,{{\\tiny 0/{value['n_planned']}}}"
    estimate = compact(value["mean"])
    uncertainty = compact(value["mcse"]) if value["mcse"] is not None else r"\mathrm{NA}"
    extra = f"\\,{{\\tiny[{value['n_success']}/{value['n_planned']}]}}" if value["n_success"] != value["n_planned"] else ""
    return f"${estimate}$\\newline{{\\tiny ($ {uncertainty} $)}}{extra}"


def build(root, output):
    root, output = Path(root), Path(output)
    plan_bytes = (root/"plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    audit = json.loads((root/"audit.json").read_text())
    if not (audit.get("audit_passed") and audit.get("all_planned_tasks_complete") and
            audit.get("plan_sha256") == hashlib.sha256(plan_bytes).hexdigest()):
        raise ValueError("Requires matching complete passing independent audit")
    rows = list(csv.DictReader((root/"per_replication.csv").open(newline="")))
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        identity = (int(row["task_index"]), row["method"])
        if identity in identities:
            raise ValueError("Duplicate task/method row")
        identities.add(identity)
        task = plan["tasks"][identity[0]]
        case = plan["cases"][task["case_index"]]
        if row["case_id"] != case["case_id"] or int(row["seed"]) != task["seed"]:
            raise ValueError("CSV identity disagrees with the audited plan")
        groups[row["case_id"], row["method"]].append(row)
    summaries = {key:summary(members) for key,members in groups.items()}
    case_counts = Counter(plan["cases"][task["case_index"]]["case_id"] for task in plan["tasks"])
    for (case_id, _), value in summaries.items():
        if value["n_planned"] != case_counts[case_id]:
            raise ValueError("Method rows do not retain every planned replication")
    output.parent.mkdir(parents=True,exist_ok=True)
    counts_text = ", ".join(str(value) for value in sorted(set(case_counts.values())))
    lines = [r"\section{Complete condition-level prediction results}",
        "Entries are mean population excess prediction risk per response, with Monte Carlo standard errors below in parentheses. The planned seed counts per condition are "+counts_text+r". Successful-only summaries show $[m/N]$ when fewer than $N$ outputs are available; a dashed cell with $0/N$ is an executed method with no eligible output. An em dash denotes a method outside that condition's protocol. All failure counts remain in the accompanying CSV. These are pointwise descriptive comparisons; the tables do not select a favorable subset of conditions.",
        r"The clean-source oracle is an information diagnostic and can have approximation bias when containment fails. Park receives raw source data and estimates an intercept; the other listed methods use the known-zero-intercept convention. Main v2 includes its target-RRR endpoint, so success of that wrapper does not imply successful transfer refinement."]
    newline = r" \\"
    for title, methods in GROUPS:
        cases = plan["cases"]
        if "Published" in title:
            cases = [case for case in cases if (case["case_id"], methods[-1][0]) in summaries]
        if "Matched" in title:
            cases = [case for case in cases if case["family"] == "sparse_stress"]
        if not cases:
            continue
        lines += [r"\clearpage", r"\subsection{"+tex(title)+"}", r"\begingroup\scriptsize",
                  r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}"]
        width = .61 / len(methods)
        spec = r"L{.04\linewidth}L{.22\linewidth}" + (f"L{{{width:.3f}\\linewidth}}"*len(methods))
        lines += [r"\begin{longtable}{"+spec+"}", r"\toprule",
                  "$n$ & Condition & "+" & ".join(tex(label) for _,label in methods)+newline,
                  r"\midrule\endhead"]
        previous_n = None
        for case in cases:
            if previous_n is not None and case["n_train"] != previous_n:
                lines.append(r"\midrule")
            previous_n = case["n_train"]
            name = FAMILY.get(case["family"],case["family"])+": "+str(case["level"])
            lines.append(str(case["n_train"])+" & "+tex(name)+" & "+" & ".join(
                cell(summaries.get((case["case_id"],method))) for method,_ in methods)+newline)
        lines += [r"\bottomrule\end{longtable}",r"\endgroup"]
    output.with_suffix('.tex').write_text("\n".join(lines)+"\n")
    payload = dict(plan_sha256=audit["plan_sha256"], conditions=len(plan["cases"]),
                   summaries=[dict(case_id=key[0],method=key[1],**value) for key,value in summaries.items()])
    output.with_suffix('.json').write_text(json.dumps(payload,indent=2)+"\n")
    return payload


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('output')
    args=parser.parse_args(); result=build(args.root,args.output)
    print(json.dumps(dict(conditions=result['conditions'],method_summaries=len(result['summaries']))))
