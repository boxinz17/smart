"""Freeze the added-comparator plan from the actual twelve reference plans."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import METHODS, RANK_INDEPENDENT, METHOD, read, sha, digest, immutable, require

REFERENCE = "/scratch2/mkolar/smart/runs/sparse-smart-v2-paper100-all-20260913T073736Z"

def build(root, references, manifest_hash, configuration):
    """Plan from metadata only. Preparation must verify every data alias."""
    groups, tasks, displays, group_lookup, task_lookup = [], [], [], {}, {}
    for reference in references:
        for case in reference["cases"]:
            key = tuple(case[k] for k in ("model_id", "n_train", "p", "q", "sigma0", "seed_id", "random_seed"))
            if key not in group_lookup:
                group = dict(group_id="data_" + digest(key)[:24], case=case,
                             reference_root=reference["root"])
                groups.append(group)
                group_lookup[key] = group["group_id"]
            gid = group_lookup[key]
            mapping = {}
            for method in METHODS:
                rank = None if method in RANK_INDEPENDENT else case["rank"]
                task_key = (gid, method, rank)
                if task_key not in task_lookup:
                    tid = len(tasks)
                    tasks.append(dict(task_id=tid, group_id=gid, method=method, rank=rank))
                    task_lookup[task_key] = tid
                mapping[method] = task_lookup[task_key]
            displays.append(dict(case=case, group_id=gid, reference_root=reference["root"], task_ids=mapping))
    plan = dict(schema_version=1, method=METHOD, root=str(root), source_manifest_sha256=manifest_hash,
                configuration=configuration, groups=groups, tasks=tasks, display_cases=displays,
                references=[{k:v for k,v in r.items() if k != "cases"} for r in references],
                n_groups=len(groups), n_tasks=len(tasks), n_display_cases=len(displays),
                methods=list(METHODS), selection="common validation MSE; no refit; deterministic first tie",
                alias_rule="same observed training/source/validation arrays; method-specific fitted rank only",
                information_policy="same saved target train plus 200 validation observations and observed source coefficient",
                inference_unit="seed within setting; aliases never increase replication count")
    plan["plan_fingerprint"] = digest(plan)
    return plan

def freeze(root, reference=REFERENCE):
    from .fitting import default_configuration
    root, reference = Path(root).resolve(), Path(reference).resolve()
    references = []
    for model in range(1, 4):
        for experiment in range(1, 5):
            folder = reference / f"model{model}-exp{experiment}"
            old = read(folder / "plan.json")
            require(old["method"] == "SparseSMARTv2Pilot", "wrong reference method")
            require(old["configuration"]["n_validation"] == 200, "different validation budget")
            require({c["seed_id"] for c in old["cases"]} == set(range(100)), "reference seed coverage differs")
            references.append(dict(root=str(folder), cases=old["cases"], plan_sha256=sha(folder / "plan.json"),
                preparation_sha256=sha(folder / "preparation.json"), summary_sha256=sha(folder / "summary/summary.json"),
                source_manifest_sha256=sha(folder / "source-manifest.json"),
                numerical_audit_sha256=sha(folder / "summary/numerical-audit.json"),
                numerical_audit_cases_sha256=sha(folder / "summary/numerical-audit-cases.jsonl")))
    plan = build(root, references, sha(root / "source-manifest.json"), default_configuration())
    require((plan["n_groups"], plan["n_tasks"], plan["n_display_cases"]) == (3000, 33000, 6600),
            "unexpected original campaign coverage")
    immutable(root / "plan.json", plan)
    contents = "".join(f"{i}\n" for i in range(plan["n_tasks"]))
    path = root / "work-items.tsv"
    if path.exists():
        require(path.read_text() == contents, "work items changed")
    else:
        path.write_text(contents)
    return plan

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference-root", default=REFERENCE, type=Path)
    args = parser.parse_args()
    plan = freeze(args.root, args.reference_root)
    print({k:plan[k] for k in ("root", "plan_fingerprint", "n_groups", "n_tasks", "n_display_cases")})

if __name__ == "__main__":
    main()
