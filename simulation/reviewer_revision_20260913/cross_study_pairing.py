"""Metadata-only pairing audit for completed main and operational-rank studies.

Reads JSON and CSV receipts, never numerical arrays, responses, or fit code.
Run after both independent numerical audits pass. This verifies equality of
their recorded typed-array fingerprints; it does not recompute those hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pairing(main_root: Path, rank_root: Path) -> dict:
    main_root, rank_root = Path(main_root), Path(rank_root)
    issues, inputs, pairs = [], {}, []

    def check(condition, message):
        if not condition:
            issues.append(message)

    def read_json(path):
        inputs[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def task_map(plan, label):
        mapping = {}
        check(plan.get("n_tasks") == len(plan["tasks"]), label + ": task count")
        check(plan.get("n_cases") == len(plan["cases"]), label + ": case count")
        for index, task in enumerate(plan["tasks"]):
            case = plan["cases"][task["case_index"]]
            key = (case["case_id"], task["seed"])
            check(key not in mapping, label + ": duplicate planned identity " + str(key))
            mapping[key] = (index, task, case)
        return mapping

    try:
        main_plan = read_json(main_root / "plan.json")
        rank_plan = read_json(rank_root / "rank-plan.json")
        rank_plan_digest = inputs[str(rank_root / "rank-plan.json")]
        main_audit = read_json(main_root / "audit.json")
        rank_audit = read_json(rank_root / "rank-audit.json")
        main_manifest = read_json(main_root / "source-manifest.json")
        rank_manifest = read_json(rank_root / "source-manifest.json")
        main_map, rank_map = task_map(main_plan, "main"), task_map(rank_plan, "rank")
        for label, root, plan, audit, plan_name in (
            ("main", main_root, main_plan, main_audit, "plan.json"),
            ("rank", rank_root, rank_plan, rank_audit, "rank-plan.json"),
        ):
            check(audit.get("audit_passed") is True, label + ": audit did not pass")
            check(audit.get("all_planned_tasks_complete") is True,
                  label + ": incomplete numerical audit")
            check(audit.get("issues") == [], label + ": numerical audit has issues")
            check(audit.get("plan_sha256") == sha256(root / plan_name),
                  label + ": numerical audit is not bound to this plan")
            audit_tasks = audit.get("tasks", [])
            check(len(audit_tasks) == len(plan["tasks"]), label + ": audit task coverage")
            seen = set()
            for receipt in audit_tasks:
                index = receipt["task_index"]
                check(index not in seen, label + ": duplicate audit task index")
                seen.add(index)
                check(receipt.get("state") == "complete", label + ": incomplete audit task")
                if not isinstance(index, int) or not 0 <= index < len(plan["tasks"]):
                    issues.append(label + ": invalid audit task index")
                    continue
                task = plan["tasks"][index]
                case = plan["cases"][task["case_index"]]
                check(receipt.get("case_id") == case["case_id"], label + ": audit case identity")
                if "seed" in receipt:
                    check(receipt["seed"] == task["seed"], label + ": audit seed identity")
            check(seen == set(range(len(plan["tasks"]))), label + ": missing audit indices")

        check(bool(rank_plan.get("code_sha256")), "rank: missing executed-source inventory")
        for name, digest in rank_plan.get("code_sha256", {}).items():
            for label, manifest in (("main", main_manifest), ("rank", rank_manifest)):
                check(manifest["files"].get(name) == digest,
                      label + ": generation/fitting source hash mismatch: " + name)

        csv_path = main_root / "per_replication.csv"
        inputs[str(csv_path)] = sha256(csv_path)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        main_rows = defaultdict(list)
        for row in rows:
            key = (row["case_id"], int(row["seed"]))
            main_rows[key].append(row)
        check(set(main_rows) == set(main_map), "main CSV: exact planned identity coverage")
        fingerprint_fields = ("fit_data_fingerprint", "truth_fingerprint")
        main_fingerprints = {}
        for key, (index, task, _) in main_map.items():
            group = main_rows[key]
            methods = [row["method"] for row in group]
            check(sorted(methods) == sorted(task["expected_methods"]),
                  "main CSV: method inventory for " + str(key))
            check(all(int(row["task_index"]) == index for row in group),
                  "main CSV: task index for " + str(key))
            values = {}
            for field in fingerprint_fields:
                digests = {row.get(field, "") for row in group}
                check(len(digests) == 1 and all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests),
                      "main CSV: empty/inconsistent/invalid " + field + " for " + str(key))
                values[field] = next(iter(digests)) if len(digests) == 1 else None
            main_fingerprints[key] = values

        expected_paths = {rank_root / "rank_tasks" / f"task-{i:06d}" / "result.json"
                          for i in range(len(rank_plan["tasks"]))}
        actual_paths = set((rank_root / "rank_tasks").glob("task-*/result.json"))
        check(actual_paths == expected_paths, "rank: missing or extra result receipts")
        for key, (rank_index, task, case) in rank_map.items():
            before = len(issues)
            if key not in main_map:
                issues.append("rank identity absent from main plan: " + str(key))
                continue
            main_index, _, main_case = main_map[key]
            check(case == main_case, "complete case dictionaries differ: " + str(key))
            path = rank_root / "rank_tasks" / f"task-{rank_index:06d}" / "result.json"
            if not path.is_file():
                issues.append("missing rank receipt: " + str(path))
                continue
            result = read_json(path)
            check(result.get("complete") is True, "rank receipt incomplete: " + str(key))
            check(result.get("plan_sha256") == rank_plan_digest,
                  "rank receipt plan mismatch: " + str(key))
            check(result.get("task") == task, "rank receipt task mismatch: " + str(key))
            check(result.get("case") == case, "rank receipt case mismatch: " + str(key))
            check(result.get("executed_code_sha256") == rank_plan.get("code_sha256"),
                  "rank receipt executed-source mismatch: " + str(key))
            for field in fingerprint_fields:
                digest = result.get(field, "")
                check(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                      "rank receipt invalid " + field + ": " + str(key))
                check(digest == main_fingerprints[key][field],
                      "cross-study " + field + " mismatch: " + str(key))
            pairs.append(dict(case_id=key[0], seed=key[1], main_task_index=main_index,
                              rank_task_index=rank_index, pair_passed=len(issues) == before,
                              **{f: result.get(f) for f in fingerprint_fields}))
        check(len(pairs) == len(rank_plan["tasks"]), "cross-study pair coverage")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        issues.append(type(error).__name__ + ": " + str(error))

    return dict(schema=1, audit_passed=not issues, n_pairs=len(pairs),
                n_passing_pairs=sum(pair["pair_passed"] for pair in pairs),
                issues=issues, main_root=str(main_root), rank_root=str(rank_root),
                scope="Metadata-only cross-study equality of recorded typed-array fingerprints; "
                      "both plans bound to complete passing numerical audits. No arrays or fits read.",
                limitations=["Fingerprint values are trusted only after their independent numerical audits; "
                             "this check does not recompute array hashes.",
                             "Equality verifies dataset reuse, not additional independent replications."],
                input_sha256=inputs, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_root", type=Path)
    parser.add_argument("rank_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_pairing(args.main_root, args.rank_root)
    report["auditor_sha256"] = sha256(Path(__file__))
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("audit_passed", "n_pairs", "n_passing_pairs", "issues")}))
    raise SystemExit(0 if report["audit_passed"] else 2)


if __name__ == "__main__":
    main()
