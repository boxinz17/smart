"""Compact synthetic audits; no raw fitting outputs, estimators or scheduler."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_campaign as campaign


def config(budgets=(500, 2000), grid=(.01, .04)):
    return dict(iteration_budgets=list(budgets), init_penalties=[.03],
                penalties_u=list(grid), penalties_v=[.01], checkpoint_interval=250,
                checkpoint_execution="continuous", validation_iterations=[])


def entry(seed, *, options=None, case_key=None, run="one", inapplicable=False):
    cell = dict(task_id=f"m0_e2_s{seed}_k0", model=0, experiment=2, setting=0,
                seed=seed, random_seed=seed+1000,
                simulation_setting=dict(suffix="rs=5", n=200, p=100, q=100, sigma0=.5, source_rank=5),
                inapplicability_reason="source_rank_exceeds_target" if inapplicable else None)
    return dict(case_key=case_key or f"case-{seed}", run_root=f"/unavailable/raw/{run}",
                plan_fingerprint=campaign._digest(run), plan_sha256=campaign._digest([run, "plan"]),
                work_table_sha256=campaign._digest([run, "work"]), cell=cell,
                configuration=deepcopy(options or config()), seed_file_sha256="e"*64,
                source=dict(scheme="frozen-study-source", implementation={
                    "true": dict(fingerprint="c"*64), "false": dict(fingerprint="c"*64)}))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_index(root, entries, *, version=1, mode="full", allowed_missing=None):
    index = dict(schema_version=version, method=campaign.INDEX_METHOD, cases=entries)
    if version >= 2:
        index.update(publication_mode=mode, scope=dict(models=sorted({e["cell"]["model"] for e in entries})))
    if allowed_missing is not None:
        index["allowed_missing_tasks"] = allowed_missing
    index["index_fingerprint"] = campaign._digest(index)
    path = root / "campaign-index.json"
    write_json(path, index)
    return path, index


def audit_for(index, planned, *, value=1., incomplete=False, archive=False, launch=False, failed=False):
    cell = planned["cell"]
    options = planned["configuration"]
    applicable = cell["inapplicability_reason"] is None
    generation = campaign._digest(dict(index_fingerprint=index["index_fingerprint"],
        case_key=planned["case_key"], input_fingerprint="a"*64, analysis_fingerprint="b"*64))
    grid_size = math.prod(len(options[name]) for name in ("init_penalties", "penalties_u", "penalties_v"))
    caps = []
    if applicable:
        for i, budget in enumerate(options["iteration_budgets"]):
            unresolved = [grid_size-1] if incomplete and i == len(options["iteration_budgets"])-1 else []
            caps.append(dict(iteration_budget=budget, success=True, coverage_complete=not unresolved,
                reached_candidates=grid_size-len(unresolved), unresolved_candidate_ids=unresolved,
                validation_mse=value, coefficient_error=value*2, selected_iteration=25,
                selected_checkpoint=min(budget, 250), winner_candidate_id=0,
                optimizer_converged=False, selected_converged=False, selected_at_cap=False, selected_near_cap=False))
    status = "inapplicable" if not applicable else "partial" if incomplete else "complete"
    summary = dict(model="model1", model_id=0, experiment="exp3", setting="rs=5", seed_id=cell["seed"],
                   applicable=applicable, inapplicability_reason=cell["inapplicability_reason"], missing=False,
                   status=status, caps=caps, configuration=deepcopy(options), implementation_fingerprint="c"*64)
    return dict(schema_version=index["schema_version"],
        **(dict(publication_mode=index["publication_mode"]) if index["schema_version"] >= 2 else {}),
        case_key=planned["case_key"], generation=generation,
        index_fingerprint=index["index_fingerprint"], plan_fingerprint=planned["plan_fingerprint"],
        identity=deepcopy(cell), configuration=deepcopy(options),
        execution=dict(fit_success=True, launch_success=not launch, archive_status="failed" if archive else "deferred",
            archive_only_failures=["task-archive-failed"] if archive else [],
            issues=[dict(error="launcher_failed")] if launch else []),
        scientific=dict(audit_passed=not failed, status="invalid" if failed else status,
            budget_coverage_complete=not failed and not incomplete,
            issues=[dict(error="factor_validation_failed")] if failed else []),
        summary=None if failed else summary, paper=dict(candidates=[]) if not failed else None)


def publish(root, audit):
    case_root = root / "cases" / audit["case_key"]
    directory = case_root / "generations" / audit["generation"]
    audit_sha = write_json(directory / "audit.json", audit)
    receipt = {key: audit[key] for key in
               ("schema_version", "case_key", "generation", "index_fingerprint", "plan_fingerprint")}
    receipt.update(input_fingerprint="a"*64, analysis_fingerprint="b"*64,
                   outputs=dict(audit=dict(path="audit.json", sha256=audit_sha),
                                merged=dict(path="merged.json", sha256="d"*64) if audit["summary"] else None),
                   completed_utc="2026-09-11T00:00:00Z")
    if audit["schema_version"] >= 2:
        receipt["publication_mode"] = audit["publication_mode"]
        receipt["outputs"]["selected_factors"] = None
        if audit["publication_mode"] == "compact":
            receipt["outputs"]["merged"] = None
            if audit["summary"]:
                receipt["outputs"]["selected_factors"] = dict(path="selected-factors.json", sha256="f"*64)
    receipt_sha = write_json(directory / "receipt.json", receipt)
    write_json(case_root / "current.json", dict(generation=audit["generation"], receipt_sha256=receipt_sha))
    # No merged.json or raw directory exists: reduction must never use either.
    return directory


def test_pools_individual_seeds_across_batches_not_batch_means(tmp_path):
    entries = [entry(0), entry(1), entry(2, run="two")]
    path, index = write_index(tmp_path, entries)
    for planned, value in zip(entries, [0., 2., 10.]):
        publish(tmp_path, audit_for(index, planned, value=value))
    report = campaign.summarize_campaign(path, output_root=tmp_path / "summary")
    assert report["primary_complete"] and report["publication_complete"]
    assert report["expected_cases"] == report["recorded_cases"] == 3
    group = report["groups"][0]
    assert group["expected_seed_ids"] == [0, 1, 2]
    stats = group["caps"][0]["complete_grid"]["validation_mse"]
    assert stats == dict(n=3, mean=4., se=pytest.approx(math.sqrt(28/3)))
    assert stats["mean"] != 5.5  # Average of the two batch means would be wrong.
    assert (tmp_path / "summary/campaign-summary.json").is_file()
    assert not list((tmp_path / "summary").glob("*.tmp"))


def test_preserves_mixed_grids_budgets_and_declared_seed_scope(tmp_path):
    entries = [entry(0), entry(1, options=config((500, 8000), (.01, .04, .16)))]
    path, index = write_index(tmp_path, entries)
    for planned in entries:
        publish(tmp_path, audit_for(index, planned))
    report = campaign.summarize_campaign(path)
    assert report["primary_complete"]
    assert len(report["groups"]) == 2
    assert {tuple(g["configuration"]["iteration_budgets"]) for g in report["groups"]} == {(500, 2000), (500, 8000)}
    assert {tuple(g["expected_seed_ids"]) for g in report["groups"]} == {(0,), (1,)}
    assert all(cap["usable"] == 1 for group in report["groups"] for cap in group["caps"])


def test_incomplete_grid_available_winners_are_separate_from_primary(tmp_path):
    entries = [entry(0), entry(1)]
    path, index = write_index(tmp_path, entries)
    publish(tmp_path, audit_for(index, entries[0], value=2.))
    publish(tmp_path, audit_for(index, entries[1], value=100., incomplete=True))
    report = campaign.summarize_campaign(path)
    first, final = report["groups"][0]["caps"]
    assert first["usable"] == 2 and first["primary_complete"]
    assert final["usable"] == 1 and final["available_winners"] == 2
    assert final["complete_grid"]["validation_mse"]["mean"] == 2.
    assert final["incomplete_grid_available"]["validation_mse"]["mean"] == 100.
    assert report["publication_complete"] and not report["primary_complete"]


def test_archive_only_errors_do_not_exclude_valid_science_but_launch_errors_do(tmp_path):
    entries = [entry(0), entry(1)]
    path, index = write_index(tmp_path, entries)
    publish(tmp_path, audit_for(index, entries[0], archive=True, value=2.))
    publish(tmp_path, audit_for(index, entries[1], launch=True, value=100.))
    report = campaign.summarize_campaign(path)
    cap = report["groups"][0]["caps"][0]
    assert cap["usable"] == 1 and cap["execution_excluded"] == 1
    assert cap["complete_grid"]["seed_ids"] == [0]
    assert report["archive_only_failed_tasks"] == 1
    assert not report["primary_complete"]
    assert report["cases"][0]["execution"]["archive_status"] == "failed"


def test_missing_corrupt_and_scientific_failures_remain_explicit(tmp_path):
    entries = [entry(i) for i in range(4)]
    path, index = write_index(tmp_path, entries)
    publish(tmp_path, audit_for(index, entries[0]))
    corrupt = publish(tmp_path, audit_for(index, entries[2]))
    (corrupt / "audit.json").write_text("{}")
    publish(tmp_path, audit_for(index, entries[3], failed=True))
    report = campaign.summarize_campaign(path)
    assert report["expected_cases"] == 4
    assert report["recorded_cases"] == 2
    assert report["missing_cases"] == report["invalid_cases"] == report["scientific_audit_failed"] == 1
    assert [row["record_state"] for row in report["cases"]] == ["published", "missing", "invalid", "published"]
    assert report["groups"][0]["caps"][0]["usable"] == 1


@pytest.mark.parametrize("mutation", [
    lambda a: a["identity"].update(seed=99),
    lambda a: a["summary"].update(seed_id=True),
    lambda a: a["summary"]["caps"][0].update(iteration_budget=True),
    lambda a: a["summary"]["caps"][0].update(success=1),
    lambda a: a["summary"]["caps"][0].update(validation_mse=float("inf")),
    lambda a: a["summary"]["caps"][0].update(selected_iteration=10000),
    lambda a: a["summary"]["caps"][0].update(winner_candidate_id=2),
    lambda a: a["summary"]["caps"][0].update(unresolved_candidate_ids=[999]),
    lambda a: a["summary"]["caps"][0].update(selected_converged=True),
    lambda a: a["scientific"].update(budget_coverage_complete=False),
    lambda a: a["summary"]["configuration"].update(penalties_u=[.5]),
    lambda a: a["execution"].update(fit_success="true"),
    lambda a: a["scientific"].update(issues=[dict(error="hidden_failure")]),
])
def test_semantically_invalid_compact_audit_is_excluded_even_with_matching_hashes(tmp_path, mutation):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    audit = audit_for(index, planned)
    mutation(audit)
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 1
    assert report["groups"][0]["caps"][0]["usable"] == 0


@pytest.mark.parametrize("kind", ["case_key", "scientific_overlap", "changed_setting_suffix", "reused_random_seed"])
def test_duplicate_index_cases_fail_before_any_reduction(tmp_path, kind):
    entries = [entry(0), entry(1)]
    if kind == "case_key":
        entries[1]["case_key"] = entries[0]["case_key"]
    elif kind == "scientific_overlap":
        entries[1] = entry(0, case_key="other-run", run="two", options=config((100, 500)))
    elif kind == "changed_setting_suffix":
        entries[1]["cell"]["seed"] = 0
        entries[1]["cell"]["setting"] = 1
    else:
        entries[1]["cell"]["random_seed"] = entries[0]["cell"]["random_seed"]
    path, _ = write_index(tmp_path, entries)
    with pytest.raises(ValueError, match="Duplicate|Overlapping"):
        campaign.summarize_campaign(path)


def test_tampered_index_and_pointer_receipt_fail_closed(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    directory = publish(tmp_path, audit_for(index, planned))
    (directory / "receipt.json").write_text("{}")
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1
    index["cases"][0]["cell"]["seed"] = 7
    write_json(path, index)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        campaign.summarize_campaign(path)


def test_inapplicable_cases_have_separate_counts_no_fit_statistics(tmp_path):
    planned = entry(0, inapplicable=True)
    path, index = write_index(tmp_path, [planned])
    publish(tmp_path, audit_for(index, planned))
    report = campaign.summarize_campaign(path)
    assert report["expected_inapplicable"] == report["audited_inapplicable"] == 1
    assert report["expected_applicable"] == 0
    assert report["primary_complete"]
    assert all(cap["expected"] == cap["usable"] == 0 for cap in report["groups"][0]["caps"])


def test_legacy_missing_validation_iterations_normalizes_only_for_comparison(tmp_path):
    options = config()
    options.pop("validation_iterations")
    entries = [entry(0, options=options), entry(1)]
    path, index = write_index(tmp_path, entries)
    for planned in entries:
        audit = audit_for(index, planned)
        audit["summary"]["configuration"]["validation_iterations"] = []
        publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["primary_complete"] and len(report["groups"]) == 1
    assert "validation_iterations" not in report["cases"][0]["configuration"]


def test_different_scientific_implementations_are_not_pooled(tmp_path):
    entries = [entry(0), entry(1)]
    entries[1]["source"]["implementation"]["true"]["fingerprint"] = "d"*64
    path, index = write_index(tmp_path, entries)
    publish(tmp_path, audit_for(index, entries[0]))
    other = audit_for(index, entries[1])
    other["summary"]["implementation_fingerprint"] = "d"*64
    publish(tmp_path, other)
    report = campaign.summarize_campaign(path)
    assert report["implementation_inconsistent_groups"] == 1
    assert not report["primary_complete"]
    assert all(cap["usable"] == 0 for cap in report["groups"][0]["caps"])


def test_cli_runs_without_site_packages_and_never_reads_raw_or_merged(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    directory = publish(tmp_path, audit_for(index, planned))
    (directory / "merged.json").write_text("This is intentionally not JSON and must never be read.")
    result = subprocess.run([sys.executable, "-S", campaign.__file__, "--index", str(path),
                             "--output-root", str(tmp_path / "out")], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["primary_complete"]
    source = Path(campaign.__file__).read_text()
    assert "import numpy" not in source and "import run_sparse" not in source


def test_publication_paths_cannot_escape_case_tree(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    publish(tmp_path, audit_for(index, planned))
    pointer = tmp_path / "cases" / planned["case_key"] / "current.json"
    write_json(pointer, dict(generation="../../outside", receipt_sha256="a"*64))
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1


def test_cli_output_root_keeps_terminal_output_compact_and_full_details_in_file(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    publish(tmp_path, audit_for(index, planned))
    command = [sys.executable, "-S", campaign.__file__, "--index", str(path)]
    result = subprocess.run(command + ["--output-root", str(tmp_path / "summary")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    compact = json.loads(result.stdout)
    assert compact["recorded_cases"] == 1 and compact["primary_complete"]
    assert "cases" not in compact and "groups" not in compact
    stored = json.loads(Path(compact["output_path"]).read_text())
    assert len(stored["cases"]) == len(stored["groups"]) == 1
    explicit_stdout = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert explicit_stdout.returncode == 0, explicit_stdout.stderr
    assert len(json.loads(explicit_stdout.stdout)["cases"]) == 1


@pytest.mark.parametrize("failed_ids", [(), (6, 7)])
@pytest.mark.parametrize("mode", ["full", "compact"])
def test_reduces_actual_case_aggregator_publication_without_rereading_factors(tmp_path, failed_ids, mode):
    from test_case_aggregation import _prepare, _aggregate, _generation
    _, output, plan, _, _ = _prepare(tmp_path, failed_ids=failed_ids, publication_mode=mode)
    envelope = _aggregate(output, plan)
    assert envelope["scientific"]["audit_passed"]
    # The reducer uses the certified compact audit even if factors are not
    # available to this stage. Full raw and merged verification belongs above.
    artifact = "merged.json" if mode == "full" else "selected-factors.json"
    (_generation(output, envelope) / artifact).unlink()
    report = campaign.summarize_campaign(output / "campaign-index.json")
    assert report["invalid_cases"] == report["missing_cases"] == 0
    assert report["recorded_cases"] == 1
    assert report["primary_complete"] is (not failed_ids)
    assert report["cases"][0]["scientific"]["status"] == ("partial" if failed_ids else "complete")


def test_v2_full_and_compact_have_identical_mixed_configuration_statistics(tmp_path):
    entries = [entry(0), entry(1), entry(2, options=config((500, 8000), (.01, .04, .16))),
               entry(3), entry(4), entry(5, inapplicable=True)]
    reports = {}
    for mode in ("full", "compact"):
        root = tmp_path / mode
        path, index = write_index(root, entries, version=2, mode=mode)
        for i, planned in enumerate(entries):
            if i == 3:
                continue
            publish(root, audit_for(index, planned, value=float(i+1), incomplete=i == 1,
                                    archive=i == 0, failed=i == 4))
        reports[mode] = campaign.summarize_campaign(path)
    full, compact = reports["full"], reports["compact"]
    assert full["groups"] == compact["groups"]
    for key in ("expected_cases", "recorded_cases", "missing_cases", "invalid_cases", "scientific_audit_failed",
                "expected_inapplicable", "audited_inapplicable", "archive_only_failed_tasks", "primary_complete"):
        assert full[key] == compact[key]
    assert full["recorded_cases"] == 5 and full["missing_cases"] == 1
    assert full["invalid_cases"] == 0 and full["scientific_audit_failed"] == 1
    assert compact["publication_mode"] == "compact" and compact["scope"] == dict(models=[0])


@pytest.mark.parametrize("mutation", [
    lambda i: i.update(publication_mode="unknown"),
    lambda i: i.update(scope=dict(models=[True])),
    lambda i: i.update(scope=dict(models=[1])),
    lambda i: i.update(scope=dict(models=[0, 1])),
    lambda i: i.update(scope=dict(models=[0, 0])),
    lambda i: i.update(scope=dict(models=[0], unexpected=True)),
])
def test_v2_invalid_scope_or_mode_fails_before_reduction(tmp_path, mutation):
    path, index = write_index(tmp_path, [entry(0)], version=2)
    mutation(index)
    index["index_fingerprint"] = campaign._digest({k: v for k, v in index.items() if k != "index_fingerprint"})
    write_json(path, index)
    with pytest.raises(ValueError, match="mode|scope"):
        campaign.summarize_campaign(path)


def _rewrite_receipt(directory, mutation):
    receipt = json.loads((directory / "receipt.json").read_text())
    mutation(receipt)
    sha = write_json(directory / "receipt.json", receipt)
    write_json(directory.parent.parent / "current.json",
               dict(generation=receipt["generation"], receipt_sha256=sha))


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(publication_mode="full"),
    lambda r: r.update(schema_version=1),
    lambda r: r["outputs"].update(merged=dict(path="merged.json", sha256="d"*64)),
    lambda r: r["outputs"].update(selected_factors=None),
    lambda r: r["outputs"]["selected_factors"].update(path="../outside.json"),
    lambda r: r["outputs"]["selected_factors"].update(sha256="unverified"),
    lambda r: r["outputs"].update(unexpected=None),
])
def test_v2_invalid_publication_descriptor_is_not_accepted(tmp_path, mutation):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=2, mode="compact")
    directory = publish(tmp_path, audit_for(index, planned))
    _rewrite_receipt(directory, mutation)
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_v2_failed_audit_must_not_advertise_scientific_artifacts(tmp_path, mode):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=2, mode=mode)
    directory = publish(tmp_path, audit_for(index, planned, failed=True))
    assert campaign.summarize_campaign(path)["scientific_audit_failed"] == 1
    key, name = ("merged", "merged.json") if mode == "full" else ("selected_factors", "selected-factors.json")
    _rewrite_receipt(directory, lambda r: r["outputs"].update({key: dict(path=name, sha256="d"*64)}))
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1


def test_lean_default_retains_choices_metrics_and_audit_reference_without_histories(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=2, mode="compact")
    audit = audit_for(index, planned)
    for cap in audit["summary"]["caps"]:
        cap["winner_candidate_id"] = 1
        cap["candidate_endpoints"] = [dict(diagnostics=["large history"]*100)]
    audit["summary"]["trajectory_final_diagnostics"] = [dict(history=["large history"]*100)]
    audit["paper"].update(selection_history=[dict(history=["large history"]*100)],
                          trajectories=[dict(validation_history=["large history"]*100)], fit_time_sec=5.)
    audit["execution"]["task_outcomes"] = [dict(task_id="task", nested_history=["large history"]*100)]
    directory = publish(tmp_path, audit)
    lean = campaign.summarize_campaign(path)
    expanded = campaign.summarize_campaign(path, include_histories=True)
    row = lean["cases"][0]
    assert "trajectory_final_diagnostics" not in row["summary"]
    assert all("candidate_endpoints" not in cap for cap in row["summary"]["caps"])
    assert "selection_history" not in row["paper"] and "trajectories" not in row["paper"]
    assert "task_outcomes" not in row["execution"] and row["execution"]["task_outcome_count"] == 1
    assert row["paper"]["fit_time_sec"] == 5.
    choice = row["paper"]["selected"][0]
    assert choice["params"] == dict(init_penalty=.03, penalty_u=.04, penalty_v=.01)
    assert choice["selected_iteration"] == 25 and choice["validation_mse"] == 1.
    assert row["provenance"]["audit_path"] == (directory / "audit.json").relative_to(tmp_path).as_posix()
    assert row["provenance"]["audit_sha256"] == hashlib.sha256((directory / "audit.json").read_bytes()).hexdigest()
    assert lean["groups"] == expanded["groups"]
    assert expanded["cases"][0]["paper"] == audit["paper"]
    assert expanded["cases"][0]["summary"] == audit["summary"]
    assert "large history" not in json.dumps(lean)


def test_compact_reducer_never_reads_or_stats_factor_artifacts(tmp_path, monkeypatch):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=2, mode="compact")
    publish(tmp_path, audit_for(index, planned))
    original_stat, original_read = Path.stat, Path.read_bytes
    def guarded_stat(self, *args, **kwargs):
        assert self.name not in ("merged.json", "selected-factors.json")
        return original_stat(self, *args, **kwargs)
    def guarded_read(self, *args, **kwargs):
        assert self.name not in ("merged.json", "selected-factors.json")
        assert str(self).startswith(str(tmp_path))
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert campaign.summarize_campaign(path)["primary_complete"]


def test_include_histories_cli_is_explicit_opt_in(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=2, mode="compact")
    audit = audit_for(index, planned)
    audit["paper"]["selection_history"] = [dict(diagnostic="retained only on request")]
    publish(tmp_path, audit)
    command = [sys.executable, "-S", campaign.__file__, "--index", str(path)]
    lean = subprocess.run(command, capture_output=True, text=True, timeout=10)
    expanded = subprocess.run(command + ["--include-histories"], capture_output=True, text=True, timeout=10)
    assert lean.returncode == expanded.returncode == 0
    assert "selection_history" not in json.loads(lean.stdout)["cases"][0]["paper"]
    assert json.loads(expanded.stdout)["cases"][0]["paper"]["selection_history"] == audit["paper"]["selection_history"]


def test_v1_publication_cannot_silently_claim_new_compact_mode(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned])
    directory = publish(tmp_path, audit_for(index, planned))
    _rewrite_receipt(directory, lambda receipt: receipt.update(publication_mode="compact"))
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1


def test_lean_selected_penalties_follow_cartesian_grid_and_original_budget(tmp_path):
    options = config(grid=(.01, .04, .16))
    options.update(init_penalties=[.03, .1], penalties_v=[.01, .04])
    planned = entry(0, options=options)
    path, index = write_index(tmp_path, [planned], version=2, mode="compact")
    audit = audit_for(index, planned)
    audit["summary"]["caps"][0]["winner_candidate_id"] = 7
    audit["summary"]["caps"][1]["winner_candidate_id"] = 23
    publish(tmp_path, audit)
    choices = campaign.summarize_campaign(path)["cases"][0]["paper"]["selected"]
    assert choices[0]["params"] == dict(init_penalty=.1, penalty_u=.01, penalty_v=.04)
    assert choices[0]["grid_candidate_id"] == 7 and choices[0]["winner_origin_budget"] == 500
    assert choices[1]["params"] == dict(init_penalty=.1, penalty_u=.16, penalty_v=.04)
    assert choices[1]["grid_candidate_id"] == 11 and choices[1]["winner_origin_budget"] == 2000


def missing_policy(planned, grid_id=1, *, external_cancellation=False):
    policy = dict(run_root=planned["run_root"], case_key=planned["case_key"],
                task_id=f"{planned['cell']['task_id']}_g{grid_id}", grid_candidate_ids=[grid_id],
                reason="Explicitly cancelled unfinished task", launcher_exit_code=137)
    if external_cancellation:
        policy.pop("launcher_exit_code")
        policy["cancellation_evidence"] = dict(path="/unavailable/cancellation/evidence.json",
            sha256="f"*64, job_id="11859590", step_id="11859590.1620", task_id=policy["task_id"])
    return policy


def policy_audit(index, planned, *, missing=False, value=1., partial_budget=False, other_failure=False):
    audit = audit_for(index, planned, value=value, incomplete=partial_budget)
    allowed = [item for item in index["allowed_missing_tasks"] if item["case_key"] == planned["case_key"]]
    unavailable = deepcopy(allowed) if missing else []
    missing_ids = sorted(i for item in unavailable for i in item["grid_candidate_ids"])
    count = campaign._grid_size(planned["configuration"])
    audit["missing_tuning"] = dict(allowed_tasks=deepcopy(allowed), unavailable_tasks=unavailable,
        unavailable_grid_candidate_ids=missing_ids, planned_candidate_count=count, available_candidate_count=count-len(missing_ids))
    execution = audit["execution"]
    execution.update(allowed_missing_task_ids=sorted(item["task_id"] for item in unavailable),
                     available_tasks_success=not other_failure, task_outcomes=[], issues=[],
                     fit_success=not missing and not other_failure, launch_success=not missing and not other_failure)
    for candidate in range(count):
        task_id = f"{planned['cell']['task_id']}_g{candidate}"
        absent = candidate in missing_ids
        external = any(item["task_id"] == task_id and "cancellation_evidence" in item for item in unavailable)
        failed = other_failure and candidate == 0
        codes = dict(zip(("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"),
                         (None, None, None if external else 137) if absent else
                         (1, 1, 1) if failed else (0, 0, 0)))
        execution["task_outcomes"].append(dict(task_id=task_id, exit_codes=codes,
                                             archive_status="deferred", archive_only_failure=False))
        if absent or failed:
            execution["issues"].extend([dict(task_id=task_id, kind="fit_execution", exit_code=codes["process-exit-code.txt"]),
                dict(task_id=task_id, kind="worker_or_launch_execution", worker_exit_code=codes["exit-code.txt"],
                     launcher_exit_code=codes["launcher-exit-code.txt"])])
    if missing:
        audit["summary"].update(status="partial", unavailable_grid_candidate_ids=missing_ids,
                                planned_candidate_count=count, available_candidate_count=count-len(missing_ids))
        audit["scientific"].update(status="partial", budget_coverage_complete=False)
        for cap in audit["summary"]["caps"]:
            ids = sorted(set(cap["unresolved_candidate_ids"]) | set(missing_ids))
            cap.update(coverage_complete=False, unresolved_candidate_ids=ids, reached_candidates=count-len(ids))
    return audit


def test_explicit_missing_policy_reports_all_100_seed_winners_and_99_seed_sensitivity(tmp_path):
    options = config(grid=(.0025, .01, .04, .16, .32))
    options.update(init_penalties=[.01, .03, .1, .3], penalties_v=[.0025, .01, .04, .16, .32])
    entries = [entry(seed, options=options) for seed in range(100)]
    allowed = [missing_policy(entries[56], grid) for grid in (15, 16, 17)]
    path, index = write_index(tmp_path, entries, version=3, mode="compact", allowed_missing=allowed)
    for planned in entries:
        publish(tmp_path, policy_audit(index, planned, missing=planned["cell"]["seed"] == 56,
                                      value=float(planned["cell"]["seed"])))
    report = campaign.summarize_campaign(path)
    assert report["recorded_cases"] == 100 and report["invalid_cases"] == 0
    assert report["primary_complete"] is False and report["allowed_missing_policy_complete"] is True
    assert report["permitted_missing_case_count"] == 1 and report["permitted_missing_task_count"] == 3
    group = report["groups"][0]
    assert group["policy_missing_seed_ids"] == [56]
    for cap in group["caps"]:
        assert cap["complete_grid"]["count"] == 99
        assert cap["policy_available"]["count"] == 100
        assert cap["policy_available"]["full_grid_coverage_count"] == 99
        assert cap["policy_available"]["available_grid_budget_complete_count"] == 100
        assert cap["policy_available"]["validation_mse"]["mean"] == 49.5
        sensitivity = cap["policy_sensitivity_excluding_missing"]
        assert sensitivity["count"] == cap["policy_sensitivity_expected"] == 99
        assert 56 not in sensitivity["seed_ids"]
        assert sensitivity["validation_mse"]["mean"] == pytest.approx((4950-56)/99)
    result = subprocess.run([sys.executable, "-S", campaign.__file__, "--index", str(path),
                             "--output-root", str(tmp_path / "summary")], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["primary_complete"] is False
    assert json.loads(result.stdout)["allowed_missing_policy_complete"] is True


def test_policy_preserves_ordinary_partial_budget_winners_and_accurate_coverage_counts(tmp_path):
    entries = [entry(0), entry(1), entry(2)]
    path, index = write_index(tmp_path, entries, version=3, allowed_missing=[missing_policy(entries[1])])
    for planned in entries:
        seed = planned["cell"]["seed"]
        publish(tmp_path, policy_audit(index, planned, missing=seed == 1, partial_budget=seed == 2))
    report = campaign.summarize_campaign(path)
    cap = report["groups"][0]["caps"][-1]
    assert report["allowed_missing_policy_complete"] and not report["primary_complete"]
    assert cap["complete_grid"]["count"] == 1 and cap["incomplete_grid_available"]["count"] == 1
    assert cap["policy_available"]["count"] == 3
    assert cap["policy_available"]["full_grid_coverage_count"] == 1
    assert cap["policy_available"]["available_grid_budget_complete_count"] == 2
    assert cap["policy_sensitivity_excluding_missing"]["count"] == 2


@pytest.mark.parametrize("external_cancellation", [False, True])
def test_policy_does_not_excuse_unrelated_execution_failure(tmp_path, external_cancellation):
    entries = [entry(0), entry(1)]
    path, index = write_index(tmp_path, entries, version=3,
        allowed_missing=[missing_policy(entries[1], external_cancellation=external_cancellation)])
    publish(tmp_path, policy_audit(index, entries[0]))
    publish(tmp_path, policy_audit(index, entries[1], missing=True, other_failure=True))
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 0
    assert not report["allowed_missing_policy_complete"]
    assert report["groups"][0]["caps"][0]["policy_available"]["count"] == 1


@pytest.mark.parametrize("mutation", [
    lambda a: a["execution"].update(available_tasks_success=True),
    lambda a: a["execution"].update(fit_success=True, launch_success=True),
    lambda a: a["missing_tuning"].update(unavailable_tasks=[]),
    lambda a: a["missing_tuning"].update(available_candidate_count=2),
    lambda a: a["summary"].update(unavailable_grid_candidate_ids=[]),
    lambda a: a["summary"]["caps"][0].update(winner_candidate_id=1),
    lambda a: a["execution"]["task_outcomes"][1]["exit_codes"].update({"launcher-exit-code.txt": 1}),
    lambda a: a["execution"]["task_outcomes"][1].update(archive_status="failed"),
    lambda a: a["execution"]["issues"].append(dict(task_id="m0_e2_s1_k0_g1", kind="corrupt_result")),
])
def test_policy_evidence_cannot_hide_missing_mismatch_or_real_failure(tmp_path, mutation):
    planned = entry(1)
    path, index = write_index(tmp_path, [planned], version=3, allowed_missing=[missing_policy(planned)])
    audit = policy_audit(index, planned, missing=True, other_failure=True)
    mutation(audit)
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 1 and not report["allowed_missing_policy_complete"]


@pytest.mark.parametrize("version", [1, 2])
def test_missing_policy_requires_new_index_version(tmp_path, version):
    planned = entry(0)
    path, _ = write_index(tmp_path, [planned], version=version, allowed_missing=[missing_policy(planned)])
    with pytest.raises(ValueError, match="version 3"):
        campaign.summarize_campaign(path)


def test_policy_declared_task_that_is_now_available_is_not_dropped_from_sensitivity(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=3, allowed_missing=[missing_policy(planned)])
    publish(tmp_path, policy_audit(index, planned))
    report = campaign.summarize_campaign(path)
    assert report["primary_complete"] and report["allowed_missing_policy_complete"]
    assert report["permitted_missing_task_count"] == 0
    assert report["groups"][0]["caps"][0]["policy_sensitivity_excluding_missing"]["count"] == 1


@pytest.mark.parametrize("missing", [False, True])
def test_external_cancellation_policy_reduces_published_metadata_without_evidence_file(tmp_path, monkeypatch, missing):
    planned = entry(0)
    policy = missing_policy(planned, external_cancellation=True)
    path, index = write_index(tmp_path, [planned], version=3, mode="compact", allowed_missing=[policy])
    publish(tmp_path, policy_audit(index, planned, missing=missing))
    original_read_bytes = Path.read_bytes

    def published_metadata_only(path):
        assert path.is_relative_to(tmp_path), "Reducer attempted to read external cancellation or fitting evidence"
        assert path.name in {"campaign-index.json", "current.json", "receipt.json", "audit.json"}
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", published_metadata_only)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 0 and report["allowed_missing_policy_complete"]
    assert report["primary_complete"] is not missing
    assert report["permitted_missing_task_count"] == int(missing)
    assert report["allowed_missing_tasks"] == [policy]
    cap = report["groups"][0]["caps"][0]
    assert cap["policy_available"]["count"] == 1
    assert cap["policy_sensitivity_excluding_missing"]["count"] == int(not missing)


@pytest.mark.parametrize("mutation", [
    lambda item: item.update(launcher_exit_code=137),
    lambda item: item.pop("cancellation_evidence"),
    lambda item: item["cancellation_evidence"].update(path="relative.json"),
    lambda item: item["cancellation_evidence"].update(sha256="invalid"),
    lambda item: item["cancellation_evidence"].update(job_id="0"),
    lambda item: item["cancellation_evidence"].update(job_id=11859590),
    lambda item: item["cancellation_evidence"].update(step_id="11859591.1620"),
    lambda item: item["cancellation_evidence"].update(step_id="11859590.batch"),
    lambda item: item["cancellation_evidence"].update(task_id="different_task"),
    lambda item: item["cancellation_evidence"].update(unexpected=True),
])
def test_external_cancellation_policy_rejects_malformed_or_ambiguous_reference(tmp_path, mutation):
    planned = entry(0)
    policy = missing_policy(planned, external_cancellation=True)
    mutation(policy)
    path, _ = write_index(tmp_path, [planned], version=3, allowed_missing=[policy])
    with pytest.raises(ValueError, match="Invalid"):
        campaign.summarize_campaign(path)


@pytest.mark.parametrize("marker", ["process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"])
@pytest.mark.parametrize("value", [0, 137])
def test_external_cancellation_policy_rejects_present_exit_markers(tmp_path, marker, value):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=3,
                             allowed_missing=[missing_policy(planned, external_cancellation=True)])
    audit = policy_audit(index, planned, missing=True)
    audit["execution"]["task_outcomes"][1]["exit_codes"][marker] = value
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 1 and not report["allowed_missing_policy_complete"]


def test_external_cancellation_audit_cannot_substitute_another_evidence_hash(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=3,
                             allowed_missing=[missing_policy(planned, external_cancellation=True)])
    audit = policy_audit(index, planned, missing=True)
    audit["missing_tuning"]["unavailable_tasks"][0]["cancellation_evidence"]["sha256"] = "a"*64
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 1 and not report["allowed_missing_policy_complete"]


def test_external_cancellation_policy_cannot_excuse_failed_archival(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=3,
                             allowed_missing=[missing_policy(planned, external_cancellation=True)])
    audit = policy_audit(index, planned, missing=True)
    audit["execution"]["task_outcomes"][1]["archive_status"] = "failed"
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 1 and not report["allowed_missing_policy_complete"]


def test_policy_missing_or_scientifically_invalid_audit_still_blocks_completion(tmp_path):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=3, allowed_missing=[missing_policy(planned)])
    report = campaign.summarize_campaign(path)
    assert report["missing_cases"] == 1 and not report["allowed_missing_policy_complete"]
    audit = policy_audit(index, planned, missing=True)
    audit.update(summary=None, paper=None)
    audit["scientific"].update(audit_passed=False, status="invalid_or_missing", issues=[dict(kind="corrupt_result")])
    publish(tmp_path, audit)
    report = campaign.summarize_campaign(path)
    assert report["invalid_cases"] == 0 and report["scientific_audit_failed"] == 1
    assert not report["allowed_missing_policy_complete"]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("field,value", [("allowed_missing_task_ids", []), ("available_tasks_success", True)])
def test_old_audit_cannot_inject_execution_missing_policy(tmp_path, version, field, value):
    planned = entry(0)
    path, index = write_index(tmp_path, [planned], version=version)
    audit = audit_for(index, planned)
    audit["execution"][field] = value
    publish(tmp_path, audit)
    assert campaign.summarize_campaign(path)["invalid_cases"] == 1


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_real_policy_publication_reduces_without_claiming_full_grid_completion(tmp_path, mode):
    from test_case_aggregation import _run, _write_markers, _aggregate, cases, study
    raw, output = tmp_path / "raw", tmp_path / "analysis"
    plan, paths = _run(raw)
    task = study.planned_tasks(plan)[-1]
    paths[-1].unlink()
    (raw / "tasks" / task["task_id"] / "results" / "budget_study_manifest.json").unlink()
    _write_markers(raw, task["task_id"], process=None, exit_code=None, launcher="137")
    policy = [dict(run_root=str(raw), task_id=task["task_id"], reason="Explicit fixture cancellation", launcher_exit_code=137)]
    cases.prepare_campaign([raw], output, publication_mode=mode, allowed_missing_tasks=policy)
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"], audit["scientific"]["issues"]
    report = campaign.summarize_campaign(output / "campaign-index.json")
    assert report["invalid_cases"] == report["missing_cases"] == 0
    assert report["primary_complete"] is False and report["allowed_missing_policy_complete"] is True
    assert report["cases"][0]["missing_tuning"]["unavailable_grid_candidate_ids"] == task["grid_candidate_ids"]
    for cap in report["groups"][0]["caps"]:
        assert cap["complete_grid"]["count"] == 0
        assert cap["policy_available"]["count"] == 1
        assert cap["policy_sensitivity_excluding_missing"]["count"] == 0


@pytest.mark.parametrize("mode", ["full", "compact"])
def test_real_external_cancellation_publication_reduces_only_saved_metadata(tmp_path, monkeypatch, mode):
    from test_case_aggregation import _slurm_cancelled_fixture, _aggregate, cases
    raw, output, plan, task, _, index, _ = _slurm_cancelled_fixture(tmp_path)
    if mode == "full":
        output = tmp_path / "full-analysis"
        policy = [{key: value for key, value in item.items() if key not in ("case_key", "grid_candidate_ids")}
                  for item in index["allowed_missing_tasks"]]
        cases.prepare_campaign([raw], output, publication_mode=mode, allowed_missing_tasks=policy)
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"], audit["scientific"]["issues"]
    original_read_bytes = Path.read_bytes

    def published_metadata_only(path):
        assert path.is_relative_to(output), "Reducer reread raw fits or cancellation evidence"
        assert path.name in {"campaign-index.json", "current.json", "receipt.json", "audit.json"}
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", published_metadata_only)
    report = campaign.summarize_campaign(output / "campaign-index.json")
    assert report["invalid_cases"] == report["missing_cases"] == 0
    assert not report["primary_complete"] and report["allowed_missing_policy_complete"]
    assert report["permitted_missing_task_count"] == 1
    assert report["cases"][0]["missing_tuning"]["unavailable_grid_candidate_ids"] == task["grid_candidate_ids"]
