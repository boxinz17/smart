"""No-fit tests for plan scope, isolated collection, and failure provenance."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import discovery_budget_study as study


def small_plan(root, *, seeds=(0,), setting=0, experiment=0):
    plan = study.make_plan(models=(0,), experiments=(experiment,), seed_ids=seeds,
                           setting_index=setting,
                           config=study.configuration(iteration_budgets=(2, 4), checkpoint_interval=2,
                                                      init_penalties=(.03,), penalties_u=(.01,), penalties_v=(.01,)))
    study.write_plan(plan, root)
    return plan


def fake_record(plan, cell, status="complete"):
    """A metadata-only fixture; intentionally no fitted coefficients or trajectories."""
    setting = cell["simulation_setting"]
    identity = dict(schema_version=1, method=study.METHOD, model=f"model{cell['model']+1}",
        experiment=f"exp{cell['experiment']+1}", rd_seed_id=cell["seed"], random_seed=cell["random_seed"],
        setting=setting, configuration=study.resolved_configuration(setting, plan["configuration"]),
        generator_arguments=dict(n=setting["n"], p=setting["p"], q=setting["q"], sigma0=setting["sigma0"],
                                 sigma=.5, r_star=5, r0_star=10, random_seed=cell["random_seed"]))
    applicable = cell["inapplicability_reason"] is None
    source = plan["source"]["implementation"][str(applicable).lower()]
    return dict(identity, configuration_fingerprint=study.digest(identity),
        implementation_fingerprint_scheme=study.SCHEME, implementation_manifest=source["manifest"],
        implementation_fingerprint=source["fingerprint"], applicable=applicable, status=status,
        success=status in ("complete", "partial"), failure_reason=cell["inapplicability_reason"])


def save_cell(root, plan, cell, *, status="complete", attempt_status="completed", errors=None):
    destination = root/"tasks"/cell["task_id"]
    output = destination/"results"
    record = fake_record(plan, cell, status)
    path = output/study.relative_result(cell)
    study.atomic_json(record, path)
    row = dict(model=record["model"], experiment=record["experiment"], seed_id=cell["seed"],
               setting=record["setting"]["suffix"], status=status, success=record["success"], path=str(path))
    manifest = dict(schema_version=1, method=study.METHOD, models=[cell["model"]],
        experiments=[cell["experiment"]], seed_ids=[cell["seed"]], profile=plan["profile"],
        setting_index=cell["setting"], expected_cells=1, configuration=plan["configuration"],
        seed_file_sha256=plan["seed_file_sha256"], attempt_status=attempt_status, cells=[row], errors=errors or [])
    manifest_path = output/"budget_study_manifest.json"
    if attempt_status != "completed":
        manifest_path = output/"budget_study_manifest_attempts"/"20260909-attempt.json"
    study.atomic_json(manifest, manifest_path)
    for name in ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"):
        (destination/name).write_text("0\n")
    return path, record, manifest_path


def test_full_plan_has_exact_existing_scope_and_explicit_exclusions():
    plan = study.make_plan()
    assert (plan["expected_cells"], plan["expected_applicable"], plan["expected_inapplicable"]) == (7200, 6300, 900)
    assert len({cell["task_id"] for cell in plan["cells"]}) == 7200
    assert plan["configuration"]["stationarity_tol"] == 1e-6
    assert len(plan["configuration"]["penalties_u"])*len(plan["configuration"]["penalties_v"]) == 16
    assert plan["configuration"]["init_penalties"] == [.01, .03, .1]
    assert (len(plan["configuration"]["init_penalties"])*len(plan["configuration"]["penalties_u"])
            *len(plan["configuration"]["penalties_v"])) == 48
    assert plan["configuration"]["validation_iterations"] == [10, 25, 50, 100, 150, 200]
    assert {cell["inapplicability_reason"] for cell in plan["cells"]} == {
        None, "source_rank_must_be_positive", "target_rank_exceeds_source_rank"}


def test_planner_runs_without_site_packages_or_estimator_imports(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import discovery_budget_study as s; "
        "s.main(['plan','--output-root',sys.argv[2],'--models','0','--experiments','0',"
        "'--seed-ids','0','--setting-index','0']); "
        "assert not any(x in sys.modules for x in ('numpy','smart','sparse_smart','run_sparse_smart_budget_study'))"
    )
    completed = subprocess.run([sys.executable, "-S", "-c", code, str(study.HERE), str(tmp_path)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert len((tmp_path/"work-items.tsv").read_text().splitlines()) == 48
    assert (tmp_path/"work-items.tsv").read_text().startswith("m0_e0_s0_k0_g0\t0\t0\t0\t0")


def test_grid_configuration_and_source_identity_match_runner_without_fits(monkeypatch):
    # The repository root contains a smart/ project directory, so select its
    # actual package root explicitly instead of the implicit namespace package.
    monkeypatch.syspath_prepend(str(study.HERE.parent/"smart"))
    import run_sparse_smart_budget_study as runner
    import sparse_smart
    from smart import generate_data
    config = runner.RunnerConfig()
    assert study.configuration() == json.loads(json.dumps(asdict(config)))
    for model in range(3):
        for experiment in range(4):
            for ours, original in zip(study.experiment_settings(model, experiment),
                                      runner.experiment_settings(model, experiment)):
                assert ours == asdict(original)
                assert study.inapplicability(ours) == original.inapplicability_reason()
                assert study.resolved_configuration(ours, study.configuration()) == json.loads(
                    json.dumps(runner.resolved_configuration(original, config)))
    source = study.source_metadata(study.HERE.parent)
    for applicable, api in ((True, sparse_smart), (False, None)):
        provenance = runner._implementation_provenance(api, generate_data)
        expected = source["implementation"][str(applicable).lower()]
        assert expected["manifest"] == provenance["implementation_manifest"]
        assert expected["fingerprint"] == provenance["implementation_fingerprint"]


@pytest.mark.parametrize("kwargs", [dict(seed_ids=(0, 0)), dict(models=(3,)), dict(experiments=(4,)),
                                    dict(setting_index=0), dict(profile="none")])
def test_invalid_scope_is_rejected(kwargs):
    with pytest.raises(ValueError):
        study.make_plan(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(iteration_budgets=(4, 2)), dict(iteration_budgets=(2, 2)),
    dict(checkpoint_interval=0), dict(stationarity_tol=float("nan")), dict(stationarity_tol=0),
    dict(init_penalties=(0.,)), dict(penalties_u=(.01, .01)),
    dict(validation_iterations=(25, 10)), dict(validation_iterations=(10, 10)),
    dict(validation_iterations=(0,)), dict(validation_iterations=(-10,)),
    dict(validation_iterations=(True,)), dict(validation_iterations=(10.,))])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        study.configuration(**kwargs)


def test_validation_schedule_is_additive_bounded_and_part_of_plan_identity():
    config = study.configuration(iteration_budgets=(40, 100), checkpoint_interval=25,
                                 validation_iterations=(10, 25, 30, 150))
    setting = study.experiment_settings(0, 0)[0]
    resolved = study.resolved_configuration(setting, config)
    assert resolved["validation_schedule"] == [0, 10, 25, 30, 40, 50, 75, 100]
    assert resolved["validation_iterations"] == [10, 25, 30, 150]
    assert resolved["validation_state_policy"] == "checkpoints_and_validation_bests"
    plan = study.make_plan(models=(0,), experiments=(0,), seed_ids=(0,), config=config)
    other = study.make_plan(models=(0,), experiments=(0,), seed_ids=(0,), config={
        **config, "validation_iterations": [10, 25, 35, 150]})
    assert plan["plan_fingerprint"] != other["plan_fingerprint"]
    study.validate_plan(plan)
    study.validate_plan(other)


@pytest.mark.parametrize("arguments, expected", [
    (["10", "25", "50"], [10, 25, 50]),
    (["10,25", "50"], [10, 25, 50]),
    (["none"], []),
])
def test_planner_cli_canonicalizes_validation_iteration_list(tmp_path, arguments, expected):
    assert study.main(["plan", "--output-root", str(tmp_path), "--models", "0",
        "--experiments", "0", "--seed-ids", "0", "--validation-iterations", *arguments]) == 0
    assert study.read_json(tmp_path/"study-plan.json")["configuration"]["validation_iterations"] == expected


@pytest.mark.parametrize("arguments", [["none", "10"], ["10,"], ["10.5"], ["0"], ["25,10"],
                                      ["+10"], [" 10"], ["10 "], ["10, 25"]])
def test_planner_cli_rejects_invalid_validation_iteration_list(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        study.main(["plan", "--output-root", str(tmp_path), "--models", "0",
            "--experiments", "0", "--seed-ids", "0", "--validation-iterations", *arguments])
    assert error.value.code == 2
    assert not (tmp_path/"study-plan.json").exists()


def test_old_plan_and_records_keep_periodic_only_identity(tmp_path):
    legacy = study.configuration(iteration_budgets=(2, 4), checkpoint_interval=2,
                                 penalties_u=(.01,), penalties_v=(.01,))
    legacy.pop("validation_iterations")
    plan = study.make_plan(models=(0,), experiments=(0,), seed_ids=(0,), setting_index=0,
                           config=legacy)
    fingerprint = plan["plan_fingerprint"]
    study.write_plan(plan, tmp_path)
    study.validate_plan(plan)
    assert "validation_iterations" not in plan["configuration"]
    path, record, _ = save_cell(tmp_path, plan, plan["cells"][0])
    assert "validation_schedule" not in record["configuration"]
    assert "validation_iterations" not in record["configuration"]
    assert "validation_state_policy" not in record["configuration"]
    assert study.aggregate(tmp_path)["execution_complete"]
    assert study.read_json(tmp_path/"study-plan.json")["plan_fingerprint"] == fingerprint


def test_resume_requires_matching_configuration_and_table(tmp_path):
    plan = small_plan(tmp_path)
    assert study.write_plan(plan, tmp_path)["created"] == plan["created"]
    other = deepcopy(plan)
    other["configuration"]["stationarity_tol"] = 1e-5
    with pytest.raises(ValueError, match="fingerprint"):
        study.write_plan(other, tmp_path)
    (tmp_path/"work-items.tsv").write_text("m0_e0_s0_k0\t0\t0\t1\t0\n")
    with pytest.raises(ValueError, match="table differs"):
        study.aggregate(tmp_path)


def test_empty_aggregate_preserves_expected_scope_for_existing_summarizer(tmp_path):
    import summarize_sparse_smart_budget_study as summary
    plan = small_plan(tmp_path)
    report = study.aggregate(tmp_path)
    assert report["recorded_cells"] == 0 and report["missing_cells"] == [plan["cells"][0]["task_id"]]
    assert not report["execution_complete"] and not report["budget_coverage_complete"]
    manifest = study.read_json(tmp_path/"results/budget_study_manifest.json")
    assert manifest["attempt_status"] == "incomplete" and manifest["expected_cells"] == 1
    result = summary.summarize(tmp_path/"results", manifest_scope=True)
    assert result["expected_cells"] == result["missing_cells"] == 1
    assert result["regenerated_unique_datasets"] == 0


def test_controlled_complete_record_collects_into_existing_scientific_summary(tmp_path, monkeypatch):
    # Reuse the existing synthetic checkpoint fixture; no estimator is fitted.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    from test_summarize_sparse_smart_budget_study import fixture, generator
    import summarize_sparse_smart_budget_study as summary
    record = fixture()
    plan = study.make_plan(models=(0,), experiments=(2,), seed_ids=(0,), setting_index=2,
                           config=record["configuration"]["runner"])
    study.write_plan(plan, tmp_path)
    cell = plan["cells"][0]
    path, metadata, _ = save_cell(tmp_path, plan, cell)
    for key in ("implementation_fingerprint_scheme", "implementation_manifest", "implementation_fingerprint"):
        record[key] = metadata[key]
    study.atomic_json(record, path)
    assert study.aggregate(tmp_path)["execution_complete"]
    report = summary.summarize(tmp_path/"results", manifest_scope=True, generate_data_fn=generator)
    assert report["expected_cells"] == report["recorded_cells"] == 1
    assert report["missing_cells"] == 0 and report["verified_factor_states"] == 10
    assert report["cells"][0]["status"] == "complete"


def test_collection_keeps_valid_partial_and_failed_cells_without_claiming_budget_completion(tmp_path):
    plan = small_plan(tmp_path, seeds=(0, 1, 2))
    for cell, status in zip(plan["cells"], ("complete", "partial", "all_candidates_failed")):
        save_cell(tmp_path, plan, cell, status=status)
    report = study.aggregate(tmp_path)
    assert report["execution_complete"] and not report["budget_coverage_complete"]
    assert report["status_counts"] == dict(complete=1, partial=1, all_candidates_failed=1)
    assert report["scientific_validation"] == "pending_existing_summarizer"
    assert len(list((tmp_path/"results").rglob("BudgetStudy_result_*.json"))) == 3


def test_inapplicable_cell_remains_an_explicit_record(tmp_path):
    plan = small_plan(tmp_path, experiment=2, setting=0)
    save_cell(tmp_path, plan, plan["cells"][0], status="inapplicable")
    report = study.aggregate(tmp_path)
    assert report["expected_inapplicable"] == report["recorded_cells"] == 1
    assert report["status_counts"] == dict(inapplicable=1)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(rd_seed_id=99),
    lambda r: r["configuration"]["runner"].update(stationarity_tol=1e-5),
    lambda r: r.update(implementation_fingerprint="b"*64),
    lambda r: r["implementation_manifest"]["files"][0].update(sha256="b"*64),
    lambda r: r.update(applicable=False),
])
def test_mismatched_record_is_reported_and_excluded(tmp_path, mutation):
    plan = small_plan(tmp_path)
    path, record, _ = save_cell(tmp_path, plan, plan["cells"][0])
    mutation(record)
    study.atomic_json(record, path)
    report = study.aggregate(tmp_path)
    assert report["recorded_cells"] == 0 and not report["execution_complete"]
    assert report["errors"][-1]["error"] == "invalid_cell_artifact"
    assert not list((tmp_path/"results").rglob("BudgetStudy_result_*.json"))


def test_duplicate_result_and_wrong_percell_manifest_mapping_are_rejected(tmp_path):
    plan = small_plan(tmp_path)
    path, record, manifest_path = save_cell(tmp_path, plan, plan["cells"][0])
    extra = path.with_name("BudgetStudy_result_unexpected.json")
    study.atomic_json(record, extra)
    assert study.aggregate(tmp_path)["recorded_cells"] == 0
    extra.unlink()
    manifest = study.read_json(manifest_path)
    manifest["cells"][0]["seed_id"] = 99
    study.atomic_json(manifest, manifest_path)
    assert study.aggregate(tmp_path)["recorded_cells"] == 0


@pytest.mark.parametrize("broken", ["record", "manifest", "manifest_row"])
def test_malformed_cell_does_not_prevent_collection_of_next_valid_cell(tmp_path, broken):
    plan = small_plan(tmp_path, seeds=(0, 1))
    path, _, manifest_path = save_cell(tmp_path, plan, plan["cells"][0])
    save_cell(tmp_path, plan, plan["cells"][1])
    if broken == "manifest_row":
        manifest = study.read_json(manifest_path)
        manifest["cells"] = [None]
        study.atomic_json(manifest, manifest_path)
    else:
        (path if broken == "record" else manifest_path).write_text("[]")
    report = study.aggregate(tmp_path)
    assert report["recorded_cells"] == 1 and not report["execution_complete"]
    assert report["cells"][0]["task_id"] == plan["cells"][1]["task_id"]


def test_nonzero_slurm_exit_preserves_record_but_marks_execution_incomplete(tmp_path):
    plan = small_plan(tmp_path)
    cell = plan["cells"][0]
    save_cell(tmp_path, plan, cell)
    (tmp_path/"tasks"/cell["task_id"]/"launcher-exit-code.txt").write_text("7\n")
    report = study.aggregate(tmp_path)
    assert report["recorded_cells"] == 1 and not report["execution_complete"]
    assert any(error.get("exit_code") == 7 for error in report["errors"])


def test_interrupted_attempt_is_audited_without_hiding_written_record(tmp_path):
    plan = small_plan(tmp_path)
    save_cell(tmp_path, plan, plan["cells"][0], attempt_status="interrupted", errors=[dict(message="interrupted")])
    report = study.aggregate(tmp_path)
    assert report["recorded_cells"] == 1 and not report["execution_complete"]
    assert {error["error"] for error in report["errors"]} == {"driver_error", "cell_execution_incomplete"}


def test_reaggregation_archives_old_output_and_does_not_reuse_now_missing_record(tmp_path):
    plan = small_plan(tmp_path)
    path, _, _ = save_cell(tmp_path, plan, plan["cells"][0])
    assert study.aggregate(tmp_path)["recorded_cells"] == 1
    path.unlink()
    assert study.aggregate(tmp_path)["recorded_cells"] == 0
    assert not list((tmp_path/"results").rglob("BudgetStudy_result_*.json"))
    assert len(list((tmp_path/"results_aggregation_attempts").rglob("BudgetStudy_result_*.json"))) == 1


def test_result_symlink_cannot_read_outside_declared_run_root(tmp_path):
    root = tmp_path/"run"
    plan = small_plan(root)
    path, record, _ = save_cell(root, plan, plan["cells"][0])
    outside = tmp_path/"outside.json"
    study.atomic_json(record, outside)
    path.unlink()
    path.symlink_to(outside)
    report = study.aggregate(root)
    assert report["recorded_cells"] == 0
    assert "escapes declared root" in report["errors"][-1]["message"]


def test_unrelated_output_and_outside_seed_paths_are_rejected(tmp_path):
    root = tmp_path/"run"
    small_plan(root)
    (root/"results").mkdir()
    (root/"results/user-file.txt").write_text("preserve")
    with pytest.raises(ValueError, match="unrelated"):
        study.aggregate(root)
    assert (root/"results/user-file.txt").read_text() == "preserve"
    with pytest.raises(ValueError, match="escapes declared root"):
        study.make_plan(seed_file=tmp_path/"outside.csv")
