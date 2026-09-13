"""Retrospective endpoint scoring, with the existing tiny operational fixture."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module(HERE / "test_summarize_sparse_smart_v2_campaign.py", "retrospective_fixtures")
campaign = fixtures.campaign
retro = module(HERE.parent / "run_sparse_smart_v2_retrospective_bic.py", "retrospective_under_test")


def retrospective_plan(old_root, old_plan, tmp_path):
    root = tmp_path / "retrospective"
    source = root / "source"
    source.mkdir(parents=True)
    (source / "fixture.py").write_text("# frozen retrospective fixture\n")
    retro.write_json(root / "source-manifest.json", dict(schema_version=1,
                     files={"fixture.py": retro.legacy.sha(source / "fixture.py")}))
    plan = dict(schema_version=1, method=retro.METHOD, root=str(root), n_cases=len(old_plan["cases"]),
                source_campaign_root=str(old_root), source_manifest_sha256=retro.legacy.sha(root / "source-manifest.json"),
                cases=[dict(case_id=case["case_id"], case=case, source_root=str(old_root),
                            reference_plan_sha256=retro.legacy.sha(old_root / "plan.json"),
                            reference_preparation_sha256=retro.legacy.sha(old_root / "preparation.json"),
                            reference_source_manifest_sha256=old_plan["source_manifest_sha256"])
                       for case in old_plan["cases"]])
    plan["plan_fingerprint"] = retro.legacy.digest(plan)
    retro.write_json(root / "plan.json", plan)
    return root / "plan.json", plan


def complete(campaign, monkeypatch, tmp_path, *, rrr=True):
    import external_validation_data
    generator = external_validation_data.generate_external_validation
    def noisy_response(**kwargs):
        data = generator(**kwargs)
        # The baseline fixture has a perfect rank-five response. Add a fixed
        # residual outside the design column space so Gaussian BIC is finite.
        data["Y"][-1, 0] = .25
        return data
    monkeypatch.setattr(external_validation_data, "generate_external_validation", noisy_response)
    old_root, old_plan, results = fixtures.completed_campaign(campaign, monkeypatch, rrr=rrr)
    path, plan = retrospective_plan(old_root, old_plan, tmp_path)
    retro.prepare(path)
    return path, plan["cases"][0]["case_id"], old_root, results


@pytest.mark.parametrize("rrr", [False, True])
def test_both_endpoints_use_existing_states_and_original_validation_baseline(campaign, monkeypatch, tmp_path, rrr):
    path, cid, old_root, originals = complete(campaign, monkeypatch, tmp_path, rrr=rrr)
    # Retrospective code has no dependency on any fit/generator entry point.
    monkeypatch.setattr(fixtures.pilot, "fit", lambda *a, **k: pytest.fail("must not refit"))
    report = retro.run_case(path, cid)
    assert report["success"], report["errors"]
    assert report["complete_execution_coverage"] and not report["complete_tuning_coverage"]
    assert report["counts"]["initializer_exclusion"] == (1 if rrr else 2)
    assert report["eligible_tasks"] == 2
    baseline = min((row for row in originals if row["success"]),
                   key=lambda row: (row["selected_metrics"]["validation_mse"], row["task"]["task_id"]))
    assert report["arms"]["original_validation"]["task_id"] == baseline["task"]["task_id"]
    assert report["arms"]["bic_validation_selected"]["endpoint"] == "selected"
    assert report["arms"]["bic_validation_terminal"]["endpoint"] == "terminal"
    assert not report["fitting_performed"] and not report["bic_uses_validation_or_truth"]
    assert all(path.suffix not in (".npz", ".gz") for path in path.parent.rglob("*"))
    if rrr:
        row = next(row for row in report["candidates"] if row["selected"]["fit_method"] == "target_rrr")
        for endpoint in ("selected", "terminal"):
            score = row[endpoint]["bic"]
            assert score["support_u"] is None and score["support_v"] is None
            assert score["model_dimension"] == 5 * (6 + 6 - 5)
            assert row[endpoint]["iteration"] == 0


def test_exact_ties_and_truth_do_not_change_bic_selection():
    def endpoint(task, score, val, truth, prefix):
        return dict(task_id=task, bic=dict(score=score), metrics=dict(validation_mse=val, coefficient_rmse=truth),
                    endpoint=prefix)
    candidates = [dict(selected=endpoint(2, 8., .1, 99., "selected"), terminal=endpoint(2, 1., 100., 88., "terminal")),
                  dict(selected=endpoint(1, 8., .1, 100., "selected"), terminal=endpoint(1, 5., .0, .0, "terminal"))]
    arms = retro.select_arms(candidates)
    assert {key: value["task_id"] for key, value in arms.items()} == dict(
        original_validation=1, bic_validation_selected=1, bic_validation_terminal=2)
    for candidate in candidates:
        for endpoint in candidate.values():
            endpoint["metrics"]["coefficient_rmse"] = np.nan
            endpoint["metrics"]["training_prediction_mse"] = -np.inf
    assert retro.select_arms(candidates) == arms
    assert retro.select_arms([]) == dict.fromkeys(retro.ARMS)


def test_selected_and_terminal_have_independent_geometry():
    from sparse_smart.chart import AnchorChart
    case = dict(p=3, q=3, rank=1, free_directions=[2, 2], support_limits=[1, 1])
    P = np.array([[.8], [.6], [0.]])
    Q = np.array([[.6], [.8], [0.]])
    d = np.array([2.])
    arrays = dict(states_schema_version=np.array(2), free_rows_u=np.array([0, 1]), free_rows_v=np.array([0, 1]))
    for endpoint, anchor in (("selected", 0), ("terminal", 1)):
        chart = AnchorChart(3, 3, [anchor], [anchor], np.eye(1), np.eye(1))
        arrays[f"{endpoint}_state"] = chart.initial_state(P, d, Q)
        for key in ("anchors_u", "anchors_v", "center_u", "center_v"):
            arrays[f"{endpoint}_{key}"] = getattr(chart, key)
        for key, value in (("P", P), ("d", d), ("Q", Q)):
            arrays[f"{endpoint}_{key}"] = value
    configuration = dict(margins=dict(anchor_min=.04, d_lower=.001, d_upper=12., gap=.0001))
    for endpoint in ("selected", "terminal"):
        coefficient, args = retro.endpoint_state(arrays, endpoint, case, dict(left=np.eye(3), right=np.eye(3)),
                                                 configuration, direct=False)
        np.testing.assert_allclose(coefficient, (P * d) @ Q.T)
        assert np.count_nonzero(args["weighted_u"][args["penalized_u"]]) == 0
    del arrays["selected_anchors_u"]
    with pytest.raises(ValueError, match="own-chart geometry"):
        retro.endpoint_state(arrays, "selected", case, dict(left=np.eye(3), right=np.eye(3)), configuration, direct=False)


@pytest.mark.parametrize("damage", ["missing", "hash", "execution", "stagnation", "metric"])
def test_failures_are_reported_not_silently_excluded(campaign, monkeypatch, tmp_path, damage):
    path, cid, old_root, originals = complete(campaign, monkeypatch, tmp_path)
    folder = old_root / "tasks/00001"
    result = originals[1]
    assert result["success"]
    if damage == "missing":
        (folder / "states.npz").unlink()
    elif damage == "hash":
        (folder / "result.json").write_text((folder / "result.json").read_text() + " ")
    elif damage == "execution":
        fixtures.execution_records(folder, result, 7)
    elif damage == "metric":
        result["terminal_metrics"]["validation_mse"] += 1.
        fixtures.seal(folder, result)
    else:
        result.update(success=False, fit_status="numerical_stagnation", avg_err=None,
                      termination_reason="numerical_stagnation")
        fixtures.seal(folder, result)
    report = retro.run_case(path, cid)
    assert not report["success"] and report["errors"]
    assert report["eligible_tasks"] == 1
    assert all(winner["task_id"] != 1 for winner in report["arms"].values())
    assert report["complete_execution_coverage"] is (damage == "stagnation")


def test_bic_scorer_receives_only_training_observations(campaign, monkeypatch, tmp_path):
    path, cid, _, _ = complete(campaign, monkeypatch, tmp_path)
    actual = retro.bic_score
    calls = []
    def checked(design, response, coefficient, **kwargs):
        assert set(kwargs) <= {"rank", "design_rank", "direct_rrr", "free_directions", "weighted_u", "weighted_v", "penalized_u", "penalized_v"}
        calls.append((design.copy(), response.copy()))
        return actual(design, response, coefficient, **kwargs)
    monkeypatch.setattr(retro, "bic_score", checked)
    report = retro.run_case(path, cid)
    assert report["success"] and len(calls) == 4
    with np.load(Path(report["reference_subroot"]) / "cases" / cid / "data.npz") as data:
        for x, y in calls:
            np.testing.assert_array_equal(x, data["X"])
            np.testing.assert_array_equal(y, data["Y"])


def test_restart_reuses_verified_output_but_rejects_changed_inputs(campaign, monkeypatch, tmp_path):
    path, cid, old_root, _ = complete(campaign, monkeypatch, tmp_path)
    first = retro.run_case(path, cid)
    assert first["success"], first["errors"]
    monkeypatch.setattr(retro, "score_endpoints", lambda *a, **k: pytest.fail("restart must reuse output"))
    assert retro.run_case(path, cid) == first
    output = path.parent / "cases" / f"{cid}.json"
    before = output.read_bytes()
    (old_root / "tasks/00001/process-status.tsv").write_text("changed")
    with pytest.raises(ValueError, match="candidate inputs changed"):
        retro.run_case(path, cid)
    assert output.read_bytes() == before


def test_prepared_slice_mutation_and_plan_mutation_are_rejected(campaign, monkeypatch, tmp_path):
    path, cid, _, _ = complete(campaign, monkeypatch, tmp_path)
    assert retro.prepare(path)["success"]
    inputs = path.parent / "inputs" / f"{cid}.json"
    inputs.write_text(inputs.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        retro.run_case(path, cid)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        retro.prepare(path)
    plan = retro.legacy.read(path)
    plan["cases"][0]["case"]["seed_id"] = 99
    retro.write_json(path, plan)
    with pytest.raises(ValueError, match="plan fingerprint"):
        retro.load_plan(path)


def test_preparation_checks_new_and_old_source_provenance(campaign, monkeypatch, tmp_path):
    old_root, old_plan, _ = fixtures.completed_campaign(campaign, monkeypatch)
    path, _ = retrospective_plan(old_root, old_plan, tmp_path)
    (path.parent / "source/fixture.py").write_text("# changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        retro.prepare(path)
    assert not (path.parent / "preparation.json").exists()


def test_real_case_result_roundtrips_through_campaign_summary(campaign, monkeypatch, tmp_path):
    path, cid, _, _ = complete(campaign, monkeypatch, tmp_path)
    report = retro.run_case(path, cid)
    assert report["success"], report["errors"]
    summary = module(HERE.parent / "summarize_sparse_smart_v2_retrospective_bic.py", "retrospective_summary_integration")
    collected = summary.summarize(path.parent, make_plots=False)
    assert collected["success"], collected["errors"]
    assert collected["planned_cases"] == 1
