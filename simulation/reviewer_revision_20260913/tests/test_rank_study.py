"""Validation-selection and bookkeeping tests; no production simulations."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))
from reviewer_revision_20260913 import rank_study


def test_rank_grid_never_uses_true_rank_and_preserves_declared_order():
    config = rank_study.rank_configuration(0, ranks=[7, 3, 5, 3], source_ranks=[5, 10])
    case = dict(p=20, q=12, target_rank=1, latent_source_rank=2, source_rank=1)
    candidates = rank_study.build_rank_candidates(case, config)
    assert [(row["fitted_rank"], row["source_rank"]) for row in candidates] == [
        (7, 5), (7, 10), (3, 5), (3, 10), (5, 5), (5, 10)]
    assert not candidates[0]["admissible"]
    assert candidates[0]["exclusion_reason"] == "fitted_rank_exceeds_source_dimension"
    assert candidates == rank_study.build_rank_candidates(dict(p=20, q=12), config)


@pytest.mark.parametrize("ranks,sources", [([True], [10]), ([0], [10]), ([], [10]),
                                          ([3], []), ([7], [5]), ([13], [15])])
def test_reject_bad_or_unreachable_operational_grids(ranks, sources):
    with pytest.raises(ValueError):
        config = rank_study.rank_configuration(0, ranks, sources)
        rank_study.build_rank_candidates(dict(p=20, q=12), config)


def test_selection_uses_validation_instead_of_claimed_scores_or_rank_labels():
    xv = np.eye(2)
    yv = np.array([[2.], [1.]])
    records = [dict(eligible=True, coefficient=np.zeros((2, 1)), validation_mse=-100.,
                    population_risk=0., fitted_rank=1),
               dict(eligible=True, coefficient=yv.copy(), validation_mse=100.,
                    population_risk=1e20, fitted_rank=7),
               dict(eligible=False, coefficient=yv.copy(), error="algorithm_failure")]
    selected = rank_study.choose_rank_results(records, xv, yv)
    assert selected["success"] and selected["selected_index"] == 1
    assert selected["validation_mse"] == 0
    assert not selected["audit"][2]["eligible"]
    np.testing.assert_array_equal(selected["coefficient"], yv)
    selected["coefficient"][0, 0] = 99
    assert records[1]["coefficient"][0, 0] == 2


def test_selection_ties_failures_and_nonfinite_predictions_have_audits():
    records = [dict(eligible=True, coefficient=np.array([[np.nan]])),
               dict(eligible=True, coefficient=np.zeros((2, 1))),
               dict(eligible=True, coefficient=np.array([[1.]])),
               dict(eligible=True, coefficient=np.array([[-1.]])),
               dict(eligible=False, error="fit_failed")]
    selected = rank_study.choose_rank_results(records, np.ones((1, 1)), np.zeros((1, 1)))
    assert selected["selected_index"] == 2
    assert len(selected["audit"]) == len(records)
    assert [row["eligible"] for row in selected["audit"]] == [False, False, True, True, False]
    failed = rank_study.choose_rank_results(records[:2], np.ones((1, 1)), np.zeros((1, 1)))
    assert not failed["success"] and failed["coefficient"] is None


def test_competitor_grids_tune_ranks_and_source_dimensions_without_truth():
    config = rank_study.rank_configuration(0)
    libraries = rank_study.competitor_libraries(dict(p=100, q=50), config)
    assert {row["rank"] for row in libraries["target_ridge_rrr"]} == {3, 5, 7}
    assert {row["source_rank"] for row in libraries["source_subspace_rrr"]} == {10, 15}
    assert len(libraries["source_subspace_rrr"]) == 6
    assert len(libraries["oracle_subspace_rrr"]) == 3
    assert all("source_rank" not in row for row in libraries["oracle_subspace_rrr"])
    assert all("rank" not in row for row in libraries["nuclear_contrast"])
    assert all("rank" not in row for row in libraries["ridge_to_source"])


def test_representative_plan_is_explicit_and_does_not_run_fits(tmp_path):
    plan = rank_study.build_plan(tmp_path, seeds=[9], iterations=0)
    assert plan["n_cases"] == plan["n_tasks"] == 6
    assert {case["n_train"] for case in plan["cases"]} == {80, 200}
    assert {case["family"] for case in plan["cases"]} == {"reference", "containment", "fitted_source"}
    assert all(case["left_angle_deg"] == 30 for case in plan["cases"] if case["family"] == "containment")
    assert not (tmp_path / "rank_tasks").exists()
    assert (tmp_path / "rank-plan.json").exists()
    first = plan["cases"][0]["case_id"]
    cells = plan["v2_libraries"][first]
    config = plan["configuration"]
    expected = 1 + len(config["init_penalties"]) * sum(
        not (left == right == 0) for left in config["penalties_u"] for right in config["penalties_v"])
    assert len(cells) == 6 and all(len(cell["candidates"]) == expected for cell in cells)
    with pytest.raises(FileExistsError):
        rank_study.build_plan(tmp_path, seeds=[9], iterations=0)


def test_slurm_guard_precedes_fitting_and_plan_reads(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        rank_study.run_task(tmp_path, 0)
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    with pytest.raises(RuntimeError, match="scratch2"):
        rank_study.run_task(Path("/not-scratch2/guard-only"), 0)


def test_cell_failure_is_preserved_and_does_not_erase_other_candidates(tmp_path, monkeypatch):
    cells = [dict(cell_index=0, fitted_rank=1, source_rank=1, admissible=True, exclusion_reason=None),
             dict(cell_index=1, fitted_rank=1, source_rank=2, admissible=True, exclusion_reason=None)]
    def fake_fit(case, data, config, directory, variant):
        if case["source_rank"] == 1:
            raise RuntimeError("deliberate cell failure")
        coefficient = np.array([[2.]])
        winner = dict(coefficient=coefficient, prediction=coefficient, candidate_index=0,
                      spec=dict(rank=1, source_rank=2), method="target_rrr", selected_iteration=0,
                      history=[])
        return {"v2": winner}, [dict(eligible=True, elapsed_seconds=.01)], .01
    monkeypatch.setattr(rank_study.runner, "fit_v2", fake_fit)
    pool, runtime = rank_study._fit_rank_v2({}, {}, {}, cells, tmp_path)
    assert len(pool["v2"]) == len(pool["initializer_only"]) == 2
    assert not pool["v2"][0]["eligible"] and pool["v2"][1]["eligible"]
    assert all(not row["eligible"] for row in pool["initializer_only"])
    selected = rank_study.choose_rank_results(pool["v2"], np.ones((1, 1)), np.array([[2.]]))
    assert selected["success"] and selected["selected_record"]["cell_index"] == 1
    assert runtime["n_candidates"] == runtime["n_eligible"] == 1
    assert (tmp_path / "rank-cells.json").exists()


def test_benchmark_evidence_saves_all_observed_frame_candidates(tmp_path):
    import json
    from sparse_smart_v2 import ObservedSource, prepare_source
    x = np.eye(4)
    source = np.zeros((4, 3)); source[0, 0] = 2.
    data = dict(X=x, Y=x @ source, X_validation=x, Y_validation=x @ source, C0=source)
    observed = prepare_source(ObservedSource(source), p=4, q=3, source_rank=2)
    library = [dict(method="source_subspace_rrr", rank=1, source_rank=rank) for rank in (1, 2)]
    selected = rank_study._fit_competitor_library(data, library, tmp_path, observed)
    assert selected.selected_index == 0
    archive = json.loads((tmp_path / "candidates.json").read_text())
    assert archive["complete"] and archive["n_eligible"] == 2
    assert all((tmp_path / f"candidate-{i:04d}.npz").exists() for i in range(2))
    diagnostics = archive["candidates"][1]["diagnostics"]
    assert diagnostics["source_frame_convention"] == "supplied_common_observed_source_decomposition"
    assert diagnostics["includes_observed_null_completion"]


def test_uncertified_baseline_is_saved_but_never_selected(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    fit = rank_study.competitors.CompetitorFit(np.zeros((1, 1)), "target_rrr", {}, {"certified": False})
    monkeypatch.setattr(rank_study.competitors, "fit_competitor", lambda *args, **kwargs: fit)
    data = dict(X=np.ones((1, 1)), Y=np.ones((1, 1)), X_validation=np.ones((1, 1)),
                Y_validation=np.ones((1, 1)), C0=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="No eligible") as caught:
        rank_study._fit_competitor_library(data, [dict(method="target_rrr", rank=1)], tmp_path,
                                          SimpleNamespace(left=np.eye(1), right=np.eye(1)))
    assert caught.value.candidate_results[0]["status"] == "uncertified"
    archive = json.loads((tmp_path / "candidates.json").read_text())
    assert archive["complete"] and archive["n_eligible"] == 0 and archive["selected_index"] is None
    assert (tmp_path / "candidate-0000.npz").exists()
