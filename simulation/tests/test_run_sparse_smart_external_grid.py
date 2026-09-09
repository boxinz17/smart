"""Batch profiles and verified reuse must preserve the declared experiment."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_sparse_smart_external_grid as batch


def test_full_and_difficult_profiles_preserve_saved_dimensions():
    full = [(m, e, s) for m in range(3) for e in range(4)
            for s in batch.settings_for_profile(m, e, "full")]
    difficult = [(m, e, s) for m in range(3) for e in range(4)
                 for s in batch.settings_for_profile(m, e, "difficult")]
    assert len(full) * 5 == 360
    assert sum(s.inapplicability_reason() is None for _, _, s in full) * 5 == 315
    assert len(difficult) * 5 == 45
    assert all(s.target_rank == 5 and s.inapplicability_reason() is None for _, _, s in difficult)
    assert {(e, s.source_rank, s.sigma0) for _, e, s in difficult} == {
        (2, 5, .01), (2, 7, .01), (3, 10, .5)}
    with pytest.raises(ValueError):
        batch.settings_for_profile(0, 0, "unknown")


def test_dry_run_passes_iteration_and_solver_configuration(tmp_path, capsys):
    assert batch.main(["--iterations", "2000", "--profile", "difficult",
        "--initialization-spectrum", "projected", "--refinement-solver", "anchor_projected",
        "--output-root", str(tmp_path / "out"), "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["expected_cells"] == record["expected_applicable"] == 45
    assert record["expected_inapplicable"] == 0
    assert record["configuration"]["iterations"] == 2000
    assert record["configuration"]["initialization_spectrum"] == "projected"
    assert record["configuration"]["refinement_solver"] == "anchor_projected"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("args", [["--iterations", "-1"],
    ["--profile", "difficult", "--experiments", "0"], ["--seed-count", "0"]])
def test_invalid_batch_fails_before_writing(tmp_path, args):
    with pytest.raises(SystemExit):
        batch.main(["--output-root", str(tmp_path / "out"), *args])
    assert not (tmp_path / "out").exists()


def test_reuse_roots_must_not_overlap(tmp_path):
    for reuse in (tmp_path, tmp_path / "child", tmp_path.parent):
        with pytest.raises(SystemExit):
            batch.main(["--output-root", str(tmp_path), "--reuse-root", str(reuse), "--dry-run"])


def fixture_task(tmp_path):
    setting = batch.settings_for_profile(0, 3, "full")[1]
    config = batch.RunnerConfig(initialization_spectrum="projected", refinement_solver="anchor_projected")
    task = (0, 3, setting, 0, 123, tmp_path / "new", config, tmp_path / "prior")
    prior = batch.result_path(task[-1], model="model1", experiment="exp4", setting=setting, seed_id=0)
    destination = batch.result_path(task[5], model="model1", experiment="exp4", setting=setting, seed_id=0)
    prior.parent.mkdir(parents=True)
    prior.write_text("checkpoint")
    return task, prior, destination


def test_reuse_occurs_only_after_normal_checkpoint_verification(tmp_path, monkeypatch):
    task, prior, destination = fixture_task(tmp_path)
    record = dict(status="complete", avg_err=.01, n_train=200, n_validation=100,
        failure_reason=None, selected_iteration=2, termination_reason="max_iterations",
        selection_history=[{}] * 9, fit_errors=[], fit_time_sec=12.)
    calls = []
    def verify(**kwargs):
        calls.append(kwargs)
        assert kwargs["destination"] == prior and not destination.exists()
        assert kwargs["config"] is task[6]
        return "skipped", record
    monkeypatch.setattr(batch, "run_setting", verify)
    result = batch.fit_cell(task)
    assert len(calls) == 1
    assert result["outcome"] == "reused" and result["reused_from"] == str(prior)
    assert json.loads(destination.read_text()) == record
    assert prior.read_text() == "checkpoint"


def test_mismatched_checkpoint_is_never_copied_or_silently_refitted(tmp_path, monkeypatch):
    task, prior, destination = fixture_task(tmp_path)
    def reject(**kwargs):
        raise ValueError("Implementation differs")
    monkeypatch.setattr(batch, "run_setting", reject)
    with pytest.raises(ValueError, match="Implementation differs"):
        batch.fit_cell(task)
    assert not destination.exists()
    assert prior.read_text() == "checkpoint"
