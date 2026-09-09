"""Tests for the deliberately minimal restricted-RRR simulation driver."""

from __future__ import annotations

from enum import Enum
import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_restricted_rrr.py"
SPEC = importlib.util.spec_from_file_location("run_restricted_rrr", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_experiment_grids_match_run_smart() -> None:
    exp1 = runner.experiment_settings(0, 0)
    assert [setting.n for setting in exp1] == [200, 400, 600, 800, 1000]
    assert [setting.suffix for setting in exp1] == [
        "n=200",
        "n=400",
        "n=600",
        "n=800",
        "n=1000",
    ]
    assert {(setting.target_rank, setting.source_rank) for setting in exp1} == {
        (5, 10)
    }

    exp2 = runner.experiment_settings(1, 1)
    assert [setting.target_rank for setting in exp2] == [1, 3, 5, 7, 9, 11]
    assert [setting.suffix for setting in exp2] == [
        "r=1",
        "r=3",
        "r=5",
        "r=7",
        "r=9",
        "r=11",
    ]
    assert exp2[-1].inapplicability_reason() == "target_rank_exceeds_source_rank"

    exp3 = runner.experiment_settings(2, 2)
    assert [setting.source_rank for setting in exp3] == [0, 3, 5, 7, 10, 15, 20]
    assert exp3[0].inapplicability_reason() == "source_rank_must_be_positive"
    assert exp3[1].inapplicability_reason() == "target_rank_exceeds_source_rank"
    assert all(setting.inapplicability_reason() is None for setting in exp3[2:])

    exp4 = runner.experiment_settings(0, 3)
    assert [setting.sigma0 for setting in exp4] == [
        0.0,
        0.01,
        0.02,
        0.05,
        0.1,
        0.5,
    ]
    assert [setting.suffix for setting in exp4] == [
        "sigma0=0.0",
        "sigma0=0.01",
        "sigma0=0.02",
        "sigma0=0.05",
        "sigma0=0.1",
        "sigma0=0.5",
    ]


def test_model_dimensions_and_sample_sizes_match_run_smart() -> None:
    expected = {
        0: ((100, 50), [200, 400, 600, 800, 1000]),
        1: ((150, 100), [300, 500, 700, 1000, 1200]),
        2: ((300, 200), [500, 700, 1000, 1200, 1500]),
    }
    for model_id, (shape, sample_sizes) in expected.items():
        settings = runner.experiment_settings(model_id, 0)
        assert [(setting.p, setting.q) for setting in settings] == [shape] * 5
        assert [setting.n for setting in settings] == sample_sizes


def test_checked_in_seed_vector_and_result_name() -> None:
    seeds = runner.load_experiment_seeds()
    assert len(seeds) == 100
    assert int(seeds[0]) == 2_103_139_804

    setting = runner.experiment_settings(0, 0)[0]
    path = runner.result_path(
        Path("results"),
        model="model1",
        experiment="exp1",
        setting=setting,
        seed_id=7,
    )
    assert path == Path(
        "results/model1/exp1/"
        "RestrictedRRR_result_model1_exp1_n=200_rd_seed_id=7.pkl"
    )


def _tiny_data(**kwargs: object) -> dict[str, np.ndarray]:
    n = int(kwargs["n"])
    p = int(kwargs["p"])
    q = int(kwargs["q"])
    return {
        "X": np.arange(n * p, dtype=float).reshape(n, p),
        "Y": np.zeros((n, q)),
        "C_star": np.zeros((p, q)),
        "C0": np.eye(p, q),
    }


def test_successful_fit_uses_whole_sample_and_writes_legacy_keys(tmp_path: Path) -> None:
    setting = runner.SimulationSetting(
        n=6,
        p=3,
        q=2,
        sigma0=0.01,
        target_rank=1,
        source_rank=2,
        suffix="tiny",
    )
    calls: dict[str, object] = {}

    def generator(**kwargs: object) -> dict[str, np.ndarray]:
        calls["generator"] = kwargs
        return _tiny_data(**kwargs)

    def estimator(
        X: np.ndarray,
        Y: np.ndarray,
        C0: np.ndarray,
        **kwargs: object,
    ) -> SimpleNamespace:
        calls["fit_shapes"] = (X.shape, Y.shape, C0.shape)
        calls["fit_kwargs"] = kwargs
        coefficient = np.ones((setting.p, setting.q))
        candidate = SimpleNamespace(
            metadata={"source_singular_values": np.array([1.0, 0.5])},
            failure_reason=None,
            message=None,
        )
        return SimpleNamespace(
            successful=True,
            coefficient=coefficient,
            candidate=candidate,
        )

    destination = tmp_path / "result.pkl"
    outcome, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=4,
        random_seed=123,
        destination=destination,
        generate_data_fn=generator,
        restricted_rrr_fn=estimator,
    )

    assert outcome == "written"
    assert result is not None
    assert calls["generator"] == {
        "n": 6,
        "p": 3,
        "q": 2,
        "sigma0": 0.01,
        "random_seed": 123,
    }
    assert calls["fit_shapes"] == ((6, 3), (6, 2), (3, 2))
    assert calls["fit_kwargs"] == {"target_rank": 1, "source_rank": 2}
    assert result["success"] is True
    assert result["status"] == "successful"
    assert result["avg_err"] == 1.0
    assert 0.0 <= result["fit_time_sec"] <= result["elapsed_time_sec"]
    assert {"C_true", "C_hat", "avg_err", "elapsed_time_sec"} <= result.keys()
    np.testing.assert_array_equal(
        result["source_singular_values"], np.array([1.0, 0.5])
    )
    assert result["rrr_cutoff_gap"] is None
    assert result["reduced_gram_condition"] is None
    assert destination.exists()

    with destination.open("rb") as handle:
        saved = pickle.load(handle)
    np.testing.assert_array_equal(saved["C_hat"], np.ones((3, 2)))
    np.testing.assert_array_equal(
        saved["fit_diagnostics"]["source_singular_values"],
        np.array([1.0, 0.5]),
    )

    # Existing cells are resume-safe unless replacement was requested.
    skipped, skipped_result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=4,
        random_seed=123,
        destination=destination,
        generate_data_fn=lambda **_: (_ for _ in ()).throw(AssertionError()),
        restricted_rrr_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError()
        ),
    )
    assert skipped == "skipped"
    assert skipped_result is None


def test_incompatible_rank_is_recorded_without_calling_estimator(
    tmp_path: Path,
) -> None:
    setting = runner.SimulationSetting(
        n=4,
        p=5,
        q=4,
        sigma0=0.01,
        target_rank=5,
        source_rank=3,
        suffix="r=5_rs=3",
    )

    def estimator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inapplicable settings must not invoke restricted_rrr")

    outcome, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp3",
        seed_id=0,
        random_seed=17,
        destination=tmp_path / "inapplicable.pkl",
        generate_data_fn=_tiny_data,
        restricted_rrr_fn=estimator,
    )

    assert outcome == "written"
    assert result is not None
    assert result["status"] == "inapplicable"
    assert result["applicable"] is False
    assert result["success"] is False
    assert result["failure_reason"] == "target_rank_exceeds_source_rank"
    assert result["C_hat"] is None
    assert result["avg_err"] is None
    assert 0.0 <= result["fit_time_sec"] <= result["elapsed_time_sec"]
    assert result["source_singular_values"] is None
    assert result["rrr_cutoff_gap"] is None
    assert result["reduced_gram_condition"] is None


def test_unsuccessful_api_result_is_saved_with_plain_failure_reason(
    tmp_path: Path,
) -> None:
    class ExampleFailure(Enum):
        SINGULAR = "singular_reduced_gram"

    setting = runner.SimulationSetting(
        n=5,
        p=3,
        q=2,
        sigma0=0.01,
        target_rank=1,
        source_rank=2,
        suffix="failure",
    )
    candidate = SimpleNamespace(
        metadata={
            "source_singular_values": (2.0, 1.0),
            "reduced_gram_condition": np.float64(np.inf),
        },
        failure_reason=ExampleFailure.SINGULAR,
        message="The reduced design Gram is singular.",
    )

    outcome, result = runner.run_setting(
        setting=setting,
        model="model1",
        experiment="exp1",
        seed_id=0,
        random_seed=17,
        destination=tmp_path / "failed.pkl",
        generate_data_fn=_tiny_data,
        restricted_rrr_fn=lambda *_args, **_kwargs: SimpleNamespace(
            successful=False,
            coefficient=None,
            candidate=candidate,
        ),
    )

    assert outcome == "written"
    assert result is not None
    assert result["status"] == "failed"
    assert result["failure_reason"] == "singular_reduced_gram"
    assert result["failure_message"] == "The reduced design Gram is singular."
    assert result["C_hat"] is None
    assert result["avg_err"] is None
    assert 0.0 <= result["fit_time_sec"] <= result["elapsed_time_sec"]
    assert result["source_singular_values"] == (2.0, 1.0)
    assert result["rrr_cutoff_gap"] is None
    assert result["reduced_gram_condition"] == np.inf


def test_unexpected_estimator_exception_propagates_without_result(
    tmp_path: Path,
) -> None:
    setting = runner.SimulationSetting(
        n=5,
        p=3,
        q=2,
        sigma0=0.01,
        target_rank=1,
        source_rank=2,
        suffix="exception",
    )
    destination = tmp_path / "must_not_exist.pkl"

    def estimator(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unexpected implementation bug")

    with pytest.raises(RuntimeError, match="unexpected implementation bug"):
        runner.run_setting(
            setting=setting,
            model="model1",
            experiment="exp1",
            seed_id=0,
            random_seed=17,
            destination=destination,
            generate_data_fn=_tiny_data,
            restricted_rrr_fn=estimator,
        )

    assert not destination.exists()


def test_dry_run_does_not_import_or_execute_estimators(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    def fail_to_load() -> None:
        raise AssertionError("dry-run must not import estimator packages")

    monkeypatch.setattr(runner, "_load_implementations", fail_to_load)
    exit_code = runner.main(
        [
            "0",
            "0",
            "0",
            "--setting-index",
            "0",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert list(tmp_path.iterdir()) == []
