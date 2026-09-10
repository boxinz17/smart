"""External-sample result auditing and historical-comparison accounting."""
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_external as summary
import summarize_sparse_smart_external_grid as grid_summary
from run_sparse_smart import _digest_json, _json_value
from run_sparse_smart_tuned import RunnerConfig as PreviousConfig
import test_summarize_sparse_smart_tuned as previous_fixture


def record(seed=0, *, error=.1, status="complete", selected=1, config=None):
    config = config or summary.external_runner.RunnerConfig(iterations=2,
        penalties_u=(.0025, .04), penalties_v=(.01,))
    previous_keys = {field.name for field in fields(PreviousConfig)}
    previous_config = PreviousConfig(**{key:value for key,value in asdict(config).items()
                                       if key in previous_keys})
    original = previous_fixture.record(seed, error=.8, config=previous_config, experiment="exp4")
    external = previous_fixture.record(seed, error=error, status=status, selected=selected,
                                       config=previous_config, experiment="exp4")
    setting = summary.experiment_settings(0, 3)[1]
    external.update(method="SparseSMARTExternal",
        configuration=_json_value(summary.external_runner.resolved_configuration(setting,config)),
        training_observed_input_fingerprint=external.pop("observed_input_fingerprint"),
        validation_observed_input_fingerprint="d"*64,
        validation_seed_metadata=dict(seed_sequence_entropy=[int(previous_fixture.SEEDS[seed]),config.validation_seed_tag],
            bit_generator="PCG64",covariance="AR1",covariance_rho=.5,design_factorization="cholesky",
            noise_std=.5,conditional_on_same_coefficient=True,source_reused=True),
        n_train=200,n_validation=config.n_validation,all_training_rows_used=True,
        training_matches_legacy=True,refit_on_all_data=False)
    split=dict(mode="independent_external_validation",n_train=200,n_validation=config.n_validation,
        train_indices=None,validation_indices=None,all_training_rows_used=True,refit_on_all_data=False)
    external["split"] = dict(**split,fingerprint=_digest_json(split))
    external["input_fingerprint"] = _digest_json(dict(training_observed=external["training_observed_input_fingerprint"],
        validation_observed=external["validation_observed_input_fingerprint"],
        evaluation_truth=external["evaluation_truth_fingerprint"]))
    identity={key:external[key] for key in ("schema_version","method","model","experiment","rd_seed_id",
        "random_seed","setting","configuration","generator_arguments")}
    external["configuration_fingerprint"] = _digest_json(identity)
    return external, original


def write(root, value, original=None):
    setting=summary.experiment_settings(0,3)[1]
    path=summary.external_runner.result_path(root/"external",model="model1",experiment="exp4",
                                           setting=setting,seed_id=value["rd_seed_id"])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    if original is not None:
        previous_fixture.write(root/"previous",original)
    return path


def summarize(root, **kwargs):
    return summary.summarize(root/"external",previous_root=root/"previous",**kwargs)


def test_six_noise_rows_preserve_failures_missingness_and_conditional_statistics(tmp_path):
    write(tmp_path,*record(0,error=.1,selected=0))
    write(tmp_path,*record(1,error=.3,selected=2))
    write(tmp_path,*record(2,status="all_candidates_failed"))
    rows,_,audit=summarize(tmp_path)
    assert [r["sigma0"] for r in rows] == [0.,.01,.02,.05,.1,.5]
    row=rows[1]
    assert (row["complete"],row["failed"],row["all_candidates_failed"],row["missing"]) == (2,1,1,2)
    assert row["final_mean"] == pytest.approx(.2) and row["final_se"] == pytest.approx(.1)
    assert row["mean_is_conditional_on_success"] and row["earlier_mean"] == pytest.approx(.8)
    assert row["n_train"] == 200 and row["n_validation"] == 100
    assert row["earlier_n_train"] == 160 and row["earlier_n_validation"] == 40
    assert row["candidate_failed"] == 2 and row["candidate_total"] == 6
    assert row["chosen_initializer"] == 1 and row["selected_at_budget"] == 1
    assert row["optimization_converged"] == row["selected_converged"] == 0
    assert rows[0]["complete"] == 0 and rows[0]["final_mean"] is None
    assert audit["expected_runs"] == 30 and audit["recorded_runs"] == 3
    assert audit["training_fingerprints_verified_against_previous"] == 3
    assert audit["comparison_has_equal_tuning_data"] is False


@pytest.mark.parametrize("mutate,match",[
    (lambda r:r.update(random_seed=12),"random seed"),
    (lambda r:r.update(configuration_fingerprint="0"*64),"Configuration fingerprint"),
    (lambda r:r.update(n_train=160),"counts or reuse"),
    (lambda r:r["split"].update(train_indices=[0]),"split metadata"),
    (lambda r:r["split"].update(fingerprint="f"*64),"split metadata"),
    (lambda r:r["validation_seed_metadata"]["seed_sequence_entropy"].__setitem__(0,123),"seed metadata"),
    (lambda r:r.update(refit_on_all_data=True),"must not refit"),
    (lambda r:r.update(best_params=deepcopy(r["selection_history"][1]["params"])),"validation winner"),
    (lambda r:r["selection_history"][0].update(selected_iteration=0),"validation minimum"),
    (lambda r:r["selected_supports"].update(u=[0,0]),"support mismatch"),
    (lambda r:r.update(success=False),"success flag"),
])
def test_corrupt_external_provenance_split_or_selection_is_rejected(tmp_path,mutate,match):
    value,original=record()
    mutate(value)
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match=match):summarize(tmp_path)


@pytest.mark.parametrize("field",["training_observed_input_fingerprint","evaluation_truth_fingerprint"])
def test_original_200_row_dataset_and_truth_must_match_even_with_consistent_combined_hash(tmp_path,field):
    value,original=record()
    value[field]="e"*64
    value["input_fingerprint"]=_digest_json(dict(training_observed=value["training_observed_input_fingerprint"],
        validation_observed=value["validation_observed_input_fingerprint"],evaluation_truth=value["evaluation_truth_fingerprint"]))
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match="differs from the original 200-row"):summarize(tmp_path)


def test_missing_earlier_artifact_cannot_be_claimed_as_verified_training_reuse(tmp_path):
    value,_=record()
    write(tmp_path,value)
    with pytest.raises(ValueError,match="Missing earlier 200-row"):summarize(tmp_path)


def test_prior_reference_is_validated_before_it_is_used(tmp_path):
    value,original=record()
    original["avg_err"]=None
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match="coefficient error"):summarize(tmp_path)


def test_failed_partial_is_never_promoted_into_reported_coefficient_error(tmp_path):
    value,original=record(status="all_candidates_failed")
    value["avg_err"]=.000001
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match="Failed partial error"):summarize(tmp_path)


def test_setup_failure_without_split_is_recorded_and_does_not_break_summary(tmp_path):
    value,original=record(status="failed")
    value.update(split=None,selection_history=[],fit_errors=[])
    write(tmp_path,value,original)
    rows,_,_=summarize(tmp_path,seed_ids=(0,))
    assert rows[1]["failed"] == 1 and rows[1]["final_mean"] is None


def test_mixed_implementations_are_rejected(tmp_path):
    write(tmp_path,*record(0))
    value,original=record(1)
    value["implementation_fingerprint"]="f"*64
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match="different external configurations or implementations"):
        summarize(tmp_path)


def test_requires_the_requested_100_additional_validation_rows(tmp_path):
    config=summary.external_runner.RunnerConfig(iterations=2,penalties_u=(.0025,.04),penalties_v=(.01,),n_validation=40)
    write(tmp_path,*record(config=config))
    with pytest.raises(ValueError,match="exactly 100"):summarize(tmp_path)


def test_report_and_plot_label_different_data_budgets_and_distinct_convergence(tmp_path):
    write(tmp_path,*record(selected=0))
    rows,paper,audit=summarize(tmp_path,seed_ids=(0,))
    destination=tmp_path/"summary"
    summary.write_outputs(rows,paper,audit,destination)
    report=(destination/"comparison.md").read_text()
    assert "200 training and 100 additional tuning" in report
    assert "160 training and 40 validation" in report
    assert "not an equal-data-budget comparison" in report
    assert "cannot be attributed to the optimizer alone" in report
    assert "no full-data refit" in report
    assert "Reaching the iteration limit is not convergence" in report
    assert "100 repetitions" in report and "paired seed-level" in report
    assert (destination/"comparison.csv").is_file()
    assert (destination/"validation_audit.json").is_file()
    assert (destination/"comparison.png").stat().st_size > 10000


def copied_paper_reference(root):
    root.mkdir(parents=True,exist_ok=True)
    reference=root/summary.DEFAULT_REFERENCE.name
    for source in (summary.DEFAULT_REFERENCE,summary.DEFAULT_REFERENCE.with_name("provenance.json")):
        (root/source.name).write_bytes(source.read_bytes())
    return reference


def test_paper_pdf_hashes_are_checked_when_a_source_pdf_is_present(tmp_path,monkeypatch):
    reference=copied_paper_reference(tmp_path/"reference")
    manuscript=tmp_path/"manuscript"
    monkeypatch.setattr(summary,"REPO_ROOT",manuscript)
    provenance=json.loads(reference.with_name("provenance.json").read_text())
    pdf=manuscript/provenance["sources"][0]["pdf"]
    pdf.parent.mkdir(parents=True,exist_ok=True)
    pdf.write_bytes(b"This present PDF does not match the reference source.")
    with pytest.raises(ValueError,match="Paper PDF hash differs"):
        summary.summarize(tmp_path/"external",reference,previous_root=tmp_path/"previous")


@pytest.mark.parametrize("consumer",["external","grid"])
def test_offline_paper_bundle_supports_both_summary_consumers(tmp_path,monkeypatch,consumer):
    reference=copied_paper_reference(tmp_path/"reference")
    manuscript=tmp_path/"manuscript"
    manuscript.mkdir()
    monkeypatch.setattr(summary,"REPO_ROOT",manuscript)
    if consumer == "external":
        write(tmp_path,*record())
        rows,paper,audit=summarize(tmp_path,reference_path=reference,seed_ids=(0,))
        assert rows[1]["complete"] == 1
    else:
        rows,paper,audit=grid_summary.summarize(tmp_path/"no-runs",reference,
                                               model_ids=(0,),experiments=(3,),seed_ids=(0,))
        assert all(row["missing"] == 1 for row in rows)
    assert len(rows) == 6 and len(paper) == 36
    assert audit["paper_pdf_hashes_verified"] == []
    verification=audit["paper_reference_verification"]
    assert verification["csv_hash_verified"] is True
    assert verification["csv_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert verification["provenance_sha256"] == hashlib.sha256(reference.with_name("provenance.json").read_bytes()).hexdigest()
    assert verification["row_count"] == 432
    assert len(verification["source_pdfs"]) == 3
    assert {source["verification"] for source in verification["source_pdfs"]} == {"unavailable"}


@pytest.mark.parametrize("consumer",["external","grid"])
def test_offline_bundle_csv_corruption_is_rejected(tmp_path,monkeypatch,consumer):
    reference=copied_paper_reference(tmp_path/"reference")
    monkeypatch.setattr(summary,"REPO_ROOT",tmp_path/"no-pdfs")
    # Appending whitespace keeps CSV parsing valid but changes its exact bytes.
    reference.write_bytes(reference.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="CSV"):
        if consumer == "external":
            summarize(tmp_path,reference_path=reference,seed_ids=(0,))
        else:
            grid_summary.summarize(tmp_path/"no-runs",reference,model_ids=(0,),experiments=(3,),seed_ids=(0,))


def _rehash_identity(value):
    identity={key:value[key] for key in ("schema_version","method","model","experiment","rd_seed_id",
        "random_seed","setting","configuration","generator_arguments")}
    value["configuration_fingerprint"]=_digest_json(identity)


@pytest.mark.parametrize("missing",[
    ("initialization_spectrum",), ("refinement_solver",),
    ("initialization_spectrum","refinement_solver"),
])
def test_historical_missing_solver_keys_decode_to_actual_old_algorithms(tmp_path,missing):
    value,original=record()
    for key in missing:del value["configuration"]["runner"][key]
    _rehash_identity(value)
    saved=deepcopy(value)
    write(tmp_path,value,original)
    _,_,audit=summarize(tmp_path,seed_ids=(0,))
    for key,historical in summary.HISTORICAL_SOLVER_SETTINGS.items():
        assert audit["runner_config"][key] == (historical if key in missing else "auto")
    assert value==saved


def test_current_explicit_solver_settings_are_preserved(tmp_path):
    value,original=record(config=summary.external_runner.RunnerConfig(iterations=2,
        penalties_u=(.0025,.04),penalties_v=(.01,),
        initialization_spectrum="projected",refinement_solver="anchor_projected"))
    write(tmp_path,value,original)
    _,_,audit=summarize(tmp_path,seed_ids=(0,))
    assert audit["runner_config"]["initialization_spectrum"]=="projected"
    assert audit["runner_config"]["refinement_solver"]=="anchor_projected"


def test_legacy_compatibility_does_not_repair_identity_hash_or_hide_other_omissions(tmp_path):
    value,original=record()
    del value["configuration"]["runner"]["initialization_spectrum"]
    write(tmp_path,value,original)
    with pytest.raises(ValueError,match="Configuration fingerprint"):
        summarize(tmp_path,seed_ids=(0,))
    value,original=record()
    del value["configuration"]["runner"]["initialization_spectrum"]
    del value["configuration"]["runner"]["inverse_step"]
    _rehash_identity(value);write(tmp_path,value,original)
    with pytest.raises(ValueError,match="Resolved configuration mismatch"):
        summarize(tmp_path,seed_ids=(0,))
