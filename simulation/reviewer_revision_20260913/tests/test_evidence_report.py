"""Regression for observed zero/all-harm Wilson interval rendering."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence_report as report


def test_harm_interval_endpoints_render(tmp_path):
    cases = []
    pairs = []
    for n_train, count in ((80, 0), (200, 30)):
        low, high = report._wilson(count, 30)
        rate = count / 30
        assert low <= rate <= high
        if count == 0:
            assert low == 0
        else:
            assert high == 1
        case_id = f"reference_exact_n{n_train}"
        cases.append(dict(case_id=case_id, family="reference", level=0, n_train=n_train))
        pairs.append(dict(case_id=case_id, method="v2", reference="target_ridge_rrr",
                          harm_frequency=rate, harm_low=low, harm_high=high,
                          harm_count=count, n_paired=30))
    figure = report.plot_harm(cases, pairs, False)
    figure.savefig(tmp_path / "endpoint-harm.pdf")
    report.plt.close(figure)


def test_solver_labels_declared_ineligible_slots_and_disjoint_selections():
    diagnostics = [dict(variant="full_caps", status="success", n_candidates=10,
        n_ineligible=2, selected_rrr=rrr, selected_iteration=iteration)
        for rrr, iteration in ((True, 0), (False, 0), (False, 10))]
    pairs = [dict(method="v2_active_caps", reference="v2_cap_control", level="exact",
        n_train=40, geometric_ratio=.9, ratio_low=.8, ratio_high=1.0)]
    figure = report.plot_solver(diagnostics, pairs, False)
    assert figure.axes[0].get_ylabel() == "Ineligible / declared candidate slots"
    assert figure.axes[0].patches[0].get_height() == pytest.approx(6/30)
    assert [bar.get_height() for bar in figure.axes[1].patches] == pytest.approx([1/3, 1/3])
    assert "Chart iteration 0" in [text.get_text() for text in figure.axes[1].get_legend().get_texts()]
    report.plt.close(figure)


def test_harm_shares_short_labels_for_identical_panel_rows():
    cases = [dict(case_id=f"fitted_{level}_{n}", family="fitted_relationship", level=level, n_train=n)
             for n in (80, 200) for level in ("coefficient_close", "spectral_shift")]
    pairs = []
    for case in cases:
        low, high = report._wilson(0, 30)
        pairs.append(dict(case_id=case["case_id"], method="v2", reference="target_ridge_rrr",
            harm_frequency=0., harm_low=low, harm_high=high, harm_count=0, n_paired=30))
    figure = report.plot_harm(cases, pairs, False)
    assert [text.get_text() for text in figure.axes[0].get_yticklabels()] == [
        "Fitted source: close coefficients", "Fitted source: spectral shift"]
    assert all(text.get_text() == "" for text in figure.axes[1].get_yticklabels())
    assert figure.axes[0].get_xlabel() == "Observed harm frequency"
    assert report._harm_label(dict(family="source_truncation", level=3)) == "Source dimension s = 3"
    report.plt.close(figure)


def test_compact_plot_csv_decodes_bool_counts_and_nested_stats(tmp_path):
    path = tmp_path / "rows.csv"
    report._write_csv(path, [dict(selected_rrr=False, selected_iteration=0, n_candidates=211,
        value=None, risk=dict(n=30, mean=.02, low=.01, high=.03))])
    row = report._read_compact_report_csv(path)[0]
    assert row == dict(selected_rrr=False, selected_iteration=0, n_candidates=211,
                       value=None, risk=dict(n=30, mean=.02, low=.01, high=.03))


def test_cached_render_rejects_mismatched_audit_before_outputs(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(dict(cases=[], tasks=[])))
    (tmp_path / "audit.json").write_text(json.dumps(dict(audit_passed=True,
        all_planned_tasks_complete=True, plan_sha256="wrong")))
    (tmp_path / "report-summary.json").write_text(json.dumps(dict(stage="audited_study", plan_sha256="wrong")))
    sentinel = tmp_path / "report-harm.png"
    sentinel.write_bytes(b"prior image")
    with pytest.raises(ValueError, match="matching complete passing audit"):
        report.render_cached_figures(tmp_path, tmp_path / "report")
    assert sentinel.read_bytes() == b"prior image"
