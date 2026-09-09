import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "summarize_sparse_smart.py"
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("summarize_sparse_smart", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record(seed, status, error):
    return dict(model="model1", experiment="exp1", rd_seed_id=seed,
                setting={"suffix":"n=200"}, configuration={"runner":{"iterations":50}},
                implementation_fingerprint="same", status=status, success=status=="complete",
                avg_err=error, initial_avg_err=.2, last_accepted_avg_err=.15,
                fit_time_sec=.1, applicable=True, diagnostics={},
                failure_reason=None if status=="complete" else "line_search_failed")


def write(root, value):
    path = root / "model1" / "exp1" / f"SparseSMART_result_{value['rd_seed_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_failed_and_missing_seeds_are_counted_without_becoming_successes(tmp_path):
    write(tmp_path, record(0,"complete",.1))
    write(tmp_path, record(1,"complete",.3))
    write(tmp_path, record(2,"failed",None))
    rows, _, _ = module.summarize(tmp_path,module.DEFAULT_REFERENCE,experiments=(0,))
    row = rows[0]
    assert (row['complete'],row['failed'],row['missing']) == (2,1,2)
    assert row['final_mean'] == pytest.approx(.2)
    assert row['final_se'] == pytest.approx(.1)
    assert row['last_accepted_mean'] == pytest.approx(.15)
    assert rows[1]['missing'] == 5 and rows[1]['final_mean'] is None


def test_mixed_tuning_and_corrupt_success_records_are_rejected(tmp_path):
    write(tmp_path, record(0,"complete",.1))
    other=record(1,"complete",.2)
    other['configuration']['runner']['iterations']=100
    write(tmp_path,other)
    with pytest.raises(ValueError,match="different tuning"):
        module.summarize(tmp_path,module.DEFAULT_REFERENCE,experiments=(0,))
    other=record(1,"complete",None)
    write(tmp_path,other)
    with pytest.raises(ValueError,match="completed-fit error"):
        module.summarize(tmp_path,module.DEFAULT_REFERENCE,experiments=(0,))
