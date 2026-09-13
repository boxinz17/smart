from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "simulation"))
from paper_source_comparators_20260913.plan import build
from paper_source_comparators_20260913.common import METHODS, digest, immutable
from paper_source_comparators_20260913.runtime import audit_selected, metrics

def case(experiment=0, rank=5, source=10, sigma=.01, seed=0):
    return dict(case_id=f"e{experiment}_r{rank}_d{source}_s{seed}_noise{sigma}", model_id=0,
                experiment_id=experiment, n_train=200, p=100, q=50, sigma0=sigma,
                seed_id=seed, random_seed=10+seed, rank=rank, source_rank=source, setting_index=0)

def test_flatline_aliases_and_source_noise_recompute():
    cs = [case(), case(2,source=5),case(2,source=20),case(1,rank=11),case(3,sigma=.5)]
    p = build(Path('/tmp/example'), [dict(root='/tmp/reference', cases=cs)], 'a'*64, {})
    assert p['n_groups'] == 2
    assert p['n_tasks'] == 22
    a,b,c,d,e = p['display_cases']
    assert a['task_ids'] == b['task_ids'] == c['task_ids']
    for method in METHODS:
        assert e['task_ids'][method] != a['task_ids'][method]
        assert (d['task_ids'][method] == a['task_ids'][method]) == (method in ('ridge_to_source','nuclear_contrast'))
    assert p['plan_fingerprint'] == digest({k:v for k,v in p.items() if k!='plan_fingerprint'})

def test_freeze_rejects_overwrite(tmp_path):
    immutable(tmp_path/'x.json',dict(a=1))
    immutable(tmp_path/'x.json',dict(a=1))
    with pytest.raises(ValueError,match='changed'):
        immutable(tmp_path/'x.json',dict(a=2))

def test_audit_rejects_wrong_winner_and_metric_uses_rmse():
    x=np.eye(3); y=np.eye(3); c=np.eye(3)
    data=dict(X=x,Y=y,X_validation=x,Y_validation=y,C_star=np.zeros((3,3)))
    selection=dict(selected_index=0,validation_mse=0.,candidate_results=[
        dict(index=0,status='eligible',validation_mse=.1,parameters={}),
        dict(index=1,status='eligible',validation_mse=0.,parameters={})])
    with pytest.raises(ValueError,match='winner'):
        audit_selected(c,selection,data,{},dict(method='target_rrr',rank=3))
    assert metrics(c,data)['coefficient_rmse'] == pytest.approx(1/np.sqrt(3))
    assert metrics(c,data)['coefficient_frobenius_squared'] == 3

@pytest.mark.parametrize('method', METHODS)
def test_fitter_runtime_audit_interface(method):
    from paper_source_comparators_20260913.fitting import fit_method
    rng=np.random.default_rng(947)
    x=rng.normal(size=(30,12)); xv=rng.normal(size=(15,12))
    source=np.zeros((12,10)); source[:2,:2]=np.diag([2.,1.])
    data=dict(X=x,Y=x@source+.01*rng.normal(size=(30,10)),C0=source,
              X_validation=xv,Y_validation=xv@source+.01*rng.normal(size=(15,10)))
    frames=dict(left=np.eye(12),right=np.eye(10))
    rank=None if method in ('ridge_to_source','nuclear_contrast') else 2
    selected,coefficient=fit_method(data,frames,method,rank)
    assert selected['status']=='success'
    assert audit_selected(coefficient,selected,data,frames,dict(method=method,rank=rank))['success']
