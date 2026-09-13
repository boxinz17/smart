import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'simulation'))
from reviewer_revision_20260913.runner import configuration, v2_specs, make_v2, evaluate


def test_rrr_endpoint_and_initializer_are_distinct():
    from sparse_smart_v2 import ObservedSource
    x=np.vstack([np.eye(4),np.eye(4)])
    c=np.zeros((4,3)); c[0,0]=3; c[1,1]=2
    y=x@c
    case=dict(p=4,q=3,target_rank=1,source_rank=2)
    config=configuration(0)
    spec=next(v2_specs(case,config,'full_caps'))
    model=make_v2(spec,config).fit(x,y,source=ObservedSource(c),validation_data=(x,y))
    assert model.success_ and model.method_=='target_rrr'
    np.testing.assert_allclose(model.coefficient_[0,0],3)
    assert np.linalg.matrix_rank(model.coefficient_)==1


def test_cap_control_matches_active_family_except_caps():
    case=dict(p=100,q=50,target_rank=3,source_rank=20)
    config=configuration()
    active=list(v2_specs(case,config,'active_caps'))
    control=list(v2_specs(case,config,'cap_control'))
    def base(s):
        return tuple((k,str(v)) for k,v in s.items() if k!='support_limits')
    assert {base(s) for s in active}=={base(s) for s in control}
    assert all(s['free_directions']==(3,3) for s in active+control)
    assert all(s['support_limits']==(291,141) for s in control)


def test_prediction_excess_includes_intercept():
    x=np.eye(2); c=np.zeros((2,1)); b=np.array([2.])
    data=dict(X=x,Y=np.zeros((2,1)),X_validation=x,Y_validation=np.zeros((2,1)))
    metrics=evaluate(c,data,dict(C_star=c,Sigma_x=np.eye(2)),b)
    assert metrics['population_prediction_excess']==4
    assert metrics['validation_mse']==4
