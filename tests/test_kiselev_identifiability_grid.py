from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bhhairml.validation.kiselev_identifiability_grid import (
    atomic_json, classify_point, derivative_at_grid_point, feature_names,
    finite_difference_weights, global_feature_scales, harmonic_features,
    jacobian_diagnostics, load_grid_config, point_id, run_parameter_point,
)


REPOSITORY=Path(__file__).resolve().parents[1]


def config(): return load_grid_config(REPOSITORY/"configs"/"kiselev_identifiability_grid.yaml")


def test_grid_configuration_completeness_and_validated_points():
    loaded=config()
    assert len(loaded["candidate_k_values"])==len(loaded["candidate_wq_values"])==9
    assert len(loaded["science_k_values"])==len(loaded["science_wq_values"])==11
    assert 0.001 in loaded["science_k_values"] and -0.5 in loaded["science_wq_values"]
    assert any(np.isclose(loaded["science_wq_values"],-2/3,atol=1e-15,rtol=0))
    assert loaded["phi_start"]==pytest.approx(np.pi,abs=1e-15)


def test_admissibility_classification_safe_marginal_invalid_and_failed():
    loaded=config(); base={"status":"completed","f_observer":.5,"min_emitter_f":.4,"min_photon_f":.3,
        "shooting_success_fraction":1.0,"finite_observable_coverage":True,"max_hit_error":1e-8,
        "max_timelike_constraint_error":1e-13,"max_null_constraint_error":1e-11,"max_impact_parameter_drift":1e-10}
    assert classify_point(base,loaded)[0]=="safe"
    assert classify_point({**base,"min_photon_f":.05},loaded)[0]=="marginal"
    assert classify_point({**base,"min_photon_f":0.0},loaded)[0]=="invalid"
    assert classify_point({**base,"status":"failed","reason":"forced"},loaded)==("failed","forced")


def test_atomic_status_and_resumable_completed_point(tmp_path):
    root=tmp_path/point_id(.001,-.5); root.mkdir()
    atomic_json(root/"status.json",{"status":"completed","phases":3})
    pd.DataFrame({"phi":[np.pi,np.pi+.1,np.pi+.2]}).to_csv(root/"phase_resolved.csv.gz",index=False)
    task={"config":config(),"config_path":str(REPOSITORY/"configs"/"kiselev_identifiability_grid.yaml"),
          "k":.001,"wq":-.5,"phases":3,"root":str(root)}
    result=run_parameter_point(task)
    assert result["status"]=="skipped_completed"
    assert json.loads((root/"status.json").read_text())["status"]=="completed"


def test_parallel_point_ids_are_unique_and_path_safe():
    ids={point_id(k,w) for k in (0,.001,.002) for w in (-.75,-.5,-.45)}
    assert len(ids)==9 and all("/" not in value and "\\" not in value for value in ids)


def test_harmonic_features_and_schema_are_fixed():
    phi=np.linspace(np.pi,3*np.pi,1001); values=2+3*np.cos(phi-np.pi)-.5*np.sin(2*(phi-np.pi))
    result=harmonic_features(phi,values,3)
    assert result["mean"]==pytest.approx(2,abs=.01)
    assert result["cos1"]==pytest.approx(3,abs=.02)
    assert result["sin2"]==pytest.approx(-.5,abs=.01)
    first=feature_names(3); second=feature_names(3)
    assert first==second and len(first["photon_geometry"])>len(first["redshift"])


def test_unequal_central_and_boundary_finite_differences():
    weights,scheme=finite_difference_weights(0.0,1.0,3.0)
    assert scheme=="central" and np.dot(weights,np.array([0.,1.,9.]))==pytest.approx(2.0)
    weights,scheme=finite_difference_weights(None,1.0,2.0)
    assert scheme=="one_sided_forward" and np.dot(weights,[1.,4.])==pytest.approx(3.0)


def _synthetic_grid():
    rows=[]
    for k in (0.,.001,.002):
        for w in (-.7,-.6,-.5): rows.append({"k":k,"wq":w,"Omega":2*k+3*w,"lambda":k-w})
    return pd.DataFrame(rows)


def test_central_one_sided_and_missing_neighbour_derivatives():
    frame=_synthetic_grid()
    value,scheme=derivative_at_grid_point(frame,.001,-.6,"Omega","k")
    assert scheme=="central" and value==pytest.approx(2)
    value,scheme=derivative_at_grid_point(frame,0.,-.6,"Omega","k")
    assert scheme=="one_sided_forward" and value==pytest.approx(2)
    missing=frame[~((frame.k==.002)&(frame.wq==-.6))]
    value,scheme=derivative_at_grid_point(missing,.001,-.6,"Omega","k")
    assert scheme=="one_sided_backward" and value==pytest.approx(2)


def test_global_scaling_is_shared_and_constant_features_excluded():
    frame=_synthetic_grid(); frame["constant"]=1.0
    loaded=config(); scales,excluded=global_feature_scales(frame,["Omega","lambda","constant"],loaded)
    assert scales["Omega"]>0 and scales["lambda"]>0
    assert "constant" in excluded


def test_k_zero_wq_insensitivity_rank_singular_values_and_condition():
    loaded=config().copy(); loaded["science_k_values"]=[0.,.001,.002]; loaded["science_wq_values"]=[-.7,-.6,-.5]
    frame=_synthetic_grid()
    # Supply the declared schema with nonconstant analytic functions.
    for names in feature_names(loaded["harmonic_count"]).values():
        for index,name in enumerate(names): frame[name]=(index+1)*frame.k+(index+2)*frame.wq
    frame["delta_r"]=frame.k; frame["r_photon"]=3+frame.k
    diagnostics,_=jacobian_diagnostics(frame,loaded)
    zero=diagnostics[np.isclose(diagnostics.k,0)]
    assert (zero.numerical_rank==1).all()
    assert (zero.derivative_scheme_wq=="analytic_k_zero").all()
    interior=diagnostics[(diagnostics.k==.001)&(diagnostics.wq==-.6)]
    assert np.isfinite(interior.sigma_min).all()
    assert (interior.condition_number>=1).all()
    assert interior.cosine_similarity.between(-1,1).all()


def test_ringdown_shooting_alignment_merge_is_one_to_one():
    shooting=_synthetic_grid(); ringdown=_synthetic_grid()[["k","wq"]].copy(); ringdown["ring"]=1
    merged=shooting.merge(ringdown,on=["k","wq"],validate="one_to_one")
    assert len(merged)==len(shooting)
    with pytest.raises(pd.errors.MergeError):
        shooting.merge(pd.concat([ringdown,ringdown.iloc[[0]]]),on=["k","wq"],validate="one_to_one")

