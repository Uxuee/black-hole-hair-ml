import numpy as np
import pandas as pd

from experiments.traditional_inverse_baselines.nearest_physical_model import NearestPhysicalModel
from experiments.traditional_inverse_baselines.jacobian_local_inverse import LocalJacobianInverse
from experiments.traditional_inverse_baselines.evaluate_nearest_baseline import registered_split_specs, summarize


def test_nearest_scaler_and_catalog_are_training_only():
    X_train=np.array([[0.0],[2.0],[4.0]]); theta=np.array([[0,-.7],[.001,-.6],[.002,-.5]])
    model=NearestPhysicalModel().fit(X_train,theta,["a","b","c"])
    assert model.scaler_.mean_[0] == 2.0
    prediction,index,_=model.predict_with_diagnostics([[100.0]])
    assert model.row_ids_[index[0]] in {"a","b","c"}
    assert tuple(prediction[0]) in set(map(tuple,theta))


def test_nearest_is_deterministic_and_explicit_distance_is_standardized():
    X=np.array([[0.,0.],[2.,4.],[5.,8.]]); theta=np.array([[0,-.7],[.001,-.6],[.002,-.5]])
    a=NearestPhysicalModel().fit(X,theta); b=NearestPhysicalModel().fit(X,theta)
    pa,ia,da=a.predict_with_diagnostics([[1.9,4.1]]); pb,ib,db=b.predict_with_diagnostics([[1.9,4.1]])
    assert np.array_equal(pa,pb) and np.array_equal(ia,ib) and np.allclose(da,db)
    expected=np.linalg.norm(a.scaler_.transform([[1.9,4.1]])[0]-a.X_train_scaled_[ia[0]])
    assert da[0] == expected


def test_registered_split_roles_are_reused_and_disjoint():
    rows=[]
    for role,ids in {"train":["a","b"],"calibration":["c"],"test":["d"]}.items():
        rows += [{"point_id":p,"protocol":"random_interpolation","direction":"","fold":"f","seed":1,"role":role} for p in ids]
    spec=registered_split_specs(pd.DataFrame(rows))[0]
    assert spec[-1] == {"train":{"a","b"},"calibration":{"c"},"test":{"d"}}


def test_k_zero_identifiable_wq_mask_and_summary():
    frame=pd.DataFrame({"method":["m"]*2,"protocol":["p"]*2,"direction":[""]*2,"fold":["f"]*2,"seed":[1]*2,"feature_set":["ringdown"]*2,
        "wq_identifiable":[False,True],"absolute_error_k":[.1,.2],"normalized_error_k":[.1,.2],"absolute_error_wq":[99.,.1],"normalized_error_wq":[99.,.2]})
    result=summarize(frame).iloc[0]
    assert result.n_wq_scored == 1 and result.MAE_identifiable_wq == .1 and result.NMAE_identifiable_wq == .2


def test_local_jacobian_reference_is_training_only_and_no_silent_clipping():
    theta=np.array([[0.,-.7],[.001,-.7],[.002,-.7],[.001,-.6],[.002,-.5]])
    X=np.column_stack([theta[:,0]/.0025,(theta[:,0]*theta[:,1])/.001])
    model=LocalJacobianInverse(neighbor_count=4).fit(X,theta,["a","b","c","d","e"])
    prediction,diag=model.predict_with_diagnostics([[5.,5.]])
    assert model.row_ids_[diag["reference_index"][0]] in set(model.row_ids_)
    assert diag["outside_domain"][0]
    assert np.any((prediction[0] < model.parameter_bounds[:,0]) | (prediction[0] > model.parameter_bounds[:,1]))


def test_rank_deficient_k_zero_reference_is_safe():
    theta=np.array([[0.,-.7],[0.,-.6],[0.,-.5],[.001,-.7],[.001,-.5]])
    X=np.column_stack([theta[:,0],theta[:,0]*theta[:,1]])
    model=LocalJacobianInverse().fit(X,theta)
    assert np.allclose(model.jacobians_[0][:,1],0)
    prediction,diag=model.predict_with_diagnostics([X[0]])
    assert np.isfinite(prediction).all() and np.isinf(diag["condition_number"][0])


def test_primary_output_schema_is_complete():
    required={"protocol","direction","fold","seed","feature_set","test_k","test_wq","predicted_k","predicted_wq",
              "nearest_training_row_identifier","standardized_nearest_distance","is_k_zero","wq_identifiable",
              "absolute_error_k","absolute_error_wq"}
    path="artifacts/traditional_baselines/nearest_physical_model_predictions.csv"
    assert required <= set(pd.read_csv(path,nrows=1).columns)


def test_archived_registered_splits_and_all_directions_are_consumed():
    assignments=pd.read_csv("artifacts/journal_phase_convergence/split_assignments_161.csv")
    specs=registered_split_specs(assignments)
    assert len(specs) == 115
    directions={s[1] for s in specs if s[0] == "directional_extrapolation"}
    assert directions == {"low_k_to_high_k","high_k_to_low_k","high_wq_to_low_wq","low_wq_to_high_wq"}
    produced=pd.read_csv("artifacts/traditional_baselines/nearest_physical_model_predictions.csv",usecols=["protocol","direction","fold","seed","point_id"])
    produced_keys=set(map(tuple,produced.fillna("").itertuples(index=False,name=None)))
    registered_test=assignments[assignments.role=="test"][["protocol","direction","fold","seed","point_id"]]
    registered_keys=set(map(tuple,registered_test.fillna("").itertuples(index=False,name=None)))
    assert produced_keys == registered_keys
