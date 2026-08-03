import numpy as np
import pandas as pd
import pytest

from bhhairml.validation.physical_shooting_ml_validation import (
    assert_no_preprocessing_leakage, bootstrap_correlation, directional_split,
    feature_sets, grouped_splits, joint_error, normalized_metrics,
    physical_groups, point_ids, random_split, split_conformal_radius,
    validate_inputs,
)


def grid():
    return pd.DataFrame([(k, w) for k in np.linspace(0, .0025, 11)
                         for w in np.linspace(-.7125, -.45, 11)], columns=["k", "wq"])


def test_deterministic_random_split_and_no_leakage():
    frame = grid(); a = random_split(frame, 7); b = random_split(frame, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert assert_no_preprocessing_leakage(*a)


def test_physical_block_integrity():
    frame = grid(); groups = physical_groups(frame)
    for train, calibration, test, _, assigned in grouped_splits(frame, {"name":"x","k_shift":0,"wq_shift":0}, 2):
        assert len(set(assigned[test])) == 1
        assert not set(assigned[test]) & set(assigned[train])


@pytest.mark.parametrize("direction,axis,high", [
    ("low_k_to_high_k", "k", True), ("high_k_to_low_k", "k", False),
    ("low_wq_to_high_wq", "wq", True), ("high_wq_to_low_wq", "wq", False)])
def test_directional_ranges(direction, axis, high):
    frame=grid(); train,cal,test,got=directional_split(frame,direction,3,8)
    assert got == axis; assert_no_preprocessing_leakage(train,cal,test)
    if high: assert frame[axis].iloc[test].min() > frame[axis].iloc[train].max()
    else: assert frame[axis].iloc[test].max() < frame[axis].iloc[train].min()


def test_metrics_and_joint_error():
    result=normalized_metrics(np.array([0.,1.]),np.array([0.,.5]),1.)
    assert result["MAE"] == .25 and result["NMAE"] == .25
    assert np.allclose(joint_error([0],[.0025],[-.7125],[-.45],(.0025,.2625)),np.sqrt(2))


def test_conformal_uses_calibration_residuals():
    radius=split_conformal_radius([0,1,2,3],[0,1,2,2],.25)
    assert radius == 1


def test_k_zero_wq_scoring_rule_is_explicit():
    frame=grid(); exact=np.isclose(frame.k,0)
    assert exact.sum()==11 and (~exact).sum()==110


def test_correlation_calculation():
    value,lo,hi=bootstrap_correlation(np.arange(20),np.arange(20),"spearman",samples=30)
    assert value == pytest.approx(1) and lo > .99 and hi <= 1


def test_feature_membership_dimensions():
    columns=["orbital__x","photon_geometry__x","redshift__x","timing__x",*('delta_r','r_photon','Omega','lambda')]
    sets=feature_sets(pd.DataFrame(columns=columns))
    assert len(sets["ringdown"])==4
    assert len(sets["all_shooting"])==4
    assert len(sets["ringdown_plus_all_shooting"])==8


def test_input_alignment_and_finite_values():
    frame=grid()
    for c in ["orbital__x","photon_geometry__x","redshift__x","timing__x",*('delta_r','r_photon','Omega','lambda')]: frame[c]=np.arange(121)
    sets=feature_sets(frame); rows=[]
    for name in (*sets.keys(), "ringdown_only"):
        for r in frame.itertuples(): rows.append({"k":r.k,"wq":r.wq,"observable_set":name})
    assert validate_inputs(frame,pd.DataFrame(rows),sets)["aligned"]


def test_failed_phase_or_output_inconsistency_is_explicit():
    frame=grid().iloc[:-1].copy(); sets=feature_sets(frame)
    with pytest.raises(ValueError,match="121"): validate_inputs(frame,pd.DataFrame(),sets)


def test_point_ids_complete_and_unique():
    ids=point_ids(grid()); assert len(ids)==121 and len(set(ids))==121
