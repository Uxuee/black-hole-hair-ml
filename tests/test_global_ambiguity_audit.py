from pathlib import Path
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from bhhairml.validation.global_ambiguity_audit import (
    all_pairs, feature_sets, load_inputs, nearest_neighbors, scaled_vectors,
)


ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/"artifacts/journal_phase_convergence/ml_ready_features_161.csv"
META=ROOT/"artifacts/journal_phase_convergence/jacobian_161_metadata.json"
OUT=ROOT/"artifacts/global_ambiguity_audit"


@lru_cache(maxsize=1)
def inputs():
    frame,sets,scales,meta=load_inputs(TABLE,META)
    vectors=scaled_vectors(frame,sets,scales)
    return frame,sets,scales,meta,vectors


@lru_cache(maxsize=1)
def pair_table():
    frame,sets,_,_,vectors=inputs()
    return all_pairs(frame,sets,vectors)


def test_mature_physical_table_and_feature_scaling():
    frame,sets,scales,meta,vectors=inputs()
    assert len(frame)==121 and frame.k.nunique()==11 and frame.wq.nunique()==11
    assert "proxy" not in str(TABLE).lower() and "161" in TABLE.name
    assert len(sets["ringdown"])==4 and len(sets["ringdown_plus_photon_geometry"])==31
    assert meta["scale_convention"].startswith("global valid-grid q95-q05")
    assert all(np.isfinite(v).all() for v in vectors.values())
    assert all(value>0 and np.isfinite(value) for value in scales.values())


def test_pairs_are_unique_finite_and_correctly_classified():
    frame,sets,_,_,vectors=inputs(); pairs=pair_table()
    assert len(pairs)==121*120//2==7260
    assert (pairs.row_id_i<pairs.row_id_j).all()
    assert not pairs[["row_id_i","row_id_j"]].duplicated().any()
    assert pairs.both_k0.sum()==55
    assert pairs.exactly_one_k0.sum()==1210
    assert pairs.both_finite_k.sum()==5995
    assert np.isfinite(pairs.filter(regex="^(d_|raw_)").to_numpy()).all()


def test_rms_distance_is_dimension_normalized():
    frame,sets,_,_,vectors=inputs(); pairs=pair_table(); row=pairs.iloc[123]
    for name in sets:
        delta=vectors[name][int(row.row_id_i)]-vectors[name][int(row.row_id_j)]
        assert np.isclose(row[f"d_O_{name}"],np.sqrt(np.mean(delta**2)))


def test_nearest_neighbor_excludes_self_and_is_deterministic():
    frame,sets,_,_,vectors=inputs(); pairs=pair_table()
    a=nearest_neighbors(frame,sets,pairs); b=nearest_neighbors(frame,sets,pairs)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==220 and (a.row_id!=a.neighbor_row_id).all()
    assert (a.k>0).all() and (a.neighbor_k>0).all()


def test_generated_outputs_are_complete_and_reproducible():
    required={"all_pair_distances.csv","nearest_observable_neighbors.csv","distant_pair_summary.csv","k0_positive_control.csv","global_ambiguity_summary.json"}
    assert required <= {p.name for p in OUT.iterdir()}
    summary=json.loads((OUT/"global_ambiguity_summary.json").read_text())
    assert summary["point_count"]==121 and summary["pair_count"]==7260
    distant=pd.read_csv(OUT/"distant_pair_summary.csv")
    assert set(distant.d_theta_threshold)=={.1,.25,.5,.75} and len(distant)==8
    assert np.isfinite(distant.select_dtypes(include=["number"]).to_numpy()).all()


def test_k0_positive_control_is_numerically_indistinguishable():
    control=pd.read_csv(OUT/"k0_positive_control.csv")
    assert len(control)==2 and (control.pair_count==55).all()
    assert (control.maximum_distance<1e-10).all()
