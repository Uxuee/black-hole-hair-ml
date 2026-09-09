"""Frozen values from audits/independent_physical_shooting_walkthrough.ipynb.

Expected values are literal independent-reference outputs, not generated with
production functions at test runtime.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.shooting.emitter import turning_point_constants, integrate_emitter_orbit
from bhhairml.shooting.photon import null_initial_momentum, null_hamiltonian, shoot_photon, static_observer_tetrad_projection
from bhhairml.validation.kiselev_identifiability_grid import extract_point_features, standardized_matrix

ROOT=Path(__file__).resolve().parents[1]

def test_metric_derivative_and_explicit_mass_reference():
    m=KiselevMetric(1.2,.001,-.6)
    np.testing.assert_allclose(m.f(10),.7536904265551981,rtol=0,atol=2e-15)
    np.testing.assert_allclose(m.f_prime(10),.023495234124415846,rtol=0,atol=2e-15)
    assert KiselevMetric(1,0,-.45).f(9)==KiselevMetric(1,0,-.7125).f(9)

def test_turning_points_and_null_initial_reference():
    metric=KiselevMetric(1,.001,-2/3); E,L=turning_point_constants(metric,8,12)
    np.testing.assert_allclose([E,L],[.9477113531045734,3.6700205577197096],rtol=0,atol=2e-14)
    x=np.array([9.,2.,-1.]); d=np.array([-.1,.02,-1.]); d/=np.linalg.norm(d); p=null_initial_momentum(metric,x,d)
    np.testing.assert_allclose(p,[-.11300469,.02260094,-1.13004686],rtol=0,atol=6e-9)
    assert abs(null_hamiltonian(metric,x,p))<1e-14

def test_static_tetrad_and_frozen_photon_redshift_geometry():
    metric=KiselevMetric(1,.001,-2/3); ph=np.array([np.pi,np.pi+.01]); orbit=integrate_emitter_orbit(metric,8,12,ph,inclination=np.deg2rad(135),omega=np.deg2rad(65),Omega=np.deg2rad(225))
    observer=np.array([0.,0.,-80.]); shot=shoot_photon(metric,orbit.position[0],observer,root_tolerance=1e-10,hit_tolerance=1e-5,photon_max_step=4,photon_max_affine_parameter=200)
    assert shot.success; integ=shot.integration; hit,p=integ.position[-1],integ.momentum[-1]; r=np.linalg.norm(hit); n=hit/r; kvec=p+(metric.f(r)-1)*n.dot(p)*n; comp,angles=static_observer_tetrad_projection(metric,hit,kvec)
    np.testing.assert_allclose(angles,[-.1215895450060836,.024951675065205202],atol=2e-9)
    np.testing.assert_allclose(integ.impact_parameter,10.41626286288726,atol=2e-9)
    u=orbit.four_velocity[0]; one_plus_z=(u[0]-u[1:].dot(integ.momentum[0]))/(1/np.sqrt(metric.f(r)))
    np.testing.assert_allclose(one_plus_z,.9938443858457906,atol=2e-11)

def test_frozen_feature_and_selected_jacobian_reference():
    cfg={"M":1.,"r_p":8.,"r_a":12.,"turning_point_phi_end":53.40707511102649,"emitter_rtol":1e-11,"emitter_atol":1e-13,"harmonic_count":3,"feature_scale_quantiles":[.05,.95],"feature_scale_floor":1e-12,"science_k_values":[0,.00025,.0005,.00075,.001,.00125,.0015,.00175,.002,.00225,.0025],"science_wq_values":[-.7125,-.69,-2/3,-.64,-.61,-.58,-.55,-.525,-.5,-.475,-.45]}; frame=pd.read_csv(ROOT/"artifacts/kiselev_identifiability_grid/points/science/k0p001000000_wqm0p666666667/phase_resolved.csv.gz")
    feat=extract_point_features(frame,cfg,.001,-2/3); np.testing.assert_allclose(feat["orbital__r_emit__mean"],9.591478071099413,atol=8e-11)
    combined=pd.read_csv(ROOT/"artifacts/kiselev_identifiability_grid/combined_features.csv"); metrics=json.loads((ROOT/"artifacts/kiselev_identifiability_grid/grid_metrics.json").read_text()); mat,names=standardized_matrix(combined,cfg,metrics["feature_scaling"],.00125,-.58,"ringdown_plus_all_shooting")
    np.testing.assert_allclose(np.linalg.svd(mat,compute_uv=False)[-1],1.6793375215780408,atol=4e-12)
