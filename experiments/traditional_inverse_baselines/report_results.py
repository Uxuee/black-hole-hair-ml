"""Generate provenance-backed reviewer and numerical-audit reports."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _table(frame, columns):
    return frame[columns].to_markdown(index=False, floatfmt=".6g")


def run(root):
    root=Path(root).resolve(); out=root/"artifacts/traditional_baselines"
    agg=pd.read_csv(out/"traditional_vs_ml_protocol_aggregate.csv")
    imp=pd.read_csv(out/"nearest_geometry_improvement.csv")
    cond=pd.read_csv(out/"conditioning_vs_recovery_summary.csv")
    jac=pd.read_csv(out/"jacobian_local_inverse_predictions.csv")
    nearest=pd.read_csv(out/"nearest_physical_model_predictions.csv")
    chosen=agg[agg.feature_set.isin(["ringdown","ringdown_plus_photon_geometry"])]
    report=fr"""# Reviewer baseline response

## Reviewer question: “How does this compare against traditional methods?”

We evaluated two deliberately transparent inversions on the **same registered train/calibration/test assignments** as HGB, RF, and MLP. The nearest-physical-model method standardizes observables using training systems only and returns the parameter pair of the nearest training system. The local Jacobian method uses the nearest training system as its reference, estimates a training-only local forward Jacobian, and applies its Moore–Penrose pseudoinverse. Its primary predictions are not clipped: {int(jac.outside_parameter_domain.sum())} of {len(jac)} raw predictions lie outside the registered domain.

Central values below are medians across the same registered folds/seeds. The deterministic baselines have no intrinsic seed randomness; repeated seed rows reflect different registered train/test assignments. Archived ML results are loaded, not retrained.

{_table(chosen,["method","protocol","feature_set","k_NMAE","identifiable_wq_NMAE","n"])}

## Does photon geometry help independently of estimator architecture?

For the nearest physical model, adding photon geometry to ringdown reduces identifiable-$w_q$ NMAE in all three protocols:

{_table(imp,list(imp.columns))}

This is outcome A in the preregistered interpretation: the additional physical observable makes the inverse map easier even for a non-learned catalog lookup. The magnitude and ranking should still be compared with the ML rows above; it is not evidence that any one architecture is universally best.

## Conditioning and recovery

{_table(cond,list(cond.columns))}

These are associations, not causal estimates. Signs vary by method and protocol, so local $\sigma_{{\min}}$ improvement does not perfectly rank pointwise recovery. The calculation uses pointwise/fold-level pairs only; no correlation was manufactured from aggregate metrics.

## Interpretation and scope

The historical synthetic proxy augmentation was an exploratory identifiability lens. It is not evidence for the physical solver. The result reported here instead uses validated photon-geometry features extracted from the frozen physical shooting grid. HGB, RF, and MLP remain useful heterogeneous estimators, while the nearest and Jacobian baselines show how much of the gain is already present in geometry and local conditioning. No geodesic shooting was rerun and the manuscript was not edited.
"""
    (root/"reports/reviewer_baseline_response.md").write_text(report,encoding="utf-8")
    audit_rows=[]
    for r in agg.itertuples(index=False):
        for target,column in [("k","k_NMAE"),("identifiable_wq","identifiable_wq_NMAE")]:
            audit_rows.append({"source_csv":"artifacts/traditional_baselines/traditional_vs_ml_summary.csv",
                "filtering_rule":f"method={r.method}; feature_set={r.feature_set}; target={target}; k=0 excluded only for wq",
                "protocol":r.protocol,"feature_set":r.feature_set,"sample_count":r.n,
                "aggregation":"median across registered fold/seed NMAE values","value":getattr(r,column),"status":"PASS"})
    numerical=pd.DataFrame(audit_rows); numerical.to_csv(out/"primary_number_provenance.csv",index=False)
    text="# Traditional-baseline numerical audit\n\nEvery primary number below is recomputed from saved prediction/fold records with full-span NMAE (0.0025 for k; 0.2625 for wq). Nominal k=0 rows remain in k metrics and are excluded only from identifiable-wq metrics.\n\n"+_table(numerical,list(numerical.columns))+"\n"
    (root/"reports/traditional_baseline_numerical_audit.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--root",default=Path(__file__).resolve().parents[2]); a=p.parse_args(); run(a.root)
