from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
TEX=ROOT/"paper/journal_identifiability_final/main.tex"


def test_nearest_improvement_numbers_match_manuscript():
    text=TEX.read_text(encoding="utf-8")
    table=pd.read_csv(ROOT/"artifacts/traditional_baselines/nearest_geometry_improvement.csv")
    for row in table.itertuples():
        assert f"{row.ringdown_identifiable_wq_NMAE:.6f}" in text
        assert f"{row.ringdown_plus_photon_geometry_identifiable_wq_NMAE:.6f}" in text
        assert f"{100*row.fractional_reduction:.1f}\\%" in text


def test_traditional_comparison_table_matches_machine_readable_values():
    text=TEX.read_text(encoding="utf-8")
    table=pd.read_csv(ROOT/"artifacts/traditional_baselines/traditional_vs_ml_protocol_aggregate.csv")
    table=table[table.feature_set=="ringdown_plus_photon_geometry"]
    labels={"Nearest physical model":"Nearest physical model","Jacobian local inverse":"Local Jacobian inverse","HGB":"HGB","RF":"RF","MLP":"MLP"}
    protocols=["random_interpolation","grouped_physical_interpolation","directional_extrapolation"]
    display=lambda value: str(Decimal(str(value)).quantize(Decimal("0.0001"),rounding=ROUND_HALF_UP))
    for method,label in labels.items():
        values=[]
        for protocol in protocols:
            row=table[(table.method==method)&(table.protocol==protocol)].iloc[0]
            values.append(f"{display(row.k_NMAE)}/{display(row.identifiable_wq_NMAE)}")
        assert f"{label} & " + " & ".join(values) in text


def test_unclipped_jacobian_count_and_figures_are_exact():
    text=TEX.read_text(encoding="utf-8")
    predictions=pd.read_csv(ROOT/"artifacts/traditional_baselines/jacobian_local_inverse_predictions.csv")
    manifest=json.loads((ROOT/"artifacts/traditional_baselines/run_manifest.json").read_text())
    outside=int(predictions.outside_parameter_domain.sum())
    assert outside == manifest["out_of_domain_jacobian_predictions"] == 1315
    assert f"{outside} of {len(predictions)}" in text and "were not clipped" in text
    for name in ["traditional_vs_ml_inverse.pdf","conditioning_vs_wq_recovery.pdf"]:
        assert (ROOT/"figures"/name).read_bytes() == (ROOT/"paper/journal_identifiability_final/figures"/name).read_bytes()


def test_conditioning_language_is_noncausal_and_appendix_only():
    text=TEX.read_text(encoding="utf-8")
    assert "do not consistently predict pointwise recovery" in text
    assert "no causal relationship is inferred" in text
    assert text.index("conditioning_vs_wq_recovery.pdf") > text.index("\\appendix")
