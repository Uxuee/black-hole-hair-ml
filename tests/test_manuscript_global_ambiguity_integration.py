from pathlib import Path
import json
import re


ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/"paper/journal_identifiability_final"
TEX=FINAL/"main.tex"
SUMMARY=ROOT/"artifacts/global_ambiguity_audit/global_ambiguity_summary.json"


def text(): return TEX.read_text(encoding="utf-8")


def test_finite_domain_section_and_caveat_exist():
    source=text()
    assert r"\subsection{Finite-domain ambiguity check}" in source
    assert r"\section{Finite-Domain Observable-Ambiguity Audit}" in source
    assert "does not establish continuous global injectivity" in source
    assert "not a proof of continuous global identifiability" in source
    assert "globally identifiable" not in source


def test_machine_readable_claims_are_integrated():
    source=text(); summary=json.loads(SUMMARY.read_text())
    assert summary["point_count"]==121 and summary["pair_count"]==7260 and summary["finite_k_pair_count"]==5995
    combined=summary["feature_sets"]["ringdown_plus_photon_geometry"]
    ring=summary["feature_sets"]["ringdown"]
    for value in ("0.03155","0.002462","12.8","0.02517","0.23216","0.21886"):
        assert value in source
    assert f"{100*ring['local_neighbor_preservation']['top_1']:.1f}\\%" in source
    assert f"{100*combined['local_neighbor_preservation']['top_1']:.1f}\\%" in source
    assert "does not preserve every local nearest-" in source


def test_positive_control_and_figure_are_appendix_only():
    source=text(); appendix=source.index(r"\appendix"); section=source.index(r"\label{app:ambiguity}")
    assert section>appendix
    assert "$2.63\\times10^{-12}$" in source[section:]
    assert "required positive control" in source[section:]
    figure="finite_domain_global_ambiguity.pdf"
    assert source.index(figure)>appendix
    assert (FINAL/"figures"/figure).is_file()


def test_all_cross_references_resolve_in_source():
    source=text(); labels=set(re.findall(r"\\label\{([^}]+)\}",source)); refs=set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",source))
    assert refs<=labels and "app:ambiguity" in labels and "fig:finite-ambiguity" in labels
