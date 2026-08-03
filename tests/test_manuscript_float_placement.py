import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper" / "ai4s2026" / "main.tex"


def _tex() -> str:
    return TEX_PATH.read_text(encoding="utf-8")


def test_all_labels_are_unique_and_all_references_resolve():
    text = _tex()
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)


def test_every_figure_is_cited_before_its_environment():
    text = _tex()
    environments = re.finditer(
        r"\\begin\{figure\*?\}.*?\\label\{([^}]+)\}.*?\\end\{figure\*?\}",
        text,
        flags=re.DOTALL,
    )
    found = 0
    for environment in environments:
        found += 1
        label = environment.group(1)
        prefix = text[: environment.start()]
        assert re.search(rf"\\ref\{{{re.escape(label)}\}}", prefix), label
    assert found == 15


def test_no_active_figure_placeholder_or_duplicate_graphic():
    text = _tex()
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    assert len(graphics) == len(set(graphics))
    assert "placeholder" not in " ".join(graphics).lower()
    assert "\\fbox" not in text
    assert "\\missingfigure" not in text


def test_no_figure_environment_after_bibliography():
    text = _tex()
    bibliography = text.index(r"\bibliographystyle{IEEEtran}")
    assert r"\begin{figure" not in text[bibliography:]


def test_required_float_barriers_are_present():
    text = _tex()
    assert re.search(
        r"\\FloatBarrier\s*\\section\{Inverse Learning from Physical Shooting Features\}",
        text,
    )
    assert re.search(r"\\FloatBarrier\s*\\section\{Limitations and Future Work\}", text)
    assert re.search(r"\\FloatBarrier\s*\\section\{Conclusion\}", text)
    assert re.search(r"\\FloatBarrier\s*\\bibliographystyle\{IEEEtran\}", text)
