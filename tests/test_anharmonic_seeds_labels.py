"""Story-024 tests: SeedsLabels provider (ADR-011 adapter) bridging
check_compatibility to lawaf.interfaces.phonopy.symmetry_seeds.

TEST-001  provider wiring: SeedsLabels implements the RepresentationLabels
          protocol (story-015) and returns labels at Gamma.
TEST-002  the BaTiO3 T1u soft triple at Gamma decomposes into P4mm daughter
          labels, one per seed direction, through check_compatibility.
TEST-003  constructed disagreement: the character-derived view of the same
          window says "T1u" while the seed labels name P4mm daughters per
          direction; the report carries the seed labels (label_source
          "provided") and the discrepancy is named by assert_compatible.
TEST-004  import boundary: no anharmonic module imports symmetry_seeds at
          module import time outside the lazy adapter in representation.py.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.anharmonic.compatibility import (  # noqa: E402
    CharacterLabels,
    RepresentationDeclaration,
    RepresentationLabels,
    assert_compatible,
    check_compatibility,
)
from lawaf.anharmonic.representation import (  # noqa: E402
    SeedsLabels,
    build_space_group_action,
)

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = np.zeros(3)
SOFT_BANDS = (0, 1, 2)  # the unstable T1u triplet (story-010 fixture)


@pytest.fixture(scope="module")
def phonon():
    pytest.importorskip("spgrep_modulation")
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def model(phonon):
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    return PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)


@pytest.fixture(scope="module")
def sga(phonon):
    return build_space_group_action(phonon)


def soft_window(model, cut=-0.05):
    """projector_fn: eigenvalue-cut window (omega^2 < cut), soft branches."""
    cache = {}

    def projector_fn(q):
        key = tuple(np.round(q, 8) % 1.0)
        if key not in cache:
            cache[key] = model.solve(q)
        evals, evecs = cache[key]
        cols = np.where(evals < cut)[0]
        V = evecs[:, cols]
        return V @ V.conj().T

    return projector_fn


def normalize(labels):
    return [str(lab).replace(" ", "") for lab in labels]


# the soft triplet's seed labels: P4mm daughter, one per seed direction
P4MM_AXES = {
    "P4mmaxis-pure(x)",
    "P4mmaxis-pure(y)",
    "P4mmaxis-pure(z)",
}


# ======================================================================
# TEST-001  provider wiring
# ======================================================================
def test_seeds_provider_implements_protocol_and_labels_gamma(phonon):
    provider = SeedsLabels(phonon, bands=SOFT_BANDS)
    assert isinstance(provider, RepresentationLabels)  # protocol satisfied
    labels = provider.irreps_at(GAMMA)
    assert len(labels) == len(SOFT_BANDS)
    assert set(normalize(labels)) == P4MM_AXES


# ======================================================================
# TEST-002  T1u soft triple -> P4mm daughter labels per seed direction
# ======================================================================
def test_check_compatibility_soft_triple_decomposes_to_p4mm_daughters(sga, phonon):
    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u"])
    report = check_compatibility(
        sga, declaration, SeedsLabels(phonon, bands=SOFT_BANDS), qpoints=[GAMMA]
    )
    assert isinstance(report.label_source, str)
    assert report.label_source == "provided"
    (check,) = report.checks
    assert check.expected == ["T1u"]  # induced declaration view
    assert len(check.found) == 3  # one label per seed direction
    assert set(normalize(check.found)) == P4MM_AXES


# ======================================================================
# TEST-003  characters vs seeds disagree: report keeps seed labels,
#           discrepancy named
# ======================================================================
def test_disagreement_report_uses_seed_labels_and_names_it(sga, model, phonon):
    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u"])
    provider = SeedsLabels(phonon, bands=SOFT_BANDS)
    character_found = normalize(
        CharacterLabels(sga, soft_window(model)).irreps_at(GAMMA)
    )
    seed_found = normalize(provider.irreps_at(GAMMA))
    # the constructed disagreement on the SAME 3-band window:
    assert character_found == ["T1u"]
    assert seed_found != character_found
    report = check_compatibility(sga, declaration, provider, qpoints=[GAMMA])
    assert report.label_source == "provided"  # the report uses the seed labels
    (check,) = report.checks
    assert normalize(check.found) == seed_found
    assert not report.passed
    with pytest.raises(ValueError) as excinfo:
        assert_compatible(report)
    msg = str(excinfo.value)
    assert "T1u" in msg and "P4mm" in msg  # discrepancy named on both sides


# ======================================================================
# TEST-004  import boundary (story-024 acceptance criterion 4)
# ======================================================================
def _mentions_symmetry_seeds(node):
    if isinstance(node, ast.Import):
        return any(
            alias.name == "symmetry_seeds" or alias.name.endswith(".symmetry_seeds")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return "symmetry_seeds" in module.split(".") or any(
            alias.name == "symmetry_seeds" for alias in node.names
        )
    return False


def test_no_anharmonic_module_imports_symmetry_seeds_outside_adapter():
    import lawaf.anharmonic as pkg

    pkg_dir = Path(pkg.__file__).parent
    adapter = "representation.py"
    offenders = []
    for path in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not _mentions_symmetry_seeds(node):
                continue
            if path.name != adapter:
                offenders.append(f"{path.name}:{node.lineno}")
            elif node.col_offset == 0:
                offenders.append(f"{path.name}:{node.lineno} (module level)")
    assert not offenders, (
        "symmetry_seeds imported outside the lazy adapter: "
        + ", ".join(offenders)
    )
