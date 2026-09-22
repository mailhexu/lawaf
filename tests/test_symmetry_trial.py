"""R2 symmetry trial orbitals: Wigner-projected seeds at Gamma."""

from pathlib import Path

import numpy as np
import pytest

from lawaf.anharmonic.compatibility import RepresentationDeclaration
from lawaf.interfaces.phonopy.symmetry_trial import symmetry_trial_orbitals

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="BaTiO3 fixture not found"
)


@pytest.fixture(scope="module")
def report_batio3():
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    phonon.symmetrize_force_constants()
    decl = RepresentationDeclaration(
        wyckoff="1b", site_irreps=["T1u"], strain_sector=False
    )
    return symmetry_trial_orbitals(phonon, decl)


def test_projection_is_exact(report_batio3):
    """The projected Ti seeds carry exactly ONE T1u copy: projector
    idempotent, projected span covariant, characters match to 1e-12."""
    r = report_batio3
    assert len(r.blocks) == 1
    b = r.blocks[0]
    assert b["name"] == "T1u"
    assert b["n_carriers"] == 3
    assert b["projector_idempotency"] < 1e-12
    assert b["sector_purity_residual"] < 1e-6
    assert b["sector_covariance"] < 1e-12
    assert r.orbitals.shape == (15, 3)
    gram = r.orbitals.conj().T @ r.orbitals
    assert np.allclose(gram, np.eye(3), atol=1e-12)
def test_sector_not_copy_finding(report_batio3):
    """Documented research finding (memo 2026-09-22): the symmetry
    projection selects the irrep sector exactly but spreads over the
    four T1u copies -- soft-subspace svmin ~ 0.68, never 1.0."""
    r = report_batio3
    t1u = [c for c in r.copy_overlaps if c["overlap_product"] > 1e-3]
    assert len(t1u) >= 3  # the seeds mix at least three copies
    assert r.soft_overlap_svmin == pytest.approx(0.6796, abs=0.01)
    assert r.soft_overlap_svmin < 0.99




def test_to_dict(report_batio3):
    import json

    d = json.loads(json.dumps(report_batio3.to_dict()))
    assert d["wyckoff"] == "1b"
    assert d["site_irreps"] == ["T1u"]
    assert d["soft_overlap_svmin"] is not None


def test_non_gamma_rejected():
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    decl = RepresentationDeclaration(
        wyckoff="1b",
        site_irreps=["T1u"],
        anchors=[(0.5, 0.0, 0.0)],
    )
    with pytest.raises(ValueError, match="Gamma point only"):
        symmetry_trial_orbitals(phonon, decl)


def test_bad_wyckoff_rejected():
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    decl = RepresentationDeclaration(wyckoff="9z", site_irreps=["T1u"])
    with pytest.raises(ValueError, match="no atom with"):
        symmetry_trial_orbitals(phonon, decl)


def test_dict_declaration_accepted():
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    r = symmetry_trial_orbitals(
        phonon, {"wyckoff": "1b", "site_irreps": ["T1u"]}
    )
    assert r.blocks[0]["n_carriers"] == 3


def test_repeated_site_irrep_extracts_independent_carriers():
    """['T1u', 'T1u'] = two copies: 6 orthonormal carriers; the second
    copy is frequency-split by the dynamical matrix when the seed
    cluster is degenerate (QT-F02), and without a reference the same
    request raises a clear error."""
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    decl = RepresentationDeclaration(
        wyckoff="1b", site_irreps=["T1u", "T1u"], strain_sector=False
    )
    r = symmetry_trial_orbitals(phonon, decl)
    assert r.orbitals.shape == (15, 6)
    gram = r.orbitals.conj().T @ r.orbitals
    assert np.allclose(gram, np.eye(6), atol=1e-10)
    with pytest.raises(ValueError, match="overshoots|cannot furnish"):
        symmetry_trial_orbitals(phonon, decl, with_reference=False)


def test_nonmaximal_site_symmetry_orbit_works():
    """Oxygen 3c site (stabilizer D4h, proper subgroup): the induced-
    sector projector must not KeyError (QT-F01)."""
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    decl = RepresentationDeclaration(
        wyckoff="3c", site_irreps=["Eu"], strain_sector=False
    )
    r = symmetry_trial_orbitals(phonon, decl)
    assert r.blocks[0]["dim_site"] == 2
    assert r.blocks[0]["n_carriers"] == 6  # 3-site orbit x Eu
    assert r.blocks[0]["projector_idempotency"] < 1e-12
    assert r.blocks[0]["sector_purity_residual"] < 1e-6
