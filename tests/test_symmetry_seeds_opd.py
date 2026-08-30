"""Tests for OPD selection + family listing — story-011.

`list_opd_families`, selection policy (default/SG-number/index,
multi-eigenspace semantics), seed-set construction (OPD line + little-group
images, QR, deterministic ordering), per-band records, determinism.

Oracle: research memo finding 3 — BaTiO3 Γ eigenspace 0 (soft T1u) lists 9
OPD families; OPD1 → P4mm (#99) order 8 axis-pure; Amm2 (#38) face-diagonal;
R3m (#160) body-diagonal; T2u (band 3 block) → P-4m2 (#115); acoustic block
→ parent group Pm-3m (#221).
"""

from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.interfaces.phonopy import symmetry_seeds as ss  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = np.zeros(3)
SOFT = (0, 1, 2)
T2U = (3, 4, 5)  # acoustic triple — see below for the real T2u block


@pytest.fixture(scope="module")
def phonon():
    pytest.importorskip("spgrep_modulation")
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


def families_by_sg(families, sg_number):
    return [f for f in families if f.sg_number == sg_number]


# ---------------------------------------------------------------- TEST-001
def test_list_opd_families_soft_block(phonon, capsys):
    families = ss.list_opd_families(phonon, GAMMA, bands=SOFT)
    # table printed with the same content
    out = capsys.readouterr().out
    assert "P4mm" in out and "R3m" in out
    p4mm = families_by_sg(families, 99)
    amm2 = families_by_sg(families, 38)
    r3m = families_by_sg(families, 160)
    assert p4mm and amm2 and r3m
    p = p4mm[0]
    assert p.subgroup_order == 8 and p.ndir == 1
    assert p.direction_summary.startswith("axis-pure")
    assert any(f.direction_summary.startswith("face-diagonal") for f in amm2)
    assert any(f.direction_summary.startswith("body-diagonal") for f in r3m)
    # a generic (low-symmetry) family exists
    assert any(f.direction_summary.startswith("generic") for f in families)
    # records are globally uniquely indexed, all ndir>=1, one eigenspace
    idx = [f.index for f in families]
    assert len(set(idx)) == len(idx)
    assert all(f.ndir >= 1 for f in families)
    assert all(f.eigenspace_index == 0 for f in families)


# ---------------------------------------------------------------- TEST-002
def test_default_selection_highest_order(phonon):
    rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT)
    assert not rep.fell_back
    sgs = [r.sg_number for r in rep.per_band]
    assert sgs == [99, 99, 99]  # P4mm default (highest order 8, ndir==1)
    # tie-break lowest index is exercised implicitly: two P4mm families exist


# ---------------------------------------------------------------- TEST-003
def test_selection_overrides(phonon):
    # SG-number override
    rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT, opd=160)
    assert not rep.fell_back
    assert [r.sg_number for r in rep.per_band] == [160, 160, 160]
    assert all(r.direction_summary.startswith("body-diagonal") for r in rep.per_band)
    # index override: pick the R3m family found in the listing
    fams = ss.list_opd_families(phonon, GAMMA, bands=SOFT)
    r3m_index = families_by_sg(fams, 160)[0].index
    rep2 = ss.get_symmetry_anchor_wfn(
        phonon, GAMMA, bands=SOFT, opd_index=r3m_index
    )
    assert [r.sg_number for r in rep2.per_band] == [160, 160, 160]
    # no match -> ValueError listing valid families
    with pytest.raises(ValueError, match="P4mm"):
        ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT, opd=1)


# ---------------------------------------------------------------- TEST-004
def test_seed_axis_purity_and_labels(phonon):
    # soft block: axis-pure P4mm seeds
    rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT)
    for band, col in enumerate(rep.psi[:, 0:3].T):
        w = np.abs(col).reshape(-1, 3).sum(axis=0)
        off = np.delete(w, np.argmax(w)).sum() / w.sum()
        assert off <= 1e-10
        assert rep.per_band[band].sg_number == 99
    # the T2u silent mode block is bands 9-11 (freq 8.534 cm^-1 block)
    rep_t2u = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(9, 10, 11))
    assert not rep_t2u.fell_back
    assert {r.sg_number for r in rep_t2u.per_band} == {115}  # P-4m2
    # acoustic triple: daughters are the parent group Pm-3m (#221)
    rep_ac = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(3, 4, 5))
    assert not rep_ac.fell_back
    assert {r.sg_number for r in rep_ac.per_band} == {221}


# ---------------------------------------------------------------- TEST-005
def test_undergeneration_falls_back(phonon, monkeypatch):
    """Images < dim → per-anchor fallback (guard)."""
    real_images = ss._distinct_images

    def starved(coeffs, irrep):
        images = real_images(coeffs, irrep)
        return images[:1]  # always undergenerate

    monkeypatch.setattr(ss, "_distinct_images", starved)
    with pytest.warns(UserWarning, match="undergenerate|images"):
        rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT)
    assert rep.fell_back


# ---------------------------------------------------------------- TEST-006
def test_determinism(phonon):
    rep1 = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT, opd=99)
    rep2 = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT, opd=99)
    np.testing.assert_array_equal(rep1.psi, rep2.psi)
    assert rep1.per_band == rep2.per_band
    assert [r.band for r in rep1.per_band] == [0, 1, 2]


# ---------------------------------------------------------------- TEST-007
def test_multi_eigenspace_selection(phonon):
    bands = (0, 1, 2, 9, 10, 11)  # soft T1u + T2u eigenspaces
    # scalar SG applies to every touched eigenspace: 99 exists only in T1u
    with pytest.raises(ValueError, match="eigenspace"):
        ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=bands, opd=99)
    # per-eigenspace mapping works
    rep = ss.get_symmetry_anchor_wfn(
        phonon,
        GAMMA,
        bands=bands,
        opd={0: 99, 3: 115},
    )
    assert not rep.fell_back
    assert [r.sg_number for r in rep.per_band] == [99, 99, 99, 115, 115, 115]


# --------------------------------------------------------- per-band record
def test_per_band_record_fields(phonon):
    rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=SOFT)
    rec = rep.per_band[0]
    fields = {
        "band", "eigenspace_index", "eigenspace_dim", "irrep_chars",
        "family_index", "sg_number", "sg_symbol", "direction_summary",
        "frequency",
    }
    assert fields == set(vars(rec).keys())
    assert rec.eigenspace_dim == 3
    assert rec.irrep_chars["little_group_order"] == 48
    assert isinstance(rec.family_index, int)
    assert rec.frequency < 0  # soft mode (freq −6.049 cm^-1)


def test_multi_block_scalar_opd_index_from_later_block(phonon):
    """SPEC-003 regression: a scalar opd_index owned by a LATER touched
    eigenspace must not raise while resolving the first block (first block
    defaults, owning block selects)."""
    bands = (0, 1, 2, 9, 10, 11)
    families = ss.list_opd_families(phonon, GAMMA, bands=bands)
    later = [
        f for f in families if f.eigenspace_index == 3 and f.ndir == 1
    ]
    assert later, "expected ndir==1 families in the T2u block"
    target = later[0]
    rep = ss.get_symmetry_anchor_wfn(
        phonon, GAMMA, bands=bands, opd_index=target.index
    )
    assert not rep.fell_back
    # first block falls back to the default rule (P4mm)
    assert [r.sg_number for r in rep.per_band[:3]] == [99] * 3
    # owning block selects exactly the requested family
    assert [r.family_index for r in rep.per_band[3:]] == [target.index] * 3


def test_soft_block_family_count_oracle(phonon):
    """Research memo finding 3: the BaTiO3 soft block lists exactly 9
    families, contiguous globally-unique indices."""
    families = ss.list_opd_families(phonon, GAMMA, bands=SOFT)
    assert len(families) == 9
    assert [f.index for f in families] == list(range(9))
    assert {f.sg_number for f in families} == {99, 6, 38, 8, 160, 1}


def test_listing_determinism(phonon, capsys):
    """Repeated listings agree bitwise in rows and printed table."""
    r1 = ss.list_opd_families(phonon, GAMMA, bands=SOFT)
    t1 = capsys.readouterr().out
    r2 = ss.list_opd_families(phonon, GAMMA, bands=SOFT)
    t2 = capsys.readouterr().out
    assert [str(f) for f in r1] == [str(f) for f in r2]
    assert t1 == t2


def test_listing_validates_bands(phonon):
    with pytest.raises(ValueError, match="invalid anchor bands"):
        ss.list_opd_families(phonon, GAMMA, bands=(999,))
    with pytest.raises(ValueError, match="invalid anchor bands"):
        ss.list_opd_families(phonon, GAMMA, bands=())


def test_listing_forwards_degeneracy_tol(phonon, monkeypatch):
    """The listing must hand degeneracy_tol to build_modulation (parity
    with the get path) — a custom tolerance changes eigenspace grouping."""
    recorded = {}

    real_build = ss.build_modulation

    def recording(phonon_obj, qpoint, **kwargs):
        recorded.update(kwargs)
        return real_build(phonon_obj, qpoint, **kwargs)

    monkeypatch.setattr(ss, "build_modulation", recording)
    ss.list_opd_families(phonon, GAMMA, bands=SOFT, degeneracy_tol=1e-2)
    assert recorded.get("degeneracy_tol") == 1e-2
