"""R1 IrrepField: per-q displacement-representation irrep content."""

from pathlib import Path

import pytest

from lawaf.anharmonic import build_space_group_action
from lawaf.anharmonic.irrepfield import evaluate_irrep_field

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="BaTiO3 fixture not found"
)


@pytest.fixture(scope="module")
def field_batio3():
    import phonopy

    phonon = phonopy.load(str(FIXTURE), is_nac=False, produce_fc=False)
    phonon.symmetrize_force_constants()
    sga = build_space_group_action(phonon)
    qpts = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.5),
    ]
    return sga, evaluate_irrep_field(sga, qpts)


def test_gamma_decomposition_matches_phonopy_labels(field_batio3):
    """Gamma displacement rep of Pm-3m BaTiO3: 4 x T1u + T2u (15 modes)."""
    _sga, f = field_batio3
    g = f.point_at((0.0, 0.0, 0.0))
    assert g.mult_of("T1u") == 4
    assert g.mult_of("T2u") == 1
    assert g.n_modes == 15
    assert g.little_group_order == 48


def test_zone_boundary_points_integer_and_complete(field_batio3):
    _sga, f = field_batio3
    for q in [(0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 0.5)]:
        p = f.point_at(q)
        assert p.n_modes == 15
        for b in p.blocks:
            assert b.multiplicity >= 1
            assert b.dim in (1, 2, 3)


def test_to_dict_round_trip(field_batio3):
    import json

    _sga, f = field_batio3
    d = json.loads(json.dumps(f.to_dict()))
    assert len(d["points"]) == 4
    names = {b["name"] for p in d["points"] for b in p["blocks"]}
    assert "T1u" in names
    g = next(p for p in d["points"] if p["qpoint"] == [0.0, 0.0, 0.0])
    assert sum(b["dim"] * b["multiplicity"] for b in g["blocks"]) == 15


def test_point_at_periodic_boundary(field_batio3):
    """Minimum-image matching across the periodic boundary (QT-F04)."""
    _sga, f = field_batio3
    p = f.point_at((1.0 - 1e-9, 0.0, 0.0))
    assert p.qpoint == (0.0, 0.0, 0.0)


def test_missing_point_raises(field_batio3):
    sga, f = field_batio3
    with pytest.raises(KeyError):
        f.point_at((0.25, 0.0, 0.0))

