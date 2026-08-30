"""Story-027 BaTiO3 Q7 downfold integration proof.

The source of truth for the harmonic check is the NAC-active fixture dynamical
matrix at all eight 2I-commensurate q points.  At off-axis q, window legality
uses the qhat-preserving little subgroup; Q7 remains legal there.  The folded
real-space LWF kernel is compared in signed omega-squared (dynamical-matrix
eigenvalue) units, rather than relative frequency: this preserves the sign
of imaginary modes and is the quantity Fourier folding is required to
reproduce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO / "example" / "anharmonic_batio3"
FIXTURE = (
    REPO / "example" / "Phonopy" / "BaTiO3" / "DM_dip_wang" / "phonopy_params.yaml"
)
sys.path.insert(0, str(CAMPAIGN_DIR))

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="BaTiO3 DM_dip_wang fixture not found"
)


@pytest.fixture(scope="module")
def downfold_2i():
    import campaign

    return campaign.downfold_fixture_2x2x2()


def test_q7_folded_harmonic_roundtrip_preserves_signed_omega_squared(downfold_2i):
    """All eight Q7 blocks, including X/M instabilities, fold exactly."""
    result = downfold_2i["roundtrip"]

    assert result["n_qpoints"] == 8
    assert result["max_relative_omega2_deviation"] <= 1e-6
    assert result["max_builder_relative_omega2_deviation"] <= 1e-6
    assert len(result["rows"]) == 8
    assert all(len(row["window_bands"]) == 3 for row in result["rows"])
    assert all(row["relative_omega2_deviation"] <= 1e-6 for row in result["rows"])
    assert all(
        row["builder_relative_omega2_deviation"] <= 1e-6
        for row in result["rows"]
    )

    by_name = {row["name"]: row for row in result["rows"]}
    assert any(freq < 0.0 for freq in by_name["X"]["source_freqs_cm1"])
    assert any(freq < 0.0 for freq in by_name["M"]["source_freqs_cm1"])


def test_q7_mapping_and_canonical_gauge_are_oh_covariant(downfold_2i):
    """The canonical Q7 frame yields a Reynolds-exact Q24-to-Cart120 mapping."""
    assert downfold_2i["nQ"] == 24
    assert downfold_2i["natom_sc"] == 40

    mapping = downfold_2i["mapping"]
    assert mapping["shape"] == [120, 24]
    assert set(mapping["raw_intertwine_defects"]) == {"Gamma", "X", "M", "R"}
    assert set(mapping["symmetrized_intertwine_defects"]) == {"Gamma", "X", "M", "R"}
    assert all(value >= 0.0 for value in mapping["raw_intertwine_defects"].values())
    assert all(
        value <= 1e-12
        for value in mapping["symmetrized_intertwine_defects"].values()
    )
    assert mapping["symmetrization"]["rank"] == 24
    assert mapping["coordinate_action"]["unitary_defect"] <= 1e-10
    assert mapping["coordinate_action"]["imaginary_component"] <= 1e-10

    gauge = downfold_2i["gauge"]
    assert gauge["gauge_eps_max"] <= 1e-10
    assert len(gauge["eps"]) == 4
    assert gauge["realification"]["takagi_residual"] <= 1e-10
    assert gauge["realification"]["time_reversal_residual"] <= 1e-10
    assert gauge["realification"]["max_imaginary_wannR"] <= 1e-10
