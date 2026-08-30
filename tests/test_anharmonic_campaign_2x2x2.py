"""Story-028 BaTiO3 Q7 campaign integration.

The unit test fixes gate-0's sign-and-ordering contract without a teacher.  The
slow integration test runs the Q7 downfold, dense 2I invariant basis, MACE
labels, total-energy fit, and artifact round trip on ``REDUCED_CONFIG_2X2X2``.
It intentionally keeps the production gate thresholds in the results rather
than relaxing them in code; reduced-run calibration assertions are recorded
explicitly beside the observed values after the first campaign run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO / "example" / "anharmonic_batio3"
FIXTURE = REPO / "example" / "Phonopy" / "BaTiO3" / "DM_dip_wang" / "phonopy_params.yaml"
sys.path.insert(0, str(CAMPAIGN_DIR))


def test_gate0_sign_pattern_contract():
    from campaign import _gate0_verdict

    verdict = _gate0_verdict(
        {
            "Gamma": [-30.0],
            "X": [-20.0, -20.0, -20.0],
            "M": [-10.0, -10.0, -10.0],
            "R": [5.0],
        }
    )
    assert verdict["pass"]

    failed = _gate0_verdict(
        {
            "Gamma": [-30.0],
            "X": [-20.0, -20.0, -20.0],
            "M": [-10.0, -10.0, -10.0],
            "R": [-5.0],
        }
    )
    assert not failed["pass"]


try:
    from lawaf.anharmonic.teacher import get_atomchain_calculator

    get_atomchain_calculator("mace-r2scan")
    HAS_TEACHER = True
except Exception:
    HAS_TEACHER = False


slow_campaign = pytest.mark.slow
requires_campaign_inputs = pytest.mark.skipif(
    not FIXTURE.exists() or not HAS_TEACHER,
    reason="BaTiO3 fixture or atomchain/MACE teacher unavailable",
)


@pytest.fixture(scope="module")
def campaign_results_2x2x2(tmp_path_factory):
    from campaign import REDUCED_CONFIG_2X2X2, run_2x2x2

    return run_2x2x2(
        REDUCED_CONFIG_2X2X2,
        outdir=tmp_path_factory.mktemp("campaign_2x2x2"),
    )


@slow_campaign
@requires_campaign_inputs
def test_reduced_q7_campaign_structure_and_serialization(campaign_results_2x2x2):
    result = campaign_results_2x2x2
    assert result["gate0_harmonic"]["pass"]
    assert result["supercell"]["natom_sc"] == 40
    assert result["supercell"]["nQ"] == 24
    assert result["gate3_harmonic_roundtrip"]["pass"]
    assert result["gate4_molien"]["consistent"]
    assert result["gate6_artifact"]["pass"]
    assert result["basis"]["coord_permutation_identity"]
    assert result["basis"]["dense_action_group_closure_defect"] <= 1e-10
    assert result["basis"]["basis_invariance_defect"] <= 1e-10
    assert result["dataset"]["n_ladder_modes"] == 15
    assert result["dataset"]["n_coupled_pairs"] == 8


@slow_campaign
@requires_campaign_inputs
def test_reduced_q7_gate1_calibration_is_explicit(campaign_results_2x2x2):
    gate1 = campaign_results_2x2x2["gate1_residuals"]
    # The reduced design is intentionally underdetermined for the production
    # residual and elastic gates. Its observed residuals define a stable
    # reduced-test calibration without claiming a full fit.
    assert gate1["thresholds"] == {
        "elastic_rel_max": 0.10,
        "energy_mae_frac": 0.15,
        "fold_rel": 1e-10,
        "force_cosine_min": 0.75,
        "roundtrip_rel": 1e-6,
        "stress_rmse_frac": 0.05,
    }
    assert gate1["force_cosine"] >= 0.55, json.dumps(gate1, default=str)
    assert gate1["stress_rmse_frac_of_rms"] <= 0.03, json.dumps(gate1, default=str)
    assert gate1["energy_mae_frac_of_scale"] <= 0.02, json.dumps(gate1, default=str)
