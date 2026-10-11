"""Fail-closed first-example policy; no executable or electronic-state mocks."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "example/Wannier90/abinit_gaas"
sys.path.insert(0, str(EXAMPLE))
spec = importlib.util.spec_from_file_location("gaas_workflow", EXAMPLE / "workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
from prepare import PINS, checked_datasets, prepare, prepare_reuse


@pytest.fixture
def manifold():
    # Ten semicore, four valence and ten conduction states at two source points.
    bands = np.array(
        [
            -30.0,
            -29.0,
            -28.0,
            -27.0,
            -26.0,
            -25.0,
            -24.0,
            -23.0,
            -22.0,
            -21.0,
            -8.0,
            -4.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ]
    )
    energies = np.stack((bands, bands + 0.1))
    windows = {
        "dis_win_min": -10.0,
        "dis_win_max": 11.0,
        "dis_froz_min": -10.0,
        "dis_froz_max": 4.5,
    }
    return energies, windows


@pytest.mark.parametrize(
    "field,value",
    [
        ("dis_win_min", -7.0),
        ("dis_win_max", 9.0),
        ("dis_froz_min", -7.0),
        ("dis_froz_max", -0.1),
        ("dis_froz_max", -10.0),
        ("dis_win_min", float("nan")),
        ("dis_win_max", float("inf")),
    ],
)
def test_window_changes_cannot_drop_valence_or_retained_states(manifold, field, value):
    energies, windows = manifold
    workflow.validate_windows(energies, windows, 0)
    windows[field] = value
    with pytest.raises(ValueError, match="D3"):
        workflow.validate_windows(energies, windows, 0)


def test_nonfinite_electronic_energy_prevents_window_admission(manifold):
    energies, windows = manifold
    energies[1, 17] = np.nan
    with pytest.raises(ValueError, match="D3"):
        workflow.validate_windows(energies, windows, 0)


@pytest.mark.parametrize("residual", ["NaN", "Infinity", "-1D-14", "2D-12"])
def test_source_residual_must_actually_meet_declared_target(tmp_path, residual):
    (tmp_path / "gaas-nc.stdout").write_text(
        f"At SCF step 12, vres2 = {residual} < 1D-12 =>converged.\nCalculation completed.\n"
    )
    with pytest.raises(ValueError, match="D2"):
        workflow.require_source(tmp_path, {"kind": "nc", "scf_residual_target": 1e-12})


@pytest.mark.parametrize("receipt", ["frozen-workflow.json", "gaas.wout"])
def test_characterization_cannot_refreeze_after_previous_freeze_or_localization(
    tmp_path, receipt
):
    path = tmp_path / receipt
    path.write_text("previous scientific receipt")
    with pytest.raises(ValueError, match="D3"):
        workflow.characterize(tmp_path, tmp_path / "unneeded-datasets")
    assert path.read_text() == "previous scientific receipt"


def test_prepare_cannot_overwrite_existing_source_plan(tmp_path):
    path = tmp_path / "source-plan.json"
    path.write_text('{"mesh":[4,4,4]}')
    with pytest.raises(ValueError, match="D3"):
        prepare("nc", tmp_path / "unneeded-datasets", tmp_path, mesh=8)
    assert json.loads(path.read_text()) == {"mesh": [4, 4, 4]}


def test_mismatched_dataset_rejected_before_input_creation(tmp_path):
    name = "Psdj_nc_fr_04_pbe_std_psp8/Ga.psp8"
    dataset = tmp_path / name
    dataset.parent.mkdir()
    dataset.write_bytes(b"wrong physical dataset")
    with pytest.raises(ValueError, match=f"D1.*{PINS[name]}.*found"):
        checked_datasets(tmp_path, "nc")


@pytest.mark.parametrize("error", [None, "input_setopt: invalid window"])
def test_external_output_missing_or_failed_never_becomes_result(tmp_path, error):
    (tmp_path / "frozen-workflow.json").write_text("{}")
    if error is not None:
        (tmp_path / "gaas.werr").write_text(error)
    with pytest.raises(ValueError, match="D4"):
        workflow.consume(tmp_path, "gaas")
    assert not (tmp_path / "gaas-consumer-report.json").exists()


def test_frozen_window_cannot_exceed_eight_states_at_any_source_point(manifold):
    energies, windows = manifold
    windows["dis_froz_max"] = float(energies[1, 18])
    with pytest.raises(ValueError, match="D3"):
        workflow.validate_windows(energies, windows, 0)


@pytest.mark.parametrize("paw", [False, True])
def test_nonorthogonal_trial_character_matches_independent_full_metric_fields(paw):
    metric = (
        np.array([[2, 1j / 3, 0], [-1j / 3, 3, 0.25], [0, 0.25, 5]], dtype=complex)
        if paw
        else np.eye(3)
    )
    trials = np.array([[1, 1j], [1j, 2], [1 + 1j, 1]], dtype=complex)
    states = np.array(
        [[1 + 1j, -2 + 3j], [2 - 1j, 0.5 + 1j], [1j, 3 - 1j]], dtype=complex
    )
    states /= np.sqrt(np.einsum("ib,ij,jb->b", states.conj(), metric, states).real)
    overlaps = states.conj().T @ metric @ trials
    gram = trials.conj().T @ metric @ trials
    actual = workflow.trial_span_character(overlaps, gram)
    # Explicit full fields in an orthonormal metric frame, independent of the
    # implementation's normal-equation contraction and nonorthogonal Gram.
    full_trials = np.linalg.cholesky(metric).conj().T @ trials
    full_states = np.linalg.cholesky(metric).conj().T @ states
    orthogonal, _ = np.linalg.qr(full_trials, mode="reduced")
    expected = np.sum(abs(full_states.conj().T @ orthogonal) ** 2, axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "gram",
    [
        np.array([[1, 1], [1, 1]], dtype=complex),
        np.array([[1, 2], [2, 1]], dtype=complex),
        np.array([[1, 1j], [1j, 1]], dtype=complex),
    ],
)
def test_invalid_trial_span_metric_cannot_certify_manifold(gram):
    with pytest.raises(ValueError, match="D3"):
        workflow.trial_span_character(np.array([[1 + 1j, 2 - 1j]]), gram)


PSPDIR = Path(__file__).resolve().parents[2] / "abinit/tests/Pspdir"


def _reuse_source(tmp_path):
    """A minimal already-run source folder a new experiment may reuse."""
    source = tmp_path / "mesh6"
    source.mkdir()
    deck = source / "gaas-nc.abi"
    deck.write_text("# completed 6^3 deck\n")
    (source / "gaas-nc.stdout").write_text(
        "At SCF step 12, vres2 = 1D-13 < tolvrs= 1D-12 =>converged.\n"
        "Calculation completed.\n"
    )
    for name in ("gaas-nco_DS2_WFK.nc", "gaas-nco_DS3_WFK.nc"):
        (source / name).write_bytes(b"real source wavefunction")
    base = {
        "kind": "nc",
        "mesh": [6, 6, 6],
        "scf_residual_target": 1e-12,
        "nscf_residual_target": 1e-20,
        "source_bands": 24,
        "retained_original_bands": list(range(10, 24)),
        "num_wann": 8,
        "held_out_reduced_kpoints": [[0.02, 0, 0]],
        "held_out_target_original_bands": list(range(10, 18)),
    }
    (source / "source-plan.json").write_text(json.dumps(base))
    return source


def test_reuse_plan_preserves_source_and_annotates_authorization(tmp_path):
    source = _reuse_source(tmp_path)
    plan = prepare_reuse(
        "nc",
        PSPDIR,
        tmp_path / "v3",
        source,
        "gaas-nc-v3-mesh6",
        "2026-10-04 explicit larger-mesh authorization; 4^3 memory/smoke unchanged",
    )
    record = json.loads(plan.read_text())
    assert record["mesh"] == [6, 6, 6]
    assert record["num_wann"] == 8
    assert record["held_out_target_original_bands"] == list(range(10, 18))
    assert record["source_reuse_dir"] == str(source)
    assert "2026-10-04" in record["user_authorization"]
    assert record["experiment_id"] == "gaas-nc-v3-mesh6"
    assert (tmp_path / "v3/gaas-nc.abi").read_text() == "# completed 6^3 deck\n"
    for name in ("gaas-nco_DS2_WFK.nc", "gaas-nco_DS3_WFK.nc"):
        assert (tmp_path / "v3" / name).read_bytes() == b"real source wavefunction"


def test_reuse_plan_cannot_overwrite_or_mix_kinds(tmp_path):
    source = _reuse_source(tmp_path)
    prepare_reuse(
        "nc", PSPDIR, tmp_path / "v3", source, "id", "authorization"
    )
    with pytest.raises(ValueError, match="D3"):
        prepare_reuse(
            "nc", tmp_path / "u", tmp_path / "v3", source, "id2", "authorization2"
        )
    with pytest.raises(ValueError, match="D2"):
        prepare_reuse(
            "paw", PSPDIR, tmp_path / "pawv3", source, "id", "authorization"
        )


def test_reuse_convergence_evidence_may_span_two_declared_reports(tmp_path):
    source = _reuse_source(tmp_path)
    # Split evidence: SCF residual in one report, completion in a remainder run.
    (source / "gaas-nc.stdout").write_text(
        "At SCF step 12, vres2 = 1D-13 < tolvrs= 1D-12 =>converged.\n"
    )
    remainder = source / "gaas-nc-remainder.stdout"
    remainder.write_text("Calculation completed.\n")
    plan = prepare_reuse(
        "nc",
        PSPDIR,
        tmp_path / "v3",
        source,
        "id",
        "authorization",
        source_stdouts=["gaas-nc.stdout", "gaas-nc-remainder.stdout"],
    )
    record = json.loads(plan.read_text())
    assert record["source_stdouts"] == [
        "gaas-nc.stdout",
        "gaas-nc-remainder.stdout",
    ]
    path = workflow.require_source(
        tmp_path / "v3", json.loads(plan.read_text())
    )
    assert path.is_file()
    # Each declared report is required; dropping the remainder fails closed.
    del record["source_stdouts"]
    (tmp_path / "v3/gaas-nc.stdout").write_text(
        "At SCF step 12, vres2 = 1D-13 < tolvrs= 1D-12 =>converged.\n"
    )
    with pytest.raises(ValueError, match="D2"):
        workflow.require_source(tmp_path / "v3", record)


class _LegacyReader:
    pass


def test_consumer_refuses_non_wsvec_hamiltonio_reader():
    with pytest.raises(ValueError, match="D5.*_gen_ham_scheme2"):
        workflow.hamiltonio_provenance(_LegacyReader)


def test_consumer_declares_supported_reader_provenance():
    pytest.importorskip("HamiltonIO")
    from HamiltonIO.wannier import WannierHam

    provenance = workflow.hamiltonio_provenance(WannierHam)
    assert provenance["wsvec_scheme2"] is True
    assert provenance["reader_module"]
