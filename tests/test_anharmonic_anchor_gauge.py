"""Story-026 anchor-general constrained-gauge contracts."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.anharmonic.compatibility import RepresentationDeclaration
from lawaf.anharmonic.gauge import (
    assert_projective_closure,
    check_window_irreps,
    constrain_amn_one_q,
    constrain_builder_amn,
)
from lawaf.anharmonic.representation import (
    WindowBands,
    WindowLegalityBlock,
    build_space_group_action,
)
from lawaf.anharmonic.io import load_symmetry, save_symmetry
from lawaf.interfaces.phonopy import phonon_downfolder as pdf
from lawaf.params import WannierParams


def _d4h_e_plus_a():
    """The E(x,y)+A(z) reducible D4h representation."""
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    reflection = np.diag([1.0, -1.0])
    reps = {}
    index = 0
    for inversion in (1.0, -1.0):
        for reflected in (False, True):
            for power in range(4):
                planar = np.linalg.matrix_power(rotation, power)
                if reflected:
                    planar = reflection @ planar
                reps[index] = np.block(
                    [
                        [inversion * planar, np.zeros((2, 1))],
                        [np.zeros((1, 2)), np.array([[inversion]])],
                    ]
                )
                index += 1
    return reps


def test_reducible_d4h_window_constraint_and_star_covariance():
    """A zone-corner E+A window is constrained without a Schur shortcut."""
    dw = _d4h_e_plus_a()
    # Two copies in the full band space make the projection nontrivial.
    window = {g: np.block([[d, np.zeros_like(d)], [np.zeros_like(d), d]]) for g, d in dw.items()}
    rng = np.random.default_rng(26)
    seed, _ = np.linalg.qr(rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3)))
    constrained, info = constrain_amn_one_q(window, dw, seed)
    assert info["eps"] <= 1e-10
    assert info["ortho"] <= 1e-10

    # An independently chosen star-frame change preserves the covariance law.
    left, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    transported = left @ constrained @ right.conj().T
    transformed_m = {g: left @ m @ left.conj().T for g, m in window.items()}
    transformed_dw = {g: right @ d @ right.conj().T for g, d in dw.items()}
    residual = max(
        np.abs(transformed_m[g] @ transported - transported @ transformed_dw[g]).max()
        for g in dw
    )
    assert residual <= 1e-10


def test_projective_class_closes_and_constraint_converges():
    """A fractional-character R-like class closes up to its cocycle."""
    projective = {0: np.eye(2), 1: 1j * np.eye(2)}
    factors = assert_projective_closure(
        projective, lambda g, h: (g + h) % 2
    )
    assert factors[(1, 1)] == pytest.approx(-1.0)
    seed = np.array([[1.0, 0.0], [0.0, 1.0]])
    constrained, info = constrain_amn_one_q(projective, projective, seed)
    assert info["eps"] <= 1e-10
    assert np.abs(constrained.conj().T @ constrained - np.eye(2)).max() <= 1e-10


def test_window_irrep_declaration_names_anchor_and_both_contents():
    """Declared anchor content must match the validated retained window."""
    qx = (0.5, 0.0, 0.0)
    windows = WindowBands(
        representatives={qx: (0, 1, 4)},
        bands={qx: (0, 1, 4)},
        legality={
            qx: (
                WindowLegalityBlock(10.0, 2, ("E",), 0.0),
                WindowLegalityBlock(12.0, 1, ("A",), 0.0),
            )
        },
    )
    declaration = RepresentationDeclaration(
        wyckoff="1b", site_irreps=["T1u"], window_irreps={qx: ["A"]}
    )
    with pytest.raises(ValueError, match=r"X.*declared \['A'\].*actual \['A', 'E'\]"):
        check_window_irreps(declaration, windows)
    check_window_irreps(
        replace(declaration, window_irreps={qx: ["E", "A"]}), windows
    )

def test_symmetry_persists_anchor_declaration_and_residuals(tmp_path):
    """Per-anchor content and residuals survive the symmetry netCDF roundtrip."""
    fixture = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
    phonon = phonopy.load(phonopy_yaml=str(fixture), is_nac=True)
    phonon.symmetrize_force_constants()
    sga = build_space_group_action(phonon)
    qx = (0.5, 0.0, 0.0)
    window = WindowBands(
        representatives={qx: (0, 1, 4)},
        bands={qx: (0, 1, 4)},
        legality={qx: (WindowLegalityBlock(10.0, 3, ("E", "A"), 0.0),)},
    )
    declaration = RepresentationDeclaration(
        wyckoff="1b",
        site_irreps=["T1u"],
        window_irreps={qx: ["E", "A"]},
    )
    path = tmp_path / "anchor-symmetry.nc"
    save_symmetry(
        path,
        sga,
        declaration=declaration,
        window_bands=window,
        gauge_diagnostics={"eps": {qx: 4.0e-13}},
    )
    record = load_symmetry(path)
    assert record.declaration == declaration
    assert record.gauge_residuals == {qx: pytest.approx(4.0e-13)}

def test_v1_gamma_only_constraint_fingerprint_is_unchanged():
    """Omitted window_bands keeps the original Gamma-only numerical path."""
    fixture = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
    phonon = phonopy.load(phonopy_yaml=str(fixture), is_nac=False)
    phonon.symmetrize_force_constants()
    sga = build_space_group_action(phonon)
    params = WannierParams(
        method="projected",
        nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func="Gauss",
        weight_func_params=(-20.0, 20.0),
        kmesh=(1, 1, 1),
    )
    downfolder = pdf.PhonopyDownfolder(phonon=phonon, params=params)
    downfolder._prepare_data()
    downfolder.atoms = downfolder.model.atoms
    downfolder.builder.prepare()
    downfolder.builder.get_Amn()
    constrain_builder_amn(
        downfolder.builder,
        RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u"]),
        sga=sga,
        params=downfolder.params,
    )
    assert hashlib.sha256(downfolder.builder.Amn.tobytes()).hexdigest() == (
        "4bcfbbc9b07d56099d6ef73636c53d5e94020450e7daf5216a9dc1be30a78273"
    )

def test_q7_anchors_constrain_and_propagate_on_full_2x2x2_mesh():
    """Q7 is constrained independently at Gamma, X, M and R."""
    fixture = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
    phonon = phonopy.load(phonopy_yaml=str(fixture), is_nac=True)
    phonon.symmetrize_force_constants()
    sga = build_space_group_action(phonon)
    q7 = {
        (0.0, 0.0, 0.0): (0, 1, 2),
        (0.5, 0.0, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.5): (0, 1, 2),
    }
    from lawaf.anharmonic.representation import check_window_legality

    validated = check_window_legality(sga, phonon, q7)
    declaration = RepresentationDeclaration(
        wyckoff="1b",
        site_irreps=["T1u"],
        window_irreps={
            q: [
                irrep
                for block in validated.legality[q]
                for irrep in block.irreps
            ]
            for q in q7
        },
    )
    params = WannierParams(
        method="projected",
        nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func="Gauss",
        weight_func_params=(-20.0, 20.0),
        kmesh=(2, 2, 2),
        window_bands=q7,
    )
    downfolder = pdf.PhonopyDownfolder(
        phonon=phonon, params=params, is_nac=True
    )
    downfolder._prepare_data()
    downfolder.atoms = downfolder.model.atoms
    downfolder.builder.prepare()
    downfolder.builder.get_Amn()
    diagnostics = constrain_builder_amn(
        downfolder.builder, declaration, sga=sga, params=downfolder.params
    )
    anchors = set(diagnostics["constrained_qs"])
    assert len(anchors) == 4
    assert {
        int(np.count_nonzero(np.isclose(q, 0.5)))
        for q in anchors
    } == {0, 1, 2, 3}
    assert all(diagnostics["eps"][q] <= 1e-10 for q in anchors)
    builder = downfolder.builder
    for q in anchors:
        ik = int(
            np.argmin(
                np.linalg.norm(
                    np.mod(builder.kpts - q + 0.5, 1.0) - 0.5,
                    axis=1,
                )
            )
        )
        selected_rows = [
            builder.ibands.index(band)
            for band in validated.bands[tuple(np.round(np.mod(q, 1.0), 8))]
        ]
        expected = np.sort(builder.get_eval_k(ik)[selected_rows])
        got = np.linalg.eigvalsh(
            builder.Amn[ik].conj().T
            @ np.diag(builder.get_eval_k(ik))
            @ builder.Amn[ik]
        )
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=0.0)
    for ik, q in enumerate(builder.kpts):
        projector = builder.Amn[ik] @ builder.Amn[ik].conj().T
        for operation in range(sga.n_ops):
            q_image = sga.qmap(operation, q)
            image_index = int(
                np.argmin(
                    np.linalg.norm(
                        np.mod(builder.kpts - q_image + 0.5, 1.0) - 0.5,
                        axis=1,
                    )
                )
            )
            transport = (
                builder.get_psi_k(image_index).conj().T
                @ sga.matrix(operation, q)
                @ builder.get_psi_k(ik)
            )
            residual = np.abs(
                builder.Amn[image_index] @ builder.Amn[image_index].conj().T
                - transport @ projector @ transport.conj().T
            ).max()
            assert residual <= 1e-10, (operation, q, residual)
