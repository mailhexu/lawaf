"""Complete PAW A/M against independent Cartesian all-electron Gaussian fields."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lawaf.interfaces.abinit.basis_datasets import sha256_file
from lawaf.interfaces.abinit.paw_coefficients import (
    SupplierMeta,
    SuppliedCoefficientArrays,
    witness_compare,
    PawCoefficientError,
)
from lawaf.interfaces.abinit.paw_kernels import HARMONIC_CONVENTION
from lawaf.interfaces.abinit.producer import (
    AbinitWannierInputs,
    GenerationPlan,
    ProducerError,
)
from lawaf.interfaces.abinit.source_arrays import (
    CallerArraySource,
    CallerMetadata,
    SymmetryMapEntry,
)
from test_abinit_paw_trials import _header

ROOT = Path(__file__).resolve().parents[2]
W90 = ROOT / ".tmp/w90-lib-research/build-dev-serial/wannier90.x"


def _gaussian_xml(path, augmented):
    a, d = 0.1, 0.004
    r = a * np.expm1(d * np.arange(int(np.ceil(np.log1p(8.0 / a) / d)) + 1))
    radial = [
        2 / np.pi**0.25 * np.exp(-r * r / 2),
        np.sqrt(8 / (3 * np.sqrt(np.pi))) * r * np.exp(-r * r / 2),
    ]
    scales = (1.2, 0.8) if augmented else (1.0, 1.0)
    lines = [
        '<paw_dataset version="0.7">',
        '<atom symbol="H"/>',
        f'<radial_grid eq="r=a*(exp(d*i)-1)" a="{a}" d="{d}" id="log1">',
        "<values>" + " ".join(f"{v:.17e}" for v in r) + "</values>",
        "</radial_grid>",
        '<paw_radius rc="8.0"/>',
        "<valence_states>",
        '<state n="1" l="0" rc="8.0" f="1" id="H0"/>',
        '<state n="2" l="1" rc="8.0" f="1" id="H1"/>',
        "</valence_states>",
    ]
    for l, values in enumerate(radial):
        for kind, scale in [
            ("ae_partial_wave", scales[l]),
            ("pseudo_partial_wave", 1.0),
            ("projector_function", 1.0),
        ]:
            lines += [
                f'<{kind} state="H{l}" grid="log1">',
                " ".join(f"{v:.17e}" for v in values * scale),
                f"</{kind}>",
            ]
    lines += ["</paw_dataset>"]
    path.write_text("\n".join(lines) + "\n")
    return np.array([scales[0], scales[1], scales[1], scales[1]])


def _fixture(folder, augmented, kpoints=None):
    xml = folder / "hydrogen.xml"
    scales = _gaussian_xml(xml, augmented)
    length = 18.0
    position = np.array([0.137, 0.193, 0.217]) * length
    if kpoints is None:
        kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    g = np.array(list(np.ndindex(41, 41, 41)), dtype=np.int32) - 20
    rng = np.random.default_rng(6503)
    vectors, _ = np.linalg.qr(rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4)))
    coefficients = vectors.reshape(4, 2, 4).transpose(0, 2, 1) / scales[:, None, None]
    spin = np.array([[np.sqrt(3) / 2, 1j / 2], [1j / 2, np.sqrt(3) / 2]])
    coefficients = coefficients @ spin.T
    betas = []
    states = {}
    for k, kpoint in enumerate(kpoints):
        beta = coefficients * np.exp(
            1j
            * (
                0.37 * k * np.arange(4)[None, :, None]
                + (0.43 + 0.21 * k) * np.arange(2)[None, None, :]
            )
        )
        q = (g + kpoint) * 2 * np.pi / length
        envelope = (
            np.pi ** (-0.75) * (2 * np.pi) ** 1.5 * np.exp(-np.sum(q * q, axis=1) / 2)
        )
        fourier = np.column_stack(
            (
                envelope,
                -1j * np.sqrt(2) * q[:, 2] * envelope,
                -1j * np.sqrt(2) * q[:, 0] * envelope,
                -1j * np.sqrt(2) * q[:, 1] * envelope,
            )
        )
        fourier *= np.exp(-1j * (q @ position))[:, None] / np.sqrt(length**3)
        states[k, 0] = np.ascontiguousarray(np.einsum("gi,ibs->bgs", fourier, beta))
        betas.append(beta)
    h = _header(
        [position / length],
        [4],
        [hashlib.md5(xml.read_bytes()).hexdigest()],
        kpoints,
        np.eye(3) * length,
        nspinor=2,
    )
    h = replace(
        h,
        znucltypat=np.array([1.0]),
        znuclpsp=np.array([1.0]),
        zionpsp=np.array([1.0]),
        title=("Authored Gaussian H PAW",),
        mband=4,
        nband=np.full(2, 4, np.int32),
        bantot=8,
        npwarr=np.full(2, len(g), np.int32),
        occ=np.ones(8),
        ecut=80.0,
        ecut_eff=80.0,
        kptopt=3,
        kptrlatt=np.diag([2, 1, 1]).astype(np.int32),
        kptrlatt_orig=np.diag([2, 1, 1]).astype(np.int32),
    )
    maps = tuple(
        SymmetryMapEntry(
            k,
            k,
            np.eye(3, dtype=int),
            np.zeros(3),
            np.zeros(3, dtype=int),
            "same",
            False,
        )
        for k in range(2)
    )
    metadata = CallerMetadata(
        header=h,
        eigenvalues=np.array(
            [[[-0.3], [-0.3]], [[0.1], [0.1]], [[0.5], [0.5]], [[0.9], [0.9]]]
        ),
        eigenvalue_units="hartree",
        length_units="bohr",
        symmetry_map=maps,
        nkpt_mesh=2,
        mesh_kpoints=kpoints,
    )
    source = CallerArraySource(
        metadata, states, {k: np.ascontiguousarray(g.T) for k in range(2)}
    )
    doc = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": HARMONIC_CONVENTION,
            "normalization": "normalized",
        },
        "trial_definition": "paw_augmented_partial_wave",
        "spin_frame_labels": True,
        "species": [
            {
                "name": "H",
                "kind": "paw_xml",
                "dataset_path": str(xml),
                "sha256": sha256_file(xml),
                "radial_ids": [
                    {"id": "H0", "label": "s", "l": 0, "harmonics": [0]},
                    {"id": "H1", "label": "pz", "l": 1, "harmonics": [0]},
                ],
            }
        ],
        "sites": [
            {
                "label": "H0",
                "species": "H",
                "position": (position / length).tolist(),
                "local_axes": np.eye(3).tolist(),
            }
        ],
    }
    manifest = folder / "trials.json"
    manifest.write_text(json.dumps(doc))
    return source, manifest, scales, np.array(betas), position


def _cartesian_references(scales, betas, position):
    nodes, weight = hermgauss(32)
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    weights = (
        weight[:, None, None] * weight[None, :, None] * weight[None, None, :]
    ).reshape(-1)
    # The Hermite measure supplies exp(-r^2) from the product of two fields.
    fields = (
        np.pi ** (-0.75)
        * np.column_stack((np.ones(len(points)), np.sqrt(2) * points[:, [2, 0, 1]]))
        * scales[None, :]
    )
    psi = np.einsum("ri,kibs->krbs", fields, betas)
    trials = np.zeros((len(points), 4, 2), complex)
    for column, (radial, spin) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        trials[:, column, spin] = fields[:, radial]

    def projection(k):
        return sum(
            psi[k, :, :, spin].conj().T @ (weights[:, None] * trials[:, :, spin])
            for spin in range(2)
        )

    def overlap(k, j, b):
        phase = np.exp(-1j * ((points + position) @ b))
        return sum(
            psi[k, :, :, spin].conj().T
            @ ((weights * phase)[:, None] * psi[j, :, :, spin])
            for spin in range(2)
        )

    return projection, overlap


@pytest.mark.skipif(not W90.exists(), reason="pinned external executable not built")
@pytest.mark.parametrize("augmented", [True, False])
def test_complete_coherent_paw_projection_neighbor_metric_and_nc_limit_against_defining_ae_fields(
    tmp_path, augmented
):
    source, basis, scales, betas, position = _fixture(tmp_path, augmented)
    plan = GenerationPlan((0, 1, 2, 3), 4, (2, 1, 1), spinors=True)
    computed = AbinitWannierInputs(source, plan, trial_manifest=basis)
    computed.write_win(tmp_path, "analytic")
    subprocess.run(
        ["rtk", "proxy", str(W90), "-pp", "analytic"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    nnkp = computed.validate_nnkp(tmp_path / "analytic.nnkp")
    computed._paw_metric.prepare_neighbors(nnkp.nnkpts)
    values = np.stack(
        [computed._paw_provider.band_block(k, 0, (0, 4)) for k in range(2)]
    )[None]
    expected_coefficients = betas.transpose(0, 2, 3, 1)[None]
    np.testing.assert_allclose(values, expected_coefficients, atol=1e-8, rtol=0)
    meta = SupplierMeta(
        schema_name="lawaf.paw_projection_coefficients",
        schema_version=1,
        function_kind="dualprojector",
        state_frame="source",
        source_channel=0,
        kpoints=source.header.kpt,
        spin_tuple=(1, 2, 4),
        band_range=(0, 4),
        dataset_md5=tuple(source.header.md5_pseudos),
        atom_species=("H",),
        projector_counts=(4,),
        projector_labels=computed._paw_provider.projector_labels,
        harmonic_convention=HARMONIC_CONVENTION,
        conjugation="trial_bra",
        raw_normalization=True,
        phase_declaration="bloch_source_frame",
    )
    record = SuppliedCoefficientArrays(values, meta, source.header).record
    with pytest.raises(PawCoefficientError) as unverified:
        AbinitWannierInputs(source, plan, trial_manifest=basis, paw_coefficients=record)
    assert unverified.value.diagnostic == "D4"
    witness_compare(record, values)
    supplied = AbinitWannierInputs(
        source, plan, trial_manifest=basis, paw_coefficients=record
    )
    supplied._paw_metric.prepare_neighbors(nnkp.nnkpts)
    reference_a, reference_m = _cartesian_references(scales, betas, position)
    reciprocal = 2 * np.pi * np.linalg.inv(source.header.rprimd).T
    worst = 0.0
    for k in range(2):
        actual = supplied._projection_tile(k)
        expected = reference_a(k)
        worst = max(worst, float(np.max(abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        states = source.coefficients(k, 0, (0, 4))
        electronic = values[0, k]
        metric = np.einsum("bgs,cgs->bc", states.conj(), states)
        metric += supplied._paw_metric.projection(electronic, electronic.transpose(2, 0, 1))
        np.testing.assert_allclose(metric, np.eye(4), atol=1e-8, rtol=0)
        for slot, row in enumerate(nnkp.nnkpts[k]):
            b = (
                source.header.kpt[row[0]] + row[1:] - source.header.kpt[k]
            ) @ reciprocal
            actual = supplied._pair_tile(k, slot, nnkp)
            expected = reference_m(k, int(row[0]), b)
            worst = max(worst, float(np.max(abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        computed._projection_tile(0), reference_a(0), atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        computed._pair_tile(0, 0, nnkp),
        supplied._pair_tile(0, 0, nnkp),
        atol=1e-12,
        rtol=0,
    )
    assert supplied._paw_provider.compute_calls == 0
    with pytest.raises(PawCoefficientError) as missing_neighbor:
        supplied._paw_metric.overlap(values[0, 0], values[0, 1], [1.111, 2, 3])
    assert missing_neighbor.value.diagnostic == "D6"
    if augmented:
        corrupted = values.copy()
        corrupted[:, :, :, 1, :] = 0
        bad = SuppliedCoefficientArrays(corrupted, meta, source.header).record
        with pytest.raises(PawCoefficientError):
            witness_compare(bad, values)
        with pytest.raises(PawCoefficientError) as rejected:
            AbinitWannierInputs(
                source, plan, trial_manifest=basis, paw_coefficients=bad
            )
        assert rejected.value.diagnostic == "D4"
        with pytest.raises(ProducerError) as error:
            AbinitWannierInputs(source, plan, paw_coefficients=record)
        assert error.value.diagnostic == "D1"
    else:
        h = replace(source.header, usepaw=0)
        nc = CallerArraySource(
            replace(source.metadata, header=h),
            {(k, 0): source.coefficients(k, 0, (0, 4)) for k in range(2)},
            {k: source.kg(k) for k in range(2)},
        )

        def nc_trials(k, g):
            spatial = supplied._paw_trials(k, g)
            result = np.zeros((len(g), 4, 2), complex)
            result[:, 0, 0] = spatial[:, 0]
            result[:, 1, 1] = spatial[:, 0]
            result[:, 2, 0] = spatial[:, 1]
            result[:, 3, 1] = spatial[:, 1]
            return result

        bare = AbinitWannierInputs(nc, plan, nc_trials)
        for k in range(2):
            np.testing.assert_allclose(
                supplied._projection_tile(k),
                bare._projection_tile(k),
                atol=1e-12,
                rtol=0,
            )
            for slot in range(nnkp.nnkpts.shape[1]):
                np.testing.assert_allclose(
                    supplied._pair_tile(k, slot, nnkp),
                    bare._pair_tile(k, slot, nnkp),
                    atol=1e-12,
                    rtol=0,
                )
    print(
        "MEASURE independent complete coherent PAW AE-field error",
        worst,
        "augmented",
        augmented,
        flush=True,
    )


@pytest.mark.parametrize("antiunitary", [False, True])
def test_supplied_ibz_projectors_transport_atom_image_orbitals_and_coherent_spin(
    tmp_path, antiunitary
):
    from lawaf.interfaces.abinit.paw_provider import PawCoefficientProvider
    from lawaf.interfaces.abinit.source_mesh import MeshSource

    kpoint = np.array([0.125, 0.25, 0.375])
    full, _, _, betas, _ = _fixture(
        tmp_path, True, np.array([np.zeros(3), kpoint])
    )
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    cell_image = np.array([1, -1, 0])
    position = full.header.xred[0]
    translation = position + cell_image - rotation.T @ position
    shift = np.array([1, -1, 0])
    sign = -1 if antiunitary else 1
    target = (sign * rotation @ kpoint + shift)[None]
    header = replace(
        full.header,
        nkpt=1,
        kpt=kpoint[None],
        nband=np.array([4], np.int32),
        bantot=4,
        npwarr=full.header.npwarr[:1],
        istwfk=full.header.istwfk[:1],
        istwfk_raw=full.header.istwfk_raw[:1],
        occ=full.header.occ[:4],
        wtk=np.ones(1),
        kptopt=1,
        kptrlatt=np.zeros((3, 3), np.int32),
        kptrlatt_orig=np.zeros((3, 3), np.int32),
    )
    metadata = replace(
        full.metadata,
        header=header,
        eigenvalues=np.ascontiguousarray(full.metadata.eigenvalues[:, 1:2]),
        nkpt_mesh=1,
        mesh_kpoints=target,
        symmetry_map=(
            SymmetryMapEntry(
                0, 0, sign * rotation, translation, shift, "same", antiunitary
            ),
        ),
    )
    source = CallerArraySource(
        metadata,
        {(0, 0): full.coefficients(1, 0, (0, 4))},
        {0: full.kg(1)},
    )
    paths = [tmp_path / "hydrogen.xml"]
    stored = PawCoefficientProvider(source, paths)
    values = stored.band_block(0, 0, (0, 4))[None, None]
    meta = SupplierMeta(
        schema_name="lawaf.paw_projection_coefficients",
        schema_version=1,
        function_kind="dualprojector",
        state_frame="ibz",
        source_channel=0,
        kpoints=header.kpt,
        spin_tuple=(1, 2, 4),
        band_range=(0, 4),
        dataset_md5=header.md5_pseudos,
        atom_species=("H",),
        projector_counts=(4,),
        projector_labels=stored.projector_labels,
        harmonic_convention=HARMONIC_CONVENTION,
        conjugation="trial_bra",
        raw_normalization=True,
        phase_declaration="bloch_source_frame",
    )
    record = SuppliedCoefficientArrays(values, meta, header).record
    mesh = MeshSource(source, target, time_reversal=antiunitary)
    supplied = PawCoefficientProvider(mesh, paths, supplied_record=record)
    computed = PawCoefficientProvider(mesh, paths)
    # Direct Cartesian fields: s, z, x, y under r -> R^T r become s, z, y, -x.
    # The atom-cell image contributes the original Bloch phase before TR.
    beta = betas[1]
    expected = np.stack((beta[0], beta[1], -beta[3], beta[2]), axis=0)
    expected *= np.exp(2j * np.pi * (kpoint @ cell_image))
    if antiunitary:
        expected = np.stack((expected[:, :, 1].conj(), -expected[:, :, 0].conj()), axis=-1)
    spin = np.diag([np.exp(-0.25j * np.pi), np.exp(0.25j * np.pi)])
    expected = (expected @ spin.T).transpose(1, 2, 0)
    np.testing.assert_allclose(supplied.band_block(0, 0, (0, 4)), expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(computed.band_block(0, 0, (0, 4)), expected, atol=1e-8, rtol=0)
    assert supplied.compute_calls == 0
    assert np.max(abs(expected - beta.transpose(1, 2, 0))) > 0.1
