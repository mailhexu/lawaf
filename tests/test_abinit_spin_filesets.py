"""Physical channel separation and coherent contractions over real array sources."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lawaf.interfaces.abinit.producer import (
    AbinitWannierInputs,
    GenerationPlan,
    HA_EV,
    ProducerError,
)
from lawaf.interfaces.abinit.spin_filesets import AbinitWannierFileSet
from lawaf.interfaces.abinit.source_arrays import CallerArraySource
from lawaf.io.w90 import read_mmn
from test_abinit_producer_writers import (
    _mesh_source,
    _gaussian_trials,
    _mesh_neighbors,
    _write_nnkp,
)


def _source(spinor=False):
    base, _ = _mesh_source(2, nband=3)
    h = base.header
    channels = 1 if spinor else 2
    components = 2 if spinor else 1
    header = replace(
        h,
        nsppol=channels,
        nspinor=components,
        nspden=4 if spinor else 2,
        nband=np.full(h.nkpt * channels, 3, dtype=np.int32),
        bantot=h.nkpt * channels * 3,
        occ=np.ones(h.nkpt * channels * 3),
        kptopt=3,
    )
    states = {}
    for k in range(h.nkpt):
        for channel in range(channels):
            g = base.kg(k).T + h.kpt[k]
            fields = []
            for band in range(3):
                row = []
                for component in range(components):
                    row.append(
                        np.exp(
                            -(0.35 + 0.08 * band + 0.17 * component)
                            * np.sum(g * g, axis=1)
                        )
                        * np.exp(
                            1j
                            * (
                                (0.13 + 0.24 * band) * g[:, 0]
                                + (0.37 + 0.11 * component) * g[:, 1]
                                + 0.7 * component
                            )
                        )
                        * (1 + (0.3 + 0.2j) * (band + channel) * g[:, 2])
                    )
                fields.append(np.stack(row, axis=-1))
            value = np.array(fields)
            # Orthonormal physical states; both coherent components stay distinct.
            q, _ = np.linalg.qr(value.reshape(3, -1).T)
            states[k, channel] = np.ascontiguousarray(
                q.T.reshape(3, len(g), components)
            )
    energy = np.empty((3, h.nkpt, channels))
    for k in range(h.nkpt):
        for channel in range(channels):
            energy[:, k, channel] = (
                np.array([-0.4, 0.1, 0.35]) + 0.013 * k + 0.2 * channel
            )
    metadata = replace(base.metadata, header=header, eigenvalues=energy)
    return CallerArraySource(metadata, states, {k: base.kg(k) for k in range(h.nkpt)})


def _spin_trials(source):
    spatial = _gaussian_trials(source, n_per_site=2)

    def trials(k, g):
        value = spatial(k, g)
        spin = np.array([[1, 1j], [0.6 + 0.2j, 0.3 - 0.7j]], complex)
        spin /= np.linalg.norm(spin, axis=1)[:, None]
        return value[:, :, None] * spin[None, :, :]

    return trials


def _reference_a(source, trials, k, channel=0):
    c = source.coefficients(k, channel, (0, 3))
    g = source.kg(k).T
    basis = trials(k, g)
    if basis.ndim == 2:
        basis = basis[:, :, None]
    result = np.zeros((3, 2), complex)
    for m in range(3):
        for n in range(2):
            for point in range(len(g)):
                for spin in range(source.header.nspinor):
                    result[m, n] += (
                        c[m, point, spin].conjugate() * basis[point, n, spin]
                    )
    return result


def _reference_m(source, k, j, image, channel=0):
    c1 = source.coefficients(k, channel, (0, 3))
    c2 = source.coefficients(j, channel, (0, 3))
    lookup = {tuple(g): row for row, g in enumerate(source.kg(j).T)}
    result = np.zeros((3, 3), complex)
    for row, g in enumerate(source.kg(k).T):
        other = lookup.get(tuple(g + image))
        if other is not None:
            for m in range(3):
                for n in range(3):
                    for spin in range(source.header.nspinor):
                        result[m, n] += (
                            c1[m, row, spin].conjugate() * c2[n, other, spin]
                        )
    return result


def _nnkp(path, source):
    table = _mesh_neighbors(source.header.kpt, 2)
    _write_nnkp(path, source, table)
    if source.header.nspinor == 2:
        path.write_text(
            path.read_text()
            .replace("begin projection", "begin spinor_projections")
            .replace("end projection", "end spinor_projections")
        )
    return path


def test_coherent_a_m_match_independent_component_and_integer_transport_sums(tmp_path):
    source = _source(spinor=True)
    trials = _spin_trials(source)
    producer = AbinitWannierInputs(
        source, GenerationPlan((0, 1, 2), 2, (2, 2, 2), spinors=True), trials
    )
    nnkp = producer.validate_nnkp(_nnkp(tmp_path / "spin.nnkp", source))
    for k in range(source.header.nkpt):
        np.testing.assert_allclose(
            producer._projection_tile(k),
            _reference_a(source, trials, k),
            atol=1e-12,
            rtol=0,
        )
        for slot, row in enumerate(nnkp.nnkpts[k]):
            np.testing.assert_allclose(
                producer._pair_tile(k, slot, nnkp),
                _reference_m(source, k, int(row[0]), row[1:]),
                atol=1e-12,
                rtol=0,
            )
    manifests = producer.write_matrices(tmp_path / "spin.nnkp", tmp_path, "spin")
    energies = np.loadtxt(manifests.files["eig"])
    assert energies.shape == (source.header.nkpt * 3, 3)
    np.testing.assert_allclose(
        energies[:, 2].reshape(-1, 3),
        source.metadata.eigenvalues[:, :, 0].T * HA_EV,
        atol=1e-10,
        rtol=0,
    )


def test_nonreal_su2_corotation_preserves_physical_a_and_m(tmp_path):
    source = _source(spinor=True)
    trials = _spin_trials(source)
    unitary = np.array([[np.sqrt(3) / 2, 1j / 2], [1j / 2, np.sqrt(3) / 2]])
    states = {
        (k, 0): np.ascontiguousarray(source.coefficients(k, 0, (0, 3)) @ unitary.T)
        for k in range(source.header.nkpt)
    }
    rotated = CallerArraySource(
        source.metadata, states, {k: source.kg(k) for k in range(source.header.nkpt)}
    )

    def rotated_trials(k, g):
        return trials(k, g) @ unitary.T

    plan = GenerationPlan((0, 1, 2), 2, (2, 2, 2), spinors=True)
    original = AbinitWannierInputs(source, plan, trials)
    changed = AbinitWannierInputs(rotated, plan, rotated_trials)
    nnkp = original.validate_nnkp(_nnkp(tmp_path / "spin.nnkp", source))
    for k in range(source.header.nkpt):
        np.testing.assert_allclose(
            changed._projection_tile(k),
            original._projection_tile(k),
            atol=1e-12,
            rtol=0,
        )
        for slot in range(nnkp.nnkpts.shape[1]):
            np.testing.assert_allclose(
                changed._pair_tile(k, slot, nnkp),
                original._pair_tile(k, slot, nnkp),
                atol=1e-12,
                rtol=0,
            )
    # Dropping or conjugating a coherent component is a real field mutation.
    for fault in ("drop", "conjugate"):
        bad = {}
        for k in range(source.header.nkpt):
            value = source.coefficients(k, 0, (0, 3)).copy()
            value[:, :, 1] = 0 if fault == "drop" else value[:, :, 1].conj()
            bad[k, 0] = value
        broken = CallerArraySource(
            source.metadata, bad, {k: source.kg(k) for k in range(source.header.nkpt)}
        )
        wrong = AbinitWannierInputs(broken, plan, trials)._projection_tile(1)
        assert np.max(abs(wrong - _reference_a(source, trials, 1))) > 1e-3


def test_collinear_files_keep_actual_channels_and_detect_swap_average_duplicate(
    tmp_path,
):
    source = _source()
    trials = _gaussian_trials(source, n_per_site=2)
    plans = {channel: GenerationPlan((0, 1, 2), 2, (2, 2, 2)) for channel in (0, 1)}
    fileset = AbinitWannierFileSet(source, plans, trial_orbitals=trials)
    seeds = {0: "up", 1: "down"}
    fileset.write_win(tmp_path, seeds)
    paths = {
        channel: _nnkp(tmp_path / f"{seed}.nnkp", source)
        for channel, seed in seeds.items()
    }
    manifests = fileset.write_matrices(paths, tmp_path, seeds)
    actual = []
    for channel in (0, 1):
        eigen = np.loadtxt(manifests[channel].files["eig"])
        np.testing.assert_allclose(
            eigen[:, 2].reshape(-1, 3),
            source.metadata.eigenvalues[:, :, channel].T * HA_EV,
            atol=1e-10,
            rtol=0,
        )
        amn = np.loadtxt(manifests[channel].files["amn"], skiprows=2)
        values = np.empty((8, 3, 2), complex)
        for m, n, k, re, im in amn:
            values[int(k) - 1, int(m) - 1, int(n) - 1] = re + 1j * im
        expected = np.stack(
            [_reference_a(source, trials, k, channel) for k in range(8)]
        )
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0)
        actual.append(values)
        matrix, _, _ = read_mmn(manifests[channel].files["mmn"])
        table = _mesh_neighbors(source.header.kpt, 2)
        for k in range(8):
            for slot, row in enumerate(table[k]):
                np.testing.assert_allclose(
                    matrix[k, slot],
                    _reference_m(source, k, int(row[0]), row[1:], channel),
                    atol=1e-12,
                    rtol=0,
                )
    assert np.max(abs(actual[0] - actual[1])) > 1e-3
    reference = np.stack(actual)
    for faulty in (
        reference[::-1],
        np.repeat(reference.mean(axis=0)[None], 2, axis=0),
        np.repeat(reference[:1], 2, axis=0),
    ):
        assert np.max(abs(faulty - reference)) > 1e-3


def test_channel_seed_collision_fails_before_writing(tmp_path):
    source = _source()
    plans = {channel: GenerationPlan((0, 1, 2), 2, (2, 2, 2)) for channel in (0, 1)}
    fileset = AbinitWannierFileSet(source, plans)
    with pytest.raises(ProducerError) as error:
        fileset.write_win(tmp_path, {0: "same", 1: "same"})
    assert error.value.diagnostic == "D2"
    assert not (tmp_path / "same.win").exists()


def test_complete_spin_tuple_and_plan_channel_diagnostics():
    scalar, _ = _mesh_source(2, nband=3)
    with pytest.raises(ProducerError) as error:
        AbinitWannierInputs(
            scalar, GenerationPlan((0, 1, 2), 2, (2, 2, 2), source_channel=1)
        )
    assert error.value.diagnostic == "D1"
    with pytest.raises(ProducerError) as error:
        AbinitWannierInputs(
            scalar, GenerationPlan((0, 1, 2), 2, (2, 2, 2), spinors=True)
        )
    assert error.value.diagnostic == "D6"
    with pytest.raises(ProducerError) as error:
        AbinitWannierFileSet(
            scalar,
            {
                0: GenerationPlan((0, 1, 2), 2, (2, 2, 2)),
                1: GenerationPlan((0, 1, 2), 2, (2, 2, 2)),
            },
        )
    assert error.value.diagnostic == "D1"


def test_real_external_spinor_preprocessing_uses_canonical_plural_projection_block(
    tmp_path,
):
    import subprocess

    executable = (
        Path(__file__).resolve().parents[2]
        / ".tmp/w90-lib-research/build-dev-serial/wannier90.x"
    )
    if not executable.exists():
        pytest.skip("pinned external executable not built")
    source = _source(spinor=True)
    producer = AbinitWannierInputs(
        source,
        GenerationPlan((0, 1, 2), 2, (2, 2, 2), spinors=True),
        _spin_trials(source),
    )
    producer.write_win(tmp_path, "coherent")
    subprocess.run(
        ["rtk", "proxy", str(executable), "-pp", "coherent"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    nnkp = producer.validate_nnkp(tmp_path / "coherent.nnkp")
    assert nnkp.spinor_projections
    np.testing.assert_array_equal(nnkp.kpoints, source.header.kpt)


def test_coherent_plan_cannot_duplicate_physical_energy_rows():
    with pytest.raises(ProducerError) as error:
        GenerationPlan((0, 1, 1, 2), 2, (2, 2, 2), spinors=True)
    assert error.value.diagnostic == "D5"


def test_explicit_spinor_packing_fails_at_real_array_source_before_file_generation():
    from lawaf.interfaces.abinit.source_arrays import ArraySourceError

    source = _source(spinor=True)
    modes = np.ones(source.header.nkpt, dtype=np.int32)
    modes[0] = 2
    metadata = replace(
        source.metadata, header=replace(source.header, istwfk=modes, istwfk_raw=modes)
    )
    with pytest.raises(ArraySourceError) as error:
        CallerArraySource(
            metadata,
            {
                (k, 0): source.coefficients(k, 0, (0, 3))
                for k in range(source.header.nkpt)
            },
            {k: source.kg(k) for k in range(source.header.nkpt)},
        )
    assert error.value.diagnostic == "D3"
