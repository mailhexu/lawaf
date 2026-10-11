"""On-demand scalar symmetry transport versus directly transformed fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from lawaf.interfaces.abinit.producer import ProducerError
from lawaf.interfaces.abinit.source_arrays import (
    CallerArraySource,
    CallerMetadata,
    SymmetryMapEntry,
)
from lawaf.interfaces.abinit.source_mesh import MeshSource


def _source(antiunitary=False):
    k = np.array([[0.125, 0.25, 0.375]])
    kg = np.array([[0, 1, -1, 0, 0], [0, 0, 0, 1, -1], [0, 0, 0, 0, 0]], dtype=np.int32)
    cg = np.array(
        [
            [0.1 + 0.2j, 0.3 - 0.1j, -0.2 + 0.4j, 0.17 + 0.31j, 0.21 - 0.43j],
            [0.22 - 0.13j, -0.14 + 0.32j, 0.26 + 0.19j, 0.41 - 0.11j, -0.17 - 0.21j],
        ],
        dtype=complex,
    )[:, :, None]
    cg /= np.linalg.norm(cg, axis=1, keepdims=True)
    operation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    translation = np.array([0.25, 0.125, 0])
    from dataclasses import replace
    from test_abinit_wfk_reader import synthetic_header

    header = replace(
        synthetic_header(k, [5], 2, ecut=2.0),
        rprimd=12 * np.eye(3),
        symrel=operation[None],
        tnons=translation[None],
        istwfk_raw=np.array([1], np.int32),
        md5_pseudos=("",),
        kptopt=0,
        kptrlatt=np.zeros((3, 3), np.int32),
        kptrlatt_orig=np.zeros((3, 3), np.int32),
        lmn_size=np.array([1], np.int32),
    )
    sign = -1 if antiunitary else 1
    shift = np.array([1, -1, 0], dtype=int)
    target = (sign * operation @ k[0] + shift)[None, :]
    meta = CallerMetadata(
        header=header,
        eigenvalues=np.array([[[0.11]], [[0.27]]]),
        eigenvalue_units="hartree",
        length_units="bohr",
        nkpt_mesh=1,
        mesh_kpoints=target,
        symmetry_map=(
            SymmetryMapEntry(
                0, 0, sign * operation, translation, shift, "same", antiunitary
            ),
        ),
    )
    return CallerArraySource(meta, {(0, 0): cg}, {0: kg}), operation, translation


@pytest.mark.parametrize("antiunitary", [False, True])
def test_reconstruction_matches_direct_real_space_image_with_translation_and_folding(
    antiunitary,
):
    source, operation, translation = _source(antiunitary)
    sign = -1 if antiunitary else 1
    shift = np.array([1, -1, 0])
    mesh = (sign * operation @ source.header.kpt[0] + shift)[None, :]
    image = MeshSource(source, mesh, time_reversal=antiunitary)
    kg, weights, cg = image.decode_full_sphere(0, 0, (0, 2))
    xyz = np.array([[0.13, 0.27, 0.31], [0.43, 0.17, 0.21], [0.09, 0.33, 0.57]])
    transformed = xyz @ operation + translation
    original_g = source.kg(0).T + source.header.kpt[0]
    expected = source.coefficients(0, 0, (0, 2))[:, :, 0] @ np.exp(
        2j * np.pi * (original_g @ transformed.T)
    )
    if antiunitary:
        expected = expected.conj()
    actual = cg[:, :, 0] @ np.exp(2j * np.pi * ((kg.T + mesh[0]) @ xyz.T))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(image.eigenvalues(0, 0, (0, 2)), [0.11, 0.27])
    np.testing.assert_array_equal(weights, np.ones(5))


def test_pure_time_reversal_not_used_without_explicit_certificate():
    source, operation, _ = _source(True)
    target = source.metadata.mesh_kpoints
    with pytest.raises(ProducerError):
        MeshSource(source, target, time_reversal=False)


@pytest.mark.parametrize("antiunitary", [False, True])
def test_coherent_rotation_and_time_reversal_match_defining_fields(antiunitary):
    from dataclasses import replace

    scalar, operation, translation = _source(antiunitary)
    h = replace(scalar.header, nspinor=2, nspden=4)
    first = scalar.coefficients(0, 0, (0, 2))[:, :, 0]
    second = first.conj() * (0.3 + 0.7j) + np.roll(first, 1, axis=1) * (0.2 - 0.4j)
    states = np.ascontiguousarray(np.stack((first, second), axis=-1))
    source = CallerArraySource(
        replace(scalar.metadata, header=h), {(0, 0): states}, {0: scalar.kg(0)}
    )
    image = MeshSource(source, source.metadata.mesh_kpoints, time_reversal=antiunitary)
    kg, _, cg = image.decode_full_sphere(0, 0, (0, 2))
    points = np.array([[0.13, 0.27, 0.31], [0.43, 0.17, 0.21], [0.09, 0.33, 0.57]])
    phase = np.exp(
        2j
        * np.pi
        * ((source.kg(0).T + h.kpt[0]) @ (points @ operation + translation).T)
    )
    expected = np.einsum("bgs,gr->brs", states, phase)
    if antiunitary:
        expected = np.stack(
            (expected[:, :, 1].conj(), -expected[:, :, 0].conj()), axis=-1
        )
    # Defining axial-spin action for a positive90-degree Cartesian z rotation.
    spin = np.diag([np.exp(-0.25j * np.pi), np.exp(0.25j * np.pi)])
    expected = expected @ spin.T
    actual = np.einsum(
        "bgs,gr->brs",
        cg,
        np.exp(2j * np.pi * ((kg.T + image.header.kpt[0]) @ points.T)),
    )
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(
        image.eigenvalues(0, 0, (0, 2)), scalar.eigenvalues(0, 0, (0, 2))
    )


def test_explicit_afm_channel_map_is_unitary_spatial_transport_not_conjugation():
    from dataclasses import replace
    from lawaf.interfaces.abinit.producer import MeshMapping

    scalar, operation, translation = _source(False)
    identity = np.eye(3, dtype=np.int32)
    mesh = scalar.metadata.mesh_kpoints
    stored_k = np.vstack((scalar.header.kpt, mesh))
    h = replace(
        scalar.header,
        nkpt=2,
        nspden=2,
        nsym=2,
        symrel=np.array([identity, operation]),
        tnons=np.array([np.zeros(3), translation]),
        symafm=np.array([1, -1], dtype=np.int32),
        kpt=stored_k,
        nband=np.array([2, 2], np.int32),
        bantot=4,
        npwarr=np.array([5, 5], np.int32),
        istwfk=np.ones(2, np.int32),
        istwfk_raw=np.ones(2, np.int32),
        occ=np.ones(4),
        wtk=np.full(2, 0.5),
    )
    first = scalar.coefficients(0, 0, (0, 2))
    second = np.ascontiguousarray(
        first
        * np.exp(1j * (0.27 * scalar.kg(0)[0] + 0.44 * scalar.kg(0)[1]))[None, :, None]
    )
    up = MeshMapping(0, 1, identity, np.zeros(3), np.zeros(3, dtype=int), False)
    entry = scalar.symmetry_map[0]
    down = MeshMapping(
        0, 0, entry.operation, entry.translation, entry.reciprocal_shift, False
    )
    declared = SymmetryMapEntry(
        0, 1, identity, np.zeros(3), np.zeros(3, dtype=int), "same", False
    )
    metadata = replace(
        scalar.metadata,
        header=h,
        eigenvalues=np.array([[[0.11], [0.41]], [[0.27], [0.67]]]),
        symmetry_map=(declared,),
    )
    source = CallerArraySource(
        metadata, {(0, 0): first, (1, 0): second}, {0: scalar.kg(0), 1: scalar.kg(0)}
    )
    with pytest.raises(ProducerError) as error:
        MeshSource(source, mesh, time_reversal=False)
    assert error.value.diagnostic == "D4"
    image = MeshSource(
        source, mesh, time_reversal=False, channel_mappings={0: (up,), 1: (down,)}
    )
    kg, _, coefficients = image.decode_full_sphere(0, 1, (0, 2))
    points = np.array([[0.13, 0.27, 0.31], [0.43, 0.17, 0.21]])
    expected = first[:, :, 0] @ np.exp(
        2j
        * np.pi
        * ((source.kg(0).T + h.kpt[0]) @ (points @ operation + translation).T)
    )
    actual = coefficients[:, :, 0] @ np.exp(2j * np.pi * ((kg.T + mesh[0]) @ points.T))
    wrong_up = second[:, :, 0] @ np.exp(
        2j * np.pi * ((source.kg(1).T + h.kpt[1]) @ points.T)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert np.max(abs(actual - expected.conj())) > 1e-2
    assert np.max(abs(actual - wrong_up)) > 1e-2
    np.testing.assert_array_equal(image.eigenvalues(0, 0, (0, 2)), [0.41, 0.67])
    np.testing.assert_array_equal(image.eigenvalues(0, 1, (0, 2)), [0.11, 0.27])


@pytest.mark.parametrize("antiunitary", [False, True])
@pytest.mark.parametrize("exchange", [False, True])
@pytest.mark.parametrize("explicit_maps", [False, True])
def test_collinear_images_route_actual_channel_fields_energies_and_occupations(
    antiunitary, exchange, explicit_maps
):
    from dataclasses import replace
    from lawaf.interfaces.abinit.producer import MeshMapping

    scalar, rotation, translation = _source(antiunitary)
    header = replace(
        scalar.header, nsppol=2, nspden=2,
        nband=np.array([2, 2], np.int32), bantot=4,
        occ=np.array([1.0, 0.8, 0.7, 0.6]),
        symafm=np.array([-1 if exchange else 1], np.int32),
    )
    up = scalar.coefficients(0, 0, (0, 2))
    down = np.ascontiguousarray(up.conj() * (0.37 + 0.91j))
    label = "TR" if exchange and antiunitary else ("flip" if exchange else "same")
    entry = replace(scalar.symmetry_map[0], spin_label=label)
    energies = np.array([[[0.11, 0.41]], [[0.27, 0.67]]])
    source = CallerArraySource(
        replace(scalar.metadata, header=header, eigenvalues=energies, symmetry_map=(entry,)),
        {(0, 0): up, (0, 1): down}, {0: scalar.kg(0)},
    )
    maps = None
    if explicit_maps:
        maps = {
            channel: (MeshMapping(
                0, 0, entry.operation, translation, entry.reciprocal_shift,
                antiunitary, source_channel=channel ^ int(exchange),
            ),)
            for channel in (0, 1)
        }
    image = MeshSource(source, source.metadata.mesh_kpoints, time_reversal=antiunitary, channel_mappings=maps)
    points = np.array([[0.13, 0.27, 0.31], [0.43, 0.17, 0.21]])
    phase = np.exp(2j * np.pi * ((source.kg(0).T + header.kpt[0]) @ (points @ rotation + translation).T))
    for channel in (0, 1):
        original = channel ^ int(exchange)
        expected = (up if original == 0 else down)[:, :, 0] @ phase
        if antiunitary:
            expected = expected.conj()
        kg, _, states = image.decode_full_sphere(0, channel, (0, 2))
        actual = states[:, :, 0] @ np.exp(2j * np.pi * ((kg.T + image.header.kpt[0]) @ points.T))
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_array_equal(image.eigenvalues(0, channel, (0, 2)), energies[:, 0, original])
        np.testing.assert_array_equal(image.header.occ[2*channel:2*channel+2], header.occ[2*original:2*original+2])
        assert image.stored_request(0, channel) == (0, original)
        with pytest.raises(ValueError):
            image.channel_mappings[channel][0].operation[0, 0] = 7
