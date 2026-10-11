"""PAW operator tests use independently integrated normalized Gaussian fields."""

from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawaf.interfaces.abinit.basis_datasets import PawChannel, PawGrid, PawXmlDataset
from lawaf.interfaces.abinit.paw_kernels import PawSpecies


def _species():
    a, d = 0.1, 0.004
    r = a * np.expm1(d * np.arange(1801))
    s = 2 / np.pi**0.25 * np.exp(-r * r / 2)
    p = np.sqrt(8 / (3 * np.sqrt(np.pi))) * r * np.exp(-r * r / 2)
    channels = tuple(
        PawChannel(str(l), 1, l, 8.0, 0.0, scale * radial, radial, radial)
        for l, scale, radial in ((0, 1.2, s), (1, 0.8, p))
    )
    return PawSpecies(
        PawXmlDataset(
            "<analytic Gaussian PAW>",
            "H",
            "0.7",
            "R",
            8.0,
            PawGrid(r, "r=a*(exp(d*i)-1)", a, d, "g"),
            channels,
        )
    )


def _cartesian_reference(b):
    nodes, weights = hermgauss(32)
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    measure = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1)
    # Hermite weight already supplies exp(-r²) in the product of two fields.
    fields = np.pi ** (-0.75) * np.column_stack(
        (np.ones(len(points)), np.sqrt(2) * points[:, [2, 0, 1]])
    )
    phase = np.exp(-1j * (points @ b))
    density_scale = np.array([1.2, 0.8, 0.8, 0.8])
    return (density_scale[:, None] * density_scale[None, :] - 1) * (
        fields.T @ ((measure * phase)[:, None] * fields)
    )


def test_finite_b_metric_matches_defining_cartesian_integral_and_reverse_pair():
    species = _species()
    for b in (np.zeros(3), np.array([0.8, 0.3, -0.2]), np.array([1.7, -0.9, 0.6])):
        value = species.form_factor(b)
        np.testing.assert_allclose(value, _cartesian_reference(b), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(
            value, species.form_factor(-b).conj().T, atol=1e-12, rtol=0
        )
    np.testing.assert_allclose(
        species.delta_s, np.diag([0.44, -0.36, -0.36, -0.36]), atol=1e-8, rtol=0
    )
    odd = species.form_factor(np.array([0.0, 0.0, 0.9]))[0, 1]
    assert abs(odd.imag) > 1e-3 and abs(odd.real) < 1e-12


def test_projector_fourier_kets_have_correct_cartesian_p_phase_and_slot_order():
    species = _species()
    q = np.array([[0.0, 0.0, 0.0], [0.17, -0.23, 0.31], [0.8, 0.3, -0.2]])
    expected_s = (
        np.pi ** (-0.75) * (2 * np.pi) ** 1.5 * np.exp(-np.sum(q * q, axis=1) / 2)
    )
    expected = np.column_stack(
        (
            expected_s,
            -1j * np.sqrt(2) * q[:, 2] * expected_s,
            -1j * np.sqrt(2) * q[:, 0] * expected_s,
            -1j * np.sqrt(2) * q[:, 1] * expected_s,
        )
    )
    np.testing.assert_allclose(
        species.projector_kets(q), expected, atol=1e-8, rtol=1e-8
    )


def test_unresolved_augmentation_cutoff_is_not_admitted_by_projector_duality():
    from dataclasses import replace
    import pytest
    from lawaf.interfaces.abinit.basis_datasets import DatasetError

    species = _species()
    # Projector/pseudo duality is unchanged, but nonzero AE-pseudo density
    # is cut at a radius between declared grid points. It is unresolved
    # at the bounded quadrature refinements, so metric admission must fail.
    unresolved = replace(species.dataset, rc=1.234567)
    with pytest.raises(DatasetError, match="D5"):
        PawSpecies(unresolved)
