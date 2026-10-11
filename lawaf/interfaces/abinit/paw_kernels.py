# SPDX-License-Identifier: BSD-2-Clause
"""PAW XML operators in a declared normalized real Wannier90 basis.

Raw XML functions are R(r). Projector Fourier coefficients, DeltaS and
finite-b Q share this radial measure; no partial-wave-to-cprj conversion.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.special import lpmv, spherical_jn

from lawaf.interfaces.abinit.basis_datasets import DatasetError, PawXmlDataset
from lawaf.interfaces.abinit.nc_trials import real_harmonics

HARMONIC_CONVENTION = "real_spherical_harmonics_wannier90"
# m_paw_atom_solve machine_zero (m_paw_atom_solve.F90: machine_zero =
# machine_precision**5): the overint trailing trim drops only machine-zero
# integrand samples of the tabulated radial functions.
_INTEGRAND_MACHINE_ZERO = float(np.finfo(np.float64).eps) ** 5


def _overint_weights(count):
    """Unit-spacing weights of the m_paw_atom_solve overint integrator.

    Composite Simpson panels terminate exactly at the last node; an even
    count applies the first trapezoid and starts the Simpson panels at the
    second node (the log-grid icorr=-1 path of the dataset generator).
    """
    weights = np.zeros(count)
    if count % 2:
        weights[0 : count - 2 : 2] += 1 / 3
        weights[1 : count - 1 : 2] += 4 / 3
        weights[2:count:2] += 1 / 3
    else:
        weights[0] = weights[1] = 1 / 2
        weights[1 : count - 2 : 2] += 1 / 3
        weights[2 : count - 1 : 2] += 4 / 3
        weights[3:count:2] += 1 / 3
    return weights


def _overint(integrand):
    """m_paw_atom_solve overint for one radial integrand (Jacobian inside).

    Trailing machine-zero samples are trimmed per integrand with the first
    following zero retained, then the trimmed support carries the overint
    weights.
    """
    significant = np.flatnonzero(np.abs(integrand) >= _INTEGRAND_MACHINE_ZERO)
    if not significant.size:
        return 0.0
    count = min(integrand.shape[0], int(significant[-1]) + 2)
    return float(_overint_weights(count) @ integrand[:count])


def _native_simpson_weights(count):
    """Unit-spacing weights of the m_pawrad simp_gen quadrature.

    Simpson anchored at the last node; when the count is even the first
    three nodes carry the Cartwright correction (5/12, 2/3, -1/12), and a
    two-node support is the plain trapezoid.
    """
    weights = np.zeros(count)
    if count % 2:
        weights[0 : count - 2 : 2] += 1 / 3
        weights[1 : count - 1 : 2] += 4 / 3
        weights[2:count:2] += 1 / 3
    elif count == 2:
        weights[:] = 1 / 2
    else:
        weights[0 : count - 3 : 2] += 1 / 3
        weights[1 : count - 2 : 2] += 4 / 3
        weights[2 : count - 1 : 2] += 1 / 3
        weights[-1] += 5 / 12
        weights[-2] += 2 / 3
        weights[-3] -= 1 / 12
    return weights


def _real_lm(l, m, direction):
    """Orthonormal CS expansion basis; signs cancel in the Gaunt expansion."""
    x, y, z = np.asarray(direction).T
    order = abs(m)
    normalization = np.sqrt(
        (2 * l + 1) / (4 * np.pi) * factorial(l - order) / factorial(l + order)
    )
    value = normalization * lpmv(order, l, np.clip(z, -1, 1))
    if m:
        phi = np.arctan2(y, x)
        value *= np.sqrt(2) * (np.cos(order * phi) if m > 0 else np.sin(order * phi))
    return value


class PawSpecies:
    """Complete one-species dual-projector and finite-b augmentation operator.

    ``projector_kets`` returns Fourier kets without atomic phase or volume
    factor. Callers apply those once. ``form_factor`` returns Q in the same
    column basis, without the external atom phase. No k-dependent cache.
    """

    def __init__(self, dataset: PawXmlDataset):
        self.dataset = dataset
        if dataset.radial_representation != "R":
            raise DatasetError(f"{dataset.path}: PAW operators require raw XML R(r)")
        self.basis = tuple(
            (radial, slot)
            for radial, channel in enumerate(dataset.channels)
            for slot in range(2 * channel.l + 1)
        )
        self.radial_index = np.array([radial for radial, _ in self.basis])
        self.lmax = max(channel.l for channel in dataset.channels)
        if not 0 <= self.lmax <= 3:
            raise DatasetError(
                f"{dataset.path}: declared real s/p/d/f operators require l<=3"
            )
        r = dataset.grid.r
        jacobian = dataset.grid.d * (r + dataset.grid.a)
        self._projector_measure = np.stack(
            [r * r * channel.dual_projector * jacobian for channel in dataset.channels]
        )
        self.duality_residual = 0.0
        # Dataset-generator overint uses per-integrand support and includes
        # the radial Jacobian and r^2 explicitly.

        for l in range(self.lmax + 1):
            channels = [channel for channel in dataset.channels if channel.l == l]
            if not channels:
                continue
            matrix = np.array(
                [
                    [
                        _overint(
                            left.dual_projector
                            * right.pseudo_partial_wave
                            * r
                            * r
                            * jacobian
                        )
                        for right in channels
                    ]
                    for left in channels
                ]
            )
            self.duality_residual = max(
                self.duality_residual,
                float(np.max(abs(matrix - np.eye(len(channels))))),
            )
        if self.duality_residual > 1e-8:
            raise DatasetError(
                f"{dataset.path}: dual-projector closure residual {self.duality_residual:.3e} exceeds 1e-8"
            )
        # Refinement checks the declared spline/grid quadrature, not the
        # physical quality of the dataset or a source SCF calculation.
        indices = np.arange(len(r))
        self._smooth_spline = CubicSpline(indices, np.stack(
            [channel.pseudo_partial_wave for channel in dataset.channels], axis=1
        ), axis=0)
        self._difference_spline = CubicSpline(indices, np.stack(
            [channel.ae_partial_wave - channel.pseudo_partial_wave for channel in dataset.channels], axis=1
        ), axis=0)
        self._r, self._density_measure = self._radial_density(4)
        self.last_quadrature_residual = None
        self.last_quadrature_refinement = None
        costheta, weights = np.polynomial.legendre.leggauss(32)
        phi = np.arange(64) * 2 * np.pi / 64
        ct, ph = np.meshgrid(costheta, phi, indexing="ij")
        directions = np.column_stack(
            (
                np.sqrt(1 - ct.ravel() ** 2) * np.cos(ph.ravel()),
                np.sqrt(1 - ct.ravel() ** 2) * np.sin(ph.ravel()),
                ct.ravel(),
            )
        )
        angular_weights = np.repeat(weights, 64) * 2 * np.pi / 64
        unique = [(l, slot) for l in range(self.lmax + 1) for slot in range(2 * l + 1)]
        harmonics = np.column_stack(
            [real_harmonics(l, directions)[:, slot] for l, slot in unique]
        )
        self._harmonic_index = np.array(
            [
                unique.index((dataset.channels[radial].l, slot))
                for radial, slot in self.basis
            ]
        )
        self._expansion = tuple(
            (l, m) for l in range(2 * self.lmax + 1) for m in range(-l, l + 1)
        )
        expansion = np.column_stack(
            [_real_lm(l, m, directions) for l, m in self._expansion]
        )
        products = (harmonics[:, :, None] * harmonics[:, None, :]).reshape(
            len(directions), -1
        )
        self._gaunt = (products.T @ (expansion * angular_weights[:, None])).reshape(
            len(unique), len(unique), len(self._expansion)
        )
        self.delta_s = self.form_factor(np.zeros(3)).real

    def projector_kets(self, momentum):
        """Bounded R-grid Fourier evaluation; output (G, local_projector)."""
        q = np.asarray(momentum, dtype=float)
        output = np.empty((len(q), len(self.basis)), complex)
        for start in range(0, len(q), 64):
            tile = q[start : start + 64]
            magnitude = np.linalg.norm(tile, axis=1)
            direction = np.divide(
                tile,
                magnitude[:, None],
                out=np.zeros_like(tile),
                where=magnitude[:, None] != 0,
            )
            radial = {}
            angular = {}
            for column, (index, slot) in enumerate(self.basis):
                channel = self.dataset.channels[index]
                if index not in radial:
                    radial[index] = (
                        4
                        * np.pi
                        * (-1j) ** channel.l
                        * simpson(
                            spherical_jn(
                                channel.l, magnitude[:, None] * self.dataset.grid.r
                            )
                            * self._projector_measure[index],
                            axis=1,
                        )
                    )
                if channel.l not in angular:
                    angular[channel.l] = real_harmonics(channel.l, direction)
                output[start : start + len(tile), column] = (
                    radial[index] * angular[channel.l][:, slot]
                )
        return output

    def _radial_density(self, refinement):
        grid = self.dataset.grid
        stop = len(grid.r) - 1
        indices = np.linspace(0, stop, refinement * stop + 1)
        r = grid.a * np.expm1(grid.d * indices)
        smooth = self._smooth_spline(indices)
        difference = self._difference_spline(indices)
        difference[r > self.dataset.rc] = 0.0
        density = (
            difference[:, :, None] * smooth[:, None, :]
            + smooth[:, :, None] * difference[:, None, :]
            + difference[:, :, None] * difference[:, None, :]
        )
        return r, density * (r * r * grid.d * (r + grid.a))[:, None, None]

    def _form_factor_at_refinement(self, magnitude, direction, refinement):
        r, measure = (self._r, self._density_measure) if refinement == 4 else self._radial_density(refinement)
        radial = [
            simpson(
                measure * spherical_jn(l, magnitude * r)[:, None, None],
                dx=1.0 / refinement, axis=0,
            )
            for l in range(2 * self.lmax + 1)
        ]
        result = np.zeros((len(self.basis), len(self.basis)), complex)
        i, j = self.radial_index[:, None], self.radial_index[None, :]
        hi, hj = self._harmonic_index[:, None], self._harmonic_index[None, :]
        for expansion, (l, m) in enumerate(self._expansion):
            factor = 4 * np.pi * (-1j) ** l * _real_lm(l, m, direction)[0]
            result += factor * self._gaunt[hi, hj, expansion] * radial[l][i, j]
        return result

    def form_factor(self, b):
        """Admit Q(b) only after an independent finer-grid integration agrees.

        This checks integration of the fixed declared radial spline. It is
        not a certificate of partial-wave closure or dataset convergence.
        Only scalar last-call evidence is retained; no state/neighbor cache.
        """
        vector = np.asarray(b, dtype=float)
        magnitude = float(np.linalg.norm(vector))
        direction = (vector / magnitude if magnitude else np.array([0.0, 0.0, 1.0]))[None, :]
        previous = self._form_factor_at_refinement(magnitude, direction, 4)
        for refinement in (8, 16, 32, 64):
            current = self._form_factor_at_refinement(magnitude, direction, refinement)
            residual = float(np.max(abs(current - previous))) / max(1.0, float(np.max(abs(current))))
            self.last_quadrature_residual = residual
            self.last_quadrature_refinement = refinement
            if np.isfinite(residual) and residual <= 1e-8:
                return current
            previous = current
        raise DatasetError(
            f"[D5] {self.dataset.path}: finite-b radial quadrature b={vector.tolist()} "
            f"residual {residual:.3e} exceeds 1e-8 after refinement{refinement}"
        )
