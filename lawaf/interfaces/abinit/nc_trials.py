# SPDX-License-Identifier: BSD-2-Clause
"""Numerical UPF trials in normalized plane waves; no mesh-sized cache."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

from lawaf.interfaces.abinit.basis_datasets import DatasetError, read_upf_projectors
from lawaf.interfaces.abinit.trial_manifest import TrialManifest


def real_harmonics(l, direction):
    """Normalized real harmonics in Wannier90's s/p/d/f slot order."""
    x, y, z = np.asarray(direction).T
    pi = np.pi
    if l == 0:
        return np.full((len(x), 1), np.sqrt(1 / (4 * pi)))
    if l == 1:
        return np.sqrt(3 / (4 * pi)) * np.column_stack((z, x, y))
    if l == 2:
        return np.column_stack(
            (
                np.sqrt(5 / (16 * pi)) * (3 * z * z - 1),
                np.sqrt(15 / (4 * pi)) * x * z,
                np.sqrt(15 / (4 * pi)) * y * z,
                np.sqrt(15 / (16 * pi)) * (x * x - y * y),
                np.sqrt(15 / (4 * pi)) * x * y,
            )
        )
    if l == 3:
        return np.column_stack(
            (
                np.sqrt(7 / (16 * pi)) * z * (5 * z * z - 3),
                np.sqrt(21 / (32 * pi)) * x * (5 * z * z - 1),
                np.sqrt(21 / (32 * pi)) * y * (5 * z * z - 1),
                np.sqrt(105 / (4 * pi)) * x * y * z,
                np.sqrt(105 / (16 * pi)) * z * (x * x - y * y),
                np.sqrt(35 / (32 * pi)) * x * (x * x - 3 * y * y),
                np.sqrt(35 / (32 * pi)) * y * (3 * x * x - y * y),
            )
        )
    raise ValueError(f"unsupported real harmonic angular momentum l={l}; expected 0..3")


class NcTrialBasis:
    """Callable ``(k_index, G_rows)`` for :class:`AbinitWannierInputs`.

    UPF PP_CHI stores u=rR. Its radial transform is integral r*u*j_l dr,
    evaluated with PP_RAB and Simpson integration on the native index grid.
    No hidden normalization, radial resampling or same-l channel averaging.
    Momentum/radial work is capped at 64 G vectors, independently of mesh size.
    """

    def __init__(self, manifest: TrialManifest, header):
        self.manifest = manifest
        self.header = header
        if manifest.trial_definition != "pseudo_radial" or header.usepaw:
            self._fail("trial_definition", "NC pseudo_radial trials and usepaw=0")
        if manifest.units.get("length") != "bohr":
            self._fail("units.length", "bohr for numerical UPF functions")
        convention = manifest.harmonic_convention
        if (
            convention.get("name") != "real_spherical_harmonics_wannier90"
            or convention.get("normalization") != "normalized"
        ):
            self._fail(
                "harmonic_convention", "normalized real_spherical_harmonics_wannier90"
            )
        if any(record.spin_label is not None for record in manifest.trial_records):
            self._fail(
                "spin_label", "scalar spatial trial labels for this NC evaluator"
            )
        lattice = np.asarray(header.rprimd, dtype=float)
        self._reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        self._prefactor = 4 * np.pi / np.sqrt(abs(np.linalg.det(lattice)))
        self._positions = np.asarray([site.position for site in manifest.sites])
        if manifest.position_coordinates == "reduced":
            self._positions = self._positions @ lattice
        self._axes = tuple(np.asarray(site.local_axes) for site in manifest.sites)
        datasets = {
            name: read_upf_projectors(path)
            for name, path in manifest.dataset_paths.items()
        }
        self._channels = {}
        for record in manifest.trial_records:
            key = record.species, record.radial_id
            channel = datasets[record.species].get_channel(record.radial_id)
            if (
                channel.l != record.l
                or not 0 <= record.harmonic_index < 2 * record.l + 1
            ):
                self._fail(
                    "radial/harmonic label",
                    f"actual dataset l={channel.l} and valid harmonic slot",
                )
            if record.l not in range(4):
                self._fail("l", "s/p/d/f angular momentum 0..3")
            self._channels[key] = channel

    def _fail(self, field, expected):
        raise DatasetError(
            f"{self.manifest.path}: field {field!r}: expected {expected}"
        )

    def __call__(self, ik, kg):
        momentum = (np.asarray(kg) + self.header.kpt[ik]) @ self._reciprocal
        result = np.empty(
            (len(momentum), len(self.manifest.trial_records)), dtype=complex
        )
        for start in range(0, len(momentum), 64):
            q = momentum[start : start + 64]
            magnitude = np.linalg.norm(q, axis=1)
            direction = np.divide(
                q,
                magnitude[:, None],
                out=np.zeros_like(q),
                where=magnitude[:, None] != 0,
            )
            radial = {}
            harmonics = {}
            phases = {}
            for column, record in enumerate(self.manifest.trial_records):
                key = record.species, record.radial_id
                if key not in radial:
                    channel = self._channels[key]
                    grid = channel.grid
                    integrand = spherical_jn(channel.l, magnitude[:, None] * grid.r)
                    integrand *= grid.r * channel.values * grid.quadrature_weights
                    radial[key] = (
                        self._prefactor
                        * (-1j) ** channel.l
                        * simpson(integrand, axis=1)
                    )
                angular_key = record.site_index, record.l
                if angular_key not in harmonics:
                    harmonics[angular_key] = real_harmonics(
                        record.l, direction @ self._axes[record.site_index].T
                    )
                if record.site_index not in phases:
                    phases[record.site_index] = np.exp(
                        -1j * (q @ self._positions[record.site_index])
                    )
                result[start : start + len(q), column] = (
                    radial[key]
                    * harmonics[angular_key][:, record.harmonic_index]
                    * phases[record.site_index]
                )
        return result
