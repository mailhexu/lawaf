# SPDX-License-Identifier: BSD-2-Clause
"""Matched PAW operator data and genuine supplied-or-computed cprj access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from lawaf.interfaces.abinit.basis_datasets import DatasetError, read_paw_xml_dataset
from lawaf.interfaces.abinit.paw_coefficients import (
    PawCoefficientError,
    _validate_meta,
    require_dataset_identity,
    require_dualprojector,
)
from lawaf.interfaces.abinit.paw_kernels import HARMONIC_CONVENTION, PawSpecies

_SLOT_M = {0: (0,), 1: (0, 1, -1), 2: (0, 1, -1, 2, -2), 3: (0, 1, -1, -2, 2, 3, -3)}


class PawCoefficientProvider:
    """Own matched species operators; borrow source and supplied record.

    ``band_block`` returns (physical_band, spinor_component, total_projector).
    ``compute_calls`` counts actual Fourier projector contractions, not a stub.
    Supplied records never enter that computation, but cannot waive the XML
    operators or their source identity/order/normalization checks.
    """

    def __init__(self, source, dataset_paths, *, supplied_record=None):
        self.source = source
        self.header = source.header
        self.record = supplied_record
        self.compute_calls = 0
        self._transport_supplied = False
        h = self.header
        if not h.usepaw:
            raise PawCoefficientError(
                "D6",
                source.path,
                "usepaw",
                "PAW classification must come from the source header",
            )
        paths = tuple(Path(path) for path in dataset_paths)
        if len(paths) != h.ntypat:
            raise DatasetError(
                f"{source.path}: matched PAW XML references required for all {h.ntypat} source species"
            )
        datasets = tuple(read_paw_xml_dataset(path) for path in paths)
        identities = tuple(hashlib.md5(path.read_bytes()).hexdigest() for path in paths)
        expected = tuple(str(value).strip() for value in h.md5_pseudos)
        if identities != expected:
            raise PawCoefficientError(
                "D4",
                source.path,
                "dataset_md5",
                f"actual XML identities{identities} must match source{expected}",
            )
        self.kernels = tuple(PawSpecies(dataset) for dataset in datasets)
        self.offsets = [0]
        self.projector_labels = []
        for atom, species_index in enumerate(h.typat):
            kernel = self.kernels[int(species_index) - 1]
            expected_count = int(h.lmn_size[int(species_index) - 1])
            if len(kernel.basis) != expected_count:
                raise PawCoefficientError(
                    "D3",
                    source.path,
                    f"atom{atom} projector count",
                    f"dataset{len(kernel.basis)} != source{expected_count}",
                )
            for radial, slot in kernel.basis:
                l = kernel.dataset.channels[radial].l
                self.projector_labels.append((atom, radial, l, _SLOT_M[l][slot]))
            self.offsets.append(self.offsets[-1] + len(kernel.basis))
        self.projector_labels = tuple(self.projector_labels)
        self.offsets = tuple(self.offsets)
        self._reciprocal = 2 * np.pi * np.linalg.inv(h.rprimd).T
        self._volume_factor = 1 / np.sqrt(abs(np.linalg.det(h.rprimd)))
        self._positions = np.asarray(h.xred) @ h.rprimd
        if self.record is not None:
            require_dualprojector(self.record)
            require_dataset_identity(self.record)
            anchor = h
            if hasattr(source, "stored_source"):
                if self.record.meta.state_frame == "expanded":
                    if (
                        self.record.meta.symmetry_map_reference
                        != source.symmetry_map_reference
                    ):
                        raise PawCoefficientError(
                            "D6",
                            source.path,
                            "symmetry_map_reference",
                            "expanded record must match the actual consumer map declaration",
                        )
                else:
                    anchor = source.stored_source.header
                    self._transport_supplied = True
            _validate_meta(
                self.record.meta,
                anchor,
                self.record.shape,
                source.path,
                symmetry_map_reference=self.record.symmetry_map_reference,
            )
            if self.record.meta.harmonic_convention != HARMONIC_CONVENTION:
                raise PawCoefficientError(
                    "D8",
                    source.path,
                    "harmonic_convention",
                    f"matched verified basis {HARMONIC_CONVENTION!r} required",
                )
            if self.record.meta.projector_labels != self.projector_labels:
                raise PawCoefficientError(
                    "D8",
                    source.path,
                    "projector_labels",
                    "actual atom/radial/angular column identities must match the XML operator basis",
                )

    def band_block(self, k, spin, band_range):
        """Read declared slabs or actually compute absent dual overlaps."""
        h = self.header
        ik, channel = int(k), int(spin)
        b0, b1 = map(int, band_range)
        if not 0 <= ik < h.nkpt or not 0 <= channel < h.nsppol:
            raise PawCoefficientError(
                "D6",
                self.source.path,
                "state index",
                f"k={k}, source channel={spin} outside source",
            )
        if not 0 <= b0 < b1 <= int(h.nband[ik + channel * h.nkpt]):
            raise PawCoefficientError(
                "D5",
                self.source.path,
                "physical band range",
                f"requested{band_range} outside source coverage",
            )
        if self.record is not None:
            meta = self.record.meta
            stored_k, stored_channel = (
                self.source.stored_request(ik, channel)
                if self._transport_supplied
                else (ik, channel)
            )
            if stored_channel != meta.source_channel:
                raise PawCoefficientError(
                    "D6",
                    self.source.path,
                    "source_channel",
                    f"supplied channel{meta.source_channel} != requested stored channel{stored_channel}",
                )
            if b0 < meta.band_range[0] or b1 > meta.band_range[1]:
                raise PawCoefficientError(
                    "D5",
                    self.source.path,
                    "physical band range",
                    f"supplied{meta.band_range} does not cover requested{band_range}; no silent recomputation",
                )
            result = np.empty((b1 - b0, h.nspinor, self.offsets[-1]), complex)
            for row, band in enumerate(range(b0, b1)):
                result[row] = self.record.band_slab(stored_k, band)
            return (
                self.source.transport_projectors(
                    ik, channel, result, self.projector_labels
                )
                if self._transport_supplied
                else result
            )
        kg, _, coefficients = self.source.decode_full_sphere(ik, channel, (b0, b1))
        momentum = (kg.T + h.kpt[ik]) @ self._reciprocal
        result = np.empty((b1 - b0, h.nspinor, self.offsets[-1]), complex)
        self.compute_calls += 1
        for atom, species_index in enumerate(h.typat):
            kernel = self.kernels[int(species_index) - 1]
            projectors = kernel.projector_kets(momentum)
            projectors *= (
                self._volume_factor * np.exp(-1j * (momentum @ self._positions[atom]))
            )[:, None]
            start, stop = self.offsets[atom : atom + 2]
            for component in range(h.nspinor):
                result[:, component, start:stop] = (
                    coefficients[:, :, component] @ projectors.conj()
                )
        return result
