# SPDX-License-Identifier: BSD-2-Clause
"""On-demand NC/PAW mesh transport; borrows, never closes its source."""

from dataclasses import replace

import numpy as np

from lawaf.interfaces.abinit.producer import MeshMapping, ProducerError, map_mesh_to_ibz
from lawaf.interfaces.abinit.source_wfk import decode_packed_sphere


class MeshSource:
    """Source seam over an explicit mesh, without a full-mesh state cache.

    ``time_reversal`` is a required caller certificate for antiunitary images.
    File callers obtain it from the actual source symmetry policy; caller-array
    inputs must declare it, never inherit a fabricated nonmagnetic default.
    Coherent transport applies spatial phases, i*sigma_y K, then SU(2).
    """

    def __init__(
        self,
        source,
        mesh_kpoints,
        *,
        time_reversal,
        channel_mappings=None,
        symmetry_map_reference=None,
    ):
        self._source = source
        self.symmetry_map_reference = symmetry_map_reference
        self._projector_rotations = {}
        self.path = source.path
        self.format = f"{source.format}:symmetry-expanded"
        h = source.header
        representation = (h.nsppol, h.nspinor, h.nspden)
        afm = representation == (1, 1, 2)
        if representation not in ((1, 1, 1), (2, 1, 2), (1, 2, 4), (1, 1, 2)):
            raise ProducerError(
                "D4",
                self.path,
                "spin representation",
                f"unsupported complete tuple{representation}",
            )
        if h.nspinor == 2 and np.any(np.asarray(h.istwfk) != 1):
            raise ProducerError(
                "D3",
                self.path,
                "spinor storage",
                "native coherent states require full storage1",
            )
        if afm and (channel_mappings is None or set(channel_mappings) != {0, 1}):
            raise ProducerError(
                "D4",
                self.path,
                "AFM channel/symmetry map",
                "explicit up/down image tables required",
            )
        if not afm and h.nsppol != 2 and channel_mappings is not None:
            raise ProducerError(
                "D4", self.path, "channel_mappings",
                "explicit channel maps require collinear or AFM states",
            )
        self._source_channels = (0, 0) if afm else tuple(range(h.nsppol))
        self._spin_rotations = {}
        mesh = np.array(mesh_kpoints, dtype=float, copy=True)
        declared = getattr(source, "symmetry_map", None)
        if afm:
            self.channel_mappings = tuple(
                self._validate_channel_maps(
                    channel_mappings[channel], mesh, channel, time_reversal
                )
                for channel in (0, 1)
            )
        elif channel_mappings is not None:
            if set(channel_mappings) != {0, 1}:
                raise ProducerError("D4", self.path, "channel_mappings", "up/down maps required")
            self.channel_mappings = tuple(
                self._validate_channel_maps(channel_mappings[channel], mesh, channel, time_reversal, collinear=True)
                for channel in (0, 1)
            )
        elif declared is not None:
            targets = source.metadata.mesh_kpoints
            if mesh.shape != targets.shape or not np.allclose(
                mesh, targets, atol=1e-10, rtol=0
            ):
                raise ProducerError(
                    "D4",
                    self.path,
                    "declared target mesh",
                    "requested mesh/order must match the explicit caller mapping",
                )
            mappings = [[] for _ in self._source_channels]
            for entry in declared:
                if entry.time_reversal and not time_reversal:
                    raise ProducerError(
                        "D4",
                        self.path,
                        "time_reversal",
                        "declared antiunitary image needs an explicit source certificate",
                    )
                if h.nsppol == 2 and entry.spin_label == "TR" and not entry.time_reversal:
                    raise ProducerError("D4", self.path, "spin_label", "physical TR requires conjugation and spin exchange")
                if entry.spin_label not in (("same", "TR", "flip") if h.nsppol == 2 else ("same", "TR")):
                    raise ProducerError(
                        "D4", self.path, "spin_label",
                        "same/TR, or explicit flip for a two-channel source",
                    )
                for channel in self._source_channels:
                    mappings[channel].append(
                        MeshMapping(
                            entry.mesh_point, entry.ibz_point, entry.operation,
                            entry.translation, entry.reciprocal_shift, entry.time_reversal,
                            source_channel=(
                                (channel ^ int(entry.spin_label in ("flip", "TR")))
                                if h.nsppol == 2 else channel
                            ),
                        )
                    )
            self.channel_mappings = tuple(tuple(rows) for rows in mappings)
        else:
            symrel, tnons, symafm = source.require_symmetry()
            allowed = np.ones(len(symafm), bool) if h.nsppol == 2 else np.asarray(symafm) == 1
            self.channel_mappings = tuple(
                tuple(map_mesh_to_ibz(
                    h.kpt, mesh, np.asarray(symrel)[allowed], np.asarray(tnons)[allowed],
                    time_reversal=time_reversal,
                    symafm=np.asarray(symafm)[allowed] if h.nsppol == 2 else None,
                    target_channel=channel,
                ))
                for channel in self._source_channels
            )
        counts = {}
        for mp in self.channel_mappings[0]:
            if mp.ibz_point not in counts:
                counts[mp.ibz_point] = self._full_g(mp.ibz_point).shape[1]
        nband = np.array(
            [
                h.nband[mp.ibz_point + mp.source_channel * h.nkpt]
                for stored, maps in zip(self._source_channels, self.channel_mappings)
                for mp in maps
            ],
            dtype=np.int32,
        )
        offsets = np.concatenate(([0], np.cumsum(h.nband)))
        occ = np.asarray(h.occ).ravel(order="F")
        occupations = np.concatenate(
            [
                occ[
                    offsets[mp.ibz_point + mp.source_channel * h.nkpt] : offsets[
                        mp.ibz_point + mp.source_channel * h.nkpt + 1
                    ]
                ]
                for stored, maps in zip(self._source_channels, self.channel_mappings)
                for mp in maps
            ]
        )
        for maps in self.channel_mappings[1:]:
            for point, mp in enumerate(maps):
                count = self._full_g(mp.ibz_point).shape[1]
                if count != counts[self.channel_mappings[0][point].ibz_point]:
                    raise ProducerError(
                        "D4",
                        self.path,
                        "channel G spheres",
                        "same physical target sphere required",
                    )
        full_modes = np.ones(len(mesh), dtype=np.int32)
        mesh.setflags(write=False)
        self.header = replace(
            h,
            kpt=mesh,
            nkpt=len(mesh),
            nsppol=len(self._source_channels),
            nband=nband,
            bantot=int(nband.sum()),
            npwarr=np.array([counts[mp.ibz_point] for mp in self.channel_mappings[0]]),
            istwfk=full_modes,
            istwfk_raw=full_modes,
            occ=occupations,
            wtk=np.full(len(mesh), 1 / len(mesh)),
            kptopt=3,
        )
        for field in ("nband", "npwarr", "istwfk", "istwfk_raw", "occ", "wtk"):
            getattr(self.header, field).flags.writeable = False

    def _validate_channel_maps(self, mappings, mesh, channel, time_reversal, *, collinear=False):
        h = self._source.header
        if len(mappings) != len(mesh) or tuple(
            mp.mesh_point for mp in mappings
        ) != tuple(range(len(mesh))):
            raise ProducerError(
                "D4",
                self.path,
                "AFM image coverage",
                "every target exactly once in target order",
            )
        admitted = []
        for mp in mappings:
            source_channel = mp.source_channel if collinear else 0
            if collinear and (
                isinstance(source_channel, (bool, np.bool_))
                or not isinstance(source_channel, (int, np.integer))
                or source_channel not in (0, 1)
            ):
                raise ProducerError("D4", self.path, "source_channel", "explicit original channel0/1")
            flags = (-1 if channel != source_channel else 1) if collinear else (1 if channel == 0 else -1)
            allowed = np.flatnonzero(np.asarray(h.symafm) == flags)
            if not 0 <= mp.ibz_point < h.nkpt or (
                mp.time_reversal and not time_reversal
            ):
                raise ProducerError(
                    "D4",
                    self.path,
                    "AFM source/time reversal",
                    "valid stored index and certified antiunitary policy",
                )
            operation = np.asarray(mp.operation) * (-1 if mp.time_reversal else 1)
            if not any(
                np.array_equal(operation, np.asarray(h.symrel)[index])
                and np.allclose(
                    mp.translation, np.asarray(h.tnons)[index], atol=1e-12, rtol=0
                )
                for index in allowed
            ):
                raise ProducerError(
                    "D4",
                    self.path,
                    "AFM operation",
                    f"actual source symafm={flags} operation/translation required",
                )
            if not np.allclose(
                mp.operation @ h.kpt[mp.ibz_point] + mp.reciprocal_shift,
                mesh[mp.mesh_point],
                atol=1e-10,
                rtol=0,
            ):
                raise ProducerError(
                    "D4",
                    self.path,
                    "AFM target k",
                    "operation/image must produce declared target",
                )
            admitted.append(replace(mp, source_channel=source_channel))
        return tuple(admitted)

    @property
    def stored_source(self):
        return self._source

    def stored_request(self, k, spin):
        mapping = self._mapping(k, spin)
        return mapping.ibz_point, mapping.source_channel

    def _cartesian_operation(self, mapping):
        reciprocal = 2 * np.pi * np.linalg.inv(self._source.header.rprimd).T
        spatial = np.asarray(mapping.operation) * (-1 if mapping.time_reversal else 1)
        cartesian = reciprocal.T @ spatial @ np.linalg.inv(reciprocal.T)
        if not np.allclose(cartesian.T @ cartesian, np.eye(3), atol=1e-10, rtol=0):
            raise ProducerError(
                "D4",
                self.path,
                "Cartesian operation",
                "metric-preserving source symmetry required",
            )
        return cartesian

    def transport_projectors(self, k, spin, coefficients, column_labels):
        """Transport local dual overlaps in the wavefunction Bloch/spin frame."""
        from lawaf.interfaces.abinit.paw_trials import _harmonic_frame_block

        h = self._source.header
        mapping = self._mapping(k, spin)
        spatial = np.asarray(mapping.operation) * (-1 if mapping.time_reversal else 1)
        cartesian = self._cartesian_operation(mapping)
        labels = np.asarray(column_labels, dtype=int)
        result = np.empty_like(coefficients)
        for atom in range(h.natom):
            source_position = spatial.T @ h.xred[atom] + mapping.translation
            candidates = []
            for other in range(h.natom):
                difference = source_position - h.xred[other]
                image = np.rint(difference).astype(int)
                if h.typat[other] == h.typat[atom] and np.allclose(
                    difference, image, atol=1e-10, rtol=0
                ):
                    candidates.append((other, image))
            if len(candidates) != 1:
                raise ProducerError(
                    "D4",
                    self.path,
                    "projector atom map",
                    f"unique same-species source atom for target{atom}",
                )
            other, image = candidates[0]
            phase = np.exp(2j * np.pi * (h.kpt[mapping.ibz_point] @ image))
            groups = sorted(
                {(radial, l) for a, radial, l, _m in column_labels if a == atom}
            )
            for radial, l in groups:
                target = np.flatnonzero(
                    (labels[:, 0] == atom)
                    & (labels[:, 1] == radial)
                    & (labels[:, 2] == l)
                )
                source = np.flatnonzero(
                    (labels[:, 0] == other)
                    & (labels[:, 1] == radial)
                    & (labels[:, 2] == l)
                )
                if len(source) != 2 * l + 1 or len(target) != 2 * l + 1:
                    raise ProducerError(
                        "D4",
                        self.path,
                        "projector harmonic group",
                        "complete matching radial/angular slots required",
                    )
                key = tuple(spatial.ravel()) + (l,)
                if key not in self._projector_rotations:
                    self._projector_rotations[key] = _harmonic_frame_block(l, cartesian)
                result[:, :, target] = (
                    coefficients[:, :, source] @ self._projector_rotations[key]
                ) * phase
        if mapping.time_reversal:
            result = result.conj()
            if h.nspinor == 2:
                result = np.stack((result[:, 1, :], -result[:, 0, :]), axis=1)
        if h.nspinor == 2:
            result = np.einsum("st,btp->bsp", self._spin_rotation(mapping), result)
        return result

    def _spin_rotation(self, mapping):
        key = tuple(np.asarray(mapping.operation).ravel()) + (mapping.time_reversal,)
        if key not in self._spin_rotations:
            from scipy.spatial.transform import Rotation

            cartesian = self._cartesian_operation(mapping)
            proper = cartesian * (1 if np.linalg.det(cartesian) > 0 else -1)
            x, y, z, w = Rotation.from_matrix(proper).as_quat()
            self._spin_rotations[key] = np.array(
                [[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]]
            )
        return self._spin_rotations[key]

    def _full_g(self, ik):
        h = self._source.header
        kg = self._source.kg(ik)
        empty = np.empty((0, kg.shape[1], h.nspinor), dtype=complex)
        full, _, _ = decode_packed_sphere(
            h, ik, int(h.istwfk[ik]), kg, empty, (0, 0), self.path
        )
        return full

    def _mapping(self, k, spin=0):
        ik, channel = int(k), int(spin)
        if not 0 <= channel < len(self.channel_mappings) or not 0 <= ik < len(
            self.channel_mappings[0]
        ):
            raise ProducerError(
                "D4",
                self.path,
                "mesh k/channel index",
                f"k={k},channel={spin} outside declared mesh",
            )
        return self.channel_mappings[channel][ik]

    @staticmethod
    def _order(mp, kg):
        image = mp.operation @ kg - mp.reciprocal_shift[:, None]
        order = np.lexsort((image[2], image[1], image[0]))
        return image[:, order], order

    def kg(self, k):
        mp = self._mapping(k)
        return self._order(mp, self._full_g(mp.ibz_point))[0]

    def coefficients(self, k, spin, band_range):
        return self.decode_full_sphere(k, spin, band_range)[2]

    def eigenvalues(self, k, spin, band_range):
        mp = self._mapping(k, spin)
        return self._source.eigenvalues(
            mp.ibz_point, mp.source_channel, band_range
        )

    def decode_full_sphere(self, k, spin, band_range):
        mp = self._mapping(k, spin)
        kg, _, cg = self._source.decode_full_sphere(
            mp.ibz_point, mp.source_channel, band_range
        )
        phase = np.exp(
            2j
            * np.pi
            * ((kg.T + self._source.header.kpt[mp.ibz_point]) @ mp.translation)
        )
        image, order = self._order(mp, kg)
        coefficients = cg[:, order, :] * phase[order][None, :, None]
        if mp.time_reversal:
            coefficients = coefficients.conj()
            if self.header.nspinor == 2:
                coefficients = np.stack(
                    (coefficients[:, :, 1], -coefficients[:, :, 0]), axis=-1
                )
        if self.header.nspinor == 2:
            coefficients = coefficients @ self._spin_rotation(mp).T
        return image, np.ones(image.shape[1]), coefficients
