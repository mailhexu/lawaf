# SPDX-License-Identifier: BSD-2-Clause
"""Caller-owned array source adapter (story 063).

Implements the Architecture §3 source seam for in-memory input: caller-owned
read-only arrays (states, ``kg``, eigenvalues) with explicit lattice,
storage, symmetry, spin, atom and projector metadata.  In-memory input and
WFK-backed input are two carriers of one identical source contract — the
adapter reuses the story-060 metadata surface (:class:`WfkHeader`) and the
shared packed decode (:func:`decode_packed_sphere`), defines no second
convention, and never copies or mutates caller arrays.

Caller arrays must be C-contiguous (complex128 states, int32 ``kg``); the
adapter holds them as ``writeable=False`` views.  Scratch routed through the
adapter is bounded; actual allocation observers measure decoder scratch/output.
Failures name the missing/inconsistent metadata field and the expected value.

No Wannier90 library, binding, ctypes or f90wrap is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from lawaf.interfaces.abinit.source_wfk import (
    _NATIVE_PACKED_RESTRICTION,
    WfkHeader,
    _gmet_from_rprimd,
    _kg_within_sphere,
    decode_packed_sphere,
    resolve_istwfk,
)

__all__ = [
    "ArraySourceError",
    "SymmetryMapEntry",
    "CallerMetadata",
    "CallerArraySource",
]


class ArraySourceError(Exception):
    """Fail-closed caller-array source diagnostic naming field/expectation."""

    def __init__(self, diagnostic, file, field, detail):
        self.diagnostic = diagnostic
        self.file = str(file)
        self.field = field
        self.detail = detail
        super().__init__(f"[{diagnostic}] file={file} field={field}: {detail}")


def _fail(diagnostic, file, field, expected, detail="") -> ArraySourceError:
    msg = f"expected {expected}. {detail}".rstrip()
    return ArraySourceError(diagnostic, file, field, msg)


@dataclass(frozen=True)
class SymmetryMapEntry:
    """One explicit IBZ->mesh map entry (FR-020 caller metadata).

    Every entry carries the spatial operation, its fractional translation,
    the reciprocal-space shift and the spin-channel label; expansion itself
    stays with the producer (story 064 / epic-048-c rules).
    """

    mesh_point: int
    ibz_point: int
    operation: np.ndarray  # (3, 3) int
    translation: np.ndarray  # (3,) fractional
    reciprocal_shift: np.ndarray  # (3,) integer G shift
    spin_label: str  # e.g. "same" | "flip" | channel name
    time_reversal: bool


@dataclass(frozen=True)
class CallerMetadata:
    """Canonical source header plus borrowed electronic arrays and target map.

    No second header schema or inferred occupations/symmetries. Target mesh
    size is explicit, independently of stored IBZ Nk. Owners keep borrowed
    buffers alive and unchanged during a consumer pass.
    """

    header: WfkHeader = None
    eigenvalues: np.ndarray = None  # (mband, stored_k, nsppol), Hartree
    eigenvalue_units: str = None
    length_units: str = None
    symmetry_map: Tuple[SymmetryMapEntry, ...] = ()
    nkpt_mesh: int = None
    mesh_kpoints: np.ndarray = None  # (nkpt_mesh,3), exact declared target order
    coefficient_carrier: object = None
    dataset_md5: Tuple[str, ...] = ()


def _readonly_view(arr, where, field, dtype):
    arr = np.asarray(arr)
    if arr.dtype != dtype:
        raise _fail(
            "D6", where, field, f"dtype {np.dtype(dtype)}, found " f"dtype {arr.dtype}"
        )
    if not arr.flags.c_contiguous:
        raise _fail(
            "D6",
            where,
            field,
            "a C-contiguous caller array; non-contiguous views are "
            "rejected rather than silently copied",
        )
    view = arr.view()
    view.flags.writeable = False
    return view


class CallerArraySource:
    """Read-only caller-owned array source implementing the story-060 seam.

    Borrows the caller arrays (no copies, no WFK write, no mutation); block
    access returns read-only views into those arrays.  The exposed header is
    the same :class:`WfkHeader` surface the WFK reader presents, so
    downstream consumers cannot distinguish the carriers.
    """

    def __init__(self, metadata: CallerMetadata, states, kg_arrays):
        self._path = "<caller-owned arrays>"
        where = self._path
        m = metadata
        if not isinstance(m.header, WfkHeader):
            raise _fail("D1", where, "header", "the complete canonical WfkHeader")
        if m.length_units != "bohr":
            raise _fail("D1", where, "length_units", "explicit bohr declaration")
        if m.eigenvalue_units != "hartree":
            raise _fail("D7", where, "eigenvalue_units", "explicit hartree declaration")
        if m.eigenvalues is None:
            raise _fail("D1", where, "eigenvalues", "caller-owned energy records")
        if type(m.nkpt_mesh) is not int or m.nkpt_mesh <= 0:
            raise _fail(
                "D4",
                where,
                "nkpt_mesh",
                "explicit positive target mesh size, not inferred IBZ count",
            )
        h = m.header
        for name in ("ecut", "ecut_eff"):
            value = getattr(h, name)
            if value is None:
                raise _fail("D1", where, name, "explicit source kinetic cutoff")
            if not np.isfinite(value) or value <= 0:
                raise _fail("D2", where, name, "finite positive source kinetic cutoff")
        for name in (
            "nkpt",
            "natom",
            "ntypat",
            "npsp",
            "nsym",
            "mband",
            "nsppol",
            "nspinor",
            "nspden",
            "nshiftk",
            "nshiftk_orig",
        ):
            if getattr(h, name) is None:
                raise _fail("D1", where, name, "explicit source count")
            minimum = 0 if name in ("nshiftk", "nshiftk_orig") else 1
            if (
                not isinstance(getattr(h, name), (int, np.integer))
                or getattr(h, name) < minimum
            ):
                raise _fail("D2", where, name, f"integer source count >= {minimum}")
        if h.usepaw is None:
            raise _fail(
                "D1", where, "usepaw", "explicit source operator classification"
            )
        if (
            h.nsppol not in (1, 2)
            or h.nspinor not in (1, 2)
            or h.nspden not in (1, 2, 4)
        ):
            raise _fail(
                "D2", where, "spin tuple", "native spin/channel/density dimensions"
            )
        nk, nsppol, nspinor = h.nkpt, h.nsppol, h.nspinor
        shapes = {
            "rprimd": (3, 3),
            "kpt": (nk, 3),
            "istwfk_raw": (nk,),
            "istwfk": (nk,),
            "npwarr": (nk,),
            "nband": (nk * nsppol,),
            "typat": (h.natom,),
            "xred": (h.natom, 3),
            "znucltypat": (h.ntypat,),
            "symrel": (h.nsym, 3, 3),
            "tnons": (h.nsym, 3),
            "symafm": (h.nsym,),
            "lmn_size": (h.npsp,),
            "so_psp": (h.npsp,),
            "wtk": (nk,),
            "kptrlatt": (3, 3),
            "kptrlatt_orig": (3, 3),
            "shiftk": (h.nshiftk, 3),
            "shiftk_orig": (h.nshiftk_orig, 3),
        }
        views = {}
        for name, shape in shapes.items():
            value = getattr(h, name)
            if value is None:
                raise _fail("D1", where, name, "explicit source header field")
            array = np.asarray(value)
            if (
                array.shape != shape
                or array.dtype.kind not in "iuf"
                or not np.all(np.isfinite(array))
            ):
                raise _fail(
                    "D2",
                    where,
                    name,
                    f"finite numeric shape{shape}; found{array.shape}",
                )
            view = array.view()
            view.flags.writeable = False
            views[name] = view
        for name in (
            "nband",
            "npwarr",
            "istwfk_raw",
            "istwfk",
            "typat",
            "symrel",
            "symafm",
            "lmn_size",
            "so_psp",
            "kptrlatt",
            "kptrlatt_orig",
        ):
            if views[name].dtype.kind not in "iu":
                raise _fail(
                    "D2", where, name, "integer metadata, no truncating conversion"
                )
        if (
            np.any(views["nband"] <= 0)
            or np.any(views["nband"] > h.mband)
            or np.any(views["npwarr"] <= 0)
        ):
            raise _fail(
                "D2", where, "nband/npwarr", "positive counts within declared mband"
            )
        if int(views["nband"].sum()) != h.bantot:
            raise _fail(
                "D2",
                where,
                "bantot",
                f"sum(nband)={int(views['nband'].sum())}; found{h.bantot}",
            )
        if np.any(views["typat"] < 1) or np.any(views["typat"] > h.ntypat):
            raise _fail("D2", where, "typat", f"atom species indices1..{h.ntypat}")
        if abs(np.linalg.det(views["rprimd"])) <= np.finfo(float).tiny:
            raise _fail("D2", where, "rprimd", "nonsingular source lattice")
        if np.any(abs(views["symafm"]) != 1):
            raise _fail("D2", where, "symafm", "native magnetic signs+1/-1")
        if len(h.md5_pseudos) != h.npsp:
            raise _fail(
                "D2", where, "md5_pseudos", f"{h.npsp} source dataset identities"
            )
        if h.occ is None or np.asarray(h.occ).size not in (
            h.bantot,
            h.mband * nk * nsppol,
        ):
            raise _fail("D2", where, "occ", "complete actual source occupations")
        # Every published header array is a read-only borrowed view, not an
        # owned copy or a writable alias. Preserve all other header fields.
        for name, value in vars(h).items():
            if isinstance(value, np.ndarray) and name not in views:
                view = value.view()
                view.flags.writeable = False
                views[name] = view
        resolved = []
        for ik in range(nk):
            try:
                mode = resolve_istwfk(
                    int(views["istwfk_raw"][ik]),
                    views["kpt"][ik],
                    nspinor=nspinor,
                    pawspnorb=h.pawspnorb,
                )
            except Exception as error:
                raise _fail(
                    "D3", where, f"istwfk[{ik}]", "legal per-point storage", str(error)
                ) from error
            if mode > 1 and (nspinor != 1 or h.pawspnorb):
                raise _fail(
                    "D3",
                    where,
                    f"istwfk[{ik}]",
                    f"full storage for nspinor={nspinor}, pawspnorb={h.pawspnorb}; {_NATIVE_PACKED_RESTRICTION}",
                )
            resolved.append(mode)
        modes = np.asarray(resolved, np.int32)
        modes.flags.writeable = False
        views["istwfk"] = modes
        metric = _gmet_from_rprimd(views["rprimd"])
        metric.flags.writeable = False
        views["gmet"] = metric
        self._header = replace(
            h, **views, title=tuple(h.title), md5_pseudos=tuple(h.md5_pseudos)
        )
        self._resolved_istwfk = modes
        self._states = {}
        self._kg = {}
        for ik in range(nk):
            if ik not in kg_arrays:
                raise _fail(
                    "D1", where, f"kg[{ik}]", "source G vectors at every stored k"
                )
            kg = _readonly_view(kg_arrays[ik], where, f"kg[{ik}]", np.int32)
            expected = (3, int(h.npwarr[ik]))
            if kg.shape != expected:
                raise _fail(
                    "D2",
                    where,
                    f"kg[{ik}]",
                    f"declared shape{expected}; found{kg.shape}",
                )
            self._kg[ik] = kg
            for spin in range(nsppol):
                key = (ik, spin)
                if key not in states:
                    raise _fail(
                        "D1",
                        where,
                        f"states[{key}]",
                        "every stored k/channel state block",
                    )
                state = _readonly_view(
                    states[key], where, f"states[{key}]", np.complex128
                )
                expected = (int(h.nband[ik + spin * nk]), int(h.npwarr[ik]), nspinor)
                if state.shape != expected:
                    raise _fail(
                        "D2",
                        where,
                        f"states[{key}]",
                        f"declared shape{expected}; found{state.shape}",
                    )
                self._states[key] = state
        eigen = _readonly_view(m.eigenvalues, where, "eigenvalues", np.float64)
        if eigen.ndim == 2:
            eigen = eigen[:, :, None]
        if eigen.shape != (h.mband, nk, nsppol):
            raise _fail(
                "D7",
                where,
                "eigenvalues",
                f"(nband, nk, nsppol)={(h.mband,nk,nsppol)}; found{eigen.shape}",
            )
        self._eigen = eigen
        if m.mesh_kpoints is None:
            raise _fail(
                "D4",
                where,
                "mesh_kpoints",
                "explicit target coordinates for the declared map",
            )
        targets = np.asarray(m.mesh_kpoints)
        if (
            targets.shape != (m.nkpt_mesh, 3)
            or targets.dtype.kind != "f"
            or not np.all(np.isfinite(targets))
        ):
            raise _fail("D4", where, "mesh_kpoints", f"finite shape{(m.nkpt_mesh,3)}")
        targets = targets.view()
        targets.flags.writeable = False
        seen = {}
        for entry in m.symmetry_map:
            point = entry.mesh_point
            if type(point) is not int or not 0 <= point < m.nkpt_mesh or point in seen:
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[mesh_point={point}]",
                    "unique target point; duplicated/out-of-range entry",
                )
            if (
                not isinstance(entry.ibz_point, (int, np.integer))
                or not 0 <= entry.ibz_point < nk
            ):
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[{point}].ibz_point",
                    f"source point0..{nk-1}",
                )
            if not isinstance(entry.time_reversal, (bool, np.bool_)):
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[{point}].time_reversal",
                    "explicit antiunitary boolean",
                )
            arrays = {}
            for field, shape, kind in (
                ("operation", (3, 3), "iu"),
                ("translation", (3,), "f"),
                ("reciprocal_shift", (3,), "iu"),
            ):
                value = getattr(entry, field)
                if value is None:
                    raise _fail(
                        "D4",
                        where,
                        f"symmetry_map[{point}].{field}",
                        f"explicit shape{shape}",
                    )
                array = np.asarray(value)
                if (
                    array.shape != shape
                    or array.dtype.kind not in kind
                    or not np.all(np.isfinite(array))
                ):
                    raise _fail(
                        "D4",
                        where,
                        f"symmetry_map[{point}].{field}",
                        f"finite shape{shape} with dtype kind{kind}",
                    )
                view = array.view()
                view.flags.writeable = False
                arrays[field] = view
            if abs(round(np.linalg.det(arrays["operation"]))) != 1:
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[{point}].operation",
                    "invertible integer lattice operation",
                )
            if not isinstance(entry.spin_label, str) or not entry.spin_label:
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[{point}].spin_label",
                    "explicit spin-channel label",
                )
            image = (
                arrays["operation"] @ self._header.kpt[entry.ibz_point]
                + arrays["reciprocal_shift"]
            )
            if not np.allclose(image, targets[point], atol=1e-10, rtol=0):
                raise _fail(
                    "D4",
                    where,
                    f"symmetry_map[{point}]",
                    "source operation/shift must produce the declared target k",
                )
            seen[point] = replace(entry, **arrays)
        missing = [point for point in range(m.nkpt_mesh) if point not in seen]
        if missing:
            raise _fail(
                "D4",
                where,
                "symmetry_map",
                f"coverage of explicit mesh size{m.nkpt_mesh}; uncovered mesh points {missing}",
            )
        self._symmetry_map = tuple(seen[point] for point in range(m.nkpt_mesh))
        if (
            h.usepaw
            and m.coefficient_carrier is None
            and not (m.dataset_md5 or any(h.md5_pseudos))
        ):
            raise _fail(
                "D5",
                where,
                "coefficient_carrier/dataset_md5",
                "matched PAW operator data; never silently treated as NC",
            )
        self._meta = replace(
            m,
            header=self._header,
            eigenvalues=self._eigen,
            symmetry_map=self._symmetry_map,
            mesh_kpoints=targets,
        )

    # -- seam (identical to WfkReader) ---------------------------------------

    @property
    def path(self):
        return self._path

    @property
    def format(self):
        return "caller-arrays"

    @property
    def header(self) -> WfkHeader:
        return self._header

    @property
    def metadata(self) -> CallerMetadata:
        return self._meta

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check_ksb(self, k, spin, band_range):
        h = self._header
        if not (0 <= int(k) < h.nkpt):
            raise _fail(
                "D6", self._path, f"k index {int(k)}", f"a k index in [0, {h.nkpt})"
            )
        if not (0 <= int(spin) < h.nsppol):
            raise _fail(
                "D6",
                self._path,
                f"spin index {int(spin)}",
                f"a spin index in [0, {h.nsppol})",
            )
        nb = int(h.nband[int(k) + int(spin) * h.nkpt])
        b0, b1 = band_range
        if not (0 <= b0 < b1 <= nb):
            raise _fail(
                "D6",
                self._path,
                f"(band-range=({b0},{b1}), k={int(k)}, spin={int(spin)})",
                f"a band range within nband({int(k)},{int(spin)})={nb}",
            )

    def kg(self, k):
        """Stored G vectors (3, npw_k), read-only view of the caller array."""
        h = self._header
        ik = int(k)
        if not (0 <= ik < h.nkpt):
            raise _fail(
                "D6", self._path, f"k index {ik}", f"a k index in [0, {h.nkpt})"
            )
        kg = self._kg[ik]
        if h.ecut_eff and not _kg_within_sphere(h.kpt[ik], kg, h.gmet, h.ecut_eff):
            raise _fail(
                "D2",
                self._path,
                f"kg[{ik}]",
                f"stored G vectors within the G-sphere bound "
                f"ecut_eff={h.ecut_eff} Hartree",
            )
        return kg

    def coefficients(self, k, spin, band_range):
        """Stored coefficient block (nband, npw, nspinor) — read-only view,
        zero copy into the caller arrays."""
        self._check_ksb(k, spin, band_range)
        b0, b1 = band_range
        return self._states[(int(k), int(spin))][b0:b1]

    def eigenvalues(self, k, spin, band_range):
        """Eigenvalues exactly as declared (Hartree) — read-only view."""
        self._check_ksb(k, spin, band_range)
        b0, b1 = band_range
        return self._eigen[b0:b1, int(k), int(spin)]

    def decode_full_sphere(self, k, spin, band_range):
        """Packed decode through the shared story-060 path (bitwise-identical
        results to the WFK reader on the same stored blocks)."""
        self._check_ksb(k, spin, band_range)
        self._check_packed_spinor(k)
        h = self._header
        ik = int(k)
        mode = int(self._resolved_istwfk[ik])
        kg_stored = self.kg(ik)
        cg_stored = self.coefficients(k, spin, band_range)
        return decode_packed_sphere(
            h, ik, mode, kg_stored, cg_stored, band_range, self._path
        )

    def _check_packed_spinor(self, ik):
        h = self._header
        mode = int(self._resolved_istwfk[int(ik)])
        so = int(np.max(np.asarray(h.so_psp, dtype=int), initial=0))
        if mode > 1 and (h.nspinor != 1 or h.pawspnorb or so > 1):
            raise _fail(
                "D3",
                self._path,
                f"k index {int(ik)}",
                f"packed storage resolved istwfk={mode} requested but "
                f"forbidden: nspinor={h.nspinor}, pawspnorb={h.pawspnorb} "
                f"(native restriction {_NATIVE_PACKED_RESTRICTION})",
            )

    @property
    def symmetry_map(self):
        """Validated target map, preserving reciprocal shifts and spin labels."""
        return self._symmetry_map

    def require_symmetry(self):
        """Actual source symmetry group, not inferred from target map entries."""
        h = self._header
        return h.symrel, h.tnons, h.symafm
