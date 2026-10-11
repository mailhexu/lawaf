# SPDX-License-Identifier: BSD-2-Clause
"""Supplied PAW dual-projector coefficient carrier (story 062).

Implements the Architecture §4.3 carrier contract
``lawaf.paw_projection_coefficients`` version 1: a declared consumer
contract — NOT a baseline native export (research
2026-10-02-w90-paw-input-readers.md F1/F7).

Two carriers present the identical validated record view to consumers:

* :class:`SuppliedCoefficientArrays` — read-only caller-memory views over
  the canonical Python axes
  ``(stored_channel, source_k, physical_band, spinor_component,
  total_projector)`` (complex128);
* :class:`PawCoefficientNetcdfCarrier` — a slab-readable NetCDF file with
  the same logical axes plus a trailing ``complex2`` (real, imag) f8 axis.
  Only requested single-band hyperslabs are read; the full tensor is never
  materialized.

Canonical record semantics (research F5/F6): ``dualprojector`` = ``<p~|psi>``
(the ABINIT cprj semantic).  Pseudo-partial-wave and AE-partial-wave
overlaps are different objects: they are identified and readable but never
relabeled or converted (no general Gram conversion exists — research F5.2).
Energies and packed-G weights are never duplicated into the record.  The
presence of a coefficient record never waives the matched PAW dataset for
derived kernels (Delta S, Q(b), one-center trial integrals); ``usepaw`` is
always read from the source header, never inferred from the coefficients.

No Wannier90 library, binding, ctypes or f90wrap is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WITNESS_RTOL",
    "REQUIRED_FUNCTION_KIND",
    "KNOWN_FUNCTION_KINDS",
    "PawCoefficientError",
    "SupplierMeta",
    "ValidatedCoefficientRecord",
    "SuppliedCoefficientArrays",
    "PawCoefficientNetcdfCarrier",
    "write_netcdf_carrier",
    "require_dataset_identity",
    "require_dualprojector",
    "witness_compare",
]

SCHEMA_NAME = "lawaf.paw_projection_coefficients"
SCHEMA_VERSION = 1
WITNESS_RTOL = 1e-12
REQUIRED_FUNCTION_KIND = "dualprojector"
KNOWN_FUNCTION_KINDS = (
    "dualprojector",
    "pseudo_partial_wave",
    "ae_partial_wave",
    "wannier_a",
)


class PawCoefficientError(Exception):
    """Fail-closed carrier diagnostic naming file/field and expectation."""

    def __init__(self, diagnostic, file, field, detail):
        self.diagnostic = diagnostic
        self.file = str(file)
        self.field = field
        self.detail = detail
        super().__init__(f"[{diagnostic}] file={file} field={field}: {detail}")


def _fail(diagnostic, file, field, expected, detail="") -> PawCoefficientError:
    msg = f"expected {expected}. {detail}".rstrip()
    return PawCoefficientError(diagnostic, file, field, msg)


@dataclass(frozen=True)
class SupplierMeta:
    """Explicit supplied-coefficient metadata (Architecture §4.3, research F6).

    Every field is a declaration; none is defaulted or guessed.  ``dataset_md5``
    entries are aligned with the source-header pseudopotential order.
    """

    schema_name: str = None
    schema_version: int = None
    function_kind: str = None
    state_frame: str = None
    kpoints: np.ndarray = None  # (nk, 3) reduced coords of the source k list
    spin_tuple: Tuple[int, int, int] = None  # (nsppol, nspinor, nspden)
    source_channel: int = (
        None  # explicit0-based source isppol for this one-channel carrier
    )
    band_range: Tuple[int, int] = None  # declared [b0, b1) coverage
    dataset_md5: Tuple[str, ...] = ()
    atom_species: Tuple[str, ...] = ()  # natom labels in typat order
    projector_counts: Tuple[int, ...] = ()  # per-atom projector counts
    projector_labels: Tuple[
        Tuple[int, int, int, int], ...
    ] = ()  # atom, radial ID, l, m
    harmonic_convention: str = ""  # identified, never guessed or converted
    conjugation: str = None  # explicit <p~|psi> declaration
    raw_normalization: bool = None
    phase_declaration: str = None
    producer: str = ""
    symmetry_map_reference: str = None  # required for expanded records

    def __post_init__(self):
        if self.kpoints is not None:
            view = np.asarray(self.kpoints).view()
            view.flags.writeable = False
            object.__setattr__(self, "kpoints", view)
        for name in (
            "spin_tuple",
            "band_range",
            "dataset_md5",
            "atom_species",
            "projector_counts",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if self.projector_labels is not None:
            object.__setattr__(
                self,
                "projector_labels",
                tuple(tuple(label) for label in self.projector_labels),
            )


# ---------------------------------------------------------------------------
# shared validation (D1-D8) over both carriers
# ---------------------------------------------------------------------------


def _meta_schema_check(meta, where):
    # D1: missing/unknown schema
    if not getattr(meta, "schema_name", None):
        raise _fail(
            "D1",
            where,
            "schema_name",
            "the declared schema name " f"{SCHEMA_NAME!r}; no default schema exists",
        )
    if meta.schema_name != SCHEMA_NAME:
        raise _fail(
            "D1", where, "schema_name", f"{SCHEMA_NAME!r}, found {meta.schema_name!r}"
        )
    version = getattr(meta, "schema_version", None)
    if (
        isinstance(version, (bool, np.bool_))
        or not isinstance(version, (int, np.integer))
        or version != SCHEMA_VERSION
    ):
        raise _fail(
            "D1",
            where,
            "schema_version",
            f"exact integer supported version {SCHEMA_VERSION}, found {version!r}",
        )


def _validate_meta(meta, header, shape, where, *, symmetry_map_reference=None):
    """D2-D8 consistency of the declared metadata against the source header
    and the coefficient shape (stored_channel, source_k, physical_band,
    spinor_component, total_projector)."""
    _meta_schema_check(meta, where)
    if meta.spin_tuple is None or len(meta.spin_tuple) != 3:
        raise _fail(
            "D6", where, "spin_tuple", "an explicit(nsppol,nspinor,nspden) tuple"
        )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in meta.spin_tuple
    ):
        raise _fail("D6", where, "spin_tuple", "three positive integral spin counts")
    _axis_check(shape, meta, where)
    if (
        isinstance(meta.source_channel, (bool, np.bool_))
        or not isinstance(meta.source_channel, (int, np.integer))
        or not 0 <= int(meta.source_channel) < int(meta.spin_tuple[0])
    ):
        raise _fail(
            "D6",
            where,
            "source_channel",
            f"an explicit source channel in [0,{meta.spin_tuple[0]}); found {meta.source_channel!r}",
        )

    # D2: function kind must at least be a known declaration
    kind = getattr(meta, "function_kind", None)
    if not kind:
        raise _fail(
            "D2",
            where,
            "function_kind",
            "an explicit function-kind declaration "
            f"(one of {list(KNOWN_FUNCTION_KINDS)}); none given",
        )
    if kind not in KNOWN_FUNCTION_KINDS:
        raise _fail(
            "D2",
            where,
            "function_kind",
            f"one of {list(KNOWN_FUNCTION_KINDS)}, found {kind!r}",
        )

    kpoints = np.asarray(meta.kpoints, dtype=float)
    nk = shape[1]
    if kpoints.shape != (nk, 3):
        raise _fail(
            "D5",
            where,
            "kpoints",
            f"declared k list covering {nk} source_k points with "
            f"reduced coords (nk, 3), found shape {kpoints.shape}",
        )
    if not np.isfinite(kpoints).all():
        raise _fail("D6", where, "kpoints", "finite source reduced coordinates")
    if meta.state_frame not in ("source", "ibz", "expanded"):
        raise _fail(
            "D6", where, "state_frame", "an explicit source, ibz or expanded frame"
        )
    if meta.state_frame == "expanded":
        if (
            not isinstance(meta.symmetry_map_reference, str)
            or not meta.symmetry_map_reference.strip()
        ):
            raise _fail(
                "D6",
                where,
                "symmetry_map_reference",
                "explicit reference for expanded states",
            )
        if meta.symmetry_map_reference != symmetry_map_reference:
            raise _fail(
                "D6",
                where,
                "symmetry_map_reference",
                f"consumer reference {symmetry_map_reference!r}, found {meta.symmetry_map_reference!r}",
            )
    elif meta.symmetry_map_reference is not None:
        raise _fail(
            "D6",
            where,
            "symmetry_map_reference",
            "no expansion reference for stored source/ibz states",
        )
    if (
        header is not None
        and meta.state_frame == "ibz"
        and header.kptopt not in (1, 2, 4)
    ):
        raise _fail(
            "D6",
            where,
            "state_frame",
            f"source reduction compatible with ibz declaration; kptopt={header.kptopt}",
        )
    if header is not None:
        # Validate spin identity before indexing a selected channel.
        expected_spin = (int(header.nsppol), int(header.nspinor), int(header.nspden))
        if tuple(int(x) for x in meta.spin_tuple) != expected_spin:
            raise _fail(
                "D6",
                where,
                "spin_tuple",
                f"source tuple{expected_spin}; found{meta.spin_tuple}",
            )
        # D5: band/k coverage vs source header
        if nk != int(header.nkpt):
            raise _fail(
                "D5",
                where,
                "source_k",
                f"coverage of nkpt={int(header.nkpt)} source k-points, " f"found {nk}",
            )
        b0, b1 = int(meta.band_range[0]), int(meta.band_range[1])
        for ik in range(nk):
            nb = int(header.nband[ik + int(meta.source_channel) * int(header.nkpt)])
            if b1 > nb:
                raise _fail(
                    "D5",
                    where,
                    f"band_range(k={ik},channel={meta.source_channel})",
                    f"bands within selected-channel header nband={nb}, "
                    f"declared range ({b0}, {b1})",
                )
        if not np.allclose(
            kpoints, np.asarray(header.kpt, dtype=float), rtol=0, atol=1e-10
        ):
            bad = int(
                np.argmax(
                    np.abs(kpoints - np.asarray(header.kpt, dtype=float)).sum(axis=1)
                )
            )
            raise _fail(
                "D6",
                where,
                f"kpoints[{bad}]",
                f"source reduced coordinates "
                f"{np.asarray(header.kpt, dtype=float)[bad].tolist()}, found "
                f"{kpoints[bad].tolist()}",
            )
        # D4: dataset identity vs header fingerprints
        if meta.dataset_md5:
            expected = list(getattr(header, "md5_pseudos", []))
            got = [str(x) for x in meta.dataset_md5]
            if len(got) != len(expected):
                raise _fail(
                    "D4",
                    where,
                    "dataset_md5",
                    f"{len(expected)} per-species fingerprints "
                    f"matching the source header, found {len(got)}",
                )
            for isp, (e, g) in enumerate(zip(expected, got)):
                if e != g:
                    raise _fail(
                        "D4",
                        where,
                        f"dataset_md5[{isp}]",
                        f"source header fingerprint {e!r}, found {g!r}",
                    )

    # D7: conjugation / raw normalization / phase declarations
    if not getattr(meta, "conjugation", None):
        raise _fail(
            "D7",
            where,
            "conjugation",
            "an explicit conjugation declaration " "(e.g. 'trial_bra' for <p~|psi>)",
        )
    if not getattr(meta, "phase_declaration", None):
        raise _fail(
            "D7",
            where,
            "phase_declaration",
            "an explicit Bloch/atom/spin frame declaration",
        )
    if getattr(meta, "raw_normalization", None) is None:
        raise _fail(
            "D7",
            where,
            "raw_normalization",
            "an explicit raw-normalization declaration "
            "(True = source-normalized overlaps, never rescaled)",
        )
    if meta.conjugation not in ("trial_bra", "state_bra"):
        raise _fail(
            "D7", where, "conjugation", "identified trial_bra or state_bra convention"
        )
    if not isinstance(meta.raw_normalization, (bool, np.bool_)):
        raise _fail("D7", where, "raw_normalization", "explicit boolean declaration")
    if meta.phase_declaration != "bloch_source_frame":
        raise _fail(
            "D6",
            where,
            "phase_declaration",
            "bloch_source_frame; no unverified gauge conversion",
        )
    if kind == REQUIRED_FUNCTION_KIND and (
        meta.conjugation != "trial_bra" or not meta.raw_normalization
    ):
        raise _fail(
            "D7",
            where,
            "conjugation/raw_normalization",
            "raw trial-bra dualprojector coefficients <p~|psi>",
        )

    # D8: atom ordering/labels vs typat order and per-atom counts vs lmn_size
    natom = len(meta.atom_species)
    counts = tuple(int(c) for c in meta.projector_counts)
    total_projector = shape[4]
    if natom == 0 or len(counts) != natom:
        raise _fail(
            "D8",
            where,
            "atom_species/projector_counts",
            f"per-atom labels and counts for all atoms in typat order "
            f"(natom labels and natom counts); found "
            f"{len(meta.atom_species)} labels and {len(counts)} counts",
        )
    if header is not None:
        typat = np.asarray(header.typat).ravel()
        if natom != int(header.natom) or len(typat) != natom:
            raise _fail(
                "D8",
                where,
                "atom_species",
                f"exactly {header.natom} atoms in source typat order, found {natom}",
            )
        lmn = np.asarray(header.lmn_size).ravel()
        for ia in range(min(natom, len(typat))):
            expected_count = int(lmn[int(typat[ia]) - 1])
            if counts[ia] != expected_count:
                raise _fail(
                    "D3",
                    where,
                    f"projector_counts[{ia}]",
                    f"header lmn_size(typat={int(typat[ia])}) = "
                    f"{expected_count}, found {counts[ia]}",
                )
    if sum(counts) != total_projector:
        raise _fail(
            "D3",
            where,
            "projector_counts",
            f"per-atom counts summing to the total_projector axis "
            f"({total_projector}), found sum {sum(counts)}",
        )
    if header is not None:
        from ase.data import chemical_symbols

        expected_labels = tuple(
            chemical_symbols[int(header.znucltypat[int(t) - 1])] for t in header.typat
        )
        if meta.atom_species != expected_labels:
            raise _fail(
                "D8",
                where,
                "atom_species",
                f"source typat species labels {expected_labels}, found {meta.atom_species}",
            )
    labels = meta.projector_labels
    if (
        labels is None
        or len(labels) != total_projector
        or not isinstance(meta.harmonic_convention, str)
        or not meta.harmonic_convention.strip()
    ):
        raise _fail(
            "D8",
            where,
            "projector_labels/harmonic_convention",
            f"{total_projector} explicit(atom,radial,l,m) labels and a named harmonic convention",
        )
    if any(
        len(label) != 4
        or any(not isinstance(value, (int, np.integer)) for value in label)
        for label in labels
    ):
        raise _fail(
            "D8", where, "projector_labels", "four integer identifiers per projector"
        )
    label_keys = tuple(tuple(int(value) for value in label) for label in labels)
    atom_order = tuple(atom for atom, count in enumerate(counts) for _ in range(count))
    if (
        tuple(label[0] for label in label_keys) != atom_order
        or len(set(label_keys)) != total_projector
        or any(radial < 0 or l < 0 or abs(m) > l for _atom, radial, l, m in label_keys)
    ):
        raise _fail(
            "D8",
            where,
            "projector_labels",
            "unique valid radial/angular IDs, grouped in the declared atom order/counts",
        )
    return shape


def _axis_check(shape, meta, where):
    if len(shape) != 5 or any(n <= 0 for n in shape):
        raise _fail(
            "D5",
            where,
            "coefficients",
            f"five nonempty canonical v1 axes; found shape{shape}",
        )
    stored_channel, nk, nband, ncomp, nproj = shape
    if stored_channel != 1:
        raise _fail(
            "D5",
            where,
            "stored_channel",
            "1 stored channel record per v1 carrier, found " f"{stored_channel}",
        )
    if ncomp not in (1, 2):
        raise _fail(
            "D6",
            where,
            "spinor_component",
            "length 1 (scalar/collinear) or 2 (coherent spinor), " f"found {ncomp}",
        )
    if meta.spin_tuple is not None and ncomp != int(meta.spin_tuple[1]):
        raise _fail(
            "D6",
            where,
            "spinor_component",
            f"declared nspinor={int(meta.spin_tuple[1])}, "
            f"found axis length {ncomp}",
        )
    if meta.band_range is None or len(meta.band_range) != 2:
        raise _fail(
            "D5", where, "band_range", "an explicit original-band interval[b0,b1)"
        )
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in meta.band_range
    ):
        raise _fail(
            "D5",
            where,
            "band_range",
            "integral original physical-band endpoints, never truncated",
        )
    b0, b1 = (int(x) for x in meta.band_range)
    if not 0 <= b0 < b1:
        raise _fail("D5", where, "band_range", f"0<=b0<b1; found({b0},{b1})")
    if nband != b1 - b0:
        raise _fail(
            "D5",
            where,
            "physical_band",
            f"declared band range ({b0}, {b1}) covering {b1 - b0} "
            f"bands, found {nband}",
        )
    return shape


# ---------------------------------------------------------------------------
# validated record view (identical for both carriers)
# ---------------------------------------------------------------------------


class ValidatedCoefficientRecord:
    """The identical validated record view presented by both carriers.

    Consumers see metadata, identity and single-band slab access only.
    Slabs are read-only and exactly equal to the stored values.
    """

    def __init__(
        self, meta, header, shape, slab_reader, where, symmetry_map_reference=None
    ):
        self._meta = meta
        self._header = header
        self._shape = tuple(int(x) for x in shape)
        self._slab_reader = slab_reader
        self._where = where
        self._symmetry_map_reference = symmetry_map_reference
        self._witness_residual = None

    # -- identity -----------------------------------------------------------

    @property
    def witness_residual(self):
        """Actual same-source witness result; metadata alone cannot set it."""
        return self._witness_residual

    def require_witness(self):
        if self._witness_residual is None:
            raise _fail(
                "D4", self._where, "coefficient witness",
                "witness_compare(record, independent_reference) must pass before PAW handoff",
            )


    @property
    def symmetry_map_reference(self):
        return self._symmetry_map_reference

    @property
    def meta(self) -> SupplierMeta:
        return self._meta

    @property
    def header(self):
        """The source header anchor (``usepaw`` is always read from here)."""
        return self._header

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        return self._shape

    @property
    def n_total_projector(self) -> int:
        return self._shape[4]

    @property
    def function_kind(self) -> str:
        return self._meta.function_kind

    @property
    def paw_cell(self) -> bool:
        """PAW classification strictly from the source header, never from
        the presence of coefficients."""
        if self._header is None:
            raise _fail(
                "D6",
                self._where,
                "header",
                "a source header anchoring the state identity; "
                "coefficient records without a source are not "
                "classifiable",
            )
        return int(self._header.usepaw) == 1

    def require_dualprojector(self):
        """D2: the operator-required kind, demanded where required."""
        if self._meta.function_kind != REQUIRED_FUNCTION_KIND:
            raise _fail(
                "D2",
                self._where,
                "function_kind",
                f"{REQUIRED_FUNCTION_KIND!r} for augmentation use; found "
                f"{self._meta.function_kind!r} (identified, never relabeled "
                "or converted)",
            )
        return self

    # -- slab access ---------------------------------------------------------

    def band_slab(self, source_k, physical_band):
        """Read one original physical-band ID within the declared coverage."""
        _channel, nk, _nband, _ncomp, _nproj = self._shape
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (source_k, physical_band)
        ):
            raise _fail(
                "D6",
                self._where,
                "source_k/physical_band",
                "integral original state identifiers, never truncated",
            )
        ik, band = int(source_k), int(physical_band)
        b0, b1 = self._meta.band_range
        if not (0 <= ik < nk):
            raise _fail(
                "D6", self._where, f"source_k={ik}", f"a source k index in [0, {nk})"
            )
        if not (b0 <= band < b1):
            raise _fail(
                "D6",
                self._where,
                f"physical_band={band}",
                f"an original physical band in [{b0}, {b1})",
            )
        return self._slab_reader(0, ik, band - b0)


class _MemorySlabs:
    def __init__(self, arr):
        self._arr = arr

    def __call__(self, channel, ik, ib):
        slab = self._arr[channel, ik, ib]
        view = slab.view()
        view.flags.writeable = False
        return view


class SuppliedCoefficientArrays:
    """Read-only caller-memory carrier over the canonical v1 axes."""

    def __init__(
        self,
        coefficients,
        meta: SupplierMeta,
        header=None,
        *,
        symmetry_map_reference=None,
    ):
        where = "<caller-memory coefficients>"
        arr = np.asarray(coefficients)
        # D9: the carrier requires contiguous complex128 caller arrays
        if arr.dtype != np.complex128:
            raise _fail(
                "D9",
                where,
                "coefficients",
                "complex128 caller arrays, found " f"dtype {arr.dtype}",
            )
        if not arr.flags.c_contiguous:
            raise _fail(
                "D9",
                where,
                "coefficients",
                "a C-contiguous caller array; non-contiguous views "
                "are rejected rather than silently copied",
            )
        view = arr.view()
        view.flags.writeable = False
        _validate_meta(
            meta,
            header,
            view.shape,
            where,
            symmetry_map_reference=symmetry_map_reference,
        )
        self._coefficients = view
        self._meta = meta
        self._header = header
        self._record = ValidatedCoefficientRecord(
            meta, header, view.shape, _MemorySlabs(view), where, symmetry_map_reference
        )

    @property
    def record(self) -> ValidatedCoefficientRecord:
        return self._record

    @property
    def coefficients(self):
        """The read-only stored view (write attempts fail closed, D9)."""
        return self._coefficients


# ---------------------------------------------------------------------------
# NetCDF file carrier
# ---------------------------------------------------------------------------

_COEFF_VAR = "coefficients"


def write_netcdf_carrier(path, coefficients, meta: SupplierMeta):
    """Author a v1 NetCDF carrier (declared-schema fixture/producer helper).

    The file layout is the declared consumer contract: attributes carry the
    schema and declarations, variables carry the per-record mappings, and
    ``coefficients`` has dimensions (stored_channel, source_k,
    physical_band, spinor_component, total_projector, complex2) in f8.
    """
    import netCDF4

    arr = np.asarray(coefficients)
    if arr.dtype != np.complex128 or arr.ndim != 5:
        raise _fail(
            "D9",
            str(path),
            "coefficients",
            "complex128 with the five canonical v1 axes, found "
            f"dtype {arr.dtype} ndim {arr.ndim}",
        )
    path = Path(path)
    _validate_meta(
        meta,
        None,
        arr.shape,
        str(path),
        symmetry_map_reference=meta.symmetry_map_reference,
    )
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        for name, dim in (
            ("stored_channel", arr.shape[0]),
            ("source_k", arr.shape[1]),
            ("physical_band", arr.shape[2]),
            ("spinor_component", arr.shape[3]),
            ("total_projector", arr.shape[4]),
            ("complex2", 2),
            ("three", 3),
            ("two", 2),
            ("natom", len(meta.atom_species)),
            ("npsp", len(meta.dataset_md5)),
        ):
            ds.createDimension(name, dim)
        ds.setncattr("schema_name", meta.schema_name)
        ds.setncattr("schema_version", np.int32(meta.schema_version))
        ds.setncattr("function_kind", meta.function_kind)
        ds.setncattr("state_frame", meta.state_frame)
        if meta.symmetry_map_reference is not None:
            ds.setncattr("symmetry_map_reference", meta.symmetry_map_reference)
        ds.setncattr("source_channel", np.int32(meta.source_channel))
        ds.setncattr("conjugation", meta.conjugation)
        ds.setncattr("raw_normalization", np.int32(bool(meta.raw_normalization)))
        ds.setncattr("harmonic_convention", meta.harmonic_convention)
        ds.setncattr("phase_declaration", meta.phase_declaration)
        if meta.producer:
            ds.setncattr("producer", meta.producer)
        ds.createVariable("kpoints", "f8", ("source_k", "three"))[:] = np.asarray(
            meta.kpoints, dtype=float
        )
        ds.createVariable("spin_tuple", "i4", ("three",))[:] = np.array(
            meta.spin_tuple, np.int32
        )
        ds.createVariable("band_range", "i4", ("two",))[:] = np.array(
            meta.band_range, np.int32
        )
        if meta.dataset_md5:
            v = ds.createVariable("dataset_md5", str, ("npsp",))
            for i, m in enumerate(meta.dataset_md5):
                v[i] = str(m)
        v = ds.createVariable("atom_species", str, ("natom",))
        for i, s in enumerate(meta.atom_species):
            v[i] = str(s)
        ds.createVariable("projector_counts", "i4", ("natom",))[:] = np.asarray(
            meta.projector_counts, np.int32
        )
        ds.createDimension("projector_label_fields", 4)
        ds.createVariable(
            "projector_labels", "i4", ("total_projector", "projector_label_fields")
        )[:] = np.asarray(meta.projector_labels, dtype=np.int32)
        var = ds.createVariable(
            _COEFF_VAR,
            "f8",
            (
                "stored_channel",
                "source_k",
                "physical_band",
                "spinor_component",
                "total_projector",
                "complex2",
            ),
        )
        var[..., 0] = arr.real
        var[..., 1] = arr.imag
    return path


def _read_meta(ds, where) -> SupplierMeta:
    def attr(name):
        if name not in ds.ncattrs():
            return None
        return ds.getncattr(name)

    raw = attr("raw_normalization")
    if raw is not None and (
        not isinstance(raw, (int, np.integer)) or raw not in (0, 1)
    ):
        raise _fail("D7", where, "raw_normalization", "NetCDF integer boolean0 or1")

    meta = SupplierMeta(
        schema_name=attr("schema_name"),
        schema_version=attr("schema_version"),
        function_kind=attr("function_kind"),
        state_frame=attr("state_frame"),
        kpoints=np.array(ds["kpoints"][:], dtype=float),
        spin_tuple=tuple(np.asarray(ds["spin_tuple"][:]).ravel()),
        source_channel=attr("source_channel"),
        band_range=tuple(np.asarray(ds["band_range"][:]).ravel()),
        dataset_md5=tuple(str(x) for x in ds["dataset_md5"][:])
        if "dataset_md5" in ds.variables
        else (),
        atom_species=tuple(str(x) for x in ds["atom_species"][:]),
        projector_counts=tuple(ds["projector_counts"][:]),
        projector_labels=tuple(tuple(label) for label in ds["projector_labels"][:])
        if "projector_labels" in ds.variables
        else (),
        harmonic_convention=attr("harmonic_convention") or "",
        conjugation=attr("conjugation"),
        raw_normalization=None if raw is None else bool(raw),
        phase_declaration=attr("phase_declaration"),
        symmetry_map_reference=attr("symmetry_map_reference"),
        producer=attr("producer") or "",
    )
    return meta


class PawCoefficientNetcdfCarrier:
    """Slab-readable v1 NetCDF carrier.

    Only requested single-band hyperslabs are read; the full coefficient
    tensor is never materialized by this carrier.
    """

    def __init__(self, path, header=None, *, symmetry_map_reference=None):
        try:
            import netCDF4
        except ImportError as exc:  # pragma: no cover
            raise PawCoefficientError(
                "D1",
                path,
                "netCDF4",
                f"the NetCDF carrier requires the netCDF4 python package: {exc}",
            ) from exc
        self._path = Path(path)
        where = str(self._path)
        self._ds = netCDF4.Dataset(str(path), "r")
        try:
            meta = _read_meta(self._ds, where)
        except KeyError as exc:
            self._ds.close()
            raise _fail(
                "D1", where, exc.args[0], "a required v1 metadata variable"
            ) from exc
        except Exception:
            self._ds.close()
            raise
        try:
            axes = (
                "stored_channel",
                "source_k",
                "physical_band",
                "spinor_component",
                "total_projector",
                "complex2",
            )
            variable = self._ds[_COEFF_VAR]
            if variable.dimensions != axes or variable.shape[-1] != 2:
                raise _fail(
                    "D5",
                    where,
                    "coefficients",
                    f"canonical dimensions{axes} with complex2 length2; found{variable.dimensions},shape{variable.shape}",
                )
            if variable.dtype != np.dtype("float64"):
                raise _fail(
                    "D9",
                    where,
                    "coefficients",
                    f"float64 real/imag pairs; found{variable.dtype}",
                )
            shape = variable.shape[:-1]
            _validate_meta(
                meta,
                header,
                shape,
                where,
                symmetry_map_reference=symmetry_map_reference,
            )
            self._meta, self._header = meta, header
            self._record = ValidatedCoefficientRecord(
                meta, header, shape, self._read_slab, where, symmetry_map_reference
            )
        except Exception:
            self._ds.close()
            raise

    def _read_slab(self, channel, ik, ib):
        var = self._ds[_COEFF_VAR]
        var.set_auto_mask(False)
        pairs = np.asarray(var[channel, ik, ib, :, :], dtype=np.float64)
        slab = pairs.view(np.complex128)[..., 0]
        slab.flags.writeable = False
        return slab

    @property
    def path(self):
        return str(self._path)

    @property
    def record(self) -> ValidatedCoefficientRecord:
        return self._record

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# anti-shortcut and witness seam
# ---------------------------------------------------------------------------


def require_dataset_identity(meta_or_record):
    """Derived-kernel gate (research F6 'not a silent NC shortcut').

    Any derived kernel (Delta S, Q(b), one-center trial integrals) demands
    the matched per-species dataset identity; supplied coefficients alone
    never waive it.
    """
    meta = getattr(meta_or_record, "meta", meta_or_record)
    if not meta.dataset_md5:
        raise _fail(
            "D4",
            "<supplied coefficients>",
            "dataset_md5",
            "matched per-species PAW dataset fingerprints for any derived "
            "kernel (Delta S, Q(b), one-center trial integrals); supplied "
            "coefficients alone never waive the operator data",
        )
    return tuple(str(x) for x in meta.dataset_md5)


def require_dualprojector(meta_or_record):
    """D2: dual-projector kind demanded wherever the operator requires it;
    other declared kinds are readable data but are never relabeled."""
    kind = getattr(meta_or_record, "function_kind", None)
    if kind is None:
        kind = meta_or_record.meta.function_kind
    if kind != REQUIRED_FUNCTION_KIND:
        raise _fail(
            "D2",
            "<supplied coefficients>",
            "function_kind",
            f"{REQUIRED_FUNCTION_KIND!r}; found {kind!r} (a different PAW "
            "function object: identified, never relabeled or converted)",
        )
    return kind


def witness_compare(record, reference, rtol=WITNESS_RTOL):
    """Supplied-versus-computed witness (Architecture §9 same-source gate).

    Compares every stored pair against an independently computed reference;
    returns the max relative deviation.  Any single-element disagreement
    beyond ``rtol`` fails closed naming the element, expected and actual.
    """
    record._witness_residual = None
    if not np.isfinite(rtol) or not 0 <= rtol <= WITNESS_RTOL:
        raise _fail("D4", "<witness>", "rtol", f"finite 0<=rtol<={WITNESS_RTOL}")
    ref = np.asarray(reference)
    if ref.shape != record.shape:
        raise _fail(
            "D5",
            "<witness>",
            "reference",
            f"reference shaped {record.shape} like the record, " f"found {ref.shape}",
        )
    max_error, max_reference = 0.0, 0.0
    worst_index, worst_expected, worst_actual = None, None, None
    band_start = record.meta.band_range[0]
    for ch in range(record.shape[0]):
        for ik in range(record.shape[1]):
            for ib in range(record.shape[2]):
                expected = ref[ch, ik, ib]
                actual = record.band_slab(ik, band_start + ib)
                for field, values in (
                    ("reference", expected),
                    ("coefficients", actual),
                ):
                    invalid = ~np.isfinite(values)
                    if invalid.any():
                        local = tuple(int(value) for value in np.argwhere(invalid)[0])
                        raise _fail(
                            "D2",
                            "<witness>",
                            f"{field}{(ch, ik, band_start+ib, *local)}",
                            "finite complex overlap values",
                        )
                max_reference = max(max_reference, float(np.max(np.abs(expected))))
                difference = np.abs(actual - expected)
                flat_index = int(np.argmax(difference))
                error = float(difference.flat[flat_index])
                if error > max_error:
                    local = np.unravel_index(flat_index, expected.shape)
                    max_error = error
                    worst_index = (ch, ik, band_start + ib, *local)
                    worst_expected, worst_actual = expected[local], actual[local]
    deviation = (
        max_error / max_reference
        if max_reference
        else (0.0 if max_error == 0.0 else float("inf"))
    )
    if deviation > rtol:
        raise _fail(
            "D2",
            "<witness>",
            f"coefficients{tuple(int(i) for i in worst_index)}",
            f"agreement with the independently computed reference within relative {rtol}; "
            f"found {deviation:.3e} (expected {worst_expected!r}, actual {worst_actual!r})",
        )
    record._witness_residual = deviation
    return deviation
