# SPDX-License-Identifier: BSD-2-Clause
"""Story-064 NC producer: WIN plan, NNKP validation, streamed A/M/E.

``AbinitWannierInputs`` turns a story-060/063 source (WfkReader or
CallerArraySource — anything exposing the shared ``header`` / ``kg`` /
``coefficients`` / ``eigenvalues`` / ``decode_full_sphere`` seam) plus a
``GenerationPlan`` into wannier90 input files:

- ``write_win`` writes the .win plan preserving the source geometry at
  original precision (source bohr rows x the pinned CODATA2022 Bohr
  constant), ``num_bands`` = retained count with NO ``exclude_bands`` and an
  explicit kpoints block in generation order;
- ``write_matrices`` accepts the external ``-pp`` NNKP (parsed, validated
  against the plan within its documented f12.7 rounding bounds; the integer
  neighbour keys/order are authoritative) and streams physically contracted
  AMN/MMN/EIG records through :func:`lawaf.io.w90.write_amn_blocks`,
  :func:`lawaf.io.w90.write_mmn_blocks` and
  :func:`lawaf.io.w90.write_eig_blocks`.

Physical conventions (Architecture section 4.2, NC contract memo F2/F5/F7):

- ``A = Psi^dagger G`` — state bra, trial ket;
- ``M(k, b) = sum_G conj(c1(G)) c2(G + g1)`` over the true transport match
  ``b = k' + g - k``: the paired integer index is offset by the image shift
  carried by the NNKP ``nnkpts`` block header, exactly wannier90's
  overlap.F90 pairing;
- spin-diagonal scalar contraction (nspinor collapsed by summation).

Bounded memory: one projection tile (one k) plus one pair tile (one
(k, neighbour)) live at a time; no dense mesh M and no U/Uopt is ever
allocated.  Fail-closed diagnostics D1-D7 (story 064) name file/field and
expectation; no partially valid file set is left behind.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lawaf.io.w90 import (
    write_amn_blocks,
    write_eig_blocks,
    write_mmn_blocks,
    write_win as _write_win,
)

# Pinned units (Architecture section 4.1; NIST 2022 = wannier90 4.x
# constants.F90 bohr_angstrom_internal).
BOHR_ANG = 0.529177210544
HA_EV = 27.211386245981

# Documented NNKP print rounding bound: wannier90's kmesh_write prints the
# lattice and fractional points at 12.7 (f12.7), i.e. a half-unit rounding
# error of 5e-8; validation accepts up to that bound plus float-parse slack.
NNKP_PRINT_TOL = 5.1e-8


class ProducerError(Exception):
    """Fail-closed producer diagnostic naming file, field and expectation."""

    def __init__(self, diagnostic, file, field, detail):
        self.diagnostic = diagnostic
        self.file = str(file)
        self.field = field
        self.detail = detail
        super().__init__(f"[{diagnostic}] file={file} field={field!r}: " f"{detail}")


def _fail(diagnostic, file, field, detail="") -> ProducerError:
    return ProducerError(diagnostic, file, field, detail)


# ---------------------------------------------------------------------------
# NNKP parsing (external -pp output; wannier90 kmesh_write grammar)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nnkp:
    """The parsed -pp NNKP; printed geometry kept verbatim for validation."""

    path: str
    real_lattice: np.ndarray  # (3, 3) Angstrom, rows, f12.7 prints
    recip_lattice: np.ndarray  # (3, 3) 2pi/A
    kpoints: np.ndarray  # (nk, 3) fractional, printed
    nnkpts: np.ndarray  # (nk, nntot, 4) int: (k_prime0, g1, g2, g3) 0-based
    has_projections: bool
    spinor_projections: bool
    exclude_band_count: int


def parse_nnkp(path) -> Nnkp:
    """Parse a wannier90 ``-pp`` NNKP file (fail-closed D1)."""
    path = Path(path)
    if not path.exists():
        raise _fail(
            "D1",
            path,
            "nnkp file",
            "the preprocessing output is missing: run the pinned "
            "external executable as '<wannier90.x> -pp <seed>' "
            "before write_matrices",
        )
    lines = path.read_text().splitlines()

    def section(name):
        for candidate in (
            [name]
            if name != "fractional kpoints"
            else ["fractional kpoints", "kpoints"]
        ):
            if f"begin {candidate}" in lines:
                i0 = lines.index(f"begin {candidate}") + 1
                i1 = lines.index(f"end {candidate}", i0)
                return lines[i0:i1]
        raise _fail(
            "D1",
            path,
            f"section {name!r}",
            "required NNKP section absent — the file is not a "
            "preprocessing (-pp) output",
        ) from None

    def try_section(name):
        try:
            return section(name)
        except ProducerError:
            return []

    real = np.array([[float(x) for x in ln.split()] for ln in section("real_lattice")])
    if real.shape != (3, 3):
        raise _fail(
            "D3", path, "real_lattice", f"3 rows x 3 columns, got shape {real.shape}"
        )
    recip = np.array(
        [[float(x) for x in ln.split()] for ln in section("recip_lattice")]
    )
    if recip.shape != (3, 3):
        raise _fail(
            "D3", path, "recip_lattice", f"3 rows x 3 columns, got shape {recip.shape}"
        )
    kbody = section("fractional kpoints")
    if len(kbody[0].split()) == 1:
        kcount = int(kbody[0].split()[0])
        kbody = kbody[1:]
    else:
        kcount = None
    kpts = np.array([[float(x) for x in ln.split()] for ln in kbody])
    if kpts.ndim != 2 or kpts.shape[1] != 3 or kpts.shape[0] == 0:
        raise _fail(
            "D3", path, "fractional kpoints", f"(nk, 3) rows, got shape {kpts.shape}"
        )
    if kcount is not None and kcount != kpts.shape[0]:
        raise _fail(
            "D3",
            path,
            "fractional kpoints count",
            f"{kcount} declared, {kpts.shape[0]} rows present",
        )
    nk = kpts.shape[0]
    nn_rows = section("nnkpts")
    if not nn_rows:
        raise _fail("D2", path, "nnkpts", "empty nnkpts block")
    # two dialects: an optional single-integer nntot line followed by flat
    # rows "k k' G1 G2 G3", or per-point header "k nntot" + rows "k' G1 G2G3"
    widths = {len(ln.split()) for ln in nn_rows}
    declared_nntot = None
    if widths <= {1, 5} and len(nn_rows[0].split()) == 1:
        declared_nntot = int(nn_rows[0].split()[0])
        nn_rows = nn_rows[1:]
        widths = {len(ln.split()) for ln in nn_rows}
    parsed = []
    for ln in nn_rows:
        t = ln.split()
        if len(t) != 5:
            raise _fail(
                "D2",
                path,
                "nnkpts",
                f"5 integer fields per row (k k' G1 G2 G3), " f"got {len(t)}: {ln!r}",
            )
        parsed.append([int(x) for x in t])
    nn = np.asarray(parsed, dtype=int).reshape(len(parsed), 5)
    nntot = len(parsed) // nk
    if declared_nntot is not None and declared_nntot != nntot:
        raise _fail(
            "D2",
            path,
            "nnkpts",
            f"declared nntot {declared_nntot} vs {nntot} rows "
            f"per point for nk={nk}",
        )
    if len(parsed) != nk * nntot:
        raise _fail(
            "D2",
            path,
            "nnkpts",
            f"row count {len(parsed)} is not nk*nntot for " f"nk={nk}",
        )
    nnkpts = np.empty((nk, nntot, 4), dtype=int)
    for ik in range(nk):
        block = nn[ik * nntot : (ik + 1) * nntot]
        if np.any(block[:, 0] != ik + 1):
            raise _fail(
                "D2",
                path,
                "nnkpts",
                f"block k index sequence for point {ik + 1}: rows "
                f"{ik * nntot + 1}..{(ik + 1) * nntot} carry k indices "
                f"{block[:, 0].tolist()}",
            )
        keys = block[:, 1:].tolist()
        if len(keys) != len({tuple(r) for r in keys}):
            raise _fail(
                "D2",
                path,
                "nnkpts",
                f"unique (k', G) neighbours for point {ik + 1}: duplicate "
                f"neighbour key present",
            )
        nnkpts[ik] = block[:, 1:] - np.array([1, 0, 0, 0])
    has_proj = bool(
        try_section("projection")
        or try_section("projections")
        or try_section("spinor_projections")
    )
    excl = 0
    excl_body = try_section("exclude_bands")
    if excl_body:
        excl = int(excl_body[0].split()[0])
    return Nnkp(
        path=str(path),
        real_lattice=real,
        recip_lattice=recip,
        kpoints=kpts,
        nnkpts=nnkpts,
        has_projections=has_proj,
        spinor_projections=bool(try_section("spinor_projections")),
        exclude_band_count=excl,
    )


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationPlan:
    """The band map / window / localization plan (counts live HERE, never
    in the NNKP).  ``band_map`` holds the retained ORIGINAL 0-based band
    indices, ascending and unique; exclusions are applied exactly once by
    the producer and recorded in provenance only."""

    band_map: Tuple[int, ...]
    num_wann: int
    mp_grid: Tuple[int, int, int]
    controls: Mapping[str, str] = field(default_factory=dict)
    kmesh_tol: float = 1e-5
    projections: Optional[Sequence[str]] = None
    orthogonalize: bool = False
    source_channel: Optional[int] = None
    spinors: Optional[bool] = None
    scratch_budget_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if (
            isinstance(self.scratch_budget_bytes, (bool, np.bool_))
            or not isinstance(self.scratch_budget_bytes, (int, np.integer))
            or self.scratch_budget_bytes <= 0
        ):
            raise _fail("D8", "<plan>", "scratch_budget_bytes", "positive integral bytes")
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))
        object.__setattr__(self, "mp_grid", tuple(self.mp_grid))
        if self.projections is not None:
            object.__setattr__(self, "projections", tuple(self.projections))
        if any(
            isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer))
            for b in self.band_map
        ):
            raise _fail(
                "D4",
                "<plan>",
                "band_map",
                "integral original physical-band IDs, never truncated",
            )
        bm = tuple(int(b) for b in self.band_map)
        object.__setattr__(self, "band_map", bm)
        if not bm or any(b < 0 for b in bm):
            raise _fail(
                "D4",
                "<plan>",
                "band_map",
                "a non-empty ascending selection of non-negative "
                f"original band indices, got {bm}",
            )
        if list(bm) != sorted(set(bm)):
            raise _fail(
                "D5" if self.spinors else "D4",
                "<plan>",
                "band_map",
                "strictly ascending unique indices, got " f"{bm}",
            )
        if int(self.num_wann) <= 0 or len(bm) < int(self.num_wann):
            raise _fail(
                "D4",
                "<plan>",
                "num_wann vs retained bands",
                f"num_wann <= {len(bm)} retained bands, got "
                f"num_wann={self.num_wann}",
            )
        if len(tuple(self.mp_grid)) != 3 or min(self.mp_grid) < 1:
            raise _fail(
                "D4", "<plan>", "mp_grid", "a 3-vector >= 1, got " f"{self.mp_grid}"
            )


@dataclass(frozen=True)
class WriteManifest:
    """What write_matrices produced: files, measures, provenance."""

    files: Dict[str, str]
    bytes: Dict[str, int]
    seconds: Dict[str, float]
    band_map: Tuple[int, ...]
    neighbor_count: int
    nnkp_sha256: str
    units: Dict[str, str]
    provenance: Dict[str, str]


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


class AbinitWannierInputs:
    """WIN/AMN/MMN/EIG producer over a story-060/063 source.

    :param source: file-backed (WfkReader) or caller-array
        (CallerArraySource) carrier exposing the shared seam;
    :param plan: the :class:`GenerationPlan`;
    :param trial_orbitals: callable ``(k_index, kg_full) -> (N, num_wann)``
        complex trial matrix G over the k-point's full-sphere integer G
        vectors (columns = wannier trial orbitals). Alternatively supply
        ``trial_manifest`` for physical numerical UPF trial evaluation.
    """

    def __init__(
        self,
        source,
        plan: GenerationPlan,
        trial_orbitals: Optional[Callable] = None,
        *,
        trial_manifest=None,
        paw_coefficients=None,
    ):
        self._source = source
        self._plan = plan
        self._trial_orbitals = trial_orbitals
        self._header = source.header
        self._paw_provider = None
        self._paw_metric = None
        self._paw_trials = None
        h = self._header
        if h.usepaw not in (0, 1) or np.asarray(h.symafm).shape != (h.nsym,):
            raise _fail(
                "D6",
                source.path,
                "spin/operator metadata",
                "complete usepaw and magnetic symmetry flags",
            )
        spin_tuple = (h.nsppol, h.nspinor, h.nspden)
        if spin_tuple == (1, 1, 2):
            raise _fail(
                "D4",
                source.path,
                "AFM channel/symmetry map",
                "explicit reconstructed two-channel source required; no duplicated up states",
            )
        if spin_tuple not in ((1, 1, 1), (2, 1, 2), (1, 2, 4)):
            raise _fail(
                "D6",
                source.path,
                "spin representation",
                f"supported complete tuple; found {spin_tuple}",
            )
        self._spinors = h.nspinor == 2
        if self._spinors and np.any(np.asarray(h.istwfk) != 1):
            point = int(np.flatnonzero(np.asarray(h.istwfk) != 1)[0])
            raise _fail(
                "D3",
                source.path,
                f"istwfk[{point}]",
                "coherent spinors require full complex storage1",
            )
        if plan.spinors is not None and plan.spinors != self._spinors:
            raise _fail(
                "D6", source.path, "plan spinors", f"source spin tuple {spin_tuple}"
            )
        declared_spinors = plan.controls.get("spinors")
        if (
            declared_spinors is not None
            and str(declared_spinors).lower().strip(".") != str(self._spinors).lower()
        ):
            raise _fail(
                "D6", source.path, "controls spinors", f"source spin tuple {spin_tuple}"
            )
        channel = plan.source_channel
        if channel is None and h.nsppol == 1:
            channel = 0
        if type(channel) is not int or not 0 <= channel < h.nsppol:
            raise _fail(
                "D1",
                source.path,
                "source_channel",
                f"explicit source channel0..{h.nsppol-1}; found {channel!r}, tuple {spin_tuple}",
            )
        self._channel = channel
        self._spin_trial_columns = None
        self._spin_trial_components = None
        if self._header.usepaw and trial_manifest is None:
            raise _fail(
                "D1",
                source.path,
                "PAW trial/operator data",
                "matched XML trial_manifest required; supplied coefficients do not waive operators",
            )
        if not self._header.usepaw and paw_coefficients is not None:
            raise _fail(
                "D4", source.path, "PAW coefficient record", "usepaw=1 source required"
            )
        if len(plan.band_map) > int(self._header.mband):
            raise _fail(
                "D4",
                source.path,
                "plan band_map vs source mband",
                f"retained count {len(plan.band_map)} <= mband "
                f"{int(self._header.mband)}",
            )
        for ik in range(int(self._header.nkpt)):
            if max(plan.band_map) >= int(
                self._header.nband[ik + self._channel * self._header.nkpt]
            ):
                raise _fail(
                    "D4",
                    source.path,
                    f"plan band_map vs nband at k={ik}",
                    f"max retained index {max(plan.band_map)} < nband "
                    f"{int(self._header.nband[ik + self._channel * self._header.nkpt])}",
                )
        if trial_manifest is not None:
            from lawaf.interfaces.abinit.nc_trials import NcTrialBasis
            from lawaf.interfaces.abinit.trial_manifest import (
                TrialManifest,
                parse_manifest,
            )

            if trial_orbitals is not None:
                raise _fail(
                    "D4",
                    source.path,
                    "trial definition",
                    "either trial_manifest or trial_orbitals, not both",
                )
            manifest = (
                trial_manifest
                if isinstance(trial_manifest, TrialManifest)
                else parse_manifest(trial_manifest)
            )
            if len(manifest.trial_records) != plan.num_wann:
                raise _fail(
                    "D4",
                    manifest.path,
                    "trial count",
                    f"{plan.num_wann} trials, got {len(manifest.trial_records)}",
                )
            if self._spinors:
                spatial = []
                indices = {}
                columns, components = [], []
                for record in manifest.trial_records:
                    if record.spin_label not in ("up", "down"):
                        raise _fail(
                            "D6",
                            manifest.path,
                            "spin trial labels",
                            "explicit up/down frame for every coherent trial",
                        )
                    key = (
                        record.site_index,
                        record.species,
                        record.radial_id,
                        record.harmonic_index,
                    )
                    if key not in indices:
                        indices[key] = len(spatial)
                        spatial.append(replace(record, spin_label=None))
                    columns.append(indices[key])
                    components.append(0 if record.spin_label == "up" else 1)
                self._spin_trial_columns = np.asarray(columns)
                self._spin_trial_components = np.asarray(components)
                manifest = replace(
                    manifest, trial_records=tuple(spatial), spin_frame_labels=False
                )
            if self._header.usepaw:
                from ase.data import chemical_symbols
                from lawaf.interfaces.abinit.paw_provider import PawCoefficientProvider
                from lawaf.interfaces.abinit.paw_trials import PawTrialBasis
                from lawaf.interfaces.abinit.physics_paw import PawOverlap

                species = [chemical_symbols[int(z)] for z in self._header.znucltypat]
                missing = [
                    name for name in species if name not in manifest.dataset_paths
                ]
                if missing:
                    raise _fail(
                        "D1",
                        manifest.path,
                        "species XML references",
                        f"missing source species{missing}",
                    )
                paths = [manifest.dataset_paths[name] for name in species]
                self._paw_provider = PawCoefficientProvider(
                    source, paths, supplied_record=paw_coefficients
                )
                if paw_coefficients is not None:
                    paw_coefficients.require_witness()
                self._paw_trials = PawTrialBasis(manifest, self._paw_provider)
                self._paw_metric = PawOverlap(self._paw_provider)
                self._trial_orbitals = self._paw_trials
            else:
                self._trial_orbitals = NcTrialBasis(manifest, self._header)

    # -- geometry ----------------------------------------------------------

    @property
    def header(self):
        return self._header

    @property
    def plan(self) -> GenerationPlan:
        return self._plan

    def lattice_ang(self) -> np.ndarray:
        """Source bohr rows x the pinned Bohr constant, full precision."""
        return np.asarray(self._header.rprimd, dtype=float) * BOHR_ANG

    def _symbols(self):
        from ase.data import chemical_symbols

        species = [chemical_symbols[int(z)] for z in self._header.znucltypat]
        return [species[int(index) - 1] for index in self._header.typat]

    # -- WIN ---------------------------------------------------------------

    def write_win(self, folder, seed) -> Path:
        """Write the .win plan: source geometry at original precision,
        ``num_bands`` = retained count, NO ``exclude_bands``, explicit
        kpoints in generation order, plan window/localization controls."""
        h = self._header
        path = Path(folder) / f"{seed}.win"
        extra = dict(self._plan.controls)
        if self._spinors:
            extra["spinors"] = "true"
        if "exclude_bands" in extra or any("exclude_bands" in str(k) for k in extra):
            raise _fail(
                "D4",
                path,
                "controls exclude_bands",
                "absent: the band map is applied exactly once via "
                "num_bands = retained count",
            )
        write_win_ = _write_win
        write_win_(
            path,
            num_wann=self._plan.num_wann,
            num_bands=len(self._plan.band_map),
            mp_grid=self._plan.mp_grid,
            lattice=self.lattice_ang(),
            symbols=self._symbols(),
            frac=h.xred,
            kpts=h.kpt,
            projections=self._plan.projections,
            kmesh_tol=self._plan.kmesh_tol,
            extra=extra,
        )
        return path

    # -- native-contract companion pair (story-067 option A) ---------------

    @staticmethod
    def _exclude_run_string(dropped: Sequence[int]) -> str:
        """1-based dropped indices as wannier90 run syntax: ``1-10,12``."""
        runs = []
        first = 0
        while first < len(dropped):
            stop = first
            while (
                stop + 1 < len(dropped) and dropped[stop + 1] == dropped[stop] + 1
            ):
                stop += 1
            runs.append(
                str(dropped[first])
                if stop == first
                else f"{dropped[first]}-{dropped[stop]}"
            )
            first = stop + 1
        return ",".join(runs)

    def write_native_companion(self, folder, seed) -> Tuple[Path, Path]:
        """Write the native-contract companion ``win``/``eig`` pair.

        Story-067 option A (user-approved): the external pair applies the
        band map exactly once via ``num_bands = retained`` (design decision
        D4); the native route instead reads the FULL source frame, so this
        companion declares ``num_bands = mband`` plus ``exclude_bands``
        naming the dropped 1-based bands, and streams the full-frame EIG
        (Hartree -> eV).  It is an ADDITIONAL output; the external pair and
        the D4 control guard are untouched.
        """
        h = self._header
        mband = int(h.mband)
        folder = Path(folder)
        retained = {int(b) + 1 for b in self._plan.band_map}
        if not retained <= set(range(1, mband + 1)):
            raise _fail(
                "D4",
                folder / f"{seed}-native.win",
                "plan band_map vs source mband",
                f"retained indices {sorted(retained)} must be within the "
                f"1-based source frame 1..{mband}",
            )
        dropped = sorted(set(range(1, mband + 1)) - retained)
        extra = dict(self._plan.controls)
        if self._spinors:
            extra["spinors"] = "true"
        if dropped:
            extra["exclude_bands"] = self._exclude_run_string(dropped)
        win_path = folder / f"{seed}-native.win"
        _write_win(
            win_path,
            num_wann=self._plan.num_wann,
            num_bands=mband,
            mp_grid=self._plan.mp_grid,
            lattice=self.lattice_ang(),
            symbols=self._symbols(),
            frac=h.xred,
            kpts=h.kpt,
            projections=self._plan.projections,
            kmesh_tol=self._plan.kmesh_tol,
            extra=extra,
        )
        eig_path = folder / f"{seed}-native.eig"
        nk = int(h.nkpt)
        part = folder / f".{seed}-native.eig.partial"

        def eig_blocks():
            for ik in range(nk):
                ev = np.asarray(
                    self._source.eigenvalues(ik, self._channel, (0, mband))
                )
                if ev.shape != (mband,):
                    raise _fail(
                        "D5",
                        self._source.path,
                        "full-frame eigenvalues",
                        f"one energy per band at k={ik}: got shape "
                        f"{ev.shape}, expected ({mband},)",
                    )
                if not np.all(np.isfinite(ev)):
                    raise _fail(
                        "D7",
                        self._source.path,
                        f"eigenvalues at k={ik}",
                        "finite Hartree eigenvalues",
                    )
                yield ik, ev * HA_EV

        write_eig_blocks(part, nk, eig_blocks())
        part.replace(eig_path)
        return win_path, eig_path

    # -- NNKP validation ---------------------------------------------------

    def validate_nnkp(self, nnkp_path) -> Nnkp:
        """Parse + validate the external NNKP against the plan.

        The printed integer neighbour keys/order are authoritative for the
        mesh/neighbour table; the printed 7-decimal lattice and fractional
        points are validated only within the documented f12.7 rounding
        bounds.  Band/Wannier counts come from the plan, never the NNKP.
        """
        path = Path(nnkp_path)
        nnkp = parse_nnkp(path)  # D1 inside
        h = self._header
        # D3: printed geometry vs plan within the documented rounding bound
        plan_lat = self.lattice_ang()
        if not np.allclose(nnkp.real_lattice, plan_lat, atol=NNKP_PRINT_TOL, rtol=0):
            raise _fail(
                "D3",
                path,
                "real_lattice",
                f"printed lattice within {NNKP_PRINT_TOL:.1e} Ang of the "
                f"plan (source bohr x {BOHR_ANG}); max deviation "
                f"{np.abs(nnkp.real_lattice - plan_lat).max():.3e}",
            )
        if nnkp.kpoints.shape != h.kpt.shape or not np.allclose(
            nnkp.kpoints, np.asarray(h.kpt), atol=NNKP_PRINT_TOL, rtol=0
        ):
            raise _fail(
                "D3",
                path,
                "fractional kpoints",
                f"printed k-points within {NNKP_PRINT_TOL:.1e} of the plan "
                f"mesh of {int(h.nkpt)} points",
            )
        # D2/D4: neighbour table vs plan
        if nnkp.nnkpts.shape[0] != int(h.nkpt):
            raise _fail(
                "D4",
                path,
                "nnkpts k count vs plan",
                f"{int(h.nkpt)} plan mesh points, NNKP carries "
                f"{nnkp.nnkpts.shape[0]}",
            )
        jk = nnkp.nnkpts[:, :, 0]
        if np.any(jk < 0) or np.any(jk >= int(h.nkpt)):
            raise _fail(
                "D2", path, "nnkpts neighbour index", f"0 <= k' < {int(h.nkpt)}"
            )
        if nnkp.exclude_band_count:
            raise _fail(
                "D4",
                path,
                "exclude_bands",
                "empty: the plan applies the band map once via num_bands; "
                f"NNKP carries {nnkp.exclude_band_count} excluded bands",
            )
        if not nnkp.has_projections:
            raise _fail(
                "D4",
                path,
                "projection",
                "a projections (or spinor_projections) block: the -pp run "
                "must see the same projection set the plan writes",
            )
        if nnkp.spinor_projections != self._spinors:
            raise _fail(
                "D6",
                path,
                "spinor_projections",
                "external preprocessing representation must match source spin tuple",
            )
        return nnkp

    # -- contraction -------------------------------------------------------

    @staticmethod
    def _band_runs(bands):
        """Contiguous source intervals, bounded even across sparse band IDs."""
        first = 0
        while first < len(bands):
            stop = first + 1
            while stop < len(bands) and stop - first < 32 and bands[stop] == bands[stop-1] + 1:
                stop += 1
            yield first, stop, (bands[first], bands[stop-1] + 1)
            first = stop

    def _sphere(self, ik, bands=None):
        """Bounded reads of selected physical bands in component-major order."""
        bands = self._plan.band_map if bands is None else tuple(bands)
        count_bound = int(self._header.npwarr[ik]) * self._header.nspinor * (2 if self._header.istwfk[ik] > 1 else 1)
        required = 16 * count_bound * (len(bands) + 3 * min(32, len(bands)))
        if required > self._plan.scratch_budget_bytes:
            raise _fail("D8", self._source.path, "producer scratch budget",
                        f"decoded point needs{required} bytes; cap{self._plan.scratch_budget_bytes}; request smaller band tiles")
        field = None
        for first, stop, interval in self._band_runs(bands):
            kg, _weights, states = self._source.decode_full_sphere(ik, self._channel, interval)
            if field is None:
                kg = np.asarray(kg).T
                order = np.lexsort((kg[:, 2], kg[:, 1], kg[:, 0]))
                field = np.empty((self._header.nspinor, len(kg), len(bands)), complex)
                ordered_g = kg[order]
            field[:, :, first:stop] = states.transpose(2, 1, 0)[:, order, :]
        return ordered_g, field.reshape(-1, len(bands))

    def _trials(self, ik, kg):
        if self._trial_orbitals is None:
            raise _fail(
                "D4",
                self._source.path,
                "trial_orbitals",
                "a trial matrix supplier for the A contraction",
            )
        g = np.asarray(self._trial_orbitals(ik, kg), dtype=complex)
        if self._spinors:
            if self._spin_trial_columns is not None:
                g = self._embed_spin_trials(g)
            if g.shape != (kg.shape[0], self._plan.num_wann, 2):
                raise _fail(
                    "D6",
                    self._source.path,
                    "spinor trial matrix",
                    f"(G,trial,2), got {g.shape}; no duplicated scalar trials",
                )
            return g.transpose(2, 0, 1).reshape(-1, self._plan.num_wann)
        if g.shape != (kg.shape[0], self._plan.num_wann):
            raise _fail(
                "D4",
                self._source.path,
                "trial_orbitals",
                f"({kg.shape[0]}, {self._plan.num_wann}), got {g.shape}",
            )
        return g

    def _embed_spin_trials(self, spatial):
        result = np.zeros((len(spatial), self._plan.num_wann, 2), complex)
        result[:, np.arange(self._plan.num_wann), self._spin_trial_components] = (
            spatial[:, self._spin_trial_columns]
        )
        return result

    def _retained_cprj(self, ik, bands=None):
        bands = self._plan.band_map if bands is None else tuple(bands)
        result = np.empty((len(bands), self._header.nspinor, self._paw_provider.offsets[-1]), complex)
        for first, stop, interval in self._band_runs(bands):
            result[first:stop] = self._paw_provider.band_block(ik, self._channel, interval)
        return result

    def _projection_tile(self, ik):
        """A = Psi^dagger G (state bra / trial ket) for one k-point."""
        kg, psi = self._sphere(ik)  # (N,3), (N,nret)
        required = 3 * psi.nbytes + 3 * 16 * psi.shape[0] * self._plan.num_wann + 2 * 16 * psi.shape[1] * self._plan.num_wann
        if required > self._plan.scratch_budget_bytes:
            raise _fail("D8", self._source.path, "projection scratch budget",
                        f"projection needs{required} bytes; cap{self._plan.scratch_budget_bytes}; adjust the declared workload budget")
        g = self._trials(ik, kg)  # (N, nw)
        a = psi.conj().T @ g  # (nret, nw)
        if self._paw_metric is not None:
            spatial = self._paw_trials.dual_coefficients(ik)
            trial = (
                self._embed_spin_trials(spatial)
                if self._spinors
                else spatial[:, :, None]
            )
            a += self._paw_metric.projection(self._retained_cprj(ik), trial)
        if self._plan.orthogonalize:
            a = self._lowdin(a, ik)
        if not (np.all(np.isfinite(a.real)) and np.all(np.isfinite(a.imag))):
            bad = np.argwhere(~(np.isfinite(a.real) & np.isfinite(a.imag)))
            raise _fail(
                "D7",
                self._source.path,
                "AMN values",
                f"non-finite projection at k={ik}, band "
                f"{self._plan.band_map[int(bad[0][0])]}, trial "
                f"{int(bad[0][1])}",
            )
        return a

    def _lowdin(self, a, ik):
        gram = a.conj().T @ a
        w, v = np.linalg.eigh(gram)
        if w.min() <= 1e-12 * max(abs(w).max(), 1.0):
            raise _fail(
                "D6",
                self._source.path,
                f"projection Gram at k={ik}",
                f"full rank; min eigenvalue {w.min():.3e}",
            )
        return a @ (v @ np.diag(w**-0.5) @ v.conj().T)

    def _pair_tile(self, ik, slot, nnkp, row_bands=None, column_bands=None, left_block=None):
        """M(k, b) = sum_G conj(c1(G)) c2(G + g) over the true transport
        match b = k' + g - k: the integer index pairing is offset by the
        NNKP image shift g (overlap.F90 pairing)."""
        h = self._header
        jk = int(nnkp.nnkpts[ik, slot, 0])
        g_img = nnkp.nnkpts[ik, slot, 1:4]
        row_bands = self._plan.band_map if row_bands is None else tuple(row_bands)
        column_bands = self._plan.band_map if column_bands is None else tuple(column_bands)
        left_count = int(h.npwarr[ik]) * (2 if h.istwfk[ik] > 1 else 1)
        right_count = int(h.npwarr[jk]) * (2 if h.istwfk[jk] > 1 else 1)
        required = 16 * h.nspinor * (
            left_count * (3 * len(row_bands) + 4 * min(32, len(row_bands)))
            + right_count * (3 * len(column_bands) + 4 * min(32, len(column_bands)))
        ) + 2 * 16 * len(row_bands) * len(column_bands)
        if required > self._plan.scratch_budget_bytes:
            raise _fail("D8", self._source.path, "pair scratch budget",
                        f"pair needs{required} bytes; cap{self._plan.scratch_budget_bytes}; request smaller band tiles")
        kg1, psi1 = self._sphere(ik, row_bands) if left_block is None else left_block[:2]
        kg2, psi2 = self._sphere(jk, column_bands)

        def flat(g):
            g = np.asarray(g, dtype=np.int64) + (1 << 20)
            return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]

        targets = kg1 + g_img  # (N1, 3) integer pairing G2 = G1 + g
        pos = np.searchsorted(flat(kg2), flat(targets))
        pos_c = np.clip(pos, 0, kg2.shape[0] - 1)
        hit = flat(kg2)[pos_c] == flat(targets)
        if self._spinors:
            left = psi1.reshape(2, len(kg1), -1)[:, hit, :]
            right = psi2.reshape(2, len(kg2), -1)[:, pos_c[hit], :]
            m = np.einsum("sgm,sgn->mn", left.conj(), right)
        else:
            m = psi1[hit].conj().T @ psi2[pos_c[hit]]
        if self._paw_metric is not None:
            shift = h.kpt[jk] + g_img - h.kpt[ik]
            m += self._paw_metric.overlap(
                self._retained_cprj(ik, row_bands) if left_block is None else left_block[2],
                self._retained_cprj(jk, column_bands), shift
            )
        if not (np.all(np.isfinite(m.real)) and np.all(np.isfinite(m.imag))):
            bad = np.argwhere(~(np.isfinite(m.real) & np.isfinite(m.imag)))
            raise _fail(
                "D7",
                self._source.path,
                "MMN values",
                f"non-finite overlap at k={ik}, neighbour slot {slot}, "
                f"band {row_bands[int(bad[0][0])]} x band "
                f"{column_bands[int(bad[0][1])]}",
            )
        return m

    # -- write_matrices ----------------------------------------------------

    def write_matrices(
        self, nnkp_path, folder, seed, slab_columns: int = 0
    ) -> WriteManifest:
        """Stream ``<seed>.amn/.mmn/.eig`` into ``folder`` and return the
        file manifest + provenance (never a Hamiltonian).

        :param slab_columns: requested MMN column width; widths are capped
            at32 and contracted before allocation. Zero selects bounded
            default columns, not a full band-pair matrix.
        """
        t0 = time.perf_counter()
        nnkp = self.validate_nnkp(nnkp_path)
        if self._paw_metric is not None:
            self._paw_metric.prepare_neighbors(nnkp.nnkpts)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        h = self._header
        nret = len(self._plan.band_map)
        nk = int(h.nkpt)
        nntot = nnkp.nnkpts.shape[1]
        neighbor_table = nnkp.nnkpts  # (nk, nntot, 4): (k'0, g1, g2, g3)
        files, sizes, seconds = {}, {}, {}
        part = lambda name: folder / f".{seed}.{name}.partial"

        def commit(tmp, final):
            Path(tmp).replace(final)

        # EIG: retained eigenvalues, Hartree -> eV
        t = time.perf_counter()
        eig_path = folder / f"{seed}.eig"

        def eig_blocks():
            for ik in range(nk):
                ev = self._source.eigenvalues(
                    ik,
                    self._channel,
                    (min(self._plan.band_map), max(self._plan.band_map) + 1),
                )
                if np.asarray(ev).shape != (
                    max(self._plan.band_map) + 1 - min(self._plan.band_map),
                ):
                    raise _fail(
                        "D5",
                        self._source.path,
                        "physical-band energies",
                        "one energy per physical band, no component stacking",
                    )
                ev = np.asarray(ev)[
                    [b - min(self._plan.band_map) for b in self._plan.band_map]
                ]
                if not np.all(np.isfinite(ev)):
                    raise _fail(
                        "D7",
                        self._source.path,
                        f"eigenvalues at k={ik}",
                        "finite Hartree eigenvalues",
                    )
                yield ik, ev * HA_EV

        write_eig_blocks(part("eig"), nk, eig_blocks())
        files["eig"] = str(eig_path)
        seconds["eig"] = time.perf_counter() - t

        # AMN: one projection tile live at a time
        t = time.perf_counter()
        amn_path = folder / f"{seed}.amn"

        def amn_blocks():
            for ik in range(nk):
                yield ik, self._projection_tile(ik)

        write_amn_blocks(part("amn"), nret, nk, self._plan.num_wann, amn_blocks())
        files["amn"] = str(amn_path)
        seconds["amn"] = time.perf_counter() - t

        # MMN: one pair tile live at a time
        t = time.perf_counter()
        mmn_path = folder / f"{seed}.mmn"
        live_slabs = {"n": 0}

        def mmn_blocks():
            columns = min(slab_columns, 32) if slab_columns > 0 else min(nret, 32)
            affordable = self._plan.scratch_budget_bytes // (4 * 16 * nret)
            if affordable < 1:
                raise _fail("D8", self._source.path, "MMN slab scratch budget",
                            f"one complete column exceeds cap{self._plan.scratch_budget_bytes}")
            columns = min(columns, affordable)
            for ik in range(nk):
                # A current-point tile is reused across its neighbors; it is
                # released at the next point, never a mesh-sized state cache.
                left_block = None
                if nret <= 32:
                    left_block = (*self._sphere(ik),
                                  self._retained_cprj(ik) if self._paw_metric is not None else None)
                for slot in range(nntot):
                    for c0 in range(0, nret, columns):
                        selected = self._plan.band_map[c0:c0+columns]
                        m = np.empty((nret, len(selected)), complex)
                        for r0 in range(0, nret, 32):
                            rows = self._plan.band_map[r0:r0+32]
                            m[r0:r0+len(rows)] = self._pair_tile(ik, slot, nnkp, rows, selected, left_block)
                        live_slabs["n"] += 1
                        yield ik, slot, c0, m
                        m = None
                left_block = None

        write_mmn_blocks(part("mmn"), nret, nk, neighbor_table, mmn_blocks())
        files["mmn"] = str(mmn_path)
        seconds["mmn"] = time.perf_counter() - t
        seconds["total"] = time.perf_counter() - t0

        # atomic publication: every file streamed successfully -> commit;
        # any failure above leaves no (partially) valid file set behind
        for name, path in files.items():
            commit(part(name), path)
            sizes[name] = Path(path).stat().st_size

        return WriteManifest(
            files=files,
            bytes=sizes,
            seconds=seconds,
            band_map=tuple(self._plan.band_map),
            neighbor_count=nntot,
            nnkp_sha256=hashlib.sha256(Path(nnkp_path).read_bytes()).hexdigest(),
            units={
                "lattice": "angstrom",
                "kpoints": "fractional",
                "eig": "eV",
                "hr": "eV over A-lattice R units",
            },
            provenance={
                "source": str(self._source.path),
                "source_format": str(self._source.format),
                "source_channel": str(self._channel),
                "spin_tuple": str((h.nsppol, h.nspinor, h.nspden)),
                "band_map_original_indices": ",".join(
                    str(b) for b in self._plan.band_map
                ),
                "band_map_policy": "applied once; never exclude_bands",
                "transport": "b = k' + g - k; M pairs G by G + g",
                "a_orientation": "A = Psi^dagger G (state bra / trial ket)",
                "bohr_angstrom": repr(BOHR_ANG),
                "win_lattice_quantization": "12 decimals => <= 5e-13 Ang; NNKP prints are f12.7 "
                "lossy validation metadata only",
                "slabs_emitted": str(live_slabs["n"]),
                "scratch_budget_bytes": str(self._plan.scratch_budget_bytes),
            },
        )


# ---------------------------------------------------------------------------
# IBZ -> full-mesh expansion (Architecture section 4.4 / epic-048-c rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshMapping:
    """Native cgtk map: k_mesh = operation @ k_source + reciprocal_shift."""

    mesh_point: int
    ibz_point: int
    operation: np.ndarray  # (3, 3) int, reciprocal reduced coords
    translation: np.ndarray  # (3,) fractional
    reciprocal_shift: np.ndarray  # (3,) int
    time_reversal: bool = False
    source_channel: Optional[int] = None

    def __post_init__(self):
        for name in ("operation", "translation", "reciprocal_shift"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def map_mesh_to_ibz(
    ibz_kpts, mesh_kpts, symrel, tnons, tol=1e-8, *, time_reversal=False,
    symafm=None, target_channel=0,
):
    """Map raw WFK symmetry rows, which are native symrel transposed.

    Native cgtk_rotate uses S_native.T @ k. Translations phase source
    states, never k coordinates. Antiunitary search needs an explicit
    caller certificate, not an inference from spin dimensions.
    """
    symrel = np.asarray(symrel, dtype=int)
    translations = np.asarray(tnons, dtype=float)
    ibz = np.asarray(ibz_kpts, dtype=float)
    out = []
    for im, k in enumerate(np.asarray(mesh_kpts, dtype=float)):
        found = None
        for isym, mat in enumerate(symrel):
            for ik, kb in enumerate(ibz):
                for sign in (1, -1) if time_reversal else (1,):
                    target = sign * (mat @ kb)
                    shift = np.rint(k - target)
                    if np.all(np.abs(k - target - shift) < tol):
                        found = MeshMapping(
                            im,
                            ik,
                            sign * mat,
                            translations[isym],
                            shift.astype(int),
                            time_reversal=sign < 0,
                            source_channel=(
                                target_channel ^ int(symafm[isym] < 0)
                                if symafm is not None else target_channel
                            ),
                        )
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            raise _fail(
                "D4",
                "<ibz expansion>",
                f"mesh point {im}",
                f"k={k.tolist()} reachable by certified source symmetries",
            )
        out.append(found)
    return out


def align_degenerate_subspace(ref_evals, vecs_ref, vecs_test, deg_tol_hartree=1e-7):
    """Unitary Procrustes alignment of ``vecs_test`` onto ``vecs_ref``
    within each degenerate subspace of ``ref_evals``.

    Both vector arrays are (nband, d): rows are band states in a common
    coefficient basis.  Within each degenerate group the residual gauge is
    a unitary W acting on the band index; the minimizer of
    ``||W t - r||_F`` over unitary W is W = U V^dagger where
    ``r t^dagger = U S V^dagger``.  Returns a (nband, d) complex copy.
    """
    ref_evals = np.asarray(ref_evals, dtype=float)
    r_all = np.asarray(vecs_ref, dtype=complex)
    t_all = np.asarray(vecs_test, dtype=complex)
    out = t_all.copy()
    n = len(ref_evals)
    start = 0
    for i in range(1, n + 1):
        if i == n or abs(ref_evals[i] - ref_evals[start]) > deg_tol_hartree:
            grp = slice(start, i)
            r, t = r_all[grp], out[grp]
            u, _s, vt = np.linalg.svd(r @ t.conj().T)
            out[grp] = (u @ vt) @ t
            start = i
    return out
