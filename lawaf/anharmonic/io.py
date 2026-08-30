"""netCDF serialization of the anharmonic effective model (story 022,
FR-012).

Scope and file ownership
------------------------
This module owns ONLY the ``anharmonic`` group of a netCDF file.  Other
programs may share the same file: :func:`save_anharmonic_model` opens it
with ``mode="w"`` only when the file does not exist yet and with
``mode="a"`` otherwise (never truncating sibling groups), and refuses to
overwrite an existing group of its own.  The story-019 ``symmetry`` group
lives alongside under the same rules (:func:`save_symmetry`); both groups
are independent and either may be written first.

Schema (version 1, group ``anharmonic``)
----------------------------------------
attrs
    schema_version (int), created, lawaf_version, conventions,
    teacher, harmonic_source (provenance strings), provenance (JSON of any
    extra provenance), selection, ridge_alpha, fingerprint, weights (JSON),
    orders (JSON), include_strain, max_strain_power (-1 = None),
    cutoff_active, even_pure_strain, nQ, nlwf, primitive_volume (NaN =
    unknown), has_harmonic, locality (JSON), cv (JSON).
variables
    coord_branch (ncoord), coord_R (ncoord, 3)     pool coordinate labels
    sc_matrix (3, 3), sc_vec (ncell, 3)            supercell definition
    rlist (nR, 3)                                  basis R list
    term_order (nterm), term_sector (nterm, code)  sector: 0 const, 1 q,
                                                   2 strain, 3 coupled
    nseed/seed_branch/seed_R/seed_strain           ClusterKey per term,
                                                   -1/0 padded index arrays
    mono_nq/mono_qidx/mono_ns/mono_sidx            global monomial table
                                                   (coordinate indices into
                                                   the pool, -1 padded)
    term_ptr (nterm+1), entry_mono/entry_num/entry_den
        CSR term/monomial structure; the exact rational basis coefficients
        as int64 numerator/denominator pairs (Fractions roundtrip exactly)
    coefficients (nterm, units="eV"), stderrs (nterm, units="eV")
    selected (nsel)
    harm_HR_re/harm_HR_im (nR2, nw, nw), harm_Rlist (nR2, 3), harm_Rdeg
    (nR2)   the harmonic LWF source, present iff has_harmonic; the baseline
        is rebuilt through the public ``harmonic_baseline`` path (bitwise
        identical fold)

Schema (version 1, group ``symmetry``, story 019)
-------------------------------------------------
attrs
    schema_version (int), created, lawaf_version, conventions,
    provenance (JSON), symprec, dataset (JSON echo: number,
    international, wyckoffs, site_symmetry_symbols, equivalent_atoms),
    action_fingerprint (sha256 of the action's matrices over the
    canonical irreducible q set), action_fp_mesh (JSON), has_declaration,
    has_report, has_basis; declaration: wyckoff, site_irreps (JSON),
    strain_sector, anchors (JSON or null); report: label_source,
    compat_checks (JSON), basis: basis_fingerprint, molien_sectors and
    molien_notes (JSON, parallel to the row arrays).
variables
    scaled_positions (natom, 3) f8, cell (3, 3) f8, numbers (natom) i8
    (canonical orbit-rank type labels; see ``_canonical_numbers``),
    compat_q (nchk, 3) f8, compat_passed (nchk) i1
    molien_order/molien_molien/molien_constructed (nrow) i8,
    molien_consistent (nrow) i1

The action is REBUILT on load from (cell, scaled_positions, numbers,
symprec) through spglib + ``build_space_group_action`` -- no phonon object
needed -- and verified against the stored op arrays and dataset echo.

Forward-compatibility policy
----------------------------
``load_anharmonic_model`` REFUSES files whose ``schema_version`` is greater
than :data:`SCHEMA_VERSION` (informative error): a silent best-effort read
of an unknown schema can quietly corrupt physics, so callers must upgrade
lawaf instead.  Lower versions are read by the matching reader.
``load_symmetry`` applies the same policy against
:data:`SYMMETRY_SCHEMA_VERSION`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Mapping, Optional

import numpy as np

from lawaf.anharmonic.basis import (
    ClusterKey,
    InvariantBasis,
    InvariantTerm,
    MolienRow,
    molien_check,
)
from lawaf.anharmonic.compatibility import (
    CompatCheck,
    RepresentationDeclaration,
    RepresentationReport,
)
from lawaf.anharmonic.fit import (
    AnharmonicCoefficients,
    CoefficientTerm,
    CVReport,
    HarmonicBaseline,
    LocalityReport,
    LocalityRow,
    basis_cell_permutation,
)
from lawaf.anharmonic.model import AnharmonicModel
from lawaf.anharmonic.representation import WindowBands, WindowLegalityBlock

__all__ = [
    "save_anharmonic_model",
    "load_anharmonic_model",
    "SCHEMA_VERSION",
    "save_symmetry",
    "load_symmetry",
    "SymmetryRecord",
    "action_fingerprint",
    "SYMMETRY_SCHEMA_VERSION",
]

SCHEMA_VERSION = 1

# story 019: the 'symmetry' group versions independently of 'anharmonic'
SYMMETRY_SCHEMA_VERSION = 2

# canonical mesh for the action fingerprint (part of its definition)
_FP_MESH = (2, 2, 2)

SECTOR_CODES = {"const": 0, "q": 1, "strain": 2, "coupled": 3}
SECTOR_NAMES = {v: k for k, v in SECTOR_CODES.items()}

_CONVENTIONS = (
    "Q: flat over M columns, c = icell*nlwf + iwann (scmaker.sc_vec order); "
    "u = M @ Q. Energy E(Q, eps) = E_harm(Q) + sum_t c_t B_t(Q, eps) with the "
    "supercell-folded harmonic kernel. Strain: Voigt (xx,yy,zz,yz,xz,xy), "
    "cell = cell0 @ (I + eps). Stress: sigma_v = (1/vol) dE/deps_v, each "
    "tensor component stored once, vol = V0_prim * det(I + eps), "
    "tension-positive (ASE)."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _lawaf_version() -> str:
    try:
        from importlib.metadata import version

        return version("lawaf")
    except Exception:
        return "unknown"


def _padded(rows: List[List[int]], width: int, pad: int) -> np.ndarray:
    out = np.full((len(rows), width), pad, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _json_attr(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def _open_root(path):
    import os

    import netCDF4

    path = str(path)
    if os.path.exists(path):
        return netCDF4.Dataset(path, "a")
    return netCDF4.Dataset(path, "w")


def _sc_lattice_of(c: AnharmonicCoefficients):
    """(sc_matrix, sc_vec) of the fitted model (harmonic or unit default)."""
    if c.harmonic is not None:
        return np.asarray(c.harmonic.sc_matrix), np.asarray(c.harmonic.sc_vec)
    return np.eye(3, dtype=int), np.zeros((1, 3), dtype=int)


class _LWFDuck:
    """Minimal LWF-like duck type for harmonic_baseline reconstruction."""

    def __init__(self, HR_total, Rlist, Rdeg):
        self.HR_total = HR_total
        self.Rlist = Rlist
        self.Rdeg = Rdeg


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def save_anharmonic_model(
    path,
    coefficients: AnharmonicCoefficients,
    provenance: Optional[Mapping[str, str]] = None,
    primitive_volume: Optional[float] = None,
) -> None:
    """Write ``coefficients`` into the netCDF group ``anharmonic``.

    Parameters
    ----------
    path:
        netCDF file (created if absent; otherwise opened in append mode so
        sibling groups such as a future ``symmetry`` group are preserved).
    coefficients:
        A fitted (or hand-assembled)
        :class:`~lawaf.anharmonic.fit.AnharmonicCoefficients`; the basis,
        the exact rational term construction, the fitted float64 arrays and
        the fit metadata are all serialized.
    provenance:
        Optional mapping; the keys ``teacher`` and ``harmonic_source`` are
        stored as dedicated attributes, everything else under
        ``provenance`` (JSON).
    primitive_volume:
        Reference per-primitive-cell volume (A^3) for
        :meth:`AnharmonicModel.stress`; stored as NaN when unknown.
    """
    c = coefficients
    basis = c.basis
    nterm = len(basis.terms)
    if nterm == 0:
        raise ValueError("refusing to serialize an empty invariant basis")
    if np.shape(c.coefficients) != (nterm,) or np.shape(c.stderrs) != (nterm,):
        raise ValueError("coefficients/stderrs width differs from the basis")

    prov = dict(provenance or {})

    # -- global monomial table (sorted, deterministic) -----------------------
    coord_index = {lab: i for i, lab in enumerate(basis.coord_labels)}
    mono_keys = sorted({key for t in basis.terms for key in t.coeffs})
    nmono = len(mono_keys)
    mono_qrows = [[coord_index[f] for f in qk] for qk, _ in mono_keys]
    mono_srows = [list(sk) for _, sk in mono_keys]
    qmax = max((len(r) for r in mono_qrows), default=1)
    smax = max((len(r) for r in mono_srows), default=1)

    mono_row = {key: i for i, key in enumerate(mono_keys)}
    entry_mono, entry_num, entry_den = [], [], []
    term_ptr = np.zeros(nterm + 1, dtype=np.int64)
    for ti, t in enumerate(basis.terms):
        for key in sorted(t.coeffs):
            frac = t.coeffs[key]
            entry_mono.append(mono_row[key])
            entry_num.append(frac.numerator)
            entry_den.append(frac.denominator)
        term_ptr[ti + 1] = len(entry_mono)

    seed_facs = [t.seed.factors for t in basis.terms]
    seed_str = [list(t.seed.strain_factors) for t in basis.terms]
    nseedmax = max((len(f) for f in seed_facs), default=1)
    nstrmax = max((len(s) for s in seed_str), default=1)

    sc_matrix, sc_vec = _sc_lattice_of(c)

    root = _open_root(path)
    try:
        if "anharmonic" in root.groups:
            raise ValueError(
                f"{path} already carries an 'anharmonic' group; refusing to "
                "overwrite (write to a fresh file)"
            )
        g = root.createGroup("anharmonic")

        # -- attributes ------------------------------------------------------
        g.setncattr("schema_version", SCHEMA_VERSION)
        g.setncattr("created", datetime.now(timezone.utc).isoformat())
        g.setncattr("lawaf_version", _lawaf_version())
        g.setncattr("conventions", _CONVENTIONS)
        g.setncattr("teacher", str(prov.pop("teacher", "")))
        g.setncattr("harmonic_source", str(prov.pop("harmonic_source", "")))
        g.setncattr("provenance", _json_attr(prov))
        g.setncattr("selection", str(c.selection))
        g.setncattr("ridge_alpha", float(c.ridge_alpha))
        g.setncattr("fingerprint", str(c.fingerprint))
        g.setncattr(
            "weights", _json_attr({k: float(v) for k, v in c.weights.items()})
        )
        g.setncattr("orders", _json_attr([int(o) for o in basis.orders]))
        g.setncattr("include_strain", int(basis.include_strain))
        g.setncattr(
            "max_strain_power",
            -1
            if basis.max_strain_power is None
            else int(basis.max_strain_power),
        )
        g.setncattr("cutoff_active", int(basis.cutoff_active))
        g.setncattr("even_pure_strain", int(basis.even_pure_strain))
        g.setncattr(
            "nQ",
            int(
                c.harmonic.nQ
                if c.harmonic is not None
                else len(basis.coord_labels)
            ),
        )
        g.setncattr("nlwf", int(basis.nlwf))
        g.setncattr(
            "primitive_volume",
            float("nan")
            if primitive_volume is None
            else float(primitive_volume),
        )
        g.setncattr("has_harmonic", int(c.harmonic is not None))
        g.setncattr("locality", _json_attr(asdict(c.locality)))
        g.setncattr("cv", _json_attr(asdict(c.cv)))

        # -- dimensions ------------------------------------------------------
        g.createDimension("three", 3)
        g.createDimension("ncoord", len(basis.coord_labels))
        g.createDimension("ncell", len(sc_vec))
        g.createDimension("nR", len(basis.rlist))
        g.createDimension("nterm", nterm)
        g.createDimension("nmono", nmono)
        g.createDimension("nnz", len(entry_mono))
        g.createDimension("nnz1", nterm + 1)
        g.createDimension("nsel", len(c.selected))
        g.createDimension("nseedmax", nseedmax)
        g.createDimension("nstrmax", nstrmax)
        g.setncattr("locality", _json_attr(asdict(c.locality)))
        g.createDimension("qmax", qmax)
        g.createDimension("smax", smax)

        # -- variables -------------------------------------------------------
        v = g.createVariable("coord_branch", "i4", ("ncoord",))
        v[:] = [int(b) for b, _ in basis.coord_labels]
        v = g.createVariable("coord_R", "i4", ("ncoord", "three"))
        v[:] = np.asarray([R for _, R in basis.coord_labels], dtype=np.int64)
        # the pool->cell coordinate permutation, stored verbatim (the fit
        # may carry it even without a harmonic baseline, from the mapping's
        # supercell; absent for identity-pool fits)
        g.setncattr("has_coord_perm", int(c.coord_perm is not None))
        if c.coord_perm is not None:
            v = g.createVariable("coord_perm", "i4", ("ncoord",))
            v[:] = np.asarray(c.coord_perm, dtype=np.int64)

        v = g.createVariable("sc_matrix", "i4", ("three", "three"))
        v[:] = np.asarray(sc_matrix, dtype=np.int64)
        v = g.createVariable("sc_vec", "i4", ("ncell", "three"))
        v[:] = np.asarray(sc_vec, dtype=np.int64)
        v = g.createVariable("rlist", "i4", ("nR", "three"))
        v[:] = np.asarray(basis.rlist, dtype=np.int64).reshape(
            len(basis.rlist), 3
        )

        v = g.createVariable("term_order", "i4", ("nterm",))
        v[:] = [int(t.order) for t in basis.terms]
        v = g.createVariable("term_sector", "i1", ("nterm",))
        v.units = "code: 0 const, 1 q, 2 strain, 3 coupled"
        v[:] = [SECTOR_CODES[t.sector] for t in basis.terms]

        v = g.createVariable("nseed", "i4", ("nterm",))
        v[:] = [len(f) for f in seed_facs]
        v = g.createVariable("seed_branch", "i4", ("nterm", "nseedmax"))
        v[:] = _padded([[int(b) for b, _ in f] for f in seed_facs], nseedmax, -1)
        seedR = np.zeros((nterm, nseedmax, 3), dtype=np.int64)
        for i, fac in enumerate(seed_facs):
            for j, (_, R) in enumerate(fac):
                seedR[i, j] = R
        v = g.createVariable("seed_R", "i4", ("nterm", "nseedmax", "three"))
        v[:] = seedR
        v = g.createVariable("seed_strain", "i4", ("nterm", "nstrmax"))
        v[:] = _padded(seed_str, nstrmax, -1)

        v = g.createVariable("mono_nq", "i4", ("nmono",))
        v[:] = [len(qk) for qk, _ in mono_keys]
        v = g.createVariable("mono_qidx", "i4", ("nmono", "qmax"))
        v[:] = _padded(mono_qrows, qmax, -1)
        v = g.createVariable("mono_ns", "i4", ("nmono",))
        v[:] = [len(sk) for _, sk in mono_keys]
        v = g.createVariable("mono_sidx", "i4", ("nmono", "smax"))
        v[:] = _padded(mono_srows, smax, -1)

        v = g.createVariable("term_ptr", "i8", ("nnz1",))
        v[:] = term_ptr
        v = g.createVariable("entry_mono", "i8", ("nnz",))
        v[:] = np.asarray(entry_mono, dtype=np.int64)
        v = g.createVariable("entry_num", "i8", ("nnz",))
        v[:] = np.asarray(entry_num, dtype=np.int64)
        v = g.createVariable("entry_den", "i8", ("nnz",))
        v[:] = np.asarray(entry_den, dtype=np.int64)

        v = g.createVariable("coefficients", "f8", ("nterm",))
        v.units = "eV"
        v[:] = np.asarray(c.coefficients, dtype=np.float64)
        v = g.createVariable("stderrs", "f8", ("nterm",))
        v.units = "eV"
        v[:] = np.asarray(c.stderrs, dtype=np.float64)
        v = g.createVariable("selected", "i4", ("nsel",))
        v[:] = np.asarray(list(c.selected), dtype=np.int64)

        if c.harmonic is not None:
            hb = c.harmonic
            HR = np.asarray(hb.HR)
            g.createDimension("nR2", len(hb.Rlist))
            g.createDimension("nwann", int(hb.nwann))
            v = g.createVariable("harm_HR_re", "f8", ("nR2", "nwann", "nwann"))
            v[:] = np.real(HR).astype(np.float64)
            v = g.createVariable("harm_HR_im", "f8", ("nR2", "nwann", "nwann"))
            v[:] = np.imag(HR).astype(np.float64)
            v = g.createVariable("harm_Rlist", "i4", ("nR2", "three"))
            v[:] = np.asarray(hb.Rlist, dtype=np.int64)
            v = g.createVariable("harm_Rdeg", "f8", ("nR2",))
            v[:] = np.asarray(hb.Rdeg, dtype=np.float64)
            v = g.createVariable("harm_sc_matrix", "i4", ("three", "three"))
            v[:] = np.asarray(hb.sc_matrix, dtype=np.int64)
    finally:
        root.close()



# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_anharmonic_model(path) -> AnharmonicModel:
    """Rebuild an :class:`AnharmonicModel` from the ``anharmonic`` group.

    Standalone: the basis (exact rational coefficients), the fitted float64
    arrays and the harmonic baseline are reconstructed from the file alone.
    The cluster action is NOT serialized (it is only needed to RE-build a
    basis, not to evaluate one); a loaded basis carries ``action=None``.
    """
    import netCDF4

    with netCDF4.Dataset(str(path), "r") as root:
        if "anharmonic" not in root.groups:
            raise ValueError(f"{path} has no 'anharmonic' group")
        g = root.groups["anharmonic"]
        sv = int(g.getncattr("schema_version"))
        if sv > SCHEMA_VERSION:
            raise ValueError(
                f"{path}: anharmonic schema_version={sv} is NEWER than the "
                f"supported version {SCHEMA_VERSION}; refusing to misread an "
                "unknown schema - upgrade lawaf"
            )
        if sv < 1:
            raise ValueError(f"{path}: invalid anharmonic schema_version={sv}")
        arrays = {name: np.array(var[:]) for name, var in g.variables.items()}
        attrs = {name: g.getncattr(name) for name in g.ncattrs()}

    nterm = len(arrays["term_order"])
    coord_R = arrays["coord_R"]
    coord_labels = tuple(
        (int(b), (int(r[0]), int(r[1]), int(r[2])))
        for b, r in zip(arrays["coord_branch"], coord_R)
    )
    nlwf = int(attrs["nlwf"])
    orders = tuple(int(o) for o in json.loads(attrs["orders"]))

    # -- global monomial table ----------------------------------------------
    mono_keys = []
    for i in range(len(arrays["mono_nq"])):
        nq = int(arrays["mono_nq"][i])
        ns = int(arrays["mono_ns"][i])
        qidx = arrays["mono_qidx"][i][:nq].astype(int)
        qk = tuple(
            (int(arrays["coord_branch"][ci]), (int(R[0]), int(R[1]), int(R[2])))
            for ci, R in zip(qidx, coord_R[qidx])
        )
        mono_keys.append(
            (qk, tuple(int(v) for v in arrays["mono_sidx"][i][:ns]))
        )

    # -- CSR terms -----------------------------------------------------------
    from fractions import Fraction

    terms: List[InvariantTerm] = []
    for ti in range(nterm):
        lo, hi = int(arrays["term_ptr"][ti]), int(arrays["term_ptr"][ti + 1])
        coeffs = {}
        for e in range(lo, hi):
            qk, sk = mono_keys[int(arrays["entry_mono"][e])]
            coeffs[(qk, sk)] = Fraction(
                int(arrays["entry_num"][e]), int(arrays["entry_den"][e])
            )
        nseed = int(arrays["nseed"][ti])
        facs = tuple(
            (
                int(arrays["seed_branch"][ti, j]),
                tuple(int(x) for x in arrays["seed_R"][ti, j]),
            )
            for j in range(nseed)
        )
        nstr = _count_pad(arrays["seed_strain"][ti])
        seed = ClusterKey(
            facs, tuple(int(v) for v in arrays["seed_strain"][ti][:nstr])
        )
        sector = SECTOR_NAMES[int(arrays["term_sector"][ti])]
        terms.append(
            InvariantTerm(
                seed=seed,
                coeffs=coeffs,
                order=int(arrays["term_order"][ti]),
                sector=sector,
            )
        )

    rlist = tuple((int(r[0]), int(r[1]), int(r[2])) for r in arrays["rlist"])
    basis = InvariantBasis(
        terms=terms,
        coord_labels=coord_labels,
        nlwf=nlwf,
        action=None,  # not serialized; evaluation never needs it
        orders=orders,
        include_strain=bool(attrs["include_strain"]),
        max_strain_power=(
            None
            if int(attrs["max_strain_power"]) < 0
            else int(attrs["max_strain_power"])
        ),
        cutoff_active=bool(attrs["cutoff_active"]),
        rlist=rlist,
        even_pure_strain=bool(attrs["even_pure_strain"]),
    )

    # -- harmonic baseline (public rebuild path, bitwise identical fold) -----
    harmonic = None
    if int(attrs["has_harmonic"]):
        HR = arrays["harm_HR_re"] + 1j * arrays["harm_HR_im"]
        lwf = _LWFDuck(
            HR_total=HR,
            Rlist=arrays["harm_Rlist"],
            Rdeg=arrays["harm_Rdeg"],
        )
        from lawaf.utils.supercell import SupercellMaker
        harm_sc = np.asarray(
            arrays.get("harm_sc_matrix", arrays.get("sc_matrix")),
            dtype=int,
        )
        harmonic = HarmonicBaseline(lwf, SupercellMaker(harm_sc))

    pc = None
    if bool(int(attrs.get("has_coord_perm", 0))) and "coord_perm" in arrays:
        # stored verbatim at write time (identity fits store nothing)
        pc = np.asarray(arrays["coord_perm"], dtype=int)
    elif harmonic is not None:
        pc = basis_cell_permutation(basis, harmonic)
    coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
    stderrs = np.asarray(arrays["stderrs"], dtype=np.float64)
    selected = tuple(int(i) for i in arrays["selected"])
    term_list = [
        CoefficientTerm(
            index=ti,
            seed=basis.terms[ti].seed,
            order=basis.terms[ti].order,
            sector=basis.terms[ti].sector,
            coeff=float(coefficients[ti]),
            stderr=None if np.isnan(stderrs[ti]) else float(stderrs[ti]),
        )
        for ti in selected
    ]
    res = AnharmonicCoefficients(
        basis=basis,
        harmonic=harmonic,
        terms=term_list,
        coefficients=coefficients,
        stderrs=stderrs,
        selected=selected,
        locality=_locality_from_json(attrs["locality"]),
        fingerprint=str(attrs["fingerprint"]),
        cv=_cv_from_json(attrs["cv"]),
        selection=str(attrs["selection"]),
        ridge_alpha=float(attrs["ridge_alpha"]),
        weights={k: float(v) for k, v in json.loads(attrs["weights"]).items()},
        coord_perm=pc,
    )
    pvol = float(attrs["primitive_volume"])
    return AnharmonicModel(res, primitive_volume=None if np.isnan(pvol) else pvol)


def _count_pad(row) -> int:
    """Number of leading entries before the -1 padding of an index row."""
    n = 0
    for v in np.asarray(row):
        if int(v) < 0:
            break
        n += 1
    return n


def _locality_from_json(blob: str) -> LocalityReport:
    data = json.loads(blob or "{}")
    rows = [
        LocalityRow(
            term_index=int(r["term_index"]),
            sector=r["sector"],
            order=int(r["order"]),
            radius_lattice=float(r["radius_lattice"]),
            radius_cart=(
                None if r["radius_cart"] is None else float(r["radius_cart"])
            ),
            abs_coeff=float(r["abs_coeff"]),
            mass_fraction=float(r["mass_fraction"]),
        )
        for r in data.get("rows", [])
    ]
    return LocalityReport(
        rows=rows,
        r90_lattice=(
            None
            if data.get("r90_lattice") is None
            else float(data["r90_lattice"])
        ),
        r90_cart=(
            None if data.get("r90_cart") is None else float(data["r90_cart"])
        ),
    )


def _cv_from_json(blob: str) -> CVReport:
    data = json.loads(blob or "{}")

    def _opt(key):
        return None if data.get(key) is None else float(data[key])

    return CVReport(
        n_train_rows=int(data.get("n_train_rows", 0)),
        n_cv_frames=int(data.get("n_cv_frames", 0)),
        energy_mae=_opt("energy_mae"),
        force_rmse=_opt("force_rmse"),
        force_cosine=_opt("force_cosine"),
        stress_rmse=_opt("stress_rmse"),
        notes=[str(n) for n in data.get("notes", [])],
    )


# ---------------------------------------------------------------------------
# story 019: the 'symmetry' group (SpaceGroupAction persistence + provenance)
# ---------------------------------------------------------------------------
_SYM_CONVENTIONS = (
    "Space-group action on primitive-cell displacements, lawaf gauge "
    "(story-014 convention).  q' = W^-T q (mod 1); "
    "[S_g(q)]_{sigma(k),k} = R_g exp(-2 pi i q'. t_k) with "
    "R_g = A W_g A^-1 (A: cell rows), t_k integer defect vectors.  The "
    "action is REBUILT on load from (cell, scaled_positions, numbers, "
    "symprec) via spglib + build_space_group_action and verified against "
    "the stored op arrays and the dataset JSON echo."
)


@dataclass(frozen=True)
class SymmetryRecord:
    """Everything persisted in (and rebuilt from) the ``symmetry`` group."""

    sga: object  # SpaceGroupAction
    declaration: Optional[RepresentationDeclaration] = None
    report: Optional[RepresentationReport] = None
    basis_fingerprint: Optional[str] = None
    action_fingerprint: Optional[str] = None
    molien_rows: Optional[List[MolienRow]] = None
    provenance: Mapping[str, str] = field(default_factory=dict)
    created: str = ""
    lawaf_version: str = ""
    window_bands: Optional[WindowBands] = None
    gauge_residuals: Optional[dict[tuple[float, float, float], float]] = None


def action_fingerprint(sga, mesh=_FP_MESH, digits: int = 12) -> str:
    """sha256 identifying the action: its matrices on a canonical q set.

    Deterministic by construction:
    * the q set is ``sga.irreducible_qpoints(mesh)`` (first-seen mesh
      enumeration; independent of the op array order because the
      membership test quantifies over the whole group),
    * operations are hashed in the canonical
      ``(rotation bytes, translation bytes)`` sort order, so RELABELING the
      operations (permuting the op arrays) leaves the fingerprint
      unchanged,
    * floats are rounded to ``digits`` decimals (with ``-0.0`` folded to
      ``+0.0``) before hashing.
    """
    rotations = np.asarray(sga.rotations, dtype=np.int64)
    translations = np.asarray(sga.translations, dtype=np.float64)
    order = sorted(
        range(len(rotations)),
        key=lambda g: (
            rotations[g].tobytes(),
            (np.round(translations[g], digits) + 0.0).tobytes(),
        ),
    )
    qs = sga.irreducible_qpoints(np.asarray(mesh, dtype=int))
    h = hashlib.sha256()
    h.update(
        (
            f"lawaf-sga-fp-v1|mesh={tuple(int(m) for m in np.asarray(mesh, dtype=int))}"
            f"|nops={len(rotations)}|natom={sga.n_atoms}|digits={digits}"
        ).encode()
    )
    for g in order:
        h.update(rotations[g].tobytes())
        h.update((np.round(translations[g], digits) + 0.0).tobytes())
        for q in qs:
            m = np.asarray(sga.matrix(g, q), dtype=complex)
            h.update((np.round(m.real, digits) + 0.0).tobytes(order="C"))
            h.update((np.round(m.imag, digits) + 0.0).tobytes(order="C"))
    return h.hexdigest()


def _dataset_echo(dataset) -> dict:
    """JSON-safe subset of the spglib dataset persisted for verification."""
    return {
        "number": int(dataset.number),
        "international": str(dataset.international),
        "wyckoffs": [str(w) for w in dataset.wyckoffs],
        "site_symmetry_symbols": [
            str(s) for s in dataset.site_symmetry_symbols
        ],
        "equivalent_atoms": [int(a) for a in dataset.equivalent_atoms],
    }

def save_symmetry(
    path,
    sga,
    declaration: Optional[RepresentationDeclaration] = None,
    report: Optional[RepresentationReport] = None,
    basis: Optional[InvariantBasis] = None,
    provenance: Optional[Mapping[str, str]] = None,
    window_bands: Optional[WindowBands] = None,
    gauge_diagnostics: Optional[Mapping] = None,
) -> None:
    """Write the space-group action and its provenance into the netCDF
    group ``symmetry``.

    Parameters
    ----------
    path:
        netCDF file (created if absent; otherwise opened in append mode so
        sibling groups such as ``anharmonic`` are preserved).  An existing
        ``symmetry`` group is never overwritten.
    sga:
        A :class:`~lawaf.anharmonic.representation.SpaceGroupAction` as
        built by ``build_space_group_action`` (its ``_aux`` build data and
        ``symmetry_dataset`` are required).
    declaration:
        Optional :class:`RepresentationDeclaration`, persisted
        field-for-field.
    report:
        Optional :class:`RepresentationReport` (``declaration`` defaults to
        ``report.declaration`` when omitted); per-anchor q/pass arrays plus
        a JSON echo of the expected/found irrep labels.
    basis:
        Optional :class:`InvariantBasis`; its fingerprint, the action
        fingerprint and the ``molien_check`` rows (order, sector,
        constructed, molien, consistent, note) are persisted as provenance.
        The Molien comparison runs at save time.
    provenance:
        Optional free-form string mapping (JSON).
    """
    rotations = np.asarray(sga.rotations, dtype=np.int64)
    translations = np.asarray(sga.translations, dtype=np.float64)
    spos = np.asarray(sga.scaled_positions, dtype=np.float64)
    aux = getattr(sga, "_aux", None) or {}
    if (
        aux.get("cell") is None
        or aux.get("sigma") is None
        or aux.get("defects") is None
        or sga.symmetry_dataset is None
    ):
        raise ValueError(
            "save_symmetry needs a SpaceGroupAction with its build-time data "
            "(_aux cell/sigma/defects) and spglib symmetry_dataset; rebuild "
            "it with build_space_group_action"
        )
    cell = np.asarray(aux["cell"], dtype=np.float64)
    atom_maps = np.asarray(aux["sigma"], dtype=np.int64)
    defects = np.rint(np.asarray(aux["defects"], dtype=np.float64)).astype(
        np.int64
    )
    if report is not None and declaration is None:
        declaration = report.declaration
    if window_bands is not None and not isinstance(window_bands, WindowBands):
        raise TypeError("window_bands must be a validated WindowBands instance")
    window_qkeys = ()
    window_rep_qkeys = ()
    window_blocks = ()
    if window_bands is not None:
        if not window_bands.representatives:
            raise ValueError("window_bands must retain at least one representative")
        if set(window_bands.legality) != set(window_bands.bands):
            raise ValueError("window_bands legality must cover every persisted q-point")
        if any(
            window_bands.bands.get(q) != bands
            for q, bands in window_bands.representatives.items()
        ):
            raise ValueError(
                "window_bands representatives must agree with expanded bands"
            )
        window_qkeys = tuple(sorted(window_bands.bands))
        window_rep_qkeys = tuple(sorted(window_bands.representatives))
        qindex = {q: i for i, q in enumerate(window_qkeys)}
        window_blocks = tuple(
            (qindex[q], block)
            for q in window_qkeys
            for block in window_bands.legality[q]
        )
    if gauge_diagnostics is not None and not isinstance(gauge_diagnostics, Mapping):
        raise TypeError("gauge_diagnostics must be a mapping returned by constrain_builder_amn")
    gauge_rows = ()
    if gauge_diagnostics is not None:
        try:
            gauge_rows = tuple(
                sorted(
                    (
                        tuple(float(x) for x in np.asarray(q, dtype=float).reshape(3)),
                        float(eps),
                    )
                    for q, eps in gauge_diagnostics["eps"].items()
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "gauge_diagnostics must contain an eps mapping keyed by q-point"
            ) from exc



    root = _open_root(path)
    try:
        if "symmetry" in root.groups:
            raise ValueError(
                f"{path} already carries a 'symmetry' group; refusing to "
                "overwrite (write to a fresh file)"
            )
        g = root.createGroup("symmetry")

        # -- attributes ------------------------------------------------------
        g.setncattr("schema_version", SYMMETRY_SCHEMA_VERSION)
        g.setncattr("created", datetime.now(timezone.utc).isoformat())
        g.setncattr("lawaf_version", _lawaf_version())
        g.setncattr("conventions", _SYM_CONVENTIONS)
        g.setncattr("provenance", _json_attr(dict(provenance or {})))
        g.setncattr("symprec", float(sga.symprec))
        g.setncattr("dataset", _json_attr(_dataset_echo(sga.symmetry_dataset)))
        g.setncattr("action_fingerprint", action_fingerprint(sga, _FP_MESH))
        g.setncattr("action_fp_mesh", _json_attr([int(m) for m in _FP_MESH]))
        g.setncattr("has_declaration", int(declaration is not None))
        g.setncattr("has_report", int(report is not None))
        g.setncattr("has_basis", int(basis is not None))
        g.setncattr("has_window_bands", int(window_bands is not None))
        g.setncattr("has_gauge_residuals", int(bool(gauge_rows)))
        if window_bands is not None:
            g.setncattr(
                "window_block_irreps",
                _json_attr([list(block.irreps) for _iq, block in window_blocks]),
            )
        if declaration is not None:
            g.setncattr("wyckoff", str(declaration.wyckoff))
            g.setncattr(
                "site_irreps",
                _json_attr([str(s) for s in declaration.site_irreps]),
            )
            g.setncattr("strain_sector", int(bool(declaration.strain_sector)))
            g.setncattr(
                "anchors",
                _json_attr(
                    None
                    if declaration.anchors is None
                    else [[float(x) for x in a] for a in declaration.anchors]
                ),
            )
            g.setncattr(
                "window_irreps",
                _json_attr(
                    None
                    if declaration.window_irreps is None
                    else [
                        [[float(x) for x in q], [str(s) for s in irreps]]
                        for q, irreps in declaration.window_irreps.items()
                    ]
                ),
            )
        if report is not None:
            g.setncattr("label_source", str(report.label_source))
            g.setncattr(
                "compat_checks",
                _json_attr(
                    [
                        {
                            "q": [float(x) for x in c.qpoint],
                            "expected": [str(s) for s in c.expected],
                            "found": [str(s) for s in c.found],
                            "passed": bool(c.passed),
                        }
                        for c in report.checks
                    ]
                ),
            )
        if basis is not None:
            g.setncattr("basis_fingerprint", str(basis.fingerprint))
            mrep = molien_check(basis)
            g.setncattr(
                "molien_sectors", _json_attr([r.sector for r in mrep.rows])
            )
            g.setncattr("molien_notes", _json_attr([r.note for r in mrep.rows]))

        # -- dimensions ------------------------------------------------------
        g.createDimension("three", 3)
        g.createDimension("nops", len(rotations))
        g.createDimension("natom", len(spos))
        if report is not None:
            g.createDimension("nchk", len(report.checks))
        if basis is not None:
            g.createDimension("nrow", len(mrep.rows))

        if window_bands is not None:
            g.createDimension("nwindow_q", len(window_qkeys))
            g.createDimension(
                "nwindow_band",
                max(len(window_bands.bands[q]) for q in window_qkeys),
            )
            g.createDimension("nwindow_rep", len(window_rep_qkeys))
            g.createDimension(
                "nwindow_rep_band",
                max(
                    len(window_bands.representatives[q])
                    for q in window_rep_qkeys
                ),
            )
            g.createDimension("nwindow_block", len(window_blocks))
        if gauge_rows:
            g.createDimension("ngauge_anchor", len(gauge_rows))
        # -- variables -------------------------------------------------------
        v = g.createVariable("rotations", "i8", ("nops", "three", "three"))
        v[:] = rotations
        v = g.createVariable("translations", "f8", ("nops", "three"))
        v[:] = translations
        v = g.createVariable("scaled_positions", "f8", ("natom", "three"))
        v[:] = spos
        v = g.createVariable("cell", "f8", ("three", "three"))
        v[:] = cell
        v = g.createVariable("numbers", "i8", ("natom",))
        v[:] = _canonical_numbers(sga)
        v = g.createVariable("atom_maps", "i8", ("nops", "natom"))
        v[:] = atom_maps
        v = g.createVariable("defect_vectors", "i8", ("nops", "natom", "three"))
        v[:] = defects


        if window_bands is not None:
            v = g.createVariable("window_q", "f8", ("nwindow_q", "three"))
            v[:] = np.asarray(window_qkeys, dtype=float)
            v = g.createVariable("window_band_count", "i8", ("nwindow_q",))
            v[:] = [len(window_bands.bands[q]) for q in window_qkeys]
            v = g.createVariable(
                "window_bands", "i8", ("nwindow_q", "nwindow_band"), fill_value=-1
            )
            v[:] = _padded(
                [list(window_bands.bands[q]) for q in window_qkeys],
                len(g.dimensions["nwindow_band"]),
                -1,
            )
            v = g.createVariable(
                "window_representative_q", "f8", ("nwindow_rep", "three")
            )
            v[:] = np.asarray(window_rep_qkeys, dtype=float)
            v = g.createVariable(
                "window_representative_band_count", "i8", ("nwindow_rep",)
            )
            v[:] = [
                len(window_bands.representatives[q]) for q in window_rep_qkeys
            ]
            v = g.createVariable(
                "window_representative_bands",
                "i8",
                ("nwindow_rep", "nwindow_rep_band"),
                fill_value=-1,
            )
            v[:] = _padded(
                [list(window_bands.representatives[q]) for q in window_rep_qkeys],
                len(g.dimensions["nwindow_rep_band"]),
                -1,
            )
            v = g.createVariable("window_block_q", "i8", ("nwindow_block",))
            v[:] = [iq for iq, _block in window_blocks]
            v = g.createVariable("window_block_frequency", "f8", ("nwindow_block",))
            v[:] = [block.frequency for _iq, block in window_blocks]
            v = g.createVariable("window_block_dimension", "i8", ("nwindow_block",))
            v[:] = [block.dimension for _iq, block in window_blocks]
            v = g.createVariable("window_block_residual", "f8", ("nwindow_block",))
            v[:] = [block.residual for _iq, block in window_blocks]
        if gauge_rows:
            v = g.createVariable("gauge_anchor_q", "f8", ("ngauge_anchor", "three"))
            v[:] = np.asarray([q for q, _eps in gauge_rows], dtype=float)
            v = g.createVariable("gauge_anchor_residual", "f8", ("ngauge_anchor",))
            v[:] = [eps for _q, eps in gauge_rows]
        if report is not None:
            v = g.createVariable("compat_q", "f8", ("nchk", "three"))
            v[:] = np.asarray([c.qpoint for c in report.checks], dtype=float)
            v = g.createVariable("compat_passed", "i1", ("nchk",))
            v[:] = [int(bool(c.passed)) for c in report.checks]
        if basis is not None:
            v = g.createVariable("molien_order", "i8", ("nrow",))
            v[:] = [int(r.order) for r in mrep.rows]
            v = g.createVariable("molien_molien", "i8", ("nrow",))
            v[:] = [int(r.molien) for r in mrep.rows]
            v = g.createVariable("molien_constructed", "i8", ("nrow",))
            v[:] = [int(r.constructed) for r in mrep.rows]
            v = g.createVariable("molien_consistent", "i1", ("nrow",))
            v[:] = [int(r.consistent) for r in mrep.rows]
    finally:
        root.close()


def _canonical_numbers(sga) -> np.ndarray:
    """Deterministic type labels that reproduce the stored symmetry.

    Atoms of one ``equivalent_atoms`` orbit are related by symmetry
    operations, hence share the (here physics-irrelevant) type label;
    ranking the orbits by first appearance gives integer labels under
    which spglib rediscovers exactly the stored group (the labels are
    only ever compared for equality by the symmetry search -- finer than
    species can only keep the op set, never grow it).
    ``load_symmetry`` verifies the rebuilt dataset echo and op arrays
    against the persisted ones, so any exotic deviation fails loudly.
    """
    rank: dict = {}
    out = []
    for a in sga.symmetry_dataset.equivalent_atoms:
        a = int(a)
        if a not in rank:
            rank[a] = len(rank)
        out.append(rank[a])
    return np.asarray(out, dtype=np.int64)


def load_symmetry(path) -> SymmetryRecord:
    """Rebuild the persisted :class:`SymmetryRecord` from the ``symmetry``
    group.

    Standalone: the :class:`SpaceGroupAction` is reconstructed from the
    stored primitive (cell, scaled_positions, numbers, symprec) through
    spglib + ``build_space_group_action`` -- the original phonon object is
    NOT needed -- and verified against the stored op arrays and the dataset
    echo (number, international, wyckoffs, site_symmetry_symbols,
    equivalent_atoms).
    """
    import netCDF4

    with netCDF4.Dataset(str(path), "r") as root:
        if "symmetry" not in root.groups:
            raise ValueError(f"{path} has no 'symmetry' group")
        g = root.groups["symmetry"]
        sv = int(g.getncattr("schema_version"))
        if sv > SYMMETRY_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: symmetry schema_version={sv} is NEWER than the "
                f"supported version {SYMMETRY_SCHEMA_VERSION}; refusing to "
                "misread an unknown schema - upgrade lawaf"
            )
        if sv < 1:
            raise ValueError(f"{path}: invalid symmetry schema_version={sv}")
        arrays = {name: np.array(var[:]) for name, var in g.variables.items()}
        attrs = {name: g.getncattr(name) for name in g.ncattrs()}

    # -- standalone rebuild through the public path --------------------------
    from ase import Atoms
    from lawaf.anharmonic.representation import build_space_group_action

    atoms = Atoms(
        numbers=[int(z) for z in arrays["numbers"]],
        cell=np.asarray(arrays["cell"], dtype=float),
        scaled_positions=np.asarray(arrays["scaled_positions"], dtype=float),
        pbc=True,
    )
    sga = build_space_group_action(atoms, symprec=float(attrs["symprec"]))

    problems = []
    if not np.array_equal(
        np.asarray(sga.rotations, dtype=np.int64), arrays["rotations"]
    ):
        problems.append("rotations")
    if not np.allclose(
        sga.translations, arrays["translations"], atol=1e-12, rtol=0
    ):
        problems.append("translations")
    if not np.array_equal(sga.atom_maps, arrays["atom_maps"]):
        problems.append("atom maps")
    if not np.array_equal(
        np.rint(sga.defect_vectors).astype(np.int64), arrays["defect_vectors"]
    ):
        problems.append("defect vectors")
    echo = json.loads(attrs["dataset"])
    got = _dataset_echo(sga.symmetry_dataset)
    for key, want in echo.items():
        if got[key] != want:
            problems.append(f"dataset.{key}")
    if problems:
        raise ValueError(
            f"{path}: rebuilt space-group action disagrees with the stored "
            f"'symmetry' group ({', '.join(problems)}); the structure or "
            "spglib version changed since the file was written"
        )

    # -- declaration / report / basis provenance -----------------------------
    declaration = None
    report = None
    if int(attrs["has_declaration"]):
        anchors = json.loads(attrs["anchors"])
        saved_window_irreps = json.loads(attrs.get("window_irreps", "null"))
        window_irreps = (
            None
            if saved_window_irreps is None
            else {
                tuple(float(x) for x in q): [str(s) for s in irreps]
                for q, irreps in saved_window_irreps
            }
        )
        declaration = RepresentationDeclaration(
            wyckoff=str(attrs["wyckoff"]),
            site_irreps=[str(s) for s in json.loads(attrs["site_irreps"])],
            strain_sector=bool(int(attrs["strain_sector"])),
            anchors=(
                None
                if anchors is None
                else tuple(tuple(float(x) for x in a) for a in anchors)
            ),
            window_irreps=window_irreps,
        )
    if int(attrs["has_report"]):
        checks = tuple(
            CompatCheck(
                qpoint=np.asarray(c["q"], dtype=float),
                expected=[str(s) for s in c["expected"]],
                found=[str(s) for s in c["found"]],
                passed=bool(c["passed"]),
            )
            for c in json.loads(attrs["compat_checks"])
        )
        report = RepresentationReport(
            declaration=declaration,
            checks=checks,
            label_source=str(attrs["label_source"]),
        )
    basis_fp = None
    molien_rows = None
    if int(attrs["has_basis"]):
        basis_fp = str(attrs["basis_fingerprint"])
        sectors = [str(s) for s in json.loads(attrs["molien_sectors"])]
        notes = [str(n) for n in json.loads(attrs["molien_notes"])]
        molien_rows = [
            MolienRow(
                order=int(arrays["molien_order"][i]),
                sector=sectors[i],
                molien=int(arrays["molien_molien"][i]),
                constructed=int(arrays["molien_constructed"][i]),
                consistent=bool(arrays["molien_consistent"][i]),
                note=notes[i],
            )
            for i in range(len(arrays["molien_order"]))
        ]

    window_bands = None
    if int(attrs.get("has_window_bands", 0)):
        qkeys = tuple(
            tuple(float(x) for x in q)
            for q in np.asarray(arrays["window_q"], dtype=float)
        )
        counts = np.asarray(arrays["window_band_count"], dtype=int)
        saved_bands = np.asarray(arrays["window_bands"], dtype=int)
        bands = {
            q: tuple(int(i) for i in saved_bands[iq, : counts[iq]])
            for iq, q in enumerate(qkeys)
        }
        if sv >= 2:
            rep_qkeys = tuple(
                tuple(float(x) for x in q)
                for q in np.asarray(
                    arrays["window_representative_q"], dtype=float
                )
            )
            rep_counts = np.asarray(
                arrays["window_representative_band_count"], dtype=int
            )
            saved_rep_bands = np.asarray(
                arrays["window_representative_bands"], dtype=int
            )
            representatives = {
                q: tuple(int(i) for i in saved_rep_bands[iq, : rep_counts[iq]])
                for iq, q in enumerate(rep_qkeys)
            }
        else:
            representatives = dict(bands)
        labels = json.loads(attrs["window_block_irreps"])
        block_q = np.asarray(arrays["window_block_q"], dtype=int)
        block_freq = np.asarray(arrays["window_block_frequency"], dtype=float)
        block_dim = np.asarray(arrays["window_block_dimension"], dtype=int)
        if sv >= 2:
            block_residual = np.asarray(
                arrays["window_block_residual"], dtype=float
            )
        else:
            block_residual = np.full(len(block_q), np.nan, dtype=float)
        legality_rows = {q: [] for q in qkeys}
        for iq, frequency, dimension, irreps, residual in zip(
            block_q, block_freq, block_dim, labels, block_residual
        ):
            legality_rows[qkeys[int(iq)]].append(
                WindowLegalityBlock(
                    frequency=float(frequency),
                    dimension=int(dimension),
                    irreps=tuple(str(name) for name in irreps),
                    residual=float(residual),
                )
            )
        window_bands = WindowBands(
            representatives=representatives,
            bands=bands,
            legality={q: tuple(rows) for q, rows in legality_rows.items()},
        )
    gauge_residuals = None
    if int(attrs.get("has_gauge_residuals", 0)):
        gauge_residuals = {
            tuple(float(x) for x in q): float(residual)
            for q, residual in zip(
                np.asarray(arrays["gauge_anchor_q"], dtype=float),
                np.asarray(arrays["gauge_anchor_residual"], dtype=float),
            )
        }


    return SymmetryRecord(
        sga=sga,
        declaration=declaration,
        report=report,
        basis_fingerprint=basis_fp,
        action_fingerprint=str(attrs["action_fingerprint"]),
        molien_rows=molien_rows,
        provenance=dict(json.loads(attrs["provenance"])),
        created=str(attrs["created"]),
        lawaf_version=str(attrs["lawaf_version"]),
        window_bands=window_bands,
        gauge_residuals=gauge_residuals,
    )
