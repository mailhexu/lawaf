"""Story 019: symmetry persistence in the netCDF ``symmetry`` group.

What is pinned
--------------
TEST-001  BaTiO3 roundtrip: ``load_symmetry`` rebuilds the
          :class:`SpaceGroupAction` STANDALONE (spglib + the stored
          primitive cell, no phonon object) and the rebuild equals the
          original on n_ops, ``matrix(g, q)`` (1e-12 over a commensurate
          grid), ``star(q)`` and ``irreducible_qpoints((3, 3, 3))``.
TEST-002  declaration + compatibility report persisted field-for-field.
TEST-003  basis provenance block: basis fingerprint, action fingerprint and
          molien rows (order/sector/counts/consistent/notes) roundtrip.
TEST-004  coexistence with the story-022 ``anharmonic`` group in BOTH write
          orders; each loader reads only its own group; sibling-group data
          bit-identical to direct saves.
TEST-005  forward-compat schema guard + overwrite refusal.

SYMPY (SYMPY RULE)
------------------
The TEST-001 grid {n/M} is closed under the star map q -> W^-T q (mod 1)
EXACTLY, because every space-group rotation is unimodular (det = +-1): the
integer matrix W^-T maps mesh numerators onto mesh numerators.  Verified
with executed sympy integer arithmetic over all 48 BaTiO3 operations and
all 27 mesh points, with the numeric ``qmap`` as the echo.
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

anh_io = pytest.importorskip("lawaf.anharmonic.io")

from lawaf.anharmonic.io import (  # noqa: E402
    SYMMETRY_SCHEMA_VERSION,
    SymmetryRecord,
    action_fingerprint,
    load_symmetry,
    save_symmetry,
)

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"

# commensurate grid incl. non-TRIM points (same set as the representation
# tests, so the persisted action is exercised where it matters)
MESH_QS = [
    np.zeros(3),
    np.array([0.5, 0.0, 0.25]),
    np.array([0.25, 0.25, 0.25]),
    np.array([0.125, 0.5, 0.25]),
    np.array([0.5, 0.5, 0.5]),
    np.array([0.0, 0.25, 0.5]),
]


@pytest.fixture(scope="module")
def phonon():
    phonopy = pytest.importorskip("phonopy")
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def model(phonon):
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    return PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)


@pytest.fixture(scope="module")
def sga(phonon):
    from lawaf.anharmonic.representation import build_space_group_action

    return build_space_group_action(phonon)


def eigencut_window(model, cut):
    """projector_fn: eigenvalue-cut window (omega^2 < cut); same as the
    compatibility tests."""
    cache = {}

    def projector_fn(q):
        key = tuple(np.round(q, 8) % 1.0)
        if key not in cache:
            cache[key] = model.solve(q)
        evals, evecs = cache[key]
        cols = np.where(evals < cut)[0]
        V = evecs[:, cols]
        return V @ V.conj().T

    return projector_fn


# ---------------------------------------------------------------------------
# tiny story-022 world for the coexistence test (same pattern as
# tests/test_anharmonic_model.py: 2-branch LWF, supercell diag(2,1,1))
# ---------------------------------------------------------------------------
def _coexistence_coefficients():
    from types import SimpleNamespace

    from ase import Atoms
    from lawaf.anharmonic.basis import (
        ClusterAction,
        build_invariant_basis,
        voigt_matrix_from_rotation,
    )
    from lawaf.anharmonic.fit import (
        AnharmonicCoefficients,
        CoefficientTerm,
        CVReport,
        LocalityReport,
        basis_cell_permutation,
        harmonic_baseline,
    )
    from lawaf.utils.supercell import SupercellMaker
    from fractions import Fraction

    class _TinyAction(ClusterAction):
        """{E, diag(1,-1,-1)}; every branch flips sign under the rotation."""

        n_ops = 2
        _rots = (np.eye(3, dtype=int), np.diag([1, -1, -1]).astype(int))
        _op1_sign = (1, 1, 1, 1, -1, -1)

        def image_factor(self, factor, op):
            b, R = factor
            R2 = tuple(int(v) for v in self._rots[op] @ np.asarray(R, dtype=int))
            return (b, R2), (-1 if op else 1)

        def point_rep(self, op):
            return self._rots[op].astype(float)

        def strain_voigt_matrix(self, op):
            return voigt_matrix_from_rotation(self._rots[op])

        def strain_row_exact(self, v, op):
            s = 1 if op == 0 else self._op1_sign[v]
            return ((v, Fraction(s)),)

        def op_inverse(self, op):
            return op

    rng = np.random.default_rng(5)
    Rlist = ((0, 0, 0), (1, 0, 0))
    HR = rng.normal(0.0, 1.0, (len(Rlist), 2, 2))
    HR = 0.5 * (HR + HR.transpose(0, 2, 1))
    lwf = SimpleNamespace(
        HR_total=HR,
        Rlist=np.array(Rlist, dtype=int),
        Rdeg=None,
        wann_masses=np.full(2, 2.0),
    )
    harmonic = harmonic_baseline(lwf, SupercellMaker(np.diag([2, 1, 1])))
    basis = build_invariant_basis(
        _TinyAction(),
        nlwf=2,
        Rlist=Rlist,
        orders=(3, 4),
        max_strain_power=1,
        include_strain=True,
    )
    nt = len(basis.terms)
    coeff = np.zeros(nt)
    selected = tuple(range(min(5, nt)))
    rng2 = np.random.default_rng(2)
    coeff[list(selected)] = rng2.normal(0.0, 0.2, len(selected))
    return AnharmonicCoefficients(
        basis=basis,
        harmonic=harmonic,
        terms=[
            CoefficientTerm(
                index=i,
                seed=basis.terms[i].seed,
                order=basis.terms[i].order,
                sector=basis.terms[i].sector,
                coeff=float(coeff[i]),
                stderr=None,
            )
            for i in selected
        ],
        coefficients=coeff,
        stderrs=np.full(nt, np.nan),
        selected=selected,
        locality=LocalityReport(rows=[]),
        fingerprint="hand022",
        cv=CVReport(n_train_rows=0, n_cv_frames=0),
        selection="hand",
        ridge_alpha=0.0,
        weights={},
        coord_perm=basis_cell_permutation(basis, harmonic),
    )


# ===========================================================================
# SYMPY verifications (SYMPY RULE)
# ===========================================================================
def test_sympy_mesh_closed_under_star_map(sga):
    """The {n/3} grid is EXACTLY closed under q -> W^-T q: unimodular W has
    an integer transpose-inverse, so integer numerators map to integer
    numerators over the same denominator.  sympy integer arithmetic over
    all 48 ops x all 27 mesh points; ``qmap`` is the numeric echo."""
    M = 3
    for g in range(sga.n_ops):
        W = __import__("sympy").Matrix(
            3, 3, [int(v) for v in sga.rotations[g].reshape(-1)]
        )
        assert W.det() in (1, -1), f"op {g} not unimodular"
        WinvT = W.inv().T
        assert all(e.is_integer for e in WinvT), f"op {g}: W^-T not integral"
        for n in itertools.product(range(M), repeat=3):
            img = WinvT * __import__("sympy").Matrix(n)
            assert all(int(e) == e for e in img), (g, n)
            qp = sga.qmap(g, np.array(n, dtype=float) / M)
            exact = np.mod(
                np.array([float(e) for e in img], dtype=float) / M, 1.0
            )
            np.testing.assert_allclose(qp, exact, atol=1e-12)


def test_action_fingerprint_relabel_invariance(sga):
    """The action fingerprint is invariant under a relabeling (permutation)
    of the op arrays: ops are hashed in canonical sorted order and the
    irreducible q set is op-order independent (discrete verification; the
    sympy-checkable identity above covers the grid closure it relies on)."""
    perm = np.arange(sga.n_ops)[::-1]

    relabeled = type(sga)(
        rotations=sga.rotations[perm],
        translations=sga.translations[perm],
        symmetry_dataset=sga.symmetry_dataset,
        scaled_positions=sga.scaled_positions,
        symprec=sga.symprec,
        _aux={
            "cell": sga._aux["cell"],
            "sigma": sga.atom_maps[perm],
            "defects": sga.defect_vectors[perm],
        },
    )
    assert action_fingerprint(relabeled) == action_fingerprint(sga)
    # deterministic across calls and carried through save/load (TEST-003)


# ===========================================================================
# TEST-001: BaTiO3 roundtrip, standalone rebuild
# ===========================================================================
def test_roundtrip_rebuilds_action_standalone(tmp_path, sga):
    path = tmp_path / "sym.nc"
    save_symmetry(path, sga, provenance={"fixture": "BaTiO3 DM_dip_wang"})
    rec = load_symmetry(path)
    assert isinstance(rec, SymmetryRecord)
    sga2 = rec.sga
    assert sga2.n_ops == sga.n_ops == 48
    assert sga2.n_atoms == sga.n_atoms == 5
    # .matrix on a commensurate grid, every operation
    for q in MESH_QS:
        for g in range(sga.n_ops):
            np.testing.assert_allclose(
                sga2.matrix(g, q),
                sga.matrix(g, q),
                atol=1e-12,
                rtol=0,
                err_msg=f"matrix mismatch at op {g}, q={q}",
            )
        st1, st2 = sga.star(q), sga2.star(q)
        assert len(st2) == len(st1)
        for (g1, q1), (g2, q2) in zip(st1, st2):
            assert g2 == g1
            np.testing.assert_allclose(q2, q1, atol=1e-12)
    # identical IBZ enumeration on the 3x3x3 mesh
    np.testing.assert_array_equal(
        sga2.irreducible_qpoints((3, 3, 3)), sga.irreducible_qpoints((3, 3, 3))
    )
    assert rec.provenance.get("fixture") == "BaTiO3 DM_dip_wang"
    assert rec.action_fingerprint == action_fingerprint(sga)


# ===========================================================================
# TEST-002: declaration + report persisted exactly
# ===========================================================================
def test_declaration_and_report_roundtrip(tmp_path, sga, model):
    from lawaf.anharmonic.compatibility import (
        RepresentationDeclaration,
        check_compatibility,
    )

    decl = RepresentationDeclaration(
        wyckoff="1b",
        site_irreps=["T1u"],
        strain_sector=True,
        anchors=((0.0, 0.0, 0.0),),
    )
    report = check_compatibility(sga, decl, eigencut_window(model, -0.05))
    assert report.passed, report

    path = tmp_path / "sym.nc"
    save_symmetry(path, sga, declaration=decl, report=report)
    rec = load_symmetry(path)

    assert rec.declaration == decl  # frozen-dataclass equality, field-for-field
    r2 = rec.report
    assert r2 is not None
    assert r2.label_source == report.label_source
    assert len(r2.checks) == len(report.checks)
    for c1, c2 in zip(report.checks, r2.checks):
        np.testing.assert_allclose(c2.qpoint, c1.qpoint, atol=1e-15)
        assert c2.expected == c1.expected
        assert c2.found == c1.found
        assert c2.passed == c1.passed
    # the report re-attaches the same declaration
    assert rec.report.declaration == decl


# ===========================================================================
# TEST-003: basis provenance block (fingerprints + molien rows/notes)
# ===========================================================================
def test_basis_provenance_roundtrip(tmp_path, sga):
    from lawaf.anharmonic.basis import build_invariant_basis, build_oh_action, molien_check

    basis = build_invariant_basis(
        build_oh_action(nlwf=3),
        nlwf=3,
        Rlist=[(0, 0, 0)],
        orders=(2, 3, 4),
        include_strain=True,
        max_strain_power=2,
    )
    path = tmp_path / "sym.nc"
    save_symmetry(path, sga, basis=basis)
    rec = load_symmetry(path)

    assert rec.basis_fingerprint == basis.fingerprint
    assert rec.action_fingerprint == action_fingerprint(sga)
    fresh = molien_check(basis)
    rows = rec.molien_rows
    assert rows is not None and len(rows) == len(fresh.rows)
    for r1, r2 in zip(fresh.rows, rows):
        assert (r2.order, r2.sector) == (r1.order, r1.sector)
        assert r2.molien == r1.molien
        assert r2.constructed == r1.constructed
        assert r2.consistent == r1.consistent
        assert r2.note == r1.note
    assert any(r.note for r in rows)
    assert any("capped" in r.note for r in rows)


# ===========================================================================
# TEST-004: coexistence with the story-022 'anharmonic' group
# ===========================================================================
def _group_bytes(path, group):
    """(variables raw bytes, attrs sans 'created') of one netCDF group.

    Float attrs are NaN-normalized: NaN payloads are bit-identical across
    files but NaN != NaN would break the dict comparison.
    """
    import netCDF4

    def _norm(v):
        return "nan" if isinstance(v, float) and np.isnan(v) else v

    with netCDF4.Dataset(str(path), "r") as root:
        g = root.groups[group]
        vars_b = {n: np.array(v[:]).tobytes() for n, v in g.variables.items()}
        attrs = {
            n: _norm(g.getncattr(n))
            for n in g.ncattrs()
            if n != "created"
        }
    return vars_b, attrs


def test_coexistence_both_write_orders(tmp_path, sga):
    from lawaf.anharmonic.io import load_anharmonic_model, save_anharmonic_model

    res = _coexistence_coefficients()
    ref_a, ref_s = tmp_path / "ref_anh.nc", tmp_path / "ref_sym.nc"
    save_anharmonic_model(ref_a, res)
    save_symmetry(ref_s, sga)

    # each loader refuses a file that only carries the OTHER group
    with pytest.raises(ValueError, match="symmetry"):
        load_symmetry(ref_a)
    with pytest.raises(ValueError, match="anharmonic"):
        load_anharmonic_model(ref_s)

    anh_b, anh_attrs = _group_bytes(ref_a, "anharmonic")
    sym_b, sym_attrs = _group_bytes(ref_s, "symmetry")

    for name, both in (
        ("anharmonic-first", tmp_path / "both1.nc"),
        ("symmetry-first", tmp_path / "both2.nc"),
    ):
        if name == "anharmonic-first":
            save_anharmonic_model(both, res)
            save_symmetry(both, sga)
        else:
            save_symmetry(both, sga)
            save_anharmonic_model(both, res)
        # both groups present; each loader reads only its own group
        rec = load_symmetry(both)
        assert rec.sga.n_ops == 48
        model = load_anharmonic_model(both)
        np.testing.assert_array_equal(
            model.coefficients.coefficients, res.coefficients
        )
        # bit-identity of the sibling group's data vs the direct save
        b, a = _group_bytes(both, "anharmonic")
        assert b == anh_b and a == anh_attrs
        b, a = _group_bytes(both, "symmetry")
        assert b == sym_b and a == sym_attrs


# ===========================================================================
# TEST-005: schema guard + overwrite refusal
# ===========================================================================
def test_schema_version_guard(tmp_path, sga):
    import netCDF4

    path = tmp_path / "sym.nc"
    save_symmetry(path, sga)
    with netCDF4.Dataset(str(path), "a") as root:
        root.groups["symmetry"].setncattr(
            "schema_version", SYMMETRY_SCHEMA_VERSION + 1
        )
    with pytest.raises(ValueError, match="NEWER"):
        load_symmetry(path)


def test_overwrite_refusal(tmp_path, sga):
    path = tmp_path / "sym.nc"
    save_symmetry(path, sga)
    with pytest.raises(ValueError, match="already carries"):
        save_symmetry(path, sga)
