"""Story-015 tests: subspace covariance (FR-002) and declared-representation
compatibility (FR-003) on the BaTiO3 ``DM_dip_wang`` fixture (Pm-3m, #221).

Fixtures and loading follow tests/test_anharmonic_representation.py (story-014).

Conventions under test (all validated here, machinery in
``lawaf/anharmonic/compatibility.py``):

Covariance (FR-002)
    A retained window with projector P(q) is *covariant* iff
    ``P(g.q) = S_g(q) P(q) S_g(q)^dagger`` for every operation g.  A window
    that is a union of complete degenerate blocks at every point of the
    (star-closed) mesh satisfies this; cropping one column of a degenerate
    block breaks it at O(1).

Declared representation (FR-003)
    ``RepresentationDeclaration(wyckoff, site_irreps)`` is compatible with a
    window iff the little-group decomposition of the induced character
    (Sakuma induction from the declared Wyckoff-orbit site irrep) equals the
    decomposition of the window characters ``chi(g) = tr S_g(q) P(q)``.
    Irreducible characters are extracted numerically from the regular
    representation of the little group (class sums + joint diagonalization)
    and named against derived reference characters (polar/axial/component
    forms); the O_h naming is checked against the published character table.

Sympy verification (AGENTS.md rule): executed asserts live in
``docs/derivations/story015_induced_characters_sympy.py`` (induced-character
formula on a symbolic group, character orthogonality, multiplicity formula,
sym^2/asym^2 identities) and are re-run by ``test_sympy_derivation_script``.
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")
sympy = pytest.importorskip("sympy")

from lawaf.anharmonic.compatibility import (  # noqa: E402
    CharacterLabels,
    CompatCheck,
    RepresentationDeclaration,
    RepresentationLabels,
    RepresentationReport,
    assert_compatible,
    assert_covariance,
    character_table,
    check_compatibility,
    check_subspace_covariance,
    induced_characters,
    little_group,
    resolve_site_irrep,
)
from lawaf.anharmonic.representation import build_space_group_action  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
DERIVATION = (
    Path(__file__).parent.parent
    / "docs"
    / "derivations"
    / "story015_induced_characters_sympy.py"
)

GAMMA = (0.0, 0.0, 0.0)
# published O_h character table, class order (E, 8C3, 6C2, 6C4, 3C2,
# i, 6S4, 8S6, 3sigma_h, 6sigma_d) -- ground truth for the extracted/named
# little-group table at Gamma (Cotton, "Chemical Applications of Group Theory").
OH_TABLE = {
    # derived from the direct product O x Ci (proper part + parity rule),
    # verified by row orthogonality (all 100 pairs) and the polar/axial
    # physical anchors chi_polar = 1 + 2cos(theta), chi_S4 = -1 + 2cos(theta)
    "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "A2g": [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
    "Eg": [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
    "T1g": [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
    "T2g": [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
    "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
    "A2u": [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
    "Eu": [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
    "T1u": [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
    "T2u": [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],
}
# canonical class keys (size, sorted (det R, tr R) over members) in the same
# class order as OH_TABLE; used to align the extracted table's class order.
OH_CLASS_KEYS = [
    # (class size, frozenset of (det R, tr R) over members), published order
    # (E, 8C3, 6C2, 6C4, 3C2, i, 6S4, 8S6, 3sigma_h, 6sigma_d)
    (1, frozenset({(1.0, 3.0)})),    # E
    (8, frozenset({(1.0, 0.0)})),    # 8C3
    (6, frozenset({(1.0, -1.0)})),   # 6C2 (face diagonals)
    (6, frozenset({(1.0, 1.0)})),    # 6C4
    (3, frozenset({(1.0, -1.0)})),   # 3C2 (coordinate axes)
    (1, frozenset({(-1.0, -3.0)})),  # i
    (6, frozenset({(-1.0, -1.0)})),  # 6S4
    (8, frozenset({(-1.0, 0.0)})),   # 8S6
    (3, frozenset({(-1.0, 1.0)})),   # 3 sigma_h
    (6, frozenset({(-1.0, 1.0)})),   # 6 sigma_d
]

@pytest.fixture(scope="module")
def phonon():
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def model(phonon):
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    return PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)


@pytest.fixture(scope="module")
def sga(phonon):
    return build_space_group_action(phonon)


@pytest.fixture(scope="module")
def atom_letters(sga):
    letters = sga.symmetry_dataset.wyckoffs
    return {let: i for i, let in enumerate(letters)}


def eigencut_window(model, cut, gamma_drop=None):
    """projector_fn: eigenvalue-cut window (omega^2 < cut) from wrapper evecs.

    ``gamma_drop``: band index excluded from the window at Gamma only
    (negative control; breaks block completeness of the 3-fold soft block).
    """

    cache = {}

    def projector_fn(q):
        key = tuple(np.round(q, 8) % 1.0)
        if key not in cache:
            cache[key] = model.solve(q)
        evals, evecs = cache[key]
        cols = np.where(evals < cut)[0]
        if gamma_drop is not None and np.linalg.norm(np.round(q, 8) % 1.0) < 1e-8:
            cols = cols[cols != gamma_drop]
        V = evecs[:, cols]
        return V @ V.conj().T

    return projector_fn


# ======================================================================
# TEST-001  subspace covariance (FR-002)
# ======================================================================
def test_covariance_block_complete_window_passes(sga, model):
    """Cut-0.2 window is block-complete at all TRIM points: passes at 1e-10."""
    result = check_subspace_covariance(sga, eigencut_window(model, 0.2), tol=1e-10)
    assert result["passed"], result
    # default point set: star closure of the irreducible 2x2x2 mesh =
    # Gamma + 3 X-half-axes + 3 M-face-diagonals + R
    assert len(result["points"]) == 8
    # one residual entry per (point, operation)
    assert len(result["residuals"]) == 8 * sga.n_ops
    assert_covariance(result)  # must not raise


def test_covariance_soft_window_passes(sga, model):
    """Soft-mode window (omega^2 < -0.05): the unstable T1u triplet is
    block-complete; empty window at R (rank 0) is handled."""
    result = check_subspace_covariance(sga, eigencut_window(model, -0.05), tol=1e-10)
    assert result["passed"], result
    assert result["worst"] <= 1e-10
    ranks = {r["rank"] for r in result["residuals"]}
    assert 0 in ranks  # R point has no unstable branch: rank-0 window


def test_covariance_cropped_window_fails_naming_op_and_q(sga, model):
    """Dropping one column of the 3-fold soft block at Gamma breaks covariance;
    assert_covariance names the offending op index and q."""
    result = check_subspace_covariance(
        sga, eigencut_window(model, 0.2, gamma_drop=8), tol=1e-10
    )
    assert not result["passed"]
    worst = result["worst_residual"]
    assert worst["residual"] > 1e-6
    assert tuple(np.round(worst["q"], 8)) == GAMMA
    with pytest.raises(ValueError, match=r"op \d+") as exc:
        assert_covariance(result)
    msg = str(exc.value)
    assert str(worst["op"]) in msg
    assert "Gamma" in msg


def test_projector_fn_must_return_hermitian_projector(sga, model):
    def bad_fn(q):
        _e, v = model.solve(q)
        # cross-block Gram matrix: not Hermitian, not idempotent
        return v[:, :3] @ v[:, 3:6].conj().T

    with pytest.raises(ValueError, match="[Hh]ermitian"):
        check_subspace_covariance(sga, bad_fn, qpoints=[GAMMA])

# ======================================================================
# TEST-002  little-group character tables (extraction + naming)
# ======================================================================
def test_gamma_table_matches_published_oh(sga):
    """Extracted O_h table: 10 classes/irreps, orthonormal rows, named rows
    equal the published character table (canonical class order)."""
    from lawaf.anharmonic.compatibility import character_table

    gamma = np.zeros(3)
    ops = [g for g in range(sga.n_ops) if np.linalg.norm(sga.qmap(g, gamma)) < 1e-8]
    ct = character_table(sga, ops)
    assert len(ct.classes) == 10
    assert ct.dims == sorted(ct.dims, reverse=True)
    assert sum(d * d for d in ct.dims) == 48
    sizes = ct.class_sizes
    gram = (ct.chars * sizes[None, :]) @ ct.chars.conj().T / len(ops)
    assert np.allclose(gram, np.eye(10), atol=1e-8)
    keys = []
    for cl in ct.classes:
        sig = frozenset(
            (round(float(np.linalg.det(sga.cart_rotations[g])), 6),
             round(float(np.trace(sga.cart_rotations[g])), 6))
            for g in cl
        )
        keys.append((len(cl), sig))
    # extracted class order is first-seen; align to the published order
    # through the canonical (size, sorted (det, tr)) class keys
    assert sorted(keys, key=lambda k: (k[0], sorted(k[1]))) == sorted(
        OH_CLASS_KEYS, key=lambda k: (k[0], sorted(k[1]))
    )
    perm = [keys.index(k) for k in OH_CLASS_KEYS]
    for name, row in OH_TABLE.items():
        i = ct.names.index(name)
        assert np.allclose(ct.chars[i][perm], row, atol=1e-6), name


def test_site_tables_and_site_vector_rep(sga, atom_letters):
    """Ti (1b, m-3m) and O (3c, 4/mm.m) site tables; the site vector rep at
    1b contains T1u and at 3c contains A2u + Eu."""
    from lawaf.anharmonic.compatibility import character_table, little_group

    gamma = np.zeros(3)
    lg = little_group(sga, gamma)
    ti = atom_letters["b"]
    ox = atom_letters["c"]
    for kappa, size, contains in ((ti, 48, ["T1u"]), (ox, 16, ["A2u", "Eu"])):
        site_ops = [g for g in lg if sga.atom_maps[g][kappa] == kappa]
        assert len(site_ops) == size
        ct = character_table(sga, site_ops)
        chi_vec = np.array(
            [np.trace(sga.cart_rotations[g]) for g in site_ops]
        )
        mult = ct.multiplicities(chi_vec, per_op=True)
        labels = ct.labels(mult)
        assert sorted(labels) == sorted(contains), (kappa, labels)


# ======================================================================
# TEST-003  induction (Sakuma) from the declared Wyckoff-orbit site irrep
# ======================================================================
def test_induced_t1u_from_1b_reproduces_soft_decomposition(sga, model, atom_letters):
    """Story acceptance: inducing T1u from the 1b Ti orbit reproduces the
    Gamma decomposition of the 3 lowest optical (soft) branches."""
    from lawaf.anharmonic.compatibility import (
        CharacterLabels,
        check_compatibility,
        induced_characters,
        little_group,
        resolve_site_irrep,
    )

    gamma = np.zeros(3)
    lg = little_group(sga, gamma)
    ti = atom_letters["b"]
    site_ops = [g for g in lg if sga.atom_maps[g][ti] == ti]
    ct_site = resolve_site_irrep(sga, site_ops, "T1u")  # (chars, name, dim)
    chi_site, name, dim = ct_site
    assert name == "T1u" and dim == 3
    chi_ind = induced_characters(sga, lg, site_ops, chi_site)
    labels = CharacterLabels(sga, eigencut_window(model, -0.05))
    found = labels.irreps_at(gamma)
    # decompose the induced character and compare
    from lawaf.anharmonic.compatibility import character_table

    ct_lg = character_table(sga, lg)
    expected = ct_lg.labels(ct_lg.multiplicities(chi_ind, per_op=True))
    assert expected == ["T1u"]
    assert found == ["T1u"]
    assert expected == found


# ======================================================================
# TEST-004  declared-representation compatibility (FR-003)
# ======================================================================
def test_correct_declaration_passes(sga, model, atom_letters):
    """(1b, ['T1u']) with the soft-mode window: compatible at Gamma."""
    declaration = RepresentationDeclaration(
        wyckoff="1b", site_irreps=["T1u"], strain_sector=True
    )
    labels = CharacterLabels(sga, eigencut_window(model, -0.05))
    report = check_compatibility(sga, declaration, labels, qpoints=[GAMMA])
    assert isinstance(report, RepresentationReport)
    assert report.label_source == "characters"
    assert report.passed, report
    (check,) = report.checks
    assert isinstance(check, CompatCheck)
    assert check.expected == ["T1u"] and check.found == ["T1u"]
    assert_compatible(report)  # must not raise
    assert isinstance(labels, RepresentationLabels)  # protocol satisfied


def test_correct_declaration_from_projector_fn(sga, model):
    """Passing a raw projector_fn wraps it in CharacterLabels ('characters')."""
    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u"])
    report = check_compatibility(
        sga, declaration, eigencut_window(model, -0.05), qpoints=[GAMMA]
    )
    assert report.label_source == "characters"
    assert report.passed, report


def test_multiplicity_window_matches_t1u_cubed(sga, model):
    """9-branch window (3 x T1u at Gamma) matches the triple declaration."""
    declaration = RepresentationDeclaration(
        wyckoff="1b", site_irreps=["T1u", "T1u", "T1u"]
    )
    report = check_compatibility(
        sga, declaration, eigencut_window(model, 0.2), qpoints=[GAMMA]
    )
    assert report.passed, report
    (check,) = report.checks
    assert check.expected == ["T1u"] * 3 and check.found == ["T1u"] * 3


def test_wrong_site_irrep_3c_eg_raises(sga, model, atom_letters):
    """O-site (3c) declaration with E_g: report fails and assert_compatible
    raises naming Gamma and the expected-vs-found irreps."""
    declaration = RepresentationDeclaration(wyckoff="3c", site_irreps=["Eg"])
    report = check_compatibility(
        sga, declaration, eigencut_window(model, -0.05), qpoints=[GAMMA]
    )
    assert not report.passed
    (check,) = report.checks
    # Eg at the D4h O site induces to even Gamma irreps: T1g + T2g
    assert "T1g" in check.expected and "T2g" in check.expected
    assert check.found == ["T1u"]
    with pytest.raises(ValueError) as exc:
        assert_compatible(report)
    msg = str(exc.value)
    assert "Gamma" in msg
    assert "Eg" in msg and "T1u" in msg


def test_wrong_site_irrep_3c_eu_raises(sga, model):
    """O-site declaration with Eu: induced T1u+T2u vs found T1u -> mismatch."""
    declaration = RepresentationDeclaration(wyckoff="3c", site_irreps=["Eu"])
    report = check_compatibility(
        sga, declaration, eigencut_window(model, -0.05), qpoints=[GAMMA]
    )
    assert not report.passed
    (check,) = report.checks
    assert sorted(check.expected) == ["T1u", "T2u"]
    assert check.found == ["T1u"]
    with pytest.raises(ValueError, match="T2u"):
        assert_compatible(report)


def test_multiplicity_mismatch_raises(sga, model):
    """(1b, 2 x T1u) against a 1 x T1u window: multiplicity mismatch."""
    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u", "T1u"])
    report = check_compatibility(
        sga, declaration, eigencut_window(model, -0.05), qpoints=[GAMMA]
    )
    assert not report.passed
    with pytest.raises(ValueError) as exc:
        assert_compatible(report)
    msg = str(exc.value)
    assert "Gamma" in msg and msg.count("T1u") >= 3  # expected twice + found once


def test_unknown_site_irrep_name_raises(sga, atom_letters):
    """An unresolvable site-irrep name lists the available names."""
    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["Xyz"])
    labels = CharacterLabels(sga, eigencut_window(None, -0.05))
    with pytest.raises(ValueError, match="Xyz"):
        check_compatibility(sga, declaration, labels, qpoints=[GAMMA])


def test_unknown_wyckoff_letter_raises(sga, model):
    declaration = RepresentationDeclaration(wyckoff="9z", site_irreps=["T1u"])
    with pytest.raises(ValueError, match="9z"):
        check_compatibility(
            sga, declaration, eigencut_window(model, -0.05), qpoints=[GAMMA]
        )


def test_provided_labels_object_is_used_verbatim(sga):
    """A custom RepresentationLabels provider is consumed directly."""

    class MyLabels:
        def irreps_at(self, qpoint):
            return ["T1u"]

    declaration = RepresentationDeclaration(wyckoff="1b", site_irreps=["T1u"])
    report = check_compatibility(sga, declaration, MyLabels(), qpoints=[GAMMA])
    assert report.label_source == "provided"
    assert report.passed, report


# ======================================================================
# TEST-005  sympy verification (AGENTS.md rule)
# ======================================================================
def test_sympy_symbolic_identities_in_process():
    """Fast in-process sympy asserts of the character-arithmetic identities
    the implementation relies on (sym^2 / asym^2 traces)."""
    a, b, c, d = sympy.symbols("a b c d")
    M = sympy.Matrix([[a, b], [c, d]])
    M2 = M @ M
    chi = M.trace()
    chi2 = M2.trace()
    # tr(Sym^2 M) = ((tr M)^2 + tr M^2) / 2  and  tr(Asym^2 M) = ((tr M)^2 - tr M^2)/2
    assert sympy.simplify(((chi**2 + chi2) / 2) - (a**2 + d**2 + b * c + a * d)) == 0
    sym2 = sympy.simplify((chi**2 + chi2) / 2)
    asym2 = sympy.simplify((chi**2 - chi2) / 2)
    assert sym2 == (a**2 + d**2 + a * d + b * c)
    assert asym2 == (a * d - b * c)
    # explicit Sym^2 / Asym^2 bases of C^2 (x^2, y^2+xy split as below):
    # Sym^2 basis {e1e1, e1e2+e2e1, e2e2}, char traces must equal sym2 above
    B_sym = [
        sympy.Matrix([[1, 0], [0, 0]]),
        sympy.Matrix([[0, 1], [1, 0]]) / sympy.sqrt(2),
        sympy.Matrix([[0, 0], [0, 1]]),
    ]
    tr_sym = sympy.trace(M * B_sym[0] * M.T * B_sym[0]) + sympy.trace(
        M * B_sym[1] * M.T * B_sym[1]
    ) + sympy.trace(M * B_sym[2] * M.T * B_sym[2])
    assert sympy.simplify(tr_sym - sym2) == 0


def test_sympy_derivation_script():
    """The full executed sympy derivation (induced-character formula on a
    symbolic group, orthogonality, multiplicity formula, O_h naming anchors)
    runs green."""
    r = subprocess.run(
        [sys.executable, str(DERIVATION)], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout
