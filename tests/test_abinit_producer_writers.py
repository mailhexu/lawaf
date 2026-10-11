"""Story-064 tests: NC producer + streaming writers (AbinitWannierInputs).

Conventions (pinned by the story and the file-handoff research memo):
- lattice/geometry: source bohr rows x the pinned CODATA2022 Bohr constant
  0.529177210544; the WIN prints 12 decimals (<= 5e-13 A quantization) and
  the NNKP f12.7 prints are lossy validation metadata only;
- A = Psi^dagger G (state bra / trial ket); MMN transport
  b = k' + g - k with the integer index pairing offset by the image shift g;
- EIG in eV (Hartree x 27.211386245981); band map applied once, no
  exclude_bands.

Real-fixture/real-executable tests need the story-060 GaAs NC WFK
(.tmp/w90-story060/nc-scalar/) and the pinned wannier90.x builds
(.tmp/w90-lib-research/build-dev-{serial,mpi}/wannier90.x); they skip with
an explicit message when absent.  No mock readers, fake libraries or fake
executables anywhere.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_LAWAF_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _LAWAF_PKG_ROOT not in sys.path:
    sys.path.insert(0, _LAWAF_PKG_ROOT)

from lawaf.io.w90 import (
    read_mmn,
    write_amn,
    write_eig,
    write_mmn,
    write_mmn_blocks,
    write_win,
)
from lawaf.interfaces.abinit.producer import (
    BOHR_ANG,
    HA_EV,
    AbinitWannierInputs,
    GenerationPlan,
    Nnkp,
    ProducerError,
    parse_nnkp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALAR_NC = REPO_ROOT / ".tmp/w90-story060/nc-scalar/gaas-nc3o_DS1_WFK.nc"
SCALAR_BIN = REPO_ROOT / ".tmp/w90-story060/nc-scalar/gaas-nco_DS1_WFK"
SI_TUT = REPO_ROOT / ".tmp/w90-lib-research/src-dev/tutorials/tutorial16-noqe"
W90_SERIAL = REPO_ROOT / ".tmp/w90-lib-research/build-dev-serial/wannier90.x"
W90_MPI = REPO_ROOT / ".tmp/w90-lib-research/build-dev-mpi/wannier90.x"
SCRATCH = REPO_ROOT / ".tmp/w90-story064"

needs_scalar = pytest.mark.skipif(
    not SCALAR_NC.exists() or not SCALAR_BIN.exists(),
    reason="scalar WFK fixtures not generated (story 060)",
)
needs_w90 = pytest.mark.skipif(
    not W90_SERIAL.exists(),
    reason="pinned serial wannier90.x not available "
    "(.tmp/w90-lib-research/build-dev-serial)",
)
needs_w90_mpi = pytest.mark.skipif(
    not W90_MPI.exists(),
    reason="pinned MPI wannier90.x not available "
    "(.tmp/w90-lib-research/build-dev-mpi)",
)


# ---------------------------------------------------------------------------
# synthetic bounded mesh source (caller-array carrier; real arrays, no mocks)
# ---------------------------------------------------------------------------


def _g_sphere(kpt, ecut, gmet):
    """Full integer-G sphere: 0.5 (k+g)' gmet (k+g) <= ecut."""
    out = []
    rng = int(np.ceil(np.sqrt(2 * ecut / min(np.linalg.eigvalsh(gmet)))) + 2)
    span = range(-rng, rng + 1)
    for g in ((i, j, k) for i in span for j in span for k in span):
        g = np.array(g, dtype=float)
        if 0.5 * (kpt + g) @ gmet @ (kpt + g) <= ecut + 1e-12:
            out.append(g.astype(int))
    return np.array(sorted(out, key=tuple), dtype=np.int32)


def _mesh_source(nmesh, nband=4, seed=7, ecut=0.8, a=10.0, finite=True):
    """Gamma-centered nmesh^3 synthetic caller-array source, full-storage
    (istwfk=1) plane-wave states with deterministic random states."""
    from lawaf.interfaces.abinit.source_arrays import (
        CallerArraySource,
        CallerMetadata,
        SymmetryMapEntry,
    )

    rng = np.random.default_rng(seed)
    gmet = (2.0 * np.pi / a) ** 2 * np.eye(3)
    kpts = np.array(
        [
            [i / nmesh, j / nmesh, k / nmesh]
            for i in range(nmesh)
            for j in range(nmesh)
            for k in range(nmesh)
        ],
        dtype=float,
    )
    nk = len(kpts)
    states, kg_arrays = {}, {}
    npw = []
    for ik, k in enumerate(kpts):
        kg = _g_sphere(k, ecut, gmet)
        npw.append(len(kg))
        cg = rng.normal(size=(nband, len(kg))) + 1j * rng.normal(size=(nband, len(kg)))
        if not finite:
            cg[0, 0] = np.nan + 1j * np.nan
        states[(ik, 0)] = np.ascontiguousarray(cg[:, :, None], dtype=np.complex128)
        kg_arrays[ik] = np.ascontiguousarray(kg.T)  # (3, npw)
    from dataclasses import replace
    from test_abinit_wfk_reader import synthetic_header

    header = replace(
        synthetic_header(kpts, npw, nband, ecut=ecut),
        rprimd=a * np.eye(3),
        istwfk_raw=np.ones(nk, np.int32),
        istwfk=np.ones(nk, np.int32),
        kptrlatt=np.diag([nmesh] * 3).astype(np.int32),
        kptrlatt_orig=np.diag([nmesh] * 3).astype(np.int32),
        md5_pseudos=("",),
    )
    meta = CallerMetadata(
        header=header,
        eigenvalues=rng.normal(size=(nband, nk, 1)),
        eigenvalue_units="hartree",
        length_units="bohr",
        nkpt_mesh=nk,
        mesh_kpoints=kpts,
        symmetry_map=tuple(
            SymmetryMapEntry(
                i,
                i,
                np.eye(3, dtype=int),
                np.zeros(3),
                np.zeros(3, dtype=int),
                "same",
                False,
            )
            for i in range(nk)
        ),
    )
    return CallerArraySource(meta, states, kg_arrays), gmet


def _gaussian_trials(source, n_per_site=1, alpha=1.0):
    """Plane-wave gaussian trial orbitals at each atom (a real, closed-form
    projection supplier).  Returns the (k, kg) -> (N, n_per_site*natom)
    callable."""
    h = source.header
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(h.rprimd)).T  # inverse bohr

    def trials(ik, kg):
        k = np.asarray(h.kpt[ik], dtype=float)
        kpg = (k + kg) @ recip
        cols = []
        for site in np.asarray(h.xred, dtype=float):
            rc = site @ np.asarray(h.rprimd)
            phase = np.exp(-1j * (kpg @ rc))
            volume = abs(np.linalg.det(h.rprimd))
            for trial in range(n_per_site):
                width = alpha * (1 + 0.3 * trial)
                norm = (2 * width / np.pi) ** 0.75
                cols.append(
                    norm
                    * (np.pi / width) ** 1.5
                    / np.sqrt(volume)
                    * np.exp(-np.sum(kpg**2, axis=1) / (4 * width))
                    * phase
                )
        return np.stack(cols, axis=1)

    return trials


def _mesh_neighbors(kpts, nmesh):
    """NNKP-style (nk, 6, 4) table: the +/- axis neighbours with their
    image shifts (0-based k', then g); g solves k2 + g = k1 + b."""
    kpts = np.asarray(kpts)
    nk = len(kpts)
    idx_exact = {
        tuple(np.round(k * nmesh).astype(int) % nmesh): i for i, k in enumerate(kpts)
    }
    cells = np.round(kpts * nmesh).astype(int)
    table = np.zeros((nk, 6, 4), dtype=int)
    for i, cell in enumerate(cells):
        for slot, (ax, sgn) in enumerate(
            ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))
        ):
            tgt = cell.copy()
            tgt[ax] += sgn
            k2 = tgt % nmesh
            g = np.zeros(3, dtype=int)
            g[ax] = (cell[ax] + sgn - k2[ax]) // nmesh
            # neighbour point: exact mesh hit; otherwise keep the listed
            # point itself with an explicit unit image shift (unique keys)
            j = idx_exact.get(tuple(k2.tolist()))
            if j is None:
                j = i
                g = np.zeros(3, dtype=int)
                g[ax] = sgn
            table[i, slot] = (j, g[0], g[1], g[2])
    return table


def _write_nnkp(path, source, table, lattice_noise=0.0, kpt_noise=0.0):
    """Write a minimal -pp-style NNKP for the synthetic source."""
    h = source.header
    lat = np.asarray(h.rprimd) * BOHR_ANG + lattice_noise
    nk = int(h.nkpt)
    lines = ["File written by test", "", "calc_only_A  :  F", "", "begin real_lattice"]
    for row in lat:
        lines.append("  " + "  ".join(f"{x:12.7f}" for x in row))
    lines += ["end real_lattice", "", "begin recip_lattice"]
    recip = 2 * np.pi * np.linalg.inv(np.asarray(h.rprimd)).T * BOHR_ANG
    for row in recip:
        lines.append("  " + "  ".join(f"{x:12.7f}" for x in row))
    lines += ["end recip_lattice", "", "begin fractional kpoints", f"{nk}"]
    for k in np.asarray(h.kpt) + kpt_noise:
        lines.append("  " + "  ".join(f"{x:12.8f}" for x in k))
    lines += [
        "end fractional kpoints",
        "",
        "begin projection",
        "  random",
        "end projection",
        "",
        "begin nnkpts",
    ]
    for ik in range(nk):
        for slot in range(table.shape[1]):
            jk, g1, g2, g3 = table[ik, slot]
            lines.append(f"{ik + 1:5d} {jk + 1:5d} " f"{g1:5d} {g2:5d} {g3:5d}")
    lines += ["end nnkpts", ""]
    Path(path).write_text("\n".join(lines))
    return Path(path)


def _plan(nband=4, nw=3, nmesh=2):
    return GenerationPlan(
        band_map=tuple(range(nband)),
        num_wann=nw,
        mp_grid=(nmesh, nmesh, nmesh),
        controls={"num_iter": "100", "num_print_cycles": "50"},
    )


def _differs(a, b, atol=1e-8):
    """Shape-aware 'is a different matrix' (mutation-fault detector)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape != b.shape or not np.allclose(a, b, atol=atol)


@pytest.fixture()
def setup(tmp_path):
    source, _gmet = _mesh_source(2, nband=4)
    table = _mesh_neighbors(source.header.kpt, 2)
    nnkp = _write_nnkp(tmp_path / "syn.nnkp", source, table)
    trials = _gaussian_trials(source, n_per_site=3, alpha=0.9)
    plan = _plan()
    prod = AbinitWannierInputs(source, plan, trial_orbitals=trials)
    return source, nnkp, trials, plan, prod


# ---------------------------------------------------------------------------
# TEST-064-01: WIN grammar/precision/no-double-exclusion
# ---------------------------------------------------------------------------


class TestWin:
    def test_win_preserves_geometry_precision_and_counts(self, tmp_path):
        source, _gmet = _mesh_source(2)
        prod = AbinitWannierInputs(source, _plan())
        win = prod.write_win(tmp_path, "syn")
        text = win.read_text()
        # no exclude_bands anywhere; num_bands == retained count
        assert "exclude_bands" not in text
        assert re.search(r"^num_bands\s*=\s*4$", text, re.M)
        assert re.search(r"^num_wann\s*=\s*3$", text, re.M)
        # lattice rows: source bohr x pinned Bohr at full writer precision
        expect = np.asarray(source.header.rprimd) * BOHR_ANG
        lat = np.array(
            [
                [float(x) for x in ln.split()]
                for ln in text.split("begin unit_cell_cart")[1]
                .split("end unit_cell_cart")[0]
                .strip()
                .splitlines()[1:]
            ]
        )
        np.testing.assert_allclose(lat, expect, atol=5e-13)
        # explicit kpoints in generation order
        kblock = text.split("begin kpoints")[1].split("end kpoints")[0]
        kpts = np.array(
            [[float(x) for x in ln.split()] for ln in kblock.strip().splitlines()]
        )
        np.testing.assert_allclose(kpts, np.asarray(source.header.kpt), atol=1e-15)
        # plan window/localization controls present
        assert "num_iter = 100" in text

    def test_win_rejects_exclude_bands_double_filtering(self, tmp_path):
        source, _gmet = _mesh_source(2)
        plan = GenerationPlan(
            band_map=(0, 1),
            num_wann=1,
            mp_grid=(2, 2, 2),
            controls={"exclude_bands": "1"},
        )
        with pytest.raises(ProducerError, match="D4.*exclude_bands"):
            AbinitWannierInputs(source, plan).write_win(tmp_path, "syn")


# ---------------------------------------------------------------------------
# TEST-067-A: native-contract companion pair (approved option A)
# ---------------------------------------------------------------------------


class TestNativeCompanion:
    def test_companion_win_full_frame_with_exclude_runs(self, tmp_path):
        """Companion WIN: num_bands = full mband, dropped bands as
        exclude_bands runs, geometry identical, D4 guard not triggered."""
        source, _gmet = _mesh_source(2, nband=6)
        plan = GenerationPlan(
            band_map=(1, 2, 4),
            num_wann=2,
            mp_grid=(2, 2, 2),
            controls={"num_iter": "100"},
        )
        prod = AbinitWannierInputs(source, plan)
        win, eig = prod.write_native_companion(tmp_path, "syn")
        assert win == tmp_path / "syn-native.win"
        text = win.read_text()
        assert re.search(r"^num_bands\s*=\s*6$", text, re.M)
        assert re.search(r"^num_wann\s*=\s*2$", text, re.M)
        # dropped original 1-based bands {1, 3, 5, 6} printed as runs
        m = re.search(r"^exclude_bands\s*=\s*(\S+)$", text, re.M)
        assert m, "companion win must carry exclude_bands"
        dropped = []
        for part in m.group(1).split(","):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                dropped.extend(range(lo, hi + 1))
            else:
                dropped.append(int(part))
        assert dropped == [1, 4, 6]
        # geometry matches the external win
        ext = prod.write_win(tmp_path, "syn")
        for block in ("unit_cell_cart", "atoms_frac", "kpoints"):
            got = text.split(f"begin {block}")[1].split(f"end {block}")[0]
            ref = ext.read_text().split(f"begin {block}")[1].split(f"end {block}")[0]
            assert got.strip() == ref.strip(), block

    def test_companion_win_no_exclude_when_all_bands_retained(self, tmp_path):
        source, _gmet = _mesh_source(2, nband=4)
        prod = AbinitWannierInputs(source, _plan())
        win, _eig = prod.write_native_companion(tmp_path, "syn")
        text = win.read_text()
        assert re.search(r"^num_bands\s*=\s*4$", text, re.M)
        assert "exclude_bands" not in text

    def test_companion_eig_full_frame_in_ev(self, tmp_path):
        """Companion EIG: all mband rows per k-point, source Hartree x
        HA_EV, strict k-outer/band-inner order."""
        source, _gmet = _mesh_source(2, nband=6)
        plan = GenerationPlan(
            band_map=(1, 3), num_wann=2, mp_grid=(2, 2, 2), controls={}
        )
        prod = AbinitWannierInputs(source, plan)
        _win, eig = prod.write_native_companion(tmp_path, "syn")
        rows = [
            (int(a), int(b), float(c))
            for a, b, c in (ln.split() for ln in eig.read_text().splitlines())
        ]
        nk = int(source.header.nkpt)
        assert len(rows) == nk * 6
        assert [r[0] for r in rows] == list(range(1, 7)) * nk
        assert [r[1] for r in rows] == [ik + 1 for ik in range(nk) for _b in range(6)]
        for ik in range(nk):
            ref = np.asarray(source.eigenvalues(ik, 0, (0, 6))) * 27.211386245981
            got = np.array([r[2] for r in rows[ik * 6:(ik + 1) * 6]])
            np.testing.assert_allclose(got, ref, atol=5e-12)

    def test_external_win_guard_unchanged(self, tmp_path):
        """The D4 no-double-filtering guard still applies to the external
        pair; the companion is an additional output, not a control path."""
        source, _gmet = _mesh_source(2, nband=4)
        prod = AbinitWannierInputs(source, _plan())
        prod.write_native_companion(tmp_path, "syn")
        plan = GenerationPlan(
            band_map=(0, 1),
            num_wann=1,
            mp_grid=(2, 2, 2),
            controls={"exclude_bands": "1"},
        )
        with pytest.raises(ProducerError, match="D4.*exclude_bands"):
            AbinitWannierInputs(source, plan).write_win(tmp_path, "other")


# ---------------------------------------------------------------------------
# TEST-064-02: NNKP parse/validation incl. rounding bounds and key mismatch
# ---------------------------------------------------------------------------


class TestNnkp:
    def test_parse_real_fixture_grammar(self):
        """The in-repo -pp fixture parses: sections, count line, nnkpts."""
        path = REPO_ROOT / "lawaf/tests/fixtures/w90/cubic444.nnkp"
        if not path.exists():
            pytest.skip("cubic444.nnkp fixture absent")
        nnkp = parse_nnkp(path)
        assert nnkp.nnkpts.shape[0] == 64
        assert nnkp.nnkpts.shape[1] > 0
        assert nnkp.real_lattice.shape == (3, 3)
        # neighbour keys are integers in range and unique per point
        assert nnkp.nnkpts[:, :, 0].min() >= 0
        assert nnkp.nnkpts.max() < 64

    def test_parse_and_validate(self, setup):
        source, nnkp = setup[:2]
        prod = AbinitWannierInputs(source, _plan())
        parsed = prod.validate_nnkp(nnkp)
        assert isinstance(parsed, Nnkp)
        assert parsed.nnkpts.shape == (8, 6, 4)

    def test_d1_missing_nnkp(self, tmp_path):
        source, _ = _mesh_source(2)
        prod = AbinitWannierInputs(source, _plan())
        with pytest.raises(ProducerError, match=r"D1.*-pp"):
            prod.validate_nnkp(tmp_path / "absent.nnkp")

    def test_d2_neighbor_key_mismatch(self, tmp_path):
        source, _, _, _, _ = None, None, None, None, None
        source, _gmet = _mesh_source(2)
        table = _mesh_neighbors(source.header.kpt, 2)
        nnkp_path = _write_nnkp(tmp_path / "syn.nnkp", source, table)
        prod = AbinitWannierInputs(source, _plan())
        text = nnkp_path.read_text().splitlines()
        i0 = next(i for i, ln in enumerate(text) if ln == "begin nnkpts")
        # corrupt the k-index field of the second nnkpts row (block 1)
        t = text[i0 + 2].split()
        text[i0 + 2] = f"{2:5d} " + " ".join(t[1:])
        bad = tmp_path / "bad.nnkp"
        bad.write_text("\n".join(text))
        with pytest.raises(ProducerError, match=r"D2.*nnkpts"):
            prod.validate_nnkp(bad)

    def test_d2_duplicate_neighbor_key(self, tmp_path):
        source, _gmet = _mesh_source(2)
        table = _mesh_neighbors(source.header.kpt, 2)
        nnkp_path = _write_nnkp(tmp_path / "syn.nnkp", source, table)
        prod = AbinitWannierInputs(source, _plan())
        text = nnkp_path.read_text().splitlines()
        i0 = next(i for i, ln in enumerate(text) if ln == "begin nnkpts")
        # duplicate the first neighbour row verbatim inside block 1
        text.insert(i0 + 2, text[i0 + 1])
        del text[-3]  # keep block sizes nk*nntot consistent is NOT possible;
        # the parser derives nntot = rows // nk: 49 rows / 8 is not integer,
        # so drop one trailing row of the last block instead
        bad = tmp_path / "bad.nnkp"
        bad.write_text("\n".join(text))
        with pytest.raises(ProducerError):
            prod.validate_nnkp(bad)

    def test_d3_lattice_print_outside_rounding_bound(self, tmp_path):
        source, _ = _mesh_source(2)
        table = _mesh_neighbors(source.header.kpt, 2)
        nnkp_path = tmp_path / "syn.nnkp"
        prod = AbinitWannierInputs(source, _plan())
        # a 7-decimal print is within the documented bound (5.1e-8);
        # 1e-6 noise is not
        _write_nnkp(nnkp_path, source, table, lattice_noise=1e-6)
        with pytest.raises(ProducerError, match=r"D3.*real_lattice"):
            prod.validate_nnkp(nnkp_path)
        # an exact 7-decimal print (what -pp actually writes) is within
        # the documented bound
        _write_nnkp(nnkp_path, source, table)
        assert prod.validate_nnkp(nnkp_path) is not None

    def test_d3_kpoint_print_outside_rounding_bound(self, tmp_path):
        source, _ = _mesh_source(2)
        nnkp_path = tmp_path / "syn.nnkp"
        _write_nnkp(
            nnkp_path, source, _mesh_neighbors(source.header.kpt, 2), kpt_noise=1e-5
        )
        prod = AbinitWannierInputs(source, _plan())
        with pytest.raises(ProducerError, match=r"D3.*kpoints"):
            prod.validate_nnkp(nnkp_path)

    def test_d4_band_selection_vs_source(self, tmp_path):
        source, _ = _mesh_source(2)
        with pytest.raises(ProducerError, match="D4.*band_map"):
            AbinitWannierInputs(
                source,
                GenerationPlan(band_map=(0, 1, 2, 3, 9), num_wann=1, mp_grid=(2, 2, 2)),
            )


# ---------------------------------------------------------------------------
# TEST-064-03: closed-form A/M orientation + transport references
# ---------------------------------------------------------------------------


class TestContraction:
    def test_a_orientation_state_bra_trial_ket(self, setup):
        source, nnkp, trials, plan, prod = setup
        for ik in range(int(source.header.nkpt)):
            kg, psi = prod._sphere(ik)
            a = prod._projection_tile(ik)
            g = trials(ik, kg)
            # independent reference: A[m, n] = sum_G conj(psi_m(G)) g_n(G)
            ref = psi.conj().T @ g
            np.testing.assert_allclose(a, ref, atol=1e-12)
            # conjugation flipped (trial bra) is a detected fault
            assert _differs(a, g.conj().T @ psi)
            # adjoint-vs-transpose confusion is a detected fault
            assert _differs(a, ref.T)

    def test_m_transport_matches_independent_reference(self, setup):
        source, nnkp, trials, plan, prod = setup
        h = source.header
        nnkp_parsed = prod.validate_nnkp(nnkp)
        pos_cache = {}
        for ik in range(int(h.nkpt)):
            kg1, psi1 = prod._sphere(ik)
            for slot in range(nnkp_parsed.nnkpts.shape[1]):
                m = prod._pair_tile(ik, slot, nnkp_parsed)
                jk = int(nnkp_parsed.nnkpts[ik, slot, 0])
                g_img = nnkp_parsed.nnkpts[ik, slot, 1:4]
                if jk not in pos_cache:
                    kg2, psi2 = prod._sphere(jk)
                    pos_cache[jk] = ({tuple(g): i for i, g in enumerate(kg2)}, psi2)
                pos2, psi2 = pos_cache[jk]
                # independent loop reference over the true transport match
                ref = np.zeros((4, 4), dtype=complex)
                for i1, g1 in enumerate(kg1):
                    i2 = pos2.get(tuple(np.asarray(g1) + g_img))
                    if i2 is not None:
                        ref += np.outer(psi1[i1].conj(), psi2[i2])
                np.testing.assert_allclose(m, ref, atol=1e-12)
                # off-by-one neighbour image is a detected fault
                ref_bad = np.zeros((4, 4), dtype=complex)
                for i1, g1 in enumerate(kg1):
                    i2 = pos2.get(tuple(np.asarray(g1) + g_img + np.array([1, 0, 0])))
                    if i2 is not None:
                        ref_bad += np.outer(psi1[i1].conj(), psi2[i2])
                assert _differs(m, ref_bad)
                # adjoint (ref^dag) and transpose (ref.T) confusions detected
                assert _differs(m, ref.conj().T)
                assert _differs(m, ref.T)

    def test_sym_reference_small_case(self):
        """Independent SymPy reference of the transport identity on a
        2-point, 1-band, 2-G toy sphere (image g = 0 pairing)."""
        sympy = pytest.importorskip("sympy")
        c1a, c1b, c2a, c2b = sympy.symbols("c1a c1b c2a c2b", complex=True)
        psi1 = np.array([[c1a], [c1b]])
        psi2 = np.array([[c2a], [c2b]])
        m = psi1.conj().T @ psi2  # M pairs G2 = G1 + g
        expected = sympy.conjugate(c1a) * c2a + sympy.conjugate(c1b) * c2b
        assert sympy.simplify(m[0, 0] - expected) == 0

    def test_write_matrices_end_to_end(self, setup, tmp_path):
        source, nnkp, trials, plan, prod = setup
        manifest = prod.write_matrices(nnkp, tmp_path, "syn")
        # EIG in eV from the retained (band-mapped) eigenvalues
        eig = np.array(
            [
                [float(x) for x in ln.split()]
                for ln in Path(manifest.files["eig"]).read_text().strip().splitlines()
            ]
        )
        ik0 = eig[:, 1] == 1
        ref = np.asarray(source.eigenvalues(0, 0, (0, 4)))[list(plan.band_map)] * HA_EV
        np.testing.assert_allclose(eig[ik0][:, 2], ref, atol=1e-9)
        # MMN header/count sanity + parseable by the reader seam
        mmn, nnlist, nncell = read_mmn(manifest.files["mmn"])
        assert mmn.shape == (8, 6, 4, 4)
        np.testing.assert_array_equal(nnlist, prod.validate_nnkp(nnkp).nnkpts[:, :, 0])
        # manifest + provenance
        assert manifest.band_map == plan.band_map
        assert manifest.neighbor_count == 6
        assert (
            manifest.nnkp_sha256 == hashlib.sha256(Path(nnkp).read_bytes()).hexdigest()
        )
        assert manifest.units["eig"] == "eV"
        assert "A = Psi^dagger G" in manifest.provenance["a_orientation"]
        assert "b = k' + g - k" in manifest.provenance["transport"]
        assert int(manifest.provenance["slabs_emitted"]) == 8 * 6
        assert all(v > 0 for v in manifest.bytes.values())


# ---------------------------------------------------------------------------
# TEST-064-05: writer coverage validation, parametrized D1-D7
# ---------------------------------------------------------------------------


def _full_blocks(kc, nntot, nb, slab=2):
    rng = np.random.default_rng(3)
    for ik in range(kc):
        for ib in range(nntot):
            m = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
            for c0 in range(0, nb, slab):
                yield (ik, ib, c0, m[:, c0 : c0 + slab])


class TestFailClosed:
    NT = np.array([[[1, 1, 0, 0], [0, -1, 0, 0]], [[0, 1, 0, 0], [1, -1, 0, 0]]])

    def test_d5_dropped_slab(self, tmp_path):
        blocks = [b for b in _full_blocks(2, 2, 4)]
        del blocks[1]  # second slab of the first pair
        with pytest.raises(ValueError, match=r"\(0, 0\).*coverage"):
            write_mmn_blocks(tmp_path / "x.mmn", 4, 2, self.NT, iter(blocks))

    def test_d5_duplicated_pair(self, tmp_path):
        blocks = list(_full_blocks(2, 2, 4))
        blocks.insert(2, blocks[0])  # duplicate pair (0, 0)
        with pytest.raises(ValueError, match="coverage"):
            write_mmn_blocks(tmp_path / "x.mmn", 4, 2, self.NT, iter(blocks))

    def test_d5_out_of_order(self, tmp_path):
        blocks = list(_full_blocks(2, 2, 4))
        blocks[2], blocks[4] = blocks[4], blocks[2]
        with pytest.raises(ValueError, match="out-of-order"):
            write_mmn_blocks(tmp_path / "x.mmn", 4, 2, self.NT, iter(blocks))

    def test_d5_missing_pair(self, tmp_path):
        blocks = [b for b in _full_blocks(2, 2, 4) if b[:2] != (1, 1)]
        with pytest.raises(ValueError, match="missing blocks"):
            write_mmn_blocks(tmp_path / "x.mmn", 4, 2, self.NT, iter(blocks))

    def test_d5_overlapping_slab(self, tmp_path):
        blocks = list(_full_blocks(2, 2, 4))
        blocks[1] = (0, 0, 1, blocks[1][3])  # overlap: starts at 1, not 2
        with pytest.raises(ValueError, match="gap or overlap"):
            write_mmn_blocks(tmp_path / "x.mmn", 4, 2, self.NT, iter(blocks))

    def test_d6_rank_deficient_gram(self, tmp_path):
        source, _ = _mesh_source(2, nband=4)

        def constant_trials(ik, kg):
            return np.ones((kg.shape[0], 2), dtype=complex)

        plan = GenerationPlan(
            band_map=(0, 1, 2, 3), num_wann=2, mp_grid=(2, 2, 2), orthogonalize=True
        )
        prod = AbinitWannierInputs(source, plan, trial_orbitals=constant_trials)
        with pytest.raises(ProducerError, match="D6"):
            prod._projection_tile(0)

    def test_d7_nonfinite_contraction_names_state(self, tmp_path):
        source, _ = _mesh_source(2, nband=4, finite=False)
        prod = AbinitWannierInputs(
            source, _plan(), trial_orbitals=_gaussian_trials(source, n_per_site=3)
        )
        with pytest.raises(ProducerError, match=r"D7.*k=0"):
            prod._projection_tile(0)

    def test_no_partially_valid_file_set(self, setup, tmp_path, monkeypatch):
        """A mid-stream MMN failure leaves no final file set behind."""
        source, nnkp_path, trials, plan, prod = setup
        real_write = write_mmn_blocks
        import lawaf.interfaces.abinit.producer as prodmod

        def bomb(path, band_count, k_count, table, blocks):
            def guard():
                for n, b in enumerate(blocks):
                    if n == 5:
                        raise OSError("simulated mid-stream failure")
                    yield b

            return real_write(path, band_count, k_count, table, guard())

        monkeypatch.setattr(prodmod, "write_mmn_blocks", bomb)
        with pytest.raises(OSError):
            prod.write_matrices(nnkp_path, tmp_path, "syn")
        assert not (tmp_path / "syn.mmn").exists()
        assert not (tmp_path / "syn.amn").exists()
        assert not (tmp_path / "syn.eig").exists()


# ---------------------------------------------------------------------------
# TEST-064-06: carrier parity + bounded-scaling measurement
# ---------------------------------------------------------------------------


def _arrays_from_reader(reader):
    from lawaf.interfaces.abinit.source_arrays import (
        CallerArraySource,
        CallerMetadata,
        SymmetryMapEntry,
    )

    h = reader.header
    nk = int(h.nkpt)
    states, kgs = {}, {}
    eigen = np.empty((h.mband, nk, h.nsppol))
    for spin in range(h.nsppol):
        for ik in range(nk):
            nb = int(h.nband[ik + spin * nk])
            states[(ik, spin)] = np.array(
                reader.coefficients(ik, spin, (0, nb)), dtype=np.complex128
            )
            kgs[ik] = np.array(reader.kg(ik), dtype=np.int32)
            eigen[:, ik, spin] = reader.eigenvalues(ik, spin, (0, nb))
    from lawaf.interfaces.abinit.producer import map_mesh_to_ibz

    mesh = np.array(list(np.ndindex(4, 4, 4)), dtype=float) / 4
    mappings = map_mesh_to_ibz(h.kpt, mesh, h.symrel, h.tnons, time_reversal=True)
    entries = tuple(
        SymmetryMapEntry(
            mp.mesh_point,
            mp.ibz_point,
            mp.operation,
            mp.translation,
            mp.reciprocal_shift,
            "TR" if mp.time_reversal else "same",
            mp.time_reversal,
        )
        for mp in mappings
    )
    meta = CallerMetadata(
        header=h,
        eigenvalues=eigen,
        eigenvalue_units="hartree",
        length_units="bohr",
        symmetry_map=entries,
        nkpt_mesh=len(mesh),
        mesh_kpoints=mesh,
    )
    return CallerArraySource(meta, states, kgs)


class TestBoundedStreaming:
    @needs_scalar
    def test_carrier_parity_identical_files(self, tmp_path):
        """The same states through the WFK-backed and caller-array carriers
        generate identical AMN/MMN/EIG files."""
        from lawaf.interfaces.abinit.source_wfk import WfkReader

        plan = GenerationPlan(
            band_map=(0, 1, 2, 3),
            num_wann=4,
            mp_grid=(4, 4, 4),
            controls={"num_iter": "50"},
        )
        files = {}
        with WfkReader.open(SCALAR_NC) as r:
            for tag, src in (("wfk", r), ("arr", _arrays_from_reader(r))):
                d = tmp_path / tag
                d.mkdir()
                trials = _gaussian_trials(src, n_per_site=2, alpha=1.5)
                prod = AbinitWannierInputs(src, plan, trial_orbitals=trials)
                nnkp = _write_nnkp(
                    d / "gaas.nnkp", src, _mesh_neighbors(src.header.kpt, 4)
                )
                manifest = prod.write_matrices(nnkp, d, "gaas")
                files[tag] = {
                    k: Path(p).read_bytes() for k, p in manifest.files.items()
                }
        assert files["wfk"] == files["arr"], "carrier parity broken: files differ"

    @pytest.mark.parametrize("nmesh", [4, 6, 8])
    def test_streamed_mesh_workloads_obey_process_scratch_cap(self, tmp_path, nmesh):
        """Actual Python/NumPy producer allocations, excluding borrowed inputs."""
        import gc
        import tracemalloc

        nband = 4
        source, _ = _mesh_source(nmesh, nband=nband, ecut=0.4)
        prod = AbinitWannierInputs(
            source,
            _plan(nband=nband, nw=3, nmesh=nmesh),
            trial_orbitals=_gaussian_trials(source, n_per_site=3),
        )
        nnkp = _write_nnkp(
            tmp_path / "syn.nnkp", source, _mesh_neighbors(source.header.kpt, nmesh)
        )
        gc.collect()
        tracemalloc.start()
        try:
            manifest = prod.write_matrices(nnkp, tmp_path, "syn")
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak <= 256 * 1024 * 1024
        assert retained < 1024 * 1024
        print(
            f"MEASURE nmesh={nmesh}: nk={int(source.header.nkpt)} "
            f"actual_peak={peak / 1024:.1f} KiB retained={retained / 1024:.1f} KiB "
            f"disk={sum(manifest.bytes.values()) / 1024:.1f} KiB "
            f"times={ {k: round(v, 3) for k, v in manifest.seconds.items()} }"
        )

    def test_slab_mode_streaming_equivalent(self, tmp_path):
        """Column-slab emission (slab_columns>0) yields the identical file."""
        source, _ = _mesh_source(2, nband=4)
        prod = AbinitWannierInputs(
            source, _plan(), trial_orbitals=_gaussian_trials(source, n_per_site=3)
        )
        nnkp = _write_nnkp(
            tmp_path / "syn.nnkp", source, _mesh_neighbors(source.header.kpt, 2)
        )
        m1 = prod.write_matrices(nnkp, tmp_path / "a", "syn")
        m2 = prod.write_matrices(nnkp, tmp_path / "b", "syn", slab_columns=2)
        assert Path(m1.files["mmn"]).read_text() == Path(m2.files["mmn"]).read_text()
        assert int(m2.provenance["slabs_emitted"]) == 2 * int(
            m1.provenance["slabs_emitted"]
        )


# ---------------------------------------------------------------------------
# TEST-064-04: real GaAs IBZ vs explicit full-mesh gates (slow, real)
# ---------------------------------------------------------------------------


ABINIT = Path(
    os.environ.get(
        "LAWAF_TEST_ABINIT",
        str(REPO_ROOT / ".tmp/w90-story064/xc-resolution/build-fixed/src/98_main/abinit"),
    )
)
needs_abinit = pytest.mark.skipif(
    not (ABINIT.exists() and SCALAR_NC.exists()),
    reason="abinit executable or story-060 fixture unavailable",
)

PSP_DIR = REPO_ROOT / "abinit/tests/Pspdir/Psdj_nc_fr_04_pbe_std_psp8"


def _fullmesh_deck(dest):
    """Symmetry IBZ run vs explicit full-mesh reference at matched solver
    quality: DS1 = the pinned story-060 SCF (identical settings), DS2 = a
    tight NSCF pass on the symmetry-reduced IBZ mesh (tolwfr 1e-16), DS3 =
    a tight NSCF pass on the explicit kptopt-0 4x4x4 mesh from the same
    DS1 density.  Both WFKs then carry states converged to the same
    residual, which is what the <= 1e-10 contracted gate presumes.
    """
    mesh = np.array(
        [[i / 4, j / 4, k / 4] for i in range(4) for j in range(4) for k in range(4)]
    )
    kpt3 = "\n".join(
        " ".join(" ".join(f"{x:.14f}" for x in k) for k in mesh[i : i + 4])
        for i in range(0, len(mesh), 4)
    )
    lines = [
        "# story-064 IBZ vs explicit full-mesh reference (pinned settings)",
        "# Solver controls match the certified cg11/matched decks: no band",
        "#buffering (nbdbuf 0) and tolwfr 1e-28 on every NSCF band, so the",
        "#retained window 10:24 is fully converged on both routes.",
        "ndtset 3",
        "jdtset 1 2 3",
        "iomode 3",
        "ecut 40",
        "nband 24",
        "nbdbuf 0",
        "nstep 300",
        "nline 30",
        "tolrde 0",
        "wfoptalg 10",
        "nshiftk 1",
        "shiftk 0 0 0",
        "toldfe 1e-14",
        "diemac 12",
        "acell 3*10.68",
        "rprim 0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0",
        "natom 2",
        "typat 1 2",
        "znucl 31 33",
        "xred 0 0 0  0.25 0.25 0.25",
        "ntypat 2",
        f'pseudos "{PSP_DIR / "Ga.psp8"}, {PSP_DIR / "As.psp8"}"',
        "so_psp 0 0",
        "nspinor 1",
        "nsppol 1",
        "nspden 1",
        # DS1: pinned IBZ SCF (density only)
        "iscf1 5",
        "kptopt1 1",
        "ngkpt1 4 4 4",
        "prtwf1 0",
        # DS2: tight NSCF pass on the symmetry IBZ mesh
        "iscf2 -2",
        "tolwfr2 1e-28",
        "kptopt2 1",
        "ngkpt2 4 4 4",
        "getden2 -1",
        "prtwf2 1",
        # DS3: tight NSCF pass on the explicit full mesh
        "nkpt3 64",  # explicit count so the parser continues kpt3 lines
        "iscf3 -2",
        "tolwfr3 1e-28",
        "kptopt3 0",
        "getden3 -2",
        "prtwf3 1",
        "kpt3 " + kpt3,
    ]
    Path(dest).write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def gaas_nscf_wfks():
    """(IBZ NSCF WFK, full-mesh NSCF WFK) generated once into the named
    story scratch (.tmp/w90-story064/gaas-nc/) and reused."""
    run = SCRATCH / "gaas-nc"
    run.mkdir(parents=True, exist_ok=True)
    ibz_wfk = run / "gaas-fullo_DS2_WFK.nc"
    full_wfk = run / "gaas-fullo_DS3_WFK.nc"
    if not full_wfk.exists():
        _fullmesh_deck(run / "gaas-full.abi")
        proc = subprocess.run(
            [str(ABINIT), "gaas-full.abi"],
            cwd=run,
            timeout=7200,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
        assert ibz_wfk.exists() and full_wfk.exists(), "NSCF passes produced no WFK"
    return ibz_wfk, full_wfk


class TestIbzVsFullMesh:
    @needs_abinit
    def test_expansion_matches_explicit_mesh(self, tmp_path, gaas_nscf_wfks):
        """AC-064-04: DS1 (IBZ, symmetry) expanded per the epic-048-c rules
        vs an explicit full-mesh NSCF: eigenvalues <= 1e-8 eV; contracted
        A/M <= 1e-10 normalized after degenerate-subspace gauge alignment.
        """
        import lawaf.interfaces.abinit.producer as prodmod
        from lawaf.interfaces.abinit.source_wfk import WfkReader
        from lawaf.interfaces.abinit.source_mesh import MeshSource

        ibz_wfk, full_wfk = gaas_nscf_wfks
        band_range = (10, 24)  # Architecture §9: exclude ten verified semicore states.
        with WfkReader.open(ibz_wfk) as ribz, WfkReader.open(full_wfk) as rfull:
            h = rfull.header
            mesh = np.asarray(h.kpt)
            nmesh = 4
            # 1) eigenvalue gate (identical source, <= 1e-8 eV)
            exp_src = MeshSource(ribz, mesh, time_reversal=ribz.header.kptopt in (1, 2))
            evals = np.array(
                [exp_src.eigenvalues(i, 0, band_range) for i in range(len(mesh))]
            )
            ev_full = np.array(
                [rfull.eigenvalues(i, 0, band_range) for i in range(len(mesh))]
            )
            dev = np.abs(evals * HA_EV - ev_full * HA_EV).max()
            print(f"MEASURE ibz-vs-mesh max|dE| = {dev:.3e} eV")
            assert dev <= 1e-8
            # 2) A/M gate through the producer on both carriers
            plan = GenerationPlan(
                band_map=tuple(range(band_range[0], band_range[1])),
                num_wann=8,
                mp_grid=(nmesh,) * 3,
            )
            from lawaf.interfaces.abinit.trial_manifest import default_nc_manifest

            datasets = PSP_DIR
            basis = default_nc_manifest(
                tmp_path / "basis.json",
                [("Ga", datasets / "Ga.upf"), ("As", datasets / "As.upf")],
                [
                    {"species": species, "position": position.tolist()}
                    for species, position in zip(("Ga", "As"), h.xred)
                ],
            )
            prod_f = AbinitWannierInputs(rfull, plan, trial_manifest=basis)
            prod_e = AbinitWannierInputs(exp_src, plan, trial_manifest=basis)
            trials = prod_f._trials
            nnkp = _write_nnkp(
                tmp_path / "mesh.nnkp", rfull, _mesh_neighbors(mesh, nmesh)
            )
            nnkp_p = prod_f.validate_nnkp(nnkp)
            # 2a) per-point degenerate-subspace gauge alignment (expanded
            # states aligned onto the explicit-mesh reference states)
            npt = len(mesh)
            psi_al = {}
            for ik in range(npt):
                _, psi_f = prod_f._sphere(ik)
                _, psi_e = prod_e._sphere(ik)
                assert psi_f.shape == psi_e.shape
                ev = np.asarray(rfull.eigenvalues(ik, 0, band_range))
                psi_al[ik] = prodmod.align_degenerate_subspace(
                    ev, psi_f.T, psi_e.T
                ).T  # (N, nband)
            # 2b) contracted A/M gates (<= 1e-10 normalized)
            worst_a = worst_m = 0.0
            for ik in range(npt):
                kg_f, psi_f = prod_f._sphere(ik)
                g = trials(ik, kg_f)
                a_f = psi_f.conj().T @ g
                a_e = psi_al[ik].conj().T @ g
                worst_a = max(
                    worst_a, np.linalg.norm(a_e - a_f) / max(1.0, np.linalg.norm(a_f))
                )
                for slot in range(nnkp_p.nnkpts.shape[1]):
                    m_f = _pair_from(prod_f, ik, slot, nnkp_p, {})
                    m_e = _pair_from(prod_e, ik, slot, nnkp_p, psi_al)
                    worst_m = max(
                        worst_m,
                        np.linalg.norm(m_e - m_f) / max(1.0, np.linalg.norm(m_f)),
                    )
            print(
                f"MEASURE worst normalized |dA| = {worst_a:.3e}, "
                f"|dM| = {worst_m:.3e}"
            )
            assert worst_a <= 1e-10
            assert worst_m <= 1e-10


def _pair_from(prod, ik, slot, nnkp, psi_override):
    """Pair tile using gauge-aligned (overridden) spheres at both ends."""
    nn = nnkp.nnkpts[ik, slot]
    jk = int(nn[0])
    g_img = nn[1:4]

    def sphere(pt):
        kg, psi = prod._sphere(pt)
        if pt in psi_override:
            psi = psi_override[pt]
        return kg, psi

    kg1, psi1 = sphere(ik)
    kg2, psi2 = sphere(jk)

    def flat(g):
        g = np.asarray(g, dtype=np.int64) + (1 << 20)
        return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]

    pos = np.clip(np.searchsorted(flat(kg2), flat(kg1 + g_img)), 0, kg2.shape[0] - 1)
    hit = flat(kg2)[pos] == flat(kg1 + g_img)
    return psi1[hit].conj().T @ psi2[pos[hit]]


# ---------------------------------------------------------------------------
# TEST-064-07: real-executable gates (slow; pinned external wannier90.x)
# ---------------------------------------------------------------------------


def _run_w90(exe, seed, cwd, timeout=3600, mpi=None, pp=False):
    cmd = ([mpi] if mpi else []) + [str(exe)] + (["-pp"] if pp else []) + [seed]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    werr = Path(cwd, f"{seed}.werr")
    assert proc.returncode == 0, (
        f"wannier90.x failed (exit {proc.returncode}); "
        f".werr={werr.read_text() if werr.exists() else '<absent>'}; "
        f"stdout tail: {proc.stdout[-2000:]} {proc.stderr[-1000:]}"
    )
    return proc


def _final_spread(wout_path):
    """Last O_TOT SPRD record (A^2) from a .wout iteration log."""
    vals = [
        float(m.group(1))
        for ln in Path(wout_path).read_text().splitlines()
        if "<-- SPRD" in ln
        for m in [re.search(r"O_TOT\s*=\s*([0-9.]+)", ln)]
        if m
    ]
    if not vals:
        raise AssertionError(f"no O_TOT SPRD line in {wout_path}")
    return vals[-1]


def _parse_amn(path):
    """Independent .amn parser -> (nb, nk, np, A[nk, nb, np])."""
    lines = Path(path).read_text().splitlines()
    nb, nk, npd = (int(x) for x in lines[1].split()[:3])
    a = np.zeros((nk, nb, npd), dtype=complex)
    for ln in lines[2:]:
        t = ln.split()
        if len(t) != 5:
            continue
        m, n, k = int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1
        a[k, m, n] = complex(float(t[3]), float(t[4]))
    return nb, nk, npd, a


def _parse_eig(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    nb = max(int(r[0]) for r in rows)
    nk = max(int(r[1]) for r in rows)
    e = np.zeros((nk, nb))
    for r in rows:
        e[int(r[1]) - 1, int(r[0]) - 1] = float(r[2])
    return nb, nk, e


class TestExecutableGates:
    @needs_w90
    def test_si_matched_input_reserialization_gate(self, tmp_path):
        """AC-064-07(i): reserialize the official tutorial Si inputs through
        the lawaf writers; -pp reproduces the reference NNKP byte-identically
        (modulo timestamp) and the full localization reproduces the baseline
        total spread within <= 1e-8 A^2, serial and external MPI.
        """
        base = tmp_path / "baseline"
        reser = tmp_path / "reserialized"
        base.mkdir()
        reser.mkdir()
        for f in ("Si.win", "Si.amn", "Si.mmn", "Si.eig"):
            shutil.copy(SI_TUT / f, base / f)
        shutil.copy(SI_TUT / "Si.win", reser / "Si.win")

        # reference: pristine -pp + baseline full run (serial)
        _run_w90(W90_SERIAL, "Si", base, pp=True)
        ref_nnkp = (base / "Si.nnkp").read_bytes()
        _run_w90(W90_SERIAL, "Si", base)  # full run (same dir, -pp cached)
        spread_base = _final_spread(base / "Si.wout")

        # reserialize through the lawaf writers at the io seam
        nb, nk, npd, amn = _parse_amn(SI_TUT / "Si.amn")
        write_amn(amn, reser / "Si.amn")
        _, _, eig = _parse_eig(SI_TUT / "Si.eig")
        write_eig(eig, reser / "Si.eig")
        mmn, nnlist, nncell = read_mmn(SI_TUT / "Si.mmn")
        write_mmn(mmn, nnlist, nncell, reser / "Si.mmn")
        # geometry: pristine Si.win bohr rows x the pinned Bohr constant
        win_text = (SI_TUT / "Si.win").read_text()
        lat_block = (
            win_text.split("begin unit_cell_cart")[1]
            .split("end unit_cell_cart")[0]
            .strip()
            .splitlines()
        )
        assert lat_block[0].strip().lower() == "bohr"
        bohr_rows = np.array([[float(x) for x in ln.split()] for ln in lat_block[1:4]])
        kblock = (
            win_text.split("begin kpoints")[1]
            .split("end kpoints")[0]
            .strip()
            .splitlines()
        )
        kpts = np.array([[float(x) for x in ln.split()] for ln in kblock])
        write_win(
            reser / "Si.win",
            num_wann=8,
            num_bands=nb,
            mp_grid=(4, 4, 4),
            lattice=bohr_rows * BOHR_ANG,
            symbols=["Si", "Si"],
            frac=[[0.0, 0.0, 0.0], [-0.25, 0.75, -0.25]],
            kpts=kpts,
            projections=["Si : sp3"],
            kmesh_tol=1e-5,
            extra={
                "dis_win_max": "17.d0",
                "dis_froz_max": "6.4d0",
                "dis_num_iter": "120",
                "dis_mix_ratio": "1.d0",
                "num_iter": "500",
                "num_print_cycles": "50",
                "write_hr": ".true.",
                "write_xyz": ".true.",
            },
        )

        # matched-input -pp: byte-identical NNKP except the timestamp line
        _run_w90(W90_SERIAL, "Si", reser, pp=True)
        got = (reser / "Si.nnkp").read_bytes().splitlines()
        want = ref_nnkp.splitlines()
        diff = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
        assert len(got) == len(want)
        assert (
            len(diff) == 1 and b"File written on" in got[diff[0]]
        ), f"NNKP differs beyond timestamp at lines {diff}"
        # full localization, serial
        _run_w90(W90_SERIAL, "Si", reser)
        spread_reser = _final_spread(reser / "Si.wout")
        d_spread = abs(spread_reser - spread_base)
        print(
            f"MEASURE Si serial: baseline O_TOT={spread_base:.8f} A^2, "
            f"reserialized O_TOT={spread_reser:.8f} A^2, "
            f"|d|={d_spread:.2e} A^2"
        )
        assert d_spread <= 1e-8
        # HamiltonIO consumption (write_hr + write_xyz products)
        ham = pytest.importorskip("HamiltonIO")
        model = ham.WannierHam.read_from_wannier_dir(str(reser), "Si")
        assert model.nbasis == 8  # 8 wannier functions consumed
        assert model.nR > 0

        # external MPI leg (build-dev-mpi), fresh reserialized copy
        mpi_dir = tmp_path / "mpi"
        shutil.copytree(reser, mpi_dir)
        (mpi_dir / "Si.wout").unlink(missing_ok=True)
        _run_w90(W90_MPI, "Si", mpi_dir, mpi="mpirun")
        spread_mpi = _final_spread(mpi_dir / "Si.wout")
        print(
            f"MEASURE Si mpirun -np 2: O_TOT={spread_mpi:.8f} A^2, "
            f"|d|={abs(spread_mpi - spread_base):.2e} A^2"
        )
        assert abs(spread_mpi - spread_base) <= 1e-8

    @needs_w90
    def test_gaas_nc_localization_from_lawaf_files(self, tmp_path, gaas_nscf_wfks):
        """AC-064-07(ii): the full lawaf pipeline (write_win -> external
        -pp -> write_matrices -> external localization) on real GaAs NC
        states (explicit 4x4x4 mesh NSCF, story-060 pinned settings)
        completes with exit 0 and a valid .wout.
        """
        from lawaf.interfaces.abinit.source_wfk import WfkReader

        run = tmp_path / "gaas"
        run.mkdir()
        plan = GenerationPlan(
            band_map=(0, 1, 2, 3),
            num_wann=4,
            mp_grid=(4, 4, 4),
            controls={"num_iter": "200", "num_print_cycles": "100", "iprint": "2"},
        )
        with WfkReader.open(gaas_nscf_wfks[1]) as src:
            trials = _gaussian_trials(src, n_per_site=2, alpha=1.5)
            prod = AbinitWannierInputs(src, plan, trial_orbitals=trials)
            prod.write_win(run, "gaas")
            # step 1: external -pp produces the authoritative NNKP
            _run_w90(W90_SERIAL, "gaas", run, pp=True)
            # step 2: lawaf streams AMN/MMN/EIG from the real WFK
            manifest = prod.write_matrices(run / "gaas.nnkp", run, "gaas")
            print(
                f"MEASURE gaas-nc producer: bytes={manifest.bytes}, "
                f"times={ {k: round(v, 2) for k, v in manifest.seconds.items()} }"
            )
        # step 3: external localization on lawaf-generated files
        _run_w90(W90_SERIAL, "gaas", run)
        wout = run / "gaas.wout"
        text = wout.read_text()
        assert "Final State" in text
        spread = _final_spread(wout)
        print(f"MEASURE gaas-nc localization O_TOT={spread:.8f} A^2")
        assert np.isfinite(spread)  # physics review is story 067


def test_streamed_eigenvalues_reject_an_exhausted_point_sequence(tmp_path):
    from lawaf.io.w90 import write_eig_blocks
    with pytest.raises(ValueError):
        write_eig_blocks(tmp_path / "incomplete.eig", 2, [(0, np.array([1.1, 2.7]))])


def test_plan_snapshots_controls_before_win_representation_check(tmp_path):
    source, _ = _mesh_source(2)
    controls = {"spinors": "false", "num_iter": "27"}
    projections = ["Ga:s"]
    plan = GenerationPlan((0, 1, 2, 3), 3, (2, 2, 2), controls=controls, projections=projections)
    producer = AbinitWannierInputs(source, plan)
    controls["spinors"] = "true"
    projections.append("As:p")
    text = producer.write_win(tmp_path, "snapshot").read_text()
    assert "spinors = false" in text
    assert "As:p" not in text


def test_sparse_physical_bands_stream_correct_neighbor_matrices_with_bounded_scratch(tmp_path):
    import tracemalloc

    source, _ = _mesh_source(1, nband=20000, ecut=2.0)
    bands = tuple(2 + 257 * index for index in range(72))
    plan = GenerationPlan(bands, 1, (1, 1, 1))
    trials = _gaussian_trials(source)
    producer = AbinitWannierInputs(source, plan, trials)
    table = _mesh_neighbors(source.header.kpt, 1)
    nnkp = _write_nnkp(tmp_path / "sparse.nnkp", source, table)
    tracemalloc.start()
    try:
        manifest = producer.write_matrices(nnkp, tmp_path, "sparse", slab_columns=7)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024**2, peak
    values, _, _ = read_mmn(manifest.files["mmn"])
    g = source.kg(0).T
    states = source.coefficients(0, 0, (0, 20000))[list(bands), :, 0]
    index = {tuple(vector): position for position, vector in enumerate(g)}
    for slot, row in enumerate(table[0]):
        expected = np.zeros((len(bands), len(bands)), complex)
        for left, vector in enumerate(g):
            right = index.get(tuple(vector + row[1:]))
            if right is not None:
                expected += np.outer(states[:, left].conj(), states[:, right])
        np.testing.assert_allclose(values[0, slot], expected, atol=2e-10, rtol=0)
    _, _, eig = _parse_eig(manifest.files["eig"])
    np.testing.assert_allclose(eig[0], source.metadata.eigenvalues[list(bands), 0, 0] * HA_EV, atol=1e-10, rtol=0)


def test_explicit_scratch_budget_fails_before_matrix_publication(tmp_path):
    from dataclasses import replace
    source, _ = _mesh_source(2)
    plan = replace(_plan(), scratch_budget_bytes=1024)
    producer = AbinitWannierInputs(source, plan, _gaussian_trials(source, n_per_site=3))
    table = _mesh_neighbors(source.header.kpt, 2)
    nnkp = _write_nnkp(tmp_path / "budget.nnkp", source, table)
    with pytest.raises(ProducerError) as error:
        producer.write_matrices(nnkp, tmp_path, "budget")
    assert error.value.diagnostic == "D8"
    assert not any((tmp_path / f"budget.{kind}").exists() for kind in ("eig", "amn", "mmn"))
