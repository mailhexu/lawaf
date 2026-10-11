"""Story 063 tests: caller-owned array source adapter.

Carrier parity uses the real story-060 GaAs NetCDF WFK: the states/kg/
eigenvalues are read through the story-060 reader into caller arrays, handed
to :class:`CallerArraySource`, and every block must match the WFK-backed
route bitwise.  No-copy is proven by planted checksums and buffer-identity
checks.  No fake source objects.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

_LAWAF_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _LAWAF_PKG_ROOT not in sys.path:
    sys.path.insert(0, _LAWAF_PKG_ROOT)

from lawaf.interfaces.abinit.source_arrays import (
    ArraySourceError,
    CallerArraySource,
    CallerMetadata,
    SymmetryMapEntry,
)
from lawaf.interfaces.abinit.source_wfk import WfkReader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALAR_NC = REPO_ROOT / ".tmp/w90-story060/nc-scalar/gaas-nc3o_DS1_WFK.nc"

needs_scalar = pytest.mark.skipif(
    not SCALAR_NC.exists(), reason="scalar WFK fixture not generated (story 060)"
)


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _symmetry_map(nk):
    """Explicit IBZ->mesh map: identity operation, zero translation, no
    reciprocal shift, spin-preserving label for every mesh point."""
    return tuple(
        SymmetryMapEntry(
            mesh_point=i,
            ibz_point=i,
            operation=np.eye(3, dtype=np.int32),
            translation=np.zeros(3),
            reciprocal_shift=np.zeros(3, dtype=np.int32),
            spin_label="same",
            time_reversal=False,
        )
        for i in range(nk)
    )


@needs_scalar
def _fixture_arrays():
    """Read the real WFK through the story-060 reader into caller arrays."""
    states, kg_arrays = {}, {}
    with WfkReader.open(SCALAR_NC) as r:
        h = r.header
        nk, nsppol = h.nkpt, h.nsppol
        for spin in range(nsppol):
            for ik in range(nk):
                nb = int(h.nband[ik + spin * nk])
                states[(ik, spin)] = np.array(
                    r.coefficients(ik, spin, (0, nb)), dtype=np.complex128
                )
                kg_arrays[ik] = np.array(r.kg(ik), dtype=np.int32)
        eigen = np.empty((h.mband, nk, nsppol))
        for spin in range(nsppol):
            for ik in range(nk):
                eigen[:, ik, spin] = r.eigenvalues(ik, spin, (0, int(h.mband)))
        header_ref = h
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
        header=header_ref,
        eigenvalues=eigen,
        eigenvalue_units="hartree",
        length_units="bohr",
        symmetry_map=entries,
        nkpt_mesh=len(mesh),
        mesh_kpoints=mesh,
    )
    return meta, states, kg_arrays, header_ref


# ---------------------------------------------------------------------------
# TEST-063-01: carrier parity (bitwise) on the real fixture deck
# ---------------------------------------------------------------------------


class TestCarrierParity:
    @needs_scalar
    def test_063_01_metadata_and_blocks_bitwise_identical(self):
        meta, states, kg_arrays, href = _fixture_arrays()
        src = CallerArraySource(meta, states, kg_arrays)
        h = src.header
        # exposed metadata identical on the fields the seam exposes
        assert h.nkpt == href.nkpt and h.nsppol == href.nsppol
        assert h.nspinor == href.nspinor and h.nspden == href.nspden
        assert h.usepaw == href.usepaw and h.natom == href.natom
        assert np.array_equal(h.kpt, href.kpt)
        assert np.array_equal(h.npwarr, href.npwarr)
        assert np.array_equal(h.nband, href.nband)
        assert np.array_equal(h.istwfk, href.istwfk)
        assert np.array_equal(h.rprimd, href.rprimd)
        assert np.array_equal(h.typat, href.typat)
        np.testing.assert_array_equal(h.znucltypat, href.znucltypat)
        assert h.ecut_eff == href.ecut_eff
        with WfkReader.open(SCALAR_NC) as r:
            for spin in range(h.nsppol):
                for ik in range(h.nkpt):
                    nb = int(h.nband[ik + spin * h.nkpt])
                    assert np.array_equal(
                        src.coefficients(ik, spin, (0, nb)),
                        r.coefficients(ik, spin, (0, nb)),
                    )
                    assert np.array_equal(src.kg(ik), r.kg(ik))
                    assert np.array_equal(
                        src.eigenvalues(ik, spin, (0, int(h.mband))),
                        r.eigenvalues(ik, spin, (0, int(h.mband))),
                    )

    @needs_scalar
    def test_063_01_packed_decode_parity(self):
        meta, states, kg_arrays, href = _fixture_arrays()
        src = CallerArraySource(meta, states, kg_arrays)
        with WfkReader.open(SCALAR_NC) as r:
            packed = [ik for ik in range(href.nkpt) if int(href.istwfk[ik]) > 1]
            assert packed, "fixture should exercise packed storage"
            for ik in packed:
                nb = int(href.nband[ik])
                a = src.decode_full_sphere(ik, 0, (0, nb))
                b = r.decode_full_sphere(ik, 0, (0, nb))
                assert np.array_equal(a[0], b[0])  # kg_full bitwise
                assert np.array_equal(a[1], b[1])  # weights bitwise
                assert np.array_equal(a[2], b[2])  # cg_full bitwise


# ---------------------------------------------------------------------------
# TEST-063-02: checksum + live-bytes instrumentation
# ---------------------------------------------------------------------------


class TestNoCopy:
    @needs_scalar
    def test_063_02_checksums_and_shared_buffers(self):
        meta, states, kg_arrays, href = _fixture_arrays()
        before = {(k, s): _digest(a) for (k, s), a in states.items()}
        kg_before = {k: _digest(a) for k, a in kg_arrays.items()}
        eig_before = _digest(meta.eigenvalues)
        src = CallerArraySource(meta, states, kg_arrays)
        # full streaming pass, twice
        per_pass = []
        for _ in range(2):
            digests = []
            for spin in range(meta.header.nsppol):
                for ik in range(meta.header.kpt.shape[0]):
                    block = src.coefficients(ik, spin, (0, int(src.header.nband[ik])))
                    _ = src.kg(ik)
                    _ = src.eigenvalues(ik, spin, (0, int(src.header.mband)))
                    if int(src.header.istwfk[ik]) > 1:
                        _ = src.decode_full_sphere(
                            ik, spin, (0, int(src.header.nband[ik]))
                        )
                    digests.append(_digest(np.asarray(block, dtype=np.complex128)))
            per_pass.append(digests)
        results = per_pass[0]
        for (k, s), a in states.items():
            assert _digest(a) == before[(k, s)], "caller state array mutated"
        for k, a in kg_arrays.items():
            assert _digest(a) == kg_before[k], "caller kg array mutated"
        assert _digest(meta.eigenvalues) == eig_before
        # zero copy: the adapter's returned blocks share caller memory
        view = src.coefficients(0, 0, (0, 3))
        assert np.shares_memory(view, states[(0, 0)])
        assert view.flags.writeable is False
        assert np.shares_memory(src.kg(0), kg_arrays[0])
        # repeated passes give identical results
        assert per_pass[0] == per_pass[1]


# ---------------------------------------------------------------------------
# TEST-063-03: mixed per-k storage + symmetry-map validation
# ---------------------------------------------------------------------------


def _tiny_arrays():
    """Small authored deck: 2 k-points, 3 bands, 4 npw, scalar."""
    rng = np.random.default_rng(7)
    kpt = np.array([[0.1, 0.2, 0.3], [-0.25, 0.0, 0.0]])
    npw = 4
    states = {
        (0, 0): (
            rng.normal(size=(3, npw, 1)) + 1j * rng.normal(size=(3, npw, 1))
        ).astype(np.complex128),
        (1, 0): (
            rng.normal(size=(3, npw, 1)) + 1j * rng.normal(size=(3, npw, 1))
        ).astype(np.complex128),
    }
    kg = {
        0: rng.integers(-1, 2, size=(3, npw)).astype(np.int32),
        1: rng.integers(-1, 2, size=(3, npw)).astype(np.int32),
    }
    from test_abinit_wfk_reader import synthetic_header

    header = replace(
        synthetic_header(kpt, [npw, npw], 3, ecut=2.0),
        rprimd=8 * np.eye(3),
        istwfk_raw=np.array([0, 1], np.int32),
        md5_pseudos=("",),
        kptopt=0,
        kptrlatt=np.zeros((3, 3), np.int32),
        kptrlatt_orig=np.zeros((3, 3), np.int32),
    )
    meta = CallerMetadata(
        header=header,
        eigenvalues=rng.normal(size=(3, 2, 1)),
        eigenvalue_units="hartree",
        length_units="bohr",
        symmetry_map=_symmetry_map(2),
        nkpt_mesh=2,
        mesh_kpoints=kpt,
    )
    return meta, states, kg


def test_skew_row_lattice_rejects_physically_outside_G():
    meta, states, kg = _tiny_arrays()
    meta = replace(
        meta,
        header=replace(
            meta.header,
            rprimd=np.array([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 1.0, 10.0]]),
            kpt=np.zeros((2, 3)),
            istwfk_raw=np.ones(2, np.int32),
            ecut=0.2,
            ecut_eff=0.2,
        ),
    )
    meta = replace(meta, mesh_kpoints=meta.header.kpt)
    for point in kg:
        kg[point][:] = 0
        kg[point][0, 1] = 1
    # Exact G100 kinetic0.2053667284Ha; transpose mutant0.197392088Ha.
    source = CallerArraySource(meta, states, kg)
    with pytest.raises(ArraySourceError) as error:
        source.kg(0)
    assert error.value.diagnostic == "D2"


def test_spinor_automatic_gamma_storage_resolves_full():
    meta, states, kg = _tiny_arrays()
    meta = replace(
        meta,
        header=replace(
            meta.header,
            kpt=np.zeros((2, 3)),
            nspinor=2,
            nspden=4,
            istwfk_raw=np.array([0, 1], np.int32),
        ),
    )
    meta = replace(meta, mesh_kpoints=meta.header.kpt)
    for point in states:
        original = states[point]
        states[point] = np.concatenate((original, 1j * original), axis=2)
    source = CallerArraySource(meta, states, kg)
    np.testing.assert_array_equal(source.header.istwfk, [1, 1])
    np.testing.assert_array_equal(source.header.istwfk_raw, [0, 1])
    for point in (0, 1):
        np.testing.assert_array_equal(
            source.coefficients(point, 0, (0, 3)), states[(point, 0)]
        )


class TestStorageAndSymmetry:
    def test_063_03_mixed_storage_records_resolved(self):
        meta, states, kg = _tiny_arrays()
        src = CallerArraySource(meta, states, kg)
        # k0 raw 0 (automatic at generic point -> 1); k1 explicit 1
        assert list(src.header.istwfk) == [1, 1]
        assert np.array_equal(src.header.istwfk_raw, [0, 1])

    def test_063_03_symmetry_map_incomplete(self):
        meta, states, kg = _tiny_arrays()
        bad = CallerMetadata(**{**vars(meta), "symmetry_map": _symmetry_map(2)[:1]})
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D4]" in str(exc.value) and "symmetry_map" in str(exc.value)
        assert "uncovered mesh points [1]" in str(exc.value)

    def test_063_03_symmetry_map_duplicated(self):
        meta, states, kg = _tiny_arrays()
        bad = CallerMetadata(
            **{**vars(meta), "symmetry_map": (_symmetry_map(2)[0], _symmetry_map(2)[0])}
        )
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D4]" in str(exc.value) and "duplicated" in str(exc.value)

    def test_063_03_symmetry_map_missing_labels(self):
        meta, states, kg = _tiny_arrays()
        entry = SymmetryMapEntry(
            0,
            0,
            np.eye(3, dtype=np.int32),
            np.zeros(3),
            np.zeros(3, np.int32),
            "",
            False,
        )
        other = SymmetryMapEntry(
            1,
            1,
            np.eye(3, dtype=np.int32),
            np.zeros(3),
            np.zeros(3, np.int32),
            "same",
            False,
        )
        bad = CallerMetadata(**{**vars(meta), "symmetry_map": (entry, other)})
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D4]" in str(exc.value) and "spin_label" in str(exc.value)
        # missing translation label
        entry = SymmetryMapEntry(
            0, 0, np.eye(3, dtype=np.int32), None, np.zeros(3, np.int32), "same", False
        )
        bad = CallerMetadata(**{**vars(meta), "symmetry_map": (entry, other)})
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D4]" in str(exc.value) and "translation" in str(exc.value)


# ---------------------------------------------------------------------------
# TEST-063-04: parametrized D1-D7 diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_063_04_d1_missing_metadata_fields(self):
        meta, states, kg = _tiny_arrays()
        for absent in (
            "rprimd",
            "kpt",
            "nband",
            "nsppol",
            "typat",
            "xred",
            "usepaw",
            "length_units",
        ):
            bad = (
                replace(meta, length_units=None)
                if absent == "length_units"
                else replace(meta, header=replace(meta.header, **{absent: None}))
            )
            with pytest.raises(ArraySourceError) as exc:
                CallerArraySource(bad, states, kg)
            assert "[D1]" in str(exc.value) and absent in str(exc.value)
        # missing state/kg arrays name the block
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, {(0, 0): states[(0, 0)]}, kg)
        assert "[D1]" in str(exc.value) and "states[(1, 0)]" in str(exc.value)
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, states, {})
        assert "[D1]" in str(exc.value) and "kg[0]" in str(exc.value)

    def test_063_04_d2_shape_inconsistencies(self):
        meta, states, kg = _tiny_arrays()
        bad_states = dict(states)
        bad_states[(1, 0)] = np.ascontiguousarray(states[(1, 0)][:, :-1])
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, bad_states, kg)
        assert "[D2]" in str(exc.value) and "states[(1, 0)]" in str(exc.value)
        bad_kg = dict(kg)
        bad_kg[0] = np.ascontiguousarray(kg[0][:, :-1])
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, states, bad_kg)
        assert "[D2]" in str(exc.value) and "kg[0]" in str(exc.value)

    def test_063_04_d3_storage_records(self):
        meta, states, kg = _tiny_arrays()
        # unresolved raw value
        bad = replace(
            meta, header=replace(meta.header, istwfk_raw=np.array([11, 1], np.int32))
        )
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D3]" in str(exc.value) and "istwfk[0]" in str(exc.value)
        # packed + nspinor=2 illegal
        bad = replace(
            meta,
            header=replace(
                meta.header, nspinor=2, nspden=4, istwfk_raw=np.array([2, 1], np.int32)
            ),
        )
        st2 = {key: np.repeat(a, 2, axis=2) for key, a in states.items()}
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, st2, kg)
        assert "[D3]" in str(exc.value) and "nspinor=2" in str(exc.value)

    def test_063_04_d5_paw_without_operator_data(self):
        meta, states, kg = _tiny_arrays()
        bad = replace(meta, header=replace(meta.header, usepaw=1))
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D5]" in str(exc.value)
        assert "never" in str(exc.value)

    def test_063_04_d6_noncontiguous_or_wrong_dtype(self):
        meta, states, kg = _tiny_arrays()
        wide = np.zeros((3, 8, 1), np.complex128)[:, ::2, :]
        assert not wide.flags.c_contiguous
        bad_states = dict(states)
        bad_states[(0, 0)] = wide
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, bad_states, kg)
        assert "[D6]" in str(exc.value) and "C-contiguous" in str(exc.value)
        bad_states = dict(states)
        bad_states[(0, 0)] = states[(0, 0)].astype(np.complex64)
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, bad_states, kg)
        assert "[D6]" in str(exc.value) and "complex128" in str(exc.value)
        bad_kg = dict(kg)
        bad_kg[0] = kg[0].astype(np.int64)
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(meta, states, bad_kg)
        assert "[D6]" in str(exc.value) and "int32" in str(exc.value)

    def test_063_04_d7_eigen_shape_and_units(self):
        meta, states, kg = _tiny_arrays()
        bad = CallerMetadata(**{**vars(meta), "eigenvalue_units": "ev"})
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D7]" in str(exc.value) and "hartree" in str(exc.value)
        bad = CallerMetadata(**{**vars(meta), "eigenvalues": np.zeros((2, 2, 1))})
        with pytest.raises(ArraySourceError) as exc:
            CallerArraySource(bad, states, kg)
        assert "[D7]" in str(exc.value) and "(nband, nk, nsppol)" in str(exc.value)

    def test_063_04_block_access_bounds(self):
        meta, states, kg = _tiny_arrays()
        src = CallerArraySource(meta, states, kg)
        with pytest.raises(ArraySourceError) as exc:
            src.coefficients(0, 0, (0, 4))
        assert "[D6]" in str(exc.value) and "band-range" in str(exc.value)


# ---------------------------------------------------------------------------
# TEST-063-05: import isolation (NFR-005)
# ---------------------------------------------------------------------------


def test_063_05_import_isolation():
    code = (
        "import sys;"
        "import lawaf.interfaces.abinit.source_arrays as m;"
        "assert not any('wannier' in k.lower() or 'f90wrap' in k.lower()"
        " for k in sys.modules);"
        "assert 'netCDF4' not in sys.modules;"
        "print('isolated')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_LAWAF_PKG_ROOT),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "isolated" in out.stdout


def test_caller_chemical_species_map_to_every_atom_in_written_geometry(tmp_path):
    from lawaf.interfaces.abinit.producer import AbinitWannierInputs, GenerationPlan

    meta, states, kg = _tiny_arrays()
    meta = replace(
        meta,
        header=replace(
            meta.header,
            natom=3,
            ntypat=2,
            npsp=2,
            typat=np.array([2, 1, 2], np.int32),
            xred=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            znucltypat=np.array([31.0, 33.0]),
            lmn_size=np.array([4, 4], np.int32),
            so_psp=np.zeros(2, np.int32),
            md5_pseudos=("", ""),
        ),
    )
    source = CallerArraySource(meta, states, kg)
    producer = AbinitWannierInputs(source, GenerationPlan((0, 1, 2), 1, (2, 1, 1)))
    path = producer.write_win(tmp_path, "geometry")
    rows = (
        path.read_text()
        .split("begin atoms_frac\n")[1]
        .split("end atoms_frac")[0]
        .splitlines()
    )
    assert [row.split()[0] for row in rows] == ["As", "Ga", "As"]
    positions = np.array([[float(value) for value in row.split()[1:]] for row in rows])
    np.testing.assert_allclose(positions, meta.header.xred, atol=5e-13, rtol=0)
    with pytest.raises(ArraySourceError, match="znucltypat"):
        CallerArraySource(
            replace(meta, header=replace(meta.header, znucltypat=None)), states, kg
        )


@needs_scalar
def test_actual_symmetry_occupations_and_transport_are_identical_across_sources():
    from lawaf.interfaces.abinit.source_mesh import MeshSource

    meta, states, kg, reference = _fixture_arrays()
    source = CallerArraySource(meta, states, kg)
    for name in (
        "symrel",
        "tnons",
        "symafm",
        "occ",
        "wtk",
        "so_psp",
        "kptrlatt",
        "shiftk",
    ):
        np.testing.assert_array_equal(
            getattr(source.header, name), getattr(reference, name)
        )
    mesh = np.array(list(np.ndindex(4, 4, 4)), dtype=float) / 4
    with WfkReader.open(SCALAR_NC) as file:
        file_mesh = MeshSource(file, mesh, time_reversal=True)
        array_mesh = MeshSource(source, mesh, time_reversal=True)
        for point in (0, 1, 7, 16, 63):
            expected = file_mesh.decode_full_sphere(point, 0, (10, 12))
            actual = array_mesh.decode_full_sphere(point, 0, (10, 12))
            for value, ref in zip(actual, expected):
                np.testing.assert_array_equal(value, ref)
            np.testing.assert_array_equal(
                array_mesh.eigenvalues(point, 0, (10, 12)),
                file_mesh.eigenvalues(point, 0, (10, 12)),
            )
        np.testing.assert_array_equal(array_mesh.header.occ, file_mesh.header.occ)
    assert len(source.symmetry_map) == 64
    for original, published in zip(meta.symmetry_map, source.symmetry_map):
        np.testing.assert_array_equal(
            published.reciprocal_shift, original.reciprocal_shift
        )
        assert published.spin_label == original.spin_label


@pytest.mark.parametrize(
    "name,value",
    [
        ("npwarr", np.array([4], np.int32)),
        ("nband", np.array([3], np.int32)),
        ("istwfk_raw", np.array([1], np.int32)),
        ("kpt", np.zeros((2, 2))),
        ("rprimd", np.eye(2)),
        ("typat", np.empty(0, np.int32)),
        ("xred", np.zeros((0, 3))),
    ],
)
def test_header_dimensions_fail_before_block_indexing(name, value):
    meta, states, kg = _tiny_arrays()
    with pytest.raises(ArraySourceError) as error:
        CallerArraySource(
            replace(meta, header=replace(meta.header, **{name: value})), states, kg
        )
    assert error.value.diagnostic == "D2"
    assert error.value.field == name


def test_target_map_size_is_explicit_not_the_stored_ibz_count():
    meta, states, kg = _tiny_arrays()
    with pytest.raises(ArraySourceError, match="nkpt_mesh"):
        CallerArraySource(replace(meta, nkpt_mesh=None), states, kg)


def test_published_geometry_and_map_views_cannot_write_into_owner():
    meta, states, kg = _tiny_arrays()
    source = CallerArraySource(meta, states, kg)
    for field in (
        "rprimd",
        "kpt",
        "xred",
        "typat",
        "istwfk_raw",
        "symrel",
        "tnons",
        "occ",
    ):
        original = getattr(meta.header, field)
        published = getattr(source.header, field)
        before = original.copy()
        assert np.shares_memory(published, original)
        with pytest.raises(ValueError):
            published.flat[0] = 99
        np.testing.assert_array_equal(original, before)
    with pytest.raises(ValueError):
        source.symmetry_map[0].translation[0] = 0.25


@needs_scalar
def test_real_packed_decoder_allocations_obey_process_scratch_cap():
    import gc
    import tracemalloc

    meta, states, kg, _ = _fixture_arrays()
    source = CallerArraySource(meta, states, kg)
    for band_range in ((0, 24), (10, 12)):
        gc.collect()
        tracemalloc.start()
        maximum_output = 0
        try:
            for point in range(source.header.nkpt):
                decoded = source.decode_full_sphere(point, 0, band_range)
                # Count only newly allocated output arrays; full-storage views
                # still borrow the caller's coefficients/G vectors.
                owned = sum(
                    value.nbytes
                    for value in decoded
                    if not any(
                        np.shares_memory(value, original)
                        for original in (states[point, 0], kg[point])
                    )
                )
                maximum_output = max(maximum_output, owned)
                del decoded
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak >= maximum_output
        assert peak <= 256 * 1024**2  # approved Architecture8 process scratch cap
        # Source output/temporary allocation must not remain mesh-sized live.
        assert current <= maximum_output + 1024 * 1024
        print(
            "MEASURE caller actual traced bytes",
            band_range,
            "owned output",
            maximum_output,
            "retained",
            current,
            "peak",
            peak,
        )


@pytest.mark.parametrize("field", ["nshiftk", "nshiftk_orig", "ecut", "ecut_eff"])
def test_required_scalar_header_fields_fail_closed_when_absent(field):
    meta, states, kg = _tiny_arrays()
    with pytest.raises(ArraySourceError) as error:
        CallerArraySource(
            replace(meta, header=replace(meta.header, **{field: None})), states, kg
        )
    assert error.value.diagnostic == "D1" and error.value.field == field


@needs_scalar
def test_declared_nondefault_gamma_map_drives_actual_rotated_field():
    from lawaf.interfaces.abinit.source_mesh import MeshSource

    meta, states, kg, h = _fixture_arrays()
    entry = meta.symmetry_map[0]
    operation = next(op for op in h.symrel if not np.array_equal(op, entry.operation))
    entries = list(meta.symmetry_map)
    entries[0] = replace(
        entry, operation=operation, time_reversal=False, spin_label="same"
    )
    source = CallerArraySource(replace(meta, symmetry_map=tuple(entries)), states, kg)
    image = MeshSource(source, meta.mesh_kpoints, time_reversal=True)
    out_g, _, out_c = image.decode_full_sphere(0, 0, (10, 12))
    points = np.array([[0.13, 0.27, 0.31], [0.43, 0.17, 0.21], [0.09, 0.33, 0.57]])
    source_g, _, source_c = source.decode_full_sphere(entry.ibz_point, 0, (10, 12))
    original_momentum = source_g.T + h.kpt[entry.ibz_point]
    expected = source_c[:, :, 0] @ np.exp(
        2j * np.pi * (original_momentum @ (points @ operation + entry.translation).T)
    )
    actual = out_c[:, :, 0] @ np.exp(
        2j * np.pi * ((out_g.T + meta.mesh_kpoints[0]) @ points.T)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_declared_source_index_and_reciprocal_shift_are_validated():
    meta, states, kg = _tiny_arrays()
    for entry in (
        replace(meta.symmetry_map[0], ibz_point=2),
        replace(meta.symmetry_map[0], reciprocal_shift=np.array([1, 0, 0])),
    ):
        with pytest.raises(ArraySourceError) as error:
            CallerArraySource(
                replace(meta, symmetry_map=(entry, meta.symmetry_map[1])), states, kg
            )
        assert error.value.diagnostic == "D4"
