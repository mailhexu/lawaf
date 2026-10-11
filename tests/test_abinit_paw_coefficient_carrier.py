"""Story 062 tests: supplied PAW coefficient carrier (lawaf.paw_projection_coefficients v1).

Real NetCDF files are authored through the declared-schema writer; the
supplied-versus-computed witness uses a synthetic analytic dual-projector ×
plane-wave fixture whose coefficients are exactly computable by an
independent in-test contraction.  State-identity negative tests use the real
story-060 GaAs NetCDF WFK header.  No mock readers or fake NetCDF handles.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_LAWAF_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _LAWAF_PKG_ROOT not in sys.path:
    sys.path.insert(0, _LAWAF_PKG_ROOT)

from lawaf.interfaces.abinit.paw_coefficients import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    WITNESS_RTOL,
    PawCoefficientError,
    PawCoefficientNetcdfCarrier,
    SupplierMeta,
    SuppliedCoefficientArrays,
    require_dataset_identity,
    require_dualprojector,
    witness_compare,
    write_netcdf_carrier,
)
from lawaf.interfaces.abinit.source_wfk import WfkHeader, WfkReader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALAR_NC = REPO_ROOT / ".tmp/w90-story060/nc-scalar/gaas-nc3o_DS1_WFK.nc"

needs_scalar = pytest.mark.skipif(
    not SCALAR_NC.exists(), reason="scalar WFK fixture not generated (story 060)"
)


# ---------------------------------------------------------------------------
# synthetic analytic PAW fixture (closed-form dual projectors × plane waves)
# ---------------------------------------------------------------------------

_GS = np.array([g for g in np.ndindex(3, 3, 3)], dtype=float) - 1.0  # 27 G


def _projectors(kpoint):
    """Four normalized Gaussian duals per atom; exact Fourier ket coefficients.

    Smooth local partial waves equal their orthonormal s/p duals in this
    analytic fixture. Odd channels carry -i q; atomic centres carry the
    Bloch-source phase. These are not general dataset Gram conversions.
    """
    q = (2 * np.pi / 10) * (_GS + np.asarray(kpoint))
    envelope = (
        np.pi ** (-0.75) * (2 * np.pi) ** 1.5 * np.exp(-np.sum(q * q, axis=1) / 2)
    )
    local = np.vstack(
        [envelope, *[-1j * np.sqrt(2) * q[:, axis] * envelope for axis in range(3)]]
    )
    return np.vstack(
        [
            local * np.exp(-1j * (q @ site))[None, :]
            for site in (np.zeros(3), np.full(3, 2.5))
        ]
    ) / np.sqrt(1000)


def _states(kpt, nband, nspinor, source_channel=0):
    """Genuinely complex asymmetric mixed-spin inputs, one normalized band."""
    r2 = np.sum(_GS**2, axis=1)
    psi = np.empty((nband, len(_GS), nspinor), complex)
    for band in range(nband):
        for component in range(nspinor):
            alpha = 0.25 + 0.025 * band + 0.25 * component
            phase = (
                0.7 * (band + 1)
                + 1.3 * component
                + 0.3 * source_channel
                + np.dot(kpt, [0.3, -0.8, 0.5])
                + (0.19 + 0.13 * band) * (_GS[:, 0] + 2 * _GS[:, 1])
                + 0.11 * band * _GS[:, 2]
            )
            psi[band, :, component] = (band + 1 + component) * np.exp(
                -alpha * r2 + 1j * phase
            )
        psi[band] /= np.linalg.norm(psi[band])
    return psi


def _reference_coefficients(meta, nspinor):
    """Independent defining Cartesian projector integral, not Fourier reuse."""
    import itertools
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(20)
    positions = np.sqrt(2) * np.array(list(itertools.product(nodes, repeat=3)))
    quadrature = 2**1.5 * np.array(
        [a * b * c for a, b, c in itertools.product(weights, repeat=3)]
    )
    # e^-r²/2 belongs to the Hermite weight; remaining s/p factor below.
    local_basis = np.pi ** (-0.75) * np.column_stack(
        (np.ones(len(positions)), np.sqrt(2) * positions)
    )
    b0, b1 = meta.band_range
    out = np.empty((1, len(meta.kpoints), b1 - b0, nspinor, 8), complex)
    for ik, kpt in enumerate(meta.kpoints):
        q = (2 * np.pi / 10) * (_GS + kpt)
        states = _states(kpt, b1, nspinor, meta.source_channel)[b0:b1]
        for atom, site in enumerate((np.zeros(3), np.full(3, 2.5))):
            plane_waves = np.exp(1j * ((positions + site) @ q.T)) / np.sqrt(1000)
            for component in range(nspinor):
                field = plane_waves @ states[:, :, component].T
                out[0, ik, :, component, 4 * atom : 4 * atom + 4] = (
                    local_basis.T @ (quadrature[:, None] * field)
                ).T
    return out


def _supplied_coefficients(meta, nspinor):
    """Fourier-space contraction against the exact analytic projector formula."""
    b0, b1 = meta.band_range
    out = np.empty((1, len(meta.kpoints), b1 - b0, nspinor, 8), complex)
    for ik, kpt in enumerate(meta.kpoints):
        projectors = _projectors(kpt)
        states = _states(kpt, b1, nspinor, meta.source_channel)[b0:b1]
        out[0, ik] = np.einsum("ig,ngs->nsi", projectors.conj(), states)
    return out


def _meta(nspinor=1, nk=2, band_range=(0, 3), **overrides):
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])[:nk]
    fields = dict(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        function_kind="dualprojector",
        state_frame="ibz",
        kpoints=kpts,
        spin_tuple=(1, nspinor, 1 if nspinor == 1 else 4),
        source_channel=0,
        band_range=band_range,
        dataset_md5=(
            "4fc0dc2916ae951ba10951578630a0be",
            "f7680fb89ccb9aa0435f9d803758fb17",
        ),
        atom_species=("Ga", "As"),
        projector_counts=(4, 4),
        projector_labels=tuple(
            (atom, radial, l, m)
            for atom in range(2)
            for radial, l, m in ((0, 0, 0), (1, 1, 1), (1, 1, -1), (1, 1, 0))
        ),
        harmonic_convention="cartesian_sp",
        conjugation="trial_bra",
        raw_normalization=True,
        phase_declaration="bloch_source_frame",
    )
    fields.update(overrides)
    return SupplierMeta(**fields)


def test_collinear_carrier_identifies_selected_source_channel(tmp_path):
    header = _header(
        nsppol=2, nspden=2, nband=np.array([6, 6, 2, 2], dtype=np.int32), bantot=16
    )
    meta = _meta(spin_tuple=(2, 1, 2), source_channel=1, band_range=(0, 2))
    coefficients = _supplied_coefficients(meta, 1)
    path = write_netcdf_carrier(tmp_path / "down.nc", coefficients, meta)
    with PawCoefficientNetcdfCarrier(path, header) as carrier:
        assert carrier.record.meta.source_channel == 1
        np.testing.assert_array_equal(
            carrier.record.band_slab(1, 1), coefficients[0, 1, 1]
        )
    # Down has2 bands: using the up-channel count6 must not admit this record.
    bad_meta = _meta(spin_tuple=(2, 1, 2), source_channel=1, band_range=(0, 3))
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(_supplied_coefficients(bad_meta, 1), bad_meta, header)
    assert error.value.diagnostic == "D5"


@pytest.mark.parametrize("channel", [None, 2])
def test_supplied_source_channel_must_be_explicit_and_in_range(channel):
    meta = _meta(source_channel=channel)
    # Values are independent of the invalid identity being validated here.
    coefficients = _supplied_coefficients(_meta(), 1)
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(coefficients, meta, _header())
    assert error.value.diagnostic == "D6" and error.value.field == "source_channel"


def test_carrier_retains_explicit_projector_labels(tmp_path):
    labels = tuple(
        (atom, radial, l, m)
        for atom in range(2)
        for radial, l, m in ((0, 0, 0), (1, 1, 1), (1, 1, -1), (1, 1, 0))
    )
    meta = _meta(projector_labels=labels, harmonic_convention="cartesian_sp")
    coefficients = _supplied_coefficients(meta, 1)
    path = write_netcdf_carrier(tmp_path / "labels.nc", coefficients, meta)
    with PawCoefficientNetcdfCarrier(path, _header()) as carrier:
        assert carrier.record.meta.projector_labels == labels
        assert carrier.record.meta.harmonic_convention == "cartesian_sp"
    # Equal total counts cannot detect an ambiguous duplicated angular label.
    duplicate = list(labels)
    duplicate[2] = duplicate[1]
    bad_meta = _meta(
        projector_labels=tuple(duplicate), harmonic_convention="cartesian_sp"
    )
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(coefficients, bad_meta, _header())
    assert error.value.diagnostic == "D8"


@pytest.mark.parametrize("corruption", ["axes", "precision"])
def test_file_carrier_rejects_wrong_axes_or_precision(tmp_path, corruption):
    import netCDF4

    meta = _meta()
    path = write_netcdf_carrier(
        tmp_path / "layout.nc", _supplied_coefficients(meta, 1), meta
    )
    with netCDF4.Dataset(path, "a") as dataset:
        variable = dataset["coefficients"]
        dimensions = variable.dimensions
        dataset.renameVariable("coefficients", "old_coefficients")
        if corruption == "axes":
            dimensions = (dimensions[0], dimensions[2], dimensions[1], *dimensions[3:])
        dataset.createVariable(
            "coefficients", "f4" if corruption == "precision" else "f8", dimensions
        )
    with pytest.raises(PawCoefficientError) as error:
        PawCoefficientNetcdfCarrier(path, _header())
    assert error.value.diagnostic == ("D5" if corruption == "axes" else "D9")


def _header(
    nspinor=1,
    usepaw=1,
    nk=2,
    nband_per_k=(6, 6),
    npsp=2,
    md5=("4fc0dc2916ae951ba10951578630a0be", "f7680fb89ccb9aa0435f9d803758fb17"),
    kpt=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
    **overrides,
):
    kpt = np.array(kpt, dtype=float)[:nk]
    nband = np.tile(np.array(nband_per_k), nk // len(nband_per_k) or 1)[:nk]
    fields = dict(
        codvsn="9.10.2",
        headform=80,
        fform=1,
        bantot=int(nband.sum()),
        date=0,
        intxc=0,
        ixc=7,
        natom=2,
        ngfft=np.array([12, 12, 12]),
        nkpt=nk,
        nspden=1 if nspinor == 1 else 4,
        nspinor=nspinor,
        nsppol=1,
        nsym=24,
        npsp=npsp,
        ntypat=2,
        occopt=1,
        pertcase=0,
        usepaw=usepaw,
        ecut=20.0,
        ecutdg=40.0,
        ecutsm=0.0,
        ecut_eff=20.0,
        qptn=np.zeros(3),
        rprimd=np.eye(3) * 10.0,
        stmbias=0.0,
        tphysel=0.0,
        tsmear=0.0,
        usewvl=0,
        nshiftk_orig=1,
        nshiftk=1,
        mband=int(nband[0]),
        istwfk=np.ones(nk, np.int32),
        istwfk_raw=np.ones(nk, np.int32),
        nband=nband,
        npwarr=np.full(nk, len(_GS), np.int32),
        so_psp=np.array([2] if nspinor == 2 else [1], np.int32),
        symafm=np.ones(24, np.int32),
        symrel=np.tile(np.eye(3, dtype=np.int32), (24, 1, 1)),
        typat=np.array([1, 2], np.int32),
        kpt=kpt,
        occ=np.ones((int(nband[0]), nk), float),
        tnons=np.zeros((24, 3)),
        znucltypat=np.array([31, 33], np.int32),
        wtk=np.full(nk, 1.0 / nk),
        residm=1e-16,
        xred=np.array([[0.0] * 3, [0.25] * 3]),
        etot=-1.0,
        fermie=0.5,
        amu=np.array([69.0, 75.0]),
        kptopt=1,
        pawcpxocc=0,
        nelect=8.0,
        cellcharge=0.0,
        icoulomb=0,
        kptrlatt=np.eye(3, dtype=np.int32),
        kptrlatt_orig=np.eye(3, dtype=np.int32),
        shiftk_orig=np.zeros((1, 3)),
        shiftk=np.zeros((1, 3)),
        title=["Ga pseudo", "As pseudo"],
        znuclpsp=np.array([31.0, 33.0]),
        zionpsp=np.array([3.0, 5.0]),
        pspso=np.array([2 if nspinor == 2 else 1] * npsp, np.int32),
        pspdat=np.array([1] * npsp, np.int32),
        pspcod=np.array([8] * npsp, np.int32),
        pspxc=np.array([11] * npsp, np.int32),
        lmn_size=np.array([4] * npsp, np.int32),
        md5_pseudos=list(md5),
        gmet=np.eye(3) * (2 * np.pi / 10.0) ** 2,
    )
    fields.update(overrides)
    return WfkHeader(**fields)


# ---------------------------------------------------------------------------
# TEST-062-01: memory + NetCDF slab roundtrip exactness; slab-only read proof
# ---------------------------------------------------------------------------


class TestCarrierSlabs:
    def test_062_01_memory_slabs_exact_and_readonly(self):
        meta = _meta()
        coeff = _supplied_coefficients(meta, 1)
        c = SuppliedCoefficientArrays(coeff, meta, _header())
        assert c.coefficients.flags.writeable is False
        for ik in range(2):
            for ib in range(3):
                slab = c.record.band_slab(ik, ib)
                assert slab.shape == (1, 8)
                assert np.array_equal(slab, coeff[0, ik, ib])
                assert slab.flags.writeable is False

    def test_062_01_netcdf_slabs_exact(self, tmp_path):
        meta = _meta()
        coeff = _supplied_coefficients(meta, 1)
        path = write_netcdf_carrier(tmp_path / "carrier.nc", coeff, meta)
        header = _header()
        with PawCoefficientNetcdfCarrier(path, header) as nc:
            for ik in range(2):
                for ib in range(3):
                    slab = nc.record.band_slab(ik, ib)
                    assert np.array_equal(slab, coeff[0, ik, ib, 0, :].reshape(1, 8))
            mem = SuppliedCoefficientArrays(coeff, meta, header)
            # identical validated record view from both carriers
            assert nc.record.shape == mem.record.shape
            assert nc.record.meta.function_kind == mem.record.meta.function_kind
            for ik in range(2):
                assert np.array_equal(
                    nc.record.band_slab(ik, 0), mem.record.band_slab(ik, 0)
                )

    def test_062_01_netcdf_bounded_slab_allocation(self, tmp_path):
        import gc
        import tracemalloc

        nk, nb = 128, 256
        kpoints = np.column_stack((np.arange(nk) / nk, np.zeros(nk), np.zeros(nk)))
        meta = _meta(nspinor=2, nk=nk, band_range=(0, nb), kpoints=kpoints)
        rng = np.random.default_rng(6201)
        coefficients = rng.normal(size=(1, nk, nb, 2, 8)).astype(np.complex128)
        coefficients.imag = rng.normal(size=coefficients.shape)
        path = write_netcdf_carrier(tmp_path / "bounded.nc", coefficients, meta)
        with PawCoefficientNetcdfCarrier(path) as carrier:
            gc.collect()
            tracemalloc.start()
            try:
                first = carrier.record.band_slab(5, 25)
                last = carrier.record.band_slab(nk - 1, nb - 1)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            np.testing.assert_array_equal(first, coefficients[0, 5, 25])
            np.testing.assert_array_equal(last, coefficients[0, nk - 1, nb - 1])
            assert peak < 64 * 1024  # two256-byte slabs; full tensor is8MiB
            assert not first.flags.writeable and not last.flags.writeable
            print(
                f"MEASURE supplied carrier slab peak={peak} bytes; full payload={coefficients.nbytes}"
            )


# ---------------------------------------------------------------------------
# TEST-062-02: supplied vs closed-form reference parity; perturbation rejected
# ---------------------------------------------------------------------------


class TestWitness:
    def test_062_02_supplied_matches_independent_reference(self):
        meta = _meta()
        supplied = _supplied_coefficients(meta, 1)
        reference = _reference_coefficients(meta, 1)
        # independent constructions agree before entering the carrier
        assert (
            np.max(np.abs(supplied - reference)) / np.max(np.abs(reference))
            < WITNESS_RTOL
        )
        c = SuppliedCoefficientArrays(supplied, meta, _header())
        dev = witness_compare(c.record, reference)
        assert dev <= WITNESS_RTOL

    def test_nonzero_physical_band_range_and_readonly_file_slab(self, tmp_path):
        meta = _meta(band_range=(2, 5))
        supplied = _supplied_coefficients(meta, 1)
        path = write_netcdf_carrier(tmp_path / "offset.nc", supplied, meta)
        with PawCoefficientNetcdfCarrier(path, _header()) as file_carrier:
            memory_carrier = SuppliedCoefficientArrays(supplied, meta, _header())
            for physical_band in range(2, 5):
                expected = supplied[0, 1, physical_band - 2]
                for record in (memory_carrier.record, file_carrier.record):
                    slab = record.band_slab(1, physical_band)
                    np.testing.assert_array_equal(slab, expected)
                    assert not slab.flags.writeable
            for outside in (1, 5):
                with pytest.raises(PawCoefficientError):
                    file_carrier.record.band_slab(0, outside)

    def test_062_02_spinor_witness(self):
        meta = _meta(nspinor=2)
        supplied = _supplied_coefficients(meta, 2)
        reference = _reference_coefficients(meta, 2)
        c = SuppliedCoefficientArrays(supplied, meta, _header(nspinor=2))
        assert witness_compare(c.record, reference) <= WITNESS_RTOL
        # genuinely distinct complex spinor components preserved
        slabs = [c.record.band_slab(0, 0)[ci] for ci in (0, 1)]
        assert not np.allclose(slabs[0], slabs[1])

    @pytest.mark.parametrize("fault", ["conjugated", "atom_duplicated"])
    def test_wrong_complex_or_atom_frame_rejected(self, fault):
        meta = _meta(nspinor=2)
        supplied = _supplied_coefficients(meta, 2)
        reference = _reference_coefficients(meta, 2)
        if fault == "conjugated":
            supplied = supplied.conj().copy()
        else:
            supplied[..., 4:] = supplied[..., :4]
        record = SuppliedCoefficientArrays(supplied, meta, _header(nspinor=2)).record
        with pytest.raises(PawCoefficientError) as error:
            witness_compare(record, reference)
        assert error.value.diagnostic == "D2"

    def test_062_02_single_element_perturbation_rejected(self):
        meta = _meta()
        supplied = _supplied_coefficients(meta, 1)
        reference = _reference_coefficients(meta, 1)
        supplied[0, 1, 2, 0, 5] += 1e-6
        c = SuppliedCoefficientArrays(supplied, meta, _header())
        with pytest.raises(PawCoefficientError) as exc:
            witness_compare(c.record, reference)
        assert "[D2]" in str(exc.value)
        assert "(0, 1, 2, 0, 5)" in str(exc.value)


@pytest.mark.parametrize("corrupted", ["supplied", "reference"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_witness_rejects_nonfinite_values(corrupted, bad):
    meta = _meta()
    supplied = _supplied_coefficients(meta, 1)
    reference = _reference_coefficients(meta, 1)
    (supplied if corrupted == "supplied" else reference)[0, 1, 2, 0, 5] = bad
    record = SuppliedCoefficientArrays(supplied, meta, _header()).record
    with pytest.raises(PawCoefficientError) as error:
        witness_compare(record, reference)
    assert error.value.diagnostic == "D2"


def test_witness_relative_gate_and_zero_reference_are_scale_safe():
    meta = _meta()
    reference = _reference_coefficients(meta, 1) * 1e-8
    supplied = _supplied_coefficients(meta, 1) * 1e-8
    record = SuppliedCoefficientArrays(supplied, meta, _header()).record
    assert witness_compare(record, reference) <= WITNESS_RTOL
    supplied[0, 1, 2, 0, 5] += 1e-13
    with pytest.raises(PawCoefficientError):
        witness_compare(record, reference)
    zeros = np.zeros_like(reference)
    record = SuppliedCoefficientArrays(zeros, meta, _header()).record
    assert witness_compare(record, zeros) == 0.0
    tiny = zeros.copy()
    tiny[0, 1, 2, 0, 5] = 1e-30
    with pytest.raises(PawCoefficientError):
        witness_compare(SuppliedCoefficientArrays(tiny, meta, _header()).record, zeros)


@pytest.mark.parametrize("band_range", [(0.5, 3.5), (False, 3)])
def test_fractional_or_boolean_band_coverage_fails_before_indexing(band_range):
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(
            _supplied_coefficients(_meta(), 1), _meta(band_range=band_range), _header()
        )
    assert error.value.diagnostic == "D5"


@pytest.mark.parametrize("state", [(0.9, 1), (0, 1.9), (False, 1)])
def test_fractional_or_boolean_state_request_cannot_select_a_different_state(
    tmp_path, state
):
    meta = _meta()
    values = _supplied_coefficients(meta, 1)
    memory = SuppliedCoefficientArrays(values, meta, _header()).record
    path = write_netcdf_carrier(tmp_path / "states.nc", values, meta)
    with PawCoefficientNetcdfCarrier(path, _header()) as carrier:
        for record in (memory, carrier.record):
            with pytest.raises(PawCoefficientError) as error:
                record.band_slab(*state)
            assert error.value.diagnostic == "D6"


def test_atom_species_swap_rejected_by_both_carriers(tmp_path):
    import netCDF4

    meta = _meta()
    values = _supplied_coefficients(meta, 1)
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(values, _meta(atom_species=("As", "Ga")), _header())
    assert error.value.diagnostic == "D8"
    path = write_netcdf_carrier(tmp_path / "species.nc", values, meta)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["atom_species"][0] = "As"
        dataset["atom_species"][1] = "Ga"
    with pytest.raises(PawCoefficientError) as error:
        PawCoefficientNetcdfCarrier(path, _header())
    assert error.value.diagnostic == "D8"


def test_file_fractional_band_identity_is_not_truncated(tmp_path):
    import netCDF4

    path = write_netcdf_carrier(
        tmp_path / "fractional-range.nc", _supplied_coefficients(_meta(), 1), _meta()
    )
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.renameVariable("band_range", "old_band_range")
        dataset.createVariable("band_range", "f8", ("two",))[:] = [0.5, 3.5]
    with pytest.raises(PawCoefficientError) as error:
        PawCoefficientNetcdfCarrier(path, _header())
    assert error.value.diagnostic == "D5"


# ---------------------------------------------------------------------------
# TEST-062-03: scalar/spinor semantics; state identity vs real story-060 header
# ---------------------------------------------------------------------------


class TestStateIdentity:
    def test_062_03_scalar_and_spinor_component_axes(self):
        s_meta = _meta(nspinor=1)
        s_coeff = _supplied_coefficients(s_meta, 1)
        s_rec = SuppliedCoefficientArrays(s_coeff, s_meta, _header()).record
        assert s_rec.shape[3] == 1
        p_meta = _meta(nspinor=2)
        p_coeff = _supplied_coefficients(p_meta, 2)
        p_rec = SuppliedCoefficientArrays(p_coeff, p_meta, _header(nspinor=2)).record
        assert p_rec.shape[3] == 2
        c0 = p_rec.band_slab(0, 0)[0]
        c1 = p_rec.band_slab(0, 0)[1]
        assert not np.allclose(c0, c1)

    @needs_scalar
    def test_062_03_state_identity_mismatch_fails(self):
        with WfkReader.open(SCALAR_NC) as r:
            h = r.header
        bad_kpt = np.array(h.kpt, dtype=float)
        bad_kpt[3] += 0.125
        meta = _meta(
            kpoints=bad_kpt,
            spin_tuple=(h.nsppol, h.nspinor, h.nspden),
            band_range=(0, 4),
        )
        coeff = np.zeros((1, h.nkpt, 4, 1, 8), complex)
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(coeff, meta, h)
        assert "[D6]" in str(exc.value) and "kpoints[3]" in str(exc.value)
        bad_spin = _meta(
            kpoints=np.array(h.kpt, dtype=float),
            spin_tuple=(2, h.nspinor, h.nspden),
            band_range=(0, 4),
        )
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(coeff, bad_spin, h)
        assert "[D6]" in str(exc.value) and "spin_tuple" in str(exc.value)


# ---------------------------------------------------------------------------
# TEST-062-04: zero-recomputation; anti-shortcut; usepaw from header
# ---------------------------------------------------------------------------


class TestReuseAndAntiShortcut:
    def test_062_04_coefficients_never_waive_dataset_identity(self):
        meta = _meta(dataset_md5=())
        c = SuppliedCoefficientArrays(
            _supplied_coefficients(meta, 1), meta, _header(usepaw=1)
        )
        # coefficients are present and validated, yet any derived kernel
        # still demands the matched dataset
        with pytest.raises(PawCoefficientError) as exc:
            require_dataset_identity(c.record)
        assert "[D4]" in str(exc.value)
        assert "never waive" in str(exc.value)
        assert "Q(b)" in str(exc.value)
        # with the matched identity the gate passes
        ok = require_dataset_identity(_meta())
        assert ok == (
            "4fc0dc2916ae951ba10951578630a0be",
            "f7680fb89ccb9aa0435f9d803758fb17",
        )

    def test_062_04_usepaw_read_from_header_not_coefficients(self):
        meta = _meta()
        c = SuppliedCoefficientArrays(
            _supplied_coefficients(meta, 1), meta, _header(usepaw=1)
        )
        assert c.record.paw_cell is True
        c_nc = SuppliedCoefficientArrays(
            _supplied_coefficients(meta, 1), meta, _header(usepaw=0)
        )
        assert c_nc.record.paw_cell is False


# ---------------------------------------------------------------------------
# TEST-062-05: parametrized D1-D9 diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_062_05_d1_schema_missing_and_unknown(self, tmp_path):
        meta = _meta(schema_name="")
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D1]" in str(exc.value) and "schema_name" in str(exc.value)
        meta = _meta(schema_name="foreign.projector.coefficients")
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "foreign.projector.coefficients" in str(exc.value)
        meta = _meta(schema_version=2)
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D1]" in str(exc.value) and "schema_version" in str(exc.value)
        # file carrier: unknown schema in the NetCDF file itself
        good = _supplied_coefficients(_meta(), 1)
        path = write_netcdf_carrier(tmp_path / "c.nc", good, _meta())
        import netCDF4

        with netCDF4.Dataset(str(path), "a") as ds:
            ds.setncattr("schema_name", "not.ours")
        with pytest.raises(PawCoefficientError) as exc:
            PawCoefficientNetcdfCarrier(path, _header())
        assert "[D1]" in str(exc.value) and "not.ours" in str(exc.value)

    def test_062_05_d2_kind_absent_unknown_and_required(self):
        meta = _meta(function_kind="")
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D2]" in str(exc.value) and "function_kind" in str(exc.value)
        meta = _meta(function_kind="mystery_kind")
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D2]" in str(exc.value) and "mystery_kind" in str(exc.value)
        # pseudo-partial-wave records are readable data but never relabeled
        meta = _meta(function_kind="pseudo_partial_wave")
        rec = SuppliedCoefficientArrays(
            _supplied_coefficients(meta, 1), meta, _header()
        ).record
        assert rec.function_kind == "pseudo_partial_wave"
        with pytest.raises(PawCoefficientError) as exc:
            rec.require_dualprojector()
        assert "[D2]" in str(exc.value)
        assert "never relabeled" in str(exc.value)
        with pytest.raises(PawCoefficientError) as exc:
            require_dualprojector(rec)
        assert "[D2]" in str(exc.value) and "dualprojector" in str(exc.value)

    def test_062_05_d3_projector_count_vs_lmn_size(self):
        meta = _meta(projector_counts=(4, 5))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D3]" in str(exc.value)
        assert "lmn_size" in str(exc.value) and "found 5" in str(exc.value)

    def test_062_05_d4_dataset_fingerprint_mismatch(self):
        meta = _meta(dataset_md5=("0" * 32, "1" * 32))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D4]" in str(exc.value)
        assert "4fc0dc2916ae951ba10951578630a0be" in str(exc.value)

    def test_062_05_d5_band_k_coverage(self):
        meta = _meta(band_range=(0, 9))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D5]" in str(exc.value) and "band_range" in str(exc.value)
        meta = _meta(kpoints=np.zeros((3, 3)))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(
                np.zeros((1, 3, 3, 1, 8), complex), meta, _header()
            )
        assert "[D5]" in str(exc.value) and "nkpt=2" in str(exc.value)

    def test_062_05_d6_state_frame_mismatch(self):
        meta = _meta(spin_tuple=(2, 1, 1))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D6]" in str(exc.value) and "spin_tuple" in str(exc.value)
        meta = _meta(kpoints=np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]]))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D6]" in str(exc.value) and "kpoints[0]" in str(exc.value)

    def test_062_05_d7_missing_declarations(self):
        for override, field in (
            ({"conjugation": ""}, "conjugation"),
            ({"phase_declaration": ""}, "phase_declaration"),
            ({"raw_normalization": None}, "raw_normalization"),
        ):
            meta = _meta(**override)
            with pytest.raises(PawCoefficientError) as exc:
                SuppliedCoefficientArrays(
                    _supplied_coefficients(meta, 1), meta, _header()
                )
            assert "[D7]" in str(exc.value) and field in str(exc.value)

    def test_062_05_d8_atom_ordering(self):
        meta = _meta(atom_species=())
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D8]" in str(exc.value)
        # duplicate labels across distinct typat entries
        meta = _meta(atom_species=("Ga", "Ga"))
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        assert "[D8]" in str(exc.value) and "atom_species" in str(exc.value)

    def test_062_05_d9_readonly_and_contiguity(self):
        meta = _meta()
        # non-contiguous caller view
        big = np.zeros((1, 2, 6, 1, 16), complex)[:, :, ::2, :, ::2]
        assert not big.flags.c_contiguous
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(big, meta, _header())
        assert "[D9]" in str(exc.value) and "C-contiguous" in str(exc.value)
        # wrong dtype
        with pytest.raises(PawCoefficientError) as exc:
            SuppliedCoefficientArrays(
                _supplied_coefficients(meta, 1).astype(np.complex64), meta, _header()
            )
        assert "[D9]" in str(exc.value) and "complex128" in str(exc.value)
        # write attempt through the read-only view
        c = SuppliedCoefficientArrays(_supplied_coefficients(meta, 1), meta, _header())
        with pytest.raises(ValueError) as exc:
            c.coefficients[0, 0, 0, 0, 0] = 1.0
        assert "read-only" in str(exc.value)


# ---------------------------------------------------------------------------
# TEST-062-06: import isolation (NFR-005)
# ---------------------------------------------------------------------------


def test_062_06_import_isolation():
    code = (
        "import sys;"
        "import lawaf.interfaces.abinit.paw_coefficients as m;"
        "assert 'netCDF4' not in sys.modules, 'netCDF4 imported eagerly';"
        "assert not any('wannier' in k.lower() or 'f90wrap' in k.lower()"
        " for k in sys.modules);"
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


@pytest.mark.parametrize("version", [1.7, "unknown", "1", True, 2])
def test_memory_schema_version_requires_exact_integral_supported_declaration(version):
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(
            _supplied_coefficients(_meta(), 1), _meta(schema_version=version), _header()
        )
    assert error.value.diagnostic == "D1" and error.value.field == "schema_version"


@pytest.mark.parametrize("version", [1.7, "unknown", "1", 2])
def test_file_schema_version_requires_exact_integral_supported_declaration(
    tmp_path, version
):
    import netCDF4

    path = write_netcdf_carrier(
        tmp_path / "version.nc", _supplied_coefficients(_meta(), 1), _meta()
    )
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.schema_version = version
    with pytest.raises(PawCoefficientError) as error:
        PawCoefficientNetcdfCarrier(path, _header())
    assert error.value.diagnostic == "D1" and error.value.field == "schema_version"


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("state_frame", None, "D6"),
        ("state_frame", "unrelated", "D6"),
        ("conjugation", "state_bra", "D7"),
        ("raw_normalization", False, "D7"),
        ("raw_normalization", "true", "D7"),
        ("phase_declaration", "other_gauge", "D6"),
    ],
)
def test_dual_carrier_rejects_mismatched_state_and_operator_declarations(
    field, value, diagnostic
):
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(
            _supplied_coefficients(_meta(), 1), _meta(**{field: value}), _header()
        )
    assert error.value.diagnostic == diagnostic


def test_file_requires_explicit_frame_and_exact_boolean_normalization(tmp_path):
    import netCDF4

    for field in ("state_frame", "raw_normalization"):
        path = write_netcdf_carrier(
            tmp_path / f"missing-{field}.nc",
            _supplied_coefficients(_meta(), 1),
            _meta(),
        )
        with netCDF4.Dataset(path, "a") as dataset:
            dataset.delncattr(field)
        with pytest.raises(PawCoefficientError) as error:
            PawCoefficientNetcdfCarrier(path, _header())
        assert error.value.field == field
    path = write_netcdf_carrier(
        tmp_path / "invalid-boolean.nc", _supplied_coefficients(_meta(), 1), _meta()
    )
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.raw_normalization = 2
    with pytest.raises(PawCoefficientError) as error:
        PawCoefficientNetcdfCarrier(path, _header())
    assert error.value.diagnostic == "D7"


def test_expanded_records_require_matching_consumer_map_reference(tmp_path):
    meta = _meta(state_frame="expanded", symmetry_map_reference="source-mesh-map-v1")
    values = _supplied_coefficients(meta, 1)
    path = write_netcdf_carrier(tmp_path / "expanded.nc", values, meta)
    for reference in (None, "different-map"):
        with pytest.raises(PawCoefficientError) as error:
            SuppliedCoefficientArrays(
                values, meta, _header(), symmetry_map_reference=reference
            )
        assert error.value.diagnostic == "D6"
        with pytest.raises(PawCoefficientError) as error:
            PawCoefficientNetcdfCarrier(
                path, _header(), symmetry_map_reference=reference
            )
        assert error.value.diagnostic == "D6"
    memory = SuppliedCoefficientArrays(
        values, meta, _header(), symmetry_map_reference="source-mesh-map-v1"
    )
    with PawCoefficientNetcdfCarrier(
        path, _header(), symmetry_map_reference="source-mesh-map-v1"
    ) as file:
        np.testing.assert_array_equal(
            memory.record.band_slab(1, 2), file.record.band_slab(1, 2)
        )
        assert file.record.meta.symmetry_map_reference == "source-mesh-map-v1"


def test_published_supplier_coordinates_are_borrowed_readonly():
    meta = _meta()
    assert not meta.kpoints.flags.writeable
    values = _supplied_coefficients(meta, 1)
    record = SuppliedCoefficientArrays(values, meta, _header()).record
    with pytest.raises(ValueError):
        record.meta.kpoints[0, 0] = 0.125
    np.testing.assert_array_equal(record.band_slab(1, 2), values[0, 1, 2])


def test_atom_label_count_must_equal_source_atom_count():
    meta = _meta(
        atom_species=("Ga",),
        projector_counts=(8,),
        projector_labels=tuple(
            (0, radial, l, m) for atom, radial, l, m in _meta().projector_labels
        ),
    )
    with pytest.raises(PawCoefficientError) as error:
        SuppliedCoefficientArrays(_supplied_coefficients(_meta(), 1), meta, _header())
    assert error.value.diagnostic == "D8"


PAW_WFK = REPO_ROOT / ".tmp/w90-story065/gaas-pawo_DS2_WFK.nc"
PAW_XML = REPO_ROOT / "abinit/tests/Pspdir/Psdj_paw_pbe_std"


@pytest.mark.skipif(
    not PAW_WFK.exists(), reason="actual PAW electronic fixture not generated"
)
def test_supplied_actual_electronic_coefficients_keep_nonzero_metric_and_finite_b_without_recomputation(
    tmp_path,
):
    from contextlib import ExitStack
    from lawaf.interfaces.abinit.basis_datasets import DatasetError
    from lawaf.interfaces.abinit.paw_provider import PawCoefficientProvider
    from lawaf.interfaces.abinit.paw_kernels import HARMONIC_CONVENTION
    from lawaf.interfaces.abinit.physics_paw import PawOverlap

    paths = tuple(PAW_XML / f"{name}.xml" for name in ("Ga", "As"))
    with WfkReader.open(PAW_WFK) as source, ExitStack() as stack:
        h = source.header
        computed = PawCoefficientProvider(source, paths)
        values = np.stack([computed.band_block(k, 0, (10, 12)) for k in range(h.nkpt)])[
            None
        ]
        meta = _meta(
            state_frame="source",
            kpoints=h.kpt,
            spin_tuple=(h.nsppol, h.nspinor, h.nspden),
            band_range=(10, 12),
            dataset_md5=tuple(h.md5_pseudos),
            projector_counts=tuple(np.diff(computed.offsets)),
            projector_labels=computed.projector_labels,
            harmonic_convention=HARMONIC_CONVENTION,
        )
        memory = SuppliedCoefficientArrays(values, meta, h).record
        path = write_netcdf_carrier(tmp_path / "actual-paw-cprj.nc", values, meta)
        file = stack.enter_context(PawCoefficientNetcdfCarrier(path, h)).record
        table = np.zeros((h.nkpt, 3, 4), int)
        for k in range(h.nkpt):
            table[k, :, 0] = [k, (k + 1) % h.nkpt, (k - 1) % h.nkpt]
        finite = []
        for record in (memory, file):
            provider = PawCoefficientProvider(source, paths, supplied_record=record)
            metric = PawOverlap(provider)
            metric.prepare_neighbors(table)
            for k in range(h.nkpt):
                _, _, psi = source.decode_full_sphere(k, 0, (10, 12))
                c = provider.band_block(k, 0, (10, 12))
                delta = metric.overlap(c, c, np.zeros(3))
                assert np.max(abs(delta)) > 1e-4
                np.testing.assert_allclose(
                    psi[:, :, 0].conj() @ psi[:, :, 0].T + delta,
                    np.eye(2),
                    atol=1e-8,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    metric.projection(c, c.transpose(2, 0, 1)),
                    delta,
                    atol=1e-12,
                    rtol=0,
                )
            left = provider.band_block(0, 0, (10, 12))
            right = provider.band_block(1, 0, (10, 12))
            shift = h.kpt[1] - h.kpt[0]
            value = metric.overlap(left, right, shift)
            assert np.max(abs(value)) > 1e-4
            np.testing.assert_allclose(
                value, metric.overlap(right, left, -shift).conj().T, atol=1e-12, rtol=0
            )
            finite.append(value)
            assert provider.compute_calls == 0
            with pytest.raises(DatasetError):
                PawCoefficientProvider(source, [], supplied_record=record)
        np.testing.assert_allclose(finite[0], finite[1], atol=1e-12, rtol=0)
