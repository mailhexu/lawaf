"""Per-q Wannier window legality and persistence (story 025)."""

from pathlib import Path
from ase import Atoms

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.anharmonic.io import load_symmetry, save_symmetry  # noqa: E402
from lawaf.anharmonic import (  # noqa: E402
    WindowBands,
    build_space_group_action,
    check_window_legality,
)
from lawaf.params import WannierParams  # noqa: E402
from lawaf.interfaces import PhonopyDownfolder  # noqa: E402
from lawaf.interfaces.downfolder import Lawaf  # noqa: E402
from lawaf.wannierization.projectedWF import ProjectedWannierizer  # noqa: E402
from lawaf.wannierization.scdmk import ScdmkWannierizer  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = (0.0, 0.0, 0.0)
X = (0.5, 0.0, 0.0)
M = (0.5, 0.5, 0.0)
R = (0.5, 0.5, 0.5)
Q7 = {GAMMA: (0, 1, 2), X: (0, 1, 4), M: (0, 1, 4), R: (0, 1, 2)}

@pytest.fixture(scope="module")
def phonon():
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=True)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def sga(phonon):
    return build_space_group_action(phonon)


def test_nac_wrapper_matches_phonopy_standard_anchor_spectra():
    """NAC downfolding must retain phonopy's full DM at every Q7 anchor."""
    native = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=True)
    native.symmetrize_force_constants()
    anchors = (GAMMA, X, M, R)
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    wrapper = PhonopyWrapper(native, mode="dm", is_nac=True, use_cache=False)
    expected = []
    for qpoint in anchors:
        native.run_qpoints([qpoint], with_eigenvectors=True)
        expected.append(np.asarray(native.qpoints.eigenvalues[0], dtype=float))
    expected = np.asarray(expected)

    actual = np.asarray([wrapper.solve(q)[0] for q in anchors])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_rejects_window_that_straddles_x_eigenblock(sga, phonon):
    """Lowest three X bands cut the 3.317 cm-1 E doublet and are illegal."""
    with pytest.raises(
        ValueError,
        match=r"q=X.*3\.317.*dim(?:ension)? 2.*irrep dimension 2",
    ):
        check_window_legality(sga, phonon, {X: (0, 1, 2)})


def test_q7_window_records_complete_little_group_decompositions(sga, phonon):
    """Q7 is a sum of whole blocks: Gamma T1u, X E+A, M A+A+A, R T."""
    window = check_window_legality(sga, phonon, Q7)

    assert isinstance(window, WindowBands)
    assert window.representatives == Q7
    assert window.bands[GAMMA] == (0, 1, 2)
    assert [block.dimension for block in window.legality[GAMMA]] == [3]
    assert [block.dimension for block in window.legality[X]] == [2, 1]
    assert [block.dimension for block in window.legality[M]] == [1, 1, 1]
    assert [block.dimension for block in window.legality[R]] == [3]
    assert sum(len(block.irreps) for block in window.legality[GAMMA]) == 1
    assert sum(len(block.irreps) for block in window.legality[X]) == 2
    assert sum(len(block.irreps) for block in window.legality[M]) == 3
    assert sum(len(block.irreps) for block in window.legality[R]) == 1
    assert all(
        block.residual < 1e-6
        for blocks in window.legality.values()
        for block in blocks
    )


def test_x_window_is_star_closed_with_identical_sorted_band_indices(sga, phonon):
    """One X representative pins its two cubic star arms to the same bands."""
    window = check_window_legality(sga, phonon, {X: (0, 1, 4)})

    arms = {tuple(np.round(q, 8)) for _, q in sga.star(X)}
    assert arms == {(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)}
    assert {q: window.bands[q] for q in arms} == {
        q: (0, 1, 4) for q in arms
    }


def test_empty_window_bands_are_rejected(sga, phonon):
    with pytest.raises(ValueError, match="at least one representative"):
        check_window_legality(sga, phonon, {})


class _AccidentallyDegenerateTetragonalPhonon:
    def solve(self, _qpoint):
        return np.ones(3), np.eye(3, dtype=complex)


def test_complete_irrep_inside_reducible_degenerate_block_is_legal():
    atoms = Atoms(
        "H",
        scaled_positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.0, 1.0, 2.0]),
        pbc=True,
    )
    tetragonal_sga = build_space_group_action(atoms)

    window = check_window_legality(
        tetragonal_sga,
        _AccidentallyDegenerateTetragonalPhonon(),
        {GAMMA: (2,)},
    )

    assert window.bands[GAMMA] == (2,)
    assert window.legality[GAMMA][0].dimension == 1
    assert window.legality[GAMMA][0].residual < 1e-10


def test_window_bands_default_has_no_legacy_behavior_hook():
    """NFR-005: omitted window_bands stays None and is not serialized by v1 params."""
    params = WannierParams(nwann=3)

    assert params.window_bands is None
    assert "window_bands" not in params.to_dict()


def test_window_bands_round_trip_in_symmetry_group(tmp_path, sga, phonon):
    """The explicit bands and derived legality are reproducible from netCDF."""
    window = check_window_legality(sga, phonon, Q7)
    path = tmp_path / "window_bands.nc"

    save_symmetry(path, sga, window_bands=window)
    record = load_symmetry(path)

    assert record.window_bands == window


def test_v1_window_record_migrates_missing_representatives_and_residuals(
    tmp_path, sga, phonon
):
    import netCDF4

    path = tmp_path / "window_bands_v1.nc"
    save_symmetry(
        path,
        sga,
        window_bands=check_window_legality(sga, phonon, Q7),
    )
    with netCDF4.Dataset(path, "a") as root:
        group = root.groups["symmetry"]
        group.setncattr("schema_version", 1)
        for name in (
            "window_representative_q",
            "window_representative_band_count",
            "window_representative_bands",
            "window_block_residual",
        ):
            group.renameVariable(name, f"v2_{name}")

    migrated = load_symmetry(path).window_bands

    assert migrated.representatives == migrated.bands
    assert all(
        np.isnan(block.residual)
        for blocks in migrated.legality.values()
        for block in blocks
    )


def test_conflicting_reciprocal_equivalent_representatives_are_rejected(
    sga, phonon
):
    """Equivalent keys must not overwrite one another during normalization."""
    with pytest.raises(ValueError, match=r"same wrapped q=X.*conflicting"):
        check_window_legality(
            sga,
            phonon,
            {X: (0, 1, 4), (-0.5, 0.0, 0.0): (0, 1, 2)},
        )


def test_window_band_indices_must_be_integers(sga, phonon):
    with pytest.raises(ValueError, match=r"X.*integer"):
        check_window_legality(sga, phonon, {X: (0, 1, 4.5)})


def test_window_bands_pass_through_dict_parameter_interface():
    """The consumer-facing flat params dict must not drop the window."""
    requested = {X: (0, 1, 4)}

    downfolder = Lawaf(model=object(), params={"nwann": 3, "window_bands": requested})

    assert downfolder.params.window_bands == requested


def test_legacy_positional_set_parameters_mapping_is_unchanged():
    downfolder = Lawaf(model=object())

    downfolder.set_parameters(
        "projected",
        (2, 2, 2),
        True,
        2,
        "unity",
        None,
        [0, 1],
    )

    assert downfolder.params.selected_basis == [0, 1]
    assert downfolder.params.window_bands is None



def test_phonopy_downfolder_expands_binary_windows_before_projection(
    tmp_path, phonon
):
    """Public downfold path validates stars, then combines them with one anchor."""
    downfolder = PhonopyDownfolder(
        phonon=phonon,
        params={
            "method": "projected",
            "kmesh": (2, 2, 2),
            "nwann": 3,
            "anchors": {GAMMA: (0, 1, 2)},
            "window_bands": Q7,
            "use_ws_distance": False,
        },
    )

    lwf = downfolder.downfold(
        output_path=tmp_path,
        write_hr_nc=None,
        write_hr_txt=None,
    )

    resolved = downfolder.params._window_bands_resolved
    assert isinstance(resolved, WindowBands)
    for ik, kpoint in enumerate(downfolder.builder.kpts):
        selected = resolved.bands[
            tuple(float(x) for x in np.round(np.mod(kpoint, 1.0), 8))
        ]
        expected = np.zeros(downfolder.builder.nband)
        expected[list(selected)] = 1.0
        np.testing.assert_array_equal(downfolder.builder.occ[ik], expected)
    assert np.isfinite(downfolder.builder.Amn).all()
    assert lwf.wannR.shape[-1] == 3

def _projected_builder(window_bands):
    """Three-k synthetic model with projectors taken only from Gamma."""
    phase = np.exp(2j * np.pi / 3)
    fourier = np.array(
        [[1, 1, 1], [1, phase, phase**2], [1, phase**2, phase]],
        dtype=complex,
    ) / np.sqrt(3)
    evecs = np.array([np.eye(3), fourier, fourier.conj()])
    evals = np.array([[0.0, 1.0, 2.0]] * 3)
    kpts = np.array([GAMMA, X, M])
    params = WannierParams(
        nwann=2,
        method="projected",
        anchors={GAMMA: (0, 2)},
        window_bands=window_bands,
    )
    return ProjectedWannierizer(
        evals=evals,
        evecs=evecs,
        kpts=kpts,
        kweights=np.full(3, 1 / 3),
        params=params,
    )


def test_binary_window_weights_coexist_with_one_anchor_projectors():
    """Listed q use exact 1/0 support; unlisted q retain projector weighting."""
    window = WindowBands(
        representatives={GAMMA: (0, 2), X: (0, 2)},
        bands={GAMMA: (0, 2), X: (0, 2)},
        legality={GAMMA: (), X: ()},
    )
    builder = _projected_builder(window)

    amn = builder.get_Amn()

    assert np.isfinite(amn).all()
    np.testing.assert_array_equal(amn[:2, 1, :], 0.0)
    assert np.linalg.norm(amn[2, 1, :]) > 0.1
    assert len(builder.projectors) == builder.nwann == 2



@pytest.mark.parametrize(
    ("builder_type", "method"),
    [(ProjectedWannierizer, "projected"), (ScdmkWannierizer, "scdmk")],
)
def test_rank_deficient_window_completion_stays_in_selected_rows(
    builder_type, method
):
    window = WindowBands(
        representatives={X: (0, 2)},
        bands={X: (0, 2)},
        legality={X: ()},
    )
    params = WannierParams(
        nwann=2,
        method=method,
        anchors={GAMMA: (0, 1)},
        window_bands=window,
    )
    builder = builder_type(
        evals=np.array([[0.0, 1.0, 2.0]] * 2),
        evecs=np.array([np.eye(3), np.eye(3)]),
        kpts=np.array([GAMMA, X]),
        kweights=np.full(2, 0.5),
        params=params,
    )

    amn = builder.get_Amn()

    np.testing.assert_array_equal(amn[1, 1], 0.0)
    np.testing.assert_allclose(
        amn[1, [0, 2]].conj().T @ amn[1, [0, 2]],
        np.eye(2),
        atol=1e-12,
    )

def test_window_band_count_must_equal_nwann():
    """A hand-selected original-band list defines the nwann-dimensional space."""
    window = WindowBands(
        representatives={GAMMA: (0,)},
        bands={GAMMA: (0,)},
        legality={GAMMA: ()},
    )

    with pytest.raises(ValueError, match=r"Gamma.*1 bands.*nwann=2"):
        _projected_builder(window)


def test_window_bands_reject_ambiguous_exclude_band_remapping():
    """Window indices cannot silently change meaning after band exclusion."""
    window = WindowBands(
        representatives={GAMMA: (0, 1)},
        bands={GAMMA: (0, 1)},
        legality={GAMMA: ()},
    )
    phase = np.exp(2j * np.pi / 3)
    fourier = np.array(
        [[1, 1, 1], [1, phase, phase**2], [1, phase**2, phase]],
        dtype=complex,
    ) / np.sqrt(3)
    params = WannierParams(
        nwann=2,
        method="projected",
        anchors={GAMMA: (0, 1)},
        exclude_bands=(1,),
        window_bands=window,
    )

    with pytest.raises(ValueError, match=r"Gamma.*excluded.*band"):
        ProjectedWannierizer(
            evals=np.array([[0.0, 1.0, 2.0]] * 3),
            evecs=np.array([np.eye(3), fourier, fourier.conj()]),
            kpts=np.array([GAMMA, X, M]),
            kweights=np.full(3, 1 / 3),
            params=params,
        )


def test_window_requires_every_expanded_star_arm_on_the_builder_mesh():
    """A validated star cannot be silently truncated by the supplied k mesh."""
    window = WindowBands(
        representatives={GAMMA: (0, 2), X: (0, 2), R: (0, 2)},
        bands={GAMMA: (0, 2), X: (0, 2), R: (0, 2)},
        legality={GAMMA: (), X: (), R: ()},
    )

    with pytest.raises(ValueError, match=r"R.*not present.*k-point mesh"):
        _projected_builder(window)


def test_scdmk_window_bands_restrict_amn_to_selected_modes():
    """The SCDM-k path applies the same binary per-q support as projected."""
    phase = np.exp(2j * np.pi / 3)
    fourier = np.array(
        [[1, 1, 1], [1, phase, phase**2], [1, phase**2, phase]],
        dtype=complex,
    ) / np.sqrt(3)
    params = WannierParams(
        nwann=2,
        method="scdmk",
        anchors={GAMMA: (0, 2)},
        window_bands=WindowBands(
            representatives={GAMMA: (0, 2), X: (0, 2)},
            bands={GAMMA: (0, 2), X: (0, 2)},
            legality={GAMMA: (), X: ()},
        ),
    )
    builder = ScdmkWannierizer(
        evals=np.array([[0.0, 1.0, 2.0]] * 3),
        evecs=np.array([np.eye(3), fourier, fourier.conj()]),
        kpts=np.array([GAMMA, X, M]),
        kweights=np.full(3, 1 / 3),
        params=params,
    )

    amn = builder.get_Amn()

    assert np.isfinite(amn).all()
    np.testing.assert_array_equal(amn[:2, 1, :], 0.0)
    assert np.linalg.norm(amn[2, 1, :]) > 0.1
