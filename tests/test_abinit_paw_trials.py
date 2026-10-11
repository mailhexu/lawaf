"""PAW dual-projector trial tests against independent Cartesian integrals.

Every reference here is a defining real-space Cartesian integral of the
authored tabulated radial functions (analytic compact polynomial times the
declared normalized real harmonic), evaluated on independent tensor
Gauss-Legendre grids with explicit periodic-image sums and Bloch phases.
No truncated plane-wave Gram matrix, partial-wave conversion or function
patching appears anywhere; the GaAs smoke uses the real provider source and
pinned XML datasets with explicit state ids.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawaf.interfaces.abinit.basis_datasets import (
    DatasetError,
    read_paw_xml_dataset,
    sha256_file,
)
from lawaf.interfaces.abinit.paw_provider import PawCoefficientProvider
from lawaf.interfaces.abinit.paw_trials import (
    PAW_GATE,
    QUADRATURE_INTERNAL_GATE,
    PawTrialBasis,
)
from lawaf.interfaces.abinit.source_arrays import (
    CallerArraySource,
    CallerMetadata,
    SymmetryMapEntry,
)
from lawaf.interfaces.abinit.source_wfk import WfkHeader, WfkReader
from lawaf.interfaces.abinit.trial_manifest import parse_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
GAAAS_DIR = REPO_ROOT / "abinit/tests/Pspdir/Psdj_paw_pbe_std"
GA_XML = GAAAS_DIR / "Ga.xml"
AS_XML = GAAAS_DIR / "As.xml"
GAAS_WFK = REPO_ROOT / ".tmp/w90-story065/gaas-pawo_DS2_WFK.nc"

needs_gaas = pytest.mark.skipif(
    not (GAAS_WFK.exists() and GA_XML.exists() and AS_XML.exists()),
    reason="GaAs PAW WFK/XML fixtures not present",
)

NODES = 96

_UNIT_NORM = None


def _dual_amplitude(scale, rc=1.55, a=0.005, d=0.006):
    """Amplitude of the authored dual: radial / int radial^2 r^2 dr."""
    global _UNIT_NORM
    if _UNIT_NORM is None:
        count = int(np.ceil(np.log1p(rc / a) / d)) + 1
        r = a * np.expm1(d * np.arange(count))
        jacobian = d * (r + a)
        _UNIT_NORM = simpson(_radial(r, rc, 1.0) ** 2 * r**2 * jacobian)
    return 1.0 / (scale * _UNIT_NORM)


# ---------------------------------------------------------------------------
# analytic compact fixtures: p(r) = scale*(rc^2 - r^2)^3 inside rc, else 0
# ---------------------------------------------------------------------------


def _radial(r, rc, scale):
    return scale * np.where(r < rc, (rc**2 - r**2) ** 3, 0.0)


def _write_species_xml(path, element, rc, scales, a=0.005, d=0.006):
    count = int(np.ceil(np.log1p(rc / a) / d)) + 1
    r = a * np.expm1(d * np.arange(count))
    values = {
        "ae_partial_wave": [],
        "pseudo_partial_wave": [],
        "projector_function": [],
    }
    channels = [(state_id, l, scale) for state_id, l, scale in scales]
    jacobian = d * (r + a)
    lines = [
        '<?xml version="1.0"?>',
        '<paw_dataset version="0.7">',
        f'  <atom symbol="{element}"/>',
        f'  <radial_grid eq="r=a*(exp(d*i)-1)" a="{a}" d="{d}" id="log1">',
        "    <values>",
        "      " + " ".join(f"{v:.16e}" for v in r),
        "    </values>",
        "  </radial_grid>",
        f'  <paw_radius rc="{rc}"/>',
        "  <valence_states>",
    ]
    for state_id, l, scale in channels:
        lines.append(f'    <state n="1" l="{l}" rc="{rc}" f="1.0" id="{state_id}"/>')
    lines.append("  </valence_states>")
    jacobian = d * (r + a)
    for state_id, l, scale in channels:
        radial = _radial(r, rc, scale)
        # genuine dual gauge: <p|t> = 1 within the one-channel l block
        dual = radial / simpson(radial**2 * r**2 * jacobian)
        lines.append(f'  <ae_partial_wave state="{state_id}" grid="log1">')
        lines.append("    " + " ".join(f"{v:.16e}" for v in radial))
        lines.append("  </ae_partial_wave>")
        lines.append(f'  <pseudo_partial_wave state="{state_id}" grid="log1">')
        lines.append("    " + " ".join(f"{v:.16e}" for v in radial))
        lines.append("  </pseudo_partial_wave>")
        lines.append(f'  <projector_function state="{state_id}" grid="log1">')
        lines.append("    " + " ".join(f"{v:.16e}" for v in dual))
        lines.append("  </projector_function>")
    lines += ["</paw_dataset>"]
    path.write_text("\n".join(lines) + "\n")
    return channels


def _header(xred, lmn_size, md5, kpt, rprimd=None, nspinor=1):
    xred = np.asarray(xred, float)
    kpt = np.asarray(kpt, float)
    fields = dict(
        codvsn="9.10.2",
        headform=80,
        fform=1,
        bantot=len(kpt),
        date=0,
        intxc=0,
        ixc=7,
        natom=len(xred),
        ngfft=np.array([12, 12, 12]),
        nkpt=len(kpt),
        nspden=1 if nspinor == 1 else 4,
        nspinor=nspinor,
        nsppol=1,
        nsym=1,
        npsp=len(lmn_size),
        ntypat=len(lmn_size),
        occopt=1,
        pertcase=0,
        usepaw=1,
        ecut=20.0,
        ecutdg=40.0,
        ecutsm=0.0,
        ecut_eff=20.0,
        qptn=np.zeros(3),
        rprimd=np.eye(3) * 3.0 if rprimd is None else np.asarray(rprimd, float),
        stmbias=0.0,
        tphysel=0.0,
        tsmear=0.0,
        usewvl=0,
        nshiftk_orig=1,
        nshiftk=1,
        mband=1,
        istwfk=np.ones(len(kpt), np.int32),
        istwfk_raw=np.ones(len(kpt), np.int32),
        nband=np.ones(len(kpt), np.int32),
        npwarr=np.full(len(kpt), 2, np.int32),
        so_psp=np.array([1] * len(lmn_size), np.int32),
        symafm=np.ones(1, np.int32),
        symrel=np.eye(3, dtype=np.int32)[None],
        typat=np.arange(1, len(lmn_size) + 1, dtype=np.int32),
        kpt=kpt,
        occ=np.ones((1, len(kpt)), float),
        tnons=np.zeros((1, 3)),
        znucltypat=np.array([13, 31, 33][: len(lmn_size)], np.int32),
        wtk=np.full(len(kpt), 1.0 / len(kpt)),
        residm=1e-16,
        xred=xred,
        etot=-1.0,
        fermie=0.5,
        amu=np.array([27.0, 69.0, 75.0][: len(lmn_size)]),
        kptopt=1,
        pawcpxocc=0,
        nelect=8.0,
        cellcharge=0.0,
        icoulomb=0,
        kptrlatt=np.eye(3, dtype=np.int32),
        kptrlatt_orig=np.eye(3, dtype=np.int32),
        shiftk_orig=np.zeros((1, 3)),
        shiftk=np.zeros((1, 3)),
        title=["fixture"] * len(lmn_size),
        znuclpsp=np.array([13.0, 31.0, 33.0][: len(lmn_size)]),
        zionpsp=np.array([3.0] * len(lmn_size)),
        pspso=np.array([1] * len(lmn_size), np.int32),
        pspdat=np.array([1] * len(lmn_size), np.int32),
        pspcod=np.array([8] * len(lmn_size), np.int32),
        pspxc=np.array([11] * len(lmn_size), np.int32),
        lmn_size=np.asarray(lmn_size, np.int32),
        md5_pseudos=list(md5),
        gmet=np.eye(3),
    )
    return WfkHeader(**fields)


def _caller_source(header, md5s):
    """Genuine CallerArraySource: one deterministic PW band per stored k."""
    nk, nsppol, nspinor = int(header.nkpt), int(header.nsppol), int(header.nspinor)
    states = {}
    kg_arrays = {}
    for ik in range(nk):
        npw = int(header.npwarr[ik])
        kg = np.zeros((3, npw), np.int32)
        kg[0, 1 % npw] = 1
        kg_arrays[ik] = np.ascontiguousarray(kg)
        for spin in range(nsppol):
            state = np.zeros((int(header.nband[ik + spin * nk]), npw, nspinor), complex)
            for component in range(nspinor):
                state[0, :, component] = (
                    (0.7 + 0.2 * component)
                    * np.exp(-0.3 * np.arange(npw))
                    * np.exp(1j * (0.9 * np.arange(npw) + 0.4 * ik + spin))
                )
            states[(ik, spin)] = np.ascontiguousarray(state)
    eigenvalues = np.zeros((1, nk, nsppol), float)
    for spin in range(nsppol):
        eigenvalues[0, :, spin] = 0.5 + 0.1 * np.arange(nk)
    symmetry_map = tuple(
        SymmetryMapEntry(
            mesh_point=point,
            ibz_point=point,
            operation=np.eye(3, dtype=np.int32),
            translation=np.zeros(3),
            reciprocal_shift=np.zeros(3, dtype=np.int32),
            spin_label="same",
            time_reversal=True,
        )
        for point in range(nk)
    )
    metadata = CallerMetadata(
        header=header,
        eigenvalues=np.ascontiguousarray(eigenvalues),
        eigenvalue_units="hartree",
        length_units="bohr",
        symmetry_map=symmetry_map,
        nkpt_mesh=nk,
        mesh_kpoints=np.ascontiguousarray(
            np.asarray(header.kpt, float)[:nk], dtype=float
        ),
        dataset_md5=tuple(md5s),
    )
    return CallerArraySource(metadata, states, kg_arrays)


def _rotation(alpha, beta):
    ca, sa, cb, sb = (
        np.cos(alpha),
        np.sin(alpha),
        np.cos(beta),
        np.sin(beta),
    )
    return np.array(
        [
            [ca * cb, sa * cb, sb],
            [-sa, ca, 0.0],
            [-ca * sb, -sa * sb, cb],
        ]
    )


def _tensor_quadrature(lo, hi, nodes=NODES):
    edges, weights = np.polynomial.legendre.leggauss(nodes)
    axes = []
    quad_axes = []
    for axis in range(3):
        half = 0.5 * (hi[axis] - lo[axis])
        mid = 0.5 * (hi[axis] + lo[axis])
        axes.append(edges * half + mid)
        quad_axes.append(weights * half)
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    quadrature = np.prod(
        np.stack(np.meshgrid(*quad_axes, indexing="ij"), axis=-1).reshape(-1, 3),
        axis=1,
    )
    return points, quadrature


def _trial_angular(xhat, axes, slot, l):
    """Normalized real harmonic, declared (z,x,y) slot order, independent."""
    local = np.asarray(xhat) @ np.asarray(axes).T
    if l == 0:
        return np.sqrt(1 / (4 * np.pi)) * np.ones(len(local))
    assert l == 1
    return np.sqrt(3 / (4 * np.pi)) * local[:, (2, 0, 1)[slot]]


def _ft_reference(
    radial_rc, radial_scale, axes, slot, l, tau, q_rows, volume, nodes=NODES
):
    """int e^{-iq.(x+tau)} p(|x|) Y_slot(xhat_local) dx / sqrt(V)."""
    lo, hi = -radial_rc, radial_rc
    points, quadrature = _tensor_quadrature([lo, lo, lo], [hi, hi, hi], nodes=nodes)
    magnitude = np.linalg.norm(points, axis=1)
    xhat = np.divide(
        points,
        magnitude[:, None],
        out=np.zeros_like(points),
        where=magnitude[:, None] != 0,
    )
    fields = _radial(
        magnitude, radial_rc, _dual_amplitude(radial_scale)
    ) * _trial_angular(xhat, axes, slot, l)
    result = np.empty(len(q_rows), complex)
    for row, q in enumerate(np.asarray(q_rows, float)):
        result[row] = (
            fields @ (quadrature * np.exp(-1j * (points + tau) @ q))
        ) / np.sqrt(volume)
    return result


def _overlap_reference(axes_s, displacement, scales=(1.0, 0.7, 0.7, 0.7), rc=1.55):
    """Full <p_a,i | g_s,j> matrix over (4 projector slots, 4 trial slots)."""
    lo = np.minimum(-rc, displacement - rc)
    hi = np.maximum(rc, displacement + rc)
    points, quadrature = _tensor_quadrature(lo, hi)
    magnitude = np.linalg.norm(points, axis=1)
    xhat = np.divide(
        points,
        magnitude[:, None],
        out=np.zeros_like(points),
        where=magnitude[:, None] != 0,
    )
    left = np.column_stack(
        [
            _radial(magnitude, rc, _dual_amplitude(scales[0]))
            * np.sqrt(1 / (4 * np.pi)),
            *[
                _radial(magnitude, rc, _dual_amplitude(scales[slot + 1]))
                * _trial_angular(xhat, np.eye(3), slot, 1)
                for slot in range(3)
            ],
        ]
    )
    relative = points - displacement
    magnitude = np.linalg.norm(relative, axis=1)
    xhat = np.divide(
        relative,
        magnitude[:, None],
        out=np.zeros_like(relative),
        where=magnitude[:, None] != 0,
    )
    right = np.column_stack(
        [
            _radial(magnitude, rc, _dual_amplitude(scales[0]))
            * np.sqrt(1 / (4 * np.pi)),
            *[
                _radial(magnitude, rc, _dual_amplitude(scales[slot + 1]))
                * _trial_angular(xhat, axes_s, slot, 1)
                for slot in range(3)
            ],
        ]
    )
    return left.T @ (quadrature[:, None] * right)


def _make_provider(tmp_path, name, xred, kpt, lattice, axes, rc=1.55, nspinor=1):
    xml = tmp_path / f"{name}.xml"
    channels = _write_species_xml(
        xml, name, rc, [(f"{name}1", 0, 1.0), (f"{name}2", 1, 0.7)]
    )
    dataset = read_paw_xml_dataset(xml)
    md5 = hashlib.md5(xml.read_bytes()).hexdigest()
    header = _header(
        xred,
        [4] * len(xred),
        [md5] * len(xred),
        kpt,
        rprimd=lattice,
        nspinor=nspinor,
    )
    provider = PawCoefficientProvider(_caller_source(header, [md5]), [xml] * len(xred))
    doc = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": "real_spherical_harmonics_wannier90",
            "normalization": "normalized",
            "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "trial_definition": "paw_dual_projector",
        "spin_frame_labels": False,
        "species": [
            {
                "name": name,
                "kind": "paw_xml",
                "dataset_path": str(xml),
                "sha256": sha256_file(xml),
                "radial_ids": [
                    {"id": f"{name}1", "label": "s", "l": 0, "harmonics": [0]},
                    {"id": f"{name}2", "label": "p", "l": 1, "harmonics": [0, 1, 2]},
                ],
            }
        ]
        * len(xred),
        "sites": [
            {
                "label": f"{name}{atom}",
                "species": name,
                "position": list(np.asarray(xred[atom], float)),
                "local_axes": [
                    list(row)
                    for row in (axes[atom] if isinstance(axes, list) else axes)
                ],
            }
            for atom in range(len(xred))
        ],
    }
    (path := tmp_path / f"{name}_manifest.json").write_text(json.dumps(doc))
    manifest = parse_manifest(path)
    basis = PawTrialBasis(manifest, provider)
    return basis, provider, dataset, channels, tmp_path


# ---------------------------------------------------------------------------
# overlap fixture: cubic 3 bohr cell, rc=1.55 -> support cutoff 3.1 bohr, so
# the six nearest periodic images genuinely overlap the trial
# ---------------------------------------------------------------------------

KPTS = np.array([[0.0, 0.0, 0.0], [0.13, -0.29, 0.41]])
AXES = _rotation(0.31, 0.62)


@pytest.fixture(scope="module")
def overlap_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paw_trials_overlap")
    basis, provider, dataset, channels, _ = _make_provider(
        tmp, "H", [[0.0, 0.0, 0.0]], KPTS, np.eye(3) * 3.0, AXES
    )
    return basis, provider, dataset, channels, tmp


def test_pw_trial_kets_match_defining_cartesian_fourier_transform(
    overlap_fixture,
):
    basis, provider, dataset, channels, _ = overlap_fixture
    volume = abs(np.linalg.det(basis.header.rprimd))
    rng = np.random.default_rng(7)
    kg = rng.uniform(-0.4, 0.4, size=(7, 3))
    for ik in range(len(KPTS)):
        kets = basis(ik, kg)
        assert kets.shape == (len(kg), 4)
        for column, (l, slot, scale) in enumerate(
            [(0, 0, 1.0), (1, 0, 0.7), (1, 1, 0.7), (1, 2, 0.7)]
        ):
            expected = _ft_reference(
                1.55,
                scale,
                AXES,
                slot,
                l,
                np.zeros(3),
                (kg + KPTS[ik]) @ basis._reciprocal,
                volume,
                nodes=160,
            )
            np.testing.assert_allclose(kets[:, column], expected, atol=1e-8, rtol=1e-8)


def test_dual_coefficients_are_defining_image_summed_real_space_integrals(
    overlap_fixture,
):
    basis, provider, dataset, channels, _ = overlap_fixture
    lattice = np.asarray(basis.header.rprimd, float)
    images = []
    for n in np.ndindex(3, 3, 3):
        shift = (np.asarray(n) - 1) @ lattice
        if np.linalg.norm(shift) <= 2 * 1.55:
            images.append(shift)
    assert sorted(np.linalg.norm(images, axis=1).round(9).tolist()) == sorted(
        [0.0] + [3.0] * 6
    )
    for ik in range(len(KPTS)):
        computed = basis.dual_coefficients(ik)
        expected = np.zeros_like(computed)
        for shift in images:
            expected += np.exp(
                1j * (KPTS[ik] @ (2 * np.pi * np.eye(3))) @ shift
            ) * _overlap_reference(AXES, shift)
        np.testing.assert_allclose(
            computed, expected, atol=1e-8, rtol=1e-8 * max(1.0, np.max(abs(expected)))
        )


def test_dual_coefficients_are_reciprocal_lattice_periodic(tmp_path):
    """D(k + reciprocal lattice vector) == D(k); genuinely k dependent otherwise."""
    lattice = np.eye(3) * 2.0  # images at 2.0 bohr overlap the rc=1.55 supports
    basis, provider, dataset, channels, _ = _make_provider(
        tmp_path,
        "H",
        [[0.0, 0.0, 0.0]],
        np.array([[0.0, 0.0, 0.0], KPTS[1], KPTS[1] + np.array([1.0, 0.0, 0.0])]),
        lattice,
        AXES,
    )
    images = [
        pair for pair in basis._neighbor_images[(0, 0)] if np.linalg.norm(pair[0]) > 0
    ]
    assert images  # genuinely overlapping periodic images exist
    gamma = basis.dual_coefficients(0)
    k1 = basis.dual_coefficients(1)
    k1_shifted = basis.dual_coefficients(2)
    np.testing.assert_allclose(k1_shifted, k1, atol=1e-12, rtol=0)
    assert np.max(abs(k1 - gamma)) > 1e-4


def test_geometry_cache_is_static_and_bounded_independent_of_nk(overlap_fixture):
    basis, *_ = overlap_fixture
    sizes = set()
    for ik in range(len(KPTS)):
        basis.dual_coefficients(ik)
        sizes.add(len(basis._geometry_cache))
    assert sizes == {7}
    # one concentric entry plus six distinct image displacements


def test_manifest_contract_failures(tmp_path):
    basis, provider, dataset, channels, _ = _make_provider(
        tmp_path, "H", [[0.0, 0.0, 0.0]], KPTS[:1], np.eye(3) * 3.0, AXES
    )
    manifest = basis.manifest
    doc = json.loads(Path(manifest.path).read_text())
    for mutate, field in [
        (lambda d: d.update(trial_definition="pseudo_radial"), "trial_definition"),
        (
            lambda d: d["species"][0]["radial_ids"][0].update(id="bogus"),
            "radial_id",
        ),
        (lambda d: d.update(spin_frame_labels=True), "spin_label"),
        (
            lambda d: d["sites"][0].update(position=[0.1, 0.0, 0.0]),
            "sites.position",
        ),
        (lambda d: d["sites"][0].update(species="Xx"), "species"),
    ]:
        broken = json.loads(json.dumps(doc))
        mutate(broken)
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken))
        with pytest.raises(DatasetError):
            PawTrialBasis(parse_manifest(path), provider)


def test_nonoverlap_fixture_uses_exact_concentric_defining_gram(tmp_path):
    """Large cell: only concentric images exist; blocks are the radial Gram."""
    lattice = np.eye(3) * 10.0
    xred = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    basis, provider, dataset, channels, _ = _make_provider(
        tmp_path, "H", xred, KPTS[:2], lattice, np.eye(3), rc=1.6
    )
    for (atom, site), images in basis._neighbor_images.items():
        if atom == site:
            assert len(images) == 1
            assert np.linalg.norm(images[0][0]) == 0.0
        else:
            assert images == ()
    grid = dataset.grid
    jacobian = grid.d * (grid.r + grid.a)
    unit = simpson(_radial(grid.r, 1.6, 1.0) ** 2 * grid.r**2 * jacobian)
    gram_unit = 1.0 / unit  # <p|p> for the authored dual gauge
    # the native-grid Gram must agree with an independent dense evaluation
    dense, weights = np.polynomial.legendre.leggauss(80)
    dense_r = 0.8 * (dense + 1)
    dense_w = 0.8 * weights
    dense_unit = dense_w @ (_radial(dense_r, 1.6, 1.0) ** 2 * dense_r**2)
    np.testing.assert_allclose(unit, dense_unit, atol=1e-6, rtol=1e-6)
    for ik in range(2):
        dual = basis.dual_coefficients(ik)
        assert dual.shape == (8, 8)
        onsite = np.zeros((4, 4))
        onsite[0, 0] = gram_unit
        onsite[1:4, 1:4] = np.eye(3) * gram_unit / 0.49
        for block in (slice(0, 4), slice(4, 8)):
            np.testing.assert_allclose(dual[block, block], onsite, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(dual[0:4, 4:8], 0.0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(dual[4:8, 0:4], 0.0, atol=1e-12, rtol=0)


def test_image_enumeration_matches_brute_force_on_displaced_and_skewed_cells(
    tmp_path,
):
    """Exact integer windows: general fractional sites and a skewed lattice."""
    cases = [
        (np.eye(3) * 3.0, [[0.37, 0.11, 0.83]], 1.2),
        (
            np.array([[3.0, 0.0, 0.0], [2.7, 0.3, 0.0], [0.9, 0.2, 3.0]]),
            [[0.0, 0.0, 0.0], [0.4, 0.5, 0.6]],
            1.2,
        ),
    ]
    for lattice, xred, rc in cases:
        basis, *_ = _make_provider(
            tmp_path, "H", xred, KPTS[:1], lattice, np.eye(3), rc=rc
        )
        positions = np.asarray(basis.header.xred, float) @ np.asarray(
            basis.header.rprimd, float
        )
        cutoff = 2 * rc
        for (atom, site), images in basis._neighbor_images.items():
            delta = positions[site] - positions[atom]
            brute = {}
            for n in np.ndindex(25, 25, 25):
                shift = (np.asarray(n) - 12) @ lattice
                displacement = delta + shift
                if np.linalg.norm(displacement) <= cutoff:
                    brute[tuple(np.round(displacement, 9))] = tuple(
                        int(v) for v in (np.asarray(n) - 12)
                    )
            code = {
                tuple(np.round(displacement, 9)): tuple(int(v) for v in n)
                for displacement, n in images
            }
            assert code == brute, (atom, site, code, brute)
            assert images  # the fixtures genuinely have overlapping images


def test_displaced_site_dual_matches_image_summed_cartesian_reference(tmp_path):
    """Off-center site: asymmetric image windows and Bloch phases e^{2pi i k.n}."""
    basis, provider, dataset, channels, _ = _make_provider(
        tmp_path,
        "H",
        [[0.37, 0.11, 0.83]],
        KPTS,
        np.eye(3) * 3.0,
        AXES,
    )
    lattice = np.asarray(basis.header.rprimd, float)
    images = []
    for n in np.ndindex(9, 9, 9):
        shift = (np.asarray(n) - 4) @ lattice
        if np.linalg.norm(shift) <= 2 * 1.55:
            images.append(shift)
    assert len(images) >= 4  # asymmetric periodic images on both sides
    for ik in range(len(KPTS)):
        computed = basis.dual_coefficients(ik)
        expected = np.zeros_like(computed)
        for shift in images:
            expected += np.exp(
                1j
                * (2 * np.pi * (KPTS[ik] @ np.rint(np.linalg.solve(lattice.T, shift))))
            ) * _overlap_reference(AXES, shift)
        np.testing.assert_allclose(
            computed, expected, atol=1e-8, rtol=1e-8 * max(1.0, np.max(abs(expected)))
        )
    for report in basis.quadrature_report.values():
        assert report["residual"] <= report["gate"] * report["scale"]
        assert report["gate"] in (QUADRATURE_INTERNAL_GATE, PAW_GATE)


def test_spinor_source_headers_are_not_rejected_here(tmp_path):
    basis, provider, *_ = _make_provider(
        tmp_path,
        "H",
        [[0.0, 0.0, 0.0]],
        KPTS[:1],
        np.eye(3) * 3.0,
        AXES,
        nspinor=2,
    )
    assert basis.dual_coefficients(0).shape == (4, 4)


# ---------------------------------------------------------------------------
# real GaAs pinned smoke (slow): actual provider source, actual XML state ids
# ---------------------------------------------------------------------------


@needs_gaas
def test_pinned_gaas_partial_wave_trials_admit_closure_and_reject_raw_dual_trials(
    tmp_path,
):
    source = WfkReader.open(GAAS_WFK)
    header = source.header
    assert header.usepaw
    xmls = [GA_XML, AS_XML]
    md5s = [hashlib.md5(p.read_bytes()).hexdigest() for p in xmls]
    assert [str(m) for m in md5s] == [str(v).strip() for v in header.md5_pseudos]
    provider = PawCoefficientProvider(source, xmls)
    elements = [provider.kernels[int(t) - 1].dataset.element for t in header.typat]
    assert elements == ["Ga", "As"]
    ga, as_ = provider.kernels
    ga_ids = [c.state_id for c in ga.dataset.channels]
    as_ids = [c.state_id for c in as_.dataset.channels]
    doc = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": "real_spherical_harmonics_wannier90",
            "normalization": "normalized",
            "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "trial_definition": "paw_augmented_partial_wave",
        "spin_frame_labels": False,
        "species": [
            {
                "name": "Ga",
                "kind": "paw_xml",
                "dataset_path": str(GA_XML),
                "sha256": sha256_file(GA_XML),
                "radial_ids": [
                    {"id": "Ga1", "label": "s", "l": 0, "harmonics": [0]},
                    {"id": "Ga3", "label": "p", "l": 1, "harmonics": [0, 1, 2]},
                ],
            },
            {
                "name": "As",
                "kind": "paw_xml",
                "dataset_path": str(AS_XML),
                "sha256": sha256_file(AS_XML),
                "radial_ids": [
                    {"id": "As1", "label": "s", "l": 0, "harmonics": [0]},
                    {"id": "As3", "label": "p", "l": 1, "harmonics": [0, 1, 2]},
                ],
            },
        ],
        "sites": [
            {
                "label": "Ga0",
                "species": "Ga",
                "position": [0.0, 0.0, 0.0],
                "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "label": "As1",
                "species": "As",
                "position": [0.25, 0.25, 0.25],
                "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        ],
    }
    (path := tmp_path / "gaas.json").write_text(json.dumps(doc))
    basis = PawTrialBasis(parse_manifest(path), provider)
    invalid = dict(doc, trial_definition="paw_dual_projector")
    bad_path = tmp_path / "inadmissible-dual-trials.json"
    bad_path.write_text(json.dumps(invalid))
    with pytest.raises(DatasetError, match="D5 trial local closure"):
        PawTrialBasis(parse_manifest(bad_path), provider)
    assert all(
        row["normalized_residual"] <= PAW_GATE for row in basis.closure_report.values()
    )

    # geometry: the pinned radii are non-overlapping; assert it explicitly.
    positions = np.asarray(header.xred, float) @ np.asarray(header.rprimd, float)
    separation = float(np.linalg.norm(positions[1] - positions[0]))
    assert separation > ga.dataset.rc + as_.dataset.rc

    dual = basis.dual_coefficients(0)
    assert dual.shape == (basis.offsets[-1], 8)
    # cross-species blocks vanish exactly under the verified non-overlap
    np.testing.assert_allclose(dual[provider.offsets[1] :, :4], 0.0, atol=1e-13, rtol=0)
    np.testing.assert_allclose(dual[: provider.offsets[1], 4:], 0.0, atol=1e-13, rtol=0)

    # onsite blocks equal the concentric radial Gram of the declared state ids
    def radial_gram(kernel, left_id, right_id):
        grid = kernel.dataset.grid
        jacobian = grid.d * (grid.r + grid.a)
        left = next(c for c in kernel.dataset.channels if c.state_id == left_id)
        right = next(c for c in kernel.dataset.channels if c.state_id == right_id)
        return simpson(
            left.dual_projector * right.pseudo_partial_wave * grid.r**2 * jacobian
        )

    for atom, (kernel, prefix) in enumerate([(ga, "Ga"), (as_, "As")]):
        rows = slice(provider.offsets[atom], provider.offsets[atom + 1])
        columns = slice(4 * atom, 4 * atom + 4)
        expected = np.zeros((rows.stop - rows.start, 4))
        expected[0, 0] = radial_gram(kernel, f"{prefix}1", f"{prefix}1")
        expected[1, 0] = radial_gram(kernel, f"{prefix}2", f"{prefix}1")
        l1_l1 = radial_gram(kernel, f"{prefix}3", f"{prefix}3")
        l1_l2 = radial_gram(kernel, f"{prefix}4", f"{prefix}3")
        expected[2:5, 1:4] = np.eye(3) * l1_l1
        expected[5:8, 1:4] = np.eye(3) * l1_l2
        np.testing.assert_allclose(dual[rows, columns], expected, atol=1e-9, rtol=1e-9)
    source.close()


# ---------------------------------------------------------------------------
# real Fe pinned admission (no WFK needed): the actual tabulated 4s/3p/3d
# states close the D5 gate only under the two certified dataset-generator
# quadrature conventions
# ---------------------------------------------------------------------------

FE_XML = GAAAS_DIR / "Fe.xml"
FE_TRIAL_IDS = ("Fe2", "Fe3", "Fe5")  # actual XML state ids of 4s, 3p, 3d

needs_fe = pytest.mark.skipif(
    not FE_XML.exists(), reason="Fe PAW XML fixture not present"
)


def _overint_reference(integrand):
    """m_paw_atom_solve integrator, re-derived from the documented Fortran:
    per-integrand trailing trim of machine_zero = machine_precision**5
    samples keeping the first following zero, Simpson anchored at the
    trimmed last node, first trapezoid if even (the log-grid icorr=-1
    path)."""
    significant = np.flatnonzero(np.abs(integrand) >= np.finfo(np.float64).eps ** 5)
    if not significant.size:
        return 0.0
    count = min(integrand.shape[0], int(significant[-1]) + 2)
    weights = np.zeros(count)
    if count % 2:
        weights[0 : count - 2 : 2] += 1 / 3
        weights[1 : count - 1 : 2] += 4 / 3
        weights[2:count:2] += 1 / 3
    else:
        weights[0] = weights[1] = 1 / 2
        weights[1 : count - 2 : 2] += 1 / 3
        weights[2 : count - 1 : 2] += 4 / 3
        weights[3:count:2] += 1 / 3
    return float(weights @ integrand[:count])


def _simpson_gen_reference(count):
    """m_pawrad simp_gen weights, re-derived from the documented Fortran:
    Simpson anchored at the last node after reversal; Cartwright correction
    (5/12, 2/3, -1/12) on the first three nodes when even; two nodes are the
    plain trapezoid."""
    weights = np.zeros(count)
    if count % 2:
        weights[0 : count - 2 : 2] += 1 / 3
        weights[1 : count - 1 : 2] += 4 / 3
        weights[2:count:2] += 1 / 3
    elif count == 2:
        weights[:] = 1 / 2
    else:
        weights[0 : count - 3 : 2] += 1 / 3
        weights[1 : count - 2 : 2] += 4 / 3
        weights[2 : count - 1 : 2] += 1 / 3
        weights[-1] += 5 / 12
        weights[-2] += 2 / 3
        weights[-3] -= 1 / 12
    return weights


def _fe_native_closure_residuals(dataset, state_ids):
    """Independent certified-convention D5 closure residuals from the raw
    tabulated dataset: overint projector coefficients E, c on the native
    grid (radial Jacobian and r^2 explicit); common simp_gen augmentation
    weights (nearest rc node) for the defining, smooth and Delta-G overlaps
    of defining versus smooth + E^T DeltaG c."""
    grid = dataset.grid
    r, jacobian = grid.r, grid.d * (grid.r + grid.a)
    ids = [channel.state_id for channel in dataset.channels]
    ls = np.array([channel.l for channel in dataset.channels])
    all_pseudo = np.array([c.pseudo_partial_wave for c in dataset.channels]).T
    all_ae = np.array([c.ae_partial_wave for c in dataset.channels]).T
    all_projectors = np.array([c.dual_projector for c in dataset.channels]).T
    nearest = int(np.argmin(np.abs(r - dataset.rc))) + 1
    native = np.zeros(r.shape[0])
    native[:nearest] = (
        _simpson_gen_reference(nearest)[::-1] * jacobian[:nearest] * r[:nearest] ** 2
    )
    residuals = {}
    for state_id in state_ids:
        column = ids.index(state_id)
        ix = np.flatnonzero(ls == ls[column])
        u, ae, p = all_pseudo[:, ix], all_ae[:, ix], all_projectors[:, ix]
        delta = ae - u
        trial = u[:, int(np.flatnonzero(ix == column)[0])].copy()
        trial[r > dataset.rc] = 0.0
        e = np.array(
            [
                [
                    _overint_reference(jacobian * r**2 * p[:, k] * u[:, m])
                    for m in range(u.shape[1])
                ]
                for k in range(p.shape[1])
            ]
        )
        c = np.array(
            [
                _overint_reference(jacobian * r**2 * p[:, k] * trial)
                for k in range(p.shape[1])
            ]
        )
        defining = (u + delta @ e).T @ (native * (trial + delta @ c))
        smooth = u.T @ (native * trial)
        delta_g = ae.T @ (native[:, None] * ae) - u.T @ (native[:, None] * u)
        metric = smooth + e.T @ delta_g @ c
        residuals[state_id] = float(
            np.max(np.abs(defining - metric))
            / max(1.0, float(np.max(np.abs(defining))))
        )
    return residuals


def _fe_manifest(tmp_path, trial_definition):
    doc = {
        "schema": "lawaf_trial_basis_manifest",
        "version": "1.0",
        "units": {"length": "bohr", "energy": "hartree"},
        "position_coordinates": "reduced",
        "harmonic_convention": {
            "name": "real_spherical_harmonics_wannier90",
            "normalization": "normalized",
            "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "trial_definition": trial_definition,
        "spin_frame_labels": False,
        "species": [
            {
                "name": "Fe",
                "kind": "paw_xml",
                "dataset_path": str(FE_XML),
                "sha256": sha256_file(FE_XML),
                "radial_ids": [
                    {"id": "Fe2", "label": "4s", "l": 0, "harmonics": [0]},
                    {"id": "Fe3", "label": "3p", "l": 1, "harmonics": [0, 1, 2]},
                    {"id": "Fe5", "label": "3d", "l": 2, "harmonics": [0, 1, 2, 3, 4]},
                ],
            }
        ],
        "sites": [
            {
                "label": "Fe0",
                "species": "Fe",
                "position": [0.0, 0.0, 0.0],
                "local_axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
    }
    path = tmp_path / "fe.json"
    path.write_text(json.dumps(doc))
    return parse_manifest(path)


@needs_fe
def test_pinned_fe_admission_closes_with_certified_native_quadrature(tmp_path):
    """Actual Fe 4s/3p/3d states: the admission D5 gate closes below 1e-8
    against an independent certified-convention closure, the reported
    residuals match that reference, and the raw dual-projector trial guard
    stays red."""
    md5 = hashlib.md5(FE_XML.read_bytes()).hexdigest()
    header = _header(
        [[0.0, 0.0, 0.0]], [18], [md5], KPTS[:1], rprimd=np.eye(3) * 10.0
    )
    provider = PawCoefficientProvider(_caller_source(header, [md5]), [FE_XML])
    basis = PawTrialBasis(
        _fe_manifest(tmp_path, "paw_augmented_partial_wave"), provider
    )
    independent = _fe_native_closure_residuals(
        read_paw_xml_dataset(FE_XML), FE_TRIAL_IDS
    )
    for state_id in FE_TRIAL_IDS:
        assert independent[state_id] < PAW_GATE
    assert sorted(state_id for _, state_id in basis.closure_report) == sorted(
        FE_TRIAL_IDS
    )
    for (_, state_id), report in basis.closure_report.items():
        assert report["gate"] == PAW_GATE
        assert report["normalized_residual"] <= 1e-12 < PAW_GATE
        assert abs(report["normalized_residual"] - independent[state_id]) < 1e-12
    with pytest.raises(DatasetError, match="D5 trial local closure"):
        PawTrialBasis(_fe_manifest(tmp_path, "paw_dual_projector"), provider)
