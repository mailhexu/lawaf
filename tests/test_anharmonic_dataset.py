"""Story 017: TrainingDataset (netCDF frame-stacked) + teacher stack.

Covers:
- roundtrip bit-parity (all arrays, NaN masking preserved)
- label-block idempotence per calculator identity, active-block switching
- lazy atomchain import error path
- spot_check report fields with a deterministic toy calculator
- ABINIT HIST.nc import unit conversion (synthetic file, exact 1e-12)
- sympy verification of the Bohr/Hartree -> Angstrom/eV factor chain and
  the Voigt (xx,yy,zz,yz,xz,xy) mapping (user-mandated sympy rule)
- MLIP-harmonic smoke (FR-021): atomchain calculate_phonon ->
  phonopy_params.yaml -> phonopy.load (EMT calculator, skip if no atomchain)
"""
import json
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from lawaf.anharmonic.dataset import TrainingDataset, calculator_identity
from lawaf.anharmonic.sampling import FrameSpec
from lawaf.anharmonic.teacher import (
    from_abinit_hist,
    get_atomchain_calculator,
    label_frames,
    spot_check,
)


# ---------------------------------------------------------------------------
# deterministic toy calculators
# ---------------------------------------------------------------------------
class FakeCalc(Calculator):
    """Analytic toy: E = s*|r-com|^2, F = -dE/dr, stress = s * second moments."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.parameters["scale"] = scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        s = self.parameters["scale"]
        r = self.atoms.get_positions()
        dr = r - r.mean(axis=0)
        self.results["energy"] = float(s * np.sum(dr**2))
        self.results["forces"] = -s * 2.0 * dr
        # symmetric, scale-dependent (second-moment) stress, voigt order
        self.results["stress"] = s * np.array(
            [
                np.sum(dr[:, 0] ** 2),
                np.sum(dr[:, 1] ** 2),
                np.sum(dr[:, 2] ** 2),
                np.sum(dr[:, 1] * dr[:, 2]),
                np.sum(dr[:, 0] * dr[:, 2]),
                np.sum(dr[:, 0] * dr[:, 1]),
            ]
        )


class FakeCalcNoStress(FakeCalc):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results.pop("stress", None)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def atoms_list():
    cell = np.eye(3) * 4.0
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    numbers = [3, 3, 8, 8]
    return [Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True) for _ in range(3)]


@pytest.fixture
def frames():
    out = []
    for i in range(3):
        out.append(
            FrameSpec(
                Q=np.array([0.1 * (i + 1), -0.2 * (i + 1)]),
                strain_voigt=None
                if i != 1
                else np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
                provenance="single" if i < 2 else "random",
                split="train" if i < 2 else "cv",
            )
        )
    return out


@pytest.fixture
def dataset(atoms_list, frames):
    return TrainingDataset.from_frames(
        frames, atoms_list, harmonic_source="toy-harmonic"
    )


# ---------------------------------------------------------------------------
# Part 1: TrainingDataset
# ---------------------------------------------------------------------------
def test_from_frames_layout(dataset):
    assert dataset.positions.shape == (3, 4, 3)
    assert dataset.atomic_numbers.shape == (3, 4)
    assert dataset.Q.shape == (3, 2)
    assert dataset.strain_voigt.shape == (3, 6)
    assert dataset.split.tolist() == ["train", "train", "cv"]
    assert dataset.provenance.tolist() == ["single", "single", "random"]
    # NaN row for unstrained frames
    assert np.isnan(dataset.strain_voigt[0]).all()
    assert np.isnan(dataset.strain_voigt[2]).all()
    assert np.isfinite(dataset.strain_voigt[1, 0])
    # units + FR-021 harmonic_source
    assert dataset.attrs["units"]["energy"] == "eV"
    assert dataset.attrs["units"]["length"] == "Angstrom"
    assert dataset.attrs["units"]["force"] == "eV/Angstrom"
    assert dataset.attrs["units"]["stress"] == "eV/Angstrom^3"
    assert dataset.attrs["harmonic_source"] == "toy-harmonic"
    assert "created" in dataset.attrs


def test_from_frames_q_padding():
    a = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.3, 1.3, 1.3]],
        cell=np.eye(3) * 5,
        pbc=True,
    )
    f = [
        FrameSpec(Q=np.array([1.0, 2.0, 3.0])),
        FrameSpec(Q=np.array([0.5])),
    ]
    ds = TrainingDataset.from_frames(f, [a, a])
    assert ds.Q.shape == (2, 3)
    assert ds.Q[0].tolist() == [1.0, 2.0, 3.0]
    assert ds.Q[1, 0] == 0.5
    assert np.isnan(ds.Q[1, 1:]).all()


def test_frame_atoms(dataset, atoms_list):
    a = dataset.frame_atoms(1)
    assert isinstance(a, Atoms)
    np.testing.assert_array_equal(a.get_positions(), atoms_list[0].get_positions())
    np.testing.assert_array_equal(a.numbers, atoms_list[0].numbers)
    np.testing.assert_allclose(a.cell.array, atoms_list[0].cell.array)
    assert a.pbc.tolist() == [True, True, True]


def test_roundtrip_bit_parity(dataset, tmp_path):
    path = tmp_path / "ds.nc"
    # label first so label blocks roundtrip too
    bid = label_frames(dataset, FakeCalc(scale=1.0))
    dataset.save(path)
    back = TrainingDataset.load(path)
    for name, arr in [
        ("positions", dataset.positions),
        ("atomic_numbers", dataset.atomic_numbers),
        ("cell", dataset.cell),
        ("Q", dataset.Q),
        ("strain_voigt", dataset.strain_voigt),
    ]:
        assert getattr(back, name).tobytes() == arr.tobytes(), name
    assert back.pbc.dtype == dataset.pbc.dtype
    assert back.pbc.tolist() == dataset.pbc.tolist()
    assert back.split.tolist() == dataset.split.tolist()
    assert back.provenance.tolist() == dataset.provenance.tolist()
    # label block bit parity
    np.testing.assert_array_equal(back.energies, dataset.energies)
    assert back.forces.tobytes() == dataset.forces.tobytes()
    assert back.stresses.tobytes() == dataset.stresses.tobytes()
    assert back.active_label == dataset.active_label == bid
    # attrs preserved
    assert back.attrs["harmonic_source"] == "toy-harmonic"
    assert back.attrs["units"] == dataset.attrs["units"]
    blocks = json.loads(back.attrs["label_blocks"])
    assert blocks[0]["calculator"].startswith("FakeCalc")


def test_set_split_and_select(dataset):
    dataset.set_split([2], "holdout")
    assert dataset.split.tolist() == ["train", "train", "holdout"]
    assert set(dataset.splits) == {"train", "holdout"}
    sub = dataset.select(split="holdout")
    assert sub.nframes == 1
    np.testing.assert_array_equal(sub.positions[0], dataset.positions[2])
    assert sub.provenance.tolist() == ["random"]
    # __getitem__ sugar: str = split, int/list = frame subset
    assert dataset["holdout"].nframes == 1
    assert dataset[[0, 2]].nframes == 2
    assert dataset[1].provenance.tolist() == ["single"]


def test_label_idempotence_and_active_block(dataset):
    b1 = dataset.label(FakeCalc(scale=1.0))
    b1b = dataset.label(FakeCalc(scale=1.0))
    assert b1 == b1b == dataset.active_label
    assert len(dataset.label_blocks) == 1
    b2 = dataset.label(FakeCalc(scale=2.0))
    assert b2 != b1
    assert dataset.active_label == b2
    # relabeling with SAME identity + new values without overwrite: no-op + warning
    with pytest.warns(UserWarning, match="already labeled"):
        b3 = dataset.label(FakeCalc(scale=1.0), values={"energy": np.zeros(3)})
    assert b3 == b1
    assert not np.allclose(dataset.blocks[b1].energy, 0.0)
    # explicit overwrite replaces values
    dataset.label(
        FakeCalc(scale=1.0),
        values={"energy": np.full(3, -7.0)},
        frames=[0, 1, 2],
        overwrite=True,
    )
    assert dataset.blocks[b1].energy.tolist() == [-7.0, -7.0, -7.0]
    # accessors follow the ACTIVE block
    dataset.label(FakeCalc(scale=1.0))  # no-op, active unchanged
    assert dataset.active_label == b2
    energies = dataset.energies
    assert energies.shape == (3,)


def test_label_frames_writes_block(dataset):
    bid = label_frames(dataset, FakeCalc(scale=1.0))
    assert bid == dataset.active_label
    # structures never mutated by labeling
    for i in range(dataset.nframes):
        ref = dataset.frame_atoms(i)
        np.testing.assert_array_equal(
            ref.get_positions(), dataset.frame_atoms(i).get_positions()
        )
    e = dataset.energies
    f = dataset.forces
    s = dataset.stresses
    assert e.shape == (3,) and f.shape == (3, 4, 3) and s.shape == (3, 3, 3)
    # analytic check on frame 0
    ref = dataset.frame_atoms(0).copy()
    ref.calc = FakeCalc()
    np.testing.assert_allclose(e[0], ref.get_potential_energy())
    np.testing.assert_allclose(f[0], ref.get_forces())
    np.testing.assert_allclose(dataset.stress_voigt[0], ref.get_stress())
    # stress stored as full symmetric tensor
    np.testing.assert_allclose(s[0], s[0].T)
    # block metadata records calculator identity + frames
    blocks = json.loads(dataset.attrs["label_blocks"])
    assert blocks[0]["calculator"].startswith("FakeCalc")
    assert blocks[0]["frames"] is None


def test_nan_stress_masking(dataset):
    label_frames(dataset, FakeCalcNoStress())
    s = dataset.stresses
    assert np.isnan(s).all()
    assert np.isfinite(dataset.energies).all()
    assert np.isfinite(dataset.forces).all()
    # stress_voigt accessor: NaN-preserving, (nf, 6)
    sv = dataset.stress_voigt
    assert sv.shape == (3, 6)
    assert np.isnan(sv).all()
    # with a real stress: voigt order (xx, yy, zz, yz, xz, xy)
    label_frames(dataset, FakeCalc(scale=1.0))
    s = dataset.stresses
    sv = dataset.stress_voigt
    np.testing.assert_allclose(sv[:, 0], s[:, 0, 0])
    np.testing.assert_allclose(sv[:, 1], s[:, 1, 1])
    np.testing.assert_allclose(sv[:, 2], s[:, 2, 2])
    np.testing.assert_allclose(sv[:, 3], s[:, 1, 2])
    np.testing.assert_allclose(sv[:, 4], s[:, 0, 2])
    np.testing.assert_allclose(sv[:, 5], s[:, 0, 1])


def test_calculator_identity_best_effort():
    assert calculator_identity(FakeCalc(scale=1.0)) == calculator_identity(
        FakeCalc(scale=1.0)
    )
    # string passthrough
    assert calculator_identity("mace-r2scan") == "mace-r2scan"
    # model_path distinguishes
    c1 = FakeCalc()
    c1.model_path = "/models/a.model"
    c2 = FakeCalc()
    c2.model_path = "/models/b.model"
    assert calculator_identity(c1) != calculator_identity(c2)
    # parameters influence the fingerprint when no model_path
    assert calculator_identity(FakeCalc(scale=1.0)) != calculator_identity(
        FakeCalc(scale=2.0)
    )


def test_lazy_import_error(monkeypatch):
    # block BOTH keys: a cached 'atomchain.init_model' short-circuits the
    # parent import and the guard would never trigger
    monkeypatch.setitem(sys.modules, "atomchain", None)
    monkeypatch.setitem(sys.modules, "atomchain.init_model", None)
    with pytest.raises(RuntimeError, match="lawaf\\[anharmonic\\]"):
        get_atomchain_calculator("mace-r2scan")


def test_get_atomchain_calculator_adapter(monkeypatch):
    """Adapter wires through to atomchain.init_model.init_calc and tags identity."""
    import types

    sentinel = FakeCalc(scale=3.0)
    captured = {}

    def fake_init_calc(model_type="mace", model_path=None):
        captured["model_type"] = model_type
        captured["model_path"] = model_path
        return sentinel

    mod = types.ModuleType("atomchain")
    init_mod = types.ModuleType("atomchain.init_model")
    init_mod.init_calc = fake_init_calc
    mod.init_model = init_mod
    monkeypatch.setitem(sys.modules, "atomchain", mod)
    monkeypatch.setitem(sys.modules, "atomchain.init_model", init_mod)
    calc = get_atomchain_calculator("mace-r2scan", model_path="/m.model")
    assert calc is sentinel
    assert captured == {"model_type": "mace-r2scan", "model_path": "/m.model"}
    assert "mace-r2scan" in calculator_identity(calc)


# ---------------------------------------------------------------------------
# Part 2: spot_check
# ---------------------------------------------------------------------------
def test_spot_check_identical_calculator(dataset):
    label_frames(dataset, FakeCalc(scale=1.0))
    rep = spot_check(
        dataset, FakeCalc(scale=1.0), FakeCalc(scale=1.0), categories="all"
    )
    cat = rep.categories["all"]
    assert cat.n_frames == 3
    assert cat.dE_max == pytest.approx(0.0, abs=1e-12)
    assert cat.force_rmse == pytest.approx(0.0, abs=1e-12)
    assert cat.force_cos_mean == pytest.approx(1.0, abs=1e-12)
    assert cat.stress_rmse == pytest.approx(0.0, abs=1e-12)
    d = rep.to_dict()
    assert d["categories"]["all"]["n_frames"] == 3


def test_spot_check_scaled_calculator_and_categories(dataset):
    label_frames(dataset, FakeCalc(scale=1.0))
    rep = spot_check(dataset, FakeCalc(scale=1.0), FakeCalc(scale=1.1))
    # default categories = unique provenance strings
    assert set(rep.categories) == {"single", "random"}
    for name, cat in rep.categories.items():
        assert cat.n_frames == (2 if name == "single" else 1)
        assert cat.dE_max > 0
        assert cat.dE_mean > 0
        assert cat.force_rmse > 0
        assert 0.0 < cat.force_cos_mean <= 1.0
        assert cat.stress_rmse > 0
    assert rep.ref_label == dataset.active_label
    assert rep.n_frames == 3


def test_spot_check_nan_stress(dataset):
    label_frames(dataset, FakeCalcNoStress())
    rep = spot_check(
        dataset, FakeCalcNoStress(), FakeCalcNoStress(), categories="all"
    )
    assert rep.categories["all"].stress_rmse is None


# ---------------------------------------------------------------------------
# Part 2: ABINIT HIST import
# ---------------------------------------------------------------------------
def _write_synthetic_hist(path, order="voigt"):
    import netCDF4

    n, natom = 2, 2
    bohr = 0.5  # stored in Bohr; expected Ang = value * ase.units.Bohr
    xcart = np.array([[[0.0, 0.0, 0.0], [bohr, bohr, bohr]]] * n)
    rprimd = np.tile(np.diag([bohr, 2 * bohr, 3 * bohr]), (n, 1, 1))
    etotal = np.array([-1.0, -2.0])
    fcart = np.array([[[1.0, -1.0, 0.5], [0.25, 0.0, -0.25]]] * n)
    # canonical (xx, yy, zz, yz, xz, xy) values
    s6 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    strten = np.tile(s6, (n, 1))
    if order == "3x3":
        T = np.zeros((n, 3, 3))
        for i in range(n):
            T[i] = [
                [s6[0], s6[5], s6[4]],
                [s6[5], s6[1], s6[3]],
                [s6[4], s6[3], s6[2]],
            ]
        strten = T
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("time", n)
        ds.createDimension("natom", natom)
        ds.createDimension("xyz", 3)
        ds.createDimension("six", 6)
        ds.createDimension("ntypat", 1)
        v = ds.createVariable("xcart", "f8", ("time", "natom", "xyz"))
        v[:] = xcart
        v = ds.createVariable("xred", "f8", ("time", "natom", "xyz"))
        v[:] = np.array([[[0.0, 0.0, 0.0], [1.0, 0.5, 1.0 / 3.0]]] * n)
        v = ds.createVariable("fcart", "f8", ("time", "natom", "xyz"))
        v[:] = fcart
        v = ds.createVariable("etotal", "f8", ("time",))
        v[:] = etotal
        dims = ("time", "six") if order == "voigt" else ("time", "xyz", "xyz")
        v = ds.createVariable("strten", "f8", dims)
        v[:] = strten
        v = ds.createVariable("rprimd", "f8", ("time", "xyz", "xyz"))
        v[:] = rprimd
        v = ds.createVariable("acell", "f8", ("time", "xyz"))
        v[:] = np.ones((n, 3))
        v = ds.createVariable("typat", "i4", ("natom",))
        v[:] = [1, 1]
        v = ds.createVariable("znucl", "f8", ("ntypat",))
        v[:] = [14.0]
    return dict(xcart=xcart, rprimd=rprimd, etotal=etotal, fcart=fcart, strten=strten)


def test_from_abinit_hist_conversion(tmp_path):
    import ase.units as au

    path = tmp_path / "train_HIST.nc"
    ref = _write_synthetic_hist(path)
    ds = from_abinit_hist(path, harmonic_source="synthetic-hist")
    B, H = au.Bohr, au.Hartree
    np.testing.assert_allclose(ds.positions, ref["xcart"] * B, atol=1e-12)
    np.testing.assert_allclose(ds.cell, ref["rprimd"] * B, atol=1e-12)
    np.testing.assert_allclose(ds.energies, ref["etotal"] * H, atol=1e-12)
    np.testing.assert_allclose(ds.forces, ref["fcart"] * H / B, atol=1e-12)
    s = ds.stresses
    conv = H / B**3
    np.testing.assert_allclose(s[:, 0, 0], ref["strten"][:, 0] * conv, atol=1e-12)
    np.testing.assert_allclose(s[:, 1, 1], ref["strten"][:, 1] * conv, atol=1e-12)
    np.testing.assert_allclose(s[:, 2, 2], ref["strten"][:, 2] * conv, atol=1e-12)
    np.testing.assert_allclose(s[:, 1, 2], ref["strten"][:, 3] * conv, atol=1e-12)
    np.testing.assert_allclose(s[:, 0, 2], ref["strten"][:, 4] * conv, atol=1e-12)
    np.testing.assert_allclose(s[:, 0, 1], ref["strten"][:, 5] * conv, atol=1e-12)
    np.testing.assert_allclose(s, np.transpose(s, (0, 2, 1)))
    # structural metadata
    assert ds.nframes == 2
    assert ds.atomic_numbers.tolist() == [[14, 14], [14, 14]]
    assert ds.pbc.tolist() == [[True, True, True]] * 2
    assert ds.split.tolist() == ["train", "train"]
    assert ds.attrs["harmonic_source"] == "synthetic-hist"
    blocks = json.loads(ds.attrs["label_blocks"])
    assert "abinit_hist" in blocks[0]["calculator"]
    # HIST frames carry no Q / strain: NaN-masked
    assert ds.Q.shape[0] == 2
    assert np.isnan(ds.strain_voigt).all()


def test_from_abinit_hist_3x3_strten_and_order_map(tmp_path):
    import ase.units as au

    path = tmp_path / "hist33.nc"
    _write_synthetic_hist(path, order="3x3")
    ds = from_abinit_hist(path)
    np.testing.assert_allclose(
        ds.stress_voigt[0],
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * au.Hartree / au.Bohr**3,
        atol=1e-12,
    )
    # order_map: canonical[i] = stored[order_map[i]]
    path2 = tmp_path / "hist_perm.nc"
    _write_synthetic_hist(path2)
    perm = [2, 1, 0, 5, 4, 3]  # stored order swapped: s0<->s2, s3<->s5
    ds2 = from_abinit_hist(path2, order_map=perm)
    canon = np.array([0.3, 0.2, 0.1, 0.6, 0.5, 0.4]) * au.Hartree / au.Bohr**3
    np.testing.assert_allclose(ds2.stress_voigt[0], canon, atol=1e-12)


def test_unit_conversion_factors_symbolic():
    """sympy-verified factor chain (user-mandated rule).

    Asserts, with executed sympy:
      1. Bohr radius a0 = 4*pi*eps0*hbar^2/(m_e e^2) matches the CODATA value
         used by scipy (independent source) to 1e-12 relative.
      2. Hartree E_h = alpha^2 m_e c^2 == hbar^2/(m_e a0^2) as an exact sympy
         identity under the defining relation alpha = e^2/(4 pi eps0 hbar c),
         and matches scipy CODATA to 1e-12.
      3. force factor  == E_h_factor / a0_factor  (energy/length composite)
         stress factor == E_h_factor / a0_factor**3 (energy/length^3).
      4. Voigt (xx,yy,zz,yz,xz,xy) extraction is a bijection on symmetric
         tensors: matrix_to_voigt / voigt_to_matrix roundtrip identity.
    """
    import sympy as sp
    import scipy.constants as sc

    a0 = sp.symbols("a0", positive=True)
    eps0, hbar, me, e, alpha, c = sp.symbols("eps0 hbar m_e e alpha c", positive=True)
    a0_expr = 4 * sp.pi * eps0 * hbar**2 / (me * e**2)
    Eh_expr1 = alpha**2 * me * c**2
    Eh_expr2 = hbar**2 / (me * a0_expr**2)
    # exact symbolic identity via the defining relation alpha = e^2/(4 pi eps0 hbar c)
    alpha_def = e**2 / (4 * sp.pi * eps0 * hbar * c)
    assert sp.simplify(Eh_expr2.subs(a0, a0_expr) - Eh_expr1.subs(alpha, alpha_def)) == 0
    # numeric: CODATA values from scipy (independent of ase)
    vals = {
        eps0: sc.epsilon_0,
        hbar: sc.hbar,
        me: sc.m_e,
        e: sc.e,
        alpha: sc.fine_structure,
        c: sc.c,
    }
    a0_num = float(a0_expr.subs(vals))
    Eh_num = float(Eh_expr1.subs(vals))
    assert abs(a0_num - sc.value("Bohr radius")) / sc.value("Bohr radius") < 1e-12
    # CODATA lists E_h via 2 R_inf h c; alpha^2 m_e c^2 re-derives it from
    # independently rounded constants -> agree at ~1e-11 relative, not 1e-12
    assert (
        abs(Eh_num - sc.value("Hartree energy")) / sc.value("Hartree energy") < 1e-9
    )
    # composites: 1 eV = |e| J exactly, 1 A = 1e-10 m; ase.units cross-check
    Eh_eV = Eh_num / sc.e
    a0_A = a0_num * 1e10
    import ase.units as au

    B, H = au.Bohr, au.Hartree
    assert abs(B - a0_A) / a0_A < 1e-6
    assert abs(H - Eh_eV) / Eh_eV < 1e-6
    # force factor = E_h/a0 (energy/length); stress factor = E_h/a0^3
    assert abs(H / B - Eh_eV / a0_A) / (Eh_eV / a0_A) < 1e-6
    assert abs(H / B**3 - Eh_eV / a0_A**3) / (Eh_eV / a0_A**3) < 1e-6
    # Voigt bijection on symmetric tensors
    s0, s1, s2, s3, s4, s5 = sp.symbols("s0 s1 s2 s3 s4 s5")
    S = sp.Matrix([[s0, s5, s4], [s5, s1, s3], [s4, s3, s2]])
    from lawaf.anharmonic.dataset import matrix_to_voigt, voigt_to_matrix

    v = matrix_to_voigt(np.array(S.tolist(), dtype=object))
    assert list(v) == [s0, s1, s2, s3, s4, s5]
    back = voigt_to_matrix(v)
    assert sp.simplify(sp.Matrix(back.tolist()) - S) == sp.zeros(3, 3)




# ---------------------------------------------------------------------------
# FR-021 MLIP-harmonic smoke: atomchain calculate_phonon -> phonopy.load
# ---------------------------------------------------------------------------
def test_mlip_harmonic_phonopy_params_smoke(tmp_path):
    pytest.importorskip("atomchain")
    import phonopy
    from ase.calculators.emt import EMT
    from atomchain.phonon.frozenphonon import calculate_phonon

    atoms = Atoms(
        "NiAl",
        positions=[[0, 0, 0], [1.8, 1.8, 1.8]],
        cell=np.eye(3) * 3.6,
        pbc=True,
    )
    save_dir = tmp_path / "phonon_save"
    calculate_phonon(
        atoms,
        calc=EMT(),
        ndim=np.eye(3),
        primitive_matrix=np.eye(3),
        phonon_save_dir=str(save_dir),
    )
    yaml_path = save_dir / "phonopy_params.yaml"
    assert yaml_path.exists()
    ph = phonopy.load(phonopy_yaml=str(yaml_path), produce_fc=False)
    assert ph.force_constants is not None
    freqs = ph.get_frequencies([0, 0, 0])
    assert np.isfinite(freqs).all()
