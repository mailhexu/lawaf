"""Teacher stack: calculator labeling, atomchain adapter, DFT spot checks.

Conventions (see ``lawaf/anharmonic/sampling.py`` / ``dataset.py``):
ASE forces (``dE/du = -F``), stress stored as the full symmetric 3x3 tensor
in eV/Angstrom^3, Voigt order ``(xx, yy, zz, yz, xz, xy)``, NaN where a
quantity is unavailable.

ABINIT HIST.nc import
    Documented ABINIT layout (verified against a real BaTiO3 MD HIST under
    atomchain example 08): variables ``rprimd (n, 3, 3)`` (lattice vectors as
    ROWS, Bohr), ``xcart (n, natom, 3)`` (Bohr), ``etotal (n,)`` (Ha),
    ``fcart (n, natom, 3)`` (Ha/Bohr), ``strten (n, 6)`` (Ha/Bohr^3, Voigt
    xx,yy,zz,yz,xz,xy — also accepts a (n, 3, 3) tensor), ``znucl (ntypat,)``
    + ``typat (natom,)`` (1-based).  ``vel``/``ekin`` are ignored (warned).
    Unit factors are ``ase.units.Bohr`` / ``ase.units.Hartree``; the factor
    chain is sympy-verified in ``docs/derivations/anharmonic_unit_
    conversions.py`` and asserted in ``tests/test_anharmonic_dataset.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .dataset import TrainingDataset, calculator_identity, matrix_to_voigt

__all__ = [
    "label_frames",
    "get_atomchain_calculator",
    "spot_check",
    "SpotCheckReport",
    "SpotCheckCategory",
    "from_abinit_hist",
    "ABINIT_VOIGT_ORDER",
]

#: ABINIT strten Voigt ordering (same as the lawaf convention).
ABINIT_VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")


# ---------------------------------------------------------------------------
# labeling
# ---------------------------------------------------------------------------
def _evaluate_atoms(atoms, calculator):
    """Fresh (E, F, S3x3) of ``atoms`` under ``calculator``.

    Works on a copy; the input atoms (and any pre-existing calculator) are
    left untouched.  A missing stress becomes NaN (NaN-masking contract);
    an energy/forces failure raises.
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    try:
        stress = _voigt_to_matrix(
            np.asarray(atoms.get_stress(voigt=True), dtype=float).reshape(6)
        )
    except Exception:
        # e.g. PropertyNotImplementedError: NaN-masked by contract
        stress = None
    return energy, forces, stress


def _voigt_to_matrix(v):
    m = np.zeros((3, 3))
    m[0, 0], m[1, 1], m[2, 2] = v[0], v[1], v[2]
    m[1, 2] = m[2, 1] = v[3]
    m[0, 2] = m[2, 0] = v[4]
    m[0, 1] = m[1, 0] = v[5]
    return m


def label_frames(
    dataset: TrainingDataset,
    calculator,
    frames: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
) -> str:
    """Label frames of ``dataset`` with ``calculator`` into a NEW label block.

    Structures are never mutated: each frame is evaluated on a copy.  Frames
    are processed (and flushed to the block) in chunks of ``batch_size``
    (ASE calculators are single-frame; the chunking bounds write granularity
    for crash-resume, not parallelism).  Returns the label block id, which
    becomes the active block.
    """
    bid = dataset.label(calculator, frames=frames)
    idx = (
        np.arange(dataset.nframes)
        if frames is None
        else np.atleast_1d(np.asarray(frames, dtype=int))
    )
    step = len(idx) if batch_size is None else int(batch_size)
    for start in range(0, len(idx), step):
        chunk = idx[start : start + step]
        energies, forces, stresses = [], [], []
        for i in chunk:
            e, f, s = _evaluate_atoms(dataset.frame_atoms(i), calculator)
            energies.append(e)
            forces.append(f)
            stresses.append(
                np.full((3, 3), np.nan) if s is None else s
            )
        dataset.fill_label_values(
            bid,
            chunk,
            energy=np.asarray(energies),
            forces=np.asarray(forces),
            stress=np.asarray(stresses),
        )
    return bid


def get_atomchain_calculator(name: str = "mace-r2scan", model_path: Optional[str] = None):
    """ASE calculator from atomchain (lazy import; never imported at module load).

    Raises RuntimeError pointing at the ``lawaf[anharmonic]`` extra when
    atomchain is not installed.
    """
    try:
        from atomchain.init_model import init_calc
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "atomchain is required for MLIP teachers; install it with "
            "`pip install 'lawaf[anharmonic]'`"
        ) from exc
    calc = init_calc(model_type=name, model_path=model_path)
    # identity hints for calculator_identity (best-effort, see dataset.py)
    try:
        calc.lawaf_model_name = name
    except Exception:
        pass
    if model_path is not None:
        try:
            calc.lawaf_model_path = model_path
        except Exception:
            pass
    return calc


# ---------------------------------------------------------------------------
# spot checks
# ---------------------------------------------------------------------------
@dataclass
class SpotCheckCategory:
    """NaN-aware per-category agreement stats (test labels vs reference)."""

    n_frames: int
    dE_max: float
    dE_mean: float
    force_rmse: float
    force_cos_mean: float
    force_cos_min: float
    stress_rmse: Optional[float]


@dataclass
class SpotCheckReport:
    ref_label: str
    test_calculator: str
    n_frames: int
    categories: Dict[str, SpotCheckCategory] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ref_label": self.ref_label,
            "test_calculator": self.test_calculator,
            "n_frames": self.n_frames,
            "categories": {k: asdict(v) for k, v in self.categories.items()},
        }


def _force_cosine(f_ref: np.ndarray, f_test: np.ndarray) -> float:
    a, b = f_ref.ravel(), f_test.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 and nb < 1e-30:
        return 1.0
    if na < 1e-30 or nb < 1e-30:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def spot_check(
    dataset: TrainingDataset,
    calculator_ref=None,
    calculator_test=None,
    categories: Union[None, str, Sequence[str]] = None,
) -> SpotCheckReport:
    """Compare stored labels against fresh labels from ``calculator_test``.

    ``calculator_ref`` selects the reference block by calculator identity
    (calculator object or identity string); ``None`` uses the ACTIVE block.
    Fresh labels are computed in memory on the same frames (no second
    dataset, no mutation).  ``categories`` defaults to the unique per-frame
    provenance strings; ``"all"`` collapses everything into one category.
    Frames without a finite reference energy are skipped.
    """
    if calculator_ref is None:
        block = dataset._active()
    else:
        ident = calculator_identity(calculator_ref)
        matches = [b for b in dataset.blocks.values() if b.calculator == ident]
        if not matches:
            raise KeyError(f"no label block for calculator identity {ident!r}")
        block = matches[0]
    e_ref, f_ref, s_ref = block.energy, block.forces, block.stress
    idx = np.flatnonzero(np.isfinite(e_ref))
    if len(idx) == 0:
        raise ValueError(f"reference label block {block.id!r} has no labeled frames")

    # fresh labels from the test calculator (in memory only)
    e_new = np.full(dataset.nframes, np.nan)
    f_new = np.full((dataset.nframes, dataset.natoms, 3), np.nan)
    s_new = np.full((dataset.nframes, 3, 3), np.nan)
    for i in idx:
        e, f, s = _evaluate_atoms(dataset.frame_atoms(i), calculator_test)
        e_new[i], f_new[i] = e, f
        if s is not None:
            s_new[i] = s

    if categories is None:
        cat_names = []
        for p in dataset.provenance[idx]:
            p = str(p)
            if p not in cat_names:
                cat_names.append(p)
    elif categories == "all":
        cat_names = ["all"]
    else:
        cat_names = list(categories)

    out: Dict[str, SpotCheckCategory] = {}
    for name in cat_names:
        if name == "all":
            sel = idx
        else:
            sel = idx[dataset.provenance[idx] == name]
        if len(sel) == 0:
            continue
        dE = e_new[sel] - e_ref[sel]
        df = (f_new[sel] - f_ref[sel]).ravel()
        ds = (s_new[sel] - s_ref[sel]).ravel()
        cos = np.array(
            [_force_cosine(f_ref[i], f_new[i]) for i in sel], dtype=float
        )
        finite_cos = cos[np.isfinite(cos)]
        out[name] = SpotCheckCategory(
            n_frames=int(len(sel)),
            dE_max=float(np.max(np.abs(dE))),
            dE_mean=float(np.mean(np.abs(dE))),
            force_rmse=float(np.sqrt(np.mean(df**2))),
            force_cos_mean=float(np.mean(finite_cos)) if finite_cos.size else float("nan"),
            force_cos_min=float(np.min(finite_cos)) if finite_cos.size else float("nan"),
            stress_rmse=(
                float(np.sqrt(np.mean(ds[np.isfinite(ds)] ** 2)))
                if np.isfinite(ds).any()
                else None
            ),
        )
    return SpotCheckReport(
        ref_label=block.id,
        test_calculator=calculator_identity(calculator_test),
        n_frames=int(len(idx)),
        categories=out,
    )


# ---------------------------------------------------------------------------
# ABINIT HIST.nc import
# ---------------------------------------------------------------------------
def _abinit_unit_factors():
    """(Bohr->Ang, Ha->eV) from ase.units; sympy-verified factor chain.

    The chain (a0 = 4 pi eps0 hbar^2/(m_e e^2), E_h = alpha^2 m_e c^2 =
    hbar^2/(m_e a0^2), composites force = E_h/a0, stress = E_h/a0^3) is
    derived and asserted in docs/derivations/anharmonic_unit_conversions.py
    and in test_unit_conversion_factors_symbolic.  ase.units values are
    cross-checked against scipy CODATA there to 1e-6 relative.
    """
    import ase.units as au

    return au.Bohr, au.Hartree


def from_abinit_hist(
    path,
    order_map: Optional[Sequence[int]] = None,
    harmonic_source: str = "",
    split: str = "train",
) -> TrainingDataset:
    """Build a TrainingDataset from an ABINIT HIST.nc trajectory/relax file.

    ``order_map`` maps stored Voigt indices to the canonical
    (xx, yy, zz, yz, xz, xy) order: ``canonical[i] = stored[order_map[i]]``;
    ``None`` means the file already uses the ABINIT/lawaf order (verified
    ordering above).  Fields that cannot be verified (``vel``, ``ekin``) are
    ignored with a warning; a (n, 3, 3) strten with a non-negligible
    antisymmetric part is symmetrized with a warning.
    """
    import netCDF4

    B, H = _abinit_unit_factors()
    with netCDF4.Dataset(str(path), "r") as ds:
        rprimd = np.array(ds["rprimd"][:], dtype=float)  # (n, 3, 3) rows, Bohr
        xcart = np.array(ds["xcart"][:], dtype=float)  # (n, natom, 3), Bohr
        etotal = np.array(ds["etotal"][:], dtype=float)  # (n,), Ha
        fcart = np.array(ds["fcart"][:], dtype=float)  # (n, natom, 3), Ha/Bohr
        strten = np.array(ds["strten"][:], dtype=float)  # (n, 6) or (n, 3, 3), Ha/Bohr^2
        typat = np.array(ds["typat"][:], dtype=int)  # (natom,), 1-based
        znucl = np.array(ds["znucl"][:], dtype=float)  # (ntypat,)
        ignored = [k for k in ("vel", "ekin", "entropy", "mdtime") if k in ds.variables]
    if ignored:
        warnings.warn(
            f"from_abinit_hist: ignoring unverified HIST fields {ignored}"
        )
    n = rprimd.shape[0]
    numbers = np.array([int(round(znucl[t - 1])) for t in typat], dtype=np.int64)

    if strten.shape == (n, 3, 3):
        anti = np.abs(strten - np.transpose(strten, (0, 2, 1)))
        scale = max(np.abs(strten).max(), 1e-30)
        if anti.max() / scale > 1e-8:
            warnings.warn(
                "from_abinit_hist: strten stored as 3x3 with antisymmetric "
                "component; symmetrizing"
            )
        strten = 0.5 * (strten + np.transpose(strten, (0, 2, 1)))
        voigt = matrix_to_voigt(strten)
    elif strten.shape == (n, 6):
        voigt = strten
    else:
        raise ValueError(f"unexpected strten shape {strten.shape}")

    if order_map is not None:
        perm = np.asarray(order_map, dtype=int)
        if sorted(perm.tolist()) != [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"order_map must be a permutation of 0..5, got {order_map}")
        voigt = voigt[:, perm]
    stress = np.stack([_voigt_to_matrix(v) for v in voigt])  # (n, 3, 3)

    ds_out = TrainingDataset(
        positions=xcart * B,  # Bohr -> Angstrom
        atomic_numbers=np.tile(numbers, (n, 1)),
        cell=rprimd * B,  # rows are lattice vectors, Bohr -> Angstrom
        pbc=np.ones((n, 3), dtype=bool),
        Q=np.full((n, 0), np.nan),  # HIST frames carry no lwf amplitudes
        strain_voigt=np.full((n, 6), np.nan),  # MD/relax frames: no strain param
        split=[split] * n,
        provenance=[f"abinit_hist:{_basename(path)}#{i}" for i in range(n)],
        attrs={"harmonic_source": harmonic_source},
    )
    ds_out.label(
        "abinit_hist",
        values={
            "energy": etotal * H,  # Ha -> eV
            "forces": fcart * H / B,  # Ha/Bohr -> eV/Angstrom
            "stress": stress * H / B**3,  # Ha/Bohr^3 -> eV/Angstrom^3
        },
    )
    return ds_out


def _basename(path) -> str:
    import os

    return os.path.basename(str(path))
