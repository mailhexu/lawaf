"""TrainingDataset: frame-stacked training structures, labels, and netCDF I/O.

Storage layout (xarray, netCDF group ``"dataset"``, one file per dataset --
same grouped convention as ``lawaf/interfaces/phonopy/lwf.py``):

Data variables
    positions       (frame, atom, xyz) float64   Angstrom
    atomic_numbers  (frame, atom)      int64
    cell            (frame, abc, xyz)  float64   rows are lattice vectors, Angstrom
    pbc             (frame, xyz)       int8      (netCDF has no bool; cast back)
    Q               (frame, q)         float64   lwf mode amplitudes, NaN when
                                                   unknown/unused (see sampling.py
                                                   for the c = icell*nlwf + iwann
                                                   ordering)
    strain_voigt    (frame, voigt)     float64   (xx,yy,zz,yz,xz,xy); NaN row for
                                                   unstrained frames
    split           (frame)            str       "train" / "cv" / "holdout"
    provenance      (frame)            str       per-frame lineage tag
    energy__<bid>   (frame)            float64   per label block, NaN where
    forces__<bid>   (frame, atom, xyz) float64   the block has no data for
    stress__<bid>   (frame, xyz, xyz)  float64   that frame (full symmetric
                                                 tensor, eV/Angstrom^3)

Dataset attributes
    units           JSON: {"energy","length","force","stress","stress_voigt_order"}
    harmonic_source FR-021 provenance string of the harmonic reference
    lawaf_version   version of lawaf that created the file
    created         ISO-8601 UTC timestamp
    label_blocks    JSON list: [{"id","calculator","frames","created"}, ...]
    active_label    id of the active label block

Label blocks
    One block per distinct calculator identity (see :func:`calculator_identity`).
    ``TrainingDataset.label`` is idempotent per identity: relabeling with the
    SAME identity is a no-op (with a warning when new values are supplied
    without ``overwrite=True``); a DIFFERENT identity appends a new block and
    makes it active.  Active-label accessors ``.energies/.forces/.stresses``
    return plain arrays with NaN wherever the block has no label.

Conventions are the ones documented in ``lawaf/anharmonic/sampling.py``:
forces follow the ASE sign (``dE/du = -F``), Voigt order is
``(xx, yy, zz, yz, xz, xy)``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "TrainingDataset",
    "LabelBlock",
    "calculator_identity",
    "voigt_to_matrix",
    "matrix_to_voigt",
    "VOIGT_ORDER",
    "SPLIT_CODES",
]

VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
SPLIT_CODES = {0: "train", 1: "cv", 2: "holdout"}

UNITS = {
    "energy": "eV",
    "length": "Angstrom",
    "force": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
    "stress_voigt_order": list(VOIGT_ORDER),
}


def voigt_to_matrix(v) -> np.ndarray:
    """Voigt (xx, yy, zz, yz, xz, xy) -> symmetric 3x3.

    Same ordering as :func:`lawaf.anharmonic.sampling.voigt_to_matrix` (the
    strain path), but dtype-agnostic so the mapping itself is sympy-verifiable
    (works on sympy symbols via object arrays).
    """
    v = np.asarray(v)
    if v.shape[-1:] != (6,):
        raise ValueError(f"expected (..., 6), got {v.shape}")
    z = np.zeros(v.shape[:-1] + (3, 3), dtype=object if v.dtype == object else v.dtype)
    z[..., 0, 0] = v[..., 0]
    z[..., 1, 1] = v[..., 1]
    z[..., 2, 2] = v[..., 2]
    z[..., 1, 2] = z[..., 2, 1] = v[..., 3]
    z[..., 0, 2] = z[..., 2, 0] = v[..., 4]
    z[..., 0, 1] = z[..., 1, 0] = v[..., 5]
    return z


def matrix_to_voigt(mat) -> np.ndarray:
    """Symmetric 3x3 -> Voigt (xx, yy, zz, yz, xz, xy)."""
    m = np.asarray(mat)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {m.shape}")
    return np.stack(
        [
            m[..., 0, 0],
            m[..., 1, 1],
            m[..., 2, 2],
            m[..., 1, 2],
            m[..., 0, 2],
            m[..., 0, 1],
        ],
        axis=-1,
    )


def calculator_identity(calculator) -> str:
    """Best-effort identity string for a calculator (or its name).

    Best-effort by design: calculators do not expose a portable fingerprint.
    Resolution order:
      1. plain string -> returned unchanged;
      2. ``lawaf_model_name`` hint (set by
         :func:`lawaf.anharmonic.teacher.get_atomchain_calculator`) plus
         ``model_path`` when present;
      3. class name plus a sha1 of the sorted JSON-serializable parameters
         (``calculator.todict()``) when that dict is small (< 64 entries,
         repr < 2048 chars).
    Two objects with the same identity are assumed to label identically;
    nothing enforces this.
    """
    if isinstance(calculator, str):
        return calculator
    name = getattr(calculator, "lawaf_model_name", None) or type(calculator).__name__
    model_path = getattr(calculator, "model_path", None)
    if model_path is None:
        model_path = getattr(calculator, "lawaf_model_path", None)
    if model_path is not None:
        return f"{name}:{model_path}"
    params = None
    try:
        params = calculator.todict()
    except Exception:
        params = None
    if isinstance(params, dict) and params:
        try:
            flat = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            flat = None
        if flat is not None and len(params) < 64 and len(flat) < 2048:
            return f"{name}:{hashlib.sha1(flat.encode()).hexdigest()[:12]}"
    return name


@dataclass
class LabelBlock:
    """One labeled block: calculator identity + NaN-masked value arrays."""

    id: str
    calculator: str
    frames: Optional[List[int]]
    energy: np.ndarray  # (nf,)
    forces: np.ndarray  # (nf, natom, 3)
    stress: np.ndarray  # (nf, 3, 3)
    created: str = ""

    def subset(self, idx) -> "LabelBlock":
        return LabelBlock(
            id=self.id,
            calculator=self.calculator,
            frames=None if self.frames is None else [self.frames[i] for i in idx],
            energy=self.energy[idx],
            forces=self.forces[idx],
            stress=self.stress[idx],
            created=self.created,
        )


class TrainingDataset:
    """Frame-stacked structures, mode amplitudes, and labeled quantities."""

    def __init__(
        self,
        positions,
        atomic_numbers,
        cell,
        pbc,
        Q,
        strain_voigt,
        split=None,
        provenance=None,
        attrs=None,
    ):
        self.positions = np.asarray(positions, dtype=float)
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.cell = np.asarray(cell, dtype=float)
        self.pbc = np.asarray(pbc, dtype=bool)
        self.Q = np.asarray(Q, dtype=float).reshape(self.nframes, -1)
        self.strain_voigt = np.asarray(strain_voigt, dtype=float).reshape(self.nframes, 6)
        # object dtype: no fixed-width truncation when writing longer codes later
        self.split = np.asarray(
            split if split is not None else ["train"] * self.nframes, dtype=object
        )
        self.provenance = np.asarray(
            provenance if provenance is not None else [""] * self.nframes, dtype=object
        )
        self.blocks: Dict[str, LabelBlock] = {}
        self.active_label: Optional[str] = None
        base = {
            "units": dict(UNITS),
            "harmonic_source": "",
            "lawaf_version": _lawaf_version(),
            "created": datetime.now(timezone.utc).isoformat(),
        }
        base.update(attrs or {})
        self.attrs = base
        self._sync_label_attrs()

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_frames(cls, frames, atoms_list, harmonic_source: str = "", **attrs):
        """Build from a list of FrameSpec and matching ase.Atoms.

        Q vectors are NaN-padded to a common width; an all-NaN strain_voigt
        row marks an unstrained frame.
        """
        frames = list(frames)
        atoms_list = list(atoms_list)
        if len(frames) != len(atoms_list):
            raise ValueError(
                f"frames ({len(frames)}) and atoms_list ({len(atoms_list)}) differ"
            )
        if not frames:
            raise ValueError("empty training set")
        natom = len(atoms_list[0])
        numbers = np.asarray(atoms_list[0].numbers)
        for i, at in enumerate(atoms_list):
            if len(at) != natom:
                raise ValueError(f"frame {i}: natom differs ({len(at)} != {natom})")
            if not np.array_equal(np.asarray(at.numbers), numbers):
                raise ValueError(f"frame {i}: atomic numbers differ")
        nQ = max(len(np.atleast_1d(f.Q)) for f in frames)
        Q = np.full((len(frames), nQ), np.nan)
        strain = np.full((len(frames), 6), np.nan)
        for i, f in enumerate(frames):
            q = np.atleast_1d(np.asarray(f.Q, dtype=float))
            Q[i, : len(q)] = q
            if f.strain_voigt is not None:
                strain[i] = f.strain_voigt
        return cls(
            positions=[at.get_positions() for at in atoms_list],
            atomic_numbers=[at.numbers for at in atoms_list],
            cell=[at.cell.array for at in atoms_list],
            pbc=[at.pbc for at in atoms_list],
            Q=Q,
            strain_voigt=strain,
            split=[f.split for f in frames],
            provenance=[f.provenance for f in frames],
            attrs={"harmonic_source": harmonic_source, **attrs},
        )

    # -- shapes -------------------------------------------------------------
    @property
    def nframes(self) -> int:
        return self.positions.shape[0]

    @property
    def natoms(self) -> int:
        return self.positions.shape[1]

    @property
    def splits(self) -> List[str]:
        seen = []
        for s in self.split:
            s = str(s)
            if s not in seen:
                seen.append(s)
        return seen

    # -- structure access ---------------------------------------------------
    def frame_atoms(self, i: int):
        """ase.Atoms of frame ``i`` (fresh object, no calculator attached)."""
        from ase import Atoms

        return Atoms(
            numbers=self.atomic_numbers[i],
            positions=self.positions[i],
            cell=self.cell[i],
            pbc=self.pbc[i],
        )

    # -- split management ---------------------------------------------------
    def set_split(self, frames: Union[int, Sequence[int]], split: Union[str, int]):
        """Assign a split code (train / cv / holdout, or 0 / 1 / 2) to frames."""
        idx = np.atleast_1d(np.asarray(frames, dtype=int))
        code = SPLIT_CODES.get(split, split) if isinstance(split, int) else split
        if code not in ("train", "cv", "holdout"):
            raise ValueError(f"split must be train/cv/holdout or 0/1/2, got {split!r}")
        self.split[idx] = code

    def select(
        self,
        frames: Union[None, int, slice, Sequence[int]] = None,
        split: Optional[str] = None,
    ) -> "TrainingDataset":
        """Subset of the dataset (copies of the row-selected arrays/blocks)."""
        if split is not None:
            idx = np.flatnonzero(np.asarray([str(s) for s in self.split]) == split)
        elif frames is None:
            idx = np.arange(self.nframes)
        elif isinstance(frames, slice):
            idx = np.arange(self.nframes)[frames]
        else:
            idx = np.atleast_1d(np.asarray(frames, dtype=int))
        idx = np.asarray(idx, dtype=int)
        ds = TrainingDataset(
            positions=self.positions[idx],
            atomic_numbers=self.atomic_numbers[idx],
            cell=self.cell[idx],
            pbc=self.pbc[idx],
            Q=self.Q[idx],
            strain_voigt=self.strain_voigt[idx],
            split=[str(s) for s in np.asarray(self.split)[idx]],
            provenance=[str(s) for s in np.asarray(self.provenance)[idx]],
            attrs=dict(self.attrs),
        )
        ds.blocks = {k: b.subset(idx) for k, b in self.blocks.items()}
        ds.active_label = self.active_label
        return ds

    def __getitem__(self, key) -> "TrainingDataset":
        """str key selects by split; int/slice/list selects frames."""
        if isinstance(key, str):
            return self.select(split=key)
        return self.select(frames=key)

    # -- labels -------------------------------------------------------------
    def label(
        self,
        calculator,
        frames: Optional[Sequence[int]] = None,
        values: Optional[dict] = None,
        overwrite: bool = False,
    ) -> str:
        """Return the label block id for ``calculator``'s identity.

        Idempotent per identity: an existing block is returned unchanged
        (new ``values`` without ``overwrite=True`` only produce a warning);
        a new identity appends an all-NaN block (or ``values``-filled) and
        makes it active.  ``frames`` restricts which frames the block covers
        (stored full-length with NaN elsewhere); ``values`` keys may be any
        of ``energy`` (k,), ``forces`` (k, natom, 3), ``stress`` (k, 3, 3).
        """
        ident = calculator_identity(calculator)
        for block in self.blocks.values():
            if block.calculator == ident:
                if values is not None:
                    if not overwrite:
                        warnings.warn(
                            f"calculator {ident!r} is already labeled (block "
                            f"{block.id!r}); relabeling with the same identity "
                            "is a no-op (pass overwrite=True to replace values)"
                        )
                    else:
                        idx = (
                            np.arange(self.nframes)
                            if frames is None
                            else np.atleast_1d(np.asarray(frames, int))
                        )
                        self.fill_label_values(block.id, idx, **values)
                return block.id
        nf_all, natom = self.nframes, self.natoms
        idx = (
            np.arange(nf_all)
            if frames is None
            else np.atleast_1d(np.asarray(frames, int))
        )
        bid = f"label_{len(self.blocks):03d}"
        block = LabelBlock(
            id=bid,
            calculator=ident,
            frames=None if frames is None else [int(i) for i in idx],
            energy=np.full(nf_all, np.nan),
            forces=np.full((nf_all, natom, 3), np.nan),
            stress=np.full((nf_all, 3, 3), np.nan),
            created=datetime.now(timezone.utc).isoformat(),
        )
        self.blocks[bid] = block
        self.active_label = bid
        if values is not None:
            self.fill_label_values(bid, idx, **values)
        self._sync_label_attrs()
        return bid

    def fill_label_values(self, bid: str, frames, energy=None, forces=None, stress=None):
        """Write values into label block ``bid`` at the given frames (frames
        not written stay NaN)."""
        block = self.blocks[bid]
        idx = np.atleast_1d(np.asarray(frames, dtype=int))
        if energy is not None:
            block.energy[idx] = np.asarray(energy, dtype=float).reshape(len(idx))
        if forces is not None:
            block.forces[idx] = np.asarray(forces, dtype=float).reshape(len(idx), -1, 3)
        if stress is not None:
            s = np.asarray(stress, dtype=float)
            block.stress[idx] = s.reshape(len(idx), 3, 3)
        self._sync_label_attrs()

    def _sync_label_attrs(self):
        self.attrs["label_blocks"] = json.dumps(
            [
                {
                    "id": b.id,
                    "calculator": b.calculator,
                    "frames": b.frames,
                    "created": b.created,
                }
                for b in self.blocks.values()
            ]
        )
        self.attrs["active_label"] = self.active_label or ""

    def _active(self) -> LabelBlock:
        if self.active_label is None:
            raise ValueError("no active label block; call .label() or label_frames()")
        return self.blocks[self.active_label]

    @property
    def label_blocks(self) -> List[LabelBlock]:
        return list(self.blocks.values())

    @property
    def energies(self) -> np.ndarray:
        """Active-block energies (nf,), NaN where unlabeled."""
        return self._active().energy

    @property
    def forces(self) -> np.ndarray:
        """Active-block forces (nf, natom, 3), NaN where unlabeled."""
        return self._active().forces

    @property
    def stresses(self) -> np.ndarray:
        """Active-block stresses (nf, 3, 3) full symmetric, NaN where absent."""
        return self._active().stress

    @property
    def stress_voigt(self) -> np.ndarray:
        """Active-block stress as (nf, 6) Voigt (xx,yy,zz,yz,xz,xy), NaN kept."""
        return matrix_to_voigt(self._active().stress)

    # -- netCDF I/O ---------------------------------------------------------
    def to_xarray(self):
        import xarray as xr

        ds = xr.Dataset(
            {
                "positions": (("frame", "atom", "xyz"), self.positions),
                "atomic_numbers": (("frame", "atom"), self.atomic_numbers),
                "cell": (("frame", "abc", "xyz"), self.cell),
                "pbc": (("frame", "xyz"), self.pbc.astype(np.int8)),
                "Q": (("frame", "q"), self.Q),
                "strain_voigt": (("frame", "voigt"), self.strain_voigt),
                "split": (("frame",), np.asarray([str(s) for s in self.split])),
                "provenance": (
                    ("frame",),
                    np.asarray([str(s) for s in self.provenance]),
                ),
            }
        )
        for bid, b in self.blocks.items():
            ds[f"energy__{bid}"] = (("frame",), b.energy)
            ds[f"forces__{bid}"] = (("frame", "atom", "xyz"), b.forces)
            ds[f"stress__{bid}"] = (("frame", "xyz", "xyz"), b.stress)
        ds.attrs = {
            "units": json.dumps(self.attrs.get("units", UNITS)),
            "harmonic_source": str(self.attrs.get("harmonic_source", "")),
            "lawaf_version": str(self.attrs.get("lawaf_version", "")),
            "created": str(self.attrs.get("created", "")),
            "label_blocks": json.dumps(
                [
                    {
                        "id": b.id,
                        "calculator": b.calculator,
                        "frames": b.frames,
                        "created": b.created,
                    }
                    for b in self.blocks.values()
                ]
            ),
            "active_label": self.active_label or "",
            "extra_attrs": json.dumps(
                {
                    k: v
                    for k, v in self.attrs.items()
                    if k
                    not in ("units", "label_blocks", "active_label", "extra_attrs")
                }
            ),
        }
        return ds

    def save(self, path):
        """Write to netCDF group ``dataset`` (mode 'w', whole file)."""
        self.to_xarray().to_netcdf(str(path), group="dataset", mode="w")

    @classmethod
    def load(cls, path) -> "TrainingDataset":
        import xarray as xr

        with xr.open_dataset(str(path), group="dataset") as ds:
            ds.load()
            attrs = dict(ds.attrs)
            arrays = {k: v.values for k, v in ds.variables.items()}
        extra = json.loads(attrs.pop("extra_attrs", "{}"))
        self = cls(
            positions=arrays["positions"],
            atomic_numbers=arrays["atomic_numbers"],
            cell=arrays["cell"],
            pbc=arrays["pbc"].astype(bool),
            Q=arrays["Q"],
            strain_voigt=arrays["strain_voigt"],
            split=[str(s) for s in arrays["split"]],
            provenance=[str(s) for s in arrays["provenance"]],
            attrs={
                "units": json.loads(attrs.get("units", "{}")) or dict(UNITS),
                "harmonic_source": attrs.get("harmonic_source", ""),
                "lawaf_version": attrs.get("lawaf_version", ""),
                "created": attrs.get("created", ""),
                **extra,
            },
        )
        self.active_label = attrs.get("active_label") or None
        for entry in json.loads(attrs.get("label_blocks", "[]")):
            bid = entry["id"]
            self.blocks[bid] = LabelBlock(
                id=bid,
                calculator=entry["calculator"],
                frames=entry["frames"],
                energy=arrays[f"energy__{bid}"],
                forces=arrays[f"forces__{bid}"],
                stress=arrays[f"stress__{bid}"],
                created=entry.get("created", ""),
            )
        self._sync_label_attrs()
        return self


def _lawaf_version() -> str:
    try:
        from importlib.metadata import version

        return version("lawaf")
    except Exception:
        return "unknown"
