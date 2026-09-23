"""Per-q irrep content of the phonon displacement representation (R1).

Research memo 2026-09-22-symmetry-projected-pwf (stage R1): at each
supplied q, decompose the 3N-dimensional displacement representation of
the little group into labeled irreps (lawaf's own character machinery,
no third-party irrep engine) and report the JSON-serializable field.

This is an *analysis* layer for window/subspace design (ADR-012-style
legality, mesh-global): it does not construct Wannier functions and does
not change the projected-Wannier path.

NOTE on compatibility: restricting the Gamma displacement representation
to another point's little group is not an endpoint compatibility
relation -- the fibers carry different q-dependent Bloch phases (on
BaTiO3 the Gamma restriction to X's group predicts u-only content while
the actual X small-rep content contains g irreps; nonsymmorphic groups
add small-rep factor systems on top). Proper endpoint compatibility
restricts both endpoint representations to the common little group of a
connecting line (CORREL/CDML machinery) and is deliberately deferred
(research memo open question, stage R3).

All public objects are importable without optional dependencies
(``lawaf.anharmonic.irrepfield``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .compatibility import character_table, little_group

__all__ = [
    "IrrepBlock",
    "IrrepFieldPoint",
    "IrrepField",
    "evaluate_irrep_field",
]


@dataclass(frozen=True)
class IrrepBlock:
    """One irrep of the little group with its multiplicity.

    ``dim * multiplicity`` modes of the displacement representation at
    this q transform under ``name``.
    """

    name: str
    dim: int
    multiplicity: int


@dataclass(frozen=True)
class IrrepFieldPoint:
    """Displacement-representation irrep content at one q point."""

    qpoint: Tuple[float, float, float]
    little_group_order: int
    blocks: Tuple[IrrepBlock, ...]
    n_modes: int

    def mult_of(self, name: str) -> int:
        return sum(b.multiplicity for b in self.blocks if b.name == name)

    def to_dict(self) -> Dict:
        return {
            "qpoint": list(self.qpoint),
            "little_group_order": self.little_group_order,
            "blocks": [
                {
                    "name": b.name,
                    "dim": b.dim,
                    "multiplicity": b.multiplicity,
                }
                for b in self.blocks
            ],
            "n_modes": self.n_modes,
        }


@dataclass
class IrrepField:
    """Labeled displacement-representation content over a q set.

    Built by :func:`evaluate_irrep_field`. The decomposition is of the
    FULL 3N displacement representation (always a band representation),
    so the field is a complete, consistent labeling -- the reference
    against which selected subspaces (windows, projectors) are compared.
    """

    points: List[IrrepFieldPoint] = field(default_factory=list)
    spacegroup_number: int = 0

    def point_at(self, qpoint, tol: float = 1e-6) -> IrrepFieldPoint:
        q = np.asarray(qpoint, dtype=float).reshape(3) % 1.0
        for p in self.points:
            # minimum-image (periodic) difference
            d = ((np.asarray(p.qpoint) - q + 0.5) % 1.0) - 0.5
            if np.all(np.abs(d) <= tol):
                return p
        raise KeyError(
            f"q {tuple(np.round(q, 6))} not in this IrrepField "
            f"({[tuple(p.qpoint) for p in self.points]})"
        )

    def to_dict(self) -> Dict:
        return {
            "spacegroup_number": self.spacegroup_number,
            "points": [p.to_dict() for p in self.points],
        }


def evaluate_irrep_field(
    sga,
    qpoints: Sequence,
    tol: float = 1e-8,
) -> IrrepField:
    """Decompose the displacement representation at each q.

    Parameters
    ----------
    sga :
        Space-group action from
        :func:`lawaf.anharmonic.build_space_group_action`.
    qpoints :
        Sequence of fractional q vectors (any representative of each
        star suffices; the decomposition is per supplied point).
    tol :
        Little-group membership tolerance (see
        :func:`lawaf.anharmonic.compatibility.little_group`).

    Returns
    -------
    IrrepField
        One :class:`IrrepFieldPoint` per input q with irrep blocks
        (name, dim, multiplicity); ``sum(dim * multiplicity) == 3N``.
    """
    out = IrrepField()
    out.spacegroup_number = int(sga.symmetry_dataset.number)
    for q in qpoints:
        q = np.asarray(q, dtype=float).reshape(3)
        lg = little_group(sga, q, tol=tol)
        ct = character_table(sga, lg)
        chi = {
            int(g): complex(np.trace(np.asarray(sga.matrix(g, q), dtype=complex)))
            for g in lg
        }
        mult = ct.multiplicities(chi)
        blocks = []
        n_modes = 0
        for name, d, m in zip(ct.names, ct.dims, mult):
            mi = int(round(m.real))
            if abs(m.real - mi) > 1e-4:
                raise ValueError(
                    f"non-integer multiplicity {m.real:.6f} for {name} "
                    f"at q={tuple(np.round(q, 6))}"
                )
            if mi > 0:
                blocks.append(IrrepBlock(name=name, dim=int(d), multiplicity=mi))
                n_modes += int(d) * mi
        out.points.append(
            IrrepFieldPoint(
                qpoint=tuple(float(x % 1.0) for x in q),
                little_group_order=len(lg),
                blocks=tuple(blocks),
                n_modes=n_modes,
            )
        )
    return out
