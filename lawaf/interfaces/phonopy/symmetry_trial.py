"""Wigner-projected symmetry trial orbitals for phonon LWFs (R2).

Research memo 2026-09-22-symmetry-projected-pwf (stage R2): build
deterministic, symmetry-exact trial orbitals at the Gamma point from a
:class:`lawaf.anharmonic.compatibility.RepresentationDeclaration` by
Wigner projection of local Cartesian displacement seeds onto the
isotypic component of each declared site irrep,

.. math::
    \\hat P^{(\\alpha)} = \\frac{d_\\alpha}{|G|}
    \\sum_{g\\in G}\\chi^{(\\alpha)}(g)^{*}\\,D(g),

followed by orthonormalization (SCMPWF-style: symmetry exact by
construction, no localization iterations). The report carries the
projector diagnostics and, against a reference eigenspectrum, the
per-copy overlaps that document the memo's central finding: the
projection selects the irrep SECTOR exactly but never the COPY (on
BaTiO3 the projected Ti seeds overlap the soft triplet at svmin
0.68 only).

Scope: Gamma point (``q=(0,0,0)``); non-Gamma q is rejected with a
clear error (general-q induction needs Bloch-phase machinery that is
deliberately out of R2 scope). This module never changes the
projected-Wannier construction path; the orbitals are consumable via
``ProjectedWannierizer.set_projectors``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from lawaf.anharmonic.compatibility import (
    RepresentationDeclaration,
    little_group,
    resolve_site_irrep,
)
from lawaf.anharmonic.gauge import site_irrep_matrices
from lawaf.anharmonic.representation import build_space_group_action

__all__ = ["TrialOrbitalReport", "symmetry_trial_orbitals"]


@dataclass
class TrialOrbitalReport:
    """Symmetry trial orbitals at Gamma plus diagnostics.

    Attributes
    ----------
    orbitals :
        Orthonormal trial vectors, shape ``(3N, nwann)`` in the lawaf
        displacement row basis (mass-weighted, ``(natom, 3)``
        flattened), ready for ``set_projectors(orbitals.T)``.
    blocks :
        One entry per declared site irrep: name, dimension, number of
        copies the projector selected (``rank``), projector
        idempotency residual, and character deviation of the projected
        span from the site irrep.
    copy_overlaps :
        Per degenerate reference-eigenspace block overlapping the
        projected span: eigenvalue, block size, and the product of the
        overlap singular values (1.0 = the span IS that block).
    soft_overlap_svmin :
        Smallest singular value of the overlap with the lowest
        reference eigenspace block (the soft modes). < 1 documents the
        sector-not-copy finding.
    """

    qpoint: tuple
    wyckoff: str
    site_irreps: List[str]
    orbitals: np.ndarray
    blocks: List[Dict] = field(default_factory=list)
    copy_overlaps: List[Dict] = field(default_factory=list)
    soft_overlap_svmin: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "qpoint": list(self.qpoint),
            "wyckoff": self.wyckoff,
            "site_irreps": list(self.site_irreps),
            "blocks": [dict(b) for b in self.blocks],
            "copy_overlaps": [dict(c) for c in self.copy_overlaps],
            "soft_overlap_svmin": self.soft_overlap_svmin,
        }


def symmetry_trial_orbitals(
    phonon,
    declaration: RepresentationDeclaration,
    seed_atoms: Optional[List[int]] = None,
    symprec: float = 1e-5,
    with_reference: bool = True,
) -> TrialOrbitalReport:
    """Wigner-projected trial orbitals at Gamma for a declaration.

    Parameters
    ----------
    phonon :
        A prepared ``phonopy.Phonopy`` object (force constants set;
        symmetrized or not -- the representation uses its symmetry
        dataset only; reference eigenvectors, if requested, come from a
        cache-free non-NAC solve of its dynamical matrix).
    declaration :
        Declared site representation (wyckoff letter + site_irreps
        with repetition = multiplicity).
    seed_atoms :
        Atom indices whose Cartesian x/y/z unit displacements seed the
        projection. Default: all atoms of the declared Wyckoff orbit.
    with_reference :
        Compute per-copy overlaps against the reference eigenspectrum
        (cheap at Gamma; set False to skip the reference solve).

    Raises
    ------
    ValueError
        For non-Gamma q, an unknown Wyckoff letter in the structure, or
        a declared site irrep that does not exist on the site.
    """
    if isinstance(declaration, dict):
        declaration = RepresentationDeclaration(**declaration)
    q = np.zeros(3)
    if declaration.anchors not in (None, ()):
        anchors = [np.asarray(a, dtype=float).reshape(3) for a in declaration.anchors]
        if any(np.any(np.abs(a) > 1e-8) for a in anchors):
            raise ValueError(
                "symmetry_trial_orbitals supports the Gamma point only "
                "(non-Gamma induction needs Bloch-phase machinery, out "
                "of R2 scope); got anchors "
                f"{[tuple(np.round(a, 6)) for a in anchors]}"
            )

    sga = build_space_group_action(phonon, symprec=symprec)
    lg = little_group(sga, q)
    nbasis = 3 * len(sga.symmetry_dataset.wyckoffs)

    wyckoffs = tuple(sga.symmetry_dataset.wyckoffs)
    letter = str(declaration.wyckoff).lstrip("0123456789")
    if not letter:
        raise ValueError(
            f"declaration wyckoff {declaration.wyckoff!r} has no letter"
        )
    orbit = [k for k, w in enumerate(wyckoffs) if w == letter]
    if not orbit:
        raise ValueError(
            f"declaration wyckoff {declaration.wyckoff!r}: no atom with "
            f"wyckoff letter {letter!r} (present: {sorted(set(wyckoffs))})"
        )
    kappa0 = orbit[0]
    site_ops = [g for g in lg if sga.atom_maps[g][kappa0] == kappa0]

    M = {g: np.asarray(sga.matrix(g, q), dtype=complex) for g in lg}

    columns = []
    blocks = []
    for name in declaration.site_irreps:
        chi_site, _, dim = resolve_site_irrep(sga, site_ops, name)
        tau = site_irrep_matrices(sga, site_ops, name)
        chi = np.array(
            [np.trace(np.asarray(tau[g], dtype=complex)) for g in lg],
            dtype=complex,
        )
        P = np.zeros((nbasis, nbasis), dtype=complex)
        for ig, g in enumerate(lg):
            P += np.conj(chi[ig]) * M[g]
        P *= dim / len(lg)
        idem = float(np.abs(P @ P - P).max())

        seeds = np.zeros((len(seed_atoms if seed_atoms else orbit) * 3, nbasis),
                         dtype=complex)
        atoms = seed_atoms if seed_atoms is not None else orbit
        for i, kappa in enumerate(atoms):
            for a in range(3):
                seeds[3 * i + a, 3 * kappa + a] = 1.0
        u, s, _ = np.linalg.svd(P @ seeds.T)
        rank = int((s > 1e-8).sum())
        U = u[:, :rank]
        # character of the projected span must equal the site irrep's
        rho_chars = [np.trace(U.conj().T @ (M[g] @ U)) for g in lg]
        char_dev = float(
            max(abs(rc - c) for rc, c in zip(rho_chars, chi))
        )
        cov = float(max(np.abs(P @ (M[g] @ U) - M[g] @ U).max() for g in lg))
        blocks.append(
            {
                "name": name,
                "dim": int(dim),
                "rank": rank,
                "singular_values": [float(x) for x in s[: rank]],
                "idempotency_residual": idem,
                "character_deviation": char_dev,
                "isotypic_covariance": cov,
            }
        )
        columns.append(U)

    orbitals = np.hstack(columns) if columns else np.zeros((nbasis, 0))
    report = TrialOrbitalReport(
        qpoint=(0.0, 0.0, 0.0),
        wyckoff=str(declaration.wyckoff),
        site_irreps=list(declaration.site_irreps),
        orbitals=orbitals,
        blocks=blocks,
    )

    if with_reference and orbitals.shape[1]:
        from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapper = PhonopyWrapper(phonon, mode="dm", is_nac=False,
                                     use_cache=False)
        _evals, evecs = wrapper.solve_all([q])
        evecs = np.asarray(evecs[0])
        U = report.orbitals
        # degenerate reference blocks (grouped by eigenvalue)
        ev = _evals[0]
        groups: List[List[int]] = []
        for i, e in enumerate(ev):
            if groups and abs(e - ev[groups[-1][0]]) <= 1e-8 * max(1.0, abs(e)):
                groups[-1].append(i)
            else:
                groups.append([i])
        overlaps = []
        for gidx in groups:
            blk = evecs[:, gidx]
            qb, _ = np.linalg.qr(blk)
            sv = np.linalg.svd(U.conj().T @ qb, compute_uv=False)
            overlaps.append(
                {
                    "eigenvalue": float(ev[gidx[0]]),
                    "block_size": len(gidx),
                    "overlap_product": float(np.prod(sv[: min(len(sv), len(gidx))])),
                }
            )
        report.copy_overlaps = overlaps
        soft = evecs[:, groups[0]]
        qs, _ = np.linalg.qr(soft)
        sv = np.linalg.svd(U.conj().T @ qs, compute_uv=False)
        report.soft_overlap_svmin = float(sv.min())
    return report
