"""Wigner-projected symmetry trial orbitals for phonon LWFs (R2).

Research memo 2026-09-22-symmetry-projected-pwf (stage R2): build
deterministic, symmetry-exact trial orbitals at the Gamma point from a
:class:`lawaf.anharmonic.compatibility.RepresentationDeclaration`.

Construction (review QT-F01): the declared site carrier induces a
representation ``Ind_H^G chi_site`` of the Gamma little group; the
trial-orbital projector is the sum of the central Wigner projectors over
every little-group irrep contained in that induced sector,

.. math::
    Q = \\sum_{\\alpha \\,\\subset\\, \\mathrm{Ind}\\,\\chi_{\\mathrm{site}}}
    \\hat P^{(\\alpha)}, \\qquad
    \\hat P^{(\\alpha)} = \\frac{d_\\alpha}{|G|}
    \\sum_{g\\in G}\\chi^{(\\alpha)}(g)^{*}\\,D(g),

applied to local Cartesian displacement seeds of the declared Wyckoff
orbit and orthonormalized (SCMPWF-style: symmetry exact by construction,
no localization iterations). ``n_orbit * d_site * multiplicity``
orthonormal carriers are extracted; the report carries projector
diagnostics and, against a reference eigenspectrum, the per-copy
overlaps documenting the memo's central finding: the projection selects
the irrep SECTOR exactly but never the COPY (on BaTiO3 the projected Ti
seeds overlap the soft triplet at svmin 0.68 only).

Scope: Gamma point (``q=(0,0,0)``); non-Gamma q is rejected with a
clear error (general-q induction needs Bloch-phase machinery that is
deliberately out of R2 scope). This module never changes the
projected-Wannier construction path; the orbitals are consumable via
``ProjectedWannierizer.set_projectors``.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from lawaf.anharmonic.compatibility import (
    RepresentationDeclaration,
    character_table,
    induced_characters,
    little_group,
    resolve_site_irrep,
)
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
        One entry per distinct declared site irrep: name, site-irrep
        dimension, induced-sector irreps (label: multiplicity), the
        number of carriers extracted, projector idempotency residual,
        and the sector-purity residual (characters of the extracted
        span decompose into integer multiplicities within the sector).
    copy_overlaps :
        Per degenerate reference-eigenspace block overlapping the
        extracted span: eigenvalue, block size, and the product of the
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
        A prepared ``phonopy.Phonopy`` object (force constants set or
        not -- the construction uses its symmetry dataset only;
        reference eigenvectors, if requested, come from a cache-free
        non-NAC solve of its dynamical matrix).
    declaration :
        Declared site representation (wyckoff letter + site_irreps
        with repetition = multiplicity of independent carriers).
    seed_atoms :
        Atom indices whose Cartesian x/y/z unit displacements seed the
        projection. Default: all atoms of the declared Wyckoff orbit.
        More seeds than the orbit are needed when the declaration
        requests several copies of one site irrep.
    with_reference :
        Compute per-copy overlaps against the reference eigenspectrum
        (cheap at Gamma; set False to skip the reference solve).

    Raises
    ------
    ValueError
        For non-Gamma anchors, an unknown Wyckoff letter, an unknown
        site-irrep name, or seeds insufficient to furnish the requested
        multiplicity.
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
    ct = character_table(sga, lg)
    nbasis = 3 * len(sga.symmetry_dataset.wyckoffs)

    # reference Gamma eigensystem (cache-free, non-NAC): used both for
    # the copy-overlap diagnostics and as the fallback copy splitter
    # (the dynamical matrix commutes with the little group, so its
    # eigenvalue clusters inside a sector-irrep range are invariant
    # subspaces -- frequency splits what symmetric seeds cannot)
    H = None
    if with_reference:
        from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapper = PhonopyWrapper(phonon, mode="dm", is_nac=False,
                                     use_cache=False)
        _evals, evecs = wrapper.solve_all([q])
        evecs = np.asarray(evecs[0])
        H = (evecs * _evals[0][None, :]) @ evecs.conj().T

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

    M = {int(g): np.asarray(sga.matrix(g, q), dtype=complex) for g in lg}

    columns = []
    blocks = []
    for name, m_decl in sorted(Counter(declaration.site_irreps).items()):
        chi_site, _, dim_site = resolve_site_irrep(sga, site_ops, name)
        chi_ind = induced_characters(sga, lg, site_ops, chi_site)
        chi_ind_d = {int(g): chi_ind[i] for i, g in enumerate(lg)}
        mult_ind = ct.multiplicities(chi_ind_d)
        if np.abs(mult_ind.real - np.round(mult_ind.real)).max() > 1e-4:
            raise ValueError(
                f"non-integer induced-sector multiplicities for {name}"
            )
        # central projector of every little-group irrep contained in the
        # induced sector (QT-F01: correct for proper site stabilizers,
        # i.e. any Wyckoff orbit, not just maximal-symmetry sites)
        P_of = {}
        for j, m in enumerate(mult_ind):
            if int(round(m.real)) >= 1:
                chi_a = {int(g): ct.chars[j, ct._cls_of[int(g)]] for g in lg}
                P = np.zeros((nbasis, nbasis), dtype=complex)
                for g in lg:
                    P += (
                        ct.dims[j] / len(lg) * np.conj(chi_a[int(g)]) * M[int(g)]
                    )
                P_of[j] = P
        Q = sum(P_of.values())
        idem = float(np.abs(Q @ Q - Q).max())

        # copy splitter: a seed-weight operator symmetrized over the
        # little group commutes with every M(g), so its eigenvalue
        # CLUSTERS inside each sector-irrep range are invariant
        # subspaces; taking whole clusters yields symmetry-exact
        # carriers and resolves the copy multiplicity (QT-F02)
        atoms = seed_atoms if seed_atoms is not None else orbit
        seeds = np.zeros((3 * len(atoms), nbasis), dtype=complex)
        for i, kappa in enumerate(atoms):
            for a in range(3):
                seeds[3 * i + a, 3 * kappa + a] = 1.0
        S = seeds.T @ seeds.conj()
        A = np.zeros((nbasis, nbasis), dtype=complex)
        for g in lg:
            A += M[int(g)] @ S @ M[int(g)].conj().T

        carriers = []
        for j, P in P_of.items():
            # declared multiplicity times the induced-copy count
            need = m_decl * int(round(mult_ind[j].real)) * ct.dims[j]
            PA = P @ A @ P
            w, v = np.linalg.eigh(PA)
            order = np.argsort(w.real)[::-1]
            take = []
            i = 0
            while i < len(order) and len(take) < need:
                ev = w.real[order[i]]
                cluster = [k for k in order[i:]
                           if abs(w.real[k] - ev) <= 1e-8 * max(1.0, abs(ev))]
                if len(take) + len(cluster) > need:
                    # seed-degenerate cluster overshoots the declared
                    # multiplicity: split it by the dynamical matrix
                    # (frequency), taking the LOWEST eigenvalues first
                    if H is None:
                        raise ValueError(
                            f"seeds cannot furnish {name} on wyckoff "
                            f"{declaration.wyckoff!r}: a seed-degenerate "
                            f"cluster of {len(cluster)} overshoots the "
                            f"{need} requested carriers of irrep "
                            f"{ct.names[j]}; rerun with "
                            "with_reference=True (frequency splitting) "
                            "or supply seed_atoms that resolve the copies"
                        )
                    Vc = v[:, cluster]
                    wc, vc = np.linalg.eigh(Vc.conj().T @ H @ Vc)
                    sub = []
                    si = 0
                    while si < len(wc) and len(take) + len(sub) < need:
                        sev = wc.real[si]
                        scl = [k for k in range(si, len(wc))
                               if abs(wc.real[k] - sev)
                               <= 1e-8 * max(1.0, abs(sev))]
                        sub.extend(scl)
                        si += len(scl)
                    take.extend(cluster[k] for k in sub[: need - len(take)])
                    i += len(cluster)
                    continue
                take.extend(cluster)
                i += len(cluster)
            if len(take) < need:
                raise ValueError(
                    f"seeds cannot furnish {name} on wyckoff "
                    f"{declaration.wyckoff!r}: only {len(take)} of "
                    f"{need} cluster vectors available in irrep "
                    f"{ct.names[j]}; supply more seed_atoms"
                )
            carriers.append(v[:, take])
        U = np.hstack(carriers) if carriers else np.zeros((nbasis, 0))
        # sector purity: characters of the extracted span decompose
        # into integer multiplicities, all within the induced sector
        rho = {
            int(g): complex(np.trace(U.conj().T @ (M[int(g)] @ U)))
            for g in lg
        }
        mult_span = ct.multiplicities(rho)
        purity = float(
            np.abs(mult_span.real - np.round(mult_span.real)).max()
        )
        outside = [
            ct.names[j]
            for j, m in enumerate(mult_span)
            if int(round(m.real)) >= 1 and j not in P_of
        ]
        if outside:
            raise ValueError(
                f"extracted span leaves the declared sector for {name}: "
                f"{outside}"
            )
        cov = float(
            max(np.abs(Q @ (M[int(g)] @ U) - M[int(g)] @ U).max() for g in lg)
        )
        blocks.append(
            {
                "name": name,
                "multiplicity": int(m_decl),
                "dim_site": int(dim_site),
                "sector_irreps": {
                    ct.names[j]: int(round(mult_ind[j].real)) for j in P_of
                },
                "n_carriers": int(U.shape[1]),
                "projector_idempotency": idem,
                "sector_purity_residual": purity,
                "sector_covariance": cov,
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
