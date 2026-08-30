"""Maximally localized Wannier functions (Marzari-Vanderbilt) backend.

Stories 006/007: the d_omega optimization and Mmn-form spread diagnostics,
built strictly on the executed sympy derivation of story-009
(``lawaf/docs/derivations/mlwf_domega_derivation.py``; check names cited
below) and reusing the overlap core in :mod:`lawaf.io.w90`
(``kmesh_nnlist``/``compute_Mmn``) per MLWF architecture ADR-001.

Spread-reporting contract (ADR-004, amended 2026-08-29): the reported
Omega/Omega_I/Omega_D/Omega_OD decomposition is computed from the Mmn form
(wannier90-identical on any mesh); R-space sums are diagnostics only.
"""

import numpy as np
from scipy.linalg import expm, qr

from .projectedWF import ProjectedWannierizer


def gauged_mmn(mmn, nnlist, U):
    """M~^{k,b} = U_k^dag M^{k,b} U_{k+b} (right action, derivation §2)."""
    out = np.empty_like(mmn)
    for ik in range(mmn.shape[0]):
        Ukdag = U[ik].conj().T
        for j in range(mmn.shape[1]):
            out[ik, j] = Ukdag @ mmn[ik, j] @ U[nnlist[ik, j]]
    return out


def _as_bk(bvecs, nkpt):
    """Normalize b-vectors to per-k ``bk[k, j] = k_j + G - k`` (w90 ``bk``
    convention). Accepts (nntot, 3) (mesh-invariant b, e.g. test models)
    broadcast across k, or (nkpt, nntot, 3) directly."""
    bvecs = np.asarray(bvecs)
    if bvecs.ndim == 2:
        return np.broadcast_to(bvecs[None], (nkpt,) + bvecs.shape)
    return bvecs


def omega_decomposition(mmn_t, wb, bvecs, guide=None):
    """Mmn-form spread decomposition (derivation checks 1a/1b/2a).

    :param mmn_t: (nkpt, nntot, N, N) gauged overlaps M~^{k,b};
    :param wb: (nntot,) neighbour weights w_b (w90 normalization
        ``sum_j w_j b_ja b_jb = delta_ab`` — load-bearing, check 1a);
    :param bvecs: (nntot, 3) mesh-invariant or (nkpt, nntot, 3) per-k
        signed b-vectors ``bk[k,j] = k_j + G - k`` (w90 convention; the
        folded image G differs per k, so real meshes need the per-k form).
    :returns: dict ``omega, omega_I, omega_D, omega_OD, rbar, omega_n``.

    ``theta~ = Im ln M~_nn`` uses the principal branch, or — when ``guide``
    (N, 3) guiding centres are given — wannier90's sheet convention
    (``wannierise.F90`` 1658-1690): theta~ = principalArg(theta_raw +
    b.rguide) - b.rguide, which keeps theta~ + b.rbar near 0 (multi-cell
    friendly). A w90-comparable decomposition uses rbar from a first
    principal-branch pass as the guide.
    """
    mmn_t = np.asarray(mmn_t)
    nkpt, nntot, nwann, _ = mmn_t.shape
    bk = _as_bk(bvecs, nkpt)  # (nkpt,nntot,3) per-k b-vectors
    diag = mmn_t.diagonal(axis1=-2, axis2=-1)  # (nkpt,nntot,N)
    if guide is not None:
        sh = np.einsum("kjx,nx->kjn", bk, np.asarray(guide))  # b.rguide
        theta = np.angle(diag * np.exp(1j * sh)) - sh
    else:
        theta = np.angle(diag)  # (nkpt,nntot,N)
    # rbar_n = -(1/Nk) sum_{k,j} w_j b_j theta~_n   (derivation section 1)
    rbar = -np.einsum("j,kjn,kjx->nx", wb, theta, bk) / nkpt
    # center-shifted phases: dtheta[k,j,n] = theta[k,j,n] + b_{k,j}.rbar[n]
    dtheta = theta + np.einsum("kjx,nx->kjn", bk, rbar)
    om_I = om_D = om_OD = 0.0
    for j in range(nntot):
        m = mmn_t[:, j]
        # check 2a: Omega_I = (1/Nk) sum w (N - Tr M~^dag M~)
        om_I += wb[j] * np.mean(nwann - np.sum(np.abs(m) ** 2, axis=(-2, -1)))
        # check 1b: sum_{m!=n}|M~_mn|^2
        offd = (np.sum(np.abs(m) ** 2, axis=(-2, -1))
                - np.sum(np.abs(m.diagonal(axis1=-2, axis2=-1)) ** 2, axis=-1))
        om_OD += wb[j] * np.mean(offd)
        # check 1a: Omega_D per band, then band-mean for the total
        om_D += wb[j] * np.mean(np.sum(dtheta[:, j, :] ** 2, axis=-1))
    # per-band contributions: Omega_n = OmegaI_n + OmegaD_n (check 1a;
    # MV97 per-band spread — Omega_OD has no per-band decomposition, so
    # sum(omega_n) == omega_I + omega_D, not omega, when Omega_OD > 0)
    omega_n = np.zeros(nwann)
    for n in range(nwann):
        om_i_n = om_d_n = 0.0
        for j in range(nntot):
            om_i_n += wb[j] * np.mean(1.0 - np.abs(mmn_t[:, j, n, n]) ** 2)
            om_d_n += wb[j] * np.mean(dtheta[:, j, n] ** 2)
        omega_n[n] = om_i_n + om_d_n
    return {
        "omega": om_I + om_D + om_OD,
        "omega_I": om_I,
        "omega_D": om_D,
        "omega_OD": om_OD,
        "rbar": rbar,
        "omega_n": omega_n,
    }


def _assert_pair_hermiticity(mmn, nnlist, nncell, bvecs, tol=1e-8):
    """NFR-002: for each neighbour pair (k, k+b) present in both
    directions, M^{k+b,-b} = (M^{k,b})^dagger (derivation §3 contract)."""
    nkpt, nntot = nnlist.shape
    for ik in range(nkpt):
        for j in range(nntot):
            kp = nnlist[ik, j]
            # find the reverse neighbour (k+b -> k with -b)
            back = [jj for jj in range(nntot)
                    if nnlist[kp, jj] == ik
                    and np.allclose(nncell[kp, jj], -nncell[ik, j])]
            if not back:
                continue
            rev = mmn[kp, back[0]]
            if not np.allclose(rev, mmn[ik, j].conj().T, atol=tol):
                raise AssertionError(
                    f"Mmn pair hermiticity violated at k={ik}, "
                    f"neighbour {j} (b={bvecs[j]})")


def _cdodq(mmn_t, ik, nnlist, wb, bvecs, rbar, nkpt):
    """w90 wann_domega cdodq transcription (derivation checks 3a/3b).

    g_k = sum_j { w(cr - cr^dag)/2 + (i/2)(crt.ln_tmp + h.c.)
                  + (i/2) w (crt.rnkb + h.c.) },  G_k = 4 g_k / Nk
    (``ln_tmp`` carries ``w_b`` — domega convention, wannierise.F90:2038;
    ``rnkb = b . rbar_n``).
    """
    nwann = mmn_t.shape[2]
    bk = _as_bk(bvecs, mmn_t.shape[0])
    g = np.zeros((nwann, nwann), dtype=complex)
    for j in range(mmn_t.shape[1]):
        m = mmn_t[ik, j]
        w = wb[j]
        mnn = m.diagonal()
        if np.min(np.abs(mnn)) < 1e-10:
            # a vanishing diagonal overlap makes crt = m/mnn and the
            # phase of M~_nn undefined; skip the pair (same convention
            # as _phase_update) rather than emit a NaN gradient
            continue
        lnt = w * np.angle(mnn)
        rnkb = bk[ik, j] @ rbar.T  # (N,) b . rbar_n
        cr = m * mnn.conj()[None, :]
        crt = m / mnn[None, :]
        t1 = w * (cr - cr.conj().T) / 2.0
        # w90 2176-2181 / derivation cdodq_k: h.c. term carries the
        # vector at index m: conj(crt[n,m] * vec[m])
        t2 = 0.5j * (crt * lnt[None, :] + (crt.T * lnt[:, None]).conj())
        t3 = 0.5j * w * (crt * rnkb[None, :]
                         + (crt.T * rnkb[:, None]).conj())
        g += t1 + t2 + t3
    return 4.0 * g / nkpt


def _phase_update(mmn_t, nnlist, wb, bvecs, rbar, U):
    """Gauge-phase update at fixed rbar (derivation check 2c).

    gamma_kn = sum_j w_j (theta~_kjn + b_kj.rbar_n) / sum_j w_j — the
    EXACT minimizer of the quadratic-in-phase Omega_D subproblem at fixed
    rbar. (w90/MV97 eq. 25 use the circular mean arg sum_j w_j
    e^{i(theta + b.rbar)}; those coincide only for symmetric residual
    sets and the circular form leaves a nonzero d(Omega_D) otherwise.)
    Vanishing overlaps (phase undefined) are excluded from both sums.
    U_k <- U_k diag(e^{i gamma})."""
    nkpt, nntot = nnlist.shape
    nwann = mmn_t.shape[2]
    bk = _as_bk(bvecs, nkpt)
    for ik in range(nkpt):
        num = np.zeros(nwann)
        den = np.zeros(nwann)
        for j in range(nntot):
            mnn = mmn_t[ik, j].diagonal()
            small = np.abs(mnn) < 1e-12
            residual = np.angle(np.where(small, 1.0, mnn)) + bk[ik, j] @ rbar.T
            num += wb[j] * np.where(small, 0.0, residual)
            den += wb[j] * (~small)
        gamma = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        U[ik] = U[ik] @ np.diag(np.exp(1j * gamma))


def d_omega_optimize(mmn, nnlist, bvecs, wb, U0=None, tol=1e-10,
                     max_iter=100, fixed_gauge=False, verbose=False):
    """Marzari-Vanderbilt d_omega sweeps (story-006).

    Alternates (a) the diagonal gauge-phase update (MV97 eq. 25 = derivation
    check 2c at fixed rbar) and (b) the Omega_OD unitary update by steepest
    descent on the derivation check-3b gradient
    ``A_k = -(i/2)(G_k - G_k^dag)`` with the right action
    ``U_k <- U_k e^{i eps A_k}`` and backtracking line search (FR-005
    monotonic acceptance).

    :param mmn: (nkpt, nntot, N, N) raw overlaps M^{k,b} (compute_Mmn);
    :param nnlist: (nkpt, nntot) neighbour k indices (kmesh_nnlist);
    :param bvecs: (nntot, 3) mesh-invariant or (nkpt, nntot, 3) per-k
        signed b-vectors (crystallographic; real meshes need per-k);
    :param wb: (nntot,) neighbour weights;
    :param U0: (nkpt, N, N) initial unitaries (None -> identity);
    :param tol: stop when |dOmega| < tol between sweeps;
    :param max_iter: sweep cap;
    :param fixed_gauge: skip the unitary Omega_OD update (FR-007).
    :returns: ``(U, history)`` with ``history`` a list of spread dicts.
    """
    mmn = np.asarray(mmn)
    nkpt, nntot, nwann, _ = mmn.shape
    rng = np.random.default_rng(20260829)
    U = ([np.eye(nwann, dtype=complex) for _ in range(nkpt)]
         if U0 is None else [u.copy() for u in U0])
    history = []
    omega_prev = np.inf
    for it in range(max_iter):
        # (a) gauge-phase update (MV97 eq. 25), iterated to
        # self-consistency within the sweep: eq. 25 is exact at FIXED
        # rbar, but rbar itself shifts after the update, so repeat
        # (phase update at current rbar -> refresh M~/rbar/spreads)
        # until Omega stops moving (single-band and phase-only gauges
        # then converge in ONE sweep, story-006 AC).
        mmn_t = gauged_mmn(mmn, nnlist, U)
        spreads = omega_decomposition(mmn_t, wb, bvecs)
        for _ in range(50):
            _phase_update(mmn_t, nnlist, wb, bvecs, spreads["rbar"], U)
            mmn_t = gauged_mmn(mmn, nnlist, U)
            new = omega_decomposition(mmn_t, wb, bvecs)
            done = abs(new["omega"] - spreads["omega"]) < 1e-15
            spreads = new
            if done:
                break
        if fixed_gauge:
            history.append(spreads)
            if abs(spreads["omega"] - omega_prev) < tol:
                break
            omega_prev = spreads["omega"]
            continue
        # (b) unitary Omega_OD update: steepest descent with backtracking
        # (derivation check 3b; right action U_k <- U_k e^{i eps A_k})
        accepted = False
        eps_step = 1.0
        step_norm = 0.0
        for _ in range(30):
            trial = [u.copy() for u in U]
            step_norm = 0.0
            for ik in range(nkpt):
                G = _cdodq(mmn_t, ik, nnlist, wb, bvecs, spreads["rbar"], nkpt)
                A = -0.5j * (G - G.conj().T)
                step_norm = max(step_norm, np.abs(A).max())
                trial[ik] = U[ik] @ expm(1j * eps_step * A)
            new = omega_decomposition(gauged_mmn(mmn, nnlist, trial), wb, bvecs)
            if (new["omega"] <= spreads["omega"] + 1e-14
                    and step_norm > 1e-13):
                accepted = True
                break
            eps_step *= 0.5
        if not accepted:
            # Steepest descent stalls at symmetric saddles (zero gradient
            # at a non-minimal gauge). Try small random Hermitian kicks;
            # accept only if Omega decreases (monotonicity preserved).
            for _ in range(10):
                trial = [u.copy() for u in U]
                for ik in range(nkpt):
                    h = (rng.normal(size=(nwann, nwann))
                         + 1j * rng.normal(size=(nwann, nwann)))
                    A = 0.05 * (h + h.conj().T) / 2.0
                    trial[ik] = U[ik] @ expm(1j * A)
                new = omega_decomposition(
                    gauged_mmn(mmn, nnlist, trial), wb, bvecs)
                if new["omega"] < spreads["omega"] - 1e-14:
                    accepted = True
                    break
            if not accepted:
                history.append(spreads)
                break  # genuine (local) minimum for the line search
        U = trial
        history.append(new)
        if verbose:
            print(f"iter {it}: Omega = {new['omega']:.10f}")
        if abs(new["omega"] - omega_prev) < tol:
            break
        omega_prev = new["omega"]
    return U, history


class MLWFWannierizer(ProjectedWannierizer):
    """Iteratively maximize localization of the selected subspace.

    The initial guess is the projected (or scdmk-fallback) Amn path
    inherited from :class:`ProjectedWannierizer` (ADR-004); the MV sweeps
    refine the gauge on top of it. After ``get_Amn()`` the refined
    ``self.Amn`` carries the optimized gauge and ``self.spreads`` holds the
    Mmn-form spread decomposition (amended ADR-004).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_orthogonal:
            raise NotImplementedError(
                "MLWFWannierizer v1 requires an orthogonal basis "
                "(Sk=None): the S-metric overlap Mmn <u_k|S|u_{k+b}> "
                "needed for a non-orthogonal basis is not implemented. "
                "Orthogonalize the basis first (e.g. Lowdin/use_proj path)."
            )
        self.selection = None

    def get_Amn(self):
        """Initial guess, then MV d_omega refinement (ADR-004).

        ``params.mlwf_initial_guess`` selects the initial gauge:

        - ``"projected"`` (default): projector guess when projectors/anchors
          are set; otherwise the ADR-004 fallback is an SCDM-k guess via the
          same anchor machinery (:class:`ScdmkWannierizer`);
        - ``"identity"``: identity gauge (nband == nwann is enforced in
          run_mlwf; useful for isolated-manifold tests).
        """
        import copy

        from lawaf.wannierization.scdmk import ScdmkWannierizer

        guess = getattr(self.params, "mlwf_initial_guess", "projected")
        has_projectors = getattr(self, "projectors", None) is not None
        if guess == "identity":
            self.Amn = np.stack(
                [np.eye(self.nband, self.nwann, dtype=complex)
                 for _ in range(self.nkpt)])
        elif guess == "projected" and has_projectors:
            super().get_Amn()
        else:
            # ADR-004 scdmk fallback (explicit 'scdmk', or 'projected'
            # without projectors). The projector weighting is meaningless
            # here (no anchors -> SCDM-k column selection only), so force
            # use_proj off in the guess builder.
            # copy (not dataclasses.replace: WannierParams has
            # unannotated attrs like kshift that replace() would drop).
            # Projector weighting is only meaningful when anchors exist;
            # force it off solely for the no-projector auto-anchor case.
            scdmk_params = copy.copy(self.params)
            if not (scdmk_params.anchors or scdmk_params.selected_basis):
                scdmk_params.use_proj = False
            scdmk = ScdmkWannierizer(
                self.evals, self.evecs, self.kpts, self.kweights,
                scdmk_params)
            scdmk.get_Amn()
            self.Amn = scdmk.Amn
        self.run_mlwf()
        return self.Amn

    # -- internals ---------------------------------------------------------

    def _recip_from_kpts(self):
        """Reciprocal basis consistent with the k-points as given.

        lawaf k-points are crystal (fractional) coordinates, so the
        consistent reciprocal basis is the identity: b-vectors come out in
        crystal units and the spreads are reported in crystal units
        (multiply by (|b_cart|/|b_crystal|)^2 per axis — scalar (a/2pi)^2
        for cubic cells — to compare with wannier90 Angstrom^2 output).
        Axes with N_d = 1 are completed with unit vectors; neighbours along
        them are self-overlaps (M = I, theta~ = 0) and contribute nothing.
        """
        return np.eye(3)

    def k_to_R(self, Rlist=None, Rdeg=None):
        """Story-007: attach the Mmn-form spread diagnostics to the result.

        ``wann_centers`` are the Mmn-form centres rbar_n (crystal units,
        matching the spread decomposition; R-space centre sums remain
        available as diagnostics via get_wannier_centers).
        """
        lwf = super().k_to_R(Rlist=Rlist, Rdeg=Rdeg)
        if self.spreads is not None:
            try:
                lwf.spreads = dict(self.spreads)
                lwf.wann_centers = self.spreads["rbar"]
            except AttributeError:
                pass  # frozen result class: diagnostics stay on self
        if self.selection is not None:
            try:
                lwf.selection = self.selection  # FR-007 diagnostics
            except AttributeError:
                pass
        return lwf
    def _has_selection_guidance(self):
        """ADR-004/F3 guard: an OUTER window or validated pins must shape
        the feasible sets. Frozen-only bounds refine the frozen core but
        leave the free complement unrestricted (F3 drift), so they do
        not enable selection on their own."""
        p = self.params
        has_outer = any(
            getattr(p, key, None) is not None
            for key in ("dis_win_min", "dis_win_max")
        )
        has_pins = (
            getattr(p, "_window_bands_resolved", None) is not None
            or getattr(p, "window_bands", None) is not None
        )
        return has_outer or has_pins

    def run_mlwf(self, kmesh_tol=1e-6):
        """Run the MV sweeps; refines ``self.Amn`` and sets ``self.spreads``."""
        from lawaf.io.w90 import compute_Mmn, kmesh_nnlist

        if self.ndim < 3 or len(self.kpts) != int(np.prod(self.kmesh[: self.ndim])):
            raise ValueError(
                "MLWF optimization requires a full Monkhorst-Pack mesh "
                f"(kmesh={self.kmesh}, nkpt={len(self.kpts)})"
            )
        if self.nband < self.nwann:
            raise ValueError(
                f"nband={self.nband} < nwann={self.nwann}: nothing to "
                "select from"
            )
        if self.nband > self.nwann and not self._has_selection_guidance():
            raise ValueError(
                f"nband={self.nband} > nwann={self.nwann}: the selection "
                "stage needs window guidance — set dis_win_min/dis_win_max "
 "(and optionally dis_froz_min/dis_froz_max), or validated per-q "
                "window_bands. Running the overlap fixed point without "
                "windows drifts to a smooth but physically wrong bundle."
            )
        recip = self._recip_from_kpts()
        nnlist, nncell, bvecs, wb = kmesh_nnlist(
            self.kpts, recip, kmesh_tol=kmesh_tol)
        # per-k b-vectors (w90 bk): k_j + G - k in kpts units
        bk = (self.kpts[nnlist] + nncell
              - self.kpts[:, None, :])
        psi = np.stack([self.get_psi_k(ik) for ik in range(self.nkpt)])
        mmn = compute_Mmn(psi, nnlist, nncell)
        _assert_pair_hermiticity(mmn, nnlist, nncell, bvecs)
        U_sel = None
        if self.nband > self.nwann:
            # ADR-004 seam: selection -> psi_sel -> existing machinery.
            from lawaf.wannierization.disentangle import (
                resolve_windows,
                select_subspace,
            )

            p = self.params
            # psi/Amn work on retained rows (exclude_bands removed);
            # windows must see the same rows (original pin indices are
            # mapped through ibands inside resolve_windows)
            evals_ret = self.evals[:, list(self.ibands)]
            window_bands = getattr(
                p, "_window_bands_resolved", p.window_bands)
            feasible, frozen = resolve_windows(
                self.kpts, evals_ret, self.nwann,
                window_bands=window_bands,
                win_min=p.dis_win_min, win_max=p.dis_win_max,
                froz_min=p.dis_froz_min, froz_max=p.dis_froz_max,
                ibands=self.ibands,
            )
            if window_bands is None and all(
                len(feasible[ik]) == self.nband for ik in range(self.nkpt)
            ):
                raise ValueError(
                    "dis_win_min/dis_win_max span the whole retained "
                    "spectrum at every k: the feasible sets are "
                    "unrestricted and the overlap fixed point would "
                    "drift (F3). Tighten the outer window or use "
                    "validated window_bands."
                )
            self.selection = select_subspace(
                mmn, nnlist, wb, feasible, frozen, self.nwann,
                eigvals=evals_ret,
                mix_ratio=p.dis_mix_ratio, max_iter=p.dis_max_iter,
                tol=p.dis_tol, min_svd=p.dis_min_svd,
                slow_tail_change=p.dis_slow_tail_change)
            self.selection.guidance["sources"] = {
                "outer_window": (p.dis_win_min, p.dis_win_max),
                "inner_window": (p.dis_froz_min, p.dis_froz_max),
                "window_bands": window_bands is not None,
            }
            U_sel = self.selection.U_sel
            # gauge guess: projected Amn expressed in the selected basis
            # (nwann x nwann GAUGE unitaries — not Hilbert-space columns)
            U0 = [qr(U_sel[ik].conj().T @ self.Amn[ik])[0]
                  for ik in range(self.nkpt)]
            mmn = np.array(
                [[U_sel[ik].conj().T @ mmn[ik, ib] @ U_sel[nnlist[ik, ib]]
                  for ib in range(mmn.shape[1])]
                 for ik in range(self.nkpt)])
        else:
            U0 = [self.Amn[ik].copy() for ik in range(self.nkpt)]
        U, history = d_omega_optimize(
            mmn, nnlist, bk, wb, U0=U0,
            tol=self.params.mlwf_tol, max_iter=self.params.mlwf_max_iter,
            fixed_gauge=self.params.mlwf_fixed_gauge,
        )
        # U is the TOTAL gauge (d_omega_optimize starts from U0 and gauges
        # the raw Mmn with U directly): with a selection, the total
        # full-manifold Amn is U_sel @ U (psi_sel @ U = psi @ (U_sel @ U)).
        if U_sel is None:
            self.Amn = np.stack([U[ik] for ik in range(self.nkpt)])
        else:
            self.Amn = np.stack([U_sel[ik] @ U[ik] for ik in range(self.nkpt)])
        mmn_t = gauged_mmn(mmn, nnlist, U)
        self.spreads = omega_decomposition(
            mmn_t, wb, bk,
            guide=omega_decomposition(mmn_t, wb, bk)["rbar"])
        self.mlwf_history = history
        return self.spreads
