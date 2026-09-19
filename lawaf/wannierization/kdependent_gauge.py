"""k-dependent GL gauge G(k) for non-orthogonal Wannier functions.

Parametrization (research memo specs/research/2026-09-19-kdependent-gauge.md,
formulation (a)): a smooth plane-wave generator on shells of the model
R-list,

.. math::

    G(\\mathbf k) = \\exp\\Big[\\sum_{\\mathbf R} \\Lambda(\\mathbf R)
        e^{2\\pi i \\mathbf k\\cdot\\mathbf R}\\Big],

applied on top of any orthonormal pipeline result,
$A'(\\mathbf k) = U(\\mathbf k) G(\\mathbf k)$. The overlap
$S^w(\\mathbf k) = G^\\dagger(\\mathbf k) G(\\mathbf k)$ becomes
k-dependent, so $S^w(\\mathbf R)$ gains finite range; mesh-k bands
remain pencil-exact (full-rank $A$ within the retained subspace), and
the constant-G path is the shell-{$0$} special case
($\\Lambda(0) = \\log G$).

The optimizer minimizes the normalized per-orbital spread
(same diagonal-position objective family as
`lawaf.wannierization.nonorthogonal_gauge`) over the $\\Lambda$
coefficients, with:

- a FROZEN, scale-invariant per-k conditioning barrier
  $-\\mu \\sum_{\\mathbf k} \\log\\det[S(\\mathbf k)/\\operatorname{diag}S(\\mathbf k)]$
  (a plain logdet barrier rewards growing $G$ without bound — verified
  runaway in the research memo);
- Tikhonov smoothness on the non-constant shells;
- column-norm pinning (the objective is 0-homogeneous per column);
- deterministic multi-start from the constant-G initialization
  ($\\Lambda(0) = \\log G_0$; the orthonormal gauge is a stationary
  point of the GL functional, so tilts are required to escape it).

The Wirtinger gradient chain
($\\Omega \\to W' \\to G \\to \\Lambda$, adjoint-Frechet for the exp)
was verified against central finite differences (1.4e-9) and sympy
(per-coefficient identity 0) in the research memo.

Off-mesh semantics: with a k-dependent gauge the congruence exactness
of the constant transform does NOT extend off the downfolding mesh;
treat the result as an R-space-consumed model (the application helper
warns accordingly).
"""

import warnings

import numpy as np
from scipy.linalg import eigh, expm, expm_frechet
from scipy.optimize import minimize

from lawaf.mathutils.kR_convert import k_to_R, R_to_k

__all__ = [
    "KDepGauge",
    "select_shells",
    "kdependent_objective",
    "optimize_kdependent_gauge",
    "apply_kdependent_gauge",
]


def select_shells(Rlist, max_shell=1):
    """Select gauge-generator R vectors from the model R-list.

    Shell 0 is R = 0 (required). Higher shells group the remaining
    vectors by the norm of their centered representation, taking the
    ``max_shell`` smallest groups. Returns the selected (nRg, 3) array.
    """
    Rlist = np.asarray(Rlist, dtype=float)
    i0 = np.where(np.all(Rlist == 0, axis=1))[0]
    if len(i0) != 1:
        raise ValueError(f"need exactly one R=0 entry in Rlist, found {len(i0)}")
    others = np.delete(Rlist, i0[0], axis=0)
    if max_shell < 0 or len(others) == 0:
        return Rlist[i0[0]][None, :]
    norms = np.linalg.norm(others, axis=1)
    order = np.argsort(norms)
    groups = []
    last = None
    for idx in order:
        v = round(float(norms[idx]), 6)
        if v != last:
            groups.append([])
            last = v
        groups[-1].append(idx)
    sel = [i0[0]]
    for g in groups[:max_shell]:
        sel.extend(g)
    return Rlist[np.sort(sel)]


class KDepGauge:
    """Container/evaluator for a k-dependent gauge: G(k) =
    expm(sum_R Lambda(R) exp(2 pi i k.R)) over the generator vectors.
    """

    def __init__(self, Rg, Lambda, method="exp"):
        if method != "exp":
            raise ValueError(f"unknown method {method!r}")
        self.Rg = np.asarray(Rg, dtype=float)
        self.Lambda = np.asarray(Lambda, dtype=complex)
        if self.Lambda.ndim != 3 or self.Lambda.shape[0] != len(self.Rg):
            raise ValueError(
                f"Lambda must have shape (nRg, nwann, nwann) with nRg = "
                f"{len(self.Rg)}, got {self.Lambda.shape}"
            )

    def L_of_k(self, kpts):
        """L(k) = sum_R Lambda(R) exp(2 pi i k.R)."""
        kpts = np.asarray(kpts, dtype=float)
        phase = np.exp(2j * np.pi * np.einsum("kd,rd->kr", kpts, self.Rg))
        return np.einsum("rlm,kr->klm", self.Lambda, phase)

    def G_of_k(self, kpts):
        return np.array([expm(L) for L in self.L_of_k(kpts)])

def kdependent_objective(wannR, Rlist, Rdeg, positions, kpts, Rg):
    """Objective + analytic Wirtinger gradient for the convolved model.

    Returns a callable ``f(Lambda, mu, pin_reg) -> (total, grad_c,
    omega, aux)`` where ``grad_c`` is dOmega_mu/dLambda* (complex, same
    shape as Lambda) and ``omega`` is the bare spread. Follows the
    lawaf FT conventions exactly (``k_to_R`` / ``R_to_k`` with Rdeg on
    the R-to-k side only).
    """
    wannR = np.asarray(wannR, dtype=complex)
    Rlist = np.asarray(Rlist, dtype=float)
    Rdeg = np.ones(len(Rlist)) if Rdeg is None else np.asarray(Rdeg, dtype=float)
    positions = np.asarray(positions, dtype=float)
    kpts = np.asarray(kpts, dtype=float)
    Rg = np.asarray(Rg, dtype=float)
    U = R_to_k(kpts, Rlist, wannR, Rdeg)  # (nk, nb, nw) orthonormal gauge
    nk, nb, nw = U.shape
    p = Rlist[:, None, :] + positions[None, :, :]  # (nR, nb, 3)
    q = [Rdeg[:, None] * p[:, :, a] for a in range(3)]
    q2 = [Rdeg[:, None] * p[:, :, a] ** 2 for a in range(3)]

    def build(Lam):
        phase = np.exp(2j * np.pi * np.einsum("kd,rd->kr", kpts, Rg))
        L = np.einsum("rlm,kr->klm", Lam, phase)
        G = np.array([expm(Lk) for Lk in L])
        A = np.einsum("kam,kml->kal", U, G)
        W = k_to_R(kpts, Rlist, A, np.full(nk, 1.0 / nk))
        return A, W, G, L

    def moments(W):
        absw = np.abs(W) ** 2  # (nR, nb, nw)
        N = np.einsum("ran,ra->n", absw, Rdeg[:, None])
        M = np.array([np.einsum("ran,ra->n", absw, qa) for qa in q])
        T = np.array([np.einsum("ran,ra->n", absw, qa2) for qa2 in q2])
        return N, M, T

    def omega(W, per_orbital=False):
        N, M, T = moments(W)
        w = (T / N[None, :] - (M / N[None, :]) ** 2).sum(axis=0)
        return w if per_orbital else w.sum()

    def f(Lam, mu=0.0, pin_reg=1e-8):
        """Total objective and d/dLambda* (frozen mu; see module doc)."""
        Lam = np.asarray(Lam, dtype=complex)
        A, W, G, L = build(Lam)
        N, M, T = moments(W)
        m = M / N[None, :]
        t = T / N[None, :]
        om = (t - m**2).sum()
        # d(omega)/dW*: per alpha
        #   [dT - t dN - 2 m dM + 2 m^2 dN]/N
        D = np.zeros_like(W)
        for a in range(3):
            coef = (q2[a][:, :, None] - t[a][None, None, :]
                    - 2.0 * m[a][None, None, :] * q[a][:, :, None]
                    + 2.0 * m[a][None, None, :] ** 2) / N[None, None, :]
            D += W * coef
        if pin_reg > 0:
            D += 2.0 * pin_reg * (N - 1.0)[None, None, :] * W
        S = np.einsum("kma,kmb->kab", G.conj(), G)
        sign, logdet = np.linalg.slogdet(S)
        if np.any(sign <= 0):
            return 1e30, np.zeros_like(Lam), 1e30, {}
        snn = np.einsum("kaa->ka", S).real
        barrier = float((logdet - np.log(snn).sum(axis=1)).sum())
        # adjoint chain: E_A(k) = (1/nk) sum_R e^{+2 pi i kR} D(R)
        EA = R_to_k(kpts, Rlist, D) / nk
        # dOmega/dG* via U^dag; plus the barrier gradient
        #   -mu G (S^{-1} - diag(1/s_nn))
        EG = np.einsum("kam,kal->kml", U.conj(), EA)
        EG = EG - mu * (np.einsum("kmb,kba->kma", G, np.linalg.inv(S))
                        - G / snn[:, None, :])
        # adjoint-Frechet of exp: X(k) = dexp(L(k)^dag)[EG(k)]
        X = np.array([
            expm_frechet(L[k].conj().T, EG[k], compute_expm=False)
            for k in range(nk)
        ])
        # synthesis L(k) = sum_R e^{+2 pi i kR} Lam(R): plain adjoint
        phase = np.exp(-2j * np.pi * np.einsum("kd,rd->kr", kpts, Rg))
        ELam = np.einsum("kr,klm->rlm", phase, X)
        total = om - mu * barrier + pin_reg * float(((N - 1.0) ** 2).sum())
        return total, ELam, om, dict(N=N, G=G, W=W)

    return f, omega, (nk, nb, nw)


def kdependent_offmesh_penalty(Lam, qpts, Rg, HwannR, Rlist, Rdeg,
                               weight=1.0):
    """Off-mesh interpolation penalty and its Wirtinger gradient.

    At each q in ``qpts`` the transformed pencil eigenvalues are matched
    (sorted) against the orthonormal control spectrum
    ``spec(eigh(H(q)))``; returns ``(penalty, dPenalty/dLambda*)`` with
    the gradient chained per eigenpair
    ``d eps_n = 2 Re[(dG w_n)^dag (H(q) - eps_n) G q w_n]`` through the
    exp adjoint.
    """
    qpts = np.asarray(qpts, dtype=float).reshape(-1, Rg.shape[1])
    phase = np.exp(2j * np.pi * np.einsum("kd,rd->kr", qpts, Rg))
    L = np.einsum("rlm,kr->klm", Lam, phase)
    nwann = Lam.shape[1]
    pen = 0.0
    cL = np.zeros_like(Lam)
    eye = np.eye(nwann)
    for iq, q in enumerate(qpts):
        Gq = expm(L[iq])
        Ho = R_to_k(q[None, :], Rlist, HwannR, Rdeg)[0]
        Hp = Gq.conj().T @ Ho @ Gq
        Sp = Gq.conj().T @ Gq
        eps, W = eigh(Hp, Sp)
        ec = np.sort(eigh(Ho)[0])
        g = eps - ec
        pen += weight * float((g**2).sum())
        c_q = np.zeros_like(Gq)
        for n in range(nwann):
            p_n = (Ho - eps[n] * eye) @ (Gq @ W[:, n])
            c_q += weight * g[n] * np.outer(W[:, n], p_n.conj())
        c_q *= 2.0
        X = expm_frechet(L[iq].conj().T, c_q, compute_expm=False)
        cL += np.einsum("r,lm->rlm", np.exp(-2j * np.pi * q @ Rg.T), X)
    return pen, cL


def optimize_kdependent_gauge(
    wannR, Rlist, Rdeg, positions, kpts, HwannR=None, shells=1, G0=None,
    barrier_weight=1e-2, tikhonov=1e-3, pin_reg=1e-8, maxiter=600,
    tol=1e-12, offmesh_weight=0.0, offmesh_points=None, offmesh_seed=19,
    verbose=False,
):
    """Optimize a smooth k-dependent GL gauge for maximal localization.

    :param wannR: (nR, nbasis, nwann) coefficients of an ORTHONORMAL
        pipeline result (any method);
    :param Rlist: (nR, 3) integer lattice vectors of ``wannR``;
    :param Rdeg: (nR,) image degeneracies (or None);
    :param positions: (nbasis, 3) parent basis positions (fractional or
        Cartesian — the spread units follow; use the same convention as
        ``lwf.wann_centers`` for comparability);
    :param kpts: (nk, 3) the downfolding mesh the model was built from
        (``lwf.kpts``);
    :param HwannR: (nR, nwann, nwann) Hamiltonian of the orthonormal
        model. Required when ``offmesh_weight > 0`` (the off-mesh
        penalty needs the control spectrum);
    :param shells: number of non-constant generator shells on the model
        R-list (1 = R0 + nearest neighbors, default; 0 = the constant-G
        special case);
    :param G0: constant-G initialization (nwann, nwann); default
        identity. ``Lambda(0) = logm(G0)`` when G0 has positive
        eigenvalues, else zeros;
    :param barrier_weight: scale-invariant per-k logdet barrier weight
        (frozen as ``mu = barrier_weight * max(Omega_start, 1)``);
    :param tikhonov: smoothness weight on the non-constant shells;
    :param offmesh_weight: weight of the off-mesh interpolation penalty
        (0 disables it). The penalty is the squared mismatch between
        the transformed pencil eigenvalues and the orthonormal control
        spectrum at ``offmesh_points`` off-mesh points — the
        smoothness-aware term that bounds the gauge-dependent
        interpolation error;
    :param offmesh_points: number of seeded uniform off-mesh points
        (default 6), or an explicit (nq, 3) array;
    :returns: ``(gauge, res, info)`` with ``gauge`` a :class:`KDepGauge`,
        and ``info`` carrying start/optimal spreads, the per-k minimum
        eigenvalue of ``S(k)``, and the off-mesh band errors
        (``offmesh_err_start`` / ``offmesh_err_opt``).
    """
    wannR = np.asarray(wannR, dtype=complex)
    nwann = wannR.shape[2]
    Rg = select_shells(Rlist, max_shell=shells)
    f, omega_fn, (nk, nb, nw) = kdependent_objective(
        wannR, Rlist, Rdeg, positions, kpts, Rg)

    Lam0 = np.zeros((len(Rg), nwann, nwann), dtype=complex)
    if G0 is not None:
        from scipy.linalg import logm

        G0 = np.asarray(G0, dtype=complex)
        if G0.shape != (nwann, nwann):
            raise ValueError(f"G0 must be {(nwann, nwann)}, got {G0.shape}")
        try:
            Lam0[0] = np.real_if_close(logm(G0), tol=1000).astype(complex)
        except Exception:
            Lam0[0] = 0.0
    _, _, start_om, _ = f(Lam0, mu=0.0, pin_reg=0.0)
    mu = barrier_weight * max(start_om, 1.0)
    off = ~np.all(Rg == 0, axis=1)

    # ---- off-mesh interpolation penalty (smoothness proxy) ------------
    # At off-mesh points the transformed pencil no longer reproduces the
    # orthonormal control spectrum; penalizing the sorted-eigenvalue
    # mismatch trades a little spread for interpolation fidelity. The
    # analytic gradient uses d eps_n = 2 Re[(dG w_n)^dag (H(q) - eps_n)
    # G(k) w_n] (S-orthonormal w_n), chained through the exp adjoint.
    if offmesh_weight > 0 and HwannR is None:
        raise ValueError(
            "offmesh_weight > 0 requires HwannR (the orthonormal model "
            "Hamiltonian) for the control spectrum"
        )
    if offmesh_points is None:
        offmesh_points = 6
    if isinstance(offmesh_points, int):
        orng = np.random.default_rng(offmesh_seed)
        qpts = orng.uniform(0.0, 1.0, size=(offmesh_points, kpts.shape[1]))
    else:
        qpts = np.asarray(offmesh_points, dtype=float).reshape(
            -1, kpts.shape[1])
    HwannR = None if HwannR is None else np.asarray(HwannR, dtype=complex)
    control = []
    if HwannR is not None:
        for q in qpts:
            Ho = R_to_k(np.asarray(q)[None, :], Rlist, HwannR, Rdeg)[0]
            control.append(np.sort(eigh(Ho)[0]))

    def offmesh_terms(Lam):
        pen, cL = kdependent_offmesh_penalty(
            Lam, qpts, Rg, HwannR, Rlist, Rdeg, weight=offmesh_weight)
        return pen, cL

    def offmesh_err(Lam):
        phase = np.exp(2j * np.pi * np.einsum("kd,rd->kr", qpts, Rg))
        L = np.einsum("rlm,kr->klm", Lam, phase)
        err = 0.0
        for iq, q in enumerate(qpts):
            Gq = expm(L[iq])
            Ho = R_to_k(np.asarray(q)[None, :], Rlist, HwannR, Rdeg)[0]
            eps = eigh(Gq.conj().T @ Ho @ Gq, Gq.conj().T @ Gq)[0]
            err = max(err, float(np.abs(eps - control[iq]).max()))
        return err

    def unpack(p):
        h = nwann * nwann * len(Rg)
        return (p[:h].reshape(len(Rg), nwann, nwann)
                + 1j * p[h:].reshape(len(Rg), nwann, nwann))

    def fun(p):
        Lam = unpack(p)
        tot, _, _, _ = f(Lam, mu=mu, pin_reg=pin_reg)
        tot = tot + tikhonov * float((np.abs(Lam[off]) ** 2).sum())
        if offmesh_weight > 0:
            pen, _ = offmesh_terms(Lam)
            tot += offmesh_weight * pen
        return float(tot)

    def jac(p):
        Lam = unpack(p)
        _, c, _, _ = f(Lam, mu=mu, pin_reg=pin_reg)
        c = c.copy()
        c[off] += 2 * tikhonov * Lam[off]
        if offmesh_weight > 0:
            _, cL = offmesh_terms(Lam)
            c += cL
        # f real, c = df/dLam*: df/dRe = 2 Re c, df/dIm = +2 Im c
        return np.concatenate([(2 * c.real).ravel(), (2 * c.imag).ravel()])

    p0 = np.concatenate([Lam0.real.ravel(), Lam0.imag.ravel()])
    res = minimize(fun, p0, jac=jac, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": tol, "ftol": tol})
    Lam = unpack(res.x)
    gauge = KDepGauge(Rg, Lam)
    _, _, om_opt, aux = f(Lam, mu=mu, pin_reg=pin_reg)
    S = aux["G"].conj().transpose(0, 2, 1) @ aux["G"]
    min_eig = float(np.min([np.linalg.eigvalsh(Sk).min() for Sk in S]))
    info = {
        "omega_start": start_om,
        "omega_opt": om_opt,
        "min_eig_S": min_eig,
        "mu": mu,
        "Rg": Rg,
    }
    if HwannR is not None:
        info["offmesh_err_start"] = offmesh_err(Lam0)
        info["offmesh_err_opt"] = offmesh_err(Lam)
    if verbose:
        msg = (f"k-dependent gauge ({len(Rg) - 1} generator vectors): "
               f"spread {start_om:.8f} -> {om_opt:.8f}; "
               f"min eig S(k) = {min_eig:.6f}; ")
        if HwannR is not None:
            msg += (f"off-mesh err "
                    f"{info['offmesh_err_start']:.3e} -> "
                    f"{info['offmesh_err_opt']:.3e}; ")
        msg += res.message
        print(msg)
    return gauge, res, info


def apply_kdependent_gauge(lwf, gauge, conditioning_floor=1e-8):
    """Apply a :class:`KDepGauge` to a finished ORTHONORMAL model.

    Returns a new object with $H \\to G^\\dagger H G$,
    $S \\to G^\\dagger G$ (finite range), amplitudes convolved with
    $G(\\mathbf k)$. Mesh-k bands are preserved exactly; off-mesh
    interpolation is gauge-dependent (a warning is emitted). NAC or
    already-non-orthogonal inputs are refused.
    """
    if getattr(lwf, "nac", False):
        raise NotImplementedError(
            "apply_kdependent_gauge: NACLWF (NAC split) is defined for "
            "orthonormal LWFs only"
        )
    if getattr(lwf, "SwannR", None) is not None:
        raise ValueError(
            "apply_kdependent_gauge expects an orthonormal model "
            "(SwannR is None)"
        )
    kpts = getattr(lwf, "kpts", None)
    if kpts is None:
        raise ValueError("the model must carry kpts (lwf.kpts)")
    G = gauge.G_of_k(kpts)  # (nk, nw, nw)
    nwann = lwf.wannR.shape[2]
    if G.shape[1:] != (nwann, nwann):
        raise ValueError(f"G(k) must be ({nwann}, {nwann}), got {G.shape[1:]}")
    min_eig = min(np.linalg.eigvalsh(Gk.conj().T @ Gk).min() for Gk in G)
    if min_eig < conditioning_floor:
        raise ValueError(
            f"conditioning floor violated: min_k lambda_min(S(k)) = "
            f"{min_eig:.3e} < {conditioning_floor:.1e}"
        )
    kw = np.full(len(kpts), 1.0 / len(kpts))
    Wk = R_to_k(kpts, lwf.Rlist, lwf.wannR, lwf.Rdeg)
    Hk = R_to_k(kpts, lwf.Rlist, lwf.HR_total
                if hasattr(lwf, "HR_total") else lwf.HwannR, lwf.Rdeg)
    wannR = k_to_R(kpts, lwf.Rlist,
                   np.einsum("kam,kml->kal", Wk, G), kw)
    HR = k_to_R(kpts, lwf.Rlist,
                np.einsum("kma,kmb,kbl->kal", G.conj(), Hk, G), kw)
    S = k_to_R(kpts, lwf.Rlist,
               np.einsum("kma,kmb->kab", G.conj(), G), kw)
    warnings.warn(
        "k-dependent gauge applied: bands are exact on the downfolding "
        "mesh; off-mesh interpolation is gauge-dependent — consume the "
        "result in real space (R-space-consumed path)",
        stacklevel=2,
    )
    if hasattr(lwf, "HR_total"):  # phonon LWF
        from lawaf.interfaces.phonopy.lwf import LWF

        return LWF(
            factor=lwf.factor,
            Rlist=lwf.Rlist,
            Rdeg=lwf.Rdeg,
            wannR=wannR,
            HR_total=HR,
            SwannR=S,
            kpts=lwf.kpts,
            kweights=lwf.kweights,
            atoms=lwf.atoms,
        )
    from HamiltonIO.lawaf import LawafHamiltonian

    return LawafHamiltonian(
        wannR=wannR,
        HwannR=HR,
        SwannR=S,
        Rlist=lwf.Rlist,
        Rdeg=lwf.Rdeg,
        atoms=lwf.atoms,
        wann_names=getattr(lwf, "wann_names", None),
        kpts=lwf.kpts,
        kweights=getattr(lwf, "kweights", None),
        is_orthogonal=False,
    )
