#!/usr/bin/env python3
"""Executable sympy derivation of the space-group action S_g(q) group law
for lawaf story-014 (anharmonic effective model, Epic 7).

Every claim below is checked by executed sympy code (simplify/expand == 0
style) with a printed PASS/FAIL and a final check count; the script exits
nonzero if any check fails.

Derived target: the BARE phase law implemented in
``lawaf/anharmonic/representation.py`` (story-014),

    [S_g(q)]_{sigma_g(kappa), kappa} = R_g * exp(-2*pi*i * q'_g . t^g_kappa),

satisfies (1) the group law S_g(q_h) S_h(q) = S_{g h}(q) -- including the
nonsymmorphic defect (translation) terms -- (2) unitarity of the
permutation-rotation-phase block structure, and (3) triviality of all phases
at Gamma.  The numeric ground truth for the same conventions is the
story-014 transport oracle (BaTiO3 DM_dip_wang, 1e-13 over 48 ops x 15
bands; see tests/test_anharmonic_representation.py).

Conventions (identical to representation.py; spglib dataset storage):
  Operations (W | w) act on REDUCED coordinates as column maps
      x |--> W x + w,   W integer 3x3 (unimodular), w fractional.
  spglib composition (apply h first, then g):
      (W_g | w_g) o (W_h | w_h) = (W_g W_h | w_g + W_g w_h).
  Atom map / nonsymmorphic defect:
      W_g r_kappa + w_g = r_{sigma_g(kappa)} + t^g_kappa,
      t^g_kappa an integer (lattice) vector; sigma_g is a bijection.
  Star / wavevector image (transpose-inverse action):
      q --g--> q'_g = W_g^{-T} q   (wrapped mod 1 in the implementation).
  Cartesian rotation: R_g = A W_g A^{-1}, A = primitive cell (rows a1,a2,a3).
  Phase law (bare, lawaf gauge): per SOURCE atom kappa of the block
      (row sigma_g(kappa), column kappa),
      phase = exp(-2*pi*i * q'_g . t^g_kappa).

Group-law identity proved here (section 3): with q_h = W_h^{-T} q and
q_gh = W_g^{-T} q_h,
    exponent(S_g(q_h) S_h(q)) = q_gh . t^g_{sigma_h(kappa)} + q_h . t^h_kappa
    exponent(S_{gh}(q))       = q_gh . t^{gh}_kappa
    t^{gh}_kappa              = t^g_{sigma_h(kappa)} + W_g t^h_kappa
  so the difference reduces to
    q_h . t^h_kappa - q_gh . (W_g t^h_kappa)
    = q_h . t^h_kappa - (W_g^T q_gh) . t^h_kappa
    = q_h . t^h_kappa - (W_g^T W_g^{-T} q_h) . t^h_kappa = 0,
  using W_g^T W_g^{-T} = I (per-op unitarity of the transpose-inverse map).

Run:  python story014_space_group_action_sympy.py   (mydev env, sympy >= 1.12)
"""
import sys

import sympy as sp

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


print("=" * 72)
print("story-014 sympy derivation: space-group action S_g(q) group law,")
print("unitarity, Gamma triviality (bare phase law of representation.py)")
print("=" * 72)

# ----------------------------------------------------------------------
# section 0: symbols
# ----------------------------------------------------------------------
# integer rotation matrices as matrix symbols (entries integral by
# construction from the spglib dataset; unimodular, hence invertible)
Wg, Wh = sp.MatrixSymbol("Wg", 3, 3), sp.MatrixSymbol("Wh", 3, 3)
A = sp.MatrixSymbol("A", 3, 3)

# reduced-coordinate vectors (column conventions)
q = sp.MatrixSymbol("q", 3, 1)  # wavevector
qh = Wh.T.inv() * q  # star image of q under h
qgh = Wg.T.inv() * qh  # star image of q_h under g

rk, rsh, rgsh = (sp.MatrixSymbol(s, 3, 1) for s in ("r_k", "r_sh", "r_gsh"))
wg, wh = sp.MatrixSymbol("wg", 3, 1), sp.MatrixSymbol("wh", 3, 1)
t_h, tg_sh, t_gh = (sp.MatrixSymbol(s, 3, 1) for s in ("t_h", "tg_sh", "t_gh"))

# ----------------------------------------------------------------------
# section 1: composition consistency and the defect (translation) law
#
# defining equations (column-vector convention):
#   (E_h)  Wh rk + wh = rsh + t_h
#   (E_g)  Wg rsh + wg = rgsh + tg_sh
#   (E_gh) (Wg Wh) rk + (Wg wh + wg) = rgsh + t_gh   [SAME image atom rgsh:
#                                                    sigma_gh = sigma_g o sigma_h
#                                                    by uniqueness of matching]
# claim: t_gh = tg_sh + Wg t_h   (nonsymmorphic defect composition)
# ----------------------------------------------------------------------
rsh_val = Wh * rk + wh - t_h  # from E_h
rgsh_val = Wg * rsh_val + wg - tg_sh  # from E_g
tgh_val = (Wg * Wh) * rk + (Wg * wh + wg) - rgsh_val  # from E_gh
diff1 = sp.simplify(sp.expand(tgh_val - (tg_sh + Wg * t_h)).as_explicit())
check("section 1: defect composition t_gh = tg_sh + Wg t_h (nonsymmorphic law)",
      diff1 == sp.zeros(3, 1))

# ----------------------------------------------------------------------
# section 2: rotation composition in the Cartesian basis
#
# R_g = A Wg A^{-1} must compose like the integer matrices:
# R_g R_h = A Wg Wh A^{-1} = R_gh.
# ----------------------------------------------------------------------
Rg = A * Wg * A.inv()
Rh = A * Wh * A.inv()
Rgh = A * (Wg * Wh) * A.inv()
diff2 = sp.simplify((Rg * Rh - Rgh).as_explicit())
check("section 2: Cartesian rotation composition Rg Rh = A (Wg Wh) Ainv = Rgh",
      diff2 == sp.zeros(3, 3))

# ----------------------------------------------------------------------
# section 3: the group-law exponent identity (nonsymmorphic terms included)
#
# phase exponents (scalar dot products, factors of -2*pi*i dropped):
#   S_h at q:            e_h   = q_h  . t_h                (source atom kappa)
#   S_g at q_h:          e_g   = q_gh . t_g_{sigma_h(kappa)}
#   product S_g S_h:     e_comp = e_g + e_h
#   S_{gh} at q:         e_gh  = q_gh . t_gh
# claim: e_comp - e_gh = 0.
# ----------------------------------------------------------------------
e_g = (qgh.T * tg_sh)[0, 0]
e_h = (qh.T * t_h)[0, 0]
e_comp = e_g + e_h
tgh_rel = tg_sh + Wg * t_h  # section 1
e_gh = (qgh.T * tgh_rel)[0, 0]

# the one substantive reduction: q_gh . (Wg t_h) = q_h . t_h, i.e.
# (Wg^{-T} q_h)^T Wg t_h = q_h^T t_h  -- a transpose/inverse cancellation.
sec3a = sp.expand((qgh.T * (Wg * t_h) - qh.T * t_h)[0, 0])
check("section 3a: image-map cancellation q_gh.(Wg t) = q_h.t  "
      "(uses Wg^T Wg^{-T} = I)", sp.simplify(sec3a) == 0)

# group law: e_comp - e_gh = q_h.t_h - q_gh.(Wg t_h) = -sec3a exactly, so the
# expanded residual must cancel sec3a identically (decomposition onto 3a).
residual = sp.simplify(sp.expand(e_comp - e_gh) + sec3a)
check("section 3b: GROUP LAW exponent identity "
      "e[S_g(q_h) S_h(q)] = e[S_{gh}(q)]", residual == 0)

# the star map itself is transitive: (Wg Wh)^{-T} q = Wg^{-T} (Wh^{-T} q),
# equivalently ((Wg Wh)^T applied to the two-step image) reproduces q.
two_step = Wg.T.inv() * (Wh.T.inv() * q)
lhs = (Wg * Wh).T * two_step
check("section 3c: star-image transitivity (Wg Wh)^T (Wg^{-T} Wh^{-T} q) = q",
      sp.simplify((lhs - q).as_explicit()) == sp.zeros(3, 1))

# ----------------------------------------------------------------------
# section 4: unitarity of the permutation-rotation-phase block structure
#
# One operation with sigma = atom swap (the nonsymmorphic-interesting case),
# 3x3 Cartesian rotation R (explicit z-rotation with symbolic angle -- any
# orthogonal R works), source phases p_k = exp(-2 pi i q'. t_k):
#   S = [[0, p1 R], [p0 R, 0]]  (6x6, atom-major xyz layout).
# claim: S^dag S = I and S S^dag = I (real rotation + unit-modulus phases).
# ----------------------------------------------------------------------
phi = sp.symbols("phi", real=True)
c, s = sp.cos(phi), sp.sin(phi)
R = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
x0, x1 = sp.symbols("x0 x1", real=True)  # stand-ins for -q'. t_kappa
p0, p1 = sp.exp(2 * sp.pi * sp.I * x0), sp.exp(2 * sp.pi * sp.I * x1)
Z = sp.zeros(3)
S = sp.Matrix(sp.BlockMatrix([[Z, p1 * R], [p0 * R, Z]]))
Id6 = sp.eye(6)
SSd = sp.simplify(sp.expand(S.conjugate().T * S))
check("section 4a: block unitarity S^dag S = I (phases unit-modulus, R orthogonal)",
      SSd == Id6)
SdS = sp.simplify(sp.expand(S * S.conjugate().T))
check("section 4b: block unitarity S S^dag = I", SdS == Id6)

# ----------------------------------------------------------------------
# section 5: Gamma triviality
#
# phase_kappa(q) = exp(-2 pi i (W^{-T} q . t_kappa)); at q = 0 the image
# W^{-T} 0 = 0, so every phase is exp(0) = 1 and S_g(0) is the real
# permutation-rotation.  Checked componentwise with explicit inverse entries.
# ----------------------------------------------------------------------
qsyms = sp.symbols("q0:3", real=True)
tsyms = sp.symbols("t0:3", real=True)
Winv = Wg.T.inv().as_explicit()
dot = sum(Winv[i, j] * qsyms[j] * tsyms[i] for i in range(3) for j in range(3))
phase_gamma = sp.exp(-2 * sp.pi * sp.I * dot).subs({v: 0 for v in qsyms})
check("section 5: Gamma triviality phase(q=0) = exp(0) = 1 for arbitrary W, t",
      sp.simplify(phase_gamma) == 1)

# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
failed = [name for name, ok in CHECKS if not ok]
n = len(CHECKS)
print("-" * 72)
print(f"{n - len(failed)}/{n} symbolic checks passed")
if failed:
    print("FAILED checks:")
    for name in failed:
        print("  -", name)
    sys.exit(1)
print("story-014 sympy derivation: all checks passed")
sys.exit(0)
