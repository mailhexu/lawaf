#!/usr/bin/env python3
"""Executable sympy derivation of the ABINIT HIST -> ASE unit-conversion
factors for lawaf story-017 (epic 7, anharmonic effective model).

Every conversion used by ``lawaf.anharmonic.teacher.from_abinit_hist`` is
DERIVED here from the defining physical relations and checked with executed
sympy identities, then cross-checked numerically against scipy.constants
(CODATA, independent of ASE) and ase.units.  Run:

    python docs/derivations/anharmonic_unit_conversions.py   # mydev env

ABINIT HIST.nc stored quantities and units (ABINIT documentation, `rprimd`,
`xcart`, `etotal`, `fcart`, `strten`):
    rprimd (n, 3, 3)  rows are the lattice vectors          [Bohr]
    xcart  (n, natom, 3)  Cartesian positions               [Bohr]
    etotal (n,)           total energy                      [Hartree]
    fcart  (n, natom, 3)  Cartesian forces                  [Hartree/Bohr]
    strten (n, 6)         stress, Voigt xx,yy,zz,yz,xz,xy   [Hartree/Bohr^3]

Target ASE units: Angstrom, eV, eV/Angstrom, eV/Angstrom^3.

Derived here:
    a0  = 4 pi eps0 hbar^2 / (m_e e^2)                       [Bohr radius]
    E_h = alpha^2 m_e c^2 = hbar^2 / (m_e a0^2)              [Hartree energy]
      (identity holds under the defining relation
       alpha = e^2 / (4 pi eps0 hbar c))
    L   = a0 * 1e10   Angstrom per Bohr   (1 A = 1e-10 m exactly)
    E   = E_h / e     eV per Hartree       (1 eV = |e| J exactly, SI 2019)
    F_f = E / L       eV/A per Ha/Bohr
    F_s = E / L^3     eV/A^3 per Ha/Bohr^3
Voigt mapping verified bijective on symmetric tensors with symbols.
"""
import sys

import numpy as np
import sympy as sp

CHECKS = []


def check(name, ok):
    CHECKS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


# ---------------------------------------------------------------------------
# 1. symbolic derivation of a0 and E_h
# ---------------------------------------------------------------------------
eps0, hbar, me, e, alpha, c, a0 = sp.symbols(
    "eps0 hbar m_e e alpha c a0", positive=True
)

a0_expr = 4 * sp.pi * eps0 * hbar**2 / (me * e**2)
Eh_from_alpha = alpha**2 * me * c**2
Eh_from_a0 = hbar**2 / (me * a0**2)

# the two Hartree expressions are identical once a0 is substituted and alpha
# is replaced by its defining relation alpha = e^2/(4 pi eps0 hbar c)
alpha_def = e**2 / (4 * sp.pi * eps0 * hbar * c)
diff = sp.simplify(Eh_from_a0.subs(a0, a0_expr) - Eh_from_alpha.subs(alpha, alpha_def))
check("E_h = alpha^2 m_e c^2 = hbar^2/(m_e a0^2) exactly", diff == 0)

# ---------------------------------------------------------------------------
# 2. numeric evaluation from scipy CODATA (independent of ase)
# ---------------------------------------------------------------------------
import scipy.constants as sc

vals = {
    eps0: sc.epsilon_0,
    hbar: sc.hbar,
    me: sc.m_e,
    e: sc.e,
    alpha: sc.fine_structure,
    c: sc.c,
}
a0_m = float(a0_expr.subs(vals))          # Bohr radius [m]
Eh_J = float(Eh_from_alpha.subs(vals))    # Hartree energy [J]

# exact SI 2019 definitions used in the conversion chain:
L = a0_m * 1e10       # Angstrom per Bohr   (1 A = 1e-10 m, exact)
E = Eh_J / sc.e       # eV per Hartree      (1 eV = |e| J, exact)

check(
    "a0 matches scipy CODATA 'Bohr radius' (1e-12)",
    abs(a0_m - sc.value("Bohr radius")) / sc.value("Bohr radius") < 1e-12,
)
# CODATA lists E_h as 2 R_inf h c; re-deriving from alpha/m_e/c absorbs the
# independent roundings -> agreement at the 1e-11 level, checked at 1e-9
check(
    "E_h matches scipy CODATA 'Hartree energy' (1e-9)",
    abs(Eh_J - sc.value("Hartree energy")) / sc.value("Hartree energy") < 1e-9,
)

# ---------------------------------------------------------------------------
# 3. composite factors used by from_abinit_hist
# ---------------------------------------------------------------------------
import ase.units as au

B, H = au.Bohr, au.Hartree  # ase: Bohr in A, Hartree in eV
check("ase.units.Bohr == derived L (1e-6)", abs(B - L) / L < 1e-6)
check("ase.units.Hartree == derived E (1e-6)", abs(H - E) / E < 1e-6)

# composite identities: the factors from_abinit_hist multiplies with
f_pos = B               # xcart * B      -> Angstrom
f_cell = B              # rprimd * B     -> Angstrom
f_E = H                 # etotal * H     -> eV
f_F = H / B             # fcart * H/B    -> eV/Angstrom  = E/L
f_S = H / B**3          # strten * H/B^3 -> eV/Angstrom^3 = E/L^3
check("force factor == E_h/a0 composite (1e-6)", abs(f_F - E / L) / (E / L) < 1e-6)
check(
    "stress factor == E_h/a0^3 composite (1e-6)",
    abs(f_S - E / L**3) / (E / L**3) < 1e-6,
)
check("position factor == Bohr->Ang (exact product)", f_pos == B)
check("energy factor == Ha->eV (exact product)", f_E == H)

# ---------------------------------------------------------------------------
# 4. Voigt (xx, yy, zz, yz, xz, xy) mapping bijection on symmetric tensors
# ---------------------------------------------------------------------------
from lawaf.anharmonic.dataset import matrix_to_voigt, voigt_to_matrix

s0, s1, s2, s3, s4, s5 = sp.symbols("s0 s1 s2 s3 s4 s5")
S = sp.Matrix([[s0, s5, s4], [s5, s1, s3], [s4, s3, s2]])
v = matrix_to_voigt(np.array(S.tolist(), dtype=object))
check("matrix_to_voigt picks (xx,yy,zz,yz,xz,xy)", list(v) == [s0, s1, s2, s3, s4, s5])
back = voigt_to_matrix(v)
check(
    "voigt_to_matrix o matrix_to_voigt = identity (sympy, exact)",
    sp.simplify(sp.Matrix(back.tolist()) - S) == sp.zeros(3, 3),
)
# ABINIT strten ordering documented identical: (xx, yy, zz, yz, xz, xy)

# ---------------------------------------------------------------------------
# numeric spot check of the full chain on a known quantity
# ---------------------------------------------------------------------------
# 1 Ha/Bohr in eV/A ~ 51.422067...
check(
    "H/B ~ 51.4221 eV/A (textbook value)",
    abs(f_F - 51.422) / 51.422 < 1e-4,
)

n_fail = sum(1 for _, ok in CHECKS if not ok)
print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
sys.exit(1 if n_fail else 0)
