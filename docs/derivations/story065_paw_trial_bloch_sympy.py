"""Story-065 symbolic check: PAW dual-projector trial phase and radial algebra.

The plane-wave ket of a localized trial displaced by tau is
    <exp(i q r) | g(r - tau)> = exp(-i q . tau) * FT(q),
the Bloch image sum carries exp(i k . R) and is therefore periodic under
k -> k + 2pi*(reciprocal integer), and the s-channel Gaussian Fourier pair
    4 pi int j0(qr) g(r) r^2 dr = (2pi)^{3/2} exp(-q^2/2)
matches the defining Cartesian transform used as the test reference.
"""
import sympy as sp

x, y, z, qx, qy, qz, tx, ty, tz = sp.symbols("x y z qx qy qz tx ty tz", real=True)
r, q, tau = sp.symbols("r q tau", positive=True)

# 1. Translation phase: FT of f(r - tau) = exp(-i q.tau) FT(f).
f = sp.exp(-(x**2 + y**2 + z**2) / 2)
shifted = f.subs({x: x - tx, y: y - ty, z: z - tz})
gaussian_ft = (2 * sp.pi) ** sp.Rational(3, 2) * sp.exp(
    -(qx**2 + qy**2 + qz**2) / 2
)
lhs = sp.integrate(
    sp.exp(-sp.I * (qx * x + qy * y + qz * z)) * shifted,
    (x, -sp.oo, sp.oo),
    (y, -sp.oo, sp.oo),
    (z, -sp.oo, sp.oo),
)
assert sp.simplify(lhs - sp.exp(-sp.I * (qx * tx + qy * ty + qz * tz)) * gaussian_ft) == 0

# 2. p-channel sign: <exp(iqr)| z g(r)> = -i q_z (2pi)^{3/2} e^{-q^2/2};
#    the dual-projector ket uses 4pi (-i)^l Y_lm(qhat) int j_l g r^2 dr,
#    whose real-harmonic l=1 slot order (z, x, y) must reproduce this sign.
p_z = z * f
lhs_p = sp.integrate(
    sp.exp(-sp.I * (qx * x + qy * y + qz * z)) * p_z,
    (x, -sp.oo, sp.oo),
    (y, -sp.oo, sp.oo),
    (z, -sp.oo, sp.oo),
)
assert sp.simplify(lhs_p + sp.I * qz * gaussian_ft) == 0

# 3. The spherical-Bessel s-channel integral equals the Cartesian transform.
j0 = sp.sin(q * r) / (q * r)
radial = 4 * sp.pi * sp.integrate(j0 * sp.exp(-r**2 / 2) * r**2, (r, 0, sp.oo))
assert sp.simplify(radial.subs(q, sp.sqrt(qx**2 + qy**2 + qz**2)) - gaussian_ft) == 0

# 4. Bloch periodicity: images R on the integer lattice carry exp(i k . R),
#    so D(k) is invariant under k -> k + 2pi m with integer m: each weight
#    exp(i 2pi m R) equals 1 for every integer pair (m, R).
kx, kx_prime, m = sp.symbols("kx kx_prime m", real=True)
R = sp.symbols("R", integer=True)
phase = sp.exp(sp.I * kx * R)
for m_value in (-2, -1, 1, 3):
    for r_value in range(-3, 4):
        weight = sp.exp(sp.I * 2 * sp.pi * m_value * r_value)
        assert weight == 1, (m_value, r_value)
assert sp.simplify(phase.subs(kx, kx + 2 * sp.pi * m) - phase * sp.exp(sp.I * 2 * sp.pi * m * R)) == 0

# 5. Defining overlap closed form and image weight: for normalized
#    Gaussians, M(d) = int p(x) g(x - d) dx = sqrt(pi) exp(-d^2/4); the
#    image at displacement d + R contributes exactly M(d + R) weighted by
#    exp(i k . R) (concrete integer R witness).
d_x, k_x = sp.symbols("d_x k_x", real=True)
base = sp.integrate(
    sp.exp(-x**2 / 2) * sp.exp(-((x - d_x) ** 2) / 2), (x, -sp.oo, sp.oo)
)
assert sp.simplify(base - sp.sqrt(sp.pi) * sp.exp(-(d_x**2) / 4)) == 0
for r_value in (-2, 1, 3):
    shifted = sp.integrate(
        sp.exp(-x**2 / 2) * sp.exp(-((x - d_x - r_value) ** 2) / 2),
        (x, -sp.oo, sp.oo),
    )
    assert sp.simplify(shifted - sp.sqrt(sp.pi) * sp.exp(-((d_x + r_value) ** 2) / 4)) == 0
    weight = sp.exp(sp.I * k_x * r_value)
    assert sp.simplify(weight * shifted.subs(d_x, sp.Rational(3, 7)) - weight * sp.sqrt(sp.pi) * sp.exp(-((sp.Rational(3, 7) + r_value) ** 2) / 4)) == 0

# 6. Image enumeration bounds: with lattice rows a_j and reciprocal columns
#    b_i of B = 2pi A^{-1} (rows of A, columns of B), (n A) B = 2pi n, so any
#    displacement d = delta + n A with |d| <= cutoff satisfies
#    n_i = (d.b_i - delta.b_i)/2pi and hence
#    |n_i + delta.b_i/2pi| <= cutoff |b_i| / 2pi (Cauchy-Schwarz via the
#    Lagrange identity).  The integer search window is therefore
#    [ceil(-delta.b_i/2pi - r_i), floor(-delta.b_i/2pi + r_i)], r_i =
#    cutoff |b_i|/2pi -- centered on -delta.b_i/2pi, not on zero.
u = sp.Matrix(sp.symbols("u1 u2 u3"))
v = sp.Matrix(sp.symbols("v1 v2 v3"))
lagrange = sp.expand(u.dot(u) * v.dot(v) - u.dot(v) ** 2)
cross_norm = sp.expand((u.cross(v)).dot(u.cross(v)))
assert sp.simplify(lagrange - cross_norm) == 0
A_conc = sp.Matrix([[4, 0, 0], [3, 1, 0], [0, 0, 2]])
B_rows = 2 * sp.pi * A_conc.inv().T  # rows are the reciprocal vectors b_i
for n_conc in [(1, 0, -2), (-3, 1, 2), (0, 0, 0)]:
    lhs = (sp.Matrix([n_conc]) * A_conc) * B_rows.T
    assert sp.simplify(lhs - 2 * sp.pi * sp.Matrix([n_conc])) == sp.zeros(1, 3)
# with d = delta + nA: d.b - delta.b = 2pi n (exact rational witness)
delta_conc = sp.Matrix([sp.Rational(7, 3), sp.Rational(-1, 5), sp.Rational(4, 7)])
shift_conc = sp.Matrix([1, 0, -2])
d_conc = delta_conc + A_conc.T * shift_conc
for i in range(3):
    b_row = B_rows[i, :]
    assert sp.simplify(
        d_conc.dot(b_row) - delta_conc.dot(b_row) - 2 * sp.pi * shift_conc[i]
    ) == 0
# k-phase of an image factors: exp(i k.(delta + nA)) = exp(i k.delta) exp(i k.nA)
kx, Rint = sp.symbols("kx Rint", real=True)
assert sp.simplify(
    sp.exp(sp.I * kx * (d_x + Rint)) - sp.exp(sp.I * kx * d_x) * sp.exp(sp.I * kx * Rint)
) == 0

print("story065 PAW trial phase/Bloch/radial algebra: PASS")
