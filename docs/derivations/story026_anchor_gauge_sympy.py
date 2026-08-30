"""Story-026 symbolic check: equal projective factors cancel in Reynolds.

For M_g M_h = omega(g,h) M_gh and D_g D_h = omega(g,h) D_gh,
the conjugation action R_h(U)=M_h U D_h^dag is an ordinary group action.
"""
import sympy as sp

I = sp.I
# Z2 with the nontrivial projective factor omega(a,a)=-1.
M = {0: sp.eye(2), 1: I * sp.eye(2)}
D = {0: sp.eye(2), 1: I * sp.eye(2)}
compose = lambda g, h: (g + h) % 2
u11, u12, u21, u22 = sp.symbols("u11 u12 u21 u22")
U = sp.Matrix([[u11, u12], [u21, u22]])

for g in M:
    for h in M:
        lhs = sp.simplify(M[g] * M[h] * U * D[h].H * D[g].H)
        rhs = sp.simplify(M[compose(g, h)] * U * D[compose(g, h)].H)
        assert lhs == rhs, (g, h, lhs, rhs)

P = sum((M[h] * U * D[h].H for h in M), sp.zeros(2)) / 2
assert sp.simplify(P - U) == sp.zeros(2)
assert sp.simplify(
    sum((M[h] * P * D[h].H for h in M), sp.zeros(2)) / 2 - P
) == sp.zeros(2)

# The character projector used to select each reducible E+A carrier.
R = sp.diag(1, -1)
P_plus = (sp.eye(2) + R) / 2
P_minus = (sp.eye(2) - R) / 2
assert sp.simplify(P_plus * P_plus - P_plus) == sp.zeros(2)
assert sp.simplify(P_minus * P_minus - P_minus) == sp.zeros(2)
assert sp.simplify(P_plus * P_minus) == sp.zeros(2)
assert sp.simplify(R * P_plus - P_plus) == sp.zeros(2)
print("story026 projective Reynolds cancellation: PASS")
