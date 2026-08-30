# MLWF dΩ/dU derivation record (story-009)

Source of truth: [`mlwf_domega_derivation.py`](mlwf_domega_derivation.py) —
executable sympy; **37/37 identity checks PASS** (exit 0, mydev env,
sympy 1.14). The output below is the script's verbatim record. Every
equation stories 006/007 will implement appears here; implementation PRs
cite the check names below.

Inputs are the MV97 discrete definitions
(`⟨r⟩_n = −(1/Nk)Σ_{k,j} w_j b_j θ̃`, `⟨r²⟩_n = (1/Nk)Σ w_j(1−|M̃_nn|²+θ̃²)`);
section 1 derives the Ω_I/Ω_D/Ω_OD decomposition from
`Ω_n = ⟨r²⟩_n − |⟨r⟩_n|²` by executed algebra (completion of the square
under the neighbor-weight constraint `Σ_j w_j b_{jα}b_{jβ} = δ_{αβ}`).

w90 ground truth: `wannier90-3.1.0/src/wannierise.F90` — `wann_omega`
(lines 1713–1984) and `wann_domega` (lines 1987–2210).

## Script output (verbatim)

```text

========================================================================
1. From Omega_n = <r^2>_n - |<r>_n|^2 (MV97 discrete definitions) to Omega_I + Omega_D + Omega_OD
========================================================================
[PASS] 1a'(1D) same with explicit 1-|M_nn|^2 entries
[PASS] 1a(3D) completion of square on the 6-neighbor star
[PASS] 1b N-sum_n|nn|^2 = (N-Tr M^dag M)+sum_{m!=n}|mn|^2
[PASS] 1c |1 - e^{-i arg z} z|^2 = (1-|z|)^2  (w90 r2ave entry algebra)

========================================================================
2. Gauge dependence of Omega_I / Omega_OD
========================================================================
[PASS] 2a Omega_I invariant under independent U_k, U_{k+b}
[PASS] 2b Omega_OD invariant under independent diagonal phase gauges D_k, D_{k+b}
[PASS] 2b Omega_OD changes under a general (real) rotation : delta = -64*sin(1/3)**2/25 - 9*cos(4/3)/50 + 64*sin(1/3)**4/25 + 9/50
[PASS] 2c (per-k, fixed rbar) optimal phase = sum_j w_j(tau_j + b_j rbar) / sum_j w_j

========================================================================
3. Gradient: delta Omega under U_k -> U_k e^{i eps A_k} (A_k Hermitian generator; eps series to first order)
========================================================================
d(Nk Omega)/deps|_0 is linear in the 12 generator parameters; record:
     d/da0_d1 (Nk*Omega) = -2*atan(1/5) + 2*atan(42)
     d/da0_d2 (Nk*Omega) = -2*atan(15/7) + 2*atan(3)
     d/da0_xr (Nk*Omega) = -37963*atan(15/7)/10275 - 299986*atan(1/5)/619515 + 114661*atan(2/5)/619515 + 414647*atan(42)/619515 + 87023*atan(35/36)/41100 + 64829*atan(3)/41100 + 3279965783*pi/1697471100 + 10991/630
     d/da0_xi (Nk*Omega) = -1177/30 - 997129*atan(3)/123300 - 77579*atan(42)/619515 + 226762*atan(1/5)/619515 + 149183*atan(2/5)/619515 + 90377*atan(35/36)/123300 + 835458739*pi/1697471100 + 226688*atan(15/7)/30825
     d/da1_d1 (Nk*Omega) = -2*pi + 2*atan(1/5) + 2*atan(2/5)
     d/da1_d2 (Nk*Omega) = -2*pi - 2*atan(35/36) + 2*atan(15/7)
     d/da1_xr (Nk*Omega) = -105803630467*pi/14062369932 - 224436815*atan(35/36)/37300716 - 2968*atan(42)/3393 + 8081*atan(1/5)/3393 + 5113*atan(2/5)/3393 + 78733225*atan(3)/37300716 + 72851795*atan(15/7)/18650358 + 155033/7560
     d/da1_xi (Nk*Omega) = -5323855*atan(15/7)/2072262 - 9324185*atan(3)/4144524 - 4603*atan(42)/6786 - 3646*atan(2/5)/3393 - 2689*atan(1/5)/6786 + 2039/756 + 19971895*atan(35/36)/4144524 + 27625191413*pi/4687456644
     d/da2_d1 (Nk*Omega) = -2*atan(42) - 2*atan(2/5) + 2*pi
     d/da2_d2 (Nk*Omega) = -2*atan(3) + 2*atan(35/36) + 2*pi
     d/da2_xr (Nk*Omega) = -7969/135 - 1606444*atan(42)/255925 - 287309*atan(3)/75630 - 3133022*atan(2/5)/255925 - 1526578*atan(1/5)/255925 + 105967*atan(15/7)/75630 + 90671*atan(35/36)/37815 + 28336040521*pi/1935560775
     d/da2_xi (Nk*Omega) = -7703/360 - 503018*atan(42)/255925 - 1336984*atan(2/5)/255925 - 178762*atan(3)/113445 - 833966*atan(1/5)/255925 + 85256*atan(35/36)/113445 + 93506*atan(15/7)/113445 + 34698658336*pi/5806682325
[PASS] 3a d(Nk Omega)/deps|_0 == sum_k Tr[ A_k . 2i (g_k - g_k^dag) ]  (g_k = w90 cdodq terms BEFORE the final 4/Nk factor)
[PASS] 3b d(Omega)/deps|_0 == sum_k Tr[ A_k . (i/2)(G_k - G_k^dag) ],  G_k = w90 cdodq INCLUDING final 4/Nk
   descent step (normalised): A_k = -(i/2)(G_k - G_k^dag); update U_k <- U_k e^{i eps A_k}

========================================================================
4. Continuum limit: why the Mmn form is the right discretisation
========================================================================
[PASS] 4a 1-|M_nn|^2 = -(alpha^2+beta) b^2 + O(b^3)
[PASS] 4b theta~ = alpha b + gamma b^2/2 + O(b^3)
[PASS] 4c(i) normalisation: d/db <u(b)|u(b)> = 0
[PASS] 4c(ii) <u'|u'> + Re<u|u''> = 0  (=> beta = -<u'|u'>)
[PASS] 4c(iii) 1-|<u_0|u_b>|^2 = b^2(<u'|u'>-alpha^2) + O(b^3)  (alpha = -<r> in the periodic gauge, MV97)
[PASS] 4d(i) counterexample: e^{ikR}-phase construction, linear moment != Mmn rbar (=0 on the ring) : 2*sqrt(3)*(-sin(16/15) - sin(1/3) + sin(7/5))/9
[PASS] 4d(ii) counterexample: R-space linear moment != Mmn rbar (=0 on the ring; M-phase + b-phase construction) : -(cos(pi/3 + 16/15) + cos(1/3 + pi/3) + sin(pi/6 + 7/5))**2/9 + (cos(pi/3 + 7/5) + sin(1/3 + pi/6) + sin(pi/6 + 16/15))**2/9

========================================================================
5. Analytic oracles
========================================================================
[PASS] 5a(Nk=4) theta~_+ = b/2, theta~_- = -b/2 (identity gauge)
[PASS] 5a(Nk=4) rbar = -1/2 (band centre at the second site) : -1/2
[PASS] 5a(Nk=4) Omega_D = sum w(theta~ + b.rbar)^2 == 0
[PASS] 5a(Nk=4) Omega_I == Nk^2/(4 pi^2) sin^2(pi/Nk)  =>  Omega = Omega_I
[PASS] 5a(Nk=8) theta~_+ = b/2, theta~_- = -b/2 (identity gauge)
[PASS] 5a(Nk=8) rbar = -1/2 (band centre at the second site) : -1/2
[PASS] 5a(Nk=8) Omega_D = sum w(theta~ + b.rbar)^2 == 0
[PASS] 5a(Nk=8) Omega_I == Nk^2/(4 pi^2) sin^2(pi/Nk)  =>  Omega = Omega_I
[PASS] 5a(symbolic) lim_{Nk->inf} Nk^2/(4 pi^2) sin^2(pi/Nk) = 1/4
[PASS] 5b(Nk=4) full two-band space: Omega_I == 0
[PASS] 5b(Nk=4) Omega_OD == 2 Nk^2/(4 pi^2) sin^2(pi/Nk)
    Omega_D(principal branch, sheet-free) = 2.0000000 -- branch-dependent, not a claimed equation
[PASS] 5b(Nk=4) full-U gauge V_k=U_k^dag: M~ = I exactly (Omega_OD = 0, atomic basis)
[PASS] 5b(Nk=8) full two-band space: Omega_I == 0
[PASS] 5b(Nk=8) Omega_OD == 2 Nk^2/(4 pi^2) sin^2(pi/Nk)
    Omega_D(principal branch, sheet-free) = 4.0000000 -- branch-dependent, not a claimed equation
[PASS] 5b(Nk=8) full-U gauge V_k=U_k^dag: M~ = I exactly (Omega_OD = 0, atomic basis)
[PASS] 5c two-orbital model Omega_I == 0 (full 2-orbital space, M unitary)
[PASS] 5c Omega_OD == (1-cos b)/b^2 = 2 Nk^2/(4 pi^2) sin^2(pi/Nk)
[PASS] 5c identity-gauge Omega_D == 0 (theta~ = +-b/2, rbar=-1/2 per band)
[PASS] 5c optimal gauge V_k=U_k^dag (atomic basis): Omega_OD == 0
[PASS] 5c optimal gauge: Omega_D == 0  =>  Omega = 0 (atomic Wannier)

========================================================================
SUMMARY
========================================================================

37/37 checks passed
```

## Derived equations (check names are the citation anchors)

| Check | Equation (lawaf implementation contract) |
|---|---|
| 1a/1a' | `Ω_n = Σ_j w_j(1−|M̃_nn|²) + Σ_j w_j(θ̃_n + b_j·r̄_n)²` — completion of the square from `Ω_n = ⟨r²⟩_n − |⟨r⟩_n|²`, **iff** `Σ_j w_j b_{jα}b_{jβ} = δ_{αβ}` (executed in 1D and on the 3D 6-neighbor star; the constraint is load-bearing) |
| 1b | `N − Σ_n|M̃_nn|² = (N − Tr M̃†M̃) + Σ_{m≠n}|M̃_mn|²` — band sum gives `Σ_n Ω_n = Ω_I + Ω_D + Ω_OD` |
| 1c | with the sheet gauge, `|1 − e^{−i arg z}z|² = (1−|z|)²` (w90 r2ave entry algebra) |
| 2a | `Ω_I = (1/Nk)Σ_{k,j} w_j (N − Tr M̃†M̃)` invariant under **independent** per-k unitaries `U_k, U_{k+b}` |
| 2b | `Ω_OD` invariant under diagonal phase gauges only; changes under full unitaries (executed) |
| 2c | per-k fixed-r̄ subproblem (one descent sweep): optimal phase `φ*_kn = mean_n θ̃-shift` = `Σ_j w_j(θ̃+b·r̄_n)/Σ_j w_j` |
| 3a | `d(Nk·Ω)/dε|₀ = Σ_k Tr[A_k·2i(g_k−g_k†)]` for the right action `U_k ← U_k e^{iεA_k}` (A_k Hermitian); g_k = w90 cdodq terms before the final `4/Nk` |
| 3b | normalized: `dΩ/dε|₀ = Σ_k Tr[A_k·(i/2)(G_k−G_k†)]` with G_k = full w90 cdodq (incl. `4/Nk`, line 2188); descent `A_k = −(i/2)(G_k−G_k†)` |
| 4a/4b | per-k continuum: `1−|M̃_nn|² = b²(⟨u′|u′⟩_k − α_k²)`, `θ̃ = −b⟨r⟩_k + O(b²)` — the Mmn form is the variance-correct discretization |
| 4c | executed on an explicit normalized family: `⟨u′|u′⟩ = −Re⟨u|u″⟩` (β identification) and `1−|⟨u₀|u_b⟩|² = b²(⟨u′|u′⟩−α²)+O(b³)` with `⟨u|u′⟩ = iα` |
| 4d | executed finite-mesh counterexamples: R-space constructions from M̃ phases alone fail to reproduce the Mmn linear moments (see Finding 1) |
| 5a | single isolated band of the xy-model: `θ̃_± = ±b/2`, `r̄ = −1/2`, `Ω_D = 0` (identity gauge), `Ω = Ω_I = Nk²/(4π²)·sin²(π/Nk)` → 1/4 |
| 5b | full two-band space (real chain): `Ω_I = 0` exactly (M̃ unitary), `Ω_OD = 2·Nk²/(4π²)sin²(π/Nk)`; full-U gauge `V_k = U_k†` → `M̃ = I` (atomic basis, executed) |
| 5c | xy-model full space: `Ω_I = 0`, `Ω_OD = (1−cos b)/b²`, identity-gauge `Ω_D = 0`; full-U gauge → `Ω = 0` (atomic site Wannier, executed) |

## w90 term-mapping table (item 6)

| Derived quantity | wannierise.F90 location | w90 expression | Match |
|---|---|---|---|
| `θ̃_n^{k,b}` (spread, w OUTSIDE) | 1749–1750 | `ln_tmp = aimag(log(csheet·M_nn)) − sheet` | ✓ identical |
| `θ̃` (gradient, w INSIDE) | 2038–2039 | same × `wb(nn)` | ✓ convention diff documented: w90's domega `ln_tmp` includes `wb`, omega's does not — lawaf must apply the weight once |
| `r̄_n` (omega) | 1755–1770 | `rave = −Σ wb·bk·ln_tmp /Nk` (`ln_tmp` EXCLUDES wb) | ✓ identical |
| `r̄_n` (domega) | 2045–2057 | `rave = −Σ bk·ln_tmp /Nk` — NO extra `wb`: it is inside `ln_tmp` (2038) | ✓ identical once the wb split is respected |
| `⟨r²⟩_n` | 1785–1798 | `r2ave = Σ wb(1 − |M_nn|² + ln_tmp²)/Nk` | ✓ = checks 1b/4a |
| `Ω_I` | 1918–1937 | `om_i = Σ wb(N − Σ_{nm}|M_nm|²)/Nk` | ✓ = check 2a form (`Tr M†M = Σ_{nm}|M|²`) |
| `Ω_OD` | 1943–1959 | `om_od = Σ wb Σ_{m≠n}|M_nm|²/Nk` | ✓ = check 1b/2b |
| `Ω_D` | 1961–1975 | `om_d = Σ wb(ln_tmp + b·rave)²/Nk` | ✓ = check 1a |
| `cr = M_mn conj(M_nn)` | 2091–2094 | `cr(:,n) = m_matrix(:,n)·conj(mnn)` | ✓ (Ω_OD gradient piece) |
| `crt = M_mn/M_nn` | 2093 | `crt(:,n) = m_matrix(:,n)/mnn` | ✓ (Ω_D gradient piece) |
| `rnkb = b·r̄_n` | 2074–2082 | `rnkb = Σ bk·rave` | ✓ (center correction) |
| `G (cdodq)` | 2168–2188 | `G = 4/Nk Σ { wb(cr−cr†)/2 + (i/2)(crt·ln_tmp + h.c.) + (i/2)wb(crt·rnkb + h.c.) }` | ✓ checks 3a/3b verify `δΩ = Σ Tr(A·2i(g−g†))` and the normalized form against this transcription **exactly** |
| update construction | internal_new_u_and_m 1204–1310 | `tmp_cdq = i·cdq` (line 1226) where `cdq` is the selected CG/SD search direction, diagonalised (ZHEEV at 1228, Schur fallback); right update `u_matrix·cdq` (1288–1290); `M ← cdq† M cdq` (1298–1302) | ✓ confirms the right-action convention `U_k ← U_k e^{iεA_k}`; lawaf v1 (ADR-002): steepest `A_k = −(i/2)(G−G†)` |
| search direction | internal_search_direction 971–1149 | w90 CG on cdodq | lawaf v1: plain steepest descent — same fixed point; check 3 proves **infinitesimal** descent (dΩ/dε < 0 along the descent direction when G ≠ 0); monotonicity of a finite step needs a line-search/backtracking acceptance test (FR-005), to be implemented in story-006 |

**MV97 vs w90 naming**: MV's `Ω_OD` = w90's `om_d + om_od`; MV's `Ω_I` = w90's `om_i`. lawaf reports the w90 triple (matches `.wout` output).

## Findings (documented, resolved — not papered over)

1. **Finite-mesh R-space sums are NOT exactly the Mmn form.** Two executed
   counterexamples (Nk=3 ring, checks 4d(i)/4d(ii)): Fourier constructions
   from `e^{ikR}·gauge phases` and from `e^{iθ̃}e^{±ib·R}` (M-phase +
   b-phase) fail to reproduce the Mmn **linear moments** (`Σ_R|C(R)|²` is
   preserved by DFT Parseval — the mismatch is in the moments, not the
   norm). Resolution: the Mmn/θ̃² form **is** the definition of the
   discrete spread (MV97; w90 computes r2ave/om_* from Mmn only). Consequence for lawaf: the optimizer (ADR-002) uses the Mmn form, as does
   w90 for ALL reported moments (r2ave/om_*, lines 1785–1975). This
   **conflicts with architecture-mlwf ADR-004**, which mandates reporting
   Ω_I via Rdeg-weighted R-space Fourier sums. **Adjudicated at the
   story-009 gate (user, 2026-08-29): ADR-004 amended — reported
   Ω_I/Ω_D/Ω_OD use the Mmn form (wannier90-identical); R-space
   quantities are diagnostics only.**
2. **w90 weight-convention split**: `wann_omega`'s `ln_tmp` excludes `wb`,
   `wann_domega`'s includes it (lines 1749 vs 2038); domega's `rave`
   (2051) correspondingly carries no extra `wb`. A transcription that
   mixes them double-counts `wb`. Checks 3a/3b use the domega convention;
   lawaf code should carry `wb` exactly once.
3. **Single-k gradient toys are degenerate**: on a toy where r̄ moves with a
   single k's phases, the diagonal-phase gradient vanishes identically
   (self-consistent average). The exact identity (checks 3a/3b) requires a
   consistent mesh with `M^{k+b,−b} = (M^{k,b})†` and r̄ fixed within a
   sweep (w90 recomputes rave per sweep) — this is the lawaf contract.
4. **Neighborhood-weight normalization is load-bearing** (check 1a):
   `Σ_j w_j b_{jα} b_{jβ} = δ_{αβ}` must hold exactly or Ω_D is wrong.
   lawaf's `kmesh_nnlist` port provides the w90 b-vectors; weights must
   reproduce w90's `wb` normalization.
5. **Principal-branch θ̃ is not a physical spread**: without w90's
   `sheet/csheet` branch tracking, identity-gauge `Ω_D` values (e.g. 2.0,
   4.0 in 5b) are branch artifacts. Only branch-independent quantities
   (`Ω_I`, `Ω_OD`) and gauge-executed results are claimed as equations;
   the oracles in 5a/5c use symmetric exact phases where θ̃ is analytic.

## Oracle model definitions (for stories 006/007/008 tests)

- **xy-plane winding model** (two-site chain, `H(k) = cos k·σ_x + sin k·σ_y`,
  `u_±k = (1, ±e^{ik})/√2`): single-band subspace {+}:
  `θ̃_± = ±b/2`, `r̄ = −1/2`, `Ω_D = 0` (identity gauge — already
  symmetric), `Ω = Ω_I = Nk²/(4π²)sin²(π/Nk)` (→ 1/4 as Nk→∞) — the
  convergence oracle for an isolated band. Full two-band space (5c):
  `Ω_I = 0`, `Ω_OD = (1−cos b)/b²`, identity `Ω_D = 0`; full-U atomic
  gauge `V_k = U_k†` → `Ω = 0` — the clean absolute-minimum oracle.
- **xz-plane real chain** (`H(k) = cos k·σ_z + sin k·σ_x`, real
  eigenvectors): full two-band space (5b): `Ω_I = 0` exactly,
  `Ω_OD = 2·Nk²/(4π²)sin²(π/Nk)`; full-U gauge → `M̃ = I` (atomic basis).
  Single-band subspace has real overlaps (no phase winding) — not used as
  an oracle.
