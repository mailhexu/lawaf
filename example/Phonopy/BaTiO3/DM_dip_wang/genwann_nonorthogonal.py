"""Non-orthogonal lattice Wannier functions for BaTiO3 (no NAC).

Demonstrates the non-orthogonal LWF path: with ``orthogonal=False`` the
projected gauge is kept raw (no Loewdin/polar orthonormalization), the
overlap ``S^w(q) = A(q)^dag A(q)`` is propagated to real space
(``SwannR``, Wigner-Seitz folded together with H and the amplitudes), and
the phonon bands come from the generalized pencil ``(D^w(q), S^w(q))``.

Run from this directory (needs phonopy_params.yaml):
    python genwann_nonorthogonal.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lawaf import PhonopyDownfolder

FNAME = "phonopy_params.yaml"

PARAMS = dict(
    method="projected",
    nwann=3,
    # the three soft Ti-dominated modes at Gamma as anchor projectors
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    use_proj=True,
    # unity weights: the raw (non-orthonormalized) projected gauge needs
    # full-rank columns at every q; energy/window weights zero out rows
    # and are designed to be undone by the polar orthonormalization
    # (that is the orthogonal=True path; SCDM-k always orthonormalizes).
    weight_func="unity",
    kmesh=(2, 2, 2),
    gamma=True,
)

KVECTORS = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5],
     [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)
KNAMES = [r"$\Gamma$", "X", "M", "R", "X", r"$\Gamma$", "R"]


def run(orthogonal):
    downfolder = PhonopyDownfolder(
        phonopy_yaml=FNAME,
        mode="DM",
        params=dict(PARAMS, orthogonal=orthogonal),
        symmetrize_fc=False,
        is_nac=False,
    )
    lwf = downfolder.downfold(
        output_path=f"nonorthogonal_{orthogonal}/",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
    )
    return downfolder, lwf


def run_gauge_transform():
    """Non-orthogonal LWFs FROM a disentangled selection: run the standard
    orthonormal pipeline (window_bands hand-selection), then apply a
    constant full-rank GL factor G. The overlap becomes onsite-only
    (G^dag G at R=0) and the pencil stays exact at every k
    (congruence invariance; here with use_ws_distance=False so the
    equivalence to the orthonormal control is machine-exact off-mesh
    too). With the default WS folding the equivalence is exact on-mesh
    and carries the usual small off-mesh gauge sensitivity."""
    rng = np.random.default_rng(9)
    G = np.eye(3) + 0.25 * (
        rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    )
    wb = {
        (0.0, 0.0, 0.0): (0, 1, 2),
        (0.5, 0.0, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.5): (0, 1, 2),
    }
    res = {}
    for tag, extra in [("selection_ctl", {}), ("selection_G",
                                               dict(nonorthogonal_gauge=G))]:
        df = PhonopyDownfolder(
            phonopy_yaml=FNAME,
            mode="DM",
            params=dict(PARAMS, use_ws_distance=False,
                        window_bands=dict(wb), **extra),
            symmetrize_fc=False,
            is_nac=False,
        )
        res[tag] = (df, df.downfold(output_path=f"gauge_{tag}/",
                                    write_hr_nc="LWF.nc",
                                    write_hr_txt="LWF.txt"))
    df, lwf = res["selection_G"]
    print("=" * 70)
    print("GL gauge transform on a disentangled selection:")
    print("  SwannR present:", lwf.SwannR is not None,
          "(control:", res["selection_ctl"][1].SwannR is not None, ")")
    i0 = int(np.where(np.all(lwf.Rlist == 0, axis=1))[0][0])
    print("  max |S(R!=0)| =", np.abs(np.delete(lwf.SwannR, i0, axis=0)).max())
    print("  ||S(0) - G^dag G|| =",
          np.linalg.norm(lwf.SwannR[i0] - G.conj().T @ G))
    q = rng.uniform(-0.5, 0.5, size=(5, 3))
    e_c = np.array([res["selection_ctl"][1].solve_k(x)[0] for x in q])
    e_g = np.array([lwf.solve_k(x)[0] for x in q])
    print(f"  selection control vs G-transformed bands at random q: "
          f"max|diff| = {np.abs(e_c - e_g).max():.2e} (pencil-exact)")
    assert np.abs(e_c - e_g).max() < 1e-10


def run_optimized_gauge():
    """Maximally localized NON-ORTHOGONAL LWFs: optimize the constant GL
    factor G to minimize the normalized per-orbital spread (the GL
    generalization of the MV objective; the orthonormal gauge is a
    stationary point, so gains come from the non-unitary directions).
    A logdet barrier keeps the set invertible - without it the spread
    infimum is degenerate (columns collapse onto the single
    best-localized function). Bands are pencil-exact throughout."""
    from lawaf import optimize_nonorthogonal_gauge
    from lawaf.wannierization.nonorthogonal_gauge import (
        apply_gauge_transform,
        nonorthogonal_spread,
        position_moment_matrices,
    )

    df = PhonopyDownfolder(
        phonopy_yaml=FNAME,
        mode="DM",
        params=dict(PARAMS, use_ws_distance=False),
        symmetrize_fc=False,
        is_nac=False,
    )
    lwf = df.downfold(output_path="gauge_optimized/", write_hr_nc=None,
                      write_hr_txt=None)
    basis_pos = np.repeat(lwf.atoms.get_scaled_positions(), 3, axis=0)
    x, y = position_moment_matrices(lwf.wannR, lwf.Rlist, lwf.Rdeg, basis_pos)
    G, res = optimize_nonorthogonal_gauge(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                          basis_pos)
    o0 = nonorthogonal_spread(np.eye(3), x, y)
    o1 = nonorthogonal_spread(G, x, y)
    print("=" * 70)
    print("optimized non-orthogonal gauge:")
    print(f"  spread (fractional units^2): {o0:.8f} -> {o1:.8f}")
    S = G.conj().T @ G
    print("  S = G^dag G off-diagonal magnitude:",
          np.abs(S - np.diag(np.diag(S))).max().round(6))
    print("  (symmetry-related triplets are near GL-optimal; large gains")
    print("   appear when the orthonormal gauge is symmetry-pinned, e.g.")
    print("   bonding/antibonding-like pairs)")
    lwf_no = apply_gauge_transform(lwf, G)
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.5, 0.5, size=(4, 3))
    e0 = np.array([lwf.solve_k(k)[0] for k in q])
    e1 = np.array([lwf_no.solve_k(k)[0] for k in q])
    print(f"  band preservation max|diff| = {np.abs(e0 - e1).max():.2e}")
    assert o1 <= o0 + 1e-12
    assert np.abs(e0 - e1).max() < 1e-10
    return lwf_no



def main():
    df_no, lwf_no = run(False)
    df_or, lwf_or = run(True)

    print("=" * 70)
    print("non-orthogonal run: SwannR present:", lwf_no.SwannR is not None)
    print("orthogonal run:    SwannR present:", lwf_or.SwannR is not None)
    S0 = lwf_no.get_Sk(np.zeros(3))
    w_S = np.linalg.eigvalsh(S0)
    print(f"S(Gamma) eigenvalues: {w_S.round(6)} (anchor gauge: orthonormal)")
    q_generic = np.array([0.25, 0.11, -0.37])
    Sq = lwf_no.get_Sk(q_generic)
    w_q = np.linalg.eigvalsh(Sq)
    print(f"S({q_generic}) eigenvalues: {w_q.round(6)}  <- non-trivial metric")
    assert np.all(w_q > 1e-6), "SwannR must stay positive definite"

    # On the downfolding mesh the pencil and the orthonormal gauge are the
    # same subspace -> identical Rayleigh-Ritz values (exact invariant).
    e_no = np.array([lwf_no.solve_k(k)[0] for k in df_no.kpts])
    e_or = np.array([lwf_or.solve_k(k)[0] for k in df_or.kpts])
    onmesh = np.abs(e_no - e_or).max()
    print(f"on-mesh max |nonortho - ortho| bands: {onmesh:.3e}")
    assert onmesh < 1e-8

    # off-mesh, the raw gauge trades k-smoothness (interpolation quality)
    # for per-function character; quantify against the orthogonal model
    qtest = np.array([[0.21, 0.13, 0.07], [0.4, -0.3, 0.15]])
    e_no_q = np.array([lwf_no.solve_k(q)[0] for q in qtest])
    e_or_q = np.array([lwf_or.solve_k(q)[0] for q in qtest])
    print(
        "off-mesh max |nonortho - ortho| bands: "
        f"{np.abs(e_no_q - e_or_q).max():.3e} (R-space truncation gauge cost)"
    )

    # netcdf round trip preserves the overlap
    from lawaf.interfaces.phonopy.lwf import LWF

    lwf_rt = LWF.load_from_netcdf("nonorthogonal_False/LWF.nc")
    assert lwf_rt.SwannR is not None
    assert np.abs(lwf_rt.SwannR - lwf_no.SwannR).max() < 1e-10
    print("netcdf round trip preserves SwannR")

    # band structure comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (df, lwf, title) in zip(
        axes,
        [(df_or, lwf_or, "orthogonal gauge"),
         (df_no, lwf_no, "non-orthogonal gauge")],
    ):
        df.plot_band_fitting(
            kvectors=KVECTORS,
            knames=KNAMES,
            npoints=60,
            unit_factor=15.6 * 33.6,
            ylabel="Frequency (cm^-1)",
            evals_to_freq=True,
            show=False,
            ax=ax,
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("LWF_BTO_nonorthogonal.png", dpi=150)
    print("wrote LWF_BTO_nonorthogonal.png")
    run_gauge_transform()
    run_optimized_gauge()
    print("=" * 70)
    print("PHONON NON-ORTHOGONAL EXAMPLE PASSED")


if __name__ == "__main__":
    main()
