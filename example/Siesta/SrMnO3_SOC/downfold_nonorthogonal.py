"""Non-orthogonal Wannier functions from a raw Siesta NAO Hamiltonian.

SrMnO3 spinor (SOC) calculation, Mn-3d + O-2p window. With
``orthogonal=False`` the LCAO basis overlap S(k) is carried through the
whole downfolding: the projection uses the dual bra ``g^dag S``, the raw
(non-orthonormalized) gauge is kept, the overlap
``S^w(k) = (psi A)^dag S (psi A) = A^dag A`` is Fourier transformed to
``SwannR`` (WS-folded with H and the amplitudes), and interpolated bands
come from the generalized pencil ``(H^w(k), S^w(k))``.

Run from this directory (needs siesta.fdf + siesta.nc):
    python downfold_nonorthogonal.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lawaf.interfaces import SiestaDownfolder

PARAMS = dict(
    method="projected",
    kmesh=[2, 2, 2],
    weight_func="unity",
    use_proj=False,
    selected_orbdict={"Mn": ["3d"], "O": ["2p"]},
    enhance_Amn=0,
    # unity weights: the raw non-orthonormalized gauge needs full-rank
    # projections at every k (energy/window weights are designed to be
    # undone by the polar orthonormalization of the orthogonal path)
    # orthogonal=False is the point of this example; the orthogonal=True
    # control run below reproduces the classic lawaf behaviour
)

KVECTORS = np.array(
    [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
)
KNAMES = [r"$\Gamma$", "X", "M", r"$\Gamma$", "R"]


def run(orthogonal):
    downfolder = SiestaDownfolder(
        fdf_fname="siesta.fdf", params=dict(PARAMS, orthogonal=orthogonal)
    )
    wann = downfolder.downfold()
    return downfolder, wann


def main():
    df_no, wann_no = run(False)
    df_or, wann_or = run(True)

    print("=" * 70)
    print("non-orthogonal run: is_orthogonal =", wann_no.is_orthogonal)
    print("orthogonal run:    is_orthogonal =", wann_or.is_orthogonal)
    nwann = wann_no.nwann
    S0 = wann_no.get_Sk(np.zeros(3))
    w_S = np.linalg.eigvalsh(S0)
    print(f"nwann = {nwann}")
    print(f"S(Gamma) eigenvalue range: [{w_S[0]:.6f}, {w_S[-1]:.6f}]")
    assert np.all(w_S > 1e-8), "SwannR must stay positive definite"
    q_generic = np.array([0.21, 0.13, -0.07])
    w_q = np.linalg.eigvalsh(wann_no.get_Sk(q_generic))
    print(f"S({q_generic}) eigenvalue range: [{w_q[0]:.6f}, {w_q[-1]:.6f}]"
          "  <- non-trivial metric")
    print(f"||S(Gamma) - I|| = {np.linalg.norm(S0 - np.eye(nwann)):.6f}")

    # same projected subspace on the mesh -> identical Rayleigh-Ritz values
    e_no = np.array([wann_no.solve_k(k)[0] for k in df_no.kpts])
    e_or = np.array([wann_or.solve_k(k)[0] for k in df_or.kpts])
    onmesh = np.abs(e_no - e_or).max()
    print(f"on-mesh max |nonortho - ortho| bands: {onmesh:.3e}")
    assert onmesh < 1e-8

    wann_no.save_pickle("wannier_nonorthogonal.pickle")

    # band comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (df, title) in zip(
        axes, [(df_or, "orthogonal gauge"), (df_no, "non-orthogonal gauge")]
    ):
        df.plot_band_fitting(
            kvectors=KVECTORS,
            knames=KNAMES,
            npoints=60,
            erange=[-8, 12],
            efermi=None,
            fullband_color="blue",
            downfolded_band_color="green",
            marker="o",
            ax=ax,
            show=False,
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("Downfolded_band_nonorthogonal.png", dpi=150)
    print("wrote Downfolded_band_nonorthogonal.png")
    print("=" * 70)
    print("ELECTRON NON-ORTHOGONAL EXAMPLE PASSED")


if __name__ == "__main__":
    main()
