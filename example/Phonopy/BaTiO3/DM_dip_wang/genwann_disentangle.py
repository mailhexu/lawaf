"""BaTiO3 lattice Wannier functions by MLWF energy-window
disentanglement (lawaf epic 9).

Unlike ``genwann.py`` (which needs ``exclude_bands`` to isolate the
manifold, nband == nwann), this script runs the 4x4x4 MLWF path with
all 15 phonon bands and selects the lowest-3 subspace by energy
windows:

    outer (feasible) window  [-210,  150] cm^-1
    inner (frozen) window    [-210, -120] cm^-1

Semantics: energy windows carry the content; the overlap fixed point
arbitrates only energy-degenerate blocks that the selection rank cuts
(ADR-002). Window bounds are given in cm^-1 and converted to the
builder's eigenvalue units with ``freqs_to_evals``.

Note: the windowed lowest-3 content of this fixture is intrinsically
exactly singular at Gamma<->X in the displacement metric (the soft
mode's continuation lives at 580/718 cm^-1, far outside the window) --
the same family the ``exclude_bands`` production path runs. The svd
guard is therefore disabled for this fixture (``dis_min_svd=0``) after
inspecting the selection diagnostics; keep the default guard (1e-8)
for other systems.

Run from this directory:
    python genwann_disentangle.py
"""

import numpy as np

from lawaf import PhonopyDownfolder
from lawaf.mathutils.evals_freq import freqs_to_evals

fname = "phonopy_params.yaml"
factor = 524.16  # cm^-1 (phonopy convention of this fixture)

# energy windows in cm^-1 -> builder eigenvalue units
win = (freqs_to_evals(-210, factor), freqs_to_evals(150, factor))
froz = (freqs_to_evals(-210, factor), freqs_to_evals(-120, factor))

params = dict(
    method="mlwf",
    nwann=3,  # select 3 out of 15 bands -- no exclude_bands needed
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    use_proj=True,
    proj_order=1,
    weight_func="unity",
    kmesh=(4, 4, 4),
    gamma=True,
    # MLWF optimizer (unchanged by the selection stage)
    mlwf_max_iter=500,
    # --- disentanglement selection (epic 9) ---
    dis_win_min=win[0],
    dis_win_max=win[1],
    dis_froz_min=froz[0],
    dis_froz_max=froz[1],
    dis_min_svd=0.0,  # intrinsically singular content, see module docstring
)

downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM", params=params)
lwf = downfolder.downfold(
    output_path="./disentangle_result",
    write_hr_nc="LWF.nc",
    write_hr_txt="LWF.txt",
)

# --- selection diagnostics (SelectionResult, FR-007) ---
sel = lwf.selection
print("\n=== disentanglement selection ===")
print(f"iterations:          {sel.n_iter}")
print(f"subspace change:     {sel.subspace_change_trace[-1]:.3e}")
print(f"tie-snapped k:       {sel.guidance['tie_snapped']}")
print(f"budget exhausted:    {sel.guidance['budget_exhausted']}")
print(f"guidance sources:    {sel.guidance['sources']}")
print(f"Omega_I (selection): {sel.omega_i_selection:.4f}")
svd_min = sel.pair_svd_min.min()
svd_mean = sel.pair_svd_mean.min()
print(f"pair svd min/mean:   {svd_min:.3e} / {svd_mean:.3f}")
print("  (exactly-singular Gamma<->X class: expected, see docstring)")

print("\n=== MLWF spreads (crystal units) ===")
for key in ("omega_I", "omega_D", "omega_OD", "omega"):
    print(f"{key:8s} = {lwf.spreads[key]:.4f}")
