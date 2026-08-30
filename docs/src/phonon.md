## Phonon

### Building Lattice Wannier functions from a phonon band structure

The workflow is the same as for electrons: build a downfolder from phonopy
outputs, set the parameters, downfold.

```python
from lawaf import PhonopyDownfolder

params = dict(method="projected",
              kmesh=[2, 2, 2],
              nwann=3,
              weight_func="Fermi",
              weight_func_params=(100, 50),
              use_proj=False,
              use_ws_distance=True)   # W90 Wigner-Seitz R-grid (default)

downfolder = PhonopyDownfolder(phonopy_yaml="phonopy_params.yaml", params=params)
downfolder.downfold(write_hr_nc="Downfolded_hr.nc",
                    write_hr_txt="Downfolded_hr.txt")
```

Here the weight function is applied in frequency space (cm$^{-1}$ by default);
`weight_func_params` gives the frequency window/Gauss center and width.

### Dipole-dipole interaction and LO-TO splitting

For polar materials the dynamical matrix is non-analytical at $\Gamma$.
Use `NACPhonopyDownfolder`, which splits the dynamical matrix as
$H_{long} = H_k - H_{short}$ with the short-range part built from the
Gonze force constants (`wang` NAC method reads the `BORN` file):

```python
from lawaf import NACPhonopyDownfolder

downfolder = NACPhonopyDownfolder(phonopy_yaml="phonopy_params.yaml",
                                  mode="IFC",               # "DM" or "IFC"
                                  nac_params={"method": "wang"},
                                  born_filename="BORN",
                                  params=params)
lwf = downfolder.downfold()
lwf.write_to_netcdf("Downfolded_hr.nc")
lwf.save_txt("Downfolded_hr.txt")
```

When plotting the fit, enable the non-analytical correction at $\Gamma$:

```python
downfolder.plot_band_fitting(..., fix_LOTO=True, evals_to_freq=True,
                             unit_factor=15.6 * 33.6)
```

### Symmetry-canonical anchor seeds

Degenerate phonon blocks get an arbitrary eigenvector basis from the
diagonalization, so anchor-based projectors are not reproducible and are
not adapted to the little group of the anchor q-point. With
`pip install lawaf[symmetry]` (spgrep-modulation + spgrep) you can replace
them by order-parameter-direction (OPD) seeds:

```python
from lawaf import PhonopyDownfolder

params = dict(method="scdmk", nwann=3,
              anchors={(0., 0., 0.): (0, 1, 2)},
              symmetry_seed=True,
              # optional selectors:
              # symmetry_seed_opd=160,     # daughter SG (R3m)
              # symmetry_seed_opd_index=3, # exact family index
              )
downfolder = PhonopyDownfolder(phonopy_yaml="phonopy_params.yaml",
                               params=params)
lwf = downfolder.downfold()
```


`symmetry_seed=True` computes, for every anchor q, the spgrep-modulation
eigenspaces of the smallest commensurate supercell, maps them onto lawaf's
band blocks by eigenvalue, and replaces the touched degenerate blocks by
their OPD-canonical images (QR-orthonormalized). The default family is the
highest-order isotropy subgroup (P4mm for the BaTiO3 soft mode); use
`symmetry_seed_opd=<SG number>` (e.g. 160 for the rhombohedral R3m
direction) or `symmetry_seed_opd_index=<index>` for an exact family. Per
band, the downfold prints the daughter space group, direction character
(axis-pure / face-diagonal / body-diagonal / generic), family index and
frequency.

Discover the available families before choosing:

```python
from phonopy import load
from lawaf import list_opd_families

phonon = load("phonopy_params.yaml", is_nac=False)
families = list_opd_families(phonon, (0., 0., 0.), bands=(0, 1, 2))
```

Both `list_opd_families` and `get_symmetry_anchor_wfn` are importable
without the extra (the import happens lazily at call time). If any guard
fails for an anchor (non-commensurate q, eigenvalue mismatch beyond
`degeneracy_tol`, subspace check), that anchor falls back to lawaf's own
eigenvectors with a warning — the legacy behavior — while other anchors
keep their canonical seeds. Without `symmetry_seed` (default) nothing
changes relative to previous versions.

### Hand-selected modes on symmetry stars

`window_bands` pins the modes used at selected q-points without changing
projector construction. Keys are representative fractional q-points; values
are exactly `nwann` zero-based indices in the original sorted phonon
eigenspectrum:

```python
params = dict(
    method="projected",
    kmesh=(2, 2, 2),
    nwann=3,
    # Projectors may all come from one q-point.
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    # Hand-selected modes at inequivalent q-points.
    window_bands={
        (0.0, 0.0, 0.0): (0, 1, 2),
        (0.5, 0.0, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.0): (0, 1, 4),
        (0.5, 0.5, 0.5): (0, 1, 2),
    },
)
```

Each representative expands to its complete symmetry star. At the
representative and every arm, the selected modes receive exact weight `1`
and all other modes weight `0`; q-points outside those stars retain the
configured energy `weight_func`. The selected span must be invariant under
the little group and decompose into complete irreps; a complete irrep inside
an accidentally degenerate reducible frequency block is legal. A split
irrep, missing mesh arm, conflicting representative, excluded selected band,
or selection count different from `nwann` raises `ValueError` before gauge
construction. The same selection works for `method="projected"` and
`method="scdmk"`.

### Energy-window disentanglement (MLWF selection)

`method="mlwf"` can now select a continuous `nwann`-dimensional subspace
out of a larger band window instead of requiring an isolated manifold
(`nband == nwann` or `exclude_bands`). Give energy windows in the
builder's own units (phonons: `freqs_to_evals` values; convert cm^-1 with
`lawaf.mathutils.evals_freq.freqs_to_evals`):

```python
from lawaf.mathutils.evals_freq import freqs_to_evals

params = dict(
    method="mlwf", kmesh=(2, 2, 2), nwann=3,
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    dis_win_min=freqs_to_evals(-210),   # outer (feasible) window
    dis_win_max=freqs_to_evals(150),
    dis_froz_min=freqs_to_evals(-210),  # inner (frozen) window
    dis_froz_max=freqs_to_evals(-120),
)
```

Semantics (ADR-002): **energy windows carry the content; the overlap
fixed point arbitrates only energy-degenerate blocks.** Per k, the free
complement is filled by whole energy-degenerate blocks in
(energy, band-index) order; the first block the rank cuts is arbitrated
by the top restricted eigenvectors of the smoothness functional
$Z_k = \sum_b w_b\, M_{kb} S_{k+b} M_{kb}^\dagger$ (the w90
disentanglement $\Omega_D$ quantity, iterated with `dis_mix_ratio`
mixing). Bands beyond the cut block are never touched, so the overlap
criterion cannot pull in non-degenerate content. An exact or
noise-level tie inside a cut block snaps to the stable energy-then-index
order (deterministic; NFR tie-break) and is recorded in the selection
guidance; a tie that coincides with an order-one budget stall raises
`DegenerateSelectionError` (settle-or-raise).

FR-004 whole-block rule: an inner-window edge that would split an
energy-degenerate block silently instead freezes the whole block when
the frozen budget allows it, else raises `InfeasibleWindowError`; an
outer-window edge expands to the whole block. Because energies are
exactly star-degenerate, the decision is identical on every star arm.

Termination (F8): the iteration stops when the largest per-k subspace
change (1 - smallest singular value of $S^{\dagger} S'$) falls below
`dis_tol`, or at `dis_max_iter` passes. A slow descent tail below
`dis_slow_tail_change` at budget is accepted (w90-style); an order-one
change at budget raises. Post-selection neighbour overlaps are checked
against `dis_min_svd` and raise `SingularOverlapError` when the
selected bundle cannot carry a well-conditioned gauge. The whole
diagnostic record (per-neighbour svd spectrum, Z gaps, iteration trace,
guidance) lands on `builder.selection` (`SelectionResult`) for
inspection.

Notes for phonons: the displacement-metric content of a tight window
can be *intrinsically* near-singular at symmetry-related pairs (e.g.
BaTiO3 lowest-3 at Gamma<->X, where the soft-triplet continuation lives
far outside the window) — the same family the `exclude_bands` path runs.
Set `dis_min_svd=0` deliberately for such fixtures after inspecting the
selection diagnostics; keep the default guard otherwise. Electron paths
(W90-HR, Siesta LCAO through the same `MLWFWannierizer`) accept the
same `dis_*` parameters.

### Output formats

The phonon downfolders write grouped netCDF files (`lwf`/`atoms` xarray
groups). `lawaf.lwf.lwf.LWF.load_nc` transparently loads both this schema
and the legacy flat (`wann_*`) schema, so older files keep working.
