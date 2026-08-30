## Release Notes


### v0.4 (unreleased)
* Phonon symmetry seeds (spgrep-modulation):

  * New `symmetry` install extra (`spgrep-modulation>=0.3,<0.4`,
    `spgrep>=0.5`).
  * `WannierParams.symmetry_seed` (default `False`) plus
    `symmetry_seed_opd` / `symmetry_seed_opd_index` selectors: every anchor
    q of `PhonopyDownfolder` / `NACPhonopyDownfolder` can use
    symmetry-canonical OPD seeds instead of the arbitrary degenerate-block
    eigenvectors (both `projected` and `scdmk` methods). Seed metadata
    (daughter space group, direction character, family index, frequency)
    is printed per band; guard failures fall back to the legacy anchors
    for that q with a warning.
  * New public API `lawaf.get_symmetry_anchor_wfn` and
    `lawaf.list_opd_families` (importable without the extra; the spgrep
    import is lazy). `lawaf.__all__` entries were fixed to strings
    (pre-existing bug).
  * `Lawaf.set_parameters` gained an `anchor_ibands` passthrough
    (default `(0, 1, 2)`, matching the `WannierParams` default).
* Anharmonic LWF effective models (`lawaf[anharmonic]` extra):

  * New `lawaf.anharmonic` package: space-group action and
    representation/compatibility checks, star-covariant constrained gauge,
    Q-space sampling with force/stress projection, `TrainingDataset` +
    teacher stack (MACE via atomchain, ASE calculators, ABINIT HIST),
    Oh-closed invariant polynomial basis with strain sector (Reynolds
    averaging + Molien cross-check), joint energy/force/stress ridge and
    screened-greedy fits, `AnharmonicModel` evaluator + ASE calculator,
    netCDF persistence (`anharmonic` + `symmetry` groups, bitwise
    round-trip), and phonon-symmetry seed labels (`SeedsLabels`).
  * BaTiO3 acceptance campaign under `example/anharmonic_batio3/`
    (symmetrized mapping, gates 3-6 pass; see
    `docs/src/lwf_anharmonic_model.md` for the measured numbers and the
    v1 architecture notes).
* MLWF energy-window disentanglement (`method="mlwf"`, nband > nwann):

  * New `lawaf/wannierization/disentangle.py`: `select_subspace`
    (overlap-guided selection with energy blocks carrying content and
    the Z fixed point arbitrating only cut degenerate blocks),
    `resolve_windows` (energy windows / `window_bands` pins to
    feasible+frozen sets with the FR-004 whole-block rule), and
    `SelectionResult` diagnostics (neighbour svd spectrum, Z gaps,
    iteration trace, tie/budget guidance).
  * New `WannierParams` fields `dis_win_min/max`, `dis_froz_min/max`,
    `dis_mix_ratio`, `dis_max_iter`, `dis_tol`, `dis_min_svd`,
    `dis_slow_tail_change` plumbed through both phonon and electron
    downfolders; `builder.selection` exposes the record; the
    `nband == nwann` isolated-manifold path is unchanged.
  * Settle-or-raise tie policy: exact/noise-level overlap ties snap to
    the stable energy-then-index order (deterministic); a tie implicated
    in an order-one budget stall raises `DegenerateSelectionError`.
    An intrinsically singular window content raises
    `SingularOverlapError` against `dis_min_svd` (default 1e-8; use 0
    deliberately for phonon fixtures whose window content is exactly
    singular by symmetry, e.g. BaTiO3 lowest-3).
* Hand-selected phonon modes:

  * New `WannierParams.window_bands` accepts explicit original-band indices
    at representative q-points, validates whole little-group irreps, expands
    them to every symmetry-star arm, and applies exact selected=`1` /
    unselected=`0` weights in both projected and SCDM-k wannierization.
    Projector `anchors` remain independent, so all projectors may still come
    from one q-point. Window definitions and legality results round-trip in
    the netCDF `symmetry` group.



### v0.3 August 2026
* Wigner-Seitz R-grid for k<->R transformations (wannier90 `ws_distance`
  semantics):

  * New `lawaf.mathutils.ws_distance` module: `ws_translate_dist` (faithful,
    vectorized reimplementation of wannier90's `ws_distance.F90`, with
    rectangular-tensor `centers_j` support), `apply_ws_distance` /
    `apply_ws_distance_tensors` (materialize `H_ij(R)/ndeg` onto the
    Wigner-Seitz image R vectors), and `fold_R_to_mesh` (fold the legacy
    extended R grid onto the plain mesh grid).
  * `WannierParams.use_ws_distance` (default `True`, wannier90 v3 default)
    wires the WS assignment into the electron (`Lawaf.downfold`) and phonon
    (`PhonopyDownfolder`, `NACPhonopyDownfolder`) paths. Set it to `False`
    to restore the legacy grid behavior.
* Full-value k<->R storage (single source of truth in
  `lawaf.mathutils.kR_convert`): DFT coefficients are stored unfolded;
  the R-vector degeneracy weights are carried as `Rdeg` metadata and
  applied at every R-sum (`R_to_k`, `R_to_onek`, `HR_to_k`,
  `LWF.get_wann_Hk`). `build_Rgrid(..., wigner_seitz=True)` supersedes the
  legacy `degeneracy` alias. Duplicate k<->R implementations were removed
  (`eigen_modifer.py`, `wannierizer.py::Hk_to_Hreal`).
* `LWF` persists `Rdeg` in its netCDF writers (`wann_Rdeg`; missing
  variable defaults to ones, so old files keep working).
* `SupercellMaker.sc_Rlist_HR(..., accumulate=True)` merges blocks that
  fold onto the same supercell R (required for WS-materialized R lists);
  `sc_RHdict` fixed (undefined `H` in the defaultdict lambda, wrong dict
  key); `build_lwf_lattice_mapping_matrix`/`MyLWFSC` gained `accumulate`.
* Guard: WS mode engages only for Gamma-centered, unshifted meshes
  (`gamma=True`, `kshift=0`); degenerate images share a phase at mesh k
  only there. Other meshes silently keep the legacy grid mode.
* Review hardening: `LWF` R-sums (`hoppings`, `force_ASR`, `Zwann`,
  masses, norms, Born-Wannier reductions) now apply `Rdeg`;
  `HamiltonIO`'s `LawafHamiltonian.get_Hk/get_Sk` apply `Rdeg`;
  `NACLWF.split_short_long_wang` is WS-safe (transforms on the plain mesh
  grid, re-scatters onto the union R list); `sc_RHdict` preserves complex
  dtype; the phonopy `lwf_supercell` duplicate builders were consolidated
  into `lawaf.lwf.lwf_supercell` (mass-scaled `wann_disps` supported).

### v0.2.5 August 2026
* Compatibility and fixes:

  * Support phonopy >= 3.0: the NAC short/long-range dynamical matrix split is
    rebuilt from `short_range_force_constants` with an explicit Gonze dataset
    build at $\Gamma$.
  * `NACPhonopyDownfolder.downfold()` returns the LWF object (the return
    statement was missing).
  * `NACLWF` sets `nbasis`, fixing `save_txt`.
  * `LWF.load_from_netcdf` fixed against the current writer (ase `numbers=`
    keyword, `factor` stored as a data variable).
  * `lawaf.lwf.lwf.LWF.load_nc` loads both the legacy flat and the grouped
    (xarray) netCDF schemas.

* Examples updated to the current API (`params` dict, `weight_func_params`);
  missing `abinito_w90_down_centres.xyz` regenerated in the Wannier90 example.

### v0.1 April 2020
* Features:

  * General interface to downfolding of tight-binding-like Hamiltonian based on the scdm-k method and projected Wannier function method.
  * Output of Hamiltonian to txt file and netcdf file.
  * Interface to Wannier90, Siesta and Phonopy.

* Examples

  * Downfolding a Wannier90 Hamiltonian from d-p model of SrMnO3 to a two band $e_g$ model.

* Tutorial

  * A tutorial on how to use banddownfolding to downfold Wannier90 Hamiltonian.


