## Release Notes


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


