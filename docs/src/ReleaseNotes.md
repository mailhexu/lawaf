## Release Notes


### v0.4 (unreleased)
* Caller-owned ABINIT sources now require the complete canonical `WfkHeader`,
  explicit target mesh coordinates/count and source-index-aware symmetry maps.
  Map entries declare operation, translation, reciprocal shift, spin label and
  antiunitary flag. `MeshSource` consumes these entries rather than rediscovering
  a possibly different map. Geometry, occupations and symmetry metadata remain
  borrowed read-only views; no flat-header defaults or compatibility aliases.
  Actual decoder allocation/lifetime measurements replace borrowed-byte counters.
  The full64-point file/caller smoke is bit-exact; it is not a localization gate.
* WFK binary body indexing now validates each point record's stored plane-wave,
  spinor and band counts against its header, and checks every trailing Fortran
  record marker, including the final coefficient record. Count disagreement
  raises D2; missing/corrupted trailing markers raise D5. Missing NetCDF
  eigenvalue variables or incomplete requested slabs raise contextual D6.
* Parsed UPF/XML radial arrays are read-only, retaining exact stored values.
  Trial manifests publish deeply immutable units, harmonic conventions and
  dataset identity/path mappings; harmonic normalization and boolean
  spin-frame declarations are mandatory rather than guessed/coerced.
* Added `PawSpecies` numerical XML operators in a declared normalized real
  Wannier90 s/p/d/f basis: bounded projector Fourier kets, zero-b DeltaS and
  finite-b Q with negative Fourier phase and reverse-pair adjoint symmetry.
  Raw `R(r)` uses `r^2 dr` and the native logarithmic-grid Jacobian; actual
  dataset duality is checked rather than converting partial-wave overlaps
  into cprj. Atomic phase and cell-volume factors are deliberately left to
  the complete-state consumer and must each be applied once.
  Independent Cartesian Gaussian fields now certify complete two-sided PAW
  A/M, both coherent components, the overlap metric and NC limit. The
  complete mixed-storage GaAs file set passes real external parsing with
  supplied/computed parity; its source SCF is not a convergence certificate.
* PAW manifests can select `paw_augmented_partial_wave`: the referenced XML
  pseudo radial function inside the augmentation radius, zero outside,
  transformed by PAW. Raw scattering waves are not localized trial functions.
  `paw_dual_projector` remains an explicitly different trial definition and
  fails admission if defining local closure exceeds the approved gate.
  Supplied electronic cprj remains `dualprojector` in either trial basis.
* Supplied stored/IBZ PAW coefficients are transported through `MeshSource`
  using the actual atom-cell image, harmonic rotation, antiunitary operation
  and coherent SU(2) action, without recomputing Fourier projector overlaps.
* `AbinitWannierFileSet` generates separate channel plans/seeds and actual
  per-channel AMN/MMN/EIG. Coherent spinors use one physical-band energy set,
  sum both same-spin components in A/M and declare `spinors=true` in WIN.
  External preprocessing uses the canonical `spinor_projections` NNKP block.
  AFM reconstruction requires explicit up/down source-symmetry maps.
  Actual GaAs collinear and SOC file sets completed external localization;
  independent mixed-channel/mixed-spin fields detect channel/component faults.
* Supplied PAW handoff now requires `witness_compare(record, reference)` to
  succeed at the unchanged same-source tolerance before constructing the
  producer. Schema validation is not a numerical certificate. Keep borrowed
  coefficient buffers and independent references unchanged while consuming
  them; a new file record must be witnessed again after opening.
  DeltaS and each distinct finite-b Q now enforce an actual finer-grid radial
  integration residual at1e-8, with bounded refinement and a named D5 error
  instead of publishing an unresolved operator. This is fixed-spline
  integration admission, not dataset/source convergence.
* MMN is contracted in bounded physical-band row/column tiles before
  allocation; sparse original-band selections no longer decode their entire
  intervening band interval. `slab_columns=0` uses bounded defaults.
  Published plans, source headers and image geometry are deeply immutable.
  Failed WFK construction closes its acquired handle immediately; missing
  streamed EIG points fail rather than publishing a truncated sequence.
  Collinear exchange/antiunitary images route actual source-channel energies,
  occupations and states; explicit maps carry original channel identity.
* `AbinitWannierInputs(..., trial_manifest=...)` now evaluates physical NC
  numerical trials directly from the referenced UPF `PP_CHI` channels. The
  bounded native-grid Fourier transform preserves `u=rR`, PP_RAB quadrature,
  normalized Wannier90 s/p/d/f harmonic order, Cartesian local axes, atomic
  Bloch phases and cell-volume normalization; no hidden radial normalization
  or same-l channel averaging occurs. A supplied callable remains an explicit
  alternative, not a fallback.
  Manifests now declare `position_coordinates` (`reduced` or `cartesian`);
  `default_nc_manifest` writes reduced site coordinates. Parsed manifests
  retain validated right-handed local axes, sites and dataset paths. Relative
  dataset references resolve against the manifest directory.
  A real GaAs WFK-to-external-localization smoke exercised this path and
  HamiltonIO output consumption; scientific symmetry certification remains
  open because the current NSCF reference did not meet its requested tolerance.
* `MeshSource` reconstructs scalar NC IBZ states on demand through the shared
  source seam. The eager full-mesh coefficient expansion/cache was removed.
  Reciprocal folding relabels G vectors; fractional translations phase the
  source Bloch field before antiunitary conjugation. Raw WFK symmetry rows
  follow native `cgtk_rotate` (transposed native symrel), not a convention
  chosen by fitting eigenvalues. Antiunitary search requires an explicit
  `time_reversal` certificate; it is no longer enabled for every source.
* Fixed ABINIT Fortran-record WFK spinor decoding: G varies fastest within
  each coherent component, matching native `cg(2,npw*nspinor)` and NetCDF
  records. Each physical band still has one eigenvalue.
* WFK symmetry reconstruction requires all three metadata fields:
  `reduced_symmetry_matrices`, `reduced_symmetry_translations`, and `symafm`.
  Missing translations or magnetic symmetry flags now raise a named `D7`
  diagnostic instead of silently substituting incomplete/nonmagnetic data.
* ABINIT file and caller-array source seams now consistently expose lattice
  vectors as rows in bohr. Native Fortran-column/NetCDF axes are converted
  once at the reader boundary; the row-lattice reciprocal metric is
  `(2*pi)^2 * inverse(A).T @ inverse(A)`. Skew-cell G-sphere checks and WIN
  geometry therefore use the same cell.
* WFK block reads avoid redundant NetCDF complex-array copies and retain
  only the requested eigenvalue slab. `peak_live_bytes` reports tracked
  numeric payload buffers; allocation measurements separately include Python
  overhead rather than mistaking that counter for total live memory.
* WFK automatic `istwfk=0` records now resolve per k point while preserving
  the raw modes. Mixed resolved/automatic records retain exactly the same
  decoded states as their explicit-storage counterparts.
* Supplied PAW carrier slab access uses original physical-band IDs within
  its declared band interval, including nonzero starts. NetCDF slabs are
  read-only; supplied/reference comparison is blockwise, not a full-tensor
  copy. Actual supplied memory/file records now feed matched PAW DeltaS and
  finite-b Q consumers with zero Fourier recomputations; XML operators remain
  mandatory. Schema/kind/frame/conjugation/phase/raw-normalization declarations
  are explicit. Expanded records require a named map reference matching the
  consumer declaration. Unsupported versions, fractional state identities,
  swapped source species and malformed metadata fail with named diagnostics.
  Analytic parity rejects non-finite overlaps and uses the true reference scale,
  with only exact zeros admitted against a zero reference. Metadata is readonly.
  This is supplied-reuse certification, not source physics or localization.
  A one-channel coefficient carrier now declares its original zero-based
  `SupplierMeta.source_channel` (NetCDF attribute `source_channel`); no
  implicit up-channel identity is substituted. Band coverage is validated
  against that selected source channel. Incorrect file axes or non-float64
  real/imag pairs fail at open with named diagnostics.
  Projector order is also explicit: `projector_labels` contains one
  `(atom, radial_id, l, m)` tuple per column, grouped by the declared atom
  counts, and `harmonic_convention` names the angular basis. Missing or
  duplicated labels fail rather than assuming that equal counts imply the
  same functions. Both fields are persisted in the v1 NetCDF carrier.
* The ABINIT PAW-XML reader preserves radial functions as `R(r)` and reports
  `PawXmlDataset.radial_representation="R"`. Partial-wave and projector
  contractions therefore require `r^2 dr`; native ABINIT multiplies XML
  arrays by `r` when constructing its internal `rR` arrays. On the supported
  logarithmic mesh `r=a*(exp(d*i)-1)`, integrate using the Jacobian
  `dr/di=d*(r+a)`. The reader does not resample or multiply the stored arrays.
* Fixed a stale-cache bug in the phonopy wrapper: `phon_cache` entries
  are now keyed by a fingerprint of the force constants, masses, and
  evaluation mode in addition to the k-point, so a cache left in the
  working directory by a run with a different configuration is never
  served (previously eigenvectors of another configuration could be
  silently reused).
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


