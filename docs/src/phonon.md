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

### Output formats

The phonon downfolders write grouped netCDF files (`lwf`/`atoms` xarray
groups). `lawaf.lwf.lwf.LWF.load_nc` transparently loads both this schema
and the legacy flat (`wann_*`) schema, so older files keep working.
