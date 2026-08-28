## Quick Start

### Electron Wannier function (from a Siesta or Wannier90 Hamiltonian)

```python
import numpy as np
from lawaf.interfaces import SiestaDownfolder

params = dict(method="scdmk",           # scdmk | projected | maxprojected
              kmesh=[3, 3, 3],          # odd divisions, Gamma-centered
              nwann=2,
              weight_func="Gauss",      # unity | Gauss | Fermi | window
              weight_func_params=(-8, 6.5),   # (mu, sigma)
              use_proj=False,
              exclude_bands=[],
              use_ws_distance=True)   # W90 Wigner-Seitz R-grid (default);
                                      # False restores the legacy grid mode

downfolder = SiestaDownfolder(fdf_fname="siesta.fdf", params=params)
lwf = downfolder.downfold(write_hr_nc="Downfolded_hr.nc",
                          write_hr_txt="Downfolded_hr.txt")

downfolder.plot_band_fitting(
    kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]),
    knames=["$\Gamma$", "X", "M", "$\Gamma$"],
    npoints=100, savefig="Downfolded_band.png")
```

For a Wannier90 Hamiltonian, use `W90Downfolder(folder=..., prefix=...)`
instead — the same `params` apply. The folder must contain the wannier90
outputs including `<prefix>_centres.xyz` (set `write_xyz = true` in the
`.win` file to have wannier90 write it).

### Lattice Wannier function (from phonons)

```python
from lawaf import PhonopyDownfolder

params = dict(method="projected",
              kmesh=[2, 2, 2],
              nwann=3,
              weight_func="Fermi",
              weight_func_params=(100, 50),
              use_proj=False)

downfolder = PhonopyDownfolder(phonopy_yaml="phonopy_params.yaml", params=params)
downfolder.downfold(write_hr_nc="Downfolded_hr.nc",
                    write_hr_txt="Downfolded_hr.txt")
```

For polar materials use `NACPhonopyDownfolder` (see [Phonon](phonon.md)).

### Loading the result

```python
from lawaf.lwf.lwf import LWF

lwf = LWF.load_nc("Downfolded_hr.nc")   # loads both the legacy flat and the
                                        # grouped (xarray) netCDF schemas
```
