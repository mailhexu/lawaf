# BaTiO3 anharmonic LWF campaign results

fixture: `/home/hexu/projects/lawaf_dev/lawaf/example/Phonopy/BaTiO3/DM_dip_wang/phonopy_params.yaml`
supercell: [3, 3, 3] (135 atoms, nQ=81), kmesh [2, 2, 2]

| gate | quantity | measured | threshold | pass |
|---|---|---|---|---|
| 1 | CV energy MAE / barrier scale | 2.043e-01 | 0.01 | False |
| 1 | CV force cosine | 0.977186 | >= 0.99 | False |
| 1 | CV stress RMSE / stress RMS | 7.530e-02 | 0.05 | False |
| 2 | C11 model vs teacher | 1.8474 vs 2.0337 eV/A^3 | rel <= 0.05 (9.16e-02) | False |
| 2 | C12 model vs teacher | 0.6792 vs 0.7188 eV/A^3 | rel <= 0.05 (5.52e-02) | False |
| 2 | C44 model vs teacher | 1.4469 vs 1.6491 eV/A^3 | rel <= 0.05 (1.23e-01) | False |
| 3 | dispersion round trip (max rel freq dev) | 2.663e-15 | 1e-06 | True |
| 4 | Molien consistent | True | - | True |
| 5 | spot check single (vs mace) | dE_mean 1.079e+03 eV, cos 0.9009 | - | measured |
| 5 | spot check coupled (vs mace) | dE_mean 1.078e+03 eV, cos 0.7252 | - | measured |
| 5 | spot check random (vs mace) | dE_mean 1.079e+03 eV, cos 0.9971 | - | measured |
| 5 | spot check reference (vs mace) | dE_mean 1.079e+03 eV, cos 0.8931 | - | measured |
| 6 | netCDF artifact (bitwise stored arrays, eval diff max 1.1e-16) | example/anharmonic_batio3/outputs/bato3_anharmonic_model.nc | <= 1e-12 | True |

all gates pass: **False**
