# banddownfolder
[![Documentation Status](https://readthedocs.org/projects/banddownfolder/badge/?version=latest)](https://banddownfolder.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/mailhexu/banddownfolder.svg?branch=master)](https://travis-ci.com/mailhexu/banddownfolder)

LaWaF is a python package for build  electron/phonon/magnon/etc Wannier functions.


It has interface to:
 * Siesta (through sisl)
 * Anaddb (part of Abinit)
 * Phonopy

The online documentation can be found at:
https://lawaf.readthedocs.io/en/latest/index.html

ABINIT PAW trial admission preserves raw XML radial functions and native
dual-projector coefficients. Projector contractions use the dataset
generator's per-integrand `overint` support; local augmentation uses the
common PAW sphere and native `simp_gen` quadrature. The defining local
overlap must match the compact metric with transposed electronic
coefficients at the unchanged normalized D5 gate of `1e-8`. Dataset
admission does not certify SCF or Wannier localization convergence.
