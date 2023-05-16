#!/usr/bin/env python3
import numpy as np
from lawaf.lwf.lwf_supercell import write_lwf_cif
import os

write_lwf_cif(lwf_fname="LWF.nc", 
              listlwf=None, 
              sc_matrix=np.diag([4,4,4]))



