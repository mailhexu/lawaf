#!/usr/bin/env python3
import numpy as np
from lawaf.lwf.lwf_supercell import write_lwf_cif

write_lwf_cif(lwf_fname="Downfolded_hr.nc", 
              #listlwf=[0,1,2], 
              listlwf=None, 
              sc_matrix=np.diag([3,3,3]))


