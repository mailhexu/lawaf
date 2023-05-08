import numpy as np
from ase import Atoms
from lawaf.plot.mcif import write_mcif
#from ase.io import write
#from ase.io.cif import write_cif

atoms=Atoms(cell=np.eye(3)*4.2, symbols="CsCl", scaled_positions=[[0,0,0], [.5,.5,.5]], pbc=True)
#atoms.set_initial_magnetic_moments([[0,0,1], [0,0,-1]])
#write("POSCAR.vasp", atoms, vasp5=True, sort=True)
#write("CsCl.cif", atoms)
write_mcif("mag.cif", atoms , vectors=[[0,0,1], [0,0,-1]], factor=1.5)
