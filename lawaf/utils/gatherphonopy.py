import numpy as np
import phonopy

fname = "phonopy_params.yaml"
phonon = phonopy.load(
    force_sets_filename="FORCE_SETS",
    # born_filename="./BORN",
    # born_filename=None,
    unitcell_filename="POSCAR-unitcell",
    supercell_matrix=np.eye(3) * 3,
)
phonon.save(settings={"force_constants": True})
