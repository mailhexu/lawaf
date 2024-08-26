from ase import Atoms
from ase.io import write

# Create a cubic perovskite lattice of LaNiO3
a = 3.8
lattice = Atoms(
    "LaNiO3",
    scaled_positions=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0, 0.5, 0.5),
        (0.5, 0, 0.5),
        (0.5, 0.5, 0),
    ],
    cell=[a, a, a],
    pbc=True,
)

lattice.write("LaNiO3.vasp")
