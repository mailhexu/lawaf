import numpy as np
from scipy.linalg import eigh
from lawaf.mathutils.linalg import Lowdin
from ase.atoms import Atoms
from lawaf.utils.symbol import symbol_number
from collections import defaultdict


class SislWrapper:
    def __init__(self, sisl_geom, spin, ispin=None, shift_fermi=None):
        self.shift_fermi = shift_fermi
        self.spin = spin
        self.ispin = ispin
        self.orbs = []
        self.orb_dict = defaultdict(lambda: [])
        _atoms = sisl_geom._atoms
        atomic_numbers = []
        atom_positions = sisl_geom.xyz
        self.cell = np.array(sisl_geom.sc.cell)
        for ia, a in enumerate(_atoms):
            atomic_numbers.append(a.Z)
        self.atoms = Atoms(
            numbers=atomic_numbers, cell=self.cell, positions=atom_positions
        )
        xred = self.atoms.get_scaled_positions()
        sdict = list(symbol_number(self.atoms).keys())
        if self.spin.is_colinear:
            if ispin is None:
                raise ValueError("For colinear spin, ispin must be given")
        else:
            if ispin is not None:
                raise ValueError(
                    "For non-colinear spin and unpolarized spin, ispin should be None"
                )

        self.positions = []
        if self.spin.is_colinear:
            for ia, a in enumerate(_atoms):
                symnum = sdict[ia]
                orb_names = []
                for x in a.orbitals:
                    name = f"{symnum}|{x.name()}|{ispin}"
                    orb_names.append(name)
                    self.positions.append(xred[ia])
                self.orbs += orb_names
                self.orb_dict[ia] += orb_names
            self.norb = len(self.orbs)
            self.nbasis = self.norb
        elif self.spin.is_spinorbit:
            for ispin in ["up", "down"]:
                for ia, a in enumerate(_atoms):
                    symnum = sdict[ia]
                    orb_names = []
                    for x in a.orbitals:
                        name = f"{symnum}|{x.name()}|{spin}"
                        orb_names.append(name)
                        self.positions.append(xred[ia])
                    self.orbs += orb_names
                    self.orb_dict[ia] += orb_names
            self.norb = len(self.orbs) / 2
            self.nbasis = len(self.orbs)
        else:
            for ia, a in enumerate(_atoms):
                symnum = sdict[ia]
                orb_names = []
                for x in a.orbitals:
                    name = f"{symnum}|{x.name()}|None"
                    orb_names.append(name)
                    self.positions.append(xred[ia])
                self.orbs += orb_names
                self.orb_dict[ia] += orb_names
            self.norb = len(self.orbs)
            self.nbasis = len(self.orbs)
        self.positions = np.array(self.positions, dtype=float)

    def print_orbs(self):
        print(self.orb_dict)

    def solve_all(self, kpts):
        """Get eigenvalues and eigenvectors for all kpts. Should be
        implemented in children."""
        pass

    def get_fermi_level(self):
        if self.shift_fermi:
            return self.shift_fermi
        else:
            return 0.0


class SislHSWrapper(SislWrapper):
    def __init__(
        self, sisl_hamiltonian, ispin=None, shift_fermi=None, format="dense", nbands=10
    ):
        self.format = format
        self.nbands = nbands
        self.ham = sisl_hamiltonian
        super().__init__(self.ham.geometry, self.ham.spin, ispin, shift_fermi)

    def solve(self, k):
        if self.ispin is None:
            evals, evecs = self.ham.eigh(k=k, eigvals_only=False)
        else:
            evals, evecs = self.ham.eigh(k=k, spin=self.ispin, eigvals_only=False)
        if self.shift_fermi:
            evals += self.shift_fermi
        return evals, evecs

    def Hk(self, k, format="dense"):
        if self.ispin is not None:
            return self.ham.Hk(k, format=format, spin=self.ispin)
        else:
            return self.ham.Hk(k, format=format)

    def solve_all(self, kpts, orth=False):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            if orth and self.ham.orthogonal:
                S = self.ham.Sk(k, format="dense")
                Smh = Lowdin(S)
                H = self.Hk(k, format="dense")
                Horth = Smh.T.conj() @ H @ Smh
                evalue, evec = eigh(Horth)
            else:
                evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs, dtype=complex, order="C")

    def get_fermi_level(self):
        if self.shift_fermi:
            return self.shift_fermi
        else:
            return 0.0


class SislWFSXWrapper(SislWrapper):
    """Wrapper for retrieving eigenvalues and eigenvectors from siesta WFSX file

    Parameters
    ----------
    geom : sisl.Geometry
        the geometry containing information about atomic positions and orbitals
    wfsx_sile: sisl.io.siesta.wfsxSileSiesta
        file reader for WFSX file
    spin : sisl.physics.Spin
        spin object carrying information about spin configurations and spin component.
    ispin : None or int
        index of spin channel to be considered. Only takes effect for collinear spin calculations (UP: 0, DOWN: 1).
        (default: None)
    shift_fermi: None or float
        energy shift to be applied to all energies. If `None` no shift is applied. (default: None)
    """

    def __init__(self, geom, wfsx_sile, spin, ispin=None, shift_fermi=None):
        super().__init__(geom, spin=spin, ispin=ispin, shift_fermi=shift_fermi)
        self.geom = geom
        self.wfsx_sile = wfsx_sile
        self.read_all()

    def read_all(self):
        """Read all wavefunctions, eigenvalues, and k-points from WFSX file."""
        evals = []
        evecs = []
        wfsx_kpts = []

        def change_gauge(k, evec):
            """Change the eigenvector gauge"""
            phase = np.dot(
                self.geom.xyz[self.geom.o2a(np.arange(self.geom.no)), :],
                np.dot(k, self.geom.rcell),
            )
            if self.spin.has_noncolinear:
                # for NC/SOC we have a 2x2 spin-box per orbital
                phase = np.repeat(phase, 2)
            # r -> R gauge tranformation.
            return evec * np.exp(1j * phase).reshape(1, -1)

        # Read wavefunctions and eigenvalues
        for wfc in self.wfsx_sile.yield_eigenstate(parent=self.geom):
            wfsx_kpts.append(wfc.info["k"])
            evals.append(wfc.c)
            # To get the same eigvecs as eigh returns we need to transpose the
            # array and change the gauge from 'r' to 'R'
            evecs.append(change_gauge(wfc.info["k"], wfc.state).T)

        wfsx_kpts = np.asarray(wfsx_kpts)

        # If any k-point occurs more than once in the WaveFuncKPoints block,
        # we discard the duplicates
        is_duplicate = self._is_duplicate(wfsx_kpts)
        self.wfsx_kpts = wfsx_kpts[~is_duplicate]
        self.evals = np.array(evals, dtype=float)[~is_duplicate]
        if self.shift_fermi is not None:
            self.evals += self.shift_fermi
        self.evecs = np.array(evecs, dtype=np.complex64, order="C")[~is_duplicate]

    def _is_duplicate(self, array):
        # TODO: Move into utils
        # Find all matches (i,j): array[i] == array[j]
        matches = np.all(np.isclose(array[None, :], array[:, None]), axis=-1)
        # Remove double counting of matches: (i,j) and (j,i)
        # Also, remove matches of elements with themselves: (i,i)
        matches = np.triu(matches, 1)

        # Finally determine elements which are duplicates
        return np.any(matches, axis=0)

    def find_all(self, kpts):
        """Find the correct eigenvectors and eigenvalues and sort them
        to match the order in kpts.

        Parameters
        ----------
        kpts : list of float (3,) or (nk, 3)
            list of k points

        Returns
        -------
        evals : list of float (nk, n)
            list of eigenvalues for every requested k point
        evecs :
            list of eiegnvector for every requested k point
        """
        kpts = np.asarray(kpts)
        sort_idx = np.where(
            np.all(np.isclose(self.wfsx_kpts[None, :], kpts[:, None]), axis=-1)
        )[1]
        if len(sort_idx) < len(kpts):
            # k-point not found
            raise ValueError(
                f"{self.__class__.__name__}._read_all unable to find at least one "
                "required k point in '{self.wfsx_sile.file}'. Please, ensure that "
                "all k points listed below are included:\n"
                + "\n".join([str(k) for k in kpts])
            )
        if not np.all(np.isclose(self.wfsx_kpts[sort_idx], kpts)):
            raise ValueError(
                f"{self.__class__.__name__}._read_all was unable to match k points "
                "in {self.wfsx_sile.file} against required kpts:"
                + "\n".join([str(k) for k in kpts])
            )

        return self.evals[sort_idx], self.evecs[sort_idx]

    # Define alias for find_all to unify in wrapper's interface
    solve_all = find_all
