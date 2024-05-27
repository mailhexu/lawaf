import copy
import pickle
import numpy as np
from scipy.linalg import eigh
from netCDF4 import Dataset
from ase import Atoms

# from lawaf.scdm.eigen_modifer import HamModifier, force_ASR_kspace
import matplotlib.pyplot as plt
from lawaf.plot import plot_band
from dataclasses import dataclass
from typing import Union


@dataclass
class TwobodyTerms:
    # nterm: int = 0
    # Rlist: np.ndarray = None
    # coeff_twobody: np.array

    def __init__(self):
        self.nterm: int = 0
        self.Rlist = None
        coeff_twobody = None

    def write_to_netcdf_file(self, ncfile: Dataset):
        Dataset.redef()
        nterm = Dataset.createDimension("nterm", self.nterm)
        iR = Dataset.createVariable("wann_twobody_iR", int, dimensions=("nterm",))
        i = Dataset.createVariable("wann_twobody_i", int, dimensions=("nterm",))
        j = Dataset.createVariable("wann_twobody_j", int, dimensions=("nterm",))
        orderi = Dataset.createVariable(
            "wann_twobody_orderi", int, dimensions=("nterm",)
        )
        orderj = Dataset.createVariable(
            "wann_twobody_orderj", int, dimensions=("nterm",)
        )
        val = Dataset.createVariable("wann_twobody_val", int, dimensions=("nterm",))
        iR[:] = self.iR
        i[:] = self.coeff_twobody[0, :]
        j[:] = self.coeff_twobody[1, :]
        orderi[:] = self.coeff_twobody[2, :]
        orderj[:] = self.coeff_twobody[3, :]
        val[:] = self.val


@dataclass
class GenericWF:
    """Generic Wannier functions class"""

    nwann: int = 0
    ndim: int = 3
    nbasis: int = 0
    nR: int = 0
    wannR: np.ndarray = None
    HwannR: np.ndarray = None
    Rlist: np.ndarray = None
    cell: np.ndarray = None
    wann_centers: Union[np.ndarray, None] = None
    Rdict: Union[dict, None] = None
    name: str = "GenericWF"


class LWF(GenericWF):
    """
    Localized Wannier function
    """

    name: str = "LWF"
    _atoms: Atoms = None
    has_nac: bool = False
    born_effective_charges: np.ndarray = None
    dielectric: np.ndarray = None
    factor: float = 1.0
    Zwann: np.ndarray = None

    def __init__(
        self,
        wannR=None,
        HwannR=None,
        Rlist=None,
        cell=None,
        wann_centers=None,
        atoms=None,
        born=None,
        has_nac=False,
        dielectric=None,
        factor=1.0,
        fortran_order=False,
        twobody_terms=None,
    ):
        self.wannR = wannR
        self.HwannR = HwannR
        self.Rlist = Rlist
        self.cell = cell

        self.nR, self.nbasis, self.nwann = np.shape(wannR)
        if wann_centers is None:
            self.wann_centers = np.zeros((self.nwann, 3))
        else:
            self.wann_centers = wann_centers
        self.natom = self.nbasis // 3
        self.ndim = np.shape(self.Rlist)[1]
        self.Rdict = {}
        for i, R in enumerate(Rlist):
            self.Rdict[tuple(R)] = i
        self.twobody_terms = twobody_terms
        self.has_nac = has_nac
        self.born_effective_charges = born
        self.dielectric = dielectric
        self.factor = factor
        self.atoms = atoms

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms
        if atoms is not None:
            assert self.nbasis == 3 * len(atoms)

    def set_born_from_full(self, born, dielectric, factor):
        self.born_effective_charges = born
        self.dielectric = dielectric
        self.factor = factor
        self.has_nac = True
        self.Zwann = np.zeros((self.nwann, 3))
        self.natom = self.nbasis // 3
        for i in range(self.nwann):
            W_R_tau_i = self.wannR[:, :, i].reshape(self.nR, self.natom, 3)
            # TODO: check the order of born effective charges (de or ed?)
            # R: Rvector. d & e: directions of dispalcement and efield
            # print(f"self.born_effective_charges: {self.born_effective_charges.shape}")
            W_R_tau_i /= np.linalg.norm(W_R_tau_i)
            self.Zwann[i, :] = np.einsum(
                "Rad,ade->e", W_R_tau_i, self.born_effective_charges
            )
        print(f"Zwann:{self.Zwann=}")

    def get_wannier_born(self):
        return self.Zwann

    def find_iR_at_zero(self):
        iR = np.argmin(np.linalg.norm(self.Rlist, axis=1))
        if not np.linalg.norm(self.Rlist[iR]) == 0:
            raise ValueError("index of R=[0,0,0] not found")
        return iR

    @property
    def hoppings(self):
        Rlist = [tuple(R) for R in self.Rlist]
        data = copy.deepcopy(dict(zip(Rlist, self.HwannR)))
        np.fill_diagonal(data[(0, 0, 0)], 0.0)
        return data

    def view_one_lwf(self, ilwf, threshold=0.01):
        rw = np.real(self.wannR[:, :, ilwf])
        for iR, R in enumerate(self.Rlist):
            print(f"R: {R}")
            ids = np.where(np.abs(rw[iR, :]) > threshold)[0]
            vals = rw[iR, ids]
            for i, val in zip(ids, vals):
                print(f"{i}: {val}")

    @property
    def site_energies(self):
        iR = self.Rdict[(0, 0, 0)]
        return np.real(np.diag(self.HwannR[iR]))

    def HR(self, R):
        iR = self.Rdict[tuple(R)]
        return self.HwannR[iR]

    def get_wann_Hk(self, k):
        hk = np.zeros((self.nwann, self.nwann), dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2j * np.pi * np.dot(R, k))
            hk += self.HwannR[iR, :, :] * phase
        return hk

    def solve_wann_k(self, k, ham=False):
        hk = self.get_wann_Hk(k)
        evals, evecs = eigh(hk)
        if not ham:
            return evals, evecs
        else:
            return evals, evecs, hk

    def solve_all(self, kpts):
        evals, evecs = [], []
        for k in kpts:
            evalue, evec = self.solve_wann_k(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals), np.array(evecs)

    def get_wann_largest_basis(self):
        ret = []
        for iwann in range(self.nwann):
            a = np.abs(self.wannR[:, :, iwann])
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            ret.append({"R": self.Rlist[ind[0]], "orb": ind[1]})
        return ret

    def save_hr_pickle(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(
                {"HR:": self.HwannR, "Rlist:": self.Rlist, "atoms": self.atoms}, f
            )

    def write_nc(self, fname, prefix="wann_", atoms=None):
        root = Dataset(fname, "w")
        ndim = root.createDimension("ndim", self.ndim)
        nR = root.createDimension("nR", self.nR)
        three = root.createDimension("three", 3)
        nwann = root.createDimension(prefix + "nwann", self.nwann)
        nbasis = root.createDimension(prefix + "nbasis", self.nbasis)

        if atoms is not None:
            natom = root.createDimension(prefix + "natom", len(atoms))

        Rlist = root.createVariable(prefix + "Rlist", int, dimensions=("nR", "ndim"))
        HamR_real = root.createVariable(
            prefix + "HamR_real",
            float,
            dimensions=("nR", prefix + "nwann", prefix + "nwann"),
            zlib=True,
        )
        HamR_imag = root.createVariable(
            prefix + "HamR_imag",
            float,
            dimensions=("nR", prefix + "nwann", prefix + "nwann"),
            zlib=True,
        )

        wannR_real = root.createVariable(
            prefix + "wannier_function_real",
            float,
            dimensions=("nR", prefix + "nbasis", prefix + "nwann"),
            zlib=True,
        )
        wannR_imag = root.createVariable(
            prefix + "wannier_function_imag",
            float,
            dimensions=("nR", prefix + "nbasis", prefix + "nwann"),
            zlib=True,
        )
        wann_center_xred = root.createVariable(
            prefix + "wannier_center_xred",
            float,
            dimensions=(prefix + "nwann", "three"),
            zlib=True,
        )

        if atoms is not None:
            cell = root.createVariable(
                prefix + "cell", float, dimensions=("three", "three"), zlib=True
            )
            numbers = root.createVariable(
                prefix + "atomic_numbers", int, dimensions=(prefix + "natom"), zlib=True
            )
            masses = root.createVariable(
                prefix + "atomic_masses",
                float,
                dimensions=(prefix + "natom"),
                zlib=True,
            )

            xred = root.createVariable(
                prefix + "atomic_xred", float, dimensions=(prefix + "natom", "three")
            )

            xcart = root.createVariable(
                prefix + "atomic_xcart", float, dimensions=(prefix + "natom", "three")
            )
            lwf_masses = root.createVariable(
                prefix + "lwf_masses", float, dimensions=(prefix + "nwann")
            )

            cell[:] = atoms.get_cell().array
            numbers[:] = atoms.get_atomic_numbers()
            xred[:] = atoms.get_scaled_positions(wrap=False)
            xcart[:] = atoms.get_positions()
            masses[:] = atoms.get_masses()
            try:
                lwf_masses[:] = np.real(self.masses_to_lwf_masses(masses))
            except:
                pass

        Rlist[:] = np.array(self.Rlist)
        HamR_real[:] = np.real(self.HwannR)
        HamR_imag[:] = np.imag(self.HwannR)
        wannR_real[:] = np.real(self.wannR)
        wannR_imag[:] = np.imag(self.wannR)
        wann_center_xred[:] = np.array(self.wann_centers)

        if self.twobody_terms is not None:
            self.twobody_terms.write_to_netcdf_file(root)

        root.close()

    def masses_to_lwf_masses(self, masses):
        m3 = np.kron(masses, [1, 1, 1])
        lwf_masses = np.einsum("rij,i->j", (self.wannR.conj() * self.wannR), m3)
        return lwf_masses

    def born_to_lwf(self, born):
        wborn = np.zeors((self.nwann, 3))
        for i in range(self.nwann):
            for j in range(self.natom):
                wborn[i]

    def write_lwf_nc(self, fname, prefix="wann_", atoms=None):
        root = Dataset(fname, "w")
        ndim = root.createDimension("ndim", self.ndim)
        nR = root.createDimension("nR", self.nR)
        three = root.createDimension("three", 3)
        nwann = root.createDimension(prefix + "nwann", self.nwann)
        nbasis = root.createDimension(prefix + "nbasis", self.nbasis)

        if self.atoms is not None:
            atoms = self.atoms
        if atoms is not None:
            natom = root.createDimension(prefix + "natom", len(atoms))

        Rlist = root.createVariable(prefix + "Rlist", float, dimensions=("nR", "ndim"))
        ifc_real = root.createVariable(
            prefix + "HamR_real",
            float,
            dimensions=("nR", prefix + "nwann", prefix + "nwann"),
            zlib=True,
        )
        ifc_imag = root.createVariable(
            prefix + "HamR_imag",
            float,
            dimensions=("nR", prefix + "nwann", prefix + "nwann"),
            zlib=True,
        )

        wannR_real = root.createVariable(
            prefix + "wannier_function_real",
            float,
            dimensions=("nR", prefix + "nbasis", prefix + "nwann"),
            zlib=True,
        )
        wannR_imag = root.createVariable(
            prefix + "wannier_function_imag",
            float,
            dimensions=("nR", prefix + "nbasis", prefix + "nwann"),
            zlib=True,
        )

        wann_centers = root.createVariable(
            prefix + "centers", float, dimensions=(prefix + "nwann", "three"), zlib=True
        )

        # root.createVariable(prefix+'xred', float64, dimensions=(nR, nwann, nwann))
        # root.createVariable(prefix+'cell', float64, dimensions=(nR, nwann, nwann))
        if atoms is not None:
            cell = root.createVariable(
                prefix + "cell", float, dimensions=("three", "three"), zlib=True
            )
            numbers = root.createVariable(
                prefix + "atomic_numbers", int, dimensions=(prefix + "natom"), zlib=True
            )
            masses = root.createVariable(
                prefix + "atomic_masses",
                float,
                dimensions=(prefix + "natom"),
                zlib=True,
            )

            xred = root.createVariable(
                prefix + "atomic_xred", float, dimensions=(prefix + "natom", "three")
            )

            xcart = root.createVariable(
                prefix + "atomic_xcart", float, dimensions=(prefix + "natom", "three")
            )
            lwf_masses = root.createVariable(
                prefix + "lwf_masses", float, dimensions=(prefix + "nwann")
            )

            cell[:] = atoms.get_cell().array
            numbers[:] = atoms.get_atomic_numbers()
            xred[:] = atoms.get_scaled_positions()
            xcart[:] = atoms.get_positions()
            masses[:] = atoms.get_masses()
            try:
                lwf_masses[:] = np.real(self.masses_to_lwf_masses(masses))
            except:
                pass

        Rlist[:] = np.array(self.Rlist)
        ifc_real[:] = np.real(self.HwannR)
        ifc_imag[:] = np.imag(self.HwannR)
        wannR_real[:] = np.real(self.wannR)
        wannR_imag[:] = np.imag(self.wannR)
        wann_centers[:, :] = self.wann_centers

        root.close()

    @staticmethod
    def load_nc(fname, prefix="wann_", order="C"):
        root = Dataset(fname, "r")
        ndim = root.dimensions["ndim"].size
        nR = root.dimensions["nR"].size
        nwann = root.dimensions[prefix + "nwann"].size
        nbasis = root.dimensions[prefix + "nbasis"].size
        try:
            natom = root.dimensions[prefix + "natom"].size
            has_atom = True
        except:
            has_atom = False
            atoms = None

        Rlist = root.variables[prefix + "Rlist"][:]
        Rlist = np.array(Rlist, dtype=int)
        try:
            ham_real = root.variables[prefix + "HamR_real"][:]
            ham_imag = root.variables[prefix + "HamR_imag"][:]
        except KeyError:
            ham_real = root.variables[prefix + "ifc_real"][:]
            ham_imag = root.variables[prefix + "ifc_imag"][:]

        Ham = ham_real + ham_imag * 1.0j

        wannR_real = root.variables[prefix + "wannier_function_real"][:]
        wannR_imag = root.variables[prefix + "wannier_function_imag"][:]
        wannR = wannR_real + wannR_imag * 1.0j
        if order.strip().capitalize().startswith("F"):
            wannR = np.swapaxes(wannR, 1, 2)
            Ham = np.swapaxes(Ham, 1, 2)

        # FIXME: cell and wann_centers
        if has_atom:
            numbers = root.variables[prefix + "atomic_numbers"][:]
            xcart = root.variables[prefix + "atomic_xcart"][:]
            # xcart = root.variables[prefix + 'xcart'][:]
            cell = root.variables[prefix + "cell"][:]
            atoms = Atoms(numbers, cell=cell, positions=xcart)
        else:
            atoms = None

        try:
            wann_centers = root.variables[prefix + "wannier_center_xred"][:]
        except:
            print(
                f"Warning: wannier centers {prefix+'wannier_center_xred'} not found, using 0 instead."
            )
            wann_centers = np.zeros((nwann, ndim))
        return LWF(
            wannR, Ham, Rlist, cell=np.eye(3), wann_centers=wann_centers, atoms=atoms
        )

    def make_supercell(self, sc_maker=None, sc_matrix=None):
        from minimulti.utils.supercell import SupercellMaker

        if sc_maker is None:
            sc_maker = SupercellMaker(sc_matrix)
        if self.atoms is not None:
            sc_atoms = sc_maker.sc_atoms(self.atoms)
        sc_Rlist, sc_HR = sc_maker.sc_Rlist_HR(
            self.Rlist, self.HwannR, n_basis=self.nwann
        )
        return sc_atoms, sc_Rlist, sc_HR

    def get_num_orbitals(self):
        return self.nwann

    def save_txt(self, fname):
        with open(fname, "w") as myfile:
            myfile.write(f"Number_of_R: {self.nR}\n")
            myfile.write(f"Number_of_Wannier_functions: {self.nwann}\n")
            # myfile.write(f"Cell parameter: {self.cell}\n")
            myfile.write("Hamiltonian:  \n" + "=" * 60 + "\n")
            for iR, R in enumerate(self.Rlist):
                myfile.write(f"index of R: {iR}.  R = {R}\n")
                d = self.HwannR[iR]
                for i in range(self.nwann):
                    for j in range(self.nwann):
                        myfile.write(
                            f"R = {R}, i = {i}, j={j} :: H(i,j,R)= {d[i,j]:.4f} \n"
                        )
                myfile.write("-" * 60 + "\n")

    def modify_evals(self, func, kmesh):
        ret = copy.deepcopy(self)
        mdf = HamModifier(HR=self.HwannR, Rlist=self.Rlist)
        ret.HwannR, ret.Rlist = mdf.modify(func, kmesh)
        return ret

    def force_ASR(self):
        iR0 = self.find_iR_at_zero()
        sumHR = np.sum(self.HwannR, axis=(0, 1))
        self.HwannR[iR0] -= np.diag(sumHR)

    def force_ASR_kspace(self, kmesh):
        self.HwannR = force_ASR_kspace(self.HwannR, self.Rlist, kmesh)
        return self


class LWF_COHP(LWF):
    def solve_all(self, k_list, eig_vectors=True, total_ham=True):
        nk = len(k_list)
        evals, evecs, hams = (
            np.zeros((self.nwann, nk)),
            np.zeros((self.nwann, nk, self.nwann), dtype=complex),
            np.zeros((nk, self.nwann, self.nwann), dtype=complex),
        )
        for ik, k in enumerate(k_list):
            evalue, evec, ham = self.solve_wann_k(k, ham=True)
            evals[:, ik] = evalue
            evecs[:, ik, :] = evec.T
            hams[ik, :, :] = ham
        return evals, evecs, hams

    @staticmethod
    def load_nc(fname, prefix="wann_"):
        root = Dataset(fname, "r")
        ndim = root.dimensions["ndim"].size
        nR = root.dimensions["nR"].size
        nwann = root.dimensions[prefix + "nwann"].size
        nbasis = root.dimensions[prefix + "nbasis"].size

        Rlist = root.variables[prefix + "Rlist"][:]
        ifc_real = root.variables[prefix + "ifc_real"][:]
        ifc_imag = root.variables[prefix + "ifc_imag"][:]
        ifc = ifc_real + ifc_imag * 1.0j

        wannR_real = root.variables[prefix + "wannier_function_real"][:]
        wannR_imag = root.variables[prefix + "wannier_function_imag"][:]
        wannR = wannR_real + wannR_imag * 1.0j

        return LWF_COHP(
            wannR, ifc, Rlist, cell=np.eye(3), wann_centers=np.zeros((nwann, ndim))
        )


def test_modify_evals():
    fname = "/home/hexu/projects/LAO/scripts/Downfolded_hr.nc"
    wf = LWF.load_nc(fname)
    ax = plot_band(wf, color="black")
    new_wf = wf.modify_evals(func=lambda x: x + 1.0, kmesh=[5, 5, 5])

    new_wf.write_lwf_nc("shifted_lwf.nc")
    ax = plot_band(new_wf, color="blue", ax=ax)
    plt.show()


def merge_two_LWF(wf1, wf2):
    nwann1 = wf1.nwann
    nwann2 = wf2.nwann
    nwann = nwann1 + nwann2
    nbasis = wf1.nbasis
    nR = wf1.nR
    wannR = np.zeros((nR, nbasis, nwann), dtype=complex)
    HwannR = np.zeros((nR, nwann, nwann), dtype=complex)
    wannR[:, :, :nwann1] = wf1.wannR
    wannR[:, :, nwann1:] = wf2.wannR
    HwannR[:, :nwann1, :nwann1] = wf1.HwannR
    HwannR[:, nwann1:, nwann1:] = wf2.HwannR
    Rlist = wf1.Rlist
    cell = wf1.cell
    wann_centers = np.vstack([wf1.wann_centers, wf2.wann_centers])
    lwf = LWF(
        wannR,
        HwannR,
        Rlist,
        cell,
        wann_centers,
        atoms=wf1.atoms,
        fortran_order=False,
        twobody_terms=None,
    )
    # TODO: add onebody terms and twobody_terms
    return lwf


if __name__ == "__main__":
    test_modify_evals()
