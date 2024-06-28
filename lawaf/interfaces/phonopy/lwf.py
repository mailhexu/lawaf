import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from lawaf.mathutils.kR_convert import k_to_R, R_to_k, R_to_onek
from .phonopywrapper import PhonopyWrapper
from ase import Atoms


@dataclass
class LWF:
    """
    LWF
    elements:
        born: Born effective charges. (natoms, 3, 3)
        dielectric: dielectric tensor. (3, 3)
        factor: factor to convert the unit.
        Rlist: list of R-vectors. (nR, 3)
        wannR: Wannier functions in real space
        HR_total: total Hamiltonian in real space
        kpts: k-points
        kweights: weights of k-points
        wann_centers: centers of Wannier functions.
        wann_masses: masses of Wannier functions.
    """

    factor: float = None
    Rlist: np.ndarray = None
    Rdeg: np.ndarray = None
    wannR: np.ndarray = None
    HR_total: np.ndarray = None
    kpts: np.ndarray = None
    kweights: np.ndarray = None
    wann_centers: np.ndarray = None
    wann_masses: np.ndarray = None
    atoms: Atoms = None

    def __post_init__(self):
        self.nR, self.nbasis, self.nwann = self.wannR.shape
        self.natoms = self.nbasis // 3
        self.nR = self.Rlist.shape[0]
        self.check_normalization()
        self.get_masses_wann()
        self.get_disp_wann()

    def write_to_netcdf(self, filename, atoms=None):
        """
        write the LWF to netcdf file.
        """
        import xarray as xr

        print(f"wann_masses: {self.wann_masses}")
        ds = xr.Dataset(
            {
                "factor": self.factor,
                "Rlist": (["nR", "dim"], self.Rlist),
                "Rdeg": (["nR"], self.Rdeg),
                "wannR": (
                    ["ncplx", "nR", "nbasis", "nwann"],
                    np.stack([np.real(self.wannR), np.imag(self.wannR)], axis=0),
                ),
                "Hwann_R": (
                    ["ncplx", "nR", "nwann", "nwann"],
                    np.stack([np.real(self.HR_total), np.imag(self.HR_total)], axis=0),
                ),
                "kpts": (["nkpt", "dim"], self.kpts),
                "kweights": (["nkpt"], self.kweights),
                "wann_centers": (["nwann", "dim"], self.wann_centers),
                "wann_masses": (["nwann"], self.wann_masses.real),
            }
        )
        ds.to_netcdf(filename, group="lwf", mode="w")

        if atoms is not None:
            ds = xr.Dataset(
                {
                    "positions": (["natom", "dim"], atoms.get_positions()),
                    "masses": (["natom"], atoms.get_masses()),
                    "cell": (["dim", "dim"], atoms.get_cell()),
                    "atomic_numbers": (["natom"], atoms.get_atomic_numbers()),
                }
            )
        ds.to_netcdf(filename, group="atoms", mode="a")

    @classmethod
    def load_from_netcdf(cls, filename):
        """
        load the LWF from netcdf file.
        """
        import xarray as xr

        ds = xr.open_dataset(filename, group="lwf")
        wannR = ds["wannR"].values[0] + 1j * ds["wannR"].values[1]
        HR_total = ds["Hwann_R"].values[0] + 1j * ds["Hwann_R"].values[1]

        ds_atoms = xr.open_dataset(filename, group="atoms")
        atoms = Atoms(
            positions=ds_atoms["positions"].values,
            masses=ds_atoms["masses"].values,
            cell=ds_atoms["cell"].values,
            atomic_numbers=ds_atoms["atomic_numbers"].values,
        )

        return cls(
            factor=ds.attrs["factor"],
            masses=ds.attrs["masses"],
            Rlist=ds["Rlist"].values,
            Rdeg=ds["Rdeg"].values,
            wannR=wannR,
            HR_total=HR_total,
            kpts=ds["kpts"].values,
            kweights=ds["kweights"].values,
            wann_centers=ds["wann_centers"].values,
        )

    def save_txt(self, fname):
        with open(fname, "w") as myfile:
            myfile.write(f"Number_of_R: {self.nR}\n")
            myfile.write(f"Number_of_Wannier_functions: {self.nwann}\n")
            # myfile.write(f"Cell parameter: {self.cell}\n")

            myfile.write("Wannier functions:  \n" + "=" * 60 + "\n")
            for iR, R in enumerate(self.Rlist):
                myfile.write(f"index of R: {iR}.  R = {R}\n")
                d = self.wannR[iR]
                for i in range(self.nwann):
                    for j in range(self.nbasis):
                        myfile.write(f"i = {i}, j={j} :: WannR(i,j,R)= {d[j,i]:.4f} \n")
                myfile.write("-" * 60 + "\n")

            myfile.write("Hamiltonian:  \n" + "=" * 60 + "\n")
            for iR, R in enumerate(self.Rlist):
                myfile.write(f"index of R: {iR}.  R = {R}\n")
                d = self.HR_total[iR]
                for i in range(self.nwann):
                    for j in range(self.nwann):
                        myfile.write(
                            f"R = {R}, i = {i}, j={j} :: H(i,j,R)= {d[i,j]:.4f} \n"
                        )
                myfile.write("-" * 60 + "\n")

    def remove_phase(self, Hk, k):
        """
        remove the phase of the R-vector
        """
        self.dr = self.wann_centers[None, :, :] - self.wann_centers[:, None, :]
        phase = np.exp(-2.0j * np.pi * np.einsum("ijk, k->ij", self.dr, k))
        return Hk * phase

    def get_masses_wann(self):
        """
        get the masses of lattice wannier functions.
        m_wann_i = sum_Rj m_j * WannR_Rji
        """
        masses = np.repeat(self.atoms.get_masses(), 3)
        self.wann_masses = (
            np.einsum("j,Rji->i", masses, self.wannR * self.wannR.conj())
            / self.wann_norm
        )

    def check_normalization(self):
        """
        check the normalization of the LWF.
        """
        self.wann_norm = np.sum(self.wannR * self.wannR.conj(), axis=(0, 1)).real
        print(f"Norm of Wannier functions: {self.wann_norm}")

    def get_disp_wann(self):
        """
        get the displacement of the LWF.
        """
        masses = np.repeat(self.atoms.get_masses(), 3)
        self.disp_wannR = self.wannR / np.sqrt(masses)[None, :, None]

    def get_volume(self):
        """
        get the volume of the unit cell.
        """
        return np.linalg.det(self.NAC_phonon.atoms.get_cell())

    def get_Hk(self, kpt):
        """
        get the Hamiltonian at k-point.
        """
        Hk = R_to_onek(kpt, self.Rlist, self.HR_total)
        return Hk

    def solve_k(self, kpt):
        """
        solve the Hamiltonian at k-point with NAC.
        """
        # if np.linalg.norm(kpt) < 1e-6:
        #    Hk = self.get_Hk_noNAC(kpt)
        # else:
        Hk = self.get_Hk(kpt)
        evals, evecs = eigh(Hk)
        return evals, evecs

    def solve_all(self, kpts):
        """
        solve the Hamiltonian at all k-points with NAC.
        """
        evals = []
        evecs = []
        for k in kpts:
            e, v = self.solve_k(k)
            evals.append(e)
            evecs.append(v)
        return np.array(evals), np.array(evecs)


@dataclass
class NACLWF(LWF):
    """
    LWF with NAC
    elements:
        born: Born effective charges. (natoms, 3, 3)
        dielectric: dielectric tensor. (3, 3)
        factor: factor to convert the unit.
        masses: masses of atoms. (natoms)
        Rlist: list of R-vectors. (nR, 3)
        wannR: Wannier functions in real space
        HR_noNAC: Hamiltonian in real space without NAC
        HR_short: short range Hamiltonian in real space
        HR_total: total Hamiltonian in real space
        NAC_phonon: PhonopyWrapper object
        nac: whether to include NAC
        kpts: k-points
        kweights: weights of k-points
        wann_centers: centers of Wannier functions.
    """

    born: np.ndarray = None
    dielectric: np.ndarray = None
    wannR: np.ndarray = None
    HR_noNAC: np.ndarray = None
    HR_short: np.ndarray = None
    HR_total: np.ndarray = None
    NAC_phonon: PhonopyWrapper = None
    nac: bool = True
    kpts: np.ndarray = None
    kweights: np.ndarray = None
    wann_centers: np.ndarray = None

    def __post_init__(self):
        self.natoms = self.born.shape[0]
        self.nwann = self.wannR.shape[2]
        self.nkpt = self.wannR.shape[0]
        self.nR = self.Rlist.shape[0]
        self.check_normalization()
        self.born_wann = self.get_born_wann()
        self.get_masses_wann()
        self.get_disp_wann()

        # nac_q = self._get_charge_sum(q=[0, 0, 0.001])
        self.split_short_long_wang()

    def set_nac(self, nac=True):
        self.nac = nac

    def get_born_wann(self):
        """
        get the Born effective charges in Wannier space.
        """
        # born = self.born.reshape(3, self.natoms * 3)
        # self.born_wan = np.einsum("Rji,kj->ik", self.wannR**2, born).real
        # born = self.born.swapaxes(1,2).reshape( self.natoms * 3, 3)
        born = self.born.reshape(self.natoms * 3, 3)
        self.born_wan = np.einsum(
            "Rji,jk->ik", self.wannR**2, born
        )  # /self.wann_norm[None, :]
        print(self.born_wan)

    def get_constant_factor_wang(self, q):
        # unit_conversion * 4.0 * np.pi / volume / np.dot(q.T, np.dot(dielectric, q))
        return (
            self.factor
            * 4.0
            * np.pi
            / np.dot(q.T, np.dot(self.dielectric, q))
            / self.get_volume()
        )

    def get_Hk_wang_long(self, qpt):
        if np.linalg.norm(qpt) < 1e-6:
            return np.zeros_like(self.HR_total[0])
        nac_q = self._get_charge_sum(qpt)
        dd = nac_q * self.get_constant_factor_wang(qpt)
        mmat = np.sqrt(np.outer(self.masses_lwf, self.masses_lwf))
        return self.remove_phase(dd / mmat, qpt)
        # return dd / mmat

    def split_short_long_wang(self):
        self.nkpt = len(self.kpts)
        Hks_short = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)
        for ik, kpt in enumerate(self.kpts):
            Hk_tot = self.get_Hk_nac_total(kpt)
            Hk_long = self.get_Hk_wang_long(kpt)
            Hks_short[ik] = Hk_tot - Hk_long
        HRs_short = k_to_R(
            self.kpts, self.Rlist, Hks_short, kweights=self.kweights, Rdeg=self.Rdeg
        )
        self.HRs_wang_short = HRs_short

    def _get_charge_sum(self, q):
        nac_q = np.zeros((self.nwann, self.nwann), dtype="double", order="C")
        A = np.dot(self.born_wan, q)
        nac_q = np.outer(A, A)
        # for i in range(num_atom):
        #    for j in range(num_atom):
        #        nac_q[i, j] = np.outer(A[i], A[j])
        return nac_q

    def get_Hk_short(self, kpt):
        """
        get the short range Hamiltonian at k-point.
        """
        Hk_short = R_to_onek(kpt, self.Rlist, self.HR_short)
        return Hk_short

    def get_wannk(self, kpt):
        """
        get the Wannier functions at k-point.
        """
        wannk = R_to_onek(kpt, self.Rlist, self.wannR)
        return wannk

    def get_Hk_long(self, kpt):
        """
        get the long range Hamiltonian at k-point.
        """
        evals, evecs, Hk, Hk_short, Hk_long = self.NAC_phonon.solve(kpt, output_H=True)
        wannk = self.get_wannk(kpt)
        Hwannk_long = wannk.conj().T @ Hk_long @ wannk
        return Hwannk_long

    def get_Hk_noNAC(self, kpt):
        """
        get the Hamiltonian at k-point without NAC.
        """
        Hk_noNAC = R_to_onek(kpt, self.Rlist, self.HR_noNAC)
        return Hk_noNAC

    def get_Hk_nac(self, kpt):
        evals, evecs, Hk, Hk_short, Hk_long = self.NAC_phonon.solve(kpt, output_H=True)
        wannk = self.get_wannk(kpt)
        Hwannk = wannk.conj().T @ Hk @ wannk
        return Hwannk

    def get_Hk_nac_total(self, kpt):
        return R_to_onek(kpt, self.Rlist, self.HR_total)

    def get_Hk(self, kpt, method="wang"):
        """
        get the Hamiltonian at k-point.
        """
        if self.nac:
            # return self.get_Hk_nac_total(kpt)
            # return self.get_Hk_nac(kpt)
            if method == "wang":
                Hk_short = R_to_onek(kpt, self.Rlist, self.HRs_wang_short)
                # Hk_short = self.get_Hk_nac_total(kpt)
                Hk_long = self.get_Hk_wang_long(kpt)
                # Hk_long =0
                Hk_tot = Hk_short + Hk_long
                return Hk_tot
            else:
                Hk_short = self.get_Hk_short(kpt)
                Hk_long = self.get_Hk_long(kpt)
                Hk_tot = Hk_short + Hk_long
                return Hk_tot

        else:
            return self.get_Hk_noNAC(kpt)
        # return self.get_Hk_nac(kpt)

    def solve_k(self, kpt):
        """
        solve the Hamiltonian at k-point with NAC.
        """
        # if np.linalg.norm(kpt) < 1e-6:
        #    Hk = self.get_Hk_noNAC(kpt)
        # else:
        Hk = self.get_Hk(kpt)
        evals, evecs = eigh(Hk)
        return evals, evecs

    def solve_all(self, kpts):
        """
        solve the Hamiltonian at all k-points with NAC.
        """
        evals = []
        evecs = []
        for k in kpts:
            e, v = self.solve_k(k)
            evals.append(e)
            evecs.append(v)
        return np.array(evals), np.array(evecs)


def get_wannier_centers(wannR, Rlist, positions, Rdeg):
    nR = len(Rlist)
    nwann = wannR.shape[2]
    wann_centers = np.zeros((nwann, 3), dtype=float)
    natom = len(positions)
    p = np.kron(positions, np.ones((3, 1)))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        # wann_centers += (c.conj() * c).real @ positions + R[None, :]
        wann_centers += (
            np.einsum("ij, ik-> jk", (c.conj() * c).real, p + R[None, :]) * Rdeg[iR]
        )
    # print(f"Wannier Centers: {wann_centers}")
    return wann_centers


class PhonopyDownfolderWrapper:
    downfolder = None
    solver: PhonopyWrapper = None

    def __init__(self, downfolder):
        self.downfolder = downfolder.builder
        self.solver = downfolder.model

    def solve_k(self, k):
        evals, evecs, Hk, Hshort, Hlong = self.solver.solve(k)
        # evals, evecs  =self.solver.solve(k)[:2]
        # return evals, evecs
        Amn = self.downfolder.get_Amn_psi(evecs)
        h = Amn.T.conj() @ np.diag(evals) @ Amn
        # h= Amn.T.conj()@evecs.T.conj()@Hshort@evecs@Amn
        evals, evecs = eigh(h)
        return evals, evecs

    def get_fermi_level(self):
        return 0.0

    def solve_all(self, kpts):
        evals = []
        evecs = []
        for k in kpts:
            e, v = self.solve_k(k)
            evals.append(e)
            evecs.append(v)
        return np.array(evals), np.array(evecs)
