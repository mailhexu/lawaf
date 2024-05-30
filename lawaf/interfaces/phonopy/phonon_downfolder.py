from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
import os
from lawaf.interfaces.downfolder import Lawaf
from lawaf.mathutils.evals_freq import freqs_to_evals
from .phonopywrapper import PhonopyWrapper
from lawaf.wannierization.wannierizer import Amnk_to_Hk, Hk_to_Hreal
from lawaf.mathutils.kR_convert import k_to_R, R_to_k, R_to_onek
from lawaf.mathutils.align_evecs import align_evecs


class PhononDownfolder(Lawaf):
    def __init__(self, model, atoms=None, params=None):
        super().__init__(model, params=params)
        self.model = model
        if atoms is not None:
            self.atoms = atoms
        else:
            try:
                self.atoms = self.model.atoms
            except Exception:
                self.atoms = None
        self.params = params


class PhonopyDownfolder(PhononDownfolder):
    def __init__(
        self, phonon=None, mode="dm", params=None, is_nac=False, *argv, **kwargs
    ):
        """
        Parameters:
        ========================================
        folder: folder of siesta calculation
        fdf_file: siesta input filename
        """
        try:
            import phonopy
        except ImportError:
            raise ImportError("phonopy is needed. Do you have phonopy installed?")
        if phonon is None:
            phonon = phonopy.load(*argv, **kwargs)
        self.params = params
        self.mode = mode
        self.factor = 524.16  # to cm-1
        self.convert_DM_parameters()
        self.is_nac = is_nac
        model = PhonopyWrapper(phonon, mode=mode, is_nac=self.is_nac)
        super().__init__(model, atoms=model.atoms, params=self.params)

    def convert_DM_parameters(self):
        if self.mode == "dm":
            # self.params["mu"] = freqs_to_evals(self.params["mu"], factor=self.factor)
            # self.params["sigma"] = freqs_to_evals(
            #    self.params["sigma"], factor=self.factor
            # )
            p = self.params["weight_func_params"]
            p = [freqs_to_evals(pe, factor=self.factor) for pe in p]
            self.params["weight_func_params"] = p
            print(self.params["weight_func_params"])

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
        **params,
    ):
        self.params.update(params)
        self.atoms = self.model.atoms
        self.ewf = self.builder.get_wannier(Rlist=self.Rlist)
        if post_func is not None:
            post_func(self.ewf)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except Exception as E:
            print(E)
            pass
        if write_hr_txt is not None:
            self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        if write_hr_nc is not None:
            # self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
            self.ewf.write_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        return self.ewf


@dataclass
class NACLWF:
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
    """

    born: np.ndarray = None
    dielectric: np.ndarray = None
    factor: float = None
    masses: np.ndarray = None
    Rlist: np.ndarray = None
    wannR: np.ndarray = None
    HR_noNAC: np.ndarray = None
    HR_short: np.ndarray = None
    HR_total: np.ndarray = None
    NAC_phonon: PhonopyWrapper = None
    nac: bool = True
    kpts: np.ndarray = None
    wann_centers: np.ndarray = None

    def __post_init__(self):
        self.natoms = self.born.shape[0]
        self.nwann = self.wannR.shape[2]
        self.nkpt = self.wannR.shape[0]
        self.nR = self.Rlist.shape[0]
        self.born_wann = self.get_born_wann()
        self.get_masses_wann()
        self.get_disp_wann()
        self.check_normalization()

        # nac_q = self._get_charge_sum(q=[0, 0, 0.001])
        self.split_short_long_wang()

    def set_nac(self, nac=True):
        self.nac = nac

    def remove_phase(self, Hk, k):
        """
        remove the phase of the R-vector
        """
        self.dr = self.wann_centers[None, :, :] - self.wann_centers[:, None, :]
        phase = np.exp(-2.0j * np.pi * np.einsum("ijk, k->ij", self.dr, k))
        return Hk * phase

    def get_born_wann(self):
        """
        get the Born effective charges in Wannier space.
        """
        # born = self.born.reshape(3, self.natoms * 3)
        # self.born_wan = np.einsum("Rji,kj->ik", self.wannR**2, born).real
        # born = self.born.swapaxes(1,2).reshape( self.natoms * 3, 3)
        born = self.born.reshape(self.natoms * 3, 3)
        self.born_wan = np.einsum("Rji,jk->ik", self.wannR**2, born).real
        print(self.born_wan)

    def get_masses_wann(self):
        """
        get the masses of lattice wannier functions.
        m_wann_i = sum_Rj m_j * WannR_Rji
        """
        masses = np.repeat(self.masses, 3).real
        self.masses_lwf = np.einsum("j,Rji->i", masses, self.wannR**2)
        print(self.masses_lwf)

    def check_normalization(self):
        """
        check the normalization of the LWF.
        """
        norm = np.sum(self.wannR**2, axis=(0, 1)).real
        print(f"Norm of Wannier functions: {norm}")

    def get_disp_wann(self):
        """
        get the displacement of the LWF.
        """
        masses = np.repeat(self.masses, 3)
        self.disp_wannR = self.wannR / np.sqrt(masses)[None, :, None]

    def get_volume(self):
        """
        get the volume of the unit cell.
        """
        return np.linalg.det(self.NAC_phonon.atoms.get_cell())

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
        # Hks_tot = k_to_R(self.kpts, self.Rlist, self.HR_total)
        self.nkpt = len(self.kpts)
        print(self.nwann)
        Hks_short = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)
        for ik, kpt in enumerate(self.kpts):
            Hk_tot = self.get_Hk_nac_total(kpt)
            Hk_long = self.get_Hk_wang_long(kpt)
            Hks_short[ik] = Hk_tot - Hk_long
        HRs_short = k_to_R(self.kpts, self.Rlist, Hks_short)
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


class NACPhonopyDownfolder(PhonopyDownfolder):
    def __init__(
        self, phonon=None, phonon_NAC=None, mode="dm", params=None, *argv, **kwargs
    ):
        """
        Parameters:
        """
        try:
            import phonopy
        except ImportError:
            raise ImportError("phonopy is needed. Do you have phonopy installed?")
        if phonon is None:
            phonon = phonopy.load(*argv, **kwargs, is_nac=True)
        # model = PhonopyWrapper(phonon, mode=mode, is_nac=True)
        super().__init__(
            phonon=phonon, mode=mode, params=params, is_nac=True, *argv, **kwargs
        )

        self.model.get_nac_params()
        self.model_NAC = self.model

        # if phonon_NAC is None:
        #    phonon_NAC = phonopy.load(*argv, **kwargs)
        # self.model_NAC = PhonopyWrapper(phonon_NAC, mode=mode, is_nac=True)

        self.born, self.dielectric, self.factor = self.model.get_nac_params()

        self.born = self.model.born
        self.dielectric = self.model.dielectric
        self.factor = self.model.factor

        self.is_nac = True
        self.set_nac_params(
            # self.model_NAC.born, self.model_NAC.dielectric, self.model_NAC.factor
            self.model.born,
            self.model.dielectric,
            self.model.factor,
        )
        print(self.model.is_nac)

    def get_Hks_with_nac(self, q):
        """
        get the dynmaical matrix at q with NAC.
        params:
            q: q-vector
        return:
            Htotal, Hshort, Hlong, eigenvalues, eigenvectors
        """
        evals, evecs, Hk, Hshort, Hlong = self.model_NAC.solve(q)
        return evals, evecs, Hk, Hshort, Hlong

    def set_nac_params(self, born, dielectic, factor):
        """set  Hamiltonians including splited Hks, Hshorts and Hlongs."""
        self.born = born
        self.dielectic = dielectic
        self.factor = factor

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
        **params,
    ):
        # self.params.update(params)
        # if "post_func" in self.params:
        #    self.params.pop("post_func")
        self.atoms = self.model.atoms
        self.builder.prepare()
        # compute the Amn matrix from phonons without NAC
        Amn = self.builder.get_Amn()
        # compute the Wannier functions and the Hamiltonian in k-space without NAC
        # wannk: (nkpt, nbasis, nwann)
        wannk, Hwannk_noNAC = self.builder.get_wannk_and_Hk()
        HwannR_noNAC = k_to_R(
            self.kpts, self.Rlist, Hwannk_noNAC, kweights=self.kweights
        )

        wannR = k_to_R(self.kpts, self.Rlist, wannk, kweights=self.kweights)
        # prepare the H and the eigens for all k-points.
        evals_nac, evecs_nac, Hk_tot, Hk_short, Hk_long = self.model_NAC.solve_all(
            self.kpts, output_H=True
        )

        # compute the short range Hamiltonian in Wannier space
        Hwannk_short = self.get_Hwannk_short(wannk, Hk_short, evecs_nac)
        HwannR_short = self.get_HwannR_short(
            Hwannk_short, self.kpts, self.Rlist, kweights=self.kweights
        )

        Hwannk_total = self.get_Hwannk_short(wannk, Hk_tot, evecs_nac)
        HwannR_total = self.get_HwannR_short(
            Hwannk_total, self.kpts, self.Rlist, kweights=self.kweights
        )

        wann_centers = get_wannier_centers(
            wannR, self.Rlist, self.atoms.get_scaled_positions()
        )
        print(wann_centers)

        # save the lwf model into a NACLWF object
        self.ewf = NACLWF(
            born=self.born,
            dielectric=self.dielectic,
            factor=self.factor,
            masses=self.atoms.get_masses(),
            Rlist=self.Rlist,
            wannR=wannR,
            HR_noNAC=HwannR_noNAC,
            HR_short=HwannR_short,
            HR_total=HwannR_total,
            NAC_phonon=self.model_NAC,
            kpts=self.kpts,
            wann_centers=wann_centers,
        )

        # if post_func is not None:
        #    post_func(self.ewf)
        # if not os.path.exists(output_path):
        #    os.makedirs(output_path)
        # try:
        #    self.save_info(output_path=output_path)
        # except:
        #    pass
        # if write_hr_txt is not None:
        #    self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        # if write_hr_nc is not None:
        #    # self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        #    self.ewf.write_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        # return self.ewf

    def get_Hwannk_short(self, wannk=None, Hk_short=None, evecs=None):
        """
        compute theh Hk_short in Wannier space
        params:
            wannk: Wannier functions in k-space, (nkpt, nbasis, nwann)
            Hk_short: short range Hamiltonian in k-space
        """
        Hk_wann_short = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)
        for ik in range(self.nkpt):
            # Hk_wann_short[ik] = wannk[ik].conj().T @ evecs[ik].conj().T@ Hk_short[ik] @ evecs[ik] @ wannk[ik]
            Hk_wann_short[ik] = wannk[ik].conj().T @ Hk_short[ik] @ wannk[ik]
        return Hk_wann_short

    def get_HwannR_short(
        self, Hk_wann_short=None, kpts=None, Rlist=None, kweights=None
    ):
        """
        compute the HR_short in Wannier space
        """
        if Hk_wann_short is None:
            Hk_wann_short = self.get_Hwannk_short()
        HwannR_short = k_to_R(kpts, Rlist, Hk_wann_short, kweights=kweights)
        return HwannR_short

    def get_wannk_interpolated(self, qpt):
        """
        Interpolate Wannier functions from real space to k-space.
        """
        wannk = R_to_onek(qpt, self.Rlist, self.ewf.wannR)
        return wannk

    def get_wannier_nac(self, Rlist=None):
        """
        Calculate Wannier functions but using non-analytic correction.
        """
        self.prepare()
        self.get_Amn()
        self.get_wannk_and_Hk_nac()
        if Rlist is not None:
            lwf = self.k_to_R(Rlist=Rlist)
            # lwf.atoms = copy.deepcopy(self.atoms)
        lwf.set_born_from_full(self.born, self.dielectic, self.factor)
        return lwf


class PhonopyDownfolderWrapper:
    downfolder = None
    solver: PhonopyWrapper = None

    def __init__(self, downfolder: PhonopyDownfolder):
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


def get_wannier_centers(wannR, Rlist, positions):
    nR = len(Rlist)
    nwann = wannR.shape[2]
    wann_centers = np.zeros((nwann, 3), dtype=float)
    natom = len(positions)
    p = np.kron(positions, np.ones((3, 1)))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        # wann_centers += (c.conj() * c).real @ positions + R[None, :]
        wann_centers += np.einsum("ij, ik-> jk", (c.conj() * c).real, p + R[None, :])
    print(f"Wannier Centers: {wann_centers}")
    return wann_centers
