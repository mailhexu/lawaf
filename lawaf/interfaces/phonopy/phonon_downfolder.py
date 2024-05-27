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
    def __init__(self, phonon=None, mode="dm", params=None, *argv, **kwargs):
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
        self.has_nac = False
        model = PhonopyWrapper(phonon, mode=mode, has_nac=self.has_nac)
        print(self.params)
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
    """

    born: np.ndarray = None
    dielectric: np.ndarray = None
    factor: float = None
    Rlist: np.ndarray = None
    wannR: np.ndarray = None
    HR_noNAC: np.ndarray = None
    HR_short: np.ndarray = None
    HR_total: np.ndarray = None
    NAC_phonon: PhonopyWrapper = None
    nac: bool = True

    def set_nac(self, nac=True):
        self.nac = nac

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
        evals, evecs, Hk, Hk_short, Hk_long = self.NAC_phonon.solve(kpt)
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
        evals, evecs, Hk, Hk_short, Hk_long = self.NAC_phonon.solve(kpt)
        wannk = self.get_wannk(kpt)
        Hwannk = wannk.conj().T @ Hk @ wannk
        return Hwannk

    def get_Hk_nac_total(self, kpt):
        return R_to_onek(kpt, self.Rlist, self.HR_total)

    def get_Hk(self, kpt):
        """
        get the Hamiltonian at k-point.
        """
        if self.nac:
            # return self.get_Hk_nac_total(kpt)
            # return self.get_Hk_nac(kpt)
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
        if np.linalg.norm(kpt) < 1e-6:
            Hk = self.get_Hk_noNAC(kpt)
        else:
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
            phonon = phonopy.load(is_nac=False, *argv, **kwargs)
        model = PhonopyWrapper(phonon, mode=mode, has_nac=False)
        super().__init__(phonon=phonon, atoms=model.atoms, params=params)

        if phonon_NAC is None:
            phonon_NAC = phonopy.load(*argv, **kwargs)
        self.model_NAC = PhonopyWrapper(phonon_NAC, mode=mode, has_nac=True)

        self.has_nac = True
        self.set_nac_params(
            self.model_NAC.born, self.model_NAC.dielectric, self.model_NAC.factor
        )

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
        self.has_nac = True
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
            self.kpts
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

        # save the lwf model into a NACLWF object
        self.ewf = NACLWF(
            born=self.born,
            dielectric=self.dielectic,
            factor=self.factor,
            Rlist=self.Rlist,
            wannR=wannR,
            HR_noNAC=HwannR_noNAC,
            HR_short=HwannR_short,
            HR_total=HwannR_total,
            NAC_phonon=self.model_NAC,
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
