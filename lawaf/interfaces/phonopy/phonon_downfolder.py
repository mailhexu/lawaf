import os

import numpy as np

from lawaf.interfaces.downfolder import Lawaf
from lawaf.mathutils.evals_freq import freqs_to_evals
from lawaf.mathutils.kR_convert import R_to_onek, k_to_R

from .lwf import LWF, NACLWF
from .phonopywrapper import PhonopyWrapper

__all__ = ["PhononDownfolder", "PhonopyDownfolder", "NACPhonopyDownfolder"]


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
        self.mode = mode
        self.factor = 524.16  # to cm-1
        self.is_nac = is_nac
        model = PhonopyWrapper(phonon, mode=mode, is_nac=self.is_nac)
        super().__init__(model, atoms=model.atoms, params=params)

    def convert_DM_parameters(self):
        """
        convert the parameters of the dynamical matrix. Unit from frequency cm-1 to
        eigenvalues in eV.
        """
        if self.mode.lower() == "dm":
            p = self.params.weight_func_params
            if p is not None:
                print(f"Converted DM parameters from: {p}")
                p = [freqs_to_evals(pe, factor=self.factor) for pe in p]
                self.params.weight_func_params = p
                print(f"Converted DM parameters to: {p}")

    def process_parameters(self):
        self.convert_DM_parameters()

    # def downfold(
    #    self,
    #    post_func=None,
    #    output_path="./",
    #    write_hr_nc="LWF.nc",
    #    write_hr_txt="LWF.txt",
    # ):
    #    # self.params.update(params)
    #    self.atoms = self.model.atoms
    #    self.lwf = self.builder.get_wannier(Rlist=self.Rlist, Rdeg=self.Rdeg)
    #    if post_func is not None:
    #        post_func(self.lwf)
    #    if not os.path.exists(output_path):
    #        os.makedirs(output_path)
    #    try:
    #        self.save_info(output_path=output_path)
    #    except Exception:
    #        pass
    #    if write_hr_txt is not None:
    #        self.lwf.save_txt(os.path.join(output_path, write_hr_txt))
    #    if write_hr_nc is not None:
    #        self.lwf.write_nc(os.path.join(output_path, write_hr_nc))
    #    return self.lwf

    def downfold(self, output_path="./", write_hr_nc="LWF.nc", write_hr_txt="LWF.txt"):
        self._prepare_data()
        self.atoms = self.model.atoms
        self.builder.prepare()
        # compute the Amn matrix from phonons without NAC
        self.builder.get_Amn()
        # compute the Wannier functions and the Hamiltonian in k-space without NAC
        # wannk: (nkpt, nbasis, nwann)
        wannk, Hwannk = self.builder.get_wannk_and_Hk()
        HwannR = k_to_R(
            self.kpts, self.Rlist, Hwannk, kweights=self.kweights, Rdeg=self.Rdeg
        )

        wannR = k_to_R(
            self.kpts, self.Rlist, wannk, kweights=self.kweights, Rdeg=self.Rdeg
        )

        wann_centers = get_wannier_centers(
            wannR, self.Rlist, self.atoms.get_scaled_positions(), Rdeg=self.Rdeg
        )
        print("wannier_centers: ")
        for i in range(self.nwann):
            print(f"{i}: {wann_centers[i]=}")

        # save the lwf model into a NACLWF object
        self.lwf = LWF(
            factor=self.factor,
            Rlist=self.Rlist,
            Rdeg=self.Rdeg,
            wannR=wannR,
            HR_total=HwannR,
            kpts=self.kpts,
            kweights=self.kweights,
            wann_centers=wann_centers,
            atoms=self.atoms,
        )

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except Exception:
            pass
        if write_hr_txt is not None:
            self.lwf.save_txt(os.path.join(output_path, write_hr_txt))
        if write_hr_nc is not None:
            self.lwf.write_to_netcdf(os.path.join(output_path, write_hr_nc))
        return self.lwf


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
            phonon = phonopy.load(*argv, **kwargs, is_nac=False)
        super().__init__(
            phonon=phonon, mode=mode, params=params, is_nac=False, *argv, **kwargs
        )

        self.model.get_nac_params()
        # self.model_NAC = self.model

        if phonon_NAC is None:
            phonon_NAC = phonopy.load(*argv, **kwargs, is_nac=True)
        self.model_NAC = PhonopyWrapper(phonon_NAC, mode=mode, is_nac=True)

        self.born, self.dielectric, self.factor = self.model_NAC.get_nac_params()

        self.is_nac = True
        self.set_nac_params(
            # self.model_NAC.born, self.model_NAC.dielectric, self.model_NAC.factor
            self.model_NAC.born,
            self.model_NAC.dielectric,
            self.model_NAC.factor,
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
        self._prepare_data()
        self.atoms = self.model.atoms
        self.builder.prepare()
        # compute the Amn matrix from phonons without NAC
        self.builder.get_Amn()
        # compute the Wannier functions and the Hamiltonian in k-space without NAC
        # wannk: (nkpt, nbasis, nwann)
        wannk, Hwannk_noNAC = self.builder.get_wannk_and_Hk()
        HwannR_noNAC = k_to_R(
            self.kpts, self.Rlist, Hwannk_noNAC, kweights=self.kweights, Rdeg=self.Rdeg
        )

        wannR = k_to_R(
            self.kpts, self.Rlist, wannk, kweights=self.kweights, Rdeg=self.Rdeg
        )
        # prepare the H and the eigens for all k-points.
        evals_nac, evecs_nac, Hk_tot, Hk_short, Hk_long = self.model_NAC.solve_all(
            self.kpts, output_H=True
        )

        # compute the short range Hamiltonian in Wannier space
        Hwannk_short = self.get_Hwannk_short(wannk, Hk_short, evecs_nac)
        HwannR_short = self.get_HwannR_short(
            Hwannk_short, self.kpts, self.Rlist, kweights=self.kweights, Rdeg=self.Rdeg
        )

        Hwannk_total = self.get_Hwannk_short(wannk, Hk_tot, evecs_nac)
        HwannR_total = self.get_HwannR_short(
            Hwannk_total, self.kpts, self.Rlist, kweights=self.kweights, Rdeg=self.Rdeg
        )

        wann_centers = get_wannier_centers(
            wannR, self.Rlist, self.atoms.get_scaled_positions(), Rdeg=self.Rdeg
        )
        # wann_centers *= 0.0
        print("wannier_centers: ")
        for i in range(self.nwann):
            print(f"{i}: {wann_centers[i]=}")

        # save the lwf model into a NACLWF object
        self.lwf = NACLWF(
            born=self.born,
            dielectric=self.dielectic,
            factor=self.factor,
            Rlist=self.Rlist,
            Rdeg=self.Rdeg,
            wannR=wannR,
            HR_noNAC=HwannR_noNAC,
            HR_short=HwannR_short,
            HR_total=HwannR_total,
            NAC_phonon=self.model_NAC,
            kpts=self.kpts,
            kweights=self.kweights,
            wann_centers=wann_centers,
            atoms=self.atoms,
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
            Hk_wann_short[ik] = wannk[ik].conj().T @ Hk_short[ik] @ wannk[ik]
        return Hk_wann_short

    def get_HwannR_short(
        self, Hk_wann_short=None, kpts=None, Rlist=None, kweights=None, Rdeg=None
    ):
        """
        compute the HR_short in Wannier space
        """
        if Rdeg is None:
            Rdeg = np.ones(len(Rlist))
        if Hk_wann_short is None:
            Hk_wann_short = self.get_Hwannk_short()
        HwannR_short = k_to_R(kpts, Rlist, Hk_wann_short, kweights=kweights, Rdeg=Rdeg)
        return HwannR_short

    def get_wannk_interpolated(self, qpt):
        """
        Interpolate Wannier functions from real space to k-space.
        """
        wannk = R_to_onek(qpt, self.Rlist, self.lwf.wannR)
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


def get_wannier_centers(wannR, Rlist, positions, Rdeg):
    # nR = len(Rlist)
    nwann = wannR.shape[2]
    wann_centers = np.zeros((nwann, 3), dtype=float)
    # natom = len(positions)
    p = np.kron(positions, np.ones((3, 1)))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        # wann_centers += (c.conj() * c).real @ positions + R[None, :]
        wann_centers += (
            np.einsum("ij, ik-> jk", (c.conj() * c).real, p + R[None, :]) * Rdeg[iR]
        )
    return wann_centers


def get_wannier_masses(masses, wannR, Rlist, Rdeg):
    """
    Get the wannier masses from the atomic mases and the Wannier functions.
    """
    nR = len(Rlist)
    nwann = wannR.shape[2]
    nR, nbasis, nwann = wannR.shape
    wann_masses = np.zeros(nwann, dtype=float)
    # masses3 = np.kron(masses, np.ones(3))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        wann_masses += np.einsum("ij, i-> j", (c.conj() * c).real, masses) * Rdeg[iR]
    return wann_masses
