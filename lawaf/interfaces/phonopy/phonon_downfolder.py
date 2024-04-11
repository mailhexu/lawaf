import numpy as np
from scipy.linalg import eigh
import os
from lawaf.interfaces.downfolder import Lawaf, make_builder
from lawaf.mathutils.evals_freq import freqs_to_evals
from .phonopywrapper import PhonopyWrapper



class PhononDownfolder(Lawaf):
    def __init__(self, model, atoms=None):
        self.model = model
        if atoms is not None:
            self.atoms = atoms
        else:
            try:
                self.atoms = self.model.atoms
            except Exception:
                self.atoms = None
        self.params = {}


class PhonopyDownfolder(PhononDownfolder):
    def __init__(self, phonon=None, mode="dm", has_nac=False, *argv, **kwargs):
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
        self.has_nac = has_nac
        model = PhonopyWrapper(phonon, mode=mode, has_nac=has_nac)
        super().__init__(model, atoms=model.atoms)

        self.mode=mode
        self.factor = 524.16 # to cm-1
        self.convert_DM_parameters()

    def convert_DM_parameters(self):
        if self.mode=="dm":
            self.params["mu"]= freqs_to_evals(self.params["mu"], factor=self.factor)
            self.params["sigma"]= freqs_to_evals(self.params["sigma"], factor=self.factor)

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
        **params,
    ):
        self.params.update(params)
        if "post_func" in self.params:
            self.params.pop("post_func")
        self.builder = make_builder(self.model, **self.params)
        self.atoms = self.model.atoms
        if self.has_nac:
            self.builder.set_nac_params(
                self.model.born, self.model.dielectric, self.model.factor
            )
            self.ewf = self.builder.get_wannier_nac()
        else:
            self.ewf = self.builder.get_wannier()
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
        #h= Amn.T.conj()@evecs.T.conj()@Hshort@evecs@Amn
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
