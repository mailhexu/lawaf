"""
Wannier Module: For building Wannier functions and Hamiltonians.
"""

import numpy as np
from lawaf.scdm.lwf import LWF
from lawaf.scdm.scdmk import Lawaf, scdm, occupation_func
from ase.dft.kpoints import monkhorst_pack
import os


class SpinDensityMatricesDownfolder(Lawaf):
    def __init__(
        self,
        evals,
        density_matrices,
        positions,
        kmesh,
        nwann,
        Hks,
        Sk=None,
        has_phase=True,
        Rgrid=None,
        sort_cols=True,
        atoms=None,
    ):
        self.nspin = 2
        self.dm = density_matrices
        self.positions = positions
        self.kmesh = kmesh
        self.kpts = monkhorst_pack(kmesh)
        self.nkpt = len(self.kpts)
        self.kweight = [1.0 / self.nkpt] * self.nkpt
        self.nwann = nwann
        self.nbasis = self.dm.shape[-1]
        self.nband = self.nbasis
        self.Hks = Hks
        self.Sk = Sk
        self.Rgrid = Rgrid
        self.spin_dm = self.dm

        self.cols = None
        self.sort_cols = sort_cols

        self.Rgrid = Rgrid
        self._prepare_Rlist()
        self.nR = self.Rlist.shape[0]
        self.Amn = np.zeros((self.nkpt, self.nband, self.nwann), dtype=complex)

        self.wannk = np.zeros((self.nkpt, self.nbasis, self.nwann), dtype=complex)
        self.Hwann_k = np.zeros(
            (self.nspin, self.nkpt, self.nwann, self.nwann), dtype=complex
        )

        self.HwannR = np.zeros(
            (self.nspin, self.nR, self.nwann, self.nwann), dtype=complex
        )
        self.wannR = np.zeros((self.nR, self.nbasis, self.nwann), dtype=complex)
        self.atoms = atoms

    @classmethod
    def from_wannier(
        cls,
        path,
        prefix_up,
        prefix_down,
        kmesh=[3, 3, 3],
        weight_func_type="Fermi",
        mu=None,
        sigma=None,
        nwann=None,
    ):
        from lawaf.wrapper.myTB import MyTB

        k = kmesh[0]
        kpts = monkhorst_pack(kmesh)

        model_up = MyTB.read_from_wannier_dir(path, prefix_up)
        model_down = MyTB.read_from_wannier_dir(path, prefix_down)

        # evals_up, evecs_up = model_up.solve_all(kpts)
        # evals_down, evecs_down = model_down.solve_all(kpts)

        hams_up, _, evals_up, evecs_up = model_up.HS_and_eigen(kpts)
        hams_down, _, evals_down, evecs_down = model_down.HS_and_eigen(kpts)
        evals = np.array([evals_up, evals_down])
        try:
            positions = model_up.positions
        except Exception:
            positions = None

        weight_func = occupation_func(ftype=weight_func_type, mu=mu, sigma=sigma)

        nk = len(kpts)
        rho = np.zeros(shape=(nk, model_up.nbasis, model_down.nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            rho[ik] = (
                evecs_up[ik]
                @ np.diag(weight_func(evals_up[ik]))
                @ evecs_up[ik].T.conj()
                - evecs_down[ik]
                @ np.diag(weight_func(evals_down[ik]))
                @ evecs_down[ik].T.conj()
            )

        if nwann is None:
            nwann = model_up.nbasis
        return SpinDensityMatricesDownfolder(
            evals=evals,
            density_matrices=rho,
            positions=positions,
            kmesh=kmesh,
            nwann=nwann,
            Hks=np.array([hams_up, hams_down]),
            Sk=None,
            has_phase=False,
            Rgrid=None,
            sort_cols=True,
        )

    def auto_set_anchors(self, kpt=(0.0, 0.0, 0.0)):
        """
        Automatically find the columns using an anchor kpoint.
        kpt: the kpoint used to set as anchor points. default is Gamma (0,0,0)
        """
        ik = self.find_k(kpt)
        density = np.real(np.diag(np.sum(self.spin_dm, axis=0))) / self.nkpt
        self.cols = scdm(self.spin_dm[ik], self.nwann)
        print(f"anchor_kpt={kpt}. Selected columns: {self.cols}.")
        if self.sort_cols:
            self.cols = np.sort(self.cols)
        # print(f"The eigenvalues at anchor k: {self.get_eval_k(ik)}")
        print(f"anchor_kpt={kpt}. Selected columns: {self.cols}.")
        return self.cols

    def get_wannk_and_Hk(self):
        """
        calculate Wannier function and H in k-space.
        """
        spin_dm = np.average(self.spin_dm[:, :], axis=0)

        spin_dmc = spin_dm[:, self.cols]
        for ik in range(self.nkpt):
            self.wannk[ik] = self.spin_dm[ik][:, self.cols]
            ##self.wannk[ik] = spin_dm
            # self.wannk[ik] = np.eye(self.nbasis)

            U, E, VT = np.linalg.svd(spin_dmc, full_matrices=False)
            # self.wannk[ik] = U[:,:self.nwann] @  VT[:self.nwann, :self.nwann]

            U, E, VT = np.linalg.svd(self.wannk[ik], full_matrices=False)
            self.wannk[ik] = U @ VT
            # self.wannk[ik], _ = qr(self.wannk[ik], mode='economic')
            # h = self.Amn[ik, :, :].T.conj() @ np.diag(
            #    self.get_eval_k(ik)) @ self.Amn[ik, :, :]
            # self.wannk[ik] = np.eye(self.nbasis)
            # self.wannk[ik] = self.wannk[ik] / np.linalg.norm(self.wannk[ik],
            #                                                 axis=0)[None, :]
            # self.wannk[ik]=np.linalg.eig(self.wannk[ik])[1]

            # print(np.real(self.wannk[ik].T.conj() @ self.wannk[ik]))
            # print(np.imag(self.wannk[ik].T.conj() @ self.wannk[ik]))
            h_up = self.wannk[ik].T.conj() @ self.Hks[0, ik] @ self.wannk[ik]
            h_dn = self.wannk[ik].T.conj() @ self.Hks[1, ik] @ self.wannk[ik]
            self.Hwann_k[0, ik] = h_up
            self.Hwann_k[1, ik] = h_dn
        return self.wannk, self.Hwann_k

    def k_to_R(self):
        """
        Calculate Wannier function and Hamiltonian from K space to R space.
        """
        for iR, R in enumerate(self.Rlist):
            for ik, k in enumerate(self.kpts):
                phase = np.exp(-2j * np.pi * np.dot(R, k))
                self.wannR[iR] += self.wannk[ik, :, :] * phase * self.kweight[ik]
                for ispin in [0, 1]:
                    self.HwannR[ispin, iR] += (
                        self.Hwann_k[ispin, ik, :, :] * phase * self.kweight[ik]
                    )
            if np.linalg.norm(R) < 0.01:
                print(self.wannR[iR])
        self.get_wannier_centers()
        self.lwf_up = LWF(
            self.wannR,
            self.HwannR[0],
            self.Rlist,
            cell=np.eye(3),
            wann_centers=self.wann_centers,
        )
        self.lwf_down = LWF(
            self.wannR,
            self.HwannR[1],
            self.Rlist,
            cell=np.eye(3),
            wann_centers=self.wann_centers,
        )
        return self.lwf_up, self.lwf_down

    def save_to_nc(self, output_path="./", prefix="Downfolded"):
        txt_fname = os.path.join(output_path, f"{prefix}_up.txt")
        nc_fname = os.path.join(output_path, f"{prefix}_up.nc")
        self.lwf_up.save_txt(txt_fname)
        self.lwf_up.write_nc(nc_fname, atoms=self.atoms)

        txt_fname = os.path.join(output_path, f"{prefix}_down.txt")
        nc_fname = os.path.join(output_path, f"{prefix}_down.nc")
        self.lwf_down.save_txt(txt_fname)
        self.lwf_down.write_nc(nc_fname, atoms=self.atoms)

        from lawaf.plot import plot_band

        ax = plot_band(self.lwf_up)

        ax = plot_band(self.lwf_down, color="red")
        # plt.show()

    def get_wannier():
        pass
