import numpy as np
from scipy.linalg import qr, svd
from scipy.special import erfc
from .wannierizer import Wannierizer


class ProjectedWannierizer(Wannierizer):
    """
    Projected Wannier functions.
    We define a set of projectors, which is a nwann*nbasis matrix.
    Each projector is vector of size nbasis.
    """
    def set_params(self,params):
        if params.selected_basis:
            self.set_projectors_with_basis(params.selected_basis)
        elif params.anchors:
            self.set_projectors_with_anchors(params.anchors)



    def set_projectors_with_anchors(self, anchors):
        """
        Use one eigen vector (defined as anchor point) as projector.
        anchors: a dictionary: {kpt1: (band1, iband2...), kpt2: ...}
        """
        self.projectors = []
        for k, ibands in anchors.items():
            if self.wfn_anchor is None:
                ik = self.find_k(k)
                for iband in ibands:
                    self.projectors.append(self.get_psi_k(ik)[:, iband])
            else:
                for iband in ibands:
                    # print("adding anchor")
                    self.projectors.append(self.wfn_anchor[tuple(k)][:, iband])
        assert (
            len(self.projectors) == self.nwann
        ), "The number of projectors != number of wannier functions"

    def set_projectors_with_basis(self, ibasis):
        self.projectors = []
        for i in ibasis:
            b = np.zeros(self.nbasis, dtype=complex)
            b[i] = 1.0
            self.projectors.append(b)
        assert (
            len(self.projectors) == self.nwann
        ), "The number of projectors != number of wannier functions"

    def set_projectors(self, projectors):
        """
        set the initial guess for Wannier functions.
        projectors: a list of wavefunctions. shape: [nwann, nbasis]
        """
        assert (
            len(projectors) == self.nwann
        ), "The number of projectors != number of wannier functions"
        self.projectors = projectors

    def get_Amn_one_k(self, ik):
        """
        Amnk_0=<gi|psi_n k>
        Amn_0 is then orthogonalized using svd.
        """
        A = np.zeros((self.nband, self.nwann), dtype=complex)
        for iband in range(self.nband):
            for iproj, psi_a in enumerate(self.projectors):
                A[iband, iproj] = (
                    np.vdot(self.get_psi_k(ik)[:, iband], psi_a) * self.occ[ik, iband]
                )
        # using einsum
        U, _S, VT = svd(A, full_matrices=False)
        return U @ VT

    def get_Amn_psi(self, psi):
        """
        Amnk_0=<gi|psi_n k>
        Amn_0 is then orthogonalized using svd.
        """
        A = np.zeros((self.nband, self.nwann), dtype=complex)
        for iband in range(self.nband):
            for iproj, psi_a in enumerate(self.projectors):
                A[iband, iproj] = np.vdot(psi[:, iband], psi_a) * self.occ[iband]
        U, _S, VT = svd(A, full_matrices=False)
        return U @ VT


class MaxProjectedWannierizer(ProjectedWannierizer):
    def get_Amn_one_k(self, ik):
        """
        Amnk_0=  <psi_m k|g_n>
        Amnk_0 is then orthogonalized using svd.
        m is the band index and n is the Wannier index.
        """
        kpt= self.kpts[ik]
        print(f"MaxProjectedWannierizer: ik={ik}, kpt={kpt}.")
        A = np.zeros((self.nband, self.nwann), dtype=complex)
        for iproj, psi_a in enumerate(self.projectors):
            for iband in range(self.nband):
                A[iband, iproj] = (
                    np.vdot(self.get_psi_k(ik)[:, iband], psi_a) * self.occ[ik, iband]
                )
            # select the maximum value of A[:, iproj] and set to 1. Others are set to 0.  
            imax=np.argmax(np.abs(np.abs(A[:, iproj])))
            imax = iproj
            print(f"MaxProjectedWannierizer: iproj={iproj}, imax={imax}.")
            tmp=A[imax, iproj]
            A[:, iproj] =0
            A[imax, iproj] = 1
        print(f"MaxProjectedWannierizer: A={A}.")
        U, _S, VT = svd(A, full_matrices=False)
        A=U @ VT
        return A
