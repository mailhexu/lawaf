import numpy as np
from scipy.linalg import svd

from .wannierizer import Wannierizer


class ProjectedWannierizer(Wannierizer):
    """
    Projected Wannier functions.
    We define a set of projectors, which is a nwann*nbasis matrix.
    Each projector is vector of size nbasis.
    """

    def set_params(self, params):
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
        self.projectors = np.array(self.projectors)

    def set_projectors_with_basis(self, ibasis):
        self.projectors = []
        for i in ibasis:
            b = np.zeros(self.nbasis, dtype=complex)
            b[i] = 1.0
            self.projectors.append(b)
        assert (
            len(self.projectors) == self.nwann
        ), "The number of projectors != number of wannier functions"
        self.projectors = np.array(self.projectors)

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
        if self.is_orthogonal:
            # A=np.einsum('ob, wo, b ->bw', self.get_psi_k(ik).conj(), self.projectors, self.occ[ik])
            A = (
                self.get_psi_k(ik).conj().T
                @ self.projectors.T
                * self.occ[ik][:, np.newaxis]
            )
        else:
            # A = self.get_psi_k(ik).conj().T @ self.projectors * self.occ[ik][:, np.newaxis]
            # A=np.einsum('ob, wo, b ->bw', self.get_psi_k(ik).conj(), self.projectors, self.occ[ik])
            # A=np.einsum('ob, ow, b ->bw', self.get_psi_k(ik).conj(), self.projectors, self.occ[ik])
            A = (
                self.get_psi_k(ik).conj().T
                # @ self.S[ik]
                @ self.projectors.T
                * self.occ[ik][:, np.newaxis]
            )
            # A = self.get_psi_k(ik).conj().T @ self.projectors.T * self.occ[ik][:, np.newaxis]
        # using einsum
        window_bands = getattr(
            self.params,
            "_window_bands_resolved",
            getattr(self.params, "window_bands", None),
        )
        if window_bands is None:
            # Preserve the legacy path bit-for-bit when pinning is disabled.
            A = (A.conj() * A) ** -0.1 * A
        else:
            magnitude = np.abs(A)
            scale = np.zeros_like(magnitude)
            np.power(magnitude, -0.2, out=scale, where=magnitude > 0.0)
            A *= scale
        return self._orthonormalize_amn(
            A, self._window_rows_by_ik.get(ik)
        )
        # return A

    def get_Amn_psi(self, psi):
        """
        Amnk_0=<gi|psi_n k>
        Amn_0 is then orthogonalized using svd.
        """
        A = np.zeros((self.nband, self.nwann), dtype=complex)
        for iband in range(self.nband):
            for iproj, psi_a in enumerate(self.projectors):
                A[iband, iproj] = (
                    np.vdot(
                        psi[:, iband],
                        psi_a,
                    )
                    * self.occ[iband]
                )
        U, _S, VT = svd(A, full_matrices=False)
        return U @ VT

    # -- story-018 parameter hook (FR-004/015, ADR-003) -------------------
    # With params.symmetry_adapted_gauge set, the standard-path gauge is
    # routed through lawaf.anharmonic.gauge: get_wannk_and_Hk reimposes the
    # little-group constraint on self.Amn in place (covering both
    # Lawaf.downfold and PhonopyDownfolder.downfold, which call this after
    # get_Amn), and get_wannier returns the fully rebuilt constrained LWF.
    # Default (flag False) behavior is untouched.
    def get_wannk_and_Hk(self, shift=0.0):
        if getattr(self.params, "symmetry_adapted_gauge", False) and not getattr(
            self, "_gauge_applied", False
        ):
            from lawaf.anharmonic.gauge import constrain_builder_amn
            constrain_builder_amn(
                self,
                getattr(self.params, "representation_declaration", None),
                params=self.params,
                sga=getattr(self, "gauge_sga", None),
            )
        return super().get_wannk_and_Hk(shift=shift)

    def get_wannier(self, Rlist=None, Rdeg=None):
        if getattr(self.params, "symmetry_adapted_gauge", False):
            from lawaf.anharmonic.gauge import constrained_localize

            return constrained_localize(
                self,
                getattr(self.params, "representation_declaration", None),
                params=self.params,
                Rlist=Rlist,
                Rdeg=Rdeg,
            )
        return super().get_wannier(Rlist=Rlist, Rdeg=Rdeg)


class MaxProjectedWannierizer(ProjectedWannierizer):
    def get_Amn_one_k(self, ik):
        """
        Amnk_0=  <psi_m k|g_n>
        Amnk_0 is then orthogonalized using svd.
        m is the band index and n is the Wannier index.
        """
        kpt = self.kpts[ik]
        print(f"MaxProjectedWannierizer: ik={ik}, kpt={kpt}.")
        A = np.zeros((self.nband, self.nwann), dtype=complex)
        for iproj, psi_a in enumerate(self.projectors):
            for iband in range(self.nband):
                A[iband, iproj] = (
                    np.vdot(self.get_psi_k(ik)[:, iband], psi_a) * self.occ[ik, iband]
                )
            # select the maximum value of A[:, iproj] and set to 1. Others are set to 0.
            imax = np.argmax(np.abs(np.abs(A[:, iproj])))
            print(f"MaxProjectedWannierizer: iproj={iproj}, imax={imax}.")
            # tmp = A[imax, iproj]
            A[:, iproj] = 0
            A[imax, iproj] = 1
        print(f"MaxProjectedWannierizer: A={A}.")
        U, _S, VT = svd(A, full_matrices=False)
        A = U @ VT
        return A
