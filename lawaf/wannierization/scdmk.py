"""
Wannier Module: For building Wannier functions and Hamiltonians.
"""

import numpy as np
from scipy.linalg import qr, svd
from scipy.special import erfc

from .wannierizer import Wannierizer


def scdm(psiT, ncol):
    """
    select columns for a psiT.
    params:
        psiT: a matrix of shape [nbasis, nband]
        ncol: number of columns to be selected.
    return:
        cols: the indices of selected columns.
    """
    _Q, _R, piv = qr(psiT, mode="full", pivoting=True)
    cols = piv[:ncol]
    return cols


class ScdmkWannierizer(Wannierizer):
    """
    Build Wannier functions using the SCDMk method.
    """

    def set_params(self, params):
        self.psi_anchors = []
        self.cols = []
        self.use_proj = params.use_proj
        self.proj_order = params.proj_order
        self.projs = np.zeros((self.nkpt, self.nband), dtype=float)
        self.sort_cols = params.sort_cols
        self.orthogonal = params.orthogonal

        if params.selected_basis:
            self.set_selected_cols(params.selected_basis)
        elif params.anchors:
            self.set_anchors(params.anchors)
        else:
            self.auto_set_anchors(params.anchor_kpt)

        print(f"Number of selected columns: {len(self.cols)}")
        print(f"Selected columns: {self.cols}")

    def set_selected_cols(self, cols):
        """
        Munually set selected Columns.
        """
        if cols is not None:
            assert (
                len(cols) == self.nwann
            ), "Number of columns should be equal to number of Wannier functions"
            self.cols = cols
            if self.sort_cols:
                self.cols = np.sort(self.cols)
            for col in cols:
                projector = np.zeros(self.nbasis, dtype=complex)
                projector[col] = 1.0
                self.psi_anchors.append(projector)

    def add_anchors(self, psi, ianchors):
        """
        psi, a wavefunction for a k point [ibasis, iband]
        ianchor: the indices of band of the anchor points.
        """
        for ia in ianchors:
            self.psi_anchors.append(psi[:, ia])

        ianchors = np.array(ianchors, dtype=int)

        proj = np.zeros((self.nband), dtype=float)
        for iband in range(self.nband):
            for ia in ianchors:
                proj[iband] += np.abs(np.vdot(psi[:, iband], psi[:, ia]))
        psi_D = psi @ np.diag(proj)

        psi_Dagger = psi_D.T.conj()  # psi.T.conj()

        cols = scdm(psi_Dagger, len(ianchors))
        self.cols = np.array(tuple(set(self.cols).union(cols)))
        if self.sort_cols:
            self.cols = np.sort(self.cols)

    def set_anchors(self, anchors):
        """
        anchor_points: a dictionary. The keys are the kpoints and the values are tuples of band indices at that kpoint.
        """
        if anchors is None:
            return
        self.psi_anchors = []
        for k, ibands in anchors.items():
            if self.wfn_anchor is None:
                ik = self.find_k(k)
                self.add_anchors(self.get_psi_k(ik), ibands)
            else:
                self.add_anchors(self.wfn_anchor[k], ibands)
        self.psi_anchors = np.array(self.psi_anchors)
        self.cols = self.cols[: self.nwann]
        print(f"Using the anchor points, these cols are selected: {self.cols}")
        assert (
            len(self.cols) == self.nwann
        ), "After adding all anchors, the number of selected columns != nwann"

    def auto_set_anchors(self, kpt=(0.0, 0.0, 0.0)):
        """
        Automatically find the columns using an anchor kpoint.
        kpt: the kpoint used to set as anchor points. default is Gamma (0,0,0)
        """
        ik = self.find_k(kpt)
        psi = self.get_psi_k(ik)[:, :] * self.occ[ik][None, :]
        psi_Dagger = psi.T.conj()
        # if not self.is_orthogonal:
        #    psi_Dagger = psi_Dagger @ self.S[ik]
        self.cols = scdm(psi_Dagger, self.nwann)
        if self.sort_cols:
            self.cols = np.sort(self.cols)
        print(f"The eigenvalues at anchor k: {self.get_eval_k(ik)}")
        print(f"anchor_kpt={kpt}. Selected columns: {self.cols}.")

    def _get_projection_to_anchors2(self):
        # anchor point wavefunctions with phase removed
        if self.use_proj and len(self.psi_anchors) > 0:
            self.projs[:, :] = 0.0
            for ikpt in range(self.nkpt):
                psik = self.get_psi_k(ikpt)
                # self.projs[ikpt] = psik.T.conj() @ (self.psi_anchors.T)
                for iband in range(self.nband):
                    psi_kb = psik[:, iband]
                    for psi_a in self.psi_anchors:
                        p = np.vdot(psi_kb, psi_a)
                        self.projs[ikpt, iband] += np.real(np.conj(p) * p) ** (
                            self.proj_order
                        )
        else:
            self.projs[:, :] = 1.0
        return self.projs

    def _get_projection_to_anchors(self):
        # anchor point wavefunctions with phase removed
        if self.use_proj and len(self.psi_anchors) > 0:
            self.projs[:, :] = 0.0
            # k: kpt, o: orbital, b: band
            self.projs_per_anchor = np.einsum(
                "kob,no->knb", self.evecs.conj(), self.psi_anchors
            )
            self.projs = np.einsum(
                "knb->kb",
                (self.projs_per_anchor * self.projs_per_anchor.conj())
                ** self.proj_order,
            )
            print(f"Projs: {self.projs}")
        else:
            self.projs[:, :] = 1.0
        return self.projs

    def get_Amn_one_k(self, ik):
        """
        calculate Amn for one k point using scdmk method.
        """
        psik = self.get_psi_k(ik)
        occ = self.occ[ik]
        projs = self.projs[ik]
        if self.is_orthogonal:
            Sk = None
        else:
            Sk = self.S[ik]
        return self.get_Amn_psi(psik, occ=occ, projs=projs, Sk=Sk)

    def get_Amn_psi(self, psik, occ=None, projs=None, Sk=None):
        psiT = psik.T.conj()
        if Sk is not None:
            psiT = psiT @ Sk
        if self.use_proj:
            # projs = np.einsum("iw,wb->b", self.psi_anchors, psik.conj())
            # projs = np.sqrt(np.abs(np.abs(projs)))
            projs = np.einsum("iw,wb->ib", self.psi_anchors, psik.conj())
            projs = np.einsum("ib-> b", np.abs(np.abs(projs)) ** self.proj_order)
            if occ is None:
                # psi = psik[self.cols, :] * (projs)[None, :]
                psiT = psiT[:, self.cols] * (projs)[:, None]
            else:
                # psi = psik[self.cols, :] * (occ * projs)[None, :]
                psiT = psiT[:, self.cols] * (occ * projs)[:, None]
        else:
            if occ is not None:
                # psi = psik[self.cols, :] * occ[None, :]
                psiT = psiT[:, self.cols] * (occ)[:, None]
            else:
                # psi = psik[self.cols, :]
                psiT = psiT[:, self.cols]
        if self.orthogonal or True:
            U, _S, VT = svd(psiT, full_matrices=False)
            Amn_k = U @ VT
        else:
            Amn_k = psiT
        return Amn_k

    def prepare(self):
        """
        Calculate projection to anchors.
        """
        self._get_projection_to_anchors()


def test():
    a = np.arange(-5, 5, 0.01)
    import matplotlib.pyplot as plt

    plt.plot(a, 0.5 * erfc((a - 0.0) / 0.5))
    plt.show()
