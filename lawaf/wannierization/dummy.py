import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from .wannierizer import Wannierizer


class DummyWannierizer(Wannierizer):
    def get_Amn_one_k(self, ik):
        pass

    def get_wannk_and_Hk(self, shift=0.0):
        """
        calculate Wannier function and H in k-space.
        """
        for ik in range(self.nkpt):
            self.wannk[ik] = np.eye(self.nbasis)
            self.Hwann_k[ik] = self.Hk[ik]
            if not self.is_orthogonal:
                if self.params.orthogonal:
                    Shalf = fractional_matrix_power(self.S[ik], -0.5)
                    self.Hwann_k[ik] = Shalf @ self.Hk[ik] @ Shalf
                else:
                    self.Hwann_k[ik] = self.Hk[ik]
                    self.Swann_k[ik] = self.S[ik]
            else:
                self.Hwann_k[ik] = self.Hk[ik]
                self.Swann_k[ik] = None
            if self.params.orthogonal:
                evals, evecs = eigh(self.Hwann_k[ik])
            else:
                evals, evecs = eigh(self.Hwann_k[ik], self.Swann_k[ik])
            diff = evals - self.get_eval_k(ik)
            print(f"Eigenvalues difference: {diff}")
        return self.wannk, self.Hwann_k, self.Swann_k
