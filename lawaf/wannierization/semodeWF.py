"""
Wannier Module: For building Wannier functions and Hamiltonians.
"""

import numpy as np
from scipy.linalg import qr, svd
from scipy.special import erfc
from .wannierizer import Wannierizer


class SelModeWannierizer(Wannierizer):
    """
    Wannier function from hand-selected modes.
    """
    def __init__(self, params):
        pass

    def get_Amn_one_k(self, ik):
        pass

    def get_Amn(self):
        return super().get_Amn()


