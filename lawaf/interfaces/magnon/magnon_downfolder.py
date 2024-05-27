import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from dataclasses import dataclass

# from lawaf.hamitonian import
from lawaf.utils.kpoints import monkhorst_pack, build_Rgrid
from lawaf.mathutils.kR_convert import k_to_R, R_to_k


@dataclass
class MagnonWrapper:
    HR: np.ndarray = None
    Rlist: np.ndarray = None
    nR: int = 0
    nbasis: int = 0

    def __init__(self, HR=None, Rlist=None):
        """ """
        self.HR = np.array(HR)
        self.Rlist = Rlist
        if HR is not None:
            self.nR, self.nbasis, _ = HR.shape

    @classmethod
    def read_from_path_k(cls, path, kmesh):
        """
        Read from a Hk file and convert to HR
        """
        p = Path(path)
        Hk = np.load(p / "H_matrix.npy")
        kpts = monkhorst_pack(kmesh)
        Rlist = build_Rgrid(kmesh)
        HR = k_to_R(kpts=kpts, Rlist=Rlist, Mk=Hk, kweights=None)
        return cls(HR=HR, Rlist=Rlist)

    def save_pickle(self, path):
        """
        save model to pickle
        """
        with open(path, "wb") as f:
            pickle.dump([np.asarray(self.HR), self.Rlist], f)

    @classmethod
    def load_from_pickle(cls, path):
        """
        load model from pickle
        """
        with open(path, "rb") as f:
            HR, Rlist = pickle.load(f)
        return cls(HR=HR, Rlist=Rlist)


class MagnonDownfolder:
    def __init__(self, path, kmesh):
        self.path = Path(path)
        pass


def test_MagnonDownfolder():
    path = "/Users/hexu/projects/magnon_downfold/examples/CoF2/ligands"
    kmesh = [9, 9, 9]
    model = MagnonWrapper.read_from_path_k(path, kmesh)


if __name__ == "__main__":
    test_MagnonDownfolder()
