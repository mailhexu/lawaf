import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from dataclasses import dataclass

# from lawaf.hamitonian import
from lawaf.utils.kpoints import monkhorst_pack, build_Rgrid
from lawaf.mathutils.kR_convert import k_to_R, R_to_k
from lawaf.interfaces.downfolder import Lawaf


@dataclass
class MagnonWrapper:
    HR: np.ndarray = None
    Rlist: np.ndarray = None
    nR: int = 0
    nbasis: int = 0
    atoms = None

    def __init__(self, HR=None, Rlist=None, atoms=None):
        """ """
        self.HR = np.array(HR)
        self.Rlist = Rlist
        if HR is not None:
            self.nR, self.nbasis, _ = HR.shape
        self.atoms = atoms

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
        with open(p / "TB2J.pickle", "rb") as f:
            tb2j = pickle.load(f)
            atoms = tb2j["atoms"]
        return cls(HR=HR, Rlist=Rlist, atoms=atoms)

    def save_pickle(self, path):
        """
        save model to pickle
        """
        with open(path, "wb") as f:
            pickle.dump([np.asarray(self.HR), self.Rlist, self.atoms], f)

    @classmethod
    def load_from_pickle(cls, path):
        """
        load model from pickle
        """
        with open(path, "rb") as f:
            HR, Rlist, atoms = pickle.load(f)
        return cls(HR=HR, Rlist=Rlist, atoms=atoms)

    def solve_all(self, kpts):
        """
        solve the model at all kpts
        """
        Hks = R_to_k(kpts, self.Rlist, self.HR)
        evals = np.zeros((len(kpts), self.nbasis), dtype=float)
        evecs = np.zeros((len(kpts), self.nbasis, self.nbasis), dtype=complex)
        for ik, Hk in enumerate(Hks):
            evals[ik], evecs[ik] = np.linalg.eigh(Hk)
        return evals, evecs


class MagnonDownfolder(Lawaf):
    def __init__(self, model):
        """
        folder   # The folder containing the Wannier function files
        prefix,   # The prefix of Wannier90 outputs. e.g. wannier90_up
        """
        self.model = model


def test_MagnonWrapper():
    path = Path("~/projects/magnon_downfold/examples/CoF2/ligands").expanduser()
    kmesh = [9, 9, 9]
    model = MagnonWrapper.read_from_path_k(path, kmesh)


def easy_downfold_magnon(
    path,
    index_metal,
    kmesh=[9, 9, 9],
    weight_func="unity",
    weight_func_params=(0.00, 0.001),
    downfolded_pickle_fname="downfolded_HR.pickle",
    savefig="Downfolded_band.png",
    **kwargs,
):
    path = Path(path).expanduser()
    model = MagnonWrapper.read_from_path_k(path, kmesh)
    wann = MagnonDownfolder(model)
    # Downfold the band structure.
    params = dict(
        method="projected",
        # method="maxprojected",
        kmesh=kmesh,
        nwann=len(index_metal),
        weight_func=weight_func,
        weight_func_params=weight_func_params,
        selected_basis=index_metal,
        # anchors={(0, 0, 0): (-1, -2, -3, -4)},
        # anchors={(0, 0, 0): ()},
        use_proj=True,
    )
    params.update(kwargs)

    wann.set_parameters(**params)
    ewf = wann.downfold()
    ewf.save_hr_pickle(downfolded_pickle_fname)

    # Plot the band structure.
    wann.plot_band_fitting(
        # kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
        #                   [0.5, 0.5, 0], [0, 0, 0],
        #                   [.5, .5, .5]]),
        # knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
        cell=model.atoms.cell,
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=None,
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig=savefig,
        show=True,
    )


def test_MagnonDownfolder():
    path = Path("~/projects/magnon_downfold/examples/CoF2/ligands").expanduser()
    kmesh = [9, 9, 9]
    model = MagnonWrapper.read_from_path_k(path, kmesh)
    wann = MagnonDownfolder(model)
    # Downfold the band structure.
    params = dict(
        method="projected",
        # method="maxprojected",
        kmesh=(9, 9, 9),
        nwann=4,
        weight_func="unity",
        weight_func_params=(0.00, 0.001),
        selected_basis=[0, 1, 6, 7],
        # anchors={(0, 0, 0): (-1, -2, -3, -4)},
        # anchors={(0, 0, 0): ()},
        use_proj=True,
    )

    wann.set_parameters(**params)
    wann.downfold()

    # Plot the band structure.
    wann.plot_band_fitting(
        # kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
        #                   [0.5, 0.5, 0], [0, 0, 0],
        #                   [.5, .5, .5]]),
        # knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=None,
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig="Downfolded_band.png",
        show=True,
    )


if __name__ == "__main__":
    # test_MagnonDownfolder()
    test_MagnonDownfolder()
