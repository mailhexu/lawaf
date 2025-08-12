import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lawaf.interfaces.downfolder import Lawaf
from lawaf.mathutils.align_evecs import align_all_degenerate_eigenvectors
from lawaf.mathutils.kR_convert import R_to_k, k_to_R

# from lawaf.hamitonian import
from lawaf.utils.kpoints import build_Rgrid, monkhorst_pack


@dataclass
class MagnonWrapper:
    HR: np.ndarray = None
    Rlist: np.ndarray = None
    nR: int = 0
    nbasis: int = 0
    atoms = None
    is_orthogonal = True

    def __init__(self, HR=None, Rlist=None, atoms=None, align_evecs=True):
        """ """
        self.HR = np.array(HR)
        self.Rlist = Rlist
        if HR is not None:
            self.nR, self.nbasis, _ = HR.shape
        self.atoms = atoms
        self.align_evecs = align_evecs

    @classmethod
    def load_from_TB2J_pickle(cls, path, fname):
        from TB2J.io_exchange import SpinIO

        exc = SpinIO.load_pickle(path=path, fname=fname)
        atoms = exc.atoms
        HR = exc.get_full_Jtensor_for_Rlist(asr=True)
        Rlist = exc.Rlist
        Rlist = np.array(Rlist)
        return cls(HR=HR, Rlist=Rlist, atoms=atoms)

    @classmethod
    def read_from_path_k(cls, path, kmesh):
        """
        Read from a Hk file and convert to HR
        """
        p = Path(path)
        Hk = np.load(p / "H_matrix.npy")
        kpts = monkhorst_pack(kmesh)
        Rlist, Rweights = build_Rgrid(kmesh)
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
        self.Rlist = np.array(self.Rlist)
        # check if there is NaN in HR
        if np.isnan(self.HR).any():
            raise ValueError("HR contains NaN values. Check the input data.")
        Hks = R_to_k(kpts, self.Rlist, self.HR)
        # check if there is NaN in Hks
        if np.isnan(Hks).any():
            raise ValueError("Hks contains NaN values. Check the input data.")

        evals = np.zeros((len(kpts), self.nbasis), dtype=float)
        evecs = np.zeros((len(kpts), self.nbasis, self.nbasis), dtype=complex)
        for ik, Hk in enumerate(Hks):
            evals[ik], evecs[ik] = np.linalg.eigh(Hk)
            if self.align_evecs:
                try:
                    evals[ik], evecs[ik] = align_all_degenerate_eigenvectors(
                        evals[ik], evecs[ik], tol=1e-8
                    )
                except Exception as e:
                    print(f"Error aligning eigenvectors at k-point {ik}: {e}")

        return evals, evecs


class MagnonDownfolder(Lawaf):
    def __init__(self, model):
        """
        folder   # The folder containing the Wannier function files
        prefix,   # The prefix of Wannier90 outputs. e.g. wannier90_up
        """
        self.model = model


def easy_downfold_magnon(
    path,
    index_basis,
    kmesh=[9, 9, 9],
    weight_func="unity",
    weight_func_params=(0.00, 0.001),
    TB2J_pickle_fname="TB2J.pickle",
    downfolded_pickle_fname="downfolded_HR.pickle",
    savefig="Downfolded_band.png",
    Jq=False,
    **kwargs,
):
    path = Path(path).expanduser()

    if Jq:
        model = MagnonWrapper.load_from_TB2J_pickle(path, TB2J_pickle_fname)
    else:
        model = MagnonWrapper.read_from_path_k(path, kmesh)
    wann = MagnonDownfolder(model)
    # Downfold the band structure.
    params = dict(
        method="projected",
        # method="maxprojected",
        kmesh=kmesh,
        nwann=len(index_basis),
        weight_func=weight_func,
        weight_func_params=weight_func_params,
        selected_basis=index_basis,
        # anchors={(0, 0, 0): (-1, -2, -3, -4)},
        # anchors={(0, 0, 0): ()},
        use_proj=True,
    )
    params.update(kwargs)

    wann.set_parameters(**params)
    print("begin downfold")
    _ewf = wann.downfold()
    # ewf.save_hr_pickle(downfolded_pickle_fname)

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
