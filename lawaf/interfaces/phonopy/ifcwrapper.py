"""
simple IFC wrapper
"""

from scipy.linalg import eigh
import numpy as np
from lawaf.plot import plot_band
import matplotlib.pyplot as plt

# from lawaf.utils import read_ifc_from_netcdf, save_ifc_to_netcdf


class IFC:
    def __init__(self, atoms, Rlist, ifc):
        self.atoms = atoms
        self.Rlist = Rlist
        self.ifc = ifc
        self.natoms = len(atoms)
        self.natoms3 = 3 * self.natoms

    @staticmethod
    def load_from_netcdf(fname):
        """
        read netcdf
        """
        atoms, Rlist, ifc = read_ifc_from_netcdf(fname)
        return IFC(atoms, Rlist, ifc)

    def save_to_netcdf(self, fname):
        save_ifc_to_netcdf(
            fname=fname,
            ifc=self.ifc,
            Rlist=self.Rlist,
            atoms=self.atoms,
            ref_energy=0.0,
        )

    def eval_modifier(self, kmesh, func):
        pass

    def solve_all(self, kpts):
        """
        calculate the eigen values and vectors
        """
        nk = len(kpts)
        Hk = np.zeros((self.natoms3, self.natoms3), dtype=complex)
        evecs = np.zeros((nk, self.natoms3, self.natoms3), dtype=complex)
        evals = np.zeros((nk, self.natoms3), dtype=float)
        for ik, k in enumerate(kpts):
            Hk[:, :] = 0.0
            for iR, R in enumerate(self.Rlist):
                phase = np.exp(-2.0j * np.pi * (k @ R))
                Hk += self.ifc[iR] * phase
            evals[ik], evecs[ik] = eigh(Hk)
        return evals, evecs

    def plot_band(self, **kwargs):
        plot_band(self, **kwargs)


def test():
    # fname = './R_VO2_ifc_scaled.nc'
    fname = "/home/hexu/projects/VO2_new/perhaps_good_potential/latt_lwf/R_VO2_ifc_scaled.nc"
    ifc = IFC.load_from_netcdf(fname)
    ifc.plot_band()
    plt.show()


if __name__ == "__main__":
    test()
