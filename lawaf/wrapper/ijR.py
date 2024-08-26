import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from netCDF4 import Dataset
from collections import defaultdict
from lawaf.utils.supercell import SupercellMaker
from ase.dft.kpoints import bandpath


class ijR(object):
    def __init__(
        self,
        nbasis,
        cell=np.eye(3, dtype=float),
        data=None,
        positions=None,
        sparse=False,
        double_site_energy=2.0,
    ):
        self.cell = cell
        self.nbasis = nbasis
        if data is not None:
            self.data = data
        else:
            self.data = defaultdict(lambda: np.zeros((nbasis, nbasis), dtype=float))
        if positions is None:
            self.positions = np.zeros((nbasis, 3))
        else:
            self.positions = positions
        self.sparse = sparse
        self.double_site_energy = double_site_energy
        if sparse:
            self._matrix = csr_matrix

    def to_sparse(self):
        for key, val in self.data:
            self.data[key] = self._matrix(val)

    def make_supercell(self, supercell_matrix=None, scmaker=None):
        if scmaker is None:
            scmaker = SupercellMaker(sc_matrix=supercell_matrix)
        ret = ijR(nbasis=self.nbasis * scmaker.ncell)
        if self.positions is None:
            ret.positions = None
        else:
            ret.positions = scmaker.sc_pos(self.positions)
        ret.cell = scmaker.sc_cell(self.cell)
        ret.sparse = self.sparse
        ret.double_site_energy = self.double_site_energy
        for R, mat in self.data.items():
            for i in range(self.nbasis):
                for j in range(self.nbasis):
                    for sc_i, sc_j, sc_R in scmaker.sc_ijR_only(i, j, R, self.nbasis):
                        ret.data[sc_R][sc_i, sc_j] = mat[i, j]
        return ret

    @property
    def Rlist(self):
        return list(self.data.keys())

    @property
    def nR(self):
        return len(self.Rlist)

    @property
    def site_energies(self):
        return self.data[(0, 0, 0)].diagonal() * self.double_site_energy

    @property
    def hoppings(self):
        data = copy.deepcopy(self.data)
        np.fill_diagonal(data[(0, 0, 0)], 0.0)
        return data

    @staticmethod
    def _positive_R_mat(R, mat):
        """
        if R<0: M(-R) = Mdagger
        if R=0: (M+Mdagger)/2
        if R>0 : M(R) = Mdagger
        """
        nzR = np.nonzero(R)[0]
        if len(nzR) != 0 and R[nzR[0]] < 0:
            newR = tuple(-np.array(R))
            newmat = mat.T.conj()
        elif len(nzR) == 0:
            newR = R
            newmat = (mat + mat.T.conj()) / 2.0
        else:
            newR = R
            newmat = mat
        return newR, newmat

    def _to_positive_R(self):
        new_ijR = ijR(
            self.nbasis, cell=self.cell, positions=self.positions, sparse=self.sparse
        )
        for R, mat in self.data:
            newR, newmat = self._positive_R_mat(R, mat)
            new_ijR.data[newR] = newmat
        return new_ijR

    def shift_position(self, rpos):
        pos = self.positions
        shift = np.zeros((self.nbasis, 3), dtype="int")
        shift[:, :] = np.round(pos - rpos)
        newpos = copy.deepcopy(pos)
        for i in range(self.nbasis):
            newpos[i] = pos[i] - shift[i]
        d = ijR(self.nbasis)
        d.positions = newpos
        for R, v in self.data.items():
            for i in range(self.nbasis):
                for j in range(self.nbasis):
                    sR = tuple(np.array(R) - shift[i] + shift[j])
                    nzR = np.nonzero(sR)[0]
                    if len(nzR) != 0 and sR[nzR[0]] < 0:
                        newR = tuple(-np.array(sR))
                        d.data[newR][j, i] += v[i, j]
                    elif len(nzR) == 0:
                        newR = sR
                        d.data[newR][i, j] += v[i, j] * 0.5
                        d.data[newR][j, i] += np.conj(v[i, j]) * 0.5
                    else:
                        d.data[sR][i, j] += v[i, j]
        return d

    def save(self, fname):
        try:
            from netCDF4 import Dataset
        except ImportError():
            print("Warning: ")
        root = Dataset(fname, "w", format="NETCDF4")
        root.createDimension("nR", self.nR)
        root.createDimension("three", 3)
        root.createDimension("nbasis", self.nbasis)
        R = root.createVariable("R", "i4", ("nR", "three"))
        data = root.createVariable("data", "f8", ("nR", "nbasis", "nbasis"))
        positions = root.createVariable("positions", "f8", ("nbasis", "three"))
        cell = root.createVariable("cell", "f8", ("three", "three"))
        R[:] = np.array(self.Rlist)
        data[:] = np.array(tuple(self.data.values()))
        positions[:] = np.array(self.positions)
        cell[:] = np.array(self.cell)
        root.close()

    def write_txt(self, fname):
        with open(fname, "w") as myfile:
            myfile.write(f"Number_of_R: {self.nR}\n")
            myfile.write(f"Number_of_basis_functions: {self.nbasis}\n")
            myfile.write(f"Cell parameter: {self.cell}\n")
            myfile.write("Hamiltonian:  \n" + "=" * 60 + "\n")
            for iR, R in enumerate(self.Rlist):
                myfile.write(f"index of R: {iR}.  R = {R}\n")
                d = self.data[R]
                for i in rang(nbais):
                    for j in range(nbasis):
                        myfile.write(f"R = {R}\t. i = {i}, j={j}, H(i,j,R)= {d[i,j]}")
                myfile.write("-" * 60 + "\n")

    @staticmethod
    def load_ijR(fname):
        root = Dataset(fname, "r")
        nbasis = root.dimensions["nbasis"].size
        Rlist = root.variables["R"][:]
        m = ijR(nbasis)
        mdata = root.variables["data"][:]
        for iR, R in enumerate(Rlist):
            m.data[tuple(R)] = mdata[iR]
        m.positions = root.variables["positions"][:]
        m.cell = root.variables["cell"][:]
        return m

    @staticmethod
    def from_tbmodel(model):
        ret = ijR(nbasis=model.size)
        for R, v in model.hop.items():
            ret.data[R] = v
        ret.positions = np.reshape(model.pos, (model.size, 3))
        ret.cell = model.uc
        return ret

    @staticmethod
    def from_tbmodel_hdf5(fname):
        from tbmodels import Model

        m = Model.from_hdf5_file(fname)
        ret = ijR(nbasis=m.size)
        for R, v in m.hop.items():
            ret.data[R] = v
        ret.positions = np.reshape(m.pos, (m.size, 3))
        ret.cell = m.uc
        return ret

    def to_spin_polarized(self, order=1):
        """
        repeat
        order =1 : orb1_up, orb1_dn, orb2_up, orb2_dn...
        order =2 : orb1_up, orb2_up, ... orb1_dn, orb2_dn...
        """
        ret = ijR(self.nbasis * 2)
        if self.positions is None:
            ret.positions = None
        else:
            ret.positions = np.repeat(self.positions, 2, axis=0)
        for R, mat in self.data.items():
            ret.data[R][::2, ::2] = mat
            ret.data[R][1::2, 1::2] = mat
        return ret

    def gen_ham(self, k):
        Hk = np.zeros((self.nbasis, self.nbasis), dtype="complex")
        np.fill_diagonal(Hk, self.site_energies)
        for R, mat in self.hoppings.items():
            phase = np.exp(2j * np.pi * np.dot(k, R))
            Hk += mat * phase + (mat * phase).T.conj()
        return Hk

    def solve_all(self, kpts):
        nk = len(kpts)
        evals = np.zeros((nk, self.nbasis))
        evecs = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            evals_k, evecs_k = eigh(self.gen_ham(k))
            evals[ik, :] = evals_k
            evecs[ik, :, :] = evecs_k
        return evals, evecs

    def plot_band(
        self,
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        ax=None,
    ):
        if ax is None:
            _fig, ax = plt.subplots()
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.cell, npoints)
        evalues, _evecs = self.solve_all(kpts=kpts)
        for i in range(evalues.shape[1]):
            ax.plot(x, evalues[:, i], color="blue", alpha=1)

        if efermi is not None:
            plt.axhline(self.get_fermi_level(), linestyle="--", color="gray")
        else:
            try:
                plt.axhline(self.get_fermi_level(), linestyle="--", color="gray")
            except:
                pass
        ax.set_xlabel("k-point")
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color="gray")
        return ax

    def validate(self):
        # make sure all R are 3d.
        for R in self.data.keys():
            if len(R) != 3:
                raise ValueError("R should be 3d")


def ijR_to_model(model, dijR):
    model._hoppings = dijR.hoppings
    model._site_energies = dijR.site_energies
    return model
