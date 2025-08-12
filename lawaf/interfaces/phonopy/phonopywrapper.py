import atexit
import copy
import lzma
import os
import pickle
from typing import Type, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack
from phonopy import Phonopy, load
from phonopy.api_phonopy import symmetrize_borns_and_epsilon
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixGL,
    DynamicalMatrixWang,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from scipy.linalg import eigh

from lawaf.mathutils.align_evecs import align_all_degenerate_eigenvectors

# from minimulti.ioput.ifc_netcdf import save_ifc_to_netcdf
from lawaf.plot import plot_band
from lawaf.utils.kpoints import kmesh_to_R

# from supercellmap import SupercellMaker
from lawaf.utils.supercell import SupercellMaker

from .ifcwrapper import IFC

__all__ = ["PhonopyWrapper", "replace_phonon_dynamics_with_myGL"]


class MyDynamicalMatrixGL(DynamicalMatrixGL):
    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        nac_params=None,
        num_G_points=None,
        with_full_terms=False,
        decimals=None,
        symprec=0.00001,
        log_level=0,
        use_openmp=False,
    ):
        super().__init__(
            supercell=supercell,
            primitive=primitive,
            force_constants=force_constants,
            nac_params=nac_params,
            num_G_points=num_G_points,
            with_full_terms=with_full_terms,
            decimals=decimals,
            log_level=log_level,
            use_openmp=use_openmp,
        )
        self._short_range_dynamical_matrix = None
        self._long_range_dynamical_matrix = None

    def get_dynamical_matrix(self, q, lang="C", split_short_long=False):
        """
        Return dynamical matrix at q.
        Parameters:
        ----------
        q: array_like, shape=(3,) q-vector in reduced coordinates
        split_short_long: bool, optional, default=False If True, return
            short-range and long-range dynamical matrices in addition to
            the total dynamical matrix.
        Returns:
        -------
        dynamical_matrix: ndarray, shape=(natom*3, natom*3) Dynamical matrix
            at q.
        short_range_dynamical_matrix: ndarray, shape=(natom*3, natom*3)
            Short-range dynamical matrix at q. This is available only when
            split_short_long is True.
        long_range_dynamical_matrix: ndarray, shape=(natom*3, natom*3)
            Long-range dynamical matrix at q. This is available only when
            split_short_long is True.
        """
        DM = super().get_dynamical_matrix(q, lang)
        if split_short_long:
            return (
                self._short_range_dynamical_matrix,
                self._long_range_dynamical_matrix,
                DM,
            )
        else:
            return DM

    def _compute_dynamical_matrix(self, q_red, q_direction):
        if self._Gonze_force_constants is None:
            self.make_Gonze_nac_dataset()

        if self._log_level > 2:
            print("%d %s" % (self._Gonze_count + 1, q_red))
        self._Gonze_count += 1
        fc = self._force_constants
        self._force_constants = self._Gonze_force_constants
        self._run(q_red)
        self._force_constants = fc
        dm_dd = self._get_Gonze_dipole_dipole(q_red, q_direction)

        self._short_range_dynamical_matrix = copy.deepcopy(self._dynamical_matrix)
        self._long_range_dynamical_matrix = copy.deepcopy(dm_dd)
        self._dynamical_matrix += dm_dd


def my_get_dynamical_matrix(
    fc2,
    supercell: PhonopyAtoms,
    primitive: Primitive,
    nac_params=None,
    frequency_scale_factor=None,
    decimals=None,
    symprec=1e-5,
    log_level=0,
    use_openmp=False,
):
    """Return dynamical matrix.

    The instance of a class inherited from DynamicalMatrix will be returned
    depending on paramters.

    """
    if frequency_scale_factor is None:
        _fc2 = fc2
    else:
        _fc2 = fc2 * frequency_scale_factor**2

    if nac_params is None:
        dm = DynamicalMatrix(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            use_openmp=use_openmp,
        )
    else:
        if "method" not in nac_params:
            method = "gonze"
        else:
            method = nac_params["method"]

        DM_cls: Union[Type[MyDynamicalMatrixGL], Type[DynamicalMatrixWang]]
        if method == "wang":
            DM_cls = DynamicalMatrixWang
        else:
            DM_cls = MyDynamicalMatrixGL
        dm = DM_cls(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            symprec=symprec,
            log_level=log_level,
            use_openmp=use_openmp,
        )
        dm.nac_params = nac_params
    return dm


class PhonopyWrapper:
    phonon: Phonopy = None

    def __init__(
        self,
        phonon=None,
        phonon_fname="phonopy_params.yaml",
        mode="ifc",
        is_nac=False,
        use_cache=True,
    ):
        if phonon is not None:
            self.phonon = phonon
        else:
            self.phonon = load(phonon_fname, is_nac=is_nac)
            print(f"Phonon loaded from file {phonon_fname} ")
        if is_nac:
            self.is_nac = True
            self.get_nac_params()
            # print("replace_phonon_dynamics_with_myGL")
            replace_phonon_dynamics_with_myGL(self.phonon)
        else:
            self.is_nac = False

        # if self.is_nac:
        #    print(
        #        f"Phonopy DM replaces with myGL. Has NAC: {self.is_nac} \n"
        #        + f" born charges: {self.born}, dielectric constant: {self.dielectric}"
        #    )

        self._prepare()
        prim = self.phonon.get_primitive()
        self.atoms = Atoms(
            prim.get_chemical_symbols(),
            positions=prim.get_positions(),
            cell=prim.get_cell(),
        )
        self.natom = len(self.atoms)
        self._positions = np.repeat(self.atoms.get_scaled_positions(), 3, axis=0)
        self.dr = self._positions[None, :, :] - self._positions[:, None, :]
        self.mode = mode.lower()
        assert self.mode in ["ifc", "dm"]
        masses = np.kron(self.atoms.get_masses(), [1, 1, 1])
        self.Mmat = np.sqrt(masses[:, None] * masses[None, :])
        self._use_cache = use_cache
        self._prepare_cache()
        atexit.register(self.save_cache)

        self.is_orthogonal = True

    def _prepare_cache(self):
        if self._use_cache:
            self._cache_file = "./phon_cache/cache.pickle"
            # make cache directory
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            if os.path.exists(self._cache_file):
                with lzma.open(self._cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            else:
                self._cache = {}

    def save_cache(self):
        if self._use_cache:
            with lzma.open(self._cache_file, "wb") as f:
                pickle.dump(self._cache, f)

    def get_nac_params(self):
        nac_params = self.phonon.get_nac_params()
        self.born = nac_params["born"]
        self.dielectric = nac_params["dielectric"]
        self.factor = nac_params["factor"]
        return self.born, self.dielectric, self.factor

    def _prepare(self):
        self.phonon.symmetrize_force_constants()
        # self.phonon.symmetrize_force_constants_by_space_group()
        pass

    def solve(self, k, output_H=False):
        # Hk = self.phonon.get_dynamical_matrix_at_q(k)
        if self.phonon._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        key = (f"{k[0]:8.5f},{k[1]:8.5f},{k[2]:8.5f}", output_H)
        if self._use_cache:
            if key in self._cache:
                return self._cache[key]

        if self.is_nac:
            # replace_phonon_dynamics_with_myGL(self.phonon)
            # self.phonon._dynamical_matrix.run(k)
            # return self.phonon._dynamical_matrix.get_dynamical_matrix()
            # self.phonon.dynamical_matrix._compute_dynamical_matrix(k, [0, 0, 0])
            # Hk, Hshort, Hlong = self.phonon.dynamical_matrix.get_dynamical_matrix(split_short_long=True)
            Hk = self.phonon.get_dynamical_matrix_at_q(k)
            # Hk=self.phonon.dynamical_matrix.dynamical_matrix
            Hshort = self.phonon.dynamical_matrix._short_range_dynamical_matrix
            Hlong = self.phonon.dynamical_matrix._long_range_dynamical_matrix
        else:
            Hk = self.phonon.get_dynamical_matrix_at_q(k)
        phase = np.exp(-2.0j * np.pi * np.einsum("ijk, k->ij", self.dr, k))
        Hk *= phase
        if self.is_nac:
            Hshort *= phase
            Hlong *= phase
        if self.mode == "ifc":
            Hk *= self.Mmat
            if self.is_nac:
                Hshort *= self.Mmat
                Hlong *= self.Mmat
        evals, evecs = eigh(Hk)
        evals, evecs = align_all_degenerate_eigenvectors(evals, evecs)
        if output_H:
            res = evals, evecs, Hk, Hshort, Hlong
        else:
            res = evals, evecs

        if self._use_cache:
            self._cache[key] = res
        return res

    def remove_phase(self, Hk, k):
        phase = np.exp(-2.0j * np.pi * np.einsum("ijk, k->ij", self.dr, k))
        Hk *= phase
        return Hk

    def assure_ASR(self, HR, Rlist):
        igamma = np.argmin(np.linalg.norm(Rlist, axis=1))
        shift = np.sum(np.sum(HR, axis=0), axis=0)
        HR[igamma] -= np.diag(shift)
        return HR

    def get_ifc(self, kmesh, eval_modify_function=None, assure_ASR=False):
        kpts = monkhorst_pack(kmesh)
        nk = len(kpts)
        Rpts = kmesh_to_R(kmesh)
        natoms = len(self.atoms)
        HR = np.zeros((len(Rpts), natoms * 3, natoms * 3), dtype=complex)
        for k in kpts:
            Hk = self.phonon.get_dynamical_matrix_at_q(k)
            Hk *= np.exp(-2.0j * np.pi * np.einsum("ijk, k->ij", self.dr, k))
            Hk *= self.Mmat
            if eval_modify_function is not None:
                evals, evecs = eigh(Hk)
                # sumne = np.sum(evals[evals < 0])
                # if sumne < -1e-18:
                #    print(k, sumne)
                evals = eval_modify_function(evals)
                Hk = evecs.conj() @ np.diag(evals) @ evecs.T
            for iR, R in enumerate(Rpts):
                HR[iR] += (Hk * np.exp(2.0j * np.pi * np.dot(R, k))) / nk
        if assure_ASR:
            HR.imag = 0.0
            HR[np.abs(HR) < 0.001] = 0.0
            HR = self.assure_ASR(HR, Rpts)
        return Rpts, HR

    def save_ifc(self, fname, kmesh, eval_modify_function=None, assure_ASR=False):
        Rpts, HR = self.get_ifc(
            kmesh, eval_modify_function=eval_modify_function, assure_ASR=assure_ASR
        )
        # save_ifc_to_netcdf(fname, HR, Rpts, self.atoms)

    def solve_all(self, kpts, output_H=False):
        evals = []
        evecs = []
        Hks = []
        Hshorts = []
        Hlongs = []
        if not output_H:
            for ik, k in enumerate(kpts):
                evalue, evec = self.solve(k)
                evals.append(evalue)
                evecs.append(evec)
        else:
            for ik, k in enumerate(kpts):
                evalue, evec, Hk, Hshort, Hlong = self.solve(k, output_H=output_H)
                evals.append(evalue)
                evecs.append(evec)
                Hks.append(Hk)
                Hshorts.append(Hshort)
                Hlongs.append(Hlong)
        if output_H:
            res = (
                np.array(evals, dtype=float),
                np.array(evecs, dtype=complex, order="C"),
                np.array(Hks, dtype=complex, order="C"),
                np.array(Hshorts, dtype=complex, order="C"),
                np.array(Hlongs, dtype=complex, order="C"),
            )
        else:
            res = (
                np.array(evals, dtype=float),
                np.array(evecs, dtype=complex, order="C"),
            )
        self.save_cache()
        return res

    @property
    def positions(self):
        return self._positions

    def plot_band(self, **kwargs):
        ax = plot_band(self, **kwargs)
        return ax

    def build_distorted_supercell2(self, supercell_matrix, modes):
        self.phonon.set_modulations(dimension=supercell_matrix, phonon_modes=modes)
        # cells=self.phonon.get_modulated_supercells()
        modulations, supercell = self.phonon.get_modulations_and_supercell()
        scatoms = Atoms(
            numbers=supercell.get_atomic_numbers(),
            cell=supercell.get_cell(),
            positions=supercell.get_positions(),
        )
        datoms = copy.deepcopy(scatoms)
        disp = sum(modulations)
        datoms.set_positions(datoms.get_positions() + disp)
        return scatoms, datoms

    def build_distorted_supercell(self, supercell_matrix, modes):
        """
        supercell_matrix
        modes: kpt, index, amp, phase, modulation_func
        """
        scmaker = SupercellMaker(supercell_matrix)
        sc_atoms = scmaker.sc_atoms(self.atoms)
        distorted_atoms = copy.deepcopy(sc_atoms)
        positions = distorted_atoms.get_positions()
        d = 0
        for mode in modes:
            kpt, index, amp, phase, modulation_func = mode
            _, evec = self.solve(kpt)[:2]
            R = scmaker.Rvector_for_each_element(n_ind=self.natom * 3)
            # print(evec[index].reshape((self.natom, 3)))
            disp = (
                scmaker.sc_trans_kvector(evec[index], kpt=kpt, phase=phase, real=True)
                * amp
            )
            p = np.exp(
                2j
                * np.pi
                * np.einsum("ij, j->i", self.atoms.get_scaled_positions(), kpt)
            )
            p = np.kron(np.ones(scmaker.ncell), np.kron([1, 1, 1], p))
            # print(p.shape)
            # print(disp.shape)
            disp *= p.real
            # print(f"{disp=}")
            if modulation_func is None:
                pass
            else:
                disp *= np.array(map(modulation_func, R))

            d += disp
        d = d.reshape((self.natom * scmaker.ncell, 3))
        positions += d
        distorted_atoms.set_positions(positions)
        return sc_atoms, distorted_atoms


def replace_phonon_dynamics_with_myGL(phonon):
    """
    replace phonon._dynamical_matrix with MyDynamicalMatrixGL, which implemnts the
    splitting of the short/long-range part of the dynamical matrix.
    """
    if phonon._is_symmetry and phonon._nac_params is not None:
        borns, epsilon = symmetrize_borns_and_epsilon(
            phonon._nac_params["born"],
            phonon._nac_params["dielectric"],
            phonon._primitive,
            symprec=phonon._symprec,
        )
        nac_params = phonon._nac_params.copy()
        nac_params.update({"born": borns, "dielectric": epsilon})
    else:
        nac_params = phonon._nac_params

    phonon._dynamical_matrix = my_get_dynamical_matrix(
        phonon._force_constants,
        phonon._supercell,
        phonon._primitive,
        nac_params,
        phonon._frequency_scale_factor,
        phonon._dynamical_matrix_decimals,
        symprec=phonon._symprec,
        log_level=phonon._log_level,
        use_openmp=False,  # phonon.use_openmp(),
    )


def save_ifc_and_show_phonon(
    fname="phonopy_params.yaml",
    ifc_fname="ifc.nc",
    kmesh=[3, 3, 3],
    assure_ASR=True,
    knames=["$\\Gamma$", "M", "A", "Z", "R", "M", "$\\Gamma$", "Z"],
    kvectors=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0],
            [0, 0, 0],
            [0.0, 0.0, 0.5],
        ]
    ),
):
    phonon = load(phonopy_yaml=fname)
    phon = PhonopyWrapper(phonon, mode="ifc")
    ax = phon.plot_band(color="blue", kvectors=kvectors, knames=knames)

    phon.save_ifc(
        fname=ifc_fname, kmesh=kmesh, eval_modify_function=None, assure_ASR=assure_ASR
    )
    Rpts, HR = phon.get_ifc(
        kmesh=kmesh, eval_modify_function=None, assure_ASR=assure_ASR
    )
    ifc = IFC(phon.atoms, Rpts, HR)
    ifc.save_to_netcdf("ifc_scaled.nc")
    ifc.plot_band(ax=ax, color="red", kvectors=kvectors, knames=knames)
    plt.ylabel(r"FC (eV/$\AA^2$) ")
    plt.show()


if __name__ == "__main__":
    save_ifc_and_show_phonon()
