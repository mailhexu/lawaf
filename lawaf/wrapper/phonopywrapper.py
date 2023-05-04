import copy
from scipy.linalg import eigh
import numpy as np
from phonopy import load, Phonopy
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack
from wannierbuilder.utils.kpoints import kmesh_to_R, build_Rgrid
#from minimulti.ioput.ifc_netcdf import save_ifc_to_netcdf
from wannierbuilder.plot import plot_band
from wannierbuilder.wrapper.ifcwrapper import IFC
import matplotlib.pyplot as plt
#from supercellmap import SupercellMaker
from wannierbuilder.utils.supercell import SupercellMaker


class PhonopyWrapper():
    def __init__(self,
                 phonon=None,
                 phonon_fname='phonopy_params.yaml',
                 mode='ifc'):
        if phonon is not None:
            self.phonon = phonon
        else:
            self.phonon = load(phonon_fname)
        self._prepare()
        prim = self.phonon.get_primitive()
        self.atoms = Atoms(prim.get_chemical_symbols(),
                           positions=prim.get_positions(),
                           cell=prim.get_cell())
        self.natom = len(self.atoms)
        self._positions = np.repeat(self.atoms.get_scaled_positions(),
                                    3,
                                    axis=0)
        self.dr = self._positions[None, :, :] - self._positions[:, None, :]
        self.mode = mode.lower()
        assert self.mode in ['ifc', 'dm']
        masses = np.kron(self.atoms.get_masses(), [1, 1, 1])
        self.Mmat = np.sqrt(masses[:, None] * masses[None, :])

    def _prepare(self):
        self.phonon.symmetrize_force_constants()
        # self.phonon.symmetrize_force_constants_by_space_group()
        pass

    def solve(self, k):
        Hk = self.phonon.get_dynamical_matrix_at_q(k)
        Hk *= np.exp(-2.0j * np.pi * np.einsum('ijk, k->ij', self.dr, k))
        if self.mode == 'ifc':
            Hk *= self.Mmat
        evals, evecs = eigh(Hk)
        return evals, evecs

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
            Hk *= np.exp(-2.0j * np.pi * np.einsum('ijk, k->ij', self.dr, k))
            Hk *= self.Mmat
            if eval_modify_function is not None:
                evals, evecs = eigh(Hk)
                sumne = np.sum(evals[evals < 0])
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

    def save_ifc(self,
                 fname,
                 kmesh,
                 eval_modify_function=None,
                 assure_ASR=False):
        Rpts, HR = self.get_ifc(kmesh,
                                eval_modify_function=eval_modify_function,
                                assure_ASR=assure_ASR)
        save_ifc_to_netcdf(fname, HR, Rpts, self.atoms)

    def solve_all(self, kpts):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs,
                                                      dtype=complex,
                                                      order='C')

    @property
    def positions(self):
        return self._positions

    def plot_band(self, **kwargs):
        ax = plot_band(self, **kwargs)
        return ax

    def build_distorted_supercell2(self, supercell_matrix, modes):
        self.phonon.set_modulations(
            dimension=supercell_matrix, phonon_modes=modes)
        # cells=self.phonon.get_modulated_supercells()
        modulations, supercell = self.phonon.get_modulations_and_supercell()
        scatoms = Atoms(numbers=supercell.get_atomic_numbers(
        ), cell=supercell.get_cell(), positions=supercell.get_positions())
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
            _, evec = self.solve(kpt)
            R = scmaker.Rvector_for_each_element(n_ind=self.natom * 3)
            #print(evec[index].reshape((self.natom, 3)))
            disp = scmaker.sc_trans_kvector(
                evec[index], kpt=kpt, phase=phase, real=True) * amp
            p = np.exp(2j * np.pi * np.einsum('ij, j->i',
                                              self.atoms.get_scaled_positions(), kpt))
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


def save_ifc_and_show_phonon(
    fname='phonopy_params.yaml',
    ifc_fname='ifc.nc',
    kmesh=[3, 3, 3],
    assure_ASR=True,
    knames=['$\\Gamma$', 'M', 'A', 'Z', 'R', 'M', '$\\Gamma$', 'Z'],
    kvectors=np.array([[0., 0., 0.], [0.5, 0.5, 0.], [0.5, 0.5, 0.5],
                       [0., 0., 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0],
                       [0, 0, 0], [0.0, 0.0, 0.5]]),
):
    phonon = load(phonopy_yaml=fname)
    phon = PhonopyWrapper(phonon, mode='ifc')
    ax = phon.plot_band(color='blue', kvectors=kvectors, knames=knames)

    phon.save_ifc(fname=ifc_fname,
                  kmesh=kmesh,
                  eval_modify_function=None,
                  assure_ASR=assure_ASR)
    Rpts, HR = phon.get_ifc(kmesh=kmesh,
                            eval_modify_function=None,
                            assure_ASR=assure_ASR)
    ifc = IFC(phon.atoms, Rpts, HR)
    ifc.save_to_netcdf("ifc_scaled.nc")
    ifc.plot_band(ax=ax, color='red', kvectors=kvectors, knames=knames)
    plt.ylabel("FC (eV/$\AA^2$) ")
    plt.show()


if __name__ == '__main__':
    save_ifc_and_show_phonon()
