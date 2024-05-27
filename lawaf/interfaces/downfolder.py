from dataclasses import dataclass
from typing import Tuple, Union, List
import os
import copy
import json
from ase import Atoms
from lawaf.utils.kpoints import monkhorst_pack
import numpy as np
from lawaf.wannierization import (
    ProjectedWannierizer,
    ScdmkWannierizer,
    MaxProjectedWannierizer,
)
from lawaf.params import WannierParams
import matplotlib.pyplot as plt
from lawaf.plot import plot_band
from lawaf.utils.kpoints import kmesh_to_R, build_Rgrid
from lawaf.utils.kpoints import autopath


def select_wannierizer(method):
    # select the Wannierizer based on the method.
    if method.lower().startswith("scdmk"):
        w = ScdmkWannierizer
    elif method.lower().startswith("projected"):
        w = ProjectedWannierizer
    elif method.lower().startswith("maxprojected"):
        w = MaxProjectedWannierizer
    else:
        raise ValueError("Unknown method")
    print(f"Using {w.__name__} method")
    return w


class Lawaf:
    params = {}
    builder: Union[ProjectedWannierizer, ScdmkWannierizer] = None
    model = None

    def __init__(self, model, params=None):
        """
        Setup the model
        """
        self.model = model
        self.params = {}
        self.builder = None
        self.Rgrid = None

    def set_parameters(
        self,
        method="scdmk",
        kmesh=(5, 5, 5),
        gamma=True,
        nwann=0,
        weight_func="unity",
        weight_func_params={0, 0.01},
        selected_basis=None,
        anchors=None,
        anchor_kpt=(0, 0, 0),
        kshift=np.array([1e-7, 2e-8, 3e-9]),
        use_proj=True,
        exclude_bands=[],
        post_func=None,
    ):
        """
        Downfold the Band structure.
        The method first get the eigenvalues and eigenvectors in a Monkhorst-Pack grid from the model.
        It then use the scdm-k or the projected wannier function method to downfold the Hamiltonian at each k-point.
        And finally it Fourier transform the new basis functions(Wannier functions) from k-space to real space.
        The Hamiltonian can be written.

        Parameters
        ====================================
        method:  the method of downfolding. scdmk|projected
        kmesh,   The k-mesh used for the BZ sampling. e.g. (5, 5, 5)
                 Note that for the moment, only odd number should be used so that the mesh is Gamma centered.
        nwann,   Number of Wannier functions to be constructed.
        weight_func='Gauss',   # The weight function type. 'unity', 'Gauss', 'Fermi', or 'window'
         - unity: all the bands are equally weighted.
         - Gauss: A gaussian centered at mu, and has the half width of sigma.
         - Fermi: A fermi function. The Fermi energy is mu, and the smearing is sigma.
         - window: A window function in the range of (mu, sigma)
        mu: see above
        sigma=2.0 : see above
        selected_basis, A list of the indexes of the Wannier functions as initial guess. The number should be equal to nwann.
        anchors: Anchor points. The index of band at one k-point. e.g.(0, 0, 0): (6, 7, 8)
        anchor_kpt: the kpoint used for automatically selecting of anchor points.
        use_proj: Whether to use projection to the anchor points in the weight function.
        write_hr_nc: write the Hamiltonian into a netcdf file. It require the NETCDF4 python library. use write_nc=None if not needed.
        write_hr_txt: write the Hamiltonian into a txt file.
        """

        self.params = WannierParams(
            method=method,
            kmesh=kmesh,
            gamma=gamma,
            nwann=nwann,
            weight_func=weight_func,
            weight_func_params=weight_func_params,
            selected_basis=selected_basis,
            anchors=anchors,
            anchor_kpt=anchor_kpt,
            kshift=kshift,
            use_proj=use_proj,
            exclude_bands=exclude_bands,
        )
        self.nwann = self.params.nwann

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare the data for the downfolding. e.g. eigen values, eigen vectors, kpoints
        """
        self._prepare_kpoints()
        self._prepare_eigen()
        self._prepare_Rlist()
        self._prepare_positions()
        self._parepare_builder()

    def _parepare_builder(self):
        params = self.params
        wannierizer = select_wannierizer(params.method)
        self.builder = wannierizer(
            evals=self.evals,
            evecs=self.evecs,
            kpts=self.kpts,
            kweights=self.kweights,
            params=self.params,
        )

    def _prepare_positions(self):
        try:
            self.atoms = self.model.atoms
        except:
            self.atoms = Atoms(cell=np.eye(3))
        try:
            positions = self.model.positions
        except Exception:
            positions = None
        self.positions = positions

    def _prepare_kpoints(self):
        """
        Prepare the kpoints
        """
        if self.params.kpts is not None:
            self.kpts = np.array(self.params.kpts)
        else:
            self.kpts = monkhorst_pack(self.params.kmesh, gamma=self.params.gamma)

        self.nkpt = len(self.kpts)
        if self.params.kshift is not None:
            kshift = np.array(self.params.kshift)
            self.kpts += kshift[None, :]
            self.params.anchor_kpt = np.array(self.params.anchor_kpt) + kshift
        if not self.params.kweights:
            self.kweights = np.ones(self.nkpt, dtype=float) / self.nkpt
        else:
            self.kweights = self.params.kweights

        self.anchor_kpt = np.array(self.params.anchor_kpt)

        abs_kdistance_to_anchor = np.abs(
            np.linalg.norm(self.kpts - self.anchor_kpt[None, :], axis=1)
        )
        if not np.any(abs_kdistance_to_anchor < 1e-5):
            self.nkpt += 1
            self.kpts = np.vstack([self.kpts, self.anchor_kpt])
            self.kweights = np.hstack([self.kweights, 0.0])
            self.anchor_kpt = self.kpts[-1]
        else:
            self.anchor_kpt = self.kpts[0]

    def _prepare_Rlist(self):
        self.Rgrid = self.params.kmesh
        if self.Rgrid is None:
            self.Rlist = kmesh_to_R(self.params.kmesh)
        else:
            self.Rlist = build_Rgrid(self.Rgrid)

    def _prepare_eigen(self, has_phase=False):
        evals, evecs = self.model.solve_all(self.kpts)
        # remove e^ikr from wfn
        self.has_phase = has_phase
        if not has_phase:
            self.psi = evecs
        else:
            self._remove_phase(evecs)
        self.evals = evals
        self.evecs = evecs
        return self.evals, self.evecs

    def _remove_phase_k(self, wfnk, k, positions):
        # phase = np.exp(-2j * np.pi * np.einsum('j, kj->k', k, self.positions))
        # return wfnk[:, :] * phase[:, None]
        nbasis = wfnk.shape[0]
        psi = np.zeros_like(wfnk)
        for ibasis in range(self.nbasis):
            phase = np.exp(-2j * np.pi * np.dot(k, positions[ibasis, :]))
            psi[ibasis, :] = wfnk[ibasis, :] * phase
        return psi

    def _remove_phase(self, wfn):
        self.psi = np.zeros_like(wfn)
        for ik, k in enumerate(self.kpts):
            self.psi[ik, :, :] = self._remove_phase_k(wfn[ik, :, :], k)

    def save_info(self, output_path="./", fname="Downfold.json"):
        results = {"params": self.params}
        with open(os.path.join(output_path, fname), "w") as myfile:
            json.dump(results, myfile, sort_keys=True, indent=2)

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc=None,
        write_hr_txt=None,
        **params,
    ):
        self.params.update(params)
        self.atoms = self.model.atoms
        self.ewf = self.builder.get_wannier(Rlist=self.Rlist)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except Exception as E:
            print(E)
            pass
        # if write_hr_txt is not None:
        #    self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        # if write_hr_nc is not None:
        #    # self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        #    self.ewf.write_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        return self.ewf

    def plot_band_fitting(
        self,
        # kvectors=np.array(
        #    [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        # ),
        # knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        kvectors=None,
        knames=None,
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=None,
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig="Downfolded_band.png",
        cell=np.eye(3),
        plot_original=True,
        plot_downfolded=True,
        plot_nonac=False,
        show=True,
        fix_LOTO=False,
        **kwargs,
    ):
        """
        Parameters:
        ========================================
        kvectors: coordinates of special k-points
        knames: names of special k-points
        supercell_matrix: If the structure is a supercell, the band can be in the primitive cell.
        npoints: number of k-points in the band.
        efermi: Fermi energy.
        erange: range of energy to be shown. e.g. [-5,5]
        fullband_color: the color of the full band structure.
        downfolded_band_color: the color of the downfolded band structure.
        marker: the marker of the downfolded band structure.
        ax: matplotlib axes object.
        savefig: the filename of the figure to be saved.
        show: whether to show the band structure.
        """
        if plot_original:
            ax = plot_band(
                self.model,
                kvectors=kvectors,
                knames=knames,
                supercell_matrix=supercell_matrix,
                npoints=npoints,
                color=fullband_color,
                alpha=0.8,
                marker="",
                erange=erange,
                efermi=efermi,
                cell=cell,
                ax=ax,
                fix_LOTO=fix_LOTO,
                **kwargs,
            )
        if plot_downfolded:
            ax = plot_band(
                self.ewf,
                kvectors=kvectors,
                knames=knames,
                supercell_matrix=supercell_matrix,
                npoints=npoints,
                efermi=efermi,
                color=downfolded_band_color,
                alpha=0.5,
                marker=marker,
                erange=erange,
                cell=cell,
                ax=ax,
                fix_LOTO=fix_LOTO,
                **kwargs,
            )
        if plot_nonac:
            self.ewf.set_nac(False)
            ax = plot_band(
                self.ewf,
                kvectors=kvectors,
                knames=knames,
                supercell_matrix=supercell_matrix,
                npoints=npoints,
                efermi=efermi,
                color="red",
                alpha=0.5,
                marker=marker,
                erange=erange,
                cell=cell,
                ax=ax,
                fix_LOTO=fix_LOTO,
                **kwargs,
            )
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        return ax
