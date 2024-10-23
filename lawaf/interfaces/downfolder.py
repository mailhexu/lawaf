import json
import os

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from HamiltonIO.lawaf import LawafHamiltonian as EWF

from lawaf.mathutils.kR_convert import k_to_R
from lawaf.params import WannierParams
from lawaf.plot import plot_band
from lawaf.utils.kpoints import build_Rgrid, monkhorst_pack
from lawaf.wannierization import (
    DummyWannierizer,
    MaxProjectedWannierizer,
    ProjectedWannierizer,
    ScdmkWannierizer,
)


def select_wannierizer(method):
    # select the Wannierizer based on the method.
    if method.lower().startswith("scdm"):
        w = ScdmkWannierizer
    elif method.lower().startswith("projected") or method.lower().startswith("pwf"):
        w = ProjectedWannierizer
    elif method.lower().startswith("maxprojected"):
        w = MaxProjectedWannierizer
    elif method.lower().startswith("dummy"):
        w = DummyWannierizer
    else:
        raise ValueError(f"Unknown method: {method}")
    print(f"Using {w.__name__} method")
    return w


class Lawaf:
    def __init__(self, model, params=None):
        """
        Setup the model
        """
        self._params = {}
        self.model = model
        self.builder = None
        self.Rgrid = None
        self.params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        if isinstance(params, dict):
            self.set_parameters(**params)
        elif isinstance(params, WannierParams):
            self._params = params
        else:
            self._params = WannierParams()
        self.nwann = self.params.nwann
        self.process_parameters()

    def process_parameters(self):
        pass

    def set_parameters(
        self,
        method="scdmk",
        kmesh=(5, 5, 5),
        gamma=True,
        nwann=0,
        weight_func="unity",
        weight_func_params=(0, 0.01),
        selected_basis=None,
        anchors=None,
        anchor_kpt=(0, 0, 0),
        kshift=np.array([0, 0, 0], dtype=float),
        use_proj=True,
        proj_order=1,
        exclude_bands=[],
        post_func=None,
        enhance_Amn=0,
        selected_orbdict=None,
        orthogonal=False,
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

        self._params = WannierParams(
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
            proj_order=proj_order,
            exclude_bands=exclude_bands,
            enhance_Amn=enhance_Amn,
            selected_orbdict=selected_orbdict,
            orthogonal=orthogonal,
        )

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
            Hk=self.Hk,
            Sk=self.Sk,
        )

    def _prepare_positions(self):
        if hasattr(self.model, "atoms"):
            self.atoms = self.model.atoms
        else:
            self.atoms = Atoms(cell=np.eye(3))

        if hasattr(self.model, "positions"):
            self.positions = self.model.positions
        else:
            self.positions = None

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
        self.Rlist, self.Rdeg = build_Rgrid(self.Rgrid, degeneracy=True)

    def _prepare_eigen(self, has_phase=False):
        self.Hk = None
        self.Sk = None
        # evals, evecs = self.model.solve_all(self.kpts)
        if self.model.is_orthogonal:
            evals, evecs = self.model.solve_all(self.kpts)
            H = S = None
        else:
            H, S, evals, evecs = self.model.HS_and_eigen(self.kpts)
        # remove e^ikr from wfn
        self.has_phase = has_phase
        if not has_phase:
            self.psi = evecs
        else:
            self._remove_phase(evecs)
        # TODO: to be removed
        self.Hk = H
        self.Sk = S
        self.evals = evals
        self.evecs = evecs
        if not self.model.is_orthogonal:
            self.is_orthogonal = False
            self.Sk = S
        else:
            self.is_orthogonal = True
            self.Sk = None
        return self.evals, self.evecs

    def _remove_phase_k(self, wfnk, k, positions):
        # phase = np.exp(-2j * np.pi * np.einsum('j, kj->k', k, self.positions))
        # return wfnk[:, :] * phase[:, None]
        # nbasis = wfnk.shape[0]
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
        **params,
    ):
        self.params.update(params)
        self._prepare_data()
        # self.params.update(params)
        self.atoms = self.model.atoms
        # self.lwf = self.builder.get_wannier(Rlist=self.Rlist, Rdeg=self.Rdeg)
        self.builder.get_Amn()
        wannk, Hwannk, Swannk = self.builder.get_wannk_and_Hk()
        wannR = k_to_R(
            self.kpts, self.Rlist, wannk, kweights=self.kweights, Rdeg=self.Rdeg
        )
        HwannR = k_to_R(
            self.kpts, self.Rlist, Hwannk, kweights=self.kweights, Rdeg=self.Rdeg
        )
        if Swannk is not None:
            SwannR = k_to_R(
                self.kpts, self.Rlist, Swannk, kweights=self.kweights, Rdeg=self.Rdeg
            )
        else:
            SwannR = None

        self.lwf = EWF(
            wannR=wannR,
            HwannR=HwannR,
            SwannR=SwannR,
            Rlist=self.Rlist,
            Rdeg=self.Rdeg,
            atoms=self.atoms,
            wann_names=None,
            is_orthogonal=(SwannR is None),
        )
        return self.lwf

    def plot_full_band(
        self,
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
        return ax

    def plot_wannier_band(
        self,
        kvectors=None,
        knames=None,
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=None,
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
        ax = plot_band(
            self.lwf,
            kvectors=kvectors,
            knames=knames,
            supercell_matrix=supercell_matrix,
            npoints=npoints,
            efermi=efermi,
            color=downfolded_band_color,
            alpha=0.3,
            marker=marker,
            erange=erange,
            cell=cell,
            ax=ax,
            fix_LOTO=fix_LOTO,
            **kwargs,
        )

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
                self.lwf,
                kvectors=kvectors,
                knames=knames,
                supercell_matrix=supercell_matrix,
                npoints=npoints,
                efermi=efermi,
                color=downfolded_band_color,
                alpha=0.3,
                marker=marker,
                erange=erange,
                cell=cell,
                ax=ax,
                fix_LOTO=fix_LOTO,
                **kwargs,
            )
        if plot_nonac:
            self.lwf.set_nac(False)
            ax = plot_band(
                self.lwf,
                kvectors=kvectors,
                knames=knames,
                supercell_matrix=supercell_matrix,
                npoints=npoints,
                efermi=efermi,
                color="red",
                alpha=0.3,
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
