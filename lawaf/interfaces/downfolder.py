from dataclasses import dataclass
from typing import Tuple, Union, List
import os
import copy
import json
from lawaf.utils.kpoints import monkhorst_pack
import numpy as np
from lawaf.wannierization.scdmk import (
    WannierProjectedBuilder,
    WannierScdmkBuilder,
    occupation_func,
)
import matplotlib.pyplot as plt
from lawaf.plot import plot_band


@dataclass
class WFParams:
    method = "scdmk"
    kmesh: Tuple[int] = (5, 5, 5)
    kshift = np.array([1e-7, 3e-6, 5e-9])
    gamma: bool = True
    nwann: int = 0
    weight_func: str = "unity"
    mu: float = 0.0
    sigma: float = 2.0
    selected_basis: Union[None, List[int]] = None
    anchors: Union[None, List[int]] = None
    anchor_kpt: Tuple[int] = (0, 0, 0)
    use_proj: bool = True
    exclude_bands: Tuple[int] = ()


def make_builder(
    model,
    kmesh,
    nwann,
    has_phase=False,
    weight_func="unity",
    mu=0,
    sigma=0.01,
    kshift=None,
    exclude_bands=[],
    use_proj=False,
    method="projected",
    selected_basis=[],
    anchor_kpt=(0, 0, 0),
    anchors=None,
    gamma=True,
):
    k = kmesh[0]
    kpts = monkhorst_pack(kmesh, gamma=gamma) 
    nk = len(kpts)
    kweights = [1.0 / nk for k in kpts]
    if kshift is not None:
        kpts+=kshift[None, :]
        anchor_kpt=np.array(anchor_kpt)+kshift

    #if model.has_nac:
    #    evals, evecs, Hks, Hshorts, Hlongs = model.solve_all(kpts)
    #else:
    evals, evecs = model.solve_all(kpts)

    anchor_kpts = []
    if anchors is not None:
        for anchor in anchors:
            anchor_kpts.append(anchor)
    if anchor_kpt is not None:
        anchor_kpts.append(anchor_kpt)
    evals_anchor, evecs_anchor = model.solve_all(anchor_kpts)[0:2]
    wfn_anchor = {}
    for ik, k in enumerate(anchor_kpts):
        wfn_anchor[tuple(k)] = evecs_anchor[ik, :, :]

    try:
        positions = model.positions
    except Exception:
        positions = None
    if isinstance(weight_func, str):
        weight_func = occupation_func(ftype=weight_func, mu=mu, sigma=sigma)

    if method == "scdmk":
        wann_builder = WannierScdmkBuilder(
            evals=evals,
            wfn=evecs,
            positions=positions,
            kpts=kpts,
            kweights=kweights,
            #kshift=kshift,
            nwann=nwann,
            weight_func=weight_func,
            has_phase=has_phase,
            Rgrid=kmesh,
            exclude_bands=exclude_bands,
            use_proj=use_proj,
            wfn_anchor=wfn_anchor,
        )
        if selected_basis:
            wann_builder.set_selected_cols(selected_basis)
        elif anchors:
            wann_builder.set_anchors(anchors)
        else:
            wann_builder.auto_set_anchors(anchor_kpt)

    elif method == "projected":
        wann_builder = WannierProjectedBuilder(
            evals=evals,
            wfn=evecs,
            has_phase=has_phase,
            positions=positions,
            kpts=kpts,
            kweights=kweights,
            kshift=kshift,
            nwann=nwann,
            weight_func=weight_func,
            Rgrid=kmesh,
            exclude_bands=exclude_bands,
            wfn_anchor=wfn_anchor,
        )
        if selected_basis:
            wann_builder.set_projectors_with_basis(selected_basis)
        elif anchors:
            wann_builder.set_projectors_with_anchors(anchors)
    else:
        raise ValueError("method should be scdmk or projected")

    #if model.has_nac:
    #    wann_builder.set_nac_Hks(Hks, Hshorts, Hlongs)
    return wann_builder


class Lawaf:
    params = {}
    builder: Union[WannierProjectedBuilder, WannierScdmkBuilder] = None
    model  = None

    def __init__(self, model):
        """
        Setup the model
        """
        self.model = model
        self.params = {}
        self.builder: Union[WannierProjectedBuilder, WannierScdmkBuilder] = None

    def set_parameters(
        self,
        method="scdmk",
        kmesh=(5, 5, 5),
        gamma=True,
        nwann=0,
        weight_func="unity",
        mu=0.0,
        sigma=2.0,
        selected_basis=None,
        anchors=None,
        anchor_kpt=(0, 0, 0),
        kshift=np.array([1e-7, 2e-8, 3e-9]),
        use_proj=True,
        exclude_bands=[],
        post_func=None,
        has_nac=False,
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

        self.params = copy.deepcopy(locals())
        self.params.pop("self")

    def save_info(self, output_path="./", fname="Downfold.json"):
        results = {"params": self.params}
        with open(os.path.join(output_path, fname), "w") as myfile:
            json.dump(results, myfile, sort_keys=True, indent=2)

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
        **params,
    ):
        self.params.update(params)
        if "post_func" in self.params:
            self.params.pop("post_func")
        self.builder = make_builder(self.model, **self.params)
        self.atoms = self.model.atoms
        self.ewf = self.builder.get_wannier()
        if post_func is not None:
            post_func(self.ewf)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except:
            pass
        if write_hr_txt is not None:
            self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        if write_hr_nc is not None:
            # self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
            self.ewf.write_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        return self.ewf

    def plot_band_fitting(
        self,
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
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
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()
        return ax


