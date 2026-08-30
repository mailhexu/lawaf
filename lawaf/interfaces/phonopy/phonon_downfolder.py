import os

import numpy as np

from lawaf.interfaces.downfolder import Lawaf
from lawaf.mathutils.evals_freq import freqs_to_evals
from lawaf.mathutils.kR_convert import R_to_onek, k_to_R
from lawaf.mathutils.ws_distance import apply_ws_distance_tensors, fold_R_to_mesh
from .lwf import LWF, NACLWF
from .phonopywrapper import PhonopyWrapper

__all__ = ["PhononDownfolder", "PhonopyDownfolder", "NACPhonopyDownfolder"]


def _ws_mesh_ok(downfolder):
    """WS materialization assumes all degenerate images share the phase at
    mesh k, which only holds for Gamma-centered (unshifted) meshes."""
    ks = getattr(downfolder.params, "kshift", None)
    kshift = np.zeros(3) if ks is None else np.asarray(ks, dtype=float)
    gamma = getattr(downfolder.params, "gamma", True)
    if not gamma:
        # half-shifted Monkhorst-Pack: k . N is half-integer -> phase -1
        return False
    return np.allclose(kshift, 0.0, atol=1e-8)


def _ws_materialize(downfolder, tensors, wann_centers):
    """Apply W90 Wigner-Seitz materialization to named R-space tensors.

    Tensors are classified by ROLE, not shape: only the key "wannR"
    (nbasis x nwann amplitudes) uses the atomic positions repeated per
    Cartesian direction as row centers; all other (Hamiltonian-like)
    tensors use the Wannier centers on both sides. This stays correct for
    full-band models where nbasis == nwann. All tensors are scattered onto
    one common union R grid; downfolder.Rlist/Rdeg are replaced.
    """
    positions = downfolder.atoms.get_scaled_positions()
    cell = np.array(downfolder.atoms.get_cell())
    centers_list = []
    centers_j_list = []
    for name, T in tensors.items():
        if name == "wannR":
            centers_list.append(np.repeat(positions, 3, axis=0)[: T.shape[1]])
            centers_j_list.append(wann_centers)
        else:
            centers_list.append(wann_centers)
            centers_j_list.append(wann_centers)
    tensors_folded, Rlist_folded = fold_R_to_mesh(
        downfolder.Rlist, list(tensors.values()), downfolder.params.kmesh,
        Rdeg=downfolder.Rdeg)
    res, Rlist_ws, Rdeg_ws = apply_ws_distance_tensors(
        tensors_folded, Rlist_folded, centers_list, cell,
        downfolder.params.kmesh, centers_j_list=centers_j_list)
    return dict(zip(tensors, res)), Rlist_ws, Rdeg_ws



class PhononDownfolder(Lawaf):
    def __init__(self, model, atoms=None, params=None):
        super().__init__(model, params=params)
        self.model = model
        if atoms is not None:
            self.atoms = atoms
        else:
            try:
                self.atoms = self.model.atoms
            except Exception:
                self.atoms = None


class PhonopyDownfolder(PhononDownfolder):
    def __init__(
        self, phonon=None, mode="dm", params=None, is_nac=False, *argv, **kwargs
    ):
        """
        Parameters:
        ========================================
        folder: folder of siesta calculation
        fdf_file: siesta input filename
        """
        try:
            import phonopy
        except ImportError:
            raise ImportError("phonopy is needed. Do you have phonopy installed?")
        if phonon is None:
            phonon = phonopy.load(*argv, **kwargs)
        self.mode = mode
        self.factor = 524.16  # to cm-1
        self.is_nac = is_nac
        model = PhonopyWrapper(phonon, mode=mode, is_nac=self.is_nac)
        super().__init__(model, atoms=model.atoms, params=params)

    def convert_DM_parameters(self):
        """
        convert the parameters of the dynamical matrix. Unit from frequency cm-1 to
        eigenvalues in eV.
        """
        if self.mode.lower() == "dm":
            p = self.params.weight_func_params
            if p is not None:
                print(f"Converted DM parameters from: {p}")
                p = [freqs_to_evals(pe, factor=self.factor) for pe in p]
                self.params.weight_func_params = p
                print(f"Converted DM parameters to: {p}")

    def process_parameters(self):
        self.convert_DM_parameters()

    def _parepare_builder(self):
        self._resolve_window_bands()
        super()._parepare_builder()
        if getattr(self.params, "symmetry_seed", False):
            self._inject_symmetry_seeds()
            # The anchor-consuming setup (set_anchors / projectors) runs
            # inside the builder's __init__→set_params chain with
            # wfn_anchor still None, so binding the seeds afterwards alone
            # would be ineffective (review round 1, REG finding). Re-run
            # the anchor setup now that the seeds are bound; both
            # wannierizer set_params overrides reset their anchor state and
            # rebuild it from params, so this is idempotent.
            self.builder.set_params(self.params)

    def _resolve_window_bands(self):
        """Validate and star-expand explicit phonon mode selections."""
        requested = getattr(self.params, "window_bands", None)
        if requested is None:
            if hasattr(self.params, "_window_bands_resolved"):
                del self.params._window_bands_resolved
            return

        from lawaf.anharmonic.representation import (
            WindowBands,
            build_space_group_action,
            check_window_legality,
        )

        if isinstance(requested, WindowBands):
            resolved = requested
        else:
            action = build_space_group_action(self.model.phonon)
            resolved = check_window_legality(
                action,
                self.model.phonon,
                requested,
            )
        self.params._window_bands_resolved = resolved
        self.window_bands = resolved

    def _inject_symmetry_seeds(self):
        """OPD-canonical anchor seeds (story-012, ADR-002/FR-008).

        Computes ``get_symmetry_anchor_wfn`` for every anchor q on the
        non-NAC model and assigns ``builder.wfn_anchor`` post-construction
        (wannierizer constructors accept no such kwarg — FEAS-001; the
        NAC subclass builds its ``self.model`` non-NAC, so this covers
        both downfolders — FEAS-002). Anchors whose report fell back inject
        the report's raw eigenvectors (= legacy behavior for that anchor).
        """
        import warnings

        from .symmetry_seeds import get_symmetry_anchor_wfn

        params = self.params
        if not params.anchors:
            # derive the mapping once and WRITE IT BACK: the set_params
            # re-run below re-reads params.anchors, and leaving it None
            # would give projected no projectors and scdmk auto-selection
            # from raw eigenvectors (review round 1)
            params.anchors = {
                tuple(np.asarray(params.anchor_kpt, dtype=float)): tuple(
                    params.anchor_ibands
                )
            }
        anchors = params.anchors
        wfn_anchor = {}
        for q, bands in anchors.items():
            q = tuple(float(x) for x in np.asarray(q, dtype=float))
            report = get_symmetry_anchor_wfn(
                self.model.phonon,
                np.asarray(q),
                tuple(bands),
                opd=params.symmetry_seed_opd,
                opd_index=params.symmetry_seed_opd_index,
            )
            for reason in report.warnings:
                warnings.warn(
                    f"symmetry_seed at q={list(q)}: {reason}; "
                    "injecting raw anchor eigenvectors",
                    stacklevel=2,
                )
            wfn_anchor[q] = report.psi
            print(f"symmetry seed at q={list(q)}:")
            for rec in report.per_band:
                print(
                    f"  band {rec.band}: eigenspace {rec.eigenspace_index} "
                    f"(dim {rec.eigenspace_dim}), family {rec.family_index} "
                    f"-> {rec.sg_symbol} (#{rec.sg_number}), "
                    f"direction {rec.direction_summary}, "
                    f"frequency {rec.frequency:.4f} cm^-1, "
                    f"irrep {rec.irrep_chars}"
                )
        self.builder.wfn_anchor = wfn_anchor

    # def downfold(
    #    self,
    #    post_func=None,
    #    output_path="./",
    #    write_hr_nc="LWF.nc",
    #    write_hr_txt="LWF.txt",
    # ):
    #    # self.params.update(params)
    #    self.atoms = self.model.atoms
    #    self.lwf = self.builder.get_wannier(Rlist=self.Rlist, Rdeg=self.Rdeg)
    #    if post_func is not None:
    #        post_func(self.lwf)
    #    if not os.path.exists(output_path):
    #        os.makedirs(output_path)
    #    try:
    #        self.save_info(output_path=output_path)
    #    except Exception:
    #        pass
    #    if write_hr_txt is not None:
    #        self.lwf.save_txt(os.path.join(output_path, write_hr_txt))
    #    if write_hr_nc is not None:
    #        self.lwf.write_nc(os.path.join(output_path, write_hr_nc))
    #    return self.lwf

    def downfold(self, output_path="./", write_hr_nc="LWF.nc", write_hr_txt="LWF.txt"):
        self._prepare_data()
        self.atoms = self.model.atoms
        self.builder.prepare()
        # compute the Amn matrix from phonons without NAC
        self.builder.get_Amn()
        # compute the Wannier functions and the Hamiltonian in k-space without NAC
        # wannk: (nkpt, nbasis, nwann)
        wannk, Hwannk, _ = self.builder.get_wannk_and_Hk()
        HwannR = k_to_R(self.kpts, self.Rlist, Hwannk, kweights=self.kweights)

        wannR = k_to_R(self.kpts, self.Rlist, wannk, kweights=self.kweights)

        wann_centers = get_wannier_centers(
            wannR, self.Rlist, self.atoms.get_scaled_positions(), Rdeg=self.Rdeg
        )
        print("wannier_centers: ")
        for i in range(self.nwann):
            print(f"{i}: {wann_centers[i]=}")

        if getattr(self.params, "use_ws_distance", False) and _ws_mesh_ok(self):
            ws, self.Rlist, self.Rdeg = _ws_materialize(
                self, {"HwannR": HwannR, "wannR": wannR}, wann_centers)
            HwannR, wannR = ws["HwannR"], ws["wannR"]

        # save the lwf model into a NACLWF object
        self.lwf = LWF(
            factor=self.factor,
            Rlist=self.Rlist,
            Rdeg=self.Rdeg,
            wannR=wannR,
            HR_total=HwannR,
            kpts=self.kpts,
            kweights=self.kweights,
            wann_centers=wann_centers,
            atoms=self.atoms,
        )

        # story-007/ADR-005: attach MLWF Mmn-form spread diagnostics
        # (ADR-004); k_to_R is bypassed here so the attachment is
        # explicit. The optimized Mmn-form centres rbar overwrite
        # wann_centers; the R-space centres stay as a diagnostic.
        spreads = getattr(self.builder, "spreads", None)
        if spreads is not None:
            try:
                self.lwf.wann_centers_rspace = wann_centers
                self.lwf.spreads = dict(spreads)
                self.lwf.wann_centers = spreads["rbar"]
            except AttributeError:
                pass  # frozen result class

        # story-031/FR-007: selection diagnostics on the result object
        selection = getattr(self.builder, "selection", None)
        if selection is not None:
            try:
                self.lwf.selection = selection
            except AttributeError:
                pass

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except Exception:
            pass
        if write_hr_txt is not None:
            self.lwf.save_txt(os.path.join(output_path, write_hr_txt))
        if write_hr_nc is not None:
            self.lwf.write_to_netcdf(os.path.join(output_path, write_hr_nc))
        return self.lwf


class NACPhonopyDownfolder(PhonopyDownfolder):
    def __init__(
        self, phonon=None, phonon_NAC=None, mode="dm", params=None, *argv, **kwargs
    ):
        """
        Parameters:
        """
        try:
            import phonopy
        except ImportError:
            raise ImportError("phonopy is needed. Do you have phonopy installed?")
        if phonon is None:
            phonon = phonopy.load(*argv, **kwargs, is_nac=False)
        super().__init__(
            phonon=phonon, mode=mode, params=params, is_nac=False, *argv, **kwargs
        )

        self.model.get_nac_params()
        # self.model_NAC = self.model

        if phonon_NAC is None:
            phonon_NAC = phonopy.load(*argv, **kwargs, is_nac=True)
        self.model_NAC = PhonopyWrapper(phonon_NAC, mode=mode, is_nac=True)

        self.born, self.dielectric, self.factor = self.model_NAC.get_nac_params()

        self.is_nac = True
        self.set_nac_params(
            # self.model_NAC.born, self.model_NAC.dielectric, self.model_NAC.factor
            self.model_NAC.born,
            self.model_NAC.dielectric,
            self.model_NAC.factor,
        )

    def get_Hks_with_nac(self, q):
        """
        get the dynmaical matrix at q with NAC.
        params:
            q: q-vector
        return:
            Htotal, Hshort, Hlong, eigenvalues, eigenvectors
        """
        evals, evecs, Hk, Hshort, Hlong = self.model_NAC.solve(q)
        return evals, evecs, Hk, Hshort, Hlong

    def set_nac_params(self, born, dielectic, factor):
        """set  Hamiltonians including splited Hks, Hshorts and Hlongs."""
        self.born = born
        self.dielectic = dielectic
        self.factor = factor

    def downfold(
        self,
        post_func=None,
        output_path="./",
        write_hr_nc="LWF.nc",
        write_hr_txt="LWF.txt",
        **params,
    ):
        self._prepare_data()
        self.atoms = self.model.atoms
        self.builder.prepare()
        # compute the Amn matrix from phonons without NAC
        self.builder.get_Amn()
        # compute the Wannier functions and the Hamiltonian in k-space without NAC
        # wannk: (nkpt, nbasis, nwann)
        wannk, Hwannk_noNAC, _ = self.builder.get_wannk_and_Hk()
        HwannR_noNAC = k_to_R(self.kpts, self.Rlist, Hwannk_noNAC, kweights=self.kweights)

        wannR = k_to_R(self.kpts, self.Rlist, wannk, kweights=self.kweights)
        # prepare the H and the eigens for all k-points.
        evals_nac, evecs_nac, Hk_tot, Hk_short, Hk_long = self.model_NAC.solve_all(
            self.kpts, output_H=True
        )

        # compute the short range Hamiltonian in Wannier space
        Hwannk_short = self.get_Hwannk_short(wannk, Hk_short, evecs_nac)
        HwannR_short = self.get_HwannR_short(
            Hwannk_short, self.kpts, self.Rlist, kweights=self.kweights, Rdeg=self.Rdeg
        )

        Hwannk_total = self.get_Hwannk_short(wannk, Hk_tot, evecs_nac)
        HwannR_total = self.get_HwannR_short(
            Hwannk_total, self.kpts, self.Rlist, kweights=self.kweights, Rdeg=self.Rdeg
        )

        wann_centers = get_wannier_centers(
            wannR, self.Rlist, self.atoms.get_scaled_positions(), Rdeg=self.Rdeg
        )
        # wann_centers *= 0.0
        print("wannier_centers: ")
        for i in range(self.nwann):
            print(f"{i}: {wann_centers[i]=}")

        if getattr(self.params, "use_ws_distance", False) and _ws_mesh_ok(self):
            ws, self.Rlist, self.Rdeg = _ws_materialize(
                self,
                {"HR_noNAC": HwannR_noNAC, "wannR": wannR,
                 "HR_short": HwannR_short, "HR_total": HwannR_total},
                wann_centers)
            HwannR_noNAC = ws["HR_noNAC"]
            wannR = ws["wannR"]
            HwannR_short = ws["HR_short"]
            HwannR_total = ws["HR_total"]

        # save the lwf model into a NACLWF object
        self.lwf = NACLWF(
            born=self.born,
            dielectric=self.dielectic,
            factor=self.factor,
            Rlist=self.Rlist,
            Rdeg=self.Rdeg,
            wannR=wannR,
            HR_noNAC=HwannR_noNAC,
            HR_short=HwannR_short,
            HR_total=HwannR_total,
            NAC_phonon=self.model_NAC,
            kpts=self.kpts,
            kweights=self.kweights,
            wann_centers=wann_centers,
            atoms=self.atoms,
        )

        # story-007/031: attach MLWF diagnostics (Mmn-form spreads,
        # ADR-004/005; selection record FR-007); both lwf-assembly seams
        # above are bypassed here
        spreads = getattr(self.builder, "spreads", None)
        if spreads is not None:
            try:
                self.lwf.wann_centers_rspace = wann_centers
                self.lwf.spreads = dict(spreads)
                self.lwf.wann_centers = spreads["rbar"]
            except AttributeError:
                pass  # frozen result class
        selection = getattr(self.builder, "selection", None)
        if selection is not None:
            try:
                self.lwf.selection = selection
            except AttributeError:
                pass

        # if post_func is not None:
        #    post_func(self.ewf)
        # if not os.path.exists(output_path):
        #    os.makedirs(output_path)
        # try:
        #    self.save_info(output_path=output_path)
        # except:
        #    pass
        # if write_hr_txt is not None:
        #    self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        # if write_hr_nc is not None:
        #    # self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        #    self.ewf.write_nc(os.path.join(output_path, write_hr_nc), atoms=self.atoms)
        # return self.ewf
        return self.lwf


    def get_Hwannk_short(self, wannk=None, Hk_short=None, evecs=None):
        """
        compute theh Hk_short in Wannier space
        params:
            wannk: Wannier functions in k-space, (nkpt, nbasis, nwann)
            Hk_short: short range Hamiltonian in k-space
        """
        Hk_wann_short = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)
        for ik in range(self.nkpt):
            Hk_wann_short[ik] = wannk[ik].conj().T @ Hk_short[ik] @ wannk[ik]
        return Hk_wann_short

    def get_HwannR_short(
        self, Hk_wann_short=None, kpts=None, Rlist=None, kweights=None, Rdeg=None
    ):
        """
        compute the HR_short in Wannier space
        """
        if Rdeg is None:
            Rdeg = np.ones(len(Rlist))
        if Hk_wann_short is None:
            Hk_wann_short = self.get_Hwannk_short()
        HwannR_short = k_to_R(kpts, Rlist, Hk_wann_short, kweights=kweights)
        return HwannR_short

    def get_wannk_interpolated(self, qpt):
        """
        Interpolate Wannier functions from real space to k-space.
        """
        wannk = R_to_onek(qpt, self.Rlist, self.lwf.wannR, self.Rdeg)
        return wannk

    def get_wannier_nac(self, Rlist=None):
        """
        Calculate Wannier functions but using non-analytic correction.
        """
        self.prepare()
        self.get_Amn()
        self.get_wannk_and_Hk_nac()
        if Rlist is not None:
            lwf = self.k_to_R(Rlist=Rlist)
            # lwf.atoms = copy.deepcopy(self.atoms)
        lwf.set_born_from_full(self.born, self.dielectic, self.factor)
        return lwf


def get_wannier_centers(wannR, Rlist, positions, Rdeg):
    # nR = len(Rlist)
    nwann = wannR.shape[2]
    wann_centers = np.zeros((nwann, 3), dtype=float)
    # natom = len(positions)
    p = np.kron(positions, np.ones((3, 1)))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        # wann_centers += (c.conj() * c).real @ positions + R[None, :]
        wann_centers += (
            np.einsum("ij, ik-> jk", (c.conj() * c).real, p + R[None, :]) * Rdeg[iR]
        )
    return wann_centers


def get_wannier_masses(masses, wannR, Rlist, Rdeg):
    """
    Get the wannier masses from the atomic mases and the Wannier functions.
    """
    nR = len(Rlist)
    nwann = wannR.shape[2]
    nR, nbasis, nwann = wannR.shape
    wann_masses = np.zeros(nwann, dtype=float)
    # masses3 = np.kron(masses, np.ones(3))
    for iR, R in enumerate(Rlist):
        c = wannR[iR, :, :]
        wann_masses += np.einsum("ij, i-> j", (c.conj() * c).real, masses) * Rdeg[iR]
    return wann_masses
