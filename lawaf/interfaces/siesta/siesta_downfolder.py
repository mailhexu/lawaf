import os
import json
import sisl
from lawaf.interfaces.downfolder import Lawaf
from lawaf.wrapper.sislwrapper import SislHSWrapper, SislWFSXWrapper

__all__ = ["SiestaDownfolder"]


class SislDownfolder(Lawaf):
    def __init__(
        self,
        folder=None,
        fdf_file=None,
        mode="HS",
        ispin=None,
        H=None,
        spin=None,
        recover_fermi=False,
        format="dense",
        nbands=10,
        atoms=None,
        wfsx_file="siesta.selected.WFSX",
    ):
        """
        Parameters:
        ========================================
        folder: folder of siesta calculation
        fdf_file: siesta input filename
        mode: switch between different modes of obtaining wavefunctions and
              eigenvalues. Supported options are:
              - 'HS' calculate from siesta hamiltonian (default)
              - 'WFSX' read from siesta wavefunction file

        ispin : index of spin channel to be considered. Only takes effect for
                collinear spin calculations (UP: 0, DOWN: 1). (default: None)
        H : Hamiltonian object, only takes effect if 'mode' is set to 'HS'
            (default: None)
        format: matrix format used internally, can be 'dense' or 'sparse',
                only takes effect if 'mode' is 'WFSX' (default: 'dense')
        nbands: number of eigenvalues calculated during diagonalization, only
                relevant if format='sparse', only takes effec if 'mode' is
                'WFSX' (default:10)
        wfsx_file: name of WFSX file to read, only takes effect if 'mode'
                   is set to 'WFSX' (default: 'siesta.selected.WFSX')
        """
        try:
            import sisl
        except ImportError:
            raise ImportError("sisl is needed. Do you have sisl installed?")
        self.shift_fermi = None

        if mode == "HS":
            fdf = sisl.get_sile(os.path.join(folder, fdf_file))
            fdf.read()
            H = fdf.read_hamiltonian()
            try:
                self.efermi = fdf.read_fermi_level().data[0]
            except:
                self.efermi = fdf.read_fermi_level()
            if recover_fermi:
                self.shift_fermi = self.efermi
            self.model = SislHSWrapper(
                H,
                shift_fermi=self.shift_fermi,
                ispin=ispin,
                format=format,
                nbands=nbands,
            )
        elif mode == "WFSX":
            wfsx_sile = sisl.get_sile(os.path.join(folder, wfsx_file))
            fdf = sisl.get_sile(os.path.join(folder, fdf_file))
            geom = fdf.read_geometry()
            spin = sisl.Spin(fdf.get("Spin"))
            try:
                self.efermi = fdf.read_fermi_level().data[0]
            except:
                self.efermi = fdf.read_fermi_level()

            # Eigen values in WFSX sile are not shifted by default
            # therefore we invert the behavior of recover_fermi/shift_fermi
            if not recover_fermi:
                self.shift_fermi = -self.efermi
            self.model = SislWFSXWrapper(
                geom,
                wfsx_sile=wfsx_sile,
                spin=spin,
                ispin=ispin,
                shift_fermi=self.shift_fermi,
            )
        else:
            raise ValueError(
                f"{self.__class__.__name__} does not support mode "
                f"{mode}. Supported options are: 'HS', 'WFSX'."
            )
        # TODO: add atoms to sisl wrapper
        # self.atoms = self.model.atoms
        try:
            positions = self.model.positions
        except Exception:
            positions = None
        self.model_info = {
            "orb_names": tuple(self.model.orbs),
            "positions": positions.tolist(),
        }
        self.params = {}

    def save_info(self, output_path="./", fname="Downfold.json"):
        cols = self.builder.cols
        self.orbs = [self.model.orbs[i] for i in cols]
        atoms = self.model.atoms
        results = {}
        results["model_info"] = self.model_info
        results["params"] = self.params
        results["results"] = {
            "selected_columns": cols.tolist(),
            "orb_names": tuple(self.orbs),
            "Efermi": self.efermi,
            "chemical_symbols": atoms.get_chemical_symbols(),
            "atom_xred": atoms.get_scaled_positions().tolist(),
            "cell": atoms.get_cell().tolist(),
        }
        results.update(self.params)
        with open(os.path.join(output_path, fname), "w") as myfile:
            json.dump(results, myfile, sort_keys=True, indent=4)

    def wannier_on_grid(self, i, k=None, grid_prec=0.2, grid=None, geom=None):
        """
        Projects the wannier function on a grid
        """

        # all_coeffs = DataArray(self.ewf.wannR, dims=('k', 'orb', 'wannier'))
        wannR = self.ewf.wannR

        # Use the geometry of the hamiltonian if the user didn't provide one (usual case)
        if geom is None:
            geom = self.model.ham.geom

        # Create a grid if the user didn't provide one
        if grid is None:
            grid = sisl.Grid(grid_prec, geometry=geom, dtype=complex)

        # Get the coefficients of that we want
        # coeffs = all_coeffs.sel(wannier=i)
        coeffi = wannR[:, :, i]
        # if k is None:
        #    coeffs = coeffs.mean('k')
        # else:
        #    coeffs = coeffs.sel(k=k)

        # Project the orbitals with the coefficients on the grid
        wavefunction(coeffs, grid, geometry=geom)

        return grid


SiestaDownfolder = SislDownfolder
