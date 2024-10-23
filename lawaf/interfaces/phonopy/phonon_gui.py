import argparse

from nicegui import ui

from lawaf.interfaces.phonopy.phonon_downfolder import (
    NACPhonopyDownfolder,
    PhonopyDownfolder,
)
from lawaf.ui.gui import ParamsGui
from lawaf.utils.port import get_unused_port

class PhononGui(ParamsGui):
    def __init__(
        self,
        phonopy_yaml="phonopy_params.yaml",
        born_filename="BORN",
        mode="DM",
        params=None,
        nac=True,
        # nac_params={"method": "wang"},
        port=None,
        *argv,
        **kwargs,
    ):
        super().__init__(port=port)
        self.mode = mode
        self.born_filename = born_filename
        self.phonopy_yaml = phonopy_yaml
        self.reset()

    def reset(self):
        if self.born_filename is None:
            self.downfolder = PhonopyDownfolder(
                params=self.params,
                mode=self.mode,
                phonopy_yaml=self.phonopy_yaml,
            )
        else:
            self.downfolder = NACPhonopyDownfolder(
                mode=self.mode,
                params=self.params,
                nac_params={"method": "wang"},
                phonopy_yaml=self.phonopy_yaml,
                born_filename=self.born_filename,
            )



    def plot_band(self, ax, pl):
        ax.clear()
        self.downfolder.plot_full_band(ax=ax, unit_factor=15.6 * 33.6, fix_LOTO=True, evals_to_freq=True,
            ylabel="Frequency (cm$^{-1}$)")
        pl.update()

    def wannierize(self, ax, pl):
        ui.notify("Wannierize")
        ui.notify(f"self.params={self.params}")
        ax.clear()
        self.reset()
        self.downfolder.params = self.params
        self.downfolder.downfold()
        self.downfolder.plot_full_band(ax=ax, fix_LOTO=True, unit_factor=15.6 * 33.6, evals_to_freq=True,ylabel="Frequency (cm$^{-1}$)")
        self.downfolder.plot_wannier_band(ax=ax, unit_factor=15.6 * 33.6, fix_LOTO=True, evals_to_freq=True,ylabel="Frequency (cm$^{-1}$)")

        pl.update()

def run_phonopy_gui():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonopy_yaml", "-f", default="phonopy_params.yaml")
    parser.add_argument("--mode", "-m", default="DM")
    parser.add_argument("--born_filename", "-b", default=None)
    parser.add_argument("--port", "-p", type=int, default=None)
    args = parser.parse_args()
    # if port is not given, use a non-blocking port
    if args.port is None:
        args.port = get_unused_port()
    ui = PhononGui(
        phonopy_yaml=args.phonopy_yaml, mode=args.mode, born_filename=args.born_filename, port=args.port
    )
    ui.run()


if __name__ in ["__main__", "__mp_main__"]:
    run_phonopy_gui()
