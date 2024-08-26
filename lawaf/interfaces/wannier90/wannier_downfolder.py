from HamiltonIO import WannierHam
from lawaf.interfaces.downfolder import Lawaf


class W90Downfolder(Lawaf):
    def __init__(self, folder, prefix, ispin=0, params=None):
        """
        folder   # The folder containing the Wannier function files
        prefix,   # The prefix of Wannier90 outputs. e.g. wannier90_up
        """
        m = WannierHam.read_from_wannier_dir(folder, prefix)
        self.model = m
        super().__init__(m, params)
