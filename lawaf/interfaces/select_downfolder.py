from .phonopy.phonon_downfolder import NACPhonopyDownfolder, PhonopyDownfolder
from .siesta import SiestaDownfolder
from .wannier90 import W90Downfolder


def select_downfolder(name):
    if name == "phonopy":
        return PhonopyDownfolder
    elif name == "nac-phonopy":
        return NACPhonopyDownfolder
    elif name == "siesta":
        return SiestaDownfolder
    elif name == "wannier90":
        return W90Downfolder
    else:
        raise ValueError("Unknown downfolder: %s" % name)
