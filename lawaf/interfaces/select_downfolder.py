# SPDX-License-Identifier: BSD-2-Clause
def select_downfolder(name):
    import lawaf.interfaces as interfaces

    classes = {
        "phonopy": "PhonopyDownfolder",
        "nac-phonopy": "NACPhonopyDownfolder",
        "siesta": "SiestaDownfolder",
        "wannier90": "W90Downfolder",
    }
    if name not in classes:
        raise ValueError("Unknown downfolder: %s" % name)
    return getattr(interfaces, classes[name])
