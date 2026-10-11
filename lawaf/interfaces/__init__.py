# SPDX-License-Identifier: BSD-2-Clause
"""Lazy public API (PEP 562).

The abinit reader subpackage (:mod:`lawaf.interfaces.abinit`) must import
without the wannier90/sisl/ctypes graph (story 060, AC-060-07), so the
downfolder classes are imported on first attribute access instead of eagerly.
"""

_LAZY = {
    "NACPhonopyDownfolder": ".phonopy.phonon_downfolder",
    "PhonopyDownfolder": ".phonopy.phonon_downfolder",
    "SiestaDownfolder": ".siesta",
    "W90Downfolder": ".wannier90",
    "select_downfolder": ".select_downfolder",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
