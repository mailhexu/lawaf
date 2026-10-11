# SPDX-License-Identifier: BSD-2-Clause
"""Lazy public API (PEP 562).

Importing :mod:`lawaf` — and in particular the
:mod:`lawaf.interfaces.abinit` reader package — must not pull heavy
dependencies (wannier90 wrappers, sisl, ctypes) onto the import graph
(story 060, AC-060-07).  All previous eager public names are preserved and
imported on first attribute access.
"""

_LAZY = {
    "W90Downfolder": "lawaf.interfaces",
    "SiestaDownfolder": "lawaf.interfaces",
    "PhonopyDownfolder": "lawaf.interfaces",
    "NACPhonopyDownfolder": "lawaf.interfaces",
    "get_symmetry_anchor_wfn": "lawaf.interfaces.phonopy.symmetry_seeds",
    "list_opd_families": "lawaf.interfaces.phonopy.symmetry_seeds",
    "LWF": "lawaf.lwf.lwf",
    "covariant_mmn_links": "lawaf.wannierization.covariant_mmn",
    "mmn_form_spread": "lawaf.wannierization.covariant_mmn",
    "phonon_mmn": "lawaf.wannierization.covariant_mmn",
    "optimize_kdependent_gauge": "lawaf.wannierization.kdependent_gauge",
    "apply_gauge_transform": "lawaf.wannierization.nonorthogonal_gauge",
    "exact_smetric_spread": "lawaf.wannierization.nonorthogonal_gauge",
    "optimize_nonorthogonal_gauge": "lawaf.wannierization.nonorthogonal_gauge",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
