import numpy as np
import copy
from ase.dft.kpoints import bandpath
from lawaf.utils.kpoints import autopath
import matplotlib.pyplot as plt


def fix_evals_LOTO(evals, xs, kpts, N=4):
    fixed_evals = copy.deepcopy(evals)
    for i, (x, kv) in enumerate(zip(xs, kpts)):
        if np.isclose(kv, np.array([0.0, 0.0, 0.0]), rtol=1e-5, atol=1e-3).all():
            if i == 0:
                fixed_evals[i] = (N * evals[i + 1] - evals[i + 2]) / (N - 1)
            # elif i==len(xs)-1:
            #    fixed_evals[i] = (10*evals[i-1] - evals[i-2])/9
            else:
                fixed_evals[i] = (N * evals[i - 1] - evals[i - 2]) / (N - 1)
                # fixed_evals[i] = evals[i-1]
    return fixed_evals


def plot_band(
    model,
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
    color="blue",
    alpha=0.8,
    marker="",
    label=None,
    cell=np.eye(3),
    evals_to_freq=False,
    unit_factor=1,
    ylabel="Energy (eV)",
    fix_LOTO=False,
    ax=None,
):
    if ax is None:
        _fig, ax = plt.subplots()

    # if supercell_matrix is None:
    #    supercell_matrix = np.eye(3)
    # kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
    if cell is None:
        if "atoms" in model.__dict__:
            cell = model.atoms.cell
        elif "cell" in model.__dict__:
            cell = model.cell

    knames, kpts, xs, Xs = autopath(
        knames=knames,
        kvectors=kvectors,
        npoints=npoints,
        supercell_matrix=supercell_matrix,
        cell=cell,
    )
    kpts = np.vstack(kpts)
    xs = np.hstack(xs)
    _evalues, _evecs = model.solve_all(kpts=kpts)[:2]

    if fix_LOTO:
        evalues = fix_evals_LOTO(_evalues, xs, kpts)
    else:
        evalues = copy.deepcopy(_evalues)
    if evals_to_freq:
        # evalues = np.where(evalues < 0, -np.sqrt(-evalues), np.sqrt(evalues))
        evalues = np.sign(evalues) * (np.sqrt(np.abs(evalues)))
    evalues *= unit_factor
    for i in range(evalues.shape[1]):
        if i == 0:
            ax.plot(
                xs, evalues[:, i], color=color, alpha=alpha, marker=marker, label=label
            )
        else:
            ax.plot(xs, evalues[:, i], color=color, alpha=alpha, marker=marker)

    if efermi is not None:
        ax.axhline(efermi, linestyle="--", color="gray")
    else:
        try:
            plt.axhline(model.get_fermi_level(), linestyle="--", color="gray")
        except AttributeError:
            pass
    ax.set_ylabel(ylabel)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xticks(Xs)
    ax.set_xticklabels(knames)
    if erange is not None:
        ax.set_ylim(erange)
    for X in Xs:
        ax.axvline(X, linewidth=0.6, color="gray")
    return ax
