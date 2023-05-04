import re
import math
import numpy as np
from ase.io import read
from ase.atoms import Atoms
from collections import defaultdict


def parse_xyz(fname):
    atoms = read(fname)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    wann_pos = []
    atoms_pos = []
    atoms_symbols = []
    for s, x in zip(symbols, pos):
        if s == 'X':
            wann_pos.append(x)
        else:
            atoms_symbols.append(s)
            atoms_pos.append(x)
    return np.array(wann_pos), atoms_symbols, np.array(atoms_pos)


def remove_comment(line: str):
    x = re.split('!|#', line)[0]
    return x


def parse_win(fname):
    """
    read atomic structure from .win file
    """
    cell = []
    xcart = []
    xred = []
    symbols = []
    with open(fname) as myfile:
        for line in myfile:
            line = remove_comment(line)
            if re.match('begin\s+unit_cell_cart', line):
                while (not re.match('end\s+unit_cell_cart', line)):
                    line = remove_comment(next(myfile))
                    ts = line.split()
                    if len(ts) == 3:
                        cell.append([float(x) for x in ts])
            elif re.match('begin\s+atoms_cart', line):
                while (not re.match('end\s+atoms_cart', line)):
                    line = remove_comment(next(myfile))
                    ts = line.split()
                    if len(ts) == 4:
                        symbols.append(ts[0])
                        xcart.append([float(x) for x in ts[1:]])
            elif re.match('begin\s+atoms_frac', line):
                while (not re.match('end\s+atoms_frac', line)):
                    line = remove_comment(next(myfile))
                    ts = line.split()
                    if len(ts) == 4:
                        symbols.append(ts[0])
                        xred.append([float(x) for x in ts[1:]])

    cell = np.array(cell)
    xcart = np.array(xcart)
    if xcart != []:
        atoms = Atoms(symbols=symbols, positions=xcart, cell=cell, pbc=True)
    elif xred != []:
        atoms = Atoms(symbols=symbols,
                      scaled_positions=xred,
                      cell=cell,
                      pbc=True)
    else:
        raise IOError("Failed to read atomic structure from %s" % fname)
    return atoms


def parse_ham(fname='wannier90_hr.dat', cutoff=None):
    """
    wannier90 hr file phaser.

    :param cutoff: the energy cutoff.  None | number | list (of Emin, Emax).
    """
    with open(fname, 'r') as myfile:
        lines = myfile.readlines()
    n_wann = int(lines[1].strip())
    n_R = int(lines[2].strip())

    # The lines of degeneracy of each R point. 15 per line.
    nline = int(math.ceil(n_R / 15.0))
    dlist = []
    for i in range(3, 3 + nline):
        d = map(float, lines[i].strip().split())
        dlist += d
    H_mnR = defaultdict(lambda: np.zeros((n_wann, n_wann), dtype=complex))
    for i in range(3 + nline, 3 + nline + n_wann**2 * n_R):
        t = lines[i].strip().split()
        R = tuple(map(int, t[:3]))
        m, n = map(int, t[3:5])
        m = m - 1
        n = n - 1
        H_real, H_imag = map(float, t[5:])
        val = H_real + 1j * H_imag
        if (m == n and np.linalg.norm(R) < 0.001):
            H_mnR[R][m, n] = val / 2.0
        elif cutoff is not None:
            if abs(val) > cutoff:
                H_mnR[R][m, n] = val
        else:
            H_mnR[R][m, n] = val / 2.0
    return n_wann, H_mnR


def auto_assign_wannier_to_atom(positions, atoms, max_distance=0.1,
                                half=False):
    """
    assign
    half: only half of orbitals. if half, only the first half is used.
    Returns:
    ind_atoms: a list of same length of n(orb).
    """
    pos = np.array(positions)
    atompos = atoms.get_scaled_positions()
    ind_atoms = []
    newpos = []
    refpos = []
    for i, p in enumerate(pos):
        # distance to all atoms
        dp = p[None, :] - atompos
        # residual of d
        r = dp - np.round(dp)
        # find the min of residual
        normd = np.linalg.norm(r, axis=1)
        iatom = np.argmin(normd)
        # ref+residual
        rmin = r[iatom]
        rpos = atompos[iatom]
        ind_atoms.append(iatom)
        refpos.append(rpos)
        newpos.append(rmin + rpos)
    return ind_atoms, newpos

