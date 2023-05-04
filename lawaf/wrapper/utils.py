from collections import OrderedDict, defaultdict
import numpy as np


def symbol_number(symbols):
    """
    symbols can be also atoms. Thus the chemical symbols will be used.
    Fe Fe Fe O -> {Fe1:0 Fe2:1 Fe3:2 O1:3}
    """
    try:
        symbs = symbols.copy().get_chemical_symbols()
    except Exception:
        symbs = symbols
    symdict = {}
    result = OrderedDict()
    for i, sym in enumerate(symbs):
        if sym in symdict:
            symdict[sym] = symdict[sym] + 1
        else:
            symdict[sym] = 1
        result[sym + str(symdict[sym])] = i
    return result


def read_basis(fname):
    """
    return basis names from file (often named as basis.txt). Return a dict. key: basis name. value: basis index, from 0
    """
    bdict = OrderedDict()
    if fname.endswith('.win'):
        with open(fname) as myfile:
            inside = False
            iline = 0
            for line in myfile.readlines():
                if line.strip().startswith('end projections'):
                    inside = False
                if inside:
                    a = line.strip().split('#')
                    assert len(
                        a) == 2, "The format should be .... # label_of_basis"
                    bdict[a[-1].strip()] = iline
                    iline += 1
                if line.strip().startswith('begin projections'):
                    inside = True
    else:
        with open(fname) as myfile:
            for iline, line in enumerate(myfile.readlines()):
                a = line.strip().split()
                if len(a) != 0:
                    bdict[a[0]] = iline
    return bdict


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


def auto_assign_wannier_to_atom2(positions,
                                 atoms,
                                 max_distance=0.1,
                                 half=False):
    """
    assign
    half: only half of orbitals. if half, only the first half is used.
    Returns:
    ind_atoms: a list of same length of n(orb).
    """
    porbs = positions
    if half:
        norbs = len(porbs) // 2
        porbs = porbs[:norbs]
    ind_atoms = []
    patoms = atoms.get_scaled_positions()
    shifted_pos = []
    for iorb, porb in enumerate(porbs):
        distances = []
        distance_vecs = []
        for iatom, patom in enumerate(patoms):
            d = porb - patom
            rd = np.min(
                np.array([d % 1.0 % 1.0, (1.0 - d) % 1.0 % 1.0]), axis=0)
            rdn = np.linalg.norm(rd)
            distance_vecs.append(rd)
            distances.append(rdn)
        iatom = np.argmin(distances)
        ind_atoms.append(iatom)
        shifted_pos.append(distance_vecs[iatom] + patoms[iatom])
        if min(distances) > max_distance:
            print(
                "Warning: the minimal distance between wannier function No. %s is large. Check if the MLWFs are well localized."
                % iorb)
    if half:
        ind_atoms = np.vstack([ind_atoms, ind_atoms], dtype=int)
        shifted_pos = np.vstack([shifted_pos, shifted_pos], dtype=float)
    return ind_atoms, shifted_pos


def auto_assign_basis_name(positions,
                           atoms,
                           max_distance=0.1,
                           write_basis_file='assigned_basis.txt',
                           half=False):
    ind_atoms, shifted_pos = auto_assign_wannier_to_atom(
        positions=positions, atoms=atoms, max_distance=max_distance, half=half)
    basis_dict = {}
    a = defaultdict(int)
    symdict = symbol_number(atoms)
    syms = list(symdict.keys())
    for i, iatom in enumerate(ind_atoms):
        a[iatom] = a[iatom] + 1
        basis_dict['%s|orb_%d' % (syms[iatom], a[iatom])] = i + 1
    if write_basis_file is not None:
        with open(write_basis_file, 'w') as myfile:
            for key, val in basis_dict.items():
                myfile.write('%s  %d \n' % (key, val))
    return basis_dict, shifted_pos


def shift_positions(p, pref):
    return p - np.round(p - pref)


def test_shift_positions():
    a = np.array((0.0, 0.1, 1.8))
    b = np.array((0.1, -1.9, 1.3))
    print(shift_positions(a, b))


def match_pos(pos, atompos):
    """
    match the  wannier position with atom positions
    pos: positions of wannier functions
    atompos: positions of atomic positions
    Returns:
      newpos: shifted  wannier positions
      refpos: the positions of atoms which the wannier functions are assigned to.
    """
    pos = np.array(pos)
    atompos = np.array(atompos)
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
        refpos.append(rpos)
        newpos.append(rmin + rpos)
    return newpos, refpos


def match_k(k, kpts):
    """
    Find the index of a kpoint to the closest k in a list of kpoints.
    The periodic structure is considered.
    k: positions of wannier functions
    kpts: positions of atomic positions
    Returns:
    imatch: the index of matching kpt
    kshift: shifted k point which matches to the kpts list
    """
    k = np.array(k)
    kpts = np.array(kpts)
    # distance to all atoms
    dp = k[None, :] - kpts
    # residual of d
    r = dp - np.round(dp)
    # find the min of residual
    normd = np.linalg.norm(r, axis=1)
    imatch = np.argmin(normd)
    # ref+residual
    rmin = r[imatch]
    kshift = kpts[imatch] + rmin
    return imatch, kshift


def match_kq_mesh(klist, qlist):
    """
    return a table of (iq, ik): i(k+q)
    """
    nk = len(klist)
    nq = len(qlist)
    ret = np.zeros((nq, nk), dtype='int')
    for iq, q in enumerate(qlist):
        for ik, k in enumerate(klist):
            ikq, kshift = match_k(k + q, klist)
            ret[iq, ik] = ikq
    return ret



def kmesh_to_R(kmesh):
    k1, k2, k3 = kmesh
    Rlist = [(R1, R2, R3) for R1 in range(-k1 // 2 + 1, k1 // 2 + 1)
             for R2 in range(-k2 // 2 + 1, k2 // 2 + 1)
             for R3 in range(-k3 // 2 + 1, k3 // 2 + 1)]
    return Rlist

