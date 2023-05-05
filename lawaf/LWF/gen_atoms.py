#!/usr/bin/env python3
import pickle
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from phonopy import load
from ase.atoms import Atoms
from ase.io import write, read
from spglib import spglib
from pyDFTutils.ase_utils.geometry import force_near_0


def gen(mod, amp):
    myphonon=load(phonopy_yaml='../phonopy_params.yaml')
    for i in [0]:
        phonon=copy.copy(myphonon)
        phonon.set_modulations(dimension=np.array([[1,0,-1],[0,1,0],[0,0,2]]).T, phonon_modes=[[(.5, 0.0, .5), i,amp, 0]])
        modes, sc = phonon.get_modulations_and_supercell()
        sc=Atoms(sc.symbols, cell=sc.cell,scaled_positions=sc.get_scaled_positions(), pbc=True)
        datoms=copy.deepcopy(sc)
        positions=sc.get_positions()
        for mode in modes:
            positions+=np.real(mode)
        datoms.set_positions(positions)
        spg=spglib.get_spacegroup(datoms,symprec=0.001)
        write(f"POSCAR_{i}.vasp", datoms, vasp5=True)
        print(datoms.cell)
        print(f"{spg}")
    return datoms

def force_near_0(positions, max=0.93):
    """
    force the atom near the "1" side (>max) to be near the 0 side
    """
    new_positions = []
    for pos in positions:
        new_pos = [(x if x < max else x - 1) for x in pos]
        new_positions.append(new_pos)
    return np.array(new_positions)


def combine(alpha, cell=False):
    R=read('R.vasp')
    M1=read('M1.vasp')
    cell_R=R.get_cell()
    spos_R=force_near_0(R.get_scaled_positions())
    print(spos_R)

    cell_M1=M1.get_cell()
    spos_M1=force_near_0(M1.get_scaled_positions())

    symbols=R.get_chemical_symbols()

    if cell:
        cell=(1-alpha)*cell_R+alpha*cell_M1
    else:
        cell=cell_R

    print(spos_M1)
    
    spos=(1-alpha)*spos_R+alpha*spos_M1
    atoms=Atoms(symbols, cell=cell, scaled_positions=spos)
    spg=spglib.get_spacegroup(atoms,symprec=0.001)
    print(f"{spg}")
    return atoms
        
atoms=combine(0.00,cell=True)

#gen(0,0)
