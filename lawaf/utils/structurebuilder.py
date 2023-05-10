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
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PhonMode():
    kpt: Tuple[float]
    ind : int




class PhonStructureGenerator():
    def __init__(self, phonon, supercell_matrix):
        self.phonon=copy.deepcopy(myphonon)

    def get_distorted_structure(self, phonon_modes):





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

def test():
