import numpy as np
from ase.io import read
import os
from collections import OrderedDict

# import lawaf.wrapper.pythtb as pythtb


def _red_to_cart(tmp, red):
    "Convert reduced to cartesian vectors."
    (a1, a2, a3) = tmp
    # cartesian coordinates
    cart = np.zeros_like(red, dtype=float)
    for i in range(0, len(cart)):
        cart[i, :] = a1 * red[i][0] + a2 * red[i][1] + a3 * red[i][2]
    return cart


def read_basis(fname):
    """
    return basis names from file (often named as basis.txt). Return a dict. key: basis name. value: basis index, from 0
    """
    bdict = OrderedDict()
    if fname.endswith(".win"):
        with open(fname) as myfile:
            inside = False
            iline = 0
            for line in myfile.readlines():
                if line.strip().startswith("end projections"):
                    inside = False
                if inside:
                    a = line.strip().split("#")
                    assert len(a) == 2, "The format should be .... # label_of_basis"
                    bdict[a[-1].strip()] = iline
                    iline += 1
                if line.strip().startswith("begin projections"):
                    inside = True
    else:
        with open(fname) as myfile:
            for iline, line in enumerate(myfile.readlines()):
                a = line.strip().split()
                if len(a) != 0:
                    bdict[a[0]] = iline
    return bdict


def wannier_to_model(
    path_up,
    path_down,
    inmodel=None,
    prefix_up="wannier90.up",
    prefix_down="wannier90.dn",
    atoms_file="POSCAR",
    atoms=None,
    zero_energy=0.0,
    min_hopping_norm=None,
    max_distance=None,
    ignorable_imaginary_part=None,
):
    if atoms is None:
        atoms = read(os.path.join(path_up, atoms_file))

    w = w90_two_spin(path_up, prefix_up, path_down, prefix_down, atoms=atoms)
    bset = w.bset
    bset.set_atoms(atoms)
    inmodel = atoms_model(atoms, basis_dict=None, basis_set=bset, nspin=2)
    model = w.model(
        inmodel=inmodel,
        zero_energy=zero_energy,
        min_hopping_norm=min_hopping_norm,
        max_distance=max_distance,
        ignorable_imaginary_part=ignorable_imaginary_part,
    )
    return model


# class w90(pythtb.w90):
#     def __init__(self, path, prefix):
#         super(w90, self).__init__(path, prefix)
#         win_fname = os.path.join(path, '%s.win' % prefix)
#         try:
#             self.read_basis(win_fname)
#         except Exception:
#             from warnings import warn
#             warn("Cannot read basis name in wannier90 input file")

#     def model(self,
#               inmodel=None,
#               zero_energy=0.0,
#               min_hopping_norm=None,
#               max_distance=None,
#               ignorable_imaginary_part=None):
#         """

#         This function returns :class:`pythtb.tb_model` object that can
#         be used to interpolate the band structure at arbitrary
#         k-point, analyze the wavefunction character, etc.

#         The tight-binding basis orbitals in the returned object are
#         maximally localized Wannier functions as computed by
#         Wannier90.  The orbital character of these functions can be
#         inferred either from the *projections* block in the
#         *prefix*.win or from the *prefix*.nnkp file.  Please note that
#         the character of the maximally localized Wannier functions is
#         not exactly the same as that specified by the initial
#         projections.  One way to ensure that the Wannier functions are
#         as close to the initial projections as possible is to first
#         choose a good set of initial projections (for these initial
#         and final spread should not differ more than 20%) and then
#         perform another Wannier90 run setting *num_iter=0* in the
#         *prefix*.win file.

#         Number of spin components is always set to 1, even if the
#         underlying DFT calculation includes spin.  Please refer to the
#         *projections* block or the *prefix*.nnkp file to see which
#         orbitals correspond to which spin.

#         Locations of the orbitals in the returned
#         :class:`pythtb.tb_model` object are equal to the centers of
#         the Wannier functions computed by Wannier90.

#         :param zero_energy: Sets the zero of the energy in the band
#           structure.  This value is typically set to the Fermi level
#           computed by the density-functional code (or to the top of the
#           valence band).  Units are electron-volts.

#         :param min_hopping_norm: Hopping terms read from Wannier90 with
#           complex norm less than *min_hopping_norm* will not be included
#           in the returned tight-binding model.  This parameters is
#           specified in electron-volts.  By default all terms regardless
#           of their norm are included.

#         :param max_distance: Hopping terms from site *i* to site *j+R* will
#           be ignored if the distance from orbital *i* to *j+R* is larger
#           than *max_distance*.  This parameter is given in Angstroms.
#           By default all terms regardless of the distance are included.

#         :param ignorable_imaginary_part: The hopping term will be assumed to
#           be exactly real if the absolute value of the imaginary part as
#           computed by Wannier90 is less than *ignorable_imaginary_part*.
#           By default imaginary terms are not ignored.  Units are again
#           eV.

#         :returns:
#            * **tb** --  The object of type :class:`pythtb.tb_model` that can be used to
#                interpolate Wannier90 band structure to an arbitrary k-point as well
#                as to analyze the character of the wavefunctions.  Please note

#         Example usage::

#           # returns tb_model with all hopping parameters
#           my_model=silicon.model()

#           # simplified model that contains only hopping terms above 0.01 eV
#           my_model_simple=silicon.model(min_hopping_norm=0.01)
#           my_model_simple.display()

#         """

#         # make the model object
#         if inmodel is None:
#             tb = etb_model(3, 3, self.lat, self.red_cen)
#         else:
#             tb = inmodel

#         # remember that this model was computed from w90
#         tb._assume_position_operator_diagonal = False

#         # add onsite energies
#         onsite = np.zeros(self.num_wan, dtype=float)
#         for i in range(self.num_wan):
#             tmp_ham = self.ham_r[(
#                 0, 0, 0)]["h"][i, i] / float(self.ham_r[(0, 0, 0)]["deg"])
#             onsite[i] = tmp_ham.real
#             if np.abs(tmp_ham.imag) > 1.0E-9:
#                 raise Exception("Onsite terms should be real!")
#         tb.set_onsite(onsite - zero_energy)

#         # add hopping terms
#         for R in self.ham_r:
#             # avoid double counting
#             use_this_R = True
#             # avoid onsite terms
#             if R[0] == 0 and R[1] == 0 and R[2] == 0:
#                 avoid_diagonal = True
#             else:
#                 avoid_diagonal = False
#                 # avoid taking both R and -R
#                 if R[0] != 0:
#                     if R[0] < 0:
#                         use_this_R = False
#                 else:
#                     if R[1] != 0:
#                         if R[1] < 0:
#                             use_this_R = False
#                     else:
#                         if R[2] < 0:
#                             use_this_R = False
#             # get R vector
#             vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]),
#                                 [R])[0]
#             # scan through unique R
#             if use_this_R == True:
#                 for i in range(self.num_wan):
#                     vec_i = self.xyz_cen[i]
#                     for j in range(self.num_wan):
#                         vec_j = self.xyz_cen[j]
#                         # get distance between orbitals
#                         dist_ijR = np.sqrt(
#                             np.dot(-vec_i + vec_j + vecR, -vec_i + vec_j +
#                                    vecR))
#                         # to prevent double counting
#                         if not (avoid_diagonal == True and j <= i):

#                             # only if distance between orbitals is small enough
#                             if max_distance is not None:
#                                 if dist_ijR > max_distance:
#                                     continue

#                             # divide the matrix element from w90 with the degeneracy
#                             tmp_ham = self.ham_r[R]["h"][i, j] / float(
#                                 self.ham_r[R]["deg"])

#                             # only if big enough matrix element
#                             if min_hopping_norm is not None:
#                                 if np.abs(tmp_ham) < min_hopping_norm:
#                                     continue

#                             # remove imaginary part if needed
#                             if ignorable_imaginary_part is not None:
#                                 if np.abs(tmp_ham.imag
#                                           ) < ignorable_imaginary_part:
#                                     tmp_ham = tmp_ham.real + 0.0j

#                             # set the hopping term
#                             tb.set_hop(tmp_ham, i, j, list(R))

#         return tb

#     def read_basis(self, fname):
#         self.basis = read_basis(fname)
#         return self.basis

#     def get_basis_set(self, atoms, spin=0):
#         bset = BasisSet()
#         sdict = symbol_number(atoms.get_chemical_symbols())
#         for b in self.basis.keys():
#             sn, label, _, _ = b.split('|')
#             site = sdict[sn]
#             bset.append(Basis(site=site, label=label, spin=spin, index=0))
#         return bset

#     def get_hamiltonian_dict(self):
#         return self.ham_r

#     def get_positions(self):
#         return self.xyz_cen

#     def get_scaled_positions(self):
#         return self.red_cen

#     def get_lat(self):
#         return self.lat

# class w90_two_spin(w90):
#     def __init__(self, path_up, prefix_up, path_down, prefix_down, atoms=None):
#         w90_up = w90(path_up, prefix_up)
#         nwan_up = w90_up.num_wan
#         hamr_up = w90_up.ham_r
#         xyz_cen_up = w90_up.xyz_cen
#         red_cen_up = w90_up.red_cen
#         lat_up = w90_up.lat

#         w90_dn = w90(path_down, prefix_down)
#         nwan_dn = w90_dn.num_wan
#         hamr_dn = w90_dn.ham_r
#         xyz_cen_dn = w90_dn.xyz_cen
#         red_cen_dn = w90_dn.red_cen
#         lat_dn = w90_dn.lat

#         assert nwan_up == nwan_dn, "number of wannier function for spin up and down not equal"
#         self.num_wan = 2 * nwan_dn
#         self.ham_r = {}
#         for R in hamr_up:
#             if R not in self.ham_r:
#                 self.ham_r[R] = {}
#                 self.ham_r[R]['h'] = np.zeros(
#                     (
#                         self.num_wan,
#                         self.num_wan, ), dtype='complex')
#             self.ham_r[R]['h'][::2, ::2] = hamr_up[R]['h']
#             self.ham_r[R]['deg'] = hamr_up[R]['deg']

#         for R in hamr_dn:
#             if R not in self.ham_r:
#                 self.ham_r[R]['h'] = np.zeros(
#                     self.num_wan, self.num_wan, dtype='complex')
#             self.ham_r[R]['h'][1::2, 1::2] = hamr_dn[R]['h']
#             self.ham_r[R]['deg'] = hamr_dn[R]['deg']

#         self.xyz_cen = np.zeros((self.num_wan, 3), dtype=float)
#         self.xyz_cen[::2, :] = xyz_cen_up
#         self.xyz_cen[1::2, :] = xyz_cen_dn

#         self.red_cen = np.zeros((self.num_wan, 3), dtype=float)
#         self.red_cen[::2, :] = red_cen_up
#         self.red_cen[1::2, :] = red_cen_dn

#         assert np.allclose(
#             np.array(lat_dn),
#             np.array(lat_dn)), "spin up and down lattice not equal"
#         self.lat = lat_up

#         if atoms is not None:
#             bset_up=w90_up.get_basis_set(atoms, spin=0)
#             bset_dn=w90_dn.get_basis_set(atoms, spin=1)
#             self.bset=BasisSet()
#             assert len(bset_up)==len(bset_dn), "length of spin up and down basis set not equal"
#             for up, dn in zip(bset_up, bset_dn):
#                 self.bset.append(up)
#                 self.bset.append(dn)
#             self.bset.set_atoms(atoms)


def test():
    path = os.path.expanduser("~/project/electy/test/wannier90/dat")
    b = read_basis(os.path.join(path, "wannier90.up.win"))
    atoms = read(os.path.join(path, "POSCAR"))
    w = w90_two_spin(
        path_up=path,
        prefix_up="wannier90.up",
        path_down=path,
        prefix_down="wannier90.up",
        atoms=atoms,
    )
    bset = w.bset
    bset.set_atoms(atoms)
    model = atoms_model(atoms, basis_dict=None, basis_set=bset, nspin=2)
    # w.get_basis_set(atoms=atoms)
    model = w.model(min_hopping_norm=0.1, inmodel=model)
    model.set_Hubbard_U(Utype="SUN", Hubbard_dict={"Ni": {"U": 1, "J": 0}})
    model.set(nel=20)
    model.set_kmesh([4, 4, 4])
    model.save("nickelate.pickle")
    model.scf_solve()


# test()
