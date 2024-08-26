"""
Read and write of xsf file.
"""

import numpy as np
from ase.units import Hartree


def transpose_last3dims(arr):
    """
    Transpose the last three dimensions of arr: (...,x,y,z) --> (...,z,y,x).
    """
    axes = np.arange(arr.ndim)
    axes[-3:] = axes[::-1][:3]

    view = np.transpose(arr, axes=axes)
    return np.ascontiguousarray(view)


def xsf_write_data(fname, data, cell, origin, ngrids=1):
    fdata = transpose_last3dims(data)
    # fdata=data
    fgrid = fdata.shape[-3:]
    # cell = structure.lattice_vectors(space="r")
    # origin = np.zeros(3)
    with open(fname, "w") as myfile:
        fwrite = myfile.write
        fwrite("BEGIN_BLOCK_DATAGRID_3D\n")
        fwrite(" data\n")

        fwrite(" BEGIN_DATAGRID_3Dgrid#" + str(1) + "\n")
        fwrite("%d %d %d\n" % fdata.shape[-3:])

        fwrite("%f %f %f\n" % tuple(origin))
        for i in range(3):
            fwrite("%f %f %f\n" % tuple(cell[i]))

        for z in range(fgrid[0]):
            for y in range(fgrid[1]):
                slice_x = fdata[z, y]
                fwrite(" ".join(["%f" % d for d in slice_x]))
                fwrite("\n")
            fwrite("\n")

        fwrite(" END_DATAGRID_3D\n")
        fwrite("END_BLOCK_DATAGRID_3D\n")


def write_xsf(fileobj, images, write_force=False, vectors=None, data=None):
    if isinstance(fileobj, str):
        fileobj = open(fileobj, "w")

    if hasattr(images, "get_positions"):
        images = [images]

    is_anim = len(images) > 1

    if is_anim:
        fileobj.write("ANIMSTEPS %d\n" % len(images))

    numbers = images[0].get_atomic_numbers()

    pbc = images[0].get_pbc()
    npbc = sum(pbc)
    if pbc[2]:
        fileobj.write("CRYSTAL\n")
        assert npbc == 3
    elif pbc[1]:
        fileobj.write("SLAB\n")
        assert npbc == 2
    elif pbc[0]:
        fileobj.write("POLYMER\n")
        assert npbc == 1
    else:
        # (Header written as part of image loop)
        assert npbc == 0

    cell_variable = False
    for image in images[1:]:
        if np.abs(images[0].cell - image.cell).max() > 1e-14:
            cell_variable = True
            break

    for n, atoms in enumerate(images):
        anim_token = " %d" % (n + 1) if is_anim else ""
        if pbc.any():
            write_cell = n == 0 or cell_variable
            if write_cell:
                if cell_variable:
                    fileobj.write("PRIMVEC%s\n" % anim_token)
                else:
                    fileobj.write("PRIMVEC\n")
                cell = atoms.get_cell()
                for i in range(3):
                    fileobj.write(" %.14f %.14f %.14f\n" % tuple(cell[i]))

            fileobj.write("PRIMCOORD%s\n" % anim_token)
        else:
            fileobj.write("ATOMS%s\n" % anim_token)

        # Get the forces if it's not too expensive:
        forces = None
        if write_force:
            calc = atoms.calc
            if calc is not None and (
                hasattr(calc, "calculation_required")
                and not calc.calculation_required(atoms, ["forces"])
            ):
                forces = atoms.get_forces() / Hartree
        pos = atoms.get_positions()

        if pbc.any():
            fileobj.write(" %d 1\n" % len(pos))
        for a in range(len(pos)):
            fileobj.write(" %2d" % numbers[a])
            fileobj.write(" %20.14f %20.14f %20.14f" % tuple(pos[a]))
            if forces is None:
                fileobj.write("\n")
            else:
                fileobj.write(" %20.14f %20.14f %20.14f\n" % tuple(forces[a]))

    if data is None:
        return

    fileobj.write("BEGIN_BLOCK_DATAGRID_3D\n")
    fileobj.write(" data\n")
    fileobj.write(" BEGIN_DATAGRID_3Dgrid#1\n")

    data = np.asarray(data)
    if data.dtype == complex:
        data = np.abs(data)

    shape = data.shape
    fileobj.write("  %d %d %d\n" % shape)

    cell = images[-1].get_cell()
    origin = np.zeros(3)
    for i in range(3):
        if not pbc[i]:
            origin += cell[i] / shape[i]
    fileobj.write("  %f %f %f\n" % tuple(origin))

    for i in range(3):
        # XXXX is this not just supposed to be the cell?
        # What's with the strange division?
        # This disagrees with the output of Octopus.  Investigate
        fileobj.write("  %f %f %f\n" % tuple(cell[i] * (shape[i] + 1) / shape[i]))

    for k in range(shape[2]):
        for j in range(shape[1]):
            fileobj.write("   ")
            fileobj.write(" ".join(["%f" % d for d in data[:, j, k]]))
            fileobj.write("\n")
        fileobj.write("\n")

    fileobj.write(" END_DATAGRID_3D\n")
    fileobj.write("END_BLOCK_DATAGRID_3D\n")
