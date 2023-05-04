#!/usr/bin/env python3

"""
Tools for manipulating netcdf files
"""
import numpy as np
import netCDF4 as nc


def add_variable(fname, varname, datatype, dimension_strings, value):
    """
    add a variable to the netcdf file.
    Note the dimensions should be defined already.
    fname: filename
    varname: variable name to be added.
    datatype: i4, f8, etc
    dimension_strings: the dimensions should be already in the netcdf file.
                   TODO: implement adding new dims.
    value: the value to be added.

    Example:
      add_variable('out2.nc', 'ref_spin_orientation', 'f8', ('nspin', 'three'),
                 np.array([[0, 0, 1.0]], dtype='float'))

    """
    with nc.Dataset(fname, 'r+', data_model='NETCDF3_CLASSIC') as myfile:
        myfile.createVariable(varname, datatype, dimension_strings)
        myfile.variables[varname][:] = value


def change_varname(fname, outfname, change_dict={}):
    """
    Rename variable names.
    fname: input filename
    outfname: output filename
    change_dict: key: the name to be changed, 
                 value: the name to be changed into
                 multi vars can be changed.
    """
    dimensions = {}
    with nc.Dataset(outfname, 'w', data_model='NETCDF3_CLASSIC') as dst:
        with nc.Dataset(fname) as src:
            # copy attributes
            for name in src.ncattrs():
                if name in change_dict:
                    newname = change_dict[name]
                else:
                    newname = name
                dst.setncattr(newname, src.getncattr(name))
                # copy dimensions
            for name, dimension in src.dimensions.items():
                if name not in dimensions:
                    dst.createDimension(
                        name, (len(dimension)
                               if not dimension.isunlimited() else 0))
                    dimensions[name] = dimension
                else:
                    pass
            # copy all file data for variables that are included in the toinclude list
            for name, variable in src.variables.items():
                # copy variables
                if name in change_dict:
                    newname = change_dict[name]
                else:
                    newname = name
                x = dst.createVariable(newname, variable.datatype,
                                       variable.dimensions)
                dst.variables[newname][:] = src.variables[name][:]
                # copy atrribute of varibles
                for attr in variable.ncattrs():
                    content = src.variables[name].getncattr(attr)
                    setattr(x, attr, content)


def merge(fnames, outfname):
    """
    Merge multi netcdf files into one.
    fnames: list of nc files.
    outfname: the output filename
    NOTE: right now, there is no consistency check. And if there are duplicate variables, 
          the one comes latter in the list of files will over-write previos ones.
    TODO: add check. 
    """
    dimensions = {}
    var_names = set()
    attrs = set()
    with nc.Dataset(outfname, 'w', data_model='NETCDF3_CLASSIC') as dst:
        for fname in fnames:
            with nc.Dataset(fname) as src:
                # copy attributes
                for name in src.ncattrs():
                    if name not in attrs:
                        dst.setncattr(name, src.getncattr(name))
                        attrs.add(name)
                    # copy dimensions
                for name, dimension in src.dimensions.items():
                    if name not in dimensions:
                        dst.createDimension(
                            name, (len(dimension)
                                   if not dimension.isunlimited() else 0))
                        dimensions[name] = dimension
                    else:
                        pass
                # copy all file data for variables that are included in the toinclude list
                for name, variable in src.variables.items():
                    # copy variables
                    if name not in var_names:
                        x = dst.createVariable(name, variable.datatype,
                                               variable.dimensions)
                        dst.variables[name][:] = src.variables[name][:]
                        for attr in variable.ncattrs():
                            content = src.variables[name].getncattr(attr)
                            setattr(x, attr, content)
                        var_names.add(name)
                    else:
                        print(
                            "Warning: Duplicate variables %s found, it will be neglected! " % name)
                    # copy atrribute of varibles


def flip_one_spin(fname, ispin):
    """
    flip a spin of the last step in the spinhist.nc file .
    """
    with nc.Dataset(fname, 'r+') as dtset:
        print("Initial spin", dtset['S'][-1, ispin, :])
        dtset['S'][-1, ispin, :] = -dtset['S'][:][-1, ispin, :]
        print("After flip:", dtset['S'][:][-1, ispin, :])


# ============================
# Examples and tests


def test_change_varname():
    """
    Examample: change the varname xcart to ref_xcart in exchange.nc. Output file: exchang2.nc

    """
    change_varname('pot.nc', 'pot2.nc', change_dict={'xcart': 'ref_xcart',
                                                     'ref_spin_qpoint': 'spin_ref_qpoint',
                                                     'ref_spin_rotate_axis': 'spin_ref_rotate_axis',
                                                     })


def test_add_variable():
    print("Adding variable ref_spin_orientation to out2.nc")
    add_variable('out2.nc', 'ref_spin_orientation', 'f8', ('nspin', 'three'),
                 np.array([[0, 0, 1.0]], dtype='float'))


def test_merge():
    """
    merge ifc.nc, Oiju... into out.nc
    """
    #merge(["ifc.nc", "Oiju_matij.nc", "exchange2.nc", "Tijuv.nc"], "out.nc")
    merge(["results/onebody.nc", "results/Downfolded_hr.nc"], 'results/LAO_wann.nc')


def test_flip_one_spin():
    """
    Flip the first spin of the last step in the hist file
    """
    flip_one_spin(fname='./test.out_spinhist.nc', ispin=0)


if __name__ == '__main__':
    test_merge()
    # test_add_variable()
    # test_change_varname()
    # test_flip_one_spin()
