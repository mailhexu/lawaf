from netCDF4 import Dataset
import numpy as np
from ase.units import Ha, eV, Angstrom, Bohr


def write_onebody_to_nc(fname, ids, orders, vals):
    root = Dataset(fname, 'w')
    nterm = len(ids)
    root.createDimension(dimname='wann_onebody_nterm', size=nterm)
    #root.dimensions['wann_onebody_nterm'] = nterm

    root.createVariable('wann_onebody_i',
                        'i4',
                        ('wann_onebody_nterm', ))
    root.variables['wann_onebody_i'][:] = np.array(ids)

    root.createVariable('wann_onebody_order',
                        'i4',
                        ('wann_onebody_nterm', ))
    root.variables['wann_onebody_order'][:] = np.array(orders)

    root.createVariable('wann_onebody_val',
                        'f8',
                        ('wann_onebody_nterm', ))
    root.variables['wann_onebody_val'][:] = np.array(vals)


def gen_terms(fname='onebody.nc'):
    ids=[]
    orders=[]
    vals=[]
    for i in [0,1]:
        for o, v in zip([4, 6, 8], [0.75, -1.1344, 0.438]):
            ids.append(i)
            orders.append(o)
            vals.append(v*Ha/Bohr**o*340.0/1200.0)
    print(f"ids: {ids}")
    print(f"orders: {orders}")
    print(f"vals: {vals}")

    write_onebody_to_nc(fname, ids, orders, vals)

if __name__=='__main__':
    gen_terms()
