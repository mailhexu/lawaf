#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='lawaf',
    version='0.1.5',
    description='Package for constructing of Lattice Wannier function and other types of Wannier functions',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    package_data={},
    install_requires=['numpy>=1.20', 'scipy',  'matplotlib', 'ase>=3.19',
                      'netcdf4', 'sisl>=0.10', 'phonopy', "ipywidgets"
                      ],
    scripts=['scripts/phonopy_to_netcdf.py'
             ],
    classifiers=[
        'Development Status :: 3 - Alpha',
    ])
