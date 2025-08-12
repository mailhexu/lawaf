# Project Structure

## Root Directory
- **lawaf/**: Main package source code
- **example/**: Example scripts and tutorials organized by interface
- **docs/**: Sphinx documentation source
- **tests/**: Unit tests and test utilities
- **scripts/**: Utility scripts for data conversion

## Core Package Structure (`lawaf/`)

### Main Modules
- **interfaces/**: External code interfaces (Siesta, Phonopy, Wannier90, Abinit)
- **lwf/**: Lattice Wannier Function core implementation
- **wannierization/**: Wannierization algorithms (SCDMK, projected WF)
- **mathutils/**: Mathematical utilities and linear algebra
- **plot/**: Visualization and plotting functions
- **io/**: File I/O operations (JSON, XSF formats)
- **utils/**: General utilities and helper functions
- **wrapper/**: Legacy wrappers and compatibility layers
- **ui/**: User interface components

### Interface Organization
Each interface follows a consistent pattern:
- `*_downfolder.py`: Main downfolding implementation
- `__init__.py`: Module exports
- Supporting modules for specific functionality

## Example Structure (`example/`)
- **Jupyter/**: Jupyter notebook tutorials
- **Phonopy/**: Phonopy interface examples with material systems
- **Siesta/**: Siesta interface examples
- **Wannier90/**: Wannier90 interface examples
- **convert_DDB/**: Abinit DDB conversion utilities

## Key Files
- **pyproject.toml**: Package configuration and dependencies
- **.ruff.toml**: Code formatting and linting configuration
- **requirements.txt**: Runtime dependencies
- **README.md**: Project overview and documentation links

## Naming Conventions
- **Classes**: PascalCase (e.g., `PhonopyDownfolder`, `LWF`)
- **Functions/Variables**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Files**: snake_case.py
- **Modules**: lowercase with underscores

## Import Patterns
Main package exports are defined in `lawaf/__init__.py`:
```python
from lawaf.interfaces import W90Downfolder, SiestaDownfolder, PhonopyDownfolder, NACPhonopyDownfolder
from lawaf.lwf.lwf import LWF
```

## Configuration Files
- JSON configuration files for downfolding parameters (e.g., `Downfold.json`)
- YAML files for phonopy parameters
- NetCDF files for data storage and exchange