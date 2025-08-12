# Technology Stack

## Build System
- **Build Backend**: setuptools with pyproject.toml configuration
- **Package Manager**: PDM supported, pip compatible
- **Python Version**: >=3.10 required

## Core Dependencies
- **numpy** (<=2.0): Numerical computations
- **scipy** (>=1.7.0): Scientific computing
- **sisl** (>=0.9.0): Interface to Siesta DFT code
- **netcdf4**: NetCDF file format support
- **HamiltonIO** (>=0.1.2): Hamiltonian I/O operations
- **matplotlib** (>=3.4.0): Plotting and visualization
- **ase** (>=3.19): Atomic Simulation Environment
- **phonopy** (>=2.11.0): Phonon calculations
- **toml**: Configuration file parsing

## Development Tools
- **Code Formatting**: Ruff (configured in .ruff.toml)
  - Line length: 88 characters
  - Python 3.8+ target
  - Double quotes for strings, space indentation
- **Testing**: pytest (>=6.2.0)
- **Documentation**: Sphinx with Read the Docs theme
- **Pre-commit**: Configured for code quality

## Common Commands
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code with ruff
ruff format .

# Lint code
ruff check .

# Build documentation
cd docs && make html

# Build package
python -m build
```

## External Interfaces
- **Siesta**: Via sisl library for DFT calculations
- **Abinit**: Via Anaddb for phonon calculations  
- **Phonopy**: Direct interface for lattice dynamics
- **Wannier90**: Support for Wannier function files