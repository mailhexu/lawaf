import numpy as np


def evals_to_freqs(evals, factor):
    """
    convert phonon eigen values to Frequencies
    """
    return np.sign(evals) * np.sqrt(np.abs(evals)) * factor


def freqs_to_evals(freqs, factor):
    """
    convert phonon frequencies to eigenvalues
    """
    return np.sign(freqs / factor) * (freqs / factor) ** 2


def test():
    evals = np.array([1, -1, 2])
    factor = 13.4
    freqs = evals_to_freqs(evals, factor)
    assert np.allclose(freqs, np.array([1, -1, np.sqrt(2)]) * factor)
