import numpy as np

def evals_to_freqs(evals, factor):
    """
    convert phonon eigen values to Frequencies
    """
    return np.sign(evals)*np.sqrt(np.abs(evals))* factor

def freqs_to_evals(freqs, factor):
    """
    convert phonon frequencies to eigenvalues
    """
    return np.sign(freqs/factor)*freqs**2

    
