import numpy as np
import scipy.sparse as ss

class MetropolisMonteCarlo():
    def __init__(self, potential):
        self.pot=potential



class ijR_potential():
    def __init__(self,  coeff : np.ndarray):
        self.coeff: np.ndarray = coeff

    def get_force(self, x):
        return -self.coeff @ x

    def get_total_energy(self, x, forces=None):
        if forces is None:
            forces = self.get_force(x)
        return -0.5 * np.dot(x, forces)

    def get_delta_energy(self, ix, x, dx):
        return -dx*(self.coeff[ix].dot(x))


class Sparse_ijR_potential():
    def __init__(self):
        pass
