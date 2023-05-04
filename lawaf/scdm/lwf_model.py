import numpy
import numba

@numba.jit
def rand():
    x=numpy.random.random_integers()
    #y=numpy.random.random_integers(1,3)
    return x

class LWF_model():
    def __init__(self):
        pass

    def make_supercell(self, ):
        pass

    def get_energy(self, x):
        pass

    def get_delta_energy(self):
        pass


print(rand())
