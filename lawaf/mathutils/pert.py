"""
This module is the perturbation to matrix.
"""
import numpy as np
from scipy.linalg import eigh


class Pert():
    def __init__(self, H0=None, evals=None, evecs=None):
        if evals is not None and evecs is not None:
            self.evals, self.evecs = evals, evecs
        elif H0 is not None:
            self.evals, self.evecs = eigh(self.H0)
        else:
            raise ValueError("at least H0| evals, evecs should be given")
        self.n = len(self.evals)
        self.evals, self.evecs = eigh(self.H0)
        self.dHH = None

    def evals1(self, dH):
        return self.Epert1(self.evecs, dH)

    def evecs1(self, dH):
        return self.Vpert1(self.evals, self.evecs, dH, self.n)

    def evals2(self, dH):
        return self.Epert2(self.evals, self.evecs, dH, self.n)

    @staticmethod
    def Epert1(evecs, dH):
        return np.diag(evecs.T.conj().dot(dH).dot(evecs))

    @staticmethod
    def Vpert1(evals, evecs, dH, n):
        dV = np.zeros((n, n), dtype='complex')
        dHH = evecs.T.conj() @ dH @ evecs
        for i in range(n):
            for k in range(n):
                if abs(evals[k] - evals[i]) > 0.000001:
                    dV[:, i] += dHH[k, i] / (evals[i] - evals[k]) * evecs[:, k]
        return dV

    @staticmethod
    def Epert2(evals, evecs, dH, n):
        d2E = np.zeros(n, dtype='complex')
        dHH = evecs.T.conj() @ dH @ evecs
        for i in range(n):
            for k in range(n):
                if abs(evals[k] - evals[i]) > 1e-10:
                    d2E[i] += dHH[i, k] * dHH[k, i] / (evals[i] - evals[k])
        return d2E

def unit2(x=0.3):
    return np.array([[np.cos(x), np.sin(x)],[-np.sin(x), np.cos(x)]])

def test_pert_degenerate_2d(x=0.01):
    H0=np.array([[2,0],[0,2]])
    evals0, evecs0=np.linalg.eigh(H0)

    H1=np.array([[1,2],[3,4]])
    H1+=H1.T
    print(evals0)

    H=H0+H1*x
    evals, evecs=eigh(H)
    print("Fdiff of eval: ",(evals-evals0)/x)
    print("Fdiff of evec: ", (evecs-evecs0))

    m=evecs0.T.conj().dot(H1).dot(evecs0)
    E1, c=eigh(m)
    print("Pert of eval: ", E1)
    c=c-np.eye(2)
    V10=np.dot( c[:,0], evecs0,)
    V11=np.dot(evecs0, c[:,1])
    print(V10)
    print(V11)





#test_pert_degenerate_2d()

def test_pert(x, n=4):
    H0 = np.random.random([n, n])
    H0 = H0 + H0.T
    evals0, evecs0 = eigh(H0)

    dH = np.random.random([n, n])
    dH = (dH + dH.T.conj())

    H = H0 + dH * x
    evals, evecs = eigh(H)

    # eigen value perturbation (1st order)
    dE = Pert.Epert1(evecs0, dH)
    dE2 = Pert.Epert2(evals0, evecs0, dH, n)
    print(evals)
    print(evals0)
    print(evals0 + x * dE)
    print(evals0 + x * dE + x * x * dE2)

    print(np.linalg.norm(evals0 - evals))
    print(np.linalg.norm(evals0 + x * dE - evals))
    print(np.linalg.norm(evals0 + x * dE + x * x * dE2 - evals))

    # eigen value perturbation (2nd order)

    # eigen vector perturbation
    dV = (Pert.Vpert1(evals0, evecs0, dH, n))
    print("dV:", dV)
    print("dV_fD:", (evecs - evecs0) / x)


def gen_degerate_mat(n):
    H0 = np.random.random([n, n])
    H0 = H0 + H0.T

    evals1, evecs = eigh(H0)
    evals = evals1
    evals[0] = evals[1]
    H0 = evecs.dot(np.diag(evals)).dot(evecs.T.conj())
    return H0


def test_pert_degenerate(x, n=2):
    H0 = gen_degerate_mat(n=n)
    evals0, evecs0 = eigh(H0)

    dH = np.random.random([n, n])
    dH = (dH + dH.T.conj()) * 1
    #dH = gen_degerate_mat(n=n)

    H = H0 + dH * x
    evals, evecs = eigh(H)

    # eigen value perturbation (1st order)
    dE = Pert.Epert1(evecs0, dH)
    dE2 = Pert.Epert2(evals0, evecs0, dH, n)
    print("dE: ", dE)
    print("dE2: ", dE2)

    print(evals)
    print(evals0 + x * dE)
    print(evals0 + x * dE + x * x * dE2)

    print("diff sumtot e0:", np.linalg.norm(evals0 - evals))
    print("diff sumtot e1:", np.linalg.norm(evals0 + x * dE - evals))
    print("diff sumtot e2:", np.linalg.norm(evals0 + x * dE - x * x * dE2 - evals))

    # eigen value perturbation (2nd order)

    # eigen vector perturbation
    dV = (Pert.Vpert1(evals0, evecs0, dH, n))
    print("dV:", dV)
    print("dV_fD:", (evecs - evecs0) / x)


def test_pert_2only(x, n=2):
    k = 2 * np.pi * 0.400050
    H0 = np.array([[0, 1 + np.exp(1j * k)], [1 + np.exp(-1j * k), 0]])
    evals0, evecs0 = eigh(H0)
    dH = np.array([[0, 1 - np.exp(1j * k)], [1 - np.exp(-1j * k), 0]])
    H = H0 + dH * x
    evals, evecs = eigh(H)

    # eigen value perturbation (1st order)
    dE = Pert.Epert1(evecs0, dH)
    dE2 = Pert.Epert2(evals0, evecs0, dH, n)
    print("dE:", dE)
    print("dE2:", dE2)

    print("Eigen value:")
    print(evals)
    print("Eigen value: first order pert")
    print(evals0 + x * dE)
    print("Eigen value: first+second order pert")
    print(evals0 + x * dE + x * x * dE2)

    print("diff: E0-E:")
    print(np.linalg.norm(evals0 - evals))
    print("diff: E0+E1-E:")
    print(np.linalg.norm(evals0 + x * dE - evals))
    print("diff: E0+E1+E2-E:")
    print(np.linalg.norm(evals0 + x * dE + x * x * dE2 - evals))

    # eigen value perturbation (2nd order)

    # eigen vector perturbation
    print(evecs0)
    dV = (Pert.Vpert1(evals0, evecs0, dH, n))
    print("dV:", dV)
    V = evecs0 + dV * x
    print(H)
    print("H:", V @ np.diag(evals) @ V.T.conj())
    print("dV_fD:", (evecs - evecs0) / x)

if __name__=="__main__":
    pass
    #test_pert(x=0.1, n=4)
    #print("========= Degenerate test=========")
    #test_pert_degenerate(x=0.11, n=4)
    #print("========= Degenerate test=========")
    #test_pert_2only(x=0.04)
