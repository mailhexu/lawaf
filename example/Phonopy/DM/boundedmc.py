import numpy as np
from numpy.random import random


class MC():
    def __init__(self, T, nlwf, effpot, avg_amp):
        self.T = T
        self.nlwf = nlwf
        self.lwf = np.zeros(nlwf, dtype=float)
        self.effpot = effpot
        self.avg_amp = avg_amp
        self.imove = 0

        self.deltaE=0.0


    def run_one_step(self):
        r = self.attempt(self.effpot)
        if random() < min(1, r):
            self.naccept += 1
            self.accept()
        else:
            self.reject()

    def accept(self):
        self.lwf[self.imove] = self.newlwf
        self.energy += self.deltaE

    def reject(self):
        pass

    def attempt(self):
        self.imove = np.random.randint(0, self.nlwf, size=1)
        self.lwf_old = self.lwf[imove]
        self.deltaE=0.0

    def move(lwf_old, lwf_new)

