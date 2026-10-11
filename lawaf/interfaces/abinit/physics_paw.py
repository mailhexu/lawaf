# SPDX-License-Identifier: BSD-2-Clause
"""Two-sided PAW metric and true-neighbor augmentation over bounded tiles.

Electronic coefficients are (band, spinor, projector); trial coefficients
are (projector, trial, spinor). The scalar metric never mixes spin axes.
Only distinct reduced neighbor displacements own cached species operators.
"""

from __future__ import annotations

import numpy as np

from lawaf.interfaces.abinit.paw_coefficients import PawCoefficientError


class PawOverlap:
    """Borrow a verified provider; own only O(distinct-neighbor) Q matrices."""

    def __init__(self, provider):
        self.provider = provider
        self.header = provider.header
        self._reciprocal = 2 * np.pi * np.linalg.inv(self.header.rprimd).T
        self._positions = np.asarray(self.header.xred) @ self.header.rprimd
        self._q = {}

    @staticmethod
    def _key(shift):
        # Equivalent mesh displacements can differ by subtraction roundoff.
        return tuple(np.round(np.asarray(shift, dtype=float), 12))

    def prepare_neighbors(self, neighbor_table):
        """Replace the cache from the authoritative NNKP integer table."""
        self._q.clear()
        for ik, rows in enumerate(neighbor_table):
            for row in rows:
                jk = int(row[0])
                shift = self.header.kpt[jk] + row[1:4] - self.header.kpt[ik]
                key = self._key(shift)
                if key not in self._q:
                    vector = np.asarray(shift) @ self._reciprocal
                    self._q[key] = tuple(
                        kernel.form_factor(vector) for kernel in self.provider.kernels
                    )

    @property
    def cached_neighbor_count(self):
        return len(self._q)

    def projection(self, electronic, trial):
        """Return sum_atom,spin c_psi* DeltaS c_trial in W90 orientation."""
        electronic = np.asarray(electronic)
        trial = np.asarray(trial)
        result = np.zeros((electronic.shape[0], trial.shape[1]), complex)
        for atom, species in enumerate(self.header.typat):
            start, stop = self.provider.offsets[atom : atom + 2]
            metric = self.provider.kernels[int(species) - 1].delta_s
            result += np.einsum(
                "bsi,ij,jws->bw",
                electronic[:, :, start:stop].conj(),
                metric,
                trial[start:stop],
                optimize=True,
            )
        return result

    def overlap(self, left, right, shift):
        """Return sum_atom,spin c_left* Q(b) c_right exp(-i b.tau)."""
        key = self._key(shift)
        if key not in self._q:
            raise PawCoefficientError(
                "D6",
                self.provider.source.path,
                "neighbor shift",
                f"requested{key} absent from admitted cached set{tuple(self._q)}",
            )
        vector = np.asarray(shift) @ self._reciprocal
        result = np.zeros((left.shape[0], right.shape[0]), complex)
        for atom, species in enumerate(self.header.typat):
            start, stop = self.provider.offsets[atom : atom + 2]
            result += np.exp(-1j * (vector @ self._positions[atom])) * np.einsum(
                "bsi,ij,csj->bc",
                left[:, :, start:stop].conj(),
                self._q[key][int(species) - 1],
                right[:, :, start:stop],
                optimize=True,
            )
        return result
