from copy import deepcopy

import numpy as np


class LOTransformation:
    def __init__(self, tm: np.ndarray):
        assert tm.ndim == 2 and tm.shape[0] == tm.shape[1]
        self._tm = tm

    @property
    def tm(self) -> np.ndarray:
        return self._tm

    @property
    def dim(self) -> int:
        return self._tm.shape[0]

    def __mul__(self, other):
        return LOTransformation(self.tm @ other.tm)

    def replace(self, tm: np.ndarray):
        self._tm = tm
        return self

    def copy(self):
        return deepcopy(self)


def fidelity(v1: np.ndarray, v2: np.ndarray):
    norm_v1 = abs(np.trace(v1.conj().T @ v1))
    norm_v2 = abs(np.trace(v2.conj().T @ v2))
    return abs(np.trace(v1.conj().T @ v2)) ** 2 / (norm_v1 * norm_v2)


def project_uni(v: np.ndarray):
    u, _, vh = np.linalg.svd(v)
    return u @ vh
