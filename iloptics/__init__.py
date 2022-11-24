from typing import Union
from copy import deepcopy

import numpy as np


class LOTransformation:
    """Class of a linear-optical (LO) transformation

    :param tm: Transfer matrix
    """

    def __init__(self, tm: np.ndarray):
        assert tm.ndim == 2 and tm.shape[0] == tm.shape[1]
        self._tm = tm

    @property
    def tm(self) -> np.ndarray:
        """Transfer matrix"""
        return self._tm

    @property
    def dim(self) -> int:
        """Dimension"""
        return self._tm.shape[0]

    def __mul__(self, other):
        """Returns the product of two LO transformations

        :param other: Other LO transformation
        :return: Resulting LO transformation
        """
        return LOTransformation(self.tm @ other.tm)

    def replace(self, tm: np.ndarray):
        """Replaces transfer matrix

        :param tm: New transfer matrix
        """
        self._tm = tm
        return self

    def copy(self):
        """Returns the copy of LO transformation"""
        return deepcopy(self)


def fidelity(v1: Union[LOTransformation, np.ndarray], v2: Union[LOTransformation, np.ndarray]) -> float:
    """Fidelity between two LO transformations

    :param v1: LO transformation or transfer matrix
    :param v2: LO transformation or transfer matrix
    :return: Fidelity value
    """
    if isinstance(v1, LOTransformation):
        v1 = v1.tm

    if isinstance(v2, LOTransformation):
        v2 = v2.tm

    norm_v1 = abs(np.trace(v1.conj().T @ v1))
    norm_v2 = abs(np.trace(v2.conj().T @ v2))
    return abs(np.trace(v1.conj().T @ v2)) ** 2 / (norm_v1 * norm_v2)


def project_uni(v: np.ndarray):
    """Projects transfer matrix onto the set of unitary matrices

    :param v: Input matrix
    :return: Output unitary matrix
    """
    u, _, vh = np.linalg.svd(v)
    return u @ vh
