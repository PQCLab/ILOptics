"""Module for working with linear optical transformation"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, NamedTuple, Dict, Any

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

    def tomo(self, noise_tomo: float = 0., measure_intensity: bool = False) -> np.ndarray:
        """Simulates the chip tomography

        :param noise_tomo: Statistical noise of transformation tomography
        :param measure_intensity: Measure output intensity only.
        :return: Returns chip transfer matrix with statistical errors.
        """
        tm = self.tm.copy()

        if noise_tomo > 0.:
            tm += noise_tomo * (np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim))

        if measure_intensity:
            out_phases = np.exp(-1j * np.angle(tm[:, 0]))
            tm = np.diag(out_phases) @ tm

        return tm

    def interferometry(
            self,
            rel_phases: np.ndarray,
            rel_port: int,
            ref_port: int = 0,
            in_intensity: float = 1.
    ) -> np.ndarray:
        assert rel_port != ref_port

        out_intensities = []
        for phase in rel_phases:
            in_state = np.zeros(self.dim, dtype=complex)
            in_state[ref_port] = 1. / np.sqrt(2)
            in_state[rel_port] = np.exp(1j * phase) / np.sqrt(2)
            out_intensities.append(in_intensity * np.abs(self.tm @ in_state) ** 2)

        return np.stack(out_intensities).T


class ReconfigurableLOTransformation(LOTransformation, ABC):
    """Class of reconfigurable LO transformation

    :param dim: Number of input/output modes
    :param num_controls: Number og transformation controls
    """
    def __init__(self, dim: int, num_controls: int):
        super(ReconfigurableLOTransformation, self).__init__(np.eye(dim, dtype=complex))
        self._num_controls = num_controls
        self._controls = [0.] * num_controls
        self.update()

    @property
    def num_controls(self) -> int:
        """Number of transformation controls"""
        return self._num_controls

    @property
    def controls(self) -> List[float]:
        """Current values of transformation controls"""
        return self._controls

    def update(self):
        """Updates transformation transfer matrix"""
        self._tm = self._get_tm()
        return self

    def reset(self):
        """Resets the controls to zero and updates transfer matrix"""
        self.control([0.] * self.num_controls)
        return self

    def control(self, controls: Optional[List[float]]):
        """Sets the transformations controls and updates its transfer matrix

        :param controls: List of controls values (if None, all controls are set to zero)
        """
        if controls is None:
            self.reset()
        else:
            self._set_controls(controls)
            self.update()
        return self

    def random_controls(self, max_value: float = 1.) -> List[float]:
        """Generates random controls

        :param max_value: Maximum control parameter value
        :return: List of controls
        """
        return list(np.random.rand(self.num_controls) * max_value)

    def _set_controls(self, controls: List[float]):
        """Sets the control values

        :param controls:
        :return:
        """
        assert len(controls) == self.num_controls
        self._controls = controls

    @abstractmethod
    def _get_tm(self) -> np.ndarray:
        """Abstract method that returns current transfer matrix taking controls into account

        :return: Transfer matrix
        """
        pass


class ProtoElement(NamedTuple):
    """Protocol element for measurement of reconfigurable transformation

    :param controls: List of transformation controls (None means all controls are disabled)
    :param meta: Element meta data
    """
    controls: List[float] = None
    meta: Dict[str, Any] = None
