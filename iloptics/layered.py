"""The module of layered architecture of reconfigurable linear optical interferometer"""
from typing import List, Union
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np
from scipy.linalg import dft, sqrtm

from iloptics import LOTransformation, project_uni


class MixLayer(LOTransformation):
    """Mixing layer of the layered chip"""
    @classmethod
    def random(cls, dim: int, hadamard_like: bool = True, hadamard_error: float = 0., max_losses: float = 0.):
        """Generates random mixing layer

        :param dim: Layer dimension
        :param hadamard_like: If True, the transfer matrix would have the form of Hadamard transform
        :param hadamard_error: The error of setting the Hadamard transform
        :param max_losses: Maximum losses inside the layer
        :return: Instance of MixingLayer
        """
        if hadamard_like:
            u = dft(dim) / np.sqrt(dim)
            if hadamard_error > 0.:
                u = project_uni(u + hadamard_error * (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)))
        else:
            u, _ = np.linalg.qr(
                np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim),
                mode="complete"
            )

        if max_losses > 0:
            squ = np.array(sqrtm(u))
            s = (1 - np.random.rand(dim) * max_losses)
            u = squ @ np.diag(s) @ squ
        return cls(u)

    @classmethod
    def dummy(cls, dim: int):
        """Generates a dummy mixing layer that does nothing (identity transformation)

        :param dim: Layer dimension
        :return: Instance of MixingLayer
        """
        return cls(np.eye(dim, dtype=complex))


class PhaseLayer(LOTransformation):
    """Phase layer of layered chip

    :param dim: Layer dimension
    :param coupling: Linear control-phase coupling constant (the phase value when the control value is 1.)
    :param noise: Relative Gaussian noise level of control parameters
    :param cross_talk: Cross-talk strength
    """
    def __init__(
            self,
            dim: int,
            coupling: float = 2 * np.pi,
            noise: float = 0.,
            cross_talk: float = 0.
    ):
        super(PhaseLayer, self).__init__(np.eye(dim))
        self._controls = np.zeros(dim)
        self._coupling = coupling
        self._noise = None
        self._cross_talk = None
        self._control_matrix = None
        self._control_matrix_inv = None
        self.set(noise=noise, cross_talk=cross_talk)

    @property
    def controls(self) -> np.ndarray:
        """Current control values"""
        return self._controls

    @property
    def coupling(self) -> float:
        """Linear control-phase coupling constant"""
        return self._coupling

    @property
    def noise(self) -> float:
        """Relative Gaussian noise level of control parameters"""
        return self._noise

    @property
    def cross_talk(self) -> float:
        """Cross-talk strength"""
        return self._cross_talk

    def control(self, controls: Union[List[float], np.ndarray]):
        """Sets the phase layer controls

        :param controls: Control values
        """
        assert len(controls) == self.dim
        self._controls = np.array(controls)
        if self.noise > 0.:
            self._controls *= (1 + np.random.randn(self.dim) * self.noise)
        # update transfer matrix
        phases = (self._control_matrix @ self._controls).astype(complex)
        self._tm = np.diag(np.exp(1j * phases))
        return self

    def phases2controls(self, phases: Union[List[float], np.ndarray]) -> np.ndarray:
        """Calculates the controls required to set the particular phases

        :param phases: Target phases
        :return: Controls
        """
        return self._control_matrix_inv @ phases

    def set(self, noise: float = None, cross_talk: float = None):
        """Sets the noise level and cross-talk strength

        :param noise: Noise level (by default the previous value holds)
        :param cross_talk: Cross-talk strength (by default the previous value holds)
        """
        if noise is not None:
            self._noise = noise

        if cross_talk is not None:
            self._cross_talk = cross_talk
            self._control_matrix = np.eye(self.dim) * self.coupling
            if cross_talk > 0.:
                for k in range(1, self.dim):
                    control_coupling_k = self.coupling * np.exp(-(k / cross_talk) ** 2 / 2)
                    self._control_matrix += np.diag(np.full(self.dim - k, control_coupling_k), +k)
                    self._control_matrix += np.diag(np.full(self.dim - k, control_coupling_k), -k)
            self._control_matrix_inv = np.linalg.inv(self._control_matrix)
        return self

    @staticmethod
    def ascending_phases(d: int) -> np.ndarray:
        """Returns the list of phases, distributed uniformly on unit circle

        :param d: Number of phases
        :return: Array of phases values
        """
        return np.array([j / d * 2 * np.pi for j in range(d)])


class LearnProtoElement(SimpleNamespace):
    """Protocol element for fast chip learning

    :param phase_layer: Instance of the phase layer to control (None means all phase layers are disabled)
    :param controls: Phase layer controls (None means all controls are disabled)
    :param mix_layer: Instance of mixing layer subjected to reconstruction
    :param columns_idx: List of indices of inverse mixing layer columns subjected to reconstruction
    :param data: The measured chip transfer matrix
    """
    phase_layer: PhaseLayer = None
    controls: np.ndarray = None
    mix_layer: MixLayer = None
    columns_idx: List[int] = None
    data: np.ndarray = None


class Chip(LOTransformation):
    """Layered chip

    :param layers: List of chip layers
    :param noise_tomo: The statistical noise of chip measurement
    """
    def __init__(self, layers: List[LOTransformation], noise_tomo: float = 0.):
        assert len(layers) > 0
        self._layers = layers
        self._dim = layers[0].dim

        self._mix_layers = [layer for layer in layers if isinstance(layer, MixLayer)]
        self._phase_layers = [layer for layer in layers if isinstance(layer, PhaseLayer)]
        self._controls = [layer.controls for layer in self._phase_layers]

        self.noise_tomo = noise_tomo

        super(Chip, self).__init__(np.eye(self.dim, dtype=complex))
        self._update()

    def _update(self):
        """Updates the chip transfer matrix taking the layers changes into account"""
        tm = np.eye(self.dim, dtype=complex)
        for layer in self.layers:
            tm = layer.tm @ tm
        tm = tm * np.exp(-1j * np.angle(tm[0, 0]))
        self._tm = tm
        return self

    @property
    def dim(self) -> int:
        """Chip dimension (number of input/output modes)"""
        return self._dim

    @property
    def layers(self) -> List[LOTransformation]:
        """List of chip layers"""
        return self._layers

    @property
    def mix_layers(self) -> List[MixLayer]:
        """List of chip mixing layers"""
        return self._mix_layers

    @property
    def phase_layers(self) -> List[PhaseLayer]:
        """List of chip phase layers"""
        return self._phase_layers

    @property
    def num_phase_layers(self) -> int:
        """Number of phase layers"""
        return len(self.phase_layers)

    def __len__(self):
        """Total number of layers"""
        return len(self._layers)

    @property
    def controls(self) -> List[np.ndarray]:
        """Current phase layers controls"""
        return self._controls

    def reset(self):
        """Resets the phase layers controls to zero"""
        self.control([np.zeros(self.dim)] * len(self.phase_layers))
        return self

    def control_layer(self, idx: int, controls: np.ndarray):
        """Sets the controls of a phase layer by index

        :param idx: Phase layer index
        :param controls: Phase layer controls
        """
        self.phase_layers[idx].control(controls)
        return self._update()

    def control(self, controls: List[np.ndarray]):
        """Sets the controls of all phase layers

        :param controls: List of phase layers controls
        """
        assert len(controls) == len(self._phase_layers)
        self._controls = controls
        for layer, layer_controls in zip(self._phase_layers, controls):
            layer.control(layer_controls)
        return self._update()

    def random_controls(self, max_value: float = 1.) -> List[np.ndarray]:
        """Generates random phase layers controls

        :param max_value: Maximum control parameter value
        :return: List of phase layers controls
        """
        return [np.random.rand(self.dim) * max_value for _ in range(len(self._phase_layers))]

    def tomo(self) -> np.ndarray:
        """Performs the chip tomography

        :return: Returns chip transfer matrix with statistical errors
        """
        tm = self.tm.copy()
        if self.noise_tomo > 0.:
            tm += self.noise_tomo * (np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim))
        return tm * np.exp(-1j * np.angle(tm[0, 0]))

    @classmethod
    def generate(
            cls,
            dim: int,
            num_phase_layers: int,
            control_coupling: float = 2 * np.pi,
            control_noise: float = 0.,
            control_cross_talk: float = 0.,
            hadamard_like: bool = True,
            hadamard_error: float = 0.,
            max_losses: float = 0.,
            noise_tomo: float = 0.
    ):
        """Generates random chip

        :param dim: Chip dimension
        :param num_phase_layers: Number of phase layers
        :param control_coupling: Phase layers control-phase coupling constant
        :param control_noise: Phase layers control noise
        :param control_cross_talk: Phase layers cross-talk strength
        :param hadamard_like: If True, the mixing layers transfer matrices would have the form of Hadamard transform
        :param hadamard_error: The error of setting the Hadamard transforms
        :param max_losses: Maximum losses inside mixing layers
        :param noise_tomo: The statistical noise of chip measurement
        :return: Chip instance
        """
        layers = []
        for _ in range(num_phase_layers):
            layers.append(MixLayer.random(dim, hadamard_like, hadamard_error, max_losses))
            layers.append(PhaseLayer(dim, control_coupling, control_noise, control_cross_talk))
        layers.append(MixLayer.random(dim, hadamard_like, hadamard_error, max_losses))
        return cls(layers, noise_tomo)

    @classmethod
    def dummy(cls, dim: int, num_phase_layers: int, phase_layer_prototype: PhaseLayer):
        """
        Creates dummy layered chip (mixing layers do nothing)

        :param dim: Chip dimension
        :param num_phase_layers: Number of phase layers
        :param phase_layer_prototype: Prototype of phase layers
        :return: Chip instance
        """
        layers = []
        for _ in range(num_phase_layers):
            layers += [MixLayer.dummy(dim), phase_layer_prototype.copy()]
        layers.append(MixLayer.dummy(dim))
        return cls(layers)

    def learn_proto(self, max_columns: int = None):
        """
        Generates the chip learning protocol

        :param max_columns: Maximum number of mixing layer columns to estimate at a time
        :return: List of protol elements
        """
        dim = self.dim
        if max_columns is None:
            max_columns = dim

        layers = []
        proto = [LearnProtoElement()]
        for phase_layer in self.phase_layers:
            mix_layer = MixLayer.dummy(dim)
            layers += [mix_layer, phase_layer]
            for offset in range(0, dim, max_columns):
                d = max_columns
                if (offset + d) >= dim:
                    d = dim - offset

                if d < dim:
                    phases_list = PhaseLayer.ascending_phases(d + 2)[:-1]
                    phases_static = phases_list[0]
                    phases_dynamic = phases_list[1:]

                    phases = np.ones(dim) * phases_static
                    phases[offset:(offset + d)] = phases_dynamic
                else:
                    phases = PhaseLayer.ascending_phases(d + 1)[:-1]

                proto.append(LearnProtoElement(
                    phase_layer=phase_layer,
                    controls=phase_layer.phases2controls(phases),
                    mix_layer=mix_layer,
                    columns_idx=list(range(offset, offset + d))
                ))
        layers.append(MixLayer.dummy(dim))

        return proto

    def learn(self, proto: List[LearnProtoElement], uni=False, disp=False):
        """Learns chip mixing layers

        :param proto: Learning protocol (including data)
        :param uni: Use unitary constraint on mixing layers
        :param disp: Display progress
        """
        # All phases are disable
        v0 = [p.data for p in proto if p.phase_layer is None][0]
        v0_inv = v0.conj().T if uni else np.linalg.inv(v0)

        v_total, v_total_inv = np.eye(self.dim), np.eye(self.dim)
        for mix_layer in tqdm(self.mix_layers[:-1], disable=not disp):
            # Reconstruct inverse matrix for a particular mix layer
            u_inv = mix_layer.tm.conj().T if uni else np.linalg.inv(mix_layer.tm)
            proto_j = [p for p in proto if p.mix_layer == mix_layer]
            for p in proto_j:
                cols = _eig_phase_sorted(v_total @ v0_inv @ p.data @ v_total_inv, len(p.columns_idx))
                u_inv[:, p.columns_idx] = cols
            # Get direct matrix
            if uni:
                u_inv = project_uni(u_inv)
                u = u_inv.conj().T
            else:
                u = np.linalg.inv(u_inv)

            mix_layer.replace(u)
            # Update total matrix
            v_total = u @ v_total
            v_total_inv = v_total_inv @ u_inv

        u = v0 @ v_total_inv
        if uni:
            u = project_uni(u)

        self.mix_layers[-1].replace(u)
        return self._update()


def _eig_phase_sorted(a: np.ndarray, col_d: int):
    w, v = np.linalg.eig(a)
    phases = np.mod(np.angle(w) + (2 * np.pi), 2 * np.pi)
    idx_sorted = np.argsort(phases)
    phases_sorted = phases.take(idx_sorted)
    idx_start = np.diff(np.append([phases_sorted[-1] - 2 * np.pi], phases_sorted)).argmax()
    idx = (list(idx_sorted[idx_start:]) + list(idx_sorted[:idx_start]))[-col_d:]
    return v.take(idx, axis=1)
