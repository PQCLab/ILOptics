"""The module of layered architecture of reconfigurable linear optical interferometer"""
from typing import List, Union
from tqdm import tqdm

import numpy as np
from scipy.linalg import dft, sqrtm

from iloptics.lt import LOTransformation, ReconfigurableLOTransformation, ProtoElement


class MixLayer(LOTransformation):
    """Mixing layer of the layered chip"""
    @classmethod
    def random(
            cls,
            dim: int,
            hadamard_like: bool = True,
            hadamard_error: float = 0.,
            uniform_losses: float = 0.,
            non_uniform_losses: float = 0.,
    ):
        """Generates random mixing layer

        :param dim: Layer dimension
        :param hadamard_like: If True, the transfer matrix would have the form of Hadamard transform
        :param hadamard_error: The error of setting the Hadamard transform
        :param uniform_losses: The level on uniform linear losses
        :param non_uniform_losses: The level of non-uniform linear losses
        :return: Instance of MixingLayer
        """
        if hadamard_like:
            u = dft(dim) / np.sqrt(dim)
            if hadamard_error > 0.:
                u = _project_uni(u + hadamard_error * (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)))
        else:
            u, _ = np.linalg.qr(
                np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim),
                mode="complete"
            )

        if uniform_losses + non_uniform_losses > 0:
            squ = np.array(sqrtm(u))
            s = (1 - uniform_losses - np.random.rand(dim) * non_uniform_losses)
            if any(s) < 0:
                raise RuntimeError('Linear losses are too high')
            u = squ @ np.diag(s) @ squ
        return cls(u)

    @classmethod
    def dummy(cls, dim: int):
        """Generates a dummy mixing layer that does nothing (identity transformation)

        :param dim: Layer dimension
        :return: Instance of MixingLayer
        """
        return cls(np.eye(dim, dtype=complex))


class PhaseLayer(ReconfigurableLOTransformation):
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
        self._coupling = coupling
        super(PhaseLayer, self).__init__(dim, dim)

        self._noise = None
        self._cross_talk = None
        self._control_matrix = None
        self._control_matrix_inv = None
        self.set(noise=noise, cross_talk=cross_talk)

    @property
    def phases(self) -> List[float]:
        return [c * self.coupling for c in self.controls]

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

    def _set_controls(self, controls: List[float]):
        controls = np.array(controls)
        if self.noise > 0.:
            controls *= (1 + np.random.randn(self.dim) * self.noise)
        controls = list(self._control_matrix @ controls)
        super(PhaseLayer, self)._set_controls(controls)

    def _get_tm(self) -> np.ndarray:
        return np.diag(np.exp(1j * np.array(self.phases, dtype=complex)))

    def phases2controls(self, phases: Union[List[float], np.ndarray]) -> List[float]:
        """Calculates the controls required to set the particular phases

        :param phases: Target phases
        :return: Controls
        """
        return list(self._control_matrix_inv @ np.array(phases) / self.coupling)

    def set(self, noise: float = None, cross_talk: float = None):
        """Sets the noise level and cross-talk strength

        :param noise: Noise level (by default the previous value holds)
        :param cross_talk: Cross-talk strength (by default the previous value holds)
        """
        if noise is not None:
            self._noise = noise

        if cross_talk is not None:
            self._cross_talk = cross_talk
            self._control_matrix = np.eye(self.dim)
            if cross_talk > 0.:
                for k in range(1, self.dim):
                    control_ct = np.exp(-(k / cross_talk) ** 2 / 2)
                    self._control_matrix += np.diag(np.full(self.dim - k, control_ct), +k)
                    self._control_matrix += np.diag(np.full(self.dim - k, control_ct), -k)
            self._control_matrix_inv = np.linalg.inv(self._control_matrix)
        return self

    @staticmethod
    def ascending_phases(d: int) -> np.ndarray:
        """Returns the list of phases, distributed uniformly on unit circle

        :param d: Number of phases
        :return: Array of phases values
        """
        return np.array([j / d * 2 * np.pi for j in range(d)])


class Layered(ReconfigurableLOTransformation):
    """Layered reconfigurable LO transformation

    :param layers: List of transformation layers
    """
    def __init__(self, layers: List[LOTransformation]):
        assert len(layers) > 0
        self._layers = layers
        self._mix_layers = [layer for layer in layers if isinstance(layer, MixLayer)]
        self._phase_layers = [layer for layer in layers if isinstance(layer, PhaseLayer)]

        dim = layers[0].dim
        super(Layered, self).__init__(
            dim=dim,
            num_controls=len(self._phase_layers) * dim
        )

    def _get_tm(self) -> np.ndarray:
        tm = np.eye(self.dim, dtype=complex)
        for layer in self.layers:
            tm = layer.tm @ tm
        return tm

    def _set_controls(self, controls: List[float]):
        super(Layered, self)._set_controls(controls)
        for layer, layer_controls in zip(self._phase_layers, np.reshape(controls, (self.num_phase_layers, self.dim))):
            layer.control(layer_controls)
        return self

    @property
    def layers(self) -> List[LOTransformation]:
        """List of transformation layers"""
        return self._layers

    @property
    def mix_layers(self) -> List[MixLayer]:
        """List of transformation mixing layers"""
        return self._mix_layers

    @property
    def phase_layers(self) -> List[PhaseLayer]:
        """List of transformation phase layers"""
        return self._phase_layers

    @property
    def num_phase_layers(self) -> int:
        """Number of phase layers"""
        return len(self.phase_layers)

    def control_layer(self, idx: int, controls: List[float]):
        """Sets the controls of a phase layer by its index

        :param idx: Phase layer index
        :param controls: Phase layer controls
        """
        self.phase_layers[idx].control(controls)
        return self.update()

    @classmethod
    def generate(
            cls,
            dim: int,
            num_phase_layers: int = None,
            *,
            control_coupling: float = 2 * np.pi,
            control_noise: float = 0.,
            control_cross_talk: float = 0.,
            hadamard_like: bool = True,
            hadamard_error: float = 0.,
            uniform_losses: float = 0.,
            non_uniform_losses: float = 0.
    ):
        """Generates random transformation.

        :param dim: Number of input/output modes.
        :param num_phase_layers: Number of phase layers (dim + 1 by default).
        :param control_coupling: Phase layers control-phase coupling constant.
        :param control_noise: Phase layers control noise.
        :param control_cross_talk: Phase layers cross-talk strength.
        :param hadamard_like: If True, the mixing layers transfer matrices would have the form of Hadamard transform.
        :param hadamard_error: The error of setting the Hadamard transforms.
        :param uniform_losses: The level on uniform linear losses.
        :param non_uniform_losses: The level of non-uniform linear losses.
        :return: Transformation instance.
        """
        if num_phase_layers is None:
            num_phase_layers = dim + 1

        layers = []
        for idx_phase_layer in range(num_phase_layers):
            layers.append(PhaseLayer(dim, control_coupling, control_noise, control_cross_talk))
            if idx_phase_layer < num_phase_layers - 1:
                layers.append(MixLayer.random(dim, hadamard_like, hadamard_error, uniform_losses, non_uniform_losses))
        return cls(layers)

    @classmethod
    def dummy(cls, dim: int, num_phase_layers: int = None, phase_layer_prototype: PhaseLayer = None):
        """
        Creates dummy layered transformation (mixing layers do nothing)

        :param dim: Number of input/output modes
        :param num_phase_layers: Number of phase layers (dim + 1 by default)
        :param phase_layer_prototype: Prototype of phase layers (PhaseLayer(dim) by default)
        :return: Transformation instance
        """
        if num_phase_layers is None:
            num_phase_layers = dim + 1

        if phase_layer_prototype is None:
            phase_layer_prototype = PhaseLayer(dim)

        layers = []
        for idx_phase_layer in range(num_phase_layers):
            layers.append(phase_layer_prototype.copy())
            if idx_phase_layer < num_phase_layers - 1:
                layers.append(MixLayer.dummy(dim))
        return cls(layers)

    # === LEARNING BASED ON FULL TOMOGRAPHY ===
    def learn_proto(self, max_columns: int = None) -> List[ProtoElement]:
        """
        Generates the layered transformation learning protocol

        :param max_columns: Maximum number of mixing layer columns to estimate at a time
        :return: List of protol elements
        """
        dim = self.dim
        if max_columns is None:
            max_columns = dim

        proto = [ProtoElement()]
        for mix_layer in reversed(self.mix_layers[1:]):
            phase_layer = self.layers[self.layers.index(mix_layer) - 1]
            assert isinstance(phase_layer, PhaseLayer)

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

                proto.append(ProtoElement(
                    controls=phase_layer.phases2controls(phases),
                    meta={
                        'phases': phases,
                        'phase_layer_idx': self.phase_layers.index(phase_layer),
                        'mix_layer_idx': self.mix_layers.index(mix_layer),
                        'columns_idx': list(range(offset, offset + d))
                    }
                ))
        return proto

    def learn(self, proto: List[ProtoElement], data: List[np.ndarray], uni=False, disp=False):
        """Learns the layered transformation mixing layers

        :param proto: Learning protocol, generated by learn_proto
        :param data: Protocol measurements results
        :param uni: Use unitary constraint on mixing layers
        :param disp: Display progress
        """
        # All phases are disable
        v0 = [d for p, d in zip(proto, data) if p.controls is None][0]
        v0_inv = _inverse(v0, uni)

        v_total, v_total_inv = np.eye(self.dim), np.eye(self.dim)
        for mix_layer in tqdm(reversed(self.mix_layers[1:]), disable=not disp):
            mix_layer_idx = self.mix_layers.index(mix_layer)
            # Reconstruct inverse matrix for a particular mix layer
            u = mix_layer.tm.copy()
            for p, vj in zip(proto, data):
                # Filter elements for current mixing layer only
                if p.meta is None or p.meta['mix_layer_idx'] != mix_layer_idx:
                    continue
                cols = _eig_phase_sorted(v_total_inv @ vj @ v0_inv @ v_total, len(p.meta['columns_idx']))
                u[:, p.meta['columns_idx']] = cols
            if uni:
                u = _project_uni(u)

            mix_layer.replace(u)
            v_total = v_total @ u
            v_total_inv = _inverse(u, uni) @ v_total_inv

        u = v_total_inv @ v0
        if uni:
            u = _project_uni(u)

        self.mix_layers[0].replace(u)
        return self.update()

    # === LEARNING BASED ON INTENSITY MEASUREMENTS ===
    def learn_intensity_based_proto(self, max_columns: int = None) -> List[ProtoElement]:
        """
        Generates the layered transformation learning protocol based on intensity measurements.

        :param max_columns: Maximum number of mixing layer columns to estimate at a time
        :return: List of protol elements
        """
        proto = []
        for elem in self.learn_proto(max_columns):
            proto.append(elem)
            if elem.controls is None:
                continue

            phase_layer = self.phase_layers[elem.meta['phase_layer_idx']]
            phases = 2 * np.pi - elem.meta['phases']
            proto.append(ProtoElement(
                controls=phase_layer.phases2controls(phases),
                meta={
                    'phases': phases,
                    'phase_layer_idx': elem.meta['phase_layer_idx'],
                    'mix_layer_idx': elem.meta['mix_layer_idx'],
                    'columns_idx': elem.meta['columns_idx'],
                    'conjugate': True
                }
            ))
        return proto

    def learn_intensity_based(self, proto: List[ProtoElement], data: List[np.ndarray], uni=False, disp=False):
        """Learns the layered transformation mixing layers using intensity measurements

        :param proto: Learning protocol, generated by learn_proto
        :param data: Protocol measurements results
        :param uni: Use unitary constraint on mixing layers
        :param disp: Display progress
        """
        # All phases are disable
        v0 = [d for p, d in zip(proto, data) if p.controls is None][0]
        v0_inv = _inverse(v0, uni)

        v_total, v_total_inv = np.eye(self.dim), np.eye(self.dim)
        for mix_layer in tqdm(reversed(self.mix_layers[1:]), disable=not disp):
            mix_layer_idx = self.mix_layers.index(mix_layer)
            # Reconstruct inverse matrix for a particular mix layer
            u = mix_layer.tm.copy()
            for idx in range(len(proto)):
                p = proto[idx]
                # Filter elements for current mixing layer only
                if p.meta is None or p.meta['mix_layer_idx'] != mix_layer_idx:
                    continue
                # Filter non-conjugate elements
                if 'conjugate' in p.meta and p.meta['conjugate']:
                    continue

                assert proto[idx + 1].meta['conjugate']
                vj, vj_conj = data[idx], data[idx + 1]

                x, y = vj @ v0_inv, v0 @ _inverse(vj_conj, uni)
                out_phases = np.angle(x[:, 0]) - np.angle(y[:, 0])
                cols = _eig_phase_sorted(np.diag(np.exp(-1j * out_phases)) @ x, len(p.meta['columns_idx']))
                cols = v_total_inv @ cols

                u[:, p.meta['columns_idx']] = cols

            mix_layer.replace(u)
            v_total = v_total @ u
            v_total_inv = _inverse(u, uni) @ v_total_inv

        u = v_total_inv @ v0
        if uni:
            u = _project_uni(u)

        self.mix_layers[0].replace(u)
        return self.update()


def _inverse(a: np.ndarray, uni: bool) -> np.ndarray:
    """Return matrix inversion

    :param a: Input matrix
    :param uni: True if input matrix is unitary
    :return: Matrix inverse
    """
    return a.conj().T if uni else np.linalg.inv(a)


def _eig_phase_sorted(a: np.ndarray, col_d: int) -> np.ndarray:
    """Evaluates the part of transfer matrix by eigenvalues phase sorting algorithm

    :param a: Input matrix
    :param col_d: Number of columns to estimate
    :return: Estimated columns matrix
    """
    w, v = np.linalg.eig(a)
    phases = np.mod(np.angle(w) + (2 * np.pi), 2 * np.pi)
    idx_sorted = np.argsort(phases)
    phases_sorted = phases.take(idx_sorted)
    idx_start = np.diff(np.append([phases_sorted[-1] - 2 * np.pi], phases_sorted)).argmax()
    idx = (list(idx_sorted[idx_start:]) + list(idx_sorted[:idx_start]))[-col_d:]
    return v.take(idx, axis=1)


def _project_uni(v: np.ndarray) -> np.ndarray:
    """Projects transfer matrix onto the set of unitary matrices

    :param v: Input matrix
    :return: Output unitary matrix
    """
    u, _, vh = np.linalg.svd(v)
    return u @ vh
