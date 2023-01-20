"""The module provides benchmarks for estimating the efficiency of LO transformations reconstruction"""
from typing import Union, List, Tuple

import numpy as np

from iloptics import LOTransformation, ReconfigurableLOTransformation
from iloptics.layered import Layered


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


def df95(
        rt_rec: ReconfigurableLOTransformation,
        rt_true: ReconfigurableLOTransformation,
        test_controls: Union[int, List[List[float]]],
        control_max_value: int = 1.,
        return_controls: bool = False
) -> Union[float, Tuple[float, List[List[float]]]]:
    """This benchmark estimated the ability of reconfigurable LO transformation model to predict transfer matrices.
    It returns the 95-th percentile of 1-F over a set of configurations.

    :param rt_rec: Reconstructed reconfigurable transformation
    :param rt_true: True reconfigurable transformation
    :param test_controls: Number of test configurations or a list of control parameters
    :param control_max_value: Maximum value of control value (if test_controls is int)
    :param return_controls: True to return the list of control parameters
    :return: Benchmark value. If return_controls is True, the list of control parameters is also returned
    """
    if type(test_controls) is int:
        test_controls = [rt_rec.random_controls(control_max_value) for _ in range(test_controls)]

    df = []
    for c in test_controls:
        df.append(1. - fidelity(rt_true.control(c), rt_rec.control(c)))
    dfq = np.quantile(df, 0.95)
    if return_controls:
        return dfq, test_controls
    else:
        return dfq


def dt_max(lt_rec: Layered, lt_true: Layered) -> float:
    """This benchmark estimates the accuracy of mixing layer reconstruction for layered architecture.
    The benchmark returns the maximum error of determining the transmission coefficients (up to global losses).

    :param lt_rec: Reconstructed layered transformation
    :param lt_true: True layered transformation
    :return: Benchmark value
    """
    assert len(lt_rec.mix_layers) == len(lt_true.mix_layers)
    dim = lt_rec.dim
    dt = 0.
    for ml1, ml2 in zip(lt_true.mix_layers, lt_rec.mix_layers):
        # Transmission coefficients
        t1, t2 = np.abs(ml1.tm) ** 2, np.abs(ml2.tm) ** 2
        # Divide by global losses
        t1, t2 = t1 / (np.sum(t1) / dim), t2 / (np.sum(t2) / dim)
        dt = max(dt, max(np.abs(t1 - t2).flatten()))
    return dt
