import pytest
from iloptics import Layered
from iloptics.benchmarks import df95, dt_max


class TestLearn:

    @pytest.mark.parametrize(['dim', 'max_columns'], [(5, 5), (5, 3), (5, 1), (20, 1), (20, 1), (20, 19)])
    def test_learn(self, dim: int, max_columns: int):
        lt = Layered.dummy(dim)
        proto = lt.learn_proto(max_columns)

        # Create an instance of true interferometer
        lt_true = Layered.generate(dim, hadamard_like=True, hadamard_error=0.1)

        # Simulate the measurements according to protocol
        data = []
        for p in proto:
            lt_true.reset()
            if p.controls is not None:
                lt_true.control_layer(p.meta['phase_layer_idx'], p.controls)
            data.append(lt_true.tomo())

        # Learn and test
        lt.learn(proto, data)
        assert df95(lt, lt_true, 100) < 1e-8
        assert dt_max(lt, lt_true) < 1e-8

    @pytest.mark.parametrize(['dim', 'max_columns'], [(5, 5), (5, 3), (5, 1), (20, 1), (20, 1), (20, 19)])
    def test_learn_intensity_based(self, dim: int, max_columns: int):
        lt = Layered.dummy(dim)
        proto = lt.learn_intensity_based_proto(max_columns)

        # Create an instance of true interferometer
        lt_true = Layered.generate(dim, hadamard_like=True, hadamard_error=0.1)

        # Simulate the measurements according to protocol
        data = []
        for p in proto:
            lt_true.reset()
            if p.controls is not None:
                lt_true.control_layer(p.meta['phase_layer_idx'], p.controls)
            data.append(lt_true.tomo(measure_intensity=True))

        # Learn and test
        lt.learn_intensity_based(proto, data)
        assert df95(lt, lt_true, 100, up_to_out_phases=True) < 1e-8
        assert dt_max(lt, lt_true) < 1e-8
