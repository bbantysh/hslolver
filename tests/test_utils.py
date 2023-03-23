import numpy as np

from hsolver.utils import mkron


def random_matrix(dim: int) -> np.ndarray:
    return np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)


class TestMKron:

    def test_mkron(self):
        dims = [5, 7, 4]
        matrices = [random_matrix(dim) for dim in dims]
        m1 = np.kron(np.kron(matrices[0], matrices[1]), matrices[2])
        m2 = mkron(matrices[0], matrices[1], matrices[2])
        assert np.allclose(m1, m2)
