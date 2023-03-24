import pytest

import scipy.sparse as sparse
from scipy.linalg import expm
import numpy as np

from hsolver.utils import transform_sv_tensor, mkron


def random_matrix(dim: int, is_sparse: bool = True):
    matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    if is_sparse:
        matrix = sparse.csc_matrix(matrix)
    return matrix


class TestTransformSVTensor:

    @pytest.fixture
    def dims(self):
        return 3, 6, 5

    @pytest.fixture
    def state(self, dims):
        return np.random.randn(*dims) + 1j * np.random.rand(*dims)

    def test_single_subsystem_unitary(self, dims, state):
        for axis in range(3):
            unitary = random_matrix(dims[axis])
            state_out1 = transform_sv_tensor(state, [axis], unitary=unitary)
            assert state_out1.shape == dims

            if axis == 0:
                state_out2 = np.einsum("ij,jmn->imn", np.array(unitary.todense()), state)
            elif axis == 1:
                state_out2 = np.einsum("ij,mjn->min", np.array(unitary.todense()), state)
            else:
                state_out2 = np.einsum("ij,mnj->mni", np.array(unitary.todense()), state)
            assert np.allclose(state_out1, state_out2)

    def test_two_subsystems_unitary(self, dims, state):
        axes = [0, 2]
        axes_dims = [dims[axis] for axis in axes]
        unitary = random_matrix(int(np.prod(axes_dims)))
        state_out1 = transform_sv_tensor(state, axes, unitary=unitary)
        assert state_out1.shape == dims

        unitary = np.array(unitary.todense()).reshape(axes_dims * 2)
        state_out2 = np.einsum("ijkl,knl->inj", unitary, state)
        assert np.allclose(state_out1, state_out2)

    def test_two_subsystems_unitary_permuted_axes(self, dims, state):
        axes = [2, 0]
        axes_dims = [dims[axis] for axis in axes]
        unitary = random_matrix(int(np.prod(axes_dims)))
        state_out1 = transform_sv_tensor(state, axes, unitary=unitary)
        assert state_out1.shape == dims

        unitary = np.array(unitary.todense()).reshape(axes_dims * 2)
        state_out2 = np.einsum("ijkl,lnk->jni", unitary, state)
        assert np.allclose(state_out1, state_out2)

    def test_full_transform_unitary(self, dims, state):
        unitary = random_matrix(int(np.prod(dims)))
        state_out1 = transform_sv_tensor(state, [0, 1, 2], unitary=unitary)

        unitary = np.array(unitary.todense()).reshape(dims * 2)
        state_out2 = np.einsum("ijklmn,lmn->ijk", unitary, state)
        assert np.allclose(state_out1, state_out2)

    def test_full_transform_unitary_permuted_axes(self, dims, state):
        axes = [2, 0, 1]
        axes_dims = [dims[axis] for axis in axes]
        unitary = random_matrix(int(np.prod(dims)))
        state_out1 = transform_sv_tensor(state, axes, unitary=unitary)
        assert state_out1.shape == dims

        unitary = np.array(unitary.todense()).reshape(axes_dims * 2)
        state_out2 = np.einsum("ijklmn,mnl->jki", unitary, state)
        assert np.allclose(state_out1, state_out2)

    def test_two_subsystems_log_unitary(self, dims, state):
        axes = [0, 2]
        axes_dims = [dims[axis] for axis in axes]
        log_unitary = random_matrix(int(np.prod(axes_dims)))
        state_out1 = transform_sv_tensor(state, axes, log_unitary=log_unitary)
        assert state_out1.shape == dims

        unitary = expm(np.array(log_unitary.todense())).reshape(axes_dims * 2)
        state_out2 = np.einsum("ijkl,knl->inj", unitary, state)
        assert np.allclose(state_out1, state_out2)


class TestMKron:

    def test_mkron(self):
        dims = [5, 7, 4]
        matrices = [random_matrix(dim) for dim in dims]
        m1 = sparse.kron(sparse.kron(matrices[0], matrices[1], format="csc"), matrices[2], format="csc")
        m2 = mkron(matrices[0], matrices[1], matrices[2])
        assert isinstance(m2, sparse.csc_matrix)
        assert np.allclose(m1.data, m2.data)

    def test_mkron_dense(self):
        dims = [5, 7, 4]
        matrices = [random_matrix(dim, is_sparse=False) for dim in dims]
        m1 = np.kron(np.kron(matrices[0], matrices[1]), matrices[2])
        m2 = mkron(matrices[0], matrices[1], matrices[2], is_sparse=False)
        assert isinstance(m2, np.ndarray)
        assert np.allclose(m1, m2)
