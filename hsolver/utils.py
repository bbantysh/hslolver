"""
Utilities module.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from typing import List

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg


def transform_sv_tensor(
        state_tensor: np.ndarray,
        transformation_axes: List[int],
        *,
        unitary: sparse.csc_matrix = None,
        log_unitary: sparse.csc_matrix = None
) -> np.ndarray:
    """Transforms state-vector tensor with some unitary matrix.

    Supports exp(A)*b fast multiplication if log_unitary is provided.

    :param state_tensor: Input state-vector tensor.
    :param transformation_axes: Axes to transform.
    :param unitary: Unitary matrix.
    :param log_unitary: Unitary logarithm.
    :return: Output state-vector tensor.
    """
    n_axes = state_tensor.ndim
    dims = state_tensor.shape
    transformation_dims = [dims[axis] for axis in transformation_axes]
    other_axes = list(set(list(range(n_axes))).difference(transformation_axes))
    other_dims = [dims[axis] for axis in other_axes]

    # move convolution axes to beginning
    state_tensor = np.moveaxis(state_tensor, transformation_axes, list(range(len(transformation_axes))))

    # reshape to matrix
    state_tensor = np.reshape(state_tensor, (np.prod(transformation_dims), np.prod(other_dims, dtype=int)))

    # transform matrix
    if unitary is not None:
        state_tensor = unitary @ state_tensor
    elif log_unitary is not None:
        state_tensor = sparse.linalg.expm_multiply(log_unitary, state_tensor)
    else:
        RuntimeError("Please specify either unitary or log_unitary argument")

    # reshape back to tensor
    state_tensor = np.reshape(state_tensor, transformation_dims + other_dims)

    # bring axes back
    state_tensor = np.moveaxis(state_tensor, list(range(len(transformation_axes))), transformation_axes)

    return state_tensor


def mkron(*matrices, is_sparse: bool = True):
    """Performs multiple matrices Kronecker product

    :param matrices: Input matrices
    :param is_sparse: Use sparse matrices
    :return: Resulting matrix
    """
    matrix = sparse.eye(1, format="csc") if is_sparse else np.eye(1)
    for m in matrices:
        matrix = sparse.kron(matrix, m, format="csc") if is_sparse else np.kron(matrix, m)
    return matrix


def sparse_sum(matrices: List[sparse.csc_matrix]) -> sparse.csc_matrix:
    """Returns the sum if several sparse matrices

    :param matrices: Input matrices
    :return: Sum of input matrices
    """
    matrix = matrices[0]
    for m in matrices[1:]:
        matrix += m
    return matrix


class ProgressPrinter:
    """Class for printing the progress of calculations

    :param min_value: Min iterator value
    :param max_value: Max iterator value
    :param title: Title of the process
    :param verbose: Set False to disable printing
    """

    def __init__(self, min_value: float, max_value: float, title: str, verbose: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        self.title = title
        self.verbose = verbose

        self.current_progress = 0
        self.num_steps = 0
        self.print(new_line=False)

    def print(self, new_line: bool):
        """Update the line with current status

        :param new_line: Place the caret on the new line
        """
        if not self.verbose:
            return
        end = "\n" if new_line else ""
        print(f"\r{self.get_line()}", end=end)

    def get_percentage(self, value: float) -> int:
        """Computes the progress

        :param value: Current iterator value
        :return: The progress in percents
        """
        return round((value - self.min_value) / (self.max_value - self.min_value) * 100)

    def get_line(self):
        """Returns the line to print

        :return: Line to print
        """
        return f"{self.title}: {self.current_progress}% ({self.num_steps} steps)"

    def update(self, value: float):
        """Updates the progress with new iterator value

        :param value: Iterator value
        """
        self.num_steps += 1
        progress = self.get_percentage(value)
        if progress == self.current_progress:
            return
        self.current_progress = progress
        self.print(new_line=False)

    def stop(self):
        """Stops the progress printer"""
        self.print(new_line=True)
