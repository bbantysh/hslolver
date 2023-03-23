"""
Utilities module.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from typing import List, Dict

import numpy as np


def push_axes(tensor: np.ndarray, new_positions: List[int]) -> np.ndarray:
    """Pushes first axes of the input tensor to specific positions

    :param tensor: Input tensor
    :param new_positions: New positions of first len(new_positions) axes
    :return: Resulting tensor
    """
    axes = list(range(len(new_positions), tensor.ndim))
    for idx_position in np.argsort(new_positions):
        axes.insert(new_positions[idx_position], idx_position)
    return tensor.transpose(axes)


def transform_sv_tensor(
        state_tensor: np.ndarray, unitary: np.ndarray, transformation_dims: Dict[int, int]
) -> np.ndarray:
    """Transforms state-vector tensor with some unitary matrix

    :param state_tensor: Input state-vector tensor
    :param unitary: Unitary matrix
    :param transformation_dims: Dimensions in which the matrix is acting
    :return: Output state-vector tensor
    """
    subsystems_indices = list(transformation_dims.keys())
    subsystems_dims = list(transformation_dims.values())
    subsystems_count = len(subsystems_indices)

    # transform tensor
    state_tensor = np.tensordot(
        unitary.reshape(2 * subsystems_dims),
        state_tensor,
        axes=(range(subsystems_count, 2 * subsystems_count), subsystems_indices)
    )

    # the convoluted axis are put in the beginning, so we need to bring them back
    state_tensor = push_axes(state_tensor, subsystems_indices)

    return state_tensor


def mkron(*matrices) -> np.ndarray:
    """Performs multiple matrices Kroneker product

    :param matrices: Input matrices
    :return: Resulting matrix
    """
    matrix = np.eye(1)
    for m in matrices:
        matrix = np.kron(matrix, m)
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
