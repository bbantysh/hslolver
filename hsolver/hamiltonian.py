"""
Module for working with hamiltonians.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import NamedTuple, List
from collections import defaultdict

import numpy as np
import scipy.sparse as sparse

from hsolver.envelopes import Envelope, ConstantEnvelope, SumEnvelope


class SubSystem(ABC):
    """Abstract class of subsystem

    :param dim: Subsystem dimension
    """

    __names_collections = defaultdict(lambda: 0)

    def __init__(self, dim: int, name: str = None):
        self._dim = dim

        if name is None:
            class_name = self.__class__.__name__
            self.__names_collections[class_name] += 1
            name = class_name + str(self.__names_collections[class_name])
        self.name = name

    @property
    def dim(self) -> int:
        """Subsystem dimension"""
        return self._dim

    @cached_property
    def i_op(self) -> sparse.csc_matrix:
        """Subsystem identity operator"""
        return sparse.eye(self.dim, format="csc")

    def basis_state(self, jb: int) -> np.ndarray:
        """Returns the vector of the subsystem basis state

        :param jb: Basis state index
        :return: Vector of the basis state
        """
        return np.eye(1, self.dim, jb).reshape((-1,))

    @abstractmethod
    def get_h0_matrix(self) -> sparse.csc_matrix:
        """Returns the H0 hamiltonian matrix"""
        pass

    def __repr__(self):
        return self.name


class Interation:
    """Class for subsystems interaction
    TODO: Make it possible to define analytical exponentiation

    :param multipliers: List of matrices corresponding to interaction hamiltonian
    """

    def __init__(self, subsystems: List[SubSystem], multipliers: List[sparse.csc_matrix]):
        assert len(subsystems) == len(multipliers), "Invalid number of multipliers"
        self.subsystems = subsystems
        self.multipliers = multipliers

    @cached_property
    def matrix(self) -> sparse.csc_matrix:
        """Tensor product of the multipliers"""
        matrix = self.multipliers[0]
        if len(self.multipliers) == 1:
            return matrix
        for matrix_j in self.multipliers[1:]:
            matrix = sparse.kron(matrix, matrix_j)
        return matrix


class HamiltonianTerm(NamedTuple):
    interaction: Interation
    envelope: Envelope


class Hamiltonian:
    """Class for working with hamiltonian

    :param subsystems: List of subsystem instances for which the hamiltonian is acting
    """

    def __init__(self, subsystems: List[SubSystem]):
        self._subsystems = subsystems
        self._subsystem_indices = {subsystem: idx for idx, subsystem in enumerate(subsystems)}
        self._h0_terms = []
        self._interaction_terms = []
        self._cache = {}

        self.use_h0()

    @property
    def subsystems(self) -> List[SubSystem]:
        """Subsystems for which the hamiltonian is acting"""
        return self._subsystems

    @property
    def interaction_terms(self) -> List[HamiltonianTerm]:
        """List of interaction terms"""
        return self._interaction_terms

    @property
    def h0_terms(self) -> List[HamiltonianTerm]:
        """List of H0 terms"""
        return self._h0_terms

    @property
    def terms(self) -> List[HamiltonianTerm]:
        """List of hamiltonian terms"""
        return self._h0_terms + self._interaction_terms

    @property
    def dims(self) -> List[int]:
        """List of subsystems dimensions"""
        if "dims" not in self._cache:
            self._cache.update({"dims": [subsystem.dim for subsystem in self.subsystems]})
        return self._cache["dims"]

    @property
    def dim(self) -> int:
        """System dimension"""
        if "dim" not in self._cache:
            self._cache.update({"dim": int(np.prod(self.dims))})
        return self._cache["dim"]

    def index(self, subsystem: SubSystem) -> int:
        """Returns the index corresponding to the specific subsystem instance

        :param subsystem: Subsystem instance
        :return: Index of the subsystem instance
        """
        return self._subsystem_indices[subsystem]

    def use_h0(self, subsystems: List[SubSystem] = None):
        """Set the usages of H0 hamiltonian

        :param subsystems: List of subsystem instances for which H0 is included.
        None to use all subsystems. [] to not use H0 at all.
        """
        if subsystems is None:
            subsystems = self.subsystems
        self._h0_terms = [
            HamiltonianTerm(Interation([subsystem], [subsystem.get_h0_matrix()]), ConstantEnvelope(1.))
            for subsystem in subsystems
        ]
        return self

    def disable_h0(self):
        """Disables H0 terms"""
        return self.use_h0([])

    def add_interaction_term(self, interaction: Interation, envelope: Envelope):
        """Adds interaction term to the hamiltonian

        :param interaction: Interaction instance
        :param envelope: Term envelope
        """
        for subsystem in interaction.subsystems:
            assert subsystem in self.subsystems, f"Hamiltonian doesnt work with subsystem {subsystem.name}"
        envelopes = envelope.expand() if isinstance(envelope, SumEnvelope) else [envelope]
        for envelope in envelopes:
            self._interaction_terms.append(HamiltonianTerm(interaction, envelope))
        return self

    def factorized_state(self, states: List[np.ndarray]) -> np.ndarray:
        """Generates the factorized state of the system

        :param states: List of subsystem states (vector-states or density matrices).
        :return: Purified factorized state of the whole system
        """
        num_subsystems = len(self.subsystems)
        assert len(states) == num_subsystems
        state = states[0]
        for state_j in states[1:]:
            state = np.kron(state, state_j)
        return state

    def subsystem_dm(self, subsystems: List[SubSystem], state: np.ndarray) -> np.ndarray:
        """Computes the subsystems density matrix by partial trace.

        :param subsystems: List of subsystems.
        :param state: Input vector-state of the whole system, or its purified density matrix.
        :return: Subsystems' density matrix.
        """
        num_subsystems = len(self.subsystems)
        if state.ndim == 1 or state.ndim == num_subsystems:  # Pure state
            state = state.reshape(self.dims + [1])
        else:  # Mixed state
            state = state.reshape(self.dims + [state.shape[-1]])

        # Put the axes to remain in the beginning of the tensor and reshape it to d1xd2 matrix.
        # The second dimension corresponds to the subsystems that should be traced out.
        axes_remain = [self.index(subsystem) for subsystem in subsystems]
        dim_remain = np.prod([subsystem.dim for subsystem in subsystems])
        state = np.moveaxis(state, axes_remain, range(len(axes_remain))).reshape([dim_remain, -1])

        return state @ state.conj().T

    def add(self, other):
        """Adds other hamiltonian.

        :param other: Hamiltonian to add.
        """
        for subsystem in other.subsystems:
            if subsystem not in self._subsystems:
                self._subsystems.append(subsystem)
        for term in other.interaction_terms:
            self._interaction_terms.append(term)
        self._cache.clear()
