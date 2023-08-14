"""
Module for working with hamiltonians.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import NamedTuple, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import scipy.sparse as sparse
from scipy import sparse as sparse

from hsolver.envelopes import Envelope, ConstantEnvelope, SumEnvelope, get_common_period
from hsolver.utils import mkron, sparse_sum


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

    def get_h0_string(self) -> str:
        """Returns string representation of H0"""
        return str(self.get_h0_matrix())

    def __repr__(self):
        return self.name


class Operator:
    """Class for time-independent subsystems operator

    :param subsystems: List of interacting subsystems
    :param multipliers: List of matrices corresponding to interaction hamiltonian
    :param multipliers_strings: List of string representation of each multiplier
    """

    def __init__(
            self,
            subsystems: List[SubSystem],
            multipliers: List[sparse.csc_matrix],
            multipliers_strings: List[str] = None
    ):
        assert len(subsystems) == len(multipliers), "Invalid number of multipliers"
        for subsystem, multiplier in zip(subsystems, multipliers):
            assert list(multiplier.shape) == [subsystem.dim] * 2, "Invalid multiplier dimension"
        self.subsystems = subsystems
        self.multipliers = multipliers
        self._multipliers_strings = multipliers_strings

    @cached_property
    def matrix(self) -> sparse.csc_matrix:
        matrix = self.multipliers[0]
        if len(self.multipliers) == 1:
            return matrix
        for matrix_j in self.multipliers[1:]:
            matrix = sparse.kron(matrix, matrix_j)
        return matrix

    @property
    def multipliers_strings(self) -> List[str]:
        if self._multipliers_strings is None:
            return ["x".join(map(str, matrix.shape)) + " matrix" for matrix in self.multipliers]
        return self._multipliers_strings


class HamiltonianTerm(NamedTuple):
    operator: Operator
    envelope: Envelope
    use_hc: bool = False


class TimeInterval:
    """Evolution time interval.

    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    :param terms: Hamiltonian terms that are acting in the interval.
    """

    def __init__(self, time_start: float, time_stop: float, terms: List[HamiltonianTerm]):
        self.time_start = time_start
        self.time_stop = time_stop
        self.time_duration = time_stop - time_start
        assert self.time_duration > 0., "Invalid time interval"

        self.terms = terms

    @cached_property
    def subsystems(self):
        """Subsystems that are touched in the interval."""
        subsystems = []
        for term in self.terms:
            for subsystem in term.operator.subsystems:
                if subsystem not in subsystems:
                    subsystems.append(subsystem)
        return subsystems

    @cached_property
    def dim(self) -> int:
        dim = 1
        for subsystem in self.subsystems:
            dim *= subsystem.dim
        return dim

    @cached_property
    def terms_full_matrices_and_envelopes(self) -> List[Tuple[sparse.csc_matrix, Envelope]]:
        """Full matrices of each interval term."""
        matrices = []
        for term in self.terms:
            matrices_to_multiply = (
                term.operator.multipliers[term.operator.subsystems.index(subsystem)]
                if subsystem in term.operator.subsystems
                else sparse.eye(subsystem.dim, format="csc")
                for subsystem in self.subsystems
            )
            matrix = mkron(*matrices_to_multiply)
            matrices.append((matrix, term.envelope))
            if term.use_hc:
                matrices.append((matrix.conj().T, term.envelope.conj()))
        return matrices

    @cached_property
    def period(self) -> Optional[float]:
        """Hamiltonian period within the interval."""
        return get_common_period([term.envelope for term in self.terms])

    def get_hamiltonian(self, t: float) -> sparse.csc_matrix:
        """Get the interval hamiltonian matrix at specific time point.

        :param t: Time point.
        :return: Hamiltonian matrix.
        """
        assert self.time_start <= t < self.time_stop, "Wrong time point"
        return sparse_sum([
            envelope(t) * matrix
            for matrix, envelope in self.terms_full_matrices_and_envelopes
        ])

    def get_hamiltonian_derivative(self, t: float) -> sparse.csc_matrix:
        """Get the interval hamiltonian matrix derivative at specific time point.

        :param t: Time point.
        :return: Hamiltonian matrix.
        """
        assert self.time_start <= t < self.time_stop, "Wrong time point"
        return sparse_sum([
            envelope.drv(t) * matrix
            for matrix, envelope in self.terms_full_matrices_and_envelopes
        ])

    @classmethod
    def search(cls, hamiltonian_terms: List[HamiltonianTerm], time_start: float = -np.inf, time_stop: float = np.inf):
        """Searches the time intervals within some time window for some list of Hamiltonian terms.

        :param hamiltonian_terms: Hamiltonian terms
        :param time_start: Window start time (-infinity by default).
        :param time_stop: Window stop time (+infinity by default).
        :return: List of time intervals.
        """
        time_points = [time_start, time_stop]
        for term in hamiltonian_terms:
            if term.envelope.time_start is not None and time_start <= term.envelope.time_start <= time_stop:
                time_points.append(term.envelope.time_start)
            if term.envelope.time_stop is not None and time_start <= term.envelope.time_stop <= time_stop:
                time_points.append(term.envelope.time_stop)
        time_points = np.sort(np.unique(time_points))

        time_intervals = []
        for idx in range(len(time_points) - 1):
            time_interval_start = time_points[idx]
            time_interval_stop = time_points[idx + 1]

            terms = []
            for term in hamiltonian_terms:
                if term.envelope.time_start is not None and term.envelope.time_start >= time_interval_stop:
                    # The term is not started yet
                    continue
                if term.envelope.time_stop is not None and term.envelope.time_stop <= time_interval_start:
                    # The term is already stopped
                    continue
                terms.append(term)

            time_intervals.append(cls(time_interval_start, time_interval_stop, terms))

        return time_intervals


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
            HamiltonianTerm(Operator([subsystem], [subsystem.get_h0_matrix()]), ConstantEnvelope(1.))
            for subsystem in subsystems
        ]
        return self

    def disable_h0(self):
        """Disables H0 terms"""
        return self.use_h0([])

    def add_interaction_term(self, operator: Operator, envelope: Envelope = None, use_hc: bool = False):
        """Adds interaction term to the hamiltonian

        :param operator: Operator instance
        :param envelope: Term envelope
        """
        for subsystem in operator.subsystems:
            assert subsystem in self.subsystems, f"Hamiltonian doesn't work with subsystem {subsystem.name}"
        if envelope is None:
            envelope = ConstantEnvelope(1.)
        envelopes = envelope.expand() if isinstance(envelope, SumEnvelope) else [envelope]
        for envelope in envelopes:
            self._interaction_terms.append(HamiltonianTerm(operator, envelope, use_hc))
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

    def print(self):
        if len(self.h0_terms) == 0:
            print("=> No H0 terms")
        else:
            print("=> H0 terms:")
            for idx, term in enumerate(self.h0_terms):
                subsystem = term.operator.subsystems[0]
                print(f"{idx + 1}) {subsystem.name}: {subsystem.get_h0_string()}")

        if len(self.interaction_terms) == 0:
            print("=> No Hint(t) terms")
        else:
            print("=> Hint(t) terms:")
            for time_interval in TimeInterval.search(self.interaction_terms):
                if len(time_interval.terms) == 0:
                    continue
                range_start = "-∞" if np.isinf(time_interval.time_start) else str(time_interval.time_start)
                range_stop = "+∞" if np.isinf(time_interval.time_stop) else str(time_interval.time_stop)
                print(f"==> t ∈ ({range_start}, {range_stop})")

                for idx, term in enumerate(time_interval.terms):
                    names = ", ".join([subsystem.name for subsystem in term.operator.subsystems])
                    multipliers = ", ".join(term.operator.multipliers_strings)
                    h_str = f"==> {idx + 1}) [{names}]: {term.envelope.to_string()} * [{multipliers}]"
                    if term.use_hc:
                        h_str += " + h.c."
                    print(h_str)

        print("")
