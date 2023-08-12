"""
State evolution solver.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from typing import List, Union, Optional
from functools import cached_property
from time import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from hsolver.hamiltonian import Hamiltonian, HamiltonianTerm, SubSystem
from hsolver.envelopes import get_common_period
from hsolver.utils import ProgressPrinter, transform_sv_tensor, mkron, sparse_sum, is_dm, purify, format_bytes


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
            for subsystem in term.interaction.subsystems:
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
    def terms_full_matrices(self) -> List[sparse.csc_matrix]:
        """Full matrices of each interval term."""
        matrices = []
        for term in self.terms:
            matrices_to_multiply = (
                term.interaction.multipliers[term.interaction.subsystems.index(subsystem)]
                if subsystem in term.interaction.subsystems else sparse.eye(subsystem.dim, format="csc")
                for subsystem in self.subsystems
            )
            matrices.append(mkron(*matrices_to_multiply))
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
            term.envelope(t) * matrix
            for term, matrix in zip(self.terms, self.terms_full_matrices)
        ])

    def get_hamiltonian_derivative(self, t: float) -> sparse.csc_matrix:
        """Get the interval hamiltonian matrix derivative at specific time point.

        :param t: Time point.
        :return: Hamiltonian matrix.
        """
        assert self.time_start <= t < self.time_stop, "Wrong time point"
        return sparse_sum([
            term.envelope.drv(t) * matrix
            for term, matrix in zip(self.terms, self.terms_full_matrices)
        ])


class SystemEvolutionSolver:
    """Class for solving the quantum state evolution.

    :param hamiltonian: System hamiltonian.
    :param init_state: Initial quantum state (list of vector states for factorized state or a full state).
    :param max_step_size: Maximum time step size.
    :param min_step_size: Minimum time step size.
    :param verbose: Display output.
    """

    HAMILTONIAN_MAX_VARIATION = 0.1
    HAMILTONIAN_MAX_AMPLITUDE = 0.1

    def __init__(
            self,
            hamiltonian: Hamiltonian,
            init_state: Union[List[np.ndarray], np.ndarray],
            max_step_size: float,
            min_step_size: float = 0.,
            min_step_size_save: float = None,
            verbose: bool = True,
    ):
        self._hamiltonian = hamiltonian
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        self.min_step_size_save = min_step_size if min_step_size_save is None else min_step_size_save
        self.verbose = verbose

        self._time_list = None
        self._state_list = None

        self.init_state = init_state

    @property
    def hamiltonian(self) -> Hamiltonian:
        """System hamiltonian."""
        return self._hamiltonian

    @property
    def time_list(self) -> List[float]:
        """List of computed time points"""
        return self._time_list

    @property
    def state_list(self) -> List[np.ndarray]:
        """List of computed quantum states tensors at time points.
        Tensor dimensions is equal to `hamiltonian.dims`."""
        return self._state_list

    def print(self, msg: str):
        """Prints message in the verbose mode.

        :param msg: Message text.
        """
        if self.verbose:
            print(msg)

    def search_time_intervals(self, time_start: float, time_stop: float) -> List[TimeInterval]:
        """Searches the time intervals within some window.

        :param time_start: Window start time.
        :param time_stop: Window stop time.
        :return: List of time intervals.
        """
        time_points = [time_start, time_stop]
        for term in self._hamiltonian.terms:
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
            for term in self._hamiltonian.terms:
                if term.envelope.time_start is not None and term.envelope.time_start >= time_interval_stop:
                    # The term is not started yet
                    continue
                if term.envelope.time_stop is not None and term.envelope.time_stop <= time_interval_start:
                    # The term is already stopped
                    continue
                terms.append(term)

            time_interval = TimeInterval(time_interval_start, time_interval_stop, terms)
            if time_interval.time_duration < self.min_step_size:
                raise RuntimeError("Too small time interval. Try decreasing min_step_size")
            time_intervals.append(time_interval)

        return time_intervals

    def _prepare_init_state(self) -> np.ndarray:
        if type(self.init_state) is list:
            init_state = self._hamiltonian.factorized_state([
                purify(state) if is_dm(state) else state.reshape([-1, 1])
                for state in self.init_state
            ])
        else:
            state = self.init_state.copy()
            init_state = purify(state) if is_dm(state) else state.reshape([-1, 1])
        return init_state.astype(complex).reshape(self._hamiltonian.dims + [init_state.shape[1]])

    def solve(self, time_stop: float, time_start: float = 0.):
        """Solves the quantum state evolution.

        :param time_stop: Evolution stop time.
        :param time_start: Evolution start time (`0.` by default).
        """
        self.print("===> STARTING SOLVER")
        solver_time_start = time()
        state = self._prepare_init_state()
        self.print(f"* State tensor dimensions: {list(state.shape)} ({format_bytes(state.nbytes)})")

        self.print("* Searching for evolution time intervals")
        time_intervals = self.search_time_intervals(time_start, time_stop)

        self._time_list = [time_start]
        self._state_list = [state]
        for idx_time_interval, time_interval in enumerate(time_intervals):
            self.print(f"===> TIME INTERVAL {idx_time_interval + 1}/{len(time_intervals)}")
            solver_interval_time_start = time()

            t_start, t_stop, period = time_interval.time_start, time_interval.time_stop, time_interval.period
            self.print(f"* t_start = {t_start:.4e}, t_stop = {t_stop:.4e}")

            if len(time_interval.terms) == 0:
                self.print("* Nothing happens here")
                time_interval_modified = None
                time_list_interval = [t_stop]
                state_list_interval = [self._state_list[-1].copy()]

            elif period is None:
                step_title = "* Non periodic interval => use plain solving" if self.verbose else None
                time_interval_modified = None
                time_list_interval, state_list_interval = \
                    self.solve_for_interval(self._state_list[-1], time_interval, False, step_title)

            elif period == 0.:
                self.print("* Constant hamiltonian => use unitary product")
                time_interval_modified = TimeInterval(
                    t_start, min(t_start + self.max_step_size, t_stop), time_interval.terms
                )
                time_list_interval, state_list_interval, unitary_list_interval = \
                    self.solve_for_interval(self._state_list[-1], time_interval_modified, True)

            else:
                self.print(f"* Periodic hamiltonian (period: {period:.4e}) => using Floquet method")
                step_title = "* Solving for single period" if self.verbose else None
                time_interval_modified = TimeInterval(
                    t_start, min(t_start + period, t_stop), time_interval.terms
                )
                time_list_interval, state_list_interval, unitary_list_interval = \
                    self.solve_for_interval(self._state_list[-1], time_interval_modified, True, step_title)

            if time_interval_modified is not None and time_list_interval[-1] < t_stop:
                progress = ProgressPrinter(
                    min_value=t_start, max_value=t_stop, title="* Repeating unitary", verbose=self.verbose
                )

                transformation_dims = [self._hamiltonian.index(subsystem) for subsystem in time_interval.subsystems]
                time_current = time_list_interval[-1]
                state_current = state_list_interval[-1]
                dt_list_interval = np.diff([t_start] + time_list_interval)
                while time_current < t_stop:
                    for dt, unitary in zip(dt_list_interval, unitary_list_interval):
                        if time_current + dt > t_stop:
                            hamiltonian = time_interval.get_hamiltonian(time_current)
                            state_current = transform_sv_tensor(
                                state_current,
                                transformation_dims,
                                log_unitary=-1j * hamiltonian * (t_stop - time_current)
                            )
                            time_current = t_stop
                        else:
                            state_current = transform_sv_tensor(state_current, transformation_dims, unitary=unitary)
                            time_current += dt

                        time_list_interval.append(time_current)
                        state_list_interval.append(state_current)

                        progress.update(time_current)
                        if time_current >= t_stop:
                            break

                progress.stop()

            # Double check
            assert time_list_interval[-1] == t_stop, "Something went wrong"
            self.print(f"* Solved in {time() - solver_interval_time_start:.1f} sec.")

            self._time_list += time_list_interval
            self._state_list += state_list_interval

        self.print("============================")
        self.print(f"* Total time elapsed {time() - solver_time_start:.1f} sec.")

        return self

    def solve_for_interval(
            self,
            init_state_tensor: np.ndarray,
            time_interval: TimeInterval,
            return_unitary: bool = False,
            pp_title: str = None
    ):
        """Solves the quantum state evolution within the time interval.

        :param init_state_tensor: Initial quantum state tensor at start of the interval.
        Tensor dimensions must be equal to `hamiltonian.dims`.
        :param time_interval: Evolution time interval.
        :param return_unitary: Return unitary matrix for each time step.
        :param pp_title: Title of the progress printer.
        :return: List of time points, quantum states and unitary matrix at each time step.
        """
        time_list = []
        state_list = []
        unitary_list = []

        transformation_dims = [self._hamiltonian.index(subsystem) for subsystem in time_interval.subsystems]

        last_save_point = time_interval.time_start
        save_unitary = sparse.eye(time_interval.dim, format="csc")
        time_current = time_interval.time_start
        state_current_tensor = init_state_tensor
        progress = ProgressPrinter(
            min_value=time_interval.time_start,
            max_value=time_interval.time_stop,
            title=pp_title,
            verbose=pp_title is not None
        )

        while time_current < time_interval.time_stop:
            hamiltonian = time_interval.get_hamiltonian(time_current)

            if time_interval.period == 0:
                # Hamiltonian is time independent
                dt = np.inf
            else:
                # Amplitude should not be too high
                hamiltonian_norm = sparse.linalg.norm(hamiltonian)
                dt_amplitude = \
                    self.HAMILTONIAN_MAX_AMPLITUDE / hamiltonian_norm \
                    if hamiltonian_norm > 0. \
                    else np.inf

                # Time variation should not be too high
                hamiltonian_derivative_norm = sparse.linalg.norm(time_interval.get_hamiltonian_derivative(time_current))
                dt_derivative = \
                    self.HAMILTONIAN_MAX_VARIATION / hamiltonian_derivative_norm \
                    if hamiltonian_derivative_norm > 0. \
                    else np.inf

                dt = min(dt_amplitude, dt_derivative)

            dt = min(dt, self.max_step_size)
            dt = max(dt, self.min_step_size)
            dt = min(dt, time_interval.time_stop - time_current)

            log_unitary = -1j * hamiltonian * dt
            if return_unitary:
                unitary = sparse.linalg.expm(log_unitary)
                state_current_tensor = transform_sv_tensor(
                    state_current_tensor, transformation_dims, unitary=unitary
                )
                save_unitary = unitary @ save_unitary
            else:
                # Calculate in a faster way if we do not need to store unitary
                state_current_tensor = transform_sv_tensor(
                    state_current_tensor, transformation_dims, log_unitary=log_unitary
                )

            time_current += dt
            progress.update(time_current)

            if time_current - last_save_point >= self.min_step_size_save or time_current == time_interval.time_stop:
                time_list.append(time_current)
                state_list.append(state_current_tensor)
                if return_unitary:
                    unitary_list.append(save_unitary)
                    save_unitary = sparse.eye(time_interval.dim, format="csc")
                last_save_point = time_current

        progress.stop()

        if return_unitary:
            return time_list, state_list, unitary_list
        else:
            return time_list, state_list

    def get_subsystem_evolution(self, subsystems: List[SubSystem] = None):
        """Computes the list of subsystem density matrices at each time point.

        :param subsystems: List of subsystems (None to take the whole system).
        :return: List of density matrices.
        """
        assert self.state_list is not None
        if subsystems is None:
            subsystems = self._hamiltonian.subsystems
        return [self._hamiltonian.subsystem_dm(subsystems, state) for state in self.state_list]

    def get_populations_evolution(self, subsystems: List[SubSystem] = None):
        """Computes the time evolution of each subsystem basis state population.

        :param subsystems: List of subsystems.
        :return: List of p(t).
        """
        dm_list = self.get_subsystem_evolution(subsystems)
        return np.array([np.diag(dm).real for dm in dm_list]).T

    def get_purity_evolution(self, subsystems: List[SubSystem]):
        """Compute the time evolution of the subsystems density matrix purity.

        :param subsystems: List of subsystems.
        :return: List of purity values at each time point.
        """
        dm_list = self.get_subsystem_evolution(subsystems)
        return [abs(np.trace(dm @ dm)) for dm in dm_list]

    def get_entanglement_evolution(self, subsystems_a: List[SubSystem], subsystems_b: List[SubSystem]):
        """Computes the time evolution of the quantum state negativity.

        :param subsystems_a: List of part A subsystems.
        :param subsystems_b: List of part B subsystems.
        :return: List of negativity values at each time point.
        """
        assert 0 < len(subsystems_a) == len(set(subsystems_a)), "Invalid list"
        assert 0 < len(subsystems_b) == len(set(subsystems_b)), "Invalid list"

        subsystems = subsystems_a + subsystems_b
        assert len(set(subsystems)) == len(subsystems), "Lists contain same subsystems"
        dims = [subsystem.dim for subsystem in subsystems]

        axes_a = [subsystems.index(subsystem) for subsystem in subsystems_a]
        axes_a_conj = [axis + len(dims) for axis in axes_a]

        axes_b = [subsystems.index(subsystem) for subsystem in subsystems_b]
        axes_b_conj = [axis + len(dims) for axis in axes_b]

        negativity = []
        for dm in self.get_subsystem_evolution(subsystems):
            dm_pt = dm\
                .reshape(dims * 2)\
                .transpose(axes_a + axes_b_conj + axes_a_conj + axes_b)\
                .reshape([np.prod(dims)] * 2)
            w = np.linalg.eigvalsh(dm_pt)
            negativity.append(sum(np.abs(w) - w) / 2)
        return negativity
