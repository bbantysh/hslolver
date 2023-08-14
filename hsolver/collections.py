"""
Collections module.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from functools import cached_property
from typing import NamedTuple, List, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from hsolver.hamiltonian import SubSystem, Hamiltonian, Operator
from hsolver.envelopes import PeriodicEnvelope


class Atom(SubSystem):
    """Atom subsystem.

    :param frequencies_0m: List of transition frequencies |0>-|m>.
    """
    def __init__(self, frequencies_0m: List[float]):
        super().__init__(len(frequencies_0m) + 1)
        self._frequencies_0m = frequencies_0m
        self._frequencies = [0] + self._frequencies_0m
        
    def frequency(self, level_a: int, level_b: int) -> float:
        """Returns transition frequency |a> -> |b>
        
        :param level_a: Level |a> frequency
        :param level_b: Level |b> frequency
        :return: Transition frequency
        """
        return self._frequencies[level_b] - self._frequencies[level_a]

    def sp(self, *levels) -> sparse.csc_matrix:
        sp = np.zeros([self.dim, self.dim])
        sp[levels[0], levels[1]] = 1.
        return sparse.csc_matrix(sp)

    @staticmethod
    def get_sp_string(*levels) -> str:
        return f"|{levels[0]}><{levels[1]}|"

    def get_h0_matrix(self) -> sparse.csc_matrix:
        return sparse.csc_matrix(np.diag(self._frequencies))

    def get_h0_string(self) -> str:
        return " + ".join([
            f"{freq}*|{level}><{level}|"
            for level, freq in enumerate(self._frequencies)
            if freq != 0.
        ])


class TwoLevelAtom(Atom):
    """Two level atom subsystem.

    :param frequency: Transition frequency.
    """
    sx = sparse.csc_matrix([[0., 1.], [1., 0.]])
    sy = sparse.csc_matrix([[0., -1j], [1j, 0.]])
    sz = sparse.csc_matrix([[1., 0.], [0., -1]])

    def __init__(self, frequency: float):
        super().__init__([frequency])
        self._frequency = frequency

    def sp01(self) -> sparse.csc_matrix:
        return super().sp(0, 1)

    def get_h0_matrix(self) -> sparse.csc_matrix:
        return -self._frequency / 2 * self.sz

    def get_h0_string(self) -> str:
        return f"-{self._frequency / 2} * σz"


class Oscillator(SubSystem):
    """Harmonic oscillator subsystem.

    :param dim: Oscillator dimension.
    :param frequency: Oscillator frequency
    :param offset: Minimum oscillator excitation number.
    """

    def __init__(self, dim: int, frequency: float, offset: int = 0):
        super(Oscillator, self).__init__(dim)
        self.frequency = frequency
        self._offset = offset

    def basis_state(self, jb: int) -> np.ndarray:
        return super(Oscillator, self).basis_state(jb - self._offset)

    def thermal_state(self, mean_excitations: float) -> np.ndarray:
        assert self._offset == 0, "Non-zero offset works bad for thermal states"
        nm = mean_excitations
        if nm <= 0.:
            pn = np.zeros(self.n.shape)
            pn[0] = 1.
        else:
            pn = np.exp(self.n * np.log(nm) - (self.n + 1) * np.log(nm + 1))
            pn /= pn.sum()
        return np.diag(pn)

    @cached_property
    def n(self) -> np.ndarray:
        return np.arange(self._offset, self._offset + self.dim)

    @cached_property
    def a_op(self) -> sparse.csc_matrix:
        """Creation operator"""
        return sparse.csc_matrix(np.diag(np.sqrt(np.arange(1, self._offset + self.dim)), 1)[self._offset:, self._offset:])

    @cached_property
    def a_op_dag(self) -> sparse.csc_matrix:
        """Annihilation operator"""
        return sparse.csc_matrix(np.diag(np.sqrt(np.arange(1, self._offset + self.dim)), -1)[self._offset:, self._offset:])

    @cached_property
    def n_op(self) -> sparse.csc_matrix:
        """Excitation number operator"""
        return sparse.csc_matrix(np.diag(self.n))

    @cached_property
    def x_op(self) -> sparse.csc_matrix:
        """X-quadrature operator"""
        return (self.a_op + self.a_op_dag) / np.sqrt(2)

    @cached_property
    def p_op(self) -> sparse.csc_matrix:
        """P-quadrature operator"""
        return -1j * (self.a_op + self.a_op_dag) / np.sqrt(2)

    def get_h0_matrix(self) -> sparse.csc_matrix:
        return self.frequency * (self.n_op + self.i_op / 2)

    def get_h0_string(self) -> str:
        return f"{self.frequency} * (n + 1/2)"

    def get_displacement_operator(self, value: float):
        """Returns the displacement operator

        :param value: Displacement value.
        :return: Displacement operator unitary matrix.
        """
        kz_op = value * (self.a_op + self.a_op_dag)
        return sparse.linalg.expm(1j * kz_op)

    @staticmethod
    def get_displacement_operator_string(value: float) -> str:
        return f"exp[i·{value}(a + a+)]"


class AtomFieldInteraction(Hamiltonian):
    """Atom-field interaction Hamiltonian

    :param atom: Atom.
    :param vibrations: List of atom vibration modes.
    :param field: Field parameters.
    :param interaction: Interaction parameters.
    :param use_atom_interaction_picture: Use interaction picture for atom (False by default).
    """

    class FieldParameters(NamedTuple):
        """Field parameters.

        :param frequency: Field frequency.
        :param phase: Field phase.
        :param pulse_front_width: Pulse front width.
        :param time_start: Pulse time start (None for -infinity).
        :param time_stop: Pulse time stop (None for +infinity).
        """
        frequency: float
        phase: float = 0.
        pulse_front_width: float = 0.
        time_start: float = None
        time_stop: float = None

    class InteractionParameters(NamedTuple):
        """Spin-field interaction parameters.

        :param rabi_frequency: Rabi frequency.
        :param ld_params: List of Lamb Dicke parameters for each mode.
        :param levels: Interaction levels of atom (|0>-|1> by default).
        """
        rabi_frequency: float
        ld_params: List[float]
        levels: Tuple[int, int] = (0, 1)

    def __init__(
            self,
            atom: Atom,
            vibrations: List[Oscillator],
            field: FieldParameters,
            interaction: InteractionParameters,
            use_atom_interaction_picture: bool = False
    ):
        super().__init__([atom] + vibrations)

        frequency = field.frequency
        if use_atom_interaction_picture:
            frequency -= atom.frequency(*interaction.levels)
        envelope = PeriodicEnvelope(
            amplitude=interaction.rabi_frequency / 2,
            frequency=frequency,
            phase=field.phase,
            time_start=field.time_start,
            time_stop=field.time_stop
        )
        if field.pulse_front_width > 0.:
            envelope = envelope.smoothen(field.pulse_front_width)

        displacement_operators = [
            vibration.get_displacement_operator(ld_param)
            for vibration, ld_param in zip(vibrations, interaction.ld_params)
        ]

        strings = [atom.get_sp_string(*interaction.levels)] + [
            vibration.get_displacement_operator_string(ld_param)
            for vibration, ld_param in zip(vibrations, interaction.ld_params)
        ]

        self.add_interaction_term(
            operator=Operator(
                subsystems=[atom] + vibrations,
                multipliers=[atom.sp(*interaction.levels)] + displacement_operators,
                multipliers_strings=strings
            ),
            envelope=envelope,
            use_hc=True
        )
        
        if use_atom_interaction_picture:
            self.use_h0(vibrations)


class MSGate(Hamiltonian):
    """Mølmer–Sørensen gate hamiltonian.

    :param atom1: Atom 1.
    :param atom2: Atom 2.
    :param vibrations: Atom common vibration modes.
    :param field_blue: Blue-shifted field parameters.
    :param field_red: Red-shifted field parameters.
    :param atom_field_int: Atom-fields interaction parameters (use if they are all the same).
    :param atom1_blue_field_int: Atom 1 and blue shifted field interaction parameters (`atom_field_int` by default).
    :param atom1_red_field_int: Atom1 and red shifted field interaction parameters (`atom1_blue_field_int` by default).
    :param atom2_blue_field_int: Atom2 and blue shifted field interaction parameters (`atom1_blue_field_int` by default).
    :param atom2_red_field_int: Atom2 and red shifted field interaction parameters (`atom1_red_field_int` by default).
    :param use_atoms_interaction_picture: Use interaction picture for atoms (False by default).
    """

    def __init__(
            self,
            atom1: Atom,
            atom2: Atom,
            vibrations: List[Oscillator],
            field_blue: AtomFieldInteraction.FieldParameters,
            field_red: AtomFieldInteraction.FieldParameters,
            atom_field_int: AtomFieldInteraction.InteractionParameters = None,
            atom1_blue_field_int: AtomFieldInteraction.InteractionParameters = None,
            atom1_red_field_int: AtomFieldInteraction.InteractionParameters = None,
            atom2_blue_field_int: AtomFieldInteraction.InteractionParameters = None,
            atom2_red_field_int: AtomFieldInteraction.InteractionParameters = None,
            use_atoms_interaction_picture: bool = False
    ):
        super(MSGate, self).__init__([atom1, atom2] + vibrations)

        if atom1_blue_field_int is None:
            atom1_blue_field_int = atom_field_int
        if atom1_red_field_int is None:
            atom1_red_field_int = atom1_blue_field_int
        if atom2_blue_field_int is None:
            atom2_blue_field_int = atom1_blue_field_int
        if atom2_red_field_int is None:
            atom2_red_field_int = atom1_red_field_int

        # Spin 1 interacting blue field
        self.add(AtomFieldInteraction(
            atom1, vibrations, field_blue, atom1_blue_field_int, use_atoms_interaction_picture
        ))

        # Spin 1 interacting red field
        self.add(AtomFieldInteraction(
            atom1, vibrations, field_red, atom1_red_field_int, use_atoms_interaction_picture
        ))

        # Spin 2 interacting blue field
        self.add(AtomFieldInteraction(
            atom2, vibrations, field_blue, atom2_blue_field_int, use_atoms_interaction_picture
        ))

        # Spin 2 interacting red field
        self.add(AtomFieldInteraction(
            atom2, vibrations, field_red, atom2_red_field_int, use_atoms_interaction_picture
        ))

        if use_atoms_interaction_picture:
            self.use_h0(vibrations)
