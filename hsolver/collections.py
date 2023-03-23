"""
Collections module.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

from hsolver.hamiltonian import SubSystem, Hamiltonian, Interation
from hsolver.envelopes import PeriodicEnvelope


class Spin(SubSystem):
    """Spin subsystem.

    :param frequency: Spin frequency.
    """
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1]])
    sp = np.array([[0, 1.], [0, 0]])
    s0 = np.eye(2)

    def __init__(self, frequency: float):
        super(Spin, self).__init__(2)
        self.frequency = frequency

    def get_h0_matrix(self) -> np.ndarray:
        return self.frequency / 2 * self.sz


class Oscillator(SubSystem):
    """Harmonic oscillator subsystem.

    :param dim: Oscillator dimension.
    :param frequency: Oscillator frequency
    :param offset: Minimum oscillator excitation number.
    """

    def __init__(self, dim: int, frequency: float, offset: int = 0):
        super(Oscillator, self).__init__(dim)
        self.frequency = frequency
        self.offset = offset

    def basis_state(self, jb: int) -> np.ndarray:
        return super(Oscillator, self).basis_state(jb - self.offset)

    @cached_property
    def a_op(self) -> np.ndarray:
        """Creation operator"""
        return np.diag(np.sqrt(np.arange(1, self.offset + self.dim)), 1)[self.offset:, self.offset:]

    @cached_property
    def a_op_dag(self) -> np.ndarray:
        """Annihilation operator"""
        return np.diag(np.sqrt(np.arange(1, self.offset + self.dim)), -1)[self.offset:, self.offset:]

    @cached_property
    def n_op(self) -> np.ndarray:
        """Excitation number operator"""
        return np.diag(np.arange(0, self.offset + self.dim))[self.offset:, self.offset:]

    @cached_property
    def x_op(self) -> np.ndarray:
        """X-quadrature operator"""
        return (self.a_op + self.a_op_dag) / np.sqrt(2)

    @cached_property
    def p_op(self) -> np.ndarray:
        """P-quadrature operator"""
        return -1j * (self.a_op + self.a_op_dag) / np.sqrt(2)

    def get_h0_matrix(self) -> np.ndarray:
        return self.frequency * (self.n_op + self.i_op / 2)

    def get_displacement_operator(self, value: float):
        """Returns the displacement operator

        :param value: Displacement value.
        :return: Displacement operator unitary matrix.
        """
        kz_op = value * (self.a_op + self.a_op_dag)
        return expm(1j * kz_op)


class SpinFieldInteractionHamiltonian(Hamiltonian):
    """Spin-field interaction hamiltonian

    :param spin: Spin.
    :param vibration: Spin vibration mode.
    :param field: Field parameters.
    :param interaction: Interaction parameters.
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

        :param rabi_frequency. Rabi frequency.
        :param ld_param: Lamb Dicke parameter.
        """
        rabi_frequency: float
        ld_param: float

    def __init__(
            self,
            spin: Spin,
            vibration: Oscillator,
            field: FieldParameters,
            interaction: InteractionParameters
    ):
        super(SpinFieldInteractionHamiltonian, self).__init__([spin, vibration])

        displacement_operator = vibration.get_displacement_operator(interaction.ld_param)
        envelope = PeriodicEnvelope(
            amplitude=interaction.rabi_frequency,
            frequency=field.frequency,
            phase=field.phase,
            time_start=field.time_start,
            time_stop=field.time_stop
        )
        if field.pulse_front_width > 0.:
            envelope = envelope.smoothen(field.pulse_front_width)

        self.add_interaction_term(
            subsystems=[spin, vibration],
            interaction=Interation([spin.sp, displacement_operator]),
            envelope=envelope
        )

        self.add_interaction_term(
            subsystems=[spin, vibration],
            interaction=Interation([spin.sp.conj().T, displacement_operator.conj().T]),
            envelope=envelope.conj()
        )


class MSGateHamiltonian(Hamiltonian):
    """Mølmer–Sørensen gate hamiltonian (in the spins interaction picture).

    :param spin1: Spin 1.
    :param spin2: Spin 2.
    :param vibration: Spins common vibration mode.
    :param field_blue: Blue-shifted field parameters.
    :param field_red: Red-shifted field parameters.
    :param spin_field: Spins-fields interaction parameters (use if they are all the same).
    :param spin1_blue_field: Spin1-blue shifted field interaction parameters (`spin_field` by default).
    :param spin1_red_field: Spin1-red shifted field interaction parameters (`spin1_blue_field` by default).
    :param spin2_blue_field: Spin2-blue shifted field interaction parameters (`spin1_blue_field` by default).
    :param spin2_red_field: Spin2-red shifted field interaction parameters (`spin1_red_field` by default).
    """

    def __init__(
            self,
            spin1: Spin,
            spin2: Spin,
            vibration: Oscillator,
            field_blue: SpinFieldInteractionHamiltonian.FieldParameters,
            field_red: SpinFieldInteractionHamiltonian.FieldParameters,
            spin_field: SpinFieldInteractionHamiltonian.InteractionParameters = None,
            spin1_blue_field: SpinFieldInteractionHamiltonian.InteractionParameters = None,
            spin1_red_field: SpinFieldInteractionHamiltonian.InteractionParameters = None,
            spin2_blue_field: SpinFieldInteractionHamiltonian.InteractionParameters = None,
            spin2_red_field: SpinFieldInteractionHamiltonian.InteractionParameters = None,
    ):
        super(MSGateHamiltonian, self).__init__([spin1, spin2, vibration])

        if spin1_blue_field is None:
            spin1_blue_field = spin_field
        if spin1_red_field is None:
            spin1_red_field = spin1_blue_field
        if spin2_blue_field is None:
            spin2_blue_field = spin1_blue_field
        if spin2_red_field is None:
            spin2_red_field = spin1_red_field

        # Spin 1 interacting blue field
        self.add(SpinFieldInteractionHamiltonian(
            spin=spin1, vibration=vibration, field=field_blue, interaction=spin1_blue_field
        ))

        # Spin 1 interacting red field
        self.add(SpinFieldInteractionHamiltonian(
            spin=spin1, vibration=vibration, field=field_red, interaction=spin1_red_field
        ))

        # Spin 2 interacting blue field
        self.add(SpinFieldInteractionHamiltonian(
            spin=spin2, vibration=vibration, field=field_blue, interaction=spin2_blue_field
        ))

        # Spin 2 interacting red field
        self.add(SpinFieldInteractionHamiltonian(
            spin=spin2, vibration=vibration, field=field_red, interaction=spin2_red_field
        ))

        self.use_h0([vibration])
