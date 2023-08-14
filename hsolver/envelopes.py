"""
Module for working with time envelopes.

Author: Boris Bantysh
E-mail: bbantysh60000@gmail.com
License: GPL-3.0
"""
from warnings import warn
from abc import ABC, abstractmethod
from typing import Optional, Callable, List
from fractions import Fraction

import numpy as np
from scipy.stats import norm


class Envelope(ABC):
    """Abstract envelope class.

    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    :param period: Envelope period (None for non-periodic envelope, 0. for constant envelope).
    """

    def __init__(
            self,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None,
            period: Optional[float] = None
    ):
        self.time_start = time_start
        self.time_stop = time_stop
        self.period = period

    def is_in_bounds(self, t: float) -> bool:
        """Checks if time point is in envelope acting bounds.

        :param t: Time point.
        :return: True if the time value is inside envelope bounds.
        """
        if self.time_start is not None and t < self.time_start:
            return False
        if self.time_stop is not None and t > self.time_stop:
            return False
        return True

    def __call__(self, t: float) -> complex:
        """Returns the envelope value at specific time point.

        :param t: Time point.
        :return: Envelope value.
        """
        return self._get_value(t) if self.is_in_bounds(t) else 0.

    def drv(self, t: float) -> complex:
        """Returns the envelope derivative at specific time point.

        :param t: Time point.
        :return: Derivative value.
        """
        return self._get_derivative(t) if self.is_in_bounds(t) else 0.

    @abstractmethod
    def _get_value(self, t: float) -> complex:
        pass

    @abstractmethod
    def _get_derivative(self, t: float) -> complex:
        pass

    def conj(self):
        """Returns the complex conjugate envelope"""
        return ConjugateEnvelope(self)

    def to_string(self) -> str:
        return "E(t)"


class ConjugateEnvelope(Envelope):
    """Complex conjugate envelope.

    :param parent_envelope: Envelope to conjugate.
    """

    def __init__(self, parent_envelope: Envelope):
        super().__init__(
            time_start=parent_envelope.time_start,
            time_stop=parent_envelope.time_stop,
            period=parent_envelope.period
        )

        self.__parent_envelope = parent_envelope

    def _get_value(self, t: float) -> complex:
        parent_value = self.__parent_envelope._get_value(t)
        if isinstance(parent_value, float):
            return parent_value
        else:
            return parent_value.conjugate()

    def _get_derivative(self, t: float) -> complex:
        parent_value = self.__parent_envelope._get_derivative(t)
        if isinstance(parent_value, float):
            return parent_value
        else:
            return parent_value.conjugate()

    def conj(self):
        return self.__parent_envelope

    def to_string(self) -> str:
        return f"conj[{self.__parent_envelope.to_string()}]"


class ConstantEnvelope(Envelope):
    """Constant value envelope.

    :param value: Envelope value.
    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    """

    def __init__(self, value: complex, time_start: float = None, time_stop: float = None):
        super().__init__(time_start, time_stop, 0.)
        self.value = value

    def _get_value(self, t: float) -> complex:
        return self.value

    def _get_derivative(self, t: float) -> complex:
        return 0.

    def smoothen(self, front_width: float):
        """Returns the smooth version of the envelope, when it starts and ends smoothly.

        :param front_width: The front width of the envelope.
        :return: New envelope with smooth bounds.
        """
        return PulseEnvelope(front_width, self.value, self.time_start, self.time_stop)

    def to_string(self) -> str:
        return str(self.value)


class CustomEnvelope(Envelope):
    """Custom envelope

    :param fun: Function handler to calculate the envelope value at specific time point.
    :param fun_derivative: Function handler to calculate the envelope time derivative value at specific time point.
    :param fun_dt: Delta time to compute the envelope derivative using the finite difference.
    Is used when `fun_derivative` is None.
    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    :param period: Envelope period (None for non-periodic envelope, 0. for constant envelope).
    """

    def __init__(
            self,
            fun: Callable[[float], complex],
            fun_derivative: Callable[[float], complex] = None,
            fun_dt: float = None,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None,
            period: Optional[float] = None
    ):
        super().__init__(time_start, time_stop, period)
        self.fun = fun
        if fun_derivative is None:
            fun_derivative = lambda t: (fun(t + fun_dt) - fun(t)) / fun_dt
        self.fun_derivative = fun_derivative

    def _get_value(self, t: float) -> complex:
        return self.fun(t)

    def _get_derivative(self, t: float) -> complex:
        return self.fun_derivative(t)


class ProductEnvelope(Envelope):
    """The product of multiple envelopes.

    :param multipliers: Product multipliers.
    :param time_start: Time the envelope is starting (None to compute from the multipliers).
    :param time_stop: Time the envelope is stopping (None to compute from the multipliers).
    :param period: Envelope period (None to compute from the multipliers).
    """

    def __init__(
            self,
            multipliers: List[Envelope],
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None,
            period: Optional[float] = None
    ):
        if time_start is None:
            time_start_list = [envelope.time_start for envelope in multipliers if envelope.time_start is not None]
            time_start = None if len(time_start_list) == 0 else max(time_start_list)
        if time_stop is None:
            time_stop_list = [envelope.time_stop for envelope in multipliers if envelope.time_stop is not None]
            time_stop = None if len(time_stop_list) == 0 else min(time_stop_list)
        if period is None:
            period = get_common_period(multipliers)
        super().__init__(time_start, time_stop, period)
        self.multipliers = multipliers

    def _get_value(self, t: float) -> complex:
        return complex(np.prod([envelope(t) for envelope in self.multipliers], dtype=complex))

    def _get_derivative(self, t: float) -> complex:
        drv = complex(0.)
        for idx, envelope in enumerate(self.multipliers):
            m1 = complex(np.prod([e(t) for e in self.multipliers[:idx] + self.multipliers[idx + 1:]], dtype=complex))
            drv += m1 * envelope.drv(t)
        return drv

    def to_string(self) -> str:
        return "·".join([envelope.to_string() for envelope in self.multipliers])


class SumEnvelope(Envelope):
    """The sum of multiple envelopes.

    :param terms: Sum terms.
    :param time_start: Time the envelope is starting (None to compute from the terms).
    :param time_stop: Time the envelope is stopping (None to compute from the terms).
    :param period: Envelope period (None to compute from the terms).
    """

    def __init__(
            self,
            terms: List[Envelope],
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None,
            period: Optional[float] = None
    ):
        if time_start is None:
            time_start_list = [envelope.time_start for envelope in terms]
            time_start = None if None in time_start_list else min(time_start_list)
        if time_stop is None:
            time_stop_list = [envelope.time_stop for envelope in terms]
            time_stop = None if None in time_stop_list else max(time_stop_list)
        super().__init__(time_start, time_stop, period)

        self.terms = terms

    def _get_value(self, t: float) -> complex:
        return sum(envelope(t) for envelope in self.terms)

    def _get_derivative(self, t: float) -> complex:
        return sum(envelope.drv(t) for envelope in self.terms)

    def conj(self):
        return SumEnvelope(
            terms=[term.conj() for term in self.terms],
            time_start=self.time_start,
            time_stop=self.time_stop,
            period=self.period
        )

    def expand(self):
        """Expands the envelope to it terms.

        :return: List of terms.
        """
        return self.terms

    def to_string(self) -> str:
        return "+".join([envelope.to_string() for envelope in self.terms])


class PulseEnvelope(SumEnvelope):
    """Smooth pulse envelope.

    :param front_width: The pulse front width.
    :param amplitude: The pulse amplitude.
    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    """

    def __init__(
            self,
            front_width: float,
            amplitude: complex = 1.,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None
    ):
        super().__init__(self.get_terms(front_width, amplitude, time_start, time_stop))

    @staticmethod
    def get_terms(
            front_width: float,
            amplitude: complex = 1.,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None
    ):
        """Compute the terms of a smooth pulse envelope.

        :param front_width: The pulse front width.
        :param amplitude: The pulse amplitude.
        :param time_start: Time the envelope is starting (None for -infinity).
        :param time_stop: Time the envelope is stopping (None for +infinity).
        :return: List of envelope terms.
        """
        if time_start is not None and time_stop is not None:
            assert time_stop - time_start > 10 * front_width, "Pulse length must be larger than the 10 widths"

        terms = []

        if time_start is not None and front_width > 0.:
            terms.append(CustomEnvelope(
                fun=lambda t: amplitude * norm.cdf(t, loc=time_start, scale=front_width),
                fun_derivative=lambda t: amplitude * norm.pdf(t, loc=time_start, scale=front_width),
                time_start=time_start - 5 * front_width,
                time_stop=time_start + 5 * front_width
            ))

        terms.append(ConstantEnvelope(
            value=amplitude,
            time_start=None if time_start is None else time_start + 5 * front_width,
            time_stop=None if time_stop is None else time_stop - 5 * front_width
        ))

        if time_stop is not None and front_width > 0.:
            terms.append( CustomEnvelope(
                fun=lambda t: amplitude * (1. - norm.cdf(t, loc=time_stop, scale=front_width)),
                fun_derivative=lambda t: -amplitude * norm.pdf(t, loc=time_stop, scale=front_width),
                time_start=time_stop - 5 * front_width,
                time_stop=time_stop + 5 * front_width
            ))

        return terms


class PeriodicEnvelope(Envelope):
    """Periodic envelope: `A * exp(-1j*w*t + p) + b`.

    :param amplitude: Envelope amplitude `A`.
    :param frequency: Envelope frequency `w`.
    :param phase: Envelope phase `p`.
    :param shift: Envelope shift `b`.
    :param time_start: Time the envelope is starting (None for -infinity).
    :param time_stop: Time the envelope is stopping (None for +infinity).
    """

    def __init__(
            self,
            amplitude: float,
            frequency: float,
            phase: Optional[float] = 0.,
            shift: Optional[float] = 0.,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None
    ):
        period = 2 * np.pi / frequency
        if time_start is not None and time_stop is not None and period > time_stop - time_start:
            period = None

        super().__init__(time_start=time_start, time_stop=time_stop, period=period)

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shift = shift

    def __new__(
            cls,
            amplitude: float,
            frequency: float,
            phase: Optional[float] = 0.,
            shift: Optional[float] = 0.,
            time_start: Optional[float] = None,
            time_stop: Optional[float] = None
    ):
        if frequency == 0:
            value = shift + amplitude * (np.cos(phase) + 1j * np.sin(phase))
            return ConstantEnvelope(value, time_start, time_stop)
        return super().__new__(cls)

    def _get_value(self, t: float) -> complex:
        phase = self.frequency * t - self.phase
        return self.shift + self.amplitude * (np.cos(phase) - 1j * np.sin(phase))

    def _get_derivative(self, t: float) -> complex:
        phase = self.frequency * t - self.phase
        return self.amplitude * self.frequency * (-np.sin(phase) - 1j * np.cos(phase))

    def to_window(self, time_start: Optional[float] = None, time_stop: Optional[float] = None):
        """Crops the periodic envelope to specific window.

        :param time_start: Time the envelope is starting (None for -infinity).
        :param time_stop: Time the envelope is stopping (None for +infinity).
        :return: New instance of the envelope with specified window.
        """
        return self.__class__(
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=self.phase,
            shift=self.shift,
            time_start=time_start,
            time_stop=time_stop
        )

    def smoothen(self, front_width: float):
        """Returns the smooth version of the envelope, when it starts and ends smoothly.

        :param front_width: The front width of the envelope.
        :return: New envelope with smooth bounds.
        """
        pulse_envelope = PulseEnvelope(front_width, time_start=self.time_start, time_stop=self.time_stop)
        return SumEnvelope([
            ProductEnvelope([self.to_window(term.time_start, term.time_stop), term])
            for term in pulse_envelope.terms
        ])

    def to_string(self) -> str:
        if self.frequency > 0.:
            if self.phase > 0.:
                string = f"-i·[{self.frequency}t-{self.phase}]"
            elif self.phase < 0.:
                string = f"-i·[{self.frequency}t+{-self.phase}]"
            else:
                string = f"-i·{self.frequency}t"
        else:
            if self.phase > 0.:
                string = f"i·[{-self.frequency}t+{self.phase}]"
            elif self.phase < 0.:
                string = f"i·[{-self.frequency}t-{self.phase}]"
            else:
                string = f"i·{-self.frequency}t"
        string = f"exp({string})"
        if self.amplitude != 1.:
            string = f"{self.amplitude}·{string}"
        if self.shift != 0.:
            string = f"[{self.shift}+{string}]"
        return string


def get_common_period(envelopes: List[Envelope]) -> Optional[float]:
    """Computes the common period of a number of envelopes.

    :param envelopes: List of envelopes.
    :return: Common period (None for non-periodic envelope, 0. for constant envelope).
    """
    periods = [envelope.period for envelope in envelopes]
    if None in periods:
        return None

    non_zero_periods = [period for period in periods if period > 0.]
    if len(non_zero_periods) == 0:
        return 0.

    period = non_zero_periods[0]

    if len(non_zero_periods) == 1:
        return period

    for period_j in non_zero_periods[1:]:
        q1, _ = Fraction.from_float(period_j / period).limit_denominator(1000).as_integer_ratio()
        period = q1 * period

    for envelope in envelopes:
        if envelope.time_start is not None:
            t0 = envelope.time_start
        elif envelope.time_stop is not None:
            t0 = envelope.time_stop - period
        else:
            t0 = 0.
        if abs(envelope(t0 + period) - envelope(t0)) > 1e-6:
            warn("Failed to approximate envelope period. Use non-periodic evolution.")
            return None
    return period
