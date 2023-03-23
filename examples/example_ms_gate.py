import matplotlib.pyplot as plt

from hsolver.collections import Spin, Oscillator, SpinFieldInteractionHamiltonian, MSGateHamiltonian
from hsolver.solver import SystemEvolutionSolver


def main():
    spin1 = Spin(frequency=10.)
    spin2 = Spin(frequency=10.)
    vibration = Oscillator(dim=10, frequency=1.)

    hamiltonian = MSGateHamiltonian(
        spin1=spin1,
        spin2=spin2,
        vibration=vibration,
        field_blue=SpinFieldInteractionHamiltonian.FieldParameters(
            frequency=0.9 * vibration.frequency, time_start=100., time_stop=800., pulse_front_width=10
        ),
        field_red=SpinFieldInteractionHamiltonian.FieldParameters(
            frequency=-0.9 * vibration.frequency, time_start=100., time_stop=800., pulse_front_width=10
        ),
        spin_field=SpinFieldInteractionHamiltonian.InteractionParameters(
            rabi_frequency=0.1,
            ld_param=0.1
        )
    )

    solver = SystemEvolutionSolver(
        hamiltonian=hamiltonian,
        init_state=[spin1.basis_state(0), spin2.basis_state(0), vibration.basis_state(0)],
        max_step_size=1.,
        min_step_size=0.1
    )

    solver.solve(time_stop=1000.)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)

    population = solver.get_populations_evolution([spin1, spin2])
    labels = ["|00>", "|01>", "|10>", "|11>"]
    for label, level_population in zip(labels, population):
        ax1.plot(solver.time_list, level_population, label=label)
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel("Population")

    purity = solver.get_purity_evolution([spin1, spin2])
    ax2.plot(solver.time_list, purity)
    ax2.grid()
    ax2.set_ylabel("Purity")

    negativity = solver.get_entanglement_evolution([spin1], [spin2])
    ax3.plot(solver.time_list, negativity)
    ax3.grid()
    ax3.set_ylabel("Entanglement")

    plt.show()


if __name__ == "__main__":
    main()
