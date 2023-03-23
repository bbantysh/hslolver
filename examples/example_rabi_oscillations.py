import matplotlib.pyplot as plt

from hsolver.collections import Spin, Oscillator, SpinFieldInteractionHamiltonian
from hsolver.solver import SystemEvolutionSolver


def main():
    spin = Spin(frequency=5.)
    vibration = Oscillator(dim=20, frequency=3.)

    hamiltonian = SpinFieldInteractionHamiltonian(
        spin=spin,
        vibration=vibration,
        field=SpinFieldInteractionHamiltonian.FieldParameters(
            frequency=0.,
            pulse_front_width=0.1,
            time_start=3.,
            time_stop=10.
        ),
        interaction=SpinFieldInteractionHamiltonian.InteractionParameters(
            rabi_frequency=1.,
            ld_param=0.01,
        )
    ).disable_h0()

    solver = SystemEvolutionSolver(
        hamiltonian=hamiltonian,
        init_state=[spin.basis_state(0), vibration.basis_state(0)],
        max_step_size=0.01
    )

    solver.solve(time_stop=15.)

    t = solver.time_list
    population = solver.get_populations_evolution([spin])

    plt.figure()
    for level, level_population in enumerate(population):
        plt.plot(t, level_population, label=f"Level {level}")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
