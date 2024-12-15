import numpy as np
from sir import read_sir_data, create_null_contact_matrix, update_beta, get_R0


def seir_simulation(contact_matrix, population, beta, epsilon_inverse, gamma_inverse,
                   initial_infected_age, seed, timesteps):
    """
    Deterministic simulation of a Susceptible-Exposed-Infected-Recovered (SEIR) model with age-contract
    matrices. Code adapted from mobs-lab/mixing-patterns.

        Parameters:
            contact_matrix: average number of contacts between an individual in age group i and j
            population: array with the number of individuals in each age-group
            beta: infectivity
            epsilon_inverse: latency period
            gamma_inverse: days to recover
            initial_infected_age: seeded age-group
            seed: fraction of infected individuals
            timesteps: maximum number of timesteps

        Returns:
            Array containing states and incidence as a function of time
    """

    # input
    epsilon = 1. / epsilon_inverse
    gamma = 1. / gamma_inverse
    num_agebrackets = len(population)

    # initial conditions
    susceptible, exposed, infected, recovered = 0, 1, 2, 3
    states = np.zeros((4, num_agebrackets, timesteps + 1))
    incidence = np.zeros((num_agebrackets, timesteps))

    states[infected][initial_infected_age][0] = population[initial_infected_age] * seed
    for a in range(num_agebrackets):
        states[susceptible][a][0] = population[a] - states[infected][a][0]

    for t in range(timesteps):
        for i in range(num_agebrackets):
            for j in range(num_agebrackets):
                incidence[i][t] += beta * contact_matrix[i][j] \
                                   * states[susceptible][i][t] \
                                   * states[infected][j][t] / population[j]

            states[susceptible][i][t + 1] = states[susceptible][i][t] - incidence[i][t]
            states[exposed][i][t + 1] = states[exposed][i][t] + incidence[i][t] - epsilon * states[exposed][i][t]
            states[infected][i][t + 1] = states[infected][i][t] \
                                         + epsilon * states[exposed][i][t] \
                                         - gamma * states[infected][i][t]
            states[recovered][i][t + 1] = states[recovered][i][t] + gamma * states[infected][i][t]

    return states, incidence

