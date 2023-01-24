import numpy as np


def read_sir_data(country: str = 'Italy') -> (np.ndarray, np.ndarray):
    """
    Read the matrix and population structure of the country.

        Parameters:
            country: name of the country, currently only Italy is supported

        Results:
            tuple containing the contact matrix and the population per age group
    """

    contact_matrix = np.genfromtxt(f'data/contact_matrix_{country}.csv', delimiter=',')
    population = np.genfromtxt(f'data/age_groups_{country}.csv', delimiter=',')

    return contact_matrix, population


def create_teen_susceptibility_vector(value: float, population: np.ndarray) -> np.ndarray:
    """
    Create a susceptibility vector with different susceptibility for people under 20 y.o.

        Parameters:
            value: susceptibility
            population: array containing the population in each age group

        Returns:
            numpy array with the same size as age groups
    """
    susceptibility = np.ones(len(population))
    susceptibility[0:2] *= value
    return susceptibility


def get_R0(matrix: np.ndarray, beta: float, gamma: float) -> float:
    """
    Calculate the value of R0 for the given matrix.
        Parameters:
            matrix: matrix with susceptibility already included
            beta: infectivity parameter
            gamma: recovery parameter

        Returns:
            The value of R0
    """
    eigenvalue = max(np.linalg.eigvals(matrix)).real
    R0 = beta * eigenvalue / gamma
    return R0


def sir_simulation(contact_matrix, population, beta, gamma_inverse,
                   susceptibility, initial_infected_age, seed,
                   timesteps):
    """
    Deterministic simulation of a Susceptible-Infected-Recovered (SIR) model with age-contract
    matrices. Code adapted from mobs-lab/mixing-patterns.

        Parameters:
            contact_matrix: average number of contacts between an individual in age group i and j
            population: array with the number of individuals in each age-group
            beta: infectivity
            gamma_inverse: days to recover
            susceptibility: susceptibility of individuals younger than 20 y.o.
            initial_infected_age: seeded age-group
            seed: fraction of infected individuals
            timesteps: maximum number of timesteps

        Returns:
            Array containing states and incidence as a function of time
    """

    # input
    gamma = 1. / gamma_inverse
    num_agebrackets = len(population)

    # initial conditions
    susceptible, infected, recovered = 0, 1, 2
    states = np.zeros((3, num_agebrackets, timesteps + 1))
    incidence = np.zeros((num_agebrackets, timesteps))
    susceptibility_vector = create_teen_susceptibility_vector(susceptibility, population)
    effective_contact_matrix = contact_matrix * susceptibility_vector

    states[infected][initial_infected_age][0] = population[initial_infected_age] * seed
    for a in range(num_agebrackets):
        states[susceptible][a][0] = population[a] - states[infected][a][0]

    for t in range(timesteps):
        for i in range(num_agebrackets):
            for j in range(num_agebrackets):
                incidence[i][t] += beta * effective_contact_matrix[i][j] \
                                   * states[susceptible][i][t] \
                                   * states[infected][j][t] / population[j]

            states[susceptible][i][t + 1] = states[susceptible][i][t] - incidence[i][t]
            states[infected][i][t + 1] = states[infected][i][t] \
                                         + incidence[i][t] \
                                         - gamma * states[infected][i][t]
            states[recovered][i][t + 1] = states[recovered][i][t] + gamma * states[infected][i][t]

    return states, incidence
