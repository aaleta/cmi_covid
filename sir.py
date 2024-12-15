import numpy as np


def read_sir_data(country: str = 'Spain') -> (np.ndarray, np.ndarray):
    """
    Read the matrix and population structure of the country.

        Parameters:
            country: name of the country, currently only Spain is supported

        Results:
            tuple containing the contact matrix and the population per age group
    """

    contact_matrix = np.genfromtxt(f'data/contact_matrix_{country}.csv', delimiter=',')
    population = np.genfromtxt(f'data/age_groups_{country}.csv', delimiter=',')

    return contact_matrix, population


def create_null_contact_matrix(population: np.ndarray) -> np.ndarray:
    """
    Create the null contact matrix to remove age-age interactions.

        Parameters:
            population: array containing the number of individuals per age group

        Retuns:
            matrix: contact matrix
    """
    matrix = np.zeros((len(population), len(population)))
    for i in range(matrix.shape[0]):
        matrix[i] = population/sum(population)

    return matrix


def update_beta(contact_matrix: np.ndarray, null_matrix: np.ndarray, settings: dict) -> float:
    """
    Renormalize beta to match the number of contacts of the contacts extracted from the data.

        Parameters:
            contact_matrix: matrix extracted from the data
            null_matrix: null model for the interactions
            settings: dictionary containing all the simulation settings

        Returns:
            beta_null: updated value of beta
    """
    beta, gamma = settings['data']['beta'], 1./settings['data']['gamma_inverse']
    r0_data = get_R0(contact_matrix, beta, gamma)
    r0_null = get_R0(null_matrix, beta, gamma)
    beta_null = beta * r0_data / r0_null

    return beta_null


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
                   initial_infected_age, seed, timesteps):
    """
    Deterministic simulation of a Susceptible-Infected-Recovered (SIR) model with age-contract
    matrices. Code adapted from mobs-lab/mixing-patterns.

        Parameters:
            contact_matrix: average number of contacts between an individual in age group i and j
            population: array with the number of individuals in each age-group
            beta: infectivity
            gamma_inverse: days to recover
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
            states[infected][i][t + 1] = states[infected][i][t] \
                                         + incidence[i][t] \
                                         - gamma * states[infected][i][t]
            states[recovered][i][t + 1] = states[recovered][i][t] + gamma * states[infected][i][t]

    return states, incidence

