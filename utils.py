import numpy as np
import pandas as pd
from scipy.stats import gamma

from sir import read_sir_data, get_R0, sir_simulation


def get_generation_time(variant: str, max_days=20) -> tuple[list[int], list[float]]:
    """Returns the generation time of the given variant.

        Parameters:
            variant: ancestral, alpha, delta or omicron
            max_days: maximum number of days to return

        Returns:
            x, y: generation time pdf (y) on each day (x)
    """
    gt = {'ancestral': [1.87, 1 / 0.28],
          'alpha': [2.53, 2.83],
          'delta': [2.49, 2.63],
          'omicron': [2.39, 2.95]}

    shape, scale = gt[variant]

    x = list(range(max_days))
    y = gamma.pdf(x, a=shape, scale=scale)
    y = y / sum(y)

    return x, y


def load_wave(wave: int) -> np.ndarray:
    """Returns a data frame with the number of cases for a specific wave.

        Parameters:
            wave: number between 1 and 6

        Returns:
            data: an array of dimension (timestep, age-groups) ordered by day
    """
    data = pd.read_csv('data/cases_age_wave.csv')
    data = data.loc[data['wave'] == f'wave_{wave}']\
        .drop(['fecha', 'wave'], axis=1)\
        .to_numpy(dtype=int)

    return data


def get_n_groups(data: np.ndarray) -> int:
    """Returns the number of age-groups contained in the data.

        Parameters:
            data: an array of dimension (timestep, age-groups)

        Returns:
            n: number of groups
    """
    return np.shape(data)[1]


def load_data(settings: dict) -> np.ndarray:
    """
    Load the dataset described in args.

        Parameters:
            settings: dictionary with the settings of the analysis

        Returns:
            numpy array of size (time, n_age_groups) with the incidence
    """

    if settings['data']['name'] == 'sir':
        contact_matrix, population = read_sir_data()
        settings['data']['R0'] = get_R0(contact_matrix,
                                        settings['data']['beta'],
                                        1. / settings['data']['gamma_inverse'])

        states, incidence = sir_simulation(contact_matrix, population,
                                           settings['data']['beta'],
                                           settings['data']['gamma_inverse'],
                                           settings['data']['susceptibility'],
                                           settings['data']['initial_infected_age'],
                                           settings['data']['seed'],
                                           settings['data']['timesteps']
                                           )

        return incidence.T

    elif settings['data']['name'] == 'data':
        data = load_wave(settings['data']['wave'])
        return data
