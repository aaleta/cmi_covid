import pickle
import numpy as np
import pandas as pd
from scipy.stats import gamma, expon
from sir import read_sir_data, get_R0, sir_simulation, create_null_contact_matrix, \
    update_beta
from seir import seir_simulation


def build_settings(args) -> dict[str, str | int | float]:
    """
    Builds the settings dictionary
        Parameters:
            args: argsparser object containing all the necessary values
    """

    settings = {
        'config':
            {
                'max_steps': args.max_steps,
                'bootstrap_samples': args.bootstrap_samples,
                'lower_confidence': args.lower_confidence,
                'upper_confidence': args.upper_confidence,
                'gt_type': None
            }
    }

    if args.simulation:
        # noinspection PyTypedDict
        settings['data'] = {'name': 'sir', 'beta': 0.035, 'gamma_inverse': 5,
                            'initial_infected_age': 2,
                            'seed': 0.001, 'timesteps': 80}

        filename = 'results_simulation'
        if args.null:
            settings['data']['name'] += '_null'
            filename += '_null'
    elif args.seir:
        # noinspection PyTypedDict
        settings['data'] = {'name': 'seir', 'beta': 0.035 * 5 / 3, 'epsilon_inverse': 2,
                            'gamma_inverse': 3, 'initial_infected_age': 2,
                            'seed': 0.001, 'timesteps': 80}

        filename = 'results_seir'
        if args.null:
            raise Exception("Null SEIR model not Implemented")
    else:
        settings['data'] = {'name': 'data', 'wave': args.wave}
        filename = f'results_wave{args.wave}'

    if args.generation_time:
        settings['data']['name'] += '_gt'
        filename += '_gt'
        # Generation time options
        if args.simulation:
            settings['config']['gt_type'] = 'exponential'
            settings['config']['gt_parameters'] = [1.0 / settings['data']['gamma_inverse']]
        elif args.seir:
            settings['config']['gt_type'] = 'erlang'
            settings['config']['gt_parameters'] = [2,
                                                   1.0 / settings['data']['epsilon_inverse'] + 1.0 / settings['data']['gamma_inverse']]
        else:
            settings['config']['gt_type'] = 'ancestral'
            settings['config']['gt_parameters'] = [1.87, 1.0 / 0.27]

    settings['results_filename'] = f'results/{filename}.pkl'

    return settings


def get_generation_time(gt_type: str, parameters: list[float], max_days=20) -> tuple[list[int], list[float]]:
    """
    Returns the generation time of the variant.

        Parameters:
            gt_type: can be exponential or ancestral
            parameters: list of parameters for the GT distribution
            max_days: maximum number of days to return

        Returns:
            x, y: generation time pdf (y) on each day (x)
    """
    x = list(range(1, max_days + 1))

    if gt_type == 'exponential':
        y = expon.pdf(x, scale=parameters[0])
    elif gt_type == 'erlang':
        y = gamma.pdf(x, a=parameters[0], scale=parameters[1])
    else:
        y = gamma.pdf(x, a=parameters[0], scale=parameters[1])

    y = y / sum(y)

    return x, y


def lag_data(data: pd.DataFrame, age: str, max_backwards: int, w_gt: np.array = None) -> np.array:
    """
    Create dictionary with the lagged data.

        Parameters:
            data: dataframe where each row represents time and columns represent age-groups
            age: column that will be lagged
            max_backwards: maximum past information to retrieve
            w_gt: generation time function

        Returns:
            A dictionary with the past data for each age-group
    """

    age_array = pd.DataFrame({f'(t-{lag})': data[age].shift(lag)
                              for lag in range(1, max_backwards + 1)
                              }).to_numpy()
    # Remove the extra rows
    age_array = age_array[max_backwards:]

    # Apply GT
    array = np.dot(age_array, w_gt) if w_gt is not None else age_array

    # Ensure it is a 2D array
    return array.reshape(array.shape[0], -1)


def extract_target(data: pd.DataFrame, col_name: str, max_steps: int) -> np.array:
    """
    Extracts a single column, shapes it as a 2D array and removes padding.

        Parameters:
            data: dataframe with the information
            col_name: column to be extracted
            max_steps: padding to be removed

        Returns:
            numpy 2D array with the column data
    """

    target = data[col_name].to_numpy().reshape(len(data), -1)[max_steps:]
    return target


def extract_wave(data: pd.DataFrame, wave: int, max_backwards: int) -> pd.DataFrame:
    """
    Extract the incidence of a single wave + some initial padding

        Parameters:
            data: dataframe with time in rows and age in columns
            wave: wave number, betweem 1 & 6
            max_backwards: maximum initial padding

        Returns:
            dataframe corresponding to the wave plus its padding
    """

    if wave == 1:
        data_past = pd.DataFrame([[0] * len(data.columns)] * max_backwards, columns=data.columns)
    else:
        data_past = data.loc[data['wave'] == f'wave_{wave - 1}'].tail(max_backwards)

    data_wave = data.loc[data['wave'] == f'wave_{wave}']
    data_past = pd.concat([data_past, data_wave], ignore_index=True)\
        .drop(['fecha', 'wave'], axis=1)

    return data_past


def load_wave(wave: int, max_backwards: int) -> pd.DataFrame:
    """
    Returns a data frame with the number of cases for a specific wave.

        Parameters:
            wave: number between 1 and 6
            max_backwards: maximum past information to retrieve

        Returns:
            dataframe with time in rows and age groups in columns
    """
    data = pd.read_csv('data/cases_age_wave.csv')
    data = extract_wave(data, wave, max_backwards)

    return data


def load_data(settings: dict) -> pd.DataFrame:
    """
    Load the dataset described in args.

        Parameters:
            settings: dictionary with the settings of the analysis

        Returns:
             dataframe with time in rows and age groups in columns
    """

    if settings['data']['name'].startswith('sir') or settings['data']['name'].startswith('seir'):
        contact_matrix, population = read_sir_data()

        if settings['data']['name'].startswith('sir_null'):
            null_matrix = create_null_contact_matrix(population)
            settings['data']['beta'] = update_beta(contact_matrix, null_matrix, settings)
            contact_matrix = null_matrix

        settings['data']['contact_matrix'] = contact_matrix
        settings['data']['R0'] = get_R0(contact_matrix, settings['data']['beta'],
                                        1.0 / settings['data']['gamma_inverse'])

        if settings['data']['name'].startswith('sir'):
            states, incidence = sir_simulation(settings['data']['contact_matrix'],
                                               population,
                                               settings['data']['beta'],
                                               settings['data']['gamma_inverse'],
                                               settings['data']['initial_infected_age'],
                                               settings['data']['seed'],
                                               settings['data']['timesteps']
                                               )
        elif settings['data']['name'].startswith('seir'):
            states, incidence = seir_simulation(settings['data']['contact_matrix'],
                                                population,
                                                settings['data']['beta'],
                                                settings['data']['epsilon_inverse'],
                                                settings['data']['gamma_inverse'],
                                                settings['data']['initial_infected_age'],
                                                settings['data']['seed'],
                                                settings['data']['timesteps']
                                                )
        else:
            raise Exception("Unknown model!")


        data = pd.DataFrame(incidence.T,
                            columns=["0-9", "10-19", "20-29", "30-39", "40-49",
                                     "50-59", "60-69", "70-79", "80+"])
        data['fecha'] = 'none'
        data['wave'] = 'wave_1'
        data = extract_wave(data, 1, settings['config']['max_steps'])

        population = np.genfromtxt('data/age_groups_Spain.csv')
        data = data / population

    elif settings['data']['name'].startswith('data'):
        data = load_wave(settings['data']['wave'], settings['config']['max_steps'])

        population = np.genfromtxt('data/age_groups_Spain.csv')
        data = data / population

    return data


def write_results(results: dict) -> None:
    """
    Write the results dict into a pickle file.
        Parameters:
            results: analysis dictionary with all the results
    """
    filename = results['settings']['results_filename']
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
