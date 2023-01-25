import pickle
import numpy as np
import concurrent.futures
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.stats import network_fdr
from utils import get_n_groups


def build_settings(args) -> dict:
    """
    Builds the settings dictionary

        Parameters:
            args: argsparser object containing all the necessary values
    """
    settings = {
        'cmi_estimator': args.estimator,
        'max_lag_sources': args.maxlag_source,
        'min_lag_sources': args.minlag_source,
        'max_lag_target': args.maxlag_target,
        'z_normalization': args.z_normalization,
    }

    if args.simulation:
        settings['data'] = {'name': 'sir', 'beta': 0.035, 'gamma_inverse': 5,
                            'susceptibility': args.susceptibility, 'initial_infected_age': 2,
                            'seed': 0.01, 'timesteps': 80}
        settings['results_filename'] = f'results/results_simulation_sus{args.susceptibility}'
    else:
        settings['data'] = {'name': 'data', 'wave': args.wave}
        settings['results_filename'] = f'results/results_wave{args.wave}_'

    settings['results_filename'] += f'_{settings["cmi_estimator"]}' \
                                    f'_maxLag{settings["max_lag_sources"]}' \
                                    f'_minLag{settings["min_lag_sources"]}' \
                                    f'{"_normalized" if settings["z_normalization"] else ""}.pkl'

    return settings


def analyze_single_target(settings: dict, data: Data, target_id: int):
    """
    Analyze one single target so that it can be paralellized.

        Parameters:
            settings: dictionary containing the settings for the mte calculation
            data: array with shape (timeseries, age-groups)
            target_id: target age_group

        Returns:
            idtxl object with the results
    """
    results = MultivariateTE().analyse_single_target(settings, data, target_id)
    return results


def write_results(analysis: dict) -> None:
    """
    Write the analysis dict into a pickle file.

        Parameters:
            analysis: analysis dictionary with all the results
    """
    filename = analysis['settings']['results_filename']
    with open(filename, 'wb') as f:
        pickle.dump(analysis, f)


def run_cmi_process(settings: dict, data: np.ndarray, max_workers: int = 100):
    """Run the analysis on multiple cores"""

    analysis = {'settings': settings,
                'data': data,
                'results': []}

    n_groups = get_n_groups(data)
    data = Data(data, dim_order='sp', normalise=settings['z_normalization'])

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_target,
                                   settings=settings,
                                   data=data,
                                   target_id=age_group):
                       age_group for age_group in range(n_groups)}

        for future in concurrent.futures.as_completed(futures):
            age_group = futures[future]
            analysis['results'].append({'age_group': age_group,
                                        'result': future.result()})

    # Combine the results nad perform the FDR correction
    res_list = list(range(n_groups))
    for result in analysis['results']:
        res_list[result['age_group']] = result['result']

    res = network_fdr({'alpha_fdr': 0.05}, *res_list)
    analysis['combined_results'] = res

    return analysis

