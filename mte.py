import pickle
import numpy as np
import concurrent.futures
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from utils import get_n_groups


def build_settings(args) -> dict:
    """Buils the settings dictionary

        Parameters:
            args: argsparser object containing all the necessary values
    """
    settings = {
        'cmi_estimator': args.estimator,
        'max_lag_sources': args.maxlag_source,
        'min_lag_sources': args.minlag_source,
        'max_lag_target': args.maxlag_target
    }

    return settings


def analyze_single_target(settings: dict, data: Data, target_id: int):
    """Analyze one single target so that it can be paralellized.

        Parameters:
            settings: dictionary containing the settings for the mte calculation
            data: array with shape (timeseries, age-groups)
            target_id: target age_group

        Returns:
            idtxl object with the results
    """
    results = MultivariateTE().analyse_single_target(settings, data, target_id)
    return results


def write_results(wave: int, analysis: dict) -> None:
    """Write the analysis dict into a pickle file.

        Parameters:
            wave: wave analyzed
            analysis: analysis dictionary with all the results
    """
    settings = analysis['settings']
    filename = f'results/results_wave{wave}_{settings["cmi_estimator"]}' \
               f'_maxLag{settings["max_lag_sources"]}' \
               f'_minLag{settings["min_lag_sources"]}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(analysis, f)


def run_cmi_process(settings: dict, data: np.ndarray, max_workers: int = 100):
    """Run the analysis on multiple cores"""

    analysis = {'settings': settings,
                'data': data,
                'results': []}

    data = Data(data, dim_order='sp')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_target,
                                   settings=settings,
                                   data=data,
                                   target_id=age_group):
                       age_group for age_group in range(get_n_groups(data))}

        for future in concurrent.futures.as_completed(futures):
            age_group = futures[future]
            analysis['results'].append({'age_group': age_group,
                                        'result': future.result()})

    return analysis
