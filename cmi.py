import numpy as np
import pandas as pd
from idtxl.estimators_jidt import JidtKraskovCMI
from utils import get_generation_time, lag_data, extract_target


def estimate_TE(source: np.ndarray, target: np.ndarray, conditional: np.ndarray = None) -> float:
    """
        Estimate the transfer entropy using the JidtKraskovCMI algorithm.

        Parameters:
            source: numpy array with the source timeseries
            target: numpy array with the target timeseries
            conditional (optional): numpy array with the conditional timeseries

        Returns:
            estimated transfer entropy
    """
    return JidtKraskovCMI().estimate(source, target, conditional)


def null_distribution_TE(data: pd.DataFrame, w_gt: np.array, col_t: str, col_s: str, max_steps: int,
                         bootstrap_samples: int, lower_confidence: float,
                         upper_confidence: float) -> dict:
    """
        Computes the mean and confidence intervals of the TE between source and target using
    bootstrap.

        Parameters:
            data: complete dataframe with time in rows and age-groups in columns
            w_gt: GT distribution or None
            col_t: name of the target column
            col_s: name of the source column
            max_steps: max backawards steps in the GT distribution
            bootstrap_samples: number of bootstrap steps
            lower_confidence: lower value of the confidence interval
            upper_confidence: upper value of the confidence interval

        Returns:
            dictionary with lower_confidence, mean, and upper_confidence
    """

    target = extract_target(data, col_t, max_steps)
    conditional = lag_data(data, col_t, max_steps, w_gt)
    source = data.loc[:, [col_s]].copy()

    samples = []
    for _ in range(bootstrap_samples):
        shuffled_source = lag_data(source, col_s, max_steps, w_gt)
        np.random.shuffle(shuffled_source)

        samples.append(estimate_TE(shuffled_source, target, conditional))

    statistics = {
        'lower_confidence': np.quantile(samples, lower_confidence),
        'mean': np.mean(samples),
        'upper_confidence': np.quantile(samples, upper_confidence)
    }
    return statistics


def estimate_TE_matrix(data, config: dict) -> np.ndarray:
    """
        Computes the significant transfer entropy for all combinations of age-groups, conditioned
        to the target's past.

        Parameters:
            data: dataframe with one column per age-group and one row per timestep
            config: dictionary with the settings for the null distribution estimation

        Returns:
            matrix with the transfer entropy between age-groups
    """
    age_groups = list(data.columns)
    n_age_groups = len(age_groups)

    # Get GT distribution
    if config['gt_type']:
        _, w_gt = get_generation_time(config['gt_type'], config['gt_parameters'],
                                      config['max_steps'])
    else:
        w_gt = None

    transfer_entropy_matrix = np.zeros((n_age_groups, n_age_groups))
    for (i, j), _ in np.ndenumerate(transfer_entropy_matrix):

        print(f'\tComputing {i} -> {j}')
        col_s, col_t = age_groups[i], age_groups[j]

        target = extract_target(data, col_t, config['max_steps'])
        conditional = lag_data(data, col_t, config['max_steps'], w_gt)
        source = lag_data(data, col_s, config['max_steps'], w_gt)

        transfer_entropy = estimate_TE(source=source,
                                       target=target,
                                       conditional=conditional)

        null = null_distribution_TE(data=data, w_gt=w_gt, col_t=col_t, col_s=col_s,
                                    max_steps=config['max_steps'],
                                    bootstrap_samples=config['bootstrap_samples'],
                                    lower_confidence=config['lower_confidence'],
                                    upper_confidence=config['upper_confidence'])

        if (transfer_entropy < null['lower_confidence'])\
                or (transfer_entropy > null['upper_confidence']):
            transfer_entropy_matrix[i][j] = transfer_entropy

    return transfer_entropy_matrix
