import argparse
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--simulation",
        action=argparse.BooleanOptionalAction,
        help="use SIR simulation input data"
    )
    parser.add_argument(
        "--susceptibility",
        default=1,
        type=float,
        help="susceptibility for teenagers in the SIR model"
    )
    parser.add_argument(
        "-w",
        "--wave",
        default=6,
        type=int,
        help="wave to study, should be between 1 and 6"
    )
    parser.add_argument(
        "-e",
        "--estimator",
        default='JidtKraskovCMI',
        help="choose either JidtDiscreteCMI for discrete data, or JidtGaussianCMI/JidtKraskovCMI"
             "for continuous data"
    )
    parser.add_argument(
        "-mS",
        "--minlag-source",
        default=1,
        type=int,
        help="minimum lag for the source"
    )
    parser.add_argument(
        "-MS",
        "--maxlag-source",
        default=10,
        type=int,
        help="maximum lag for the source"
    )
    parser.add_argument(
        "-MT",
        "--maxlag-target",
        default=10,
        type=int,
        help="maximum lag for the target"
    )
    parser.add_argument(
        "-z",
        "--z-normalization",
        action=argparse.BooleanOptionalAction,
        help="flag to indicate if data should be z-normalized"
    )
    parser.add_argument(
        "-gt",
        "--generation-time",
        action=argparse.BooleanOptionalAction,
        help="flag to indicate if the aggregated data by GT should be used"
    )
    return parser.parse_args(args)

