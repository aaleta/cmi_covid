import argparse
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    # Simulation parameters
    parser.add_argument(
        "-s",
        "--simulation",
        action=argparse.BooleanOptionalAction,
        help="run a SIR model with input data"
    )

    parser.add_argument(
        "-se",
        "--seir",
        action=argparse.BooleanOptionalAction,
        help="run a SEIR model with input data"
    )

    parser.add_argument(
        "-n",
        "--null",
        action=argparse.BooleanOptionalAction,
        help="override the input data with a null matrix in the SIR model"
    )

    # Data
    parser.add_argument(
        "-w",
        "--wave",
        default=6,
        type=int,
        help="wave to study, should be between 1 and 6"
    )

    # Estimation configuration
    parser.add_argument(
        "-gt",
        "--generation-time",
        action=argparse.BooleanOptionalAction,
        help="flag to indicate if past data should be aggregated by GT"
    )

    parser.add_argument(
        "-mt",
        "--max-steps",
        default=10,
        type=int,
        help="number of steps used in the aproximation of the generation time"
    )

    parser.add_argument(
        "-b",
        "--bootstrap-samples",
        default=200,
        type=int,
        help="number of bootstrap samples for the confidence interval estimation"
    )

    parser.add_argument(
        "-l",
        "--lower-confidence",
        default=0.05,
        type=float,
        help="lower confidence interval"
    )

    parser.add_argument(
        "-u",
        "--upper-confidence",
        default=0.95,
        type=float,
        help="upper confidence interval"
    )

    return parser.parse_args(args)

