import argparse
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

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
        action=argparse.BooleanOptionalAction
    )

    return parser.parse_args(args)

