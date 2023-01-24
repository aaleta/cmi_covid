from mte import run_cmi_process, write_results, build_settings
from parser import parse_args
from utils import load_wave


if __name__ == "__main__":
    args = parse_args()

    data = load_wave(args.wave, normalise=args.z_normalization)
    settings = build_settings(args)

    a = run_cmi_process(settings, data)
    write_results(args.wave, a)
