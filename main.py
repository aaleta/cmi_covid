from mte import build_settings, run_cmi_process, write_results
from parser import parse_args
from utils import load_data

if __name__ == "__main__":
    args = parse_args()

    settings = build_settings(args)
    data = load_data(settings)

    a = run_cmi_process(settings, data)
    write_results(args.wave, a)



