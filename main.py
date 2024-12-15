from cmi import estimate_TE_matrix
from parser import parse_args
from utils import load_data, build_settings, write_results

if __name__ == "__main__":
    args = parse_args()

    settings = build_settings(args)

    data = load_data(settings)

    TE_matrix = estimate_TE_matrix(data, settings['config'])
    results = {'settings': settings,
               'data': data.tail(-settings['config']['max_steps']),
               'results': TE_matrix}

    write_results(results)
