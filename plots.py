import numpy as np
import matplotlib.pyplot as plt


def plot_curves_matrix(data: np.ndarray, susceptibility=1) -> plt.Axes:
    """
    Creates a graph with the curves for the new cases and the age-interaction matrix.

        Parameters:
            data: numpy ndarray with timesteps in rows and age groups in columns.
            susceptibility: value of the susceptibility for individuals < 20 y.o.

        Returns:
            A plt axis
    :return:
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Matrix
    matrix = np.genfromtxt('data/contact_matrix_Italy.csv', delimiter=',')
    susceptibility_vector = np.ones(matrix.shape[0])
    susceptibility_vector[0:2] = susceptibility
    matrix = (matrix.T * susceptibility_vector).T

    c = ax[1].matshow(matrix)
    for (i, j), z in np.ndenumerate(matrix):
        ax[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    cax = fig.add_axes([ax[1].get_position().x1 + 0.01,
                        ax[1].get_position().y0,
                        0.02,
                        ax[1].get_position().height])
    cbar = fig.colorbar(c, cax=cax)
    cbar.set_label('contacts')

    # Incidence
    population = np.genfromtxt('data/age_groups_Italy.csv')
    data = data / population

    lines = ax[0].plot(data)
    ax[0].set_xlabel("timestep")
    ax[0].set_ylabel('normalized incidence')
    ax[0].legend(lines, [age for age in range(len(lines))])

    return fig, ax


def plot_TE_matrix(analysis: dict) -> plt.Axes:
    """
    Creates the transfer entropy matrix based on the data in an analysis

        Parameters:
            analysis: dictionary containing the results

        Returns:
            The matplolib variables necessary for plotting
    """
    n_age_groups = len(analysis['results'])
    transfer = np.zeros((n_age_groups, n_age_groups))
    for result in analysis['results']:
        target = result['age_group']
        result = result['result']._single_target
        if len(result[target]['selected_vars_sources']):
            for source, te in zip(result[target]['selected_vars_sources'],
                                  result[target]['selected_sources_te']):
                transfer[target][source[0]] += te

    transfer = transfer.T
    fig, ax = plt.subplots()
    c = ax.matshow(transfer)
    for (i, j), z in np.ndenumerate(transfer):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(c, cax=cax)
    cbar.set_label('Transfer Entropy')
    return fig, ax

