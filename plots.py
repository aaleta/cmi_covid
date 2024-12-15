import numpy as np
import matplotlib.pyplot as plt


def plot_curves_matrix(data: np.ndarray, matrix: np.ndarray, title=None) -> plt.Axes:
    """
    Creates a graph with the curves for the new cases and the age-interaction matrix.

        Parameters:
            data: numpy ndarray with timesteps in rows and age groups in columns.
            matrix: numpy ndarray with the contact matrix used in the simulation.
            title: title for the plot

        Returns:
            A plt axis
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    c = ax[1].matshow(matrix)
    for (i, j), z in np.ndenumerate(matrix):
        ax[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    x0, y0, w, h = ax[1].get_position().bounds
    cax = fig.add_axes([x0 + w + 0.01, y0, 0.02, h])
    cbar = fig.colorbar(c, cax=cax)
    cbar.set_label('contacts')

    lines = ax[0].plot(data)
    ax[0].set_xlabel("timestep")
    ax[0].set_ylabel('normalized incidence')
    ax[0].legend(lines, [age for age in range(len(lines))])

    if title:
        fig.suptitle(title)

    return fig, ax


def plot_TE_matrix(analysis: dict) -> plt.Axes:
    """
    Creates the transfer entropy matrix based on the data in an analysis

        Parameters:
            analysis: dictionary containing the results

        Returns:
            The matplolib variables necessary for plotting
    """
    transfer = analysis['results']
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
