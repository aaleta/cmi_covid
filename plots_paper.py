import math
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from scipy.special import expit

COLORSG = {'0-9': '#9b0b1d', '10-19': '#b9470f', '20-29': '#f96802', '30-39': '#efc29e', '40-49': '#799da1',
          '50-59': '#007342', '60-69': '#05b092', '70-79': '#7eb958', '80+': '#c08f4e'}

COLORSS = {'0': '#9b0b1d', '1': '#b9470f', '2': '#f96802', '3': '#efc29e', '4': '#799da1',
          '5': '#007342', '6': '#05b092', '7': '#7eb958', '8': '#c08f4e'}

def fig1():
    data = pd.read_csv('data/cases_age_wave.csv')

    population = np.genfromtxt('data/age_groups_Spain.csv')
    data['0-9'] = data['0-9'] * 10000 / population[0]
    data['10-19'] = data['10-19'] * 10000 / population[1]
    data['20-29'] = data['20-29'] * 10000 / population[2]
    data['30-39'] = data['30-39'] * 10000 / population[3]
    data['40-49'] = data['40-49'] * 10000 / population[4]
    data['50-59'] = data['50-59'] * 10000 / population[5]
    data['60-69'] = data['60-69'] * 10000 / population[6]
    data['70-79'] = data['70-79'] * 10000 / population[7]
    data['80+'] = data['80+'] * 10000 / population[8]

    data = data.drop('wave', axis=1)
    data['fecha'] = pd.to_datetime(data['fecha'])

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in data.columns[1:]:
        ax.plot(data['fecha'], data[col], label=col, linewidth=1, c=COLORSG[col])

    ax.set_ylim([0, 70])
    ax.set_xlim([pd.to_datetime('2020-01-01'), pd.to_datetime('2022-04-01')])

    dates = ['2020-06-21', '2020-12-06', '2021-03-14', '2021-06-19', '2021-10-13', '2022-03-27']
    labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5', 'Wave 6']
    days = [120, 110, 75, 75, 85, 100]
    for i, date in enumerate(dates):
        ax.axvline(pd.Timestamp(date), linestyle='--')
        ax.text(pd.Timestamp(date) - pd.Timedelta(days=days[i]), ax.get_ylim()[1] * 0.95, labels[i])

    # add
    ax.set_ylabel('Incidence per 10,000 inhabitants')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.subplots_adjust(left=0.05)

    # print the plot
    plt.savefig('plots/fig1.pdf')


def fig2():
    results = pickle.load(open('results/results_simulation.pkl', 'rb'))

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 14})

    # Contact matrix
    contact_matrix = results['settings']['data']['contact_matrix']

    c1 = ax.imshow(contact_matrix, vmin=0, vmax=10, cmap='YlGnBu')

    # Axis
    ax.set_title('Age-mixing matrix (Spain)')
    ax.set_xticks(list(range(9)))
    ax.set_xticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)
    ax.set_yticks(list(range(9)))
    ax.set_yticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)

    # Color bar options
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(c1, cax=cax)
    cbar.set_label('average number of contacts')

    # print the plot
    plt.savefig('plots/fig2.pdf')


def fig3(results, results_gt, plotname):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.05)

    # Incidence curves
    data = results['data'].reset_index(drop=True)

    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    # Plot each line with the corresponding color from COLORSG
    lines = []
    for age_group in age_groups:
        line, = axs[0, 0].plot(data[age_group] * 10000, c=COLORSG[age_group], label=age_group)
        lines.append(line)

    # Set labels and legend
    axs[0, 0].set_xlabel("Timestep")
    axs[0, 0].set_ylabel('Incidence per 10,000 inhabitants')
    axs[0, 0].legend(lines, age_groups)

    # Contact matrix
    contact_matrix = results['settings']['data']['contact_matrix']
    np.fill_diagonal(contact_matrix, np.nan)

    vmax = 3 if plotname == 'fig3' else 0.3
    c1 = axs[0, 1].imshow(contact_matrix, vmin=0, vmax=vmax, cmap='YlGnBu')

    # Axis
    axs[0, 1].set_title('Non-diagonal contacts')
    axs[0, 1].set_xticks(list(range(9)))
    axs[0, 1].set_xticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)
    axs[0, 1].set_yticks(list(range(9)))
    axs[0, 1].set_yticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)

    # Color bar options
    cax = fig.add_axes([axs[0, 1].get_position().x1 + 0.01,
                        axs[0, 1].get_position().y0 + 0.01,
                        0.02,
                        axs[0, 1].get_position().height])
    cbar = fig.colorbar(c1, cax=cax)
    cbar.set_label('average number of contacts')

    # TE without GT
    transfer = results['results']
    transfer = np.clip(transfer, 0, None)

    print(f'max fig3 bot-left = {np.max(transfer)}')
    c2 = axs[1, 0].imshow(transfer, vmin=0, vmax=0.4, cmap='YlGnBu')

    # Axis
    axs[1, 0].set_title('Transfer entropy on raw incidence')
    axs[1, 0].set_xticks(list(range(9)))
    axs[1, 0].set_xticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)
    axs[1, 0].set_yticks(list(range(9)))
    axs[1, 0].set_yticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)

    # Color bar options
    cax = fig.add_axes([axs[1, 0].get_position().x1 + 0.01,
                        axs[1, 0].get_position().y0,
                        0.02,
                        axs[1, 0].get_position().height])
    cbar = fig.colorbar(c2, cax=cax)
    cbar.set_label('TE')
    cbar.remove()

    # TE with GT
    transfer = results_gt['results']
    transfer = np.clip(transfer, 0, None)

    print(f'max fig3 bot-right = {np.max(transfer)}')
    c3 = axs[1, 1].imshow(transfer, vmin=0, vmax=0.4, cmap='YlGnBu')

    # Axis
    axs[1, 1].set_title('Transfer entropy with GT aggregation')
    axs[1, 1].set_xticks(list(range(9)))
    axs[1, 1].set_xticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)
    axs[1, 1].set_yticks(list(range(9)))
    axs[1, 1].set_yticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)

    # Color bar options
    cax = fig.add_axes([axs[1, 1].get_position().x1 + 0.01,
                        axs[1, 1].get_position().y0 - 0.009,
                        0.02,
                        axs[1, 1].get_position().height])
    cbar = fig.colorbar(c3, cax=cax)
    cbar.set_label('transfer entropy')

    # Change font sizes
    axs[0, 0].xaxis.label.set_fontsize(16)
    axs[0, 0].yaxis.label.set_fontsize(16)
    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Padding
    fig.subplots_adjust(hspace=0.3)

    # Labels
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axs[0, 0].text(-0.2, 1.1, '(a)', transform=axs[0, 0].transAxes + trans, verticalalignment='top')
    axs[0, 1].text(-0.2, 1.1, '(b)', transform=axs[0, 1].transAxes + trans, verticalalignment='top')
    axs[1, 0].text(-0.2, 1.1, '(c)', transform=axs[1, 0].transAxes + trans, verticalalignment='top')
    axs[1, 1].text(-0.2, 1.1, '(d)', transform=axs[1, 1].transAxes + trans, verticalalignment='top')

    plt.savefig(f'plots/{plotname}.pdf')


def fig4(waves, incidences, plotname):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.rcParams.update({'font.size': 14})

    for idx in range(len(waves)):
        age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

        # Incidence plot
        lines = []
        for age_group in age_groups:
            # Apply the rolling mean for each column (age group)
            incidences[idx][age_group] = incidences[idx][age_group].rolling(14, min_periods=1).mean()

            # Plot each line with the corresponding color from COLORSG
            line, = axs[0, idx].plot(incidences[idx][age_group] * 10000, c=COLORSG[age_group], label=age_group)
            lines.append(line)

        # Set title and labels
        axs[0, idx].set_title(f'Wave {idx + 2}')

        # Matrices
        transfer = waves[idx]
        transfer = np.clip(transfer, 0, None)

        print(f'max fig4 wave {idx + 1}= {np.max(transfer)}')
        c = axs[1, idx].imshow(transfer, vmin=0, vmax=0.32, cmap='YlGnBu')

        # Axis
        axs[1, idx].set_xticks(list(range(9)))
        axs[1, idx].set_xticklabels(
            ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
            rotation=30)
        axs[1, idx].set_yticks(list(range(9)))
        axs[1, idx].set_yticklabels(
            ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
            rotation=30)
        axs[1, idx].grid(which='minor', color='w', linestyle='-', linewidth=2)

        # Color bar options
        cax = fig.add_axes([axs[1, idx].get_position().x1 + 0.01,
                            axs[1, idx].get_position().y0,
                            0.02,
                            axs[1, idx].get_position().height])
        cbar = fig.colorbar(c, cax=cax)
        cbar.set_label('transfer entropy')
        if idx < (len(waves) - 1):
            cbar.remove()
        if idx > 0:
            axs[1, idx].set_yticklabels([])

    axs[0, 0].set_ylabel('Incidence per 10,000 inhabitants')
    axs[0, 4].legend(lines,
                     ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
                     loc='center left', bbox_to_anchor=(1, 0.5))

    start = ['2020-06-21', '2020-12-06', '2021-03-14', '2021-06-19', '2021-10-13']
    for idx, incidence in enumerate(incidences):
        ticks = [0, len(incidence) / 3, len(incidence) * 2 / 3, len(incidence)]
        axs[0, idx].set_xticks(ticks)
        dates = [pd.to_datetime(start[idx]) + pd.Timedelta(days=x) for x in ticks]
        axs[0, idx].set_xticklabels([date.strftime('%d/%m/%y') for date in dates])

    plt.savefig(f'plots/{plotname}.pdf', bbox_inches='tight')


def sigmoid_interpolation(x1, x2, y1, y2, n_points=100):
    t = np.linspace(-6, 6, n_points)

    x_values = np.linspace(x1, x2, n_points)
    y_values = y1 + (y2 - y1) * expit(t)
    return x_values, y_values


def fig5():
    drivers_df, drivens_df = pd.DataFrame(), pd.DataFrame()
    for wave in range(2, 7):
        matrix = pickle.load(open(f'results/results_wave{wave}_gt.pkl', 'rb'))['results']
        matrix = np.clip(matrix, 0, None)

        driver = np.sum(matrix, axis=1)
        position = driver.argsort()

        data = pd.DataFrame({'x': wave,
                             'y': list(range(len(driver))),
                             'value': driver[position],
                             'group': [str(p) for p in position]})

        drivers_df = pd.concat([drivers_df, data], ignore_index=True)

        driven = np.sum(matrix, axis=0)
        position = driven.argsort()

        data = pd.DataFrame({'x': wave,
                             'y': list(range(len(driven))),
                             'value': driven[position],
                             'group': [str(p) for p in position]})

        drivens_df = pd.concat([drivens_df, data], ignore_index=True)

    drivers_df = drivers_df[drivers_df['value'] > 0]
    drivens_df = drivens_df[drivens_df['value'] > 0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.03, hspace=0.03)

    colors = {'0': '#9b0b1d', '1': '#b9470f', '2': '#f96802', '3': '#efc29e', '4': '#799da1',
              '5': '#007342', '6': '#05b092', '7': '#7eb958', '8': '#c08f4e'}
    labels = {'0': '0-9', '1': '10-19', '2': '20-29', '3': '30-39', '4': '40-49',
              '5': '50-59', '6': '60-69', '7': '70-79', '8': '80+'}

    added_to_legend_drivers = set()
    added_to_legend_drivens = set()

    for group, data in drivers_df.groupby('group'):
        previous_x, previous_y = None, None
        for _, row in data.iterrows():
            x, y = row['x'], row['y']
            if group not in added_to_legend_drivers:
                label = labels[group]
                added_to_legend_drivers.add(group)
            else:
                label = None
            axs[0].scatter(x, y, c=colors[group], s=np.log(1 + row['value']) * 200, label=label)
            if previous_x is not None and previous_y is not None:
                x_sigmoid, y_sigmoid = sigmoid_interpolation(previous_x, x, previous_y, y)
                axs[0].plot(x_sigmoid, y_sigmoid, c=colors[group])
            previous_x, previous_y = x, y

    axs[0].set_ylim([-0.5, 8.5])
    axs[0].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    axs[0].set_yticklabels(['Lowest', '', '', '', '', '', '', '', 'Highest'])
    axs[0].set_xticks([2, 3, 4, 5, 6])
    axs[0].set_xticklabels([2, 3, 4, 5, 6])
    axs[0].set_xlabel('Wave')
    axs[0].set_ylabel('Ranking')
    axs[0].set_title('Drivers')
    # axs[0].legend()

    for group, data in drivens_df.groupby('group'):
        previous_x, previous_y = None, None
        for _, row in data.iterrows():
            x, y = row['x'], row['y']
            if group not in added_to_legend_drivens:
                label = labels[group]
                added_to_legend_drivens.add(group)
            else:
                label = None
            axs[1].scatter(x, y, c=colors[group], s=np.log(1 + row['value']) * 200, label=label)
            if previous_x is not None and previous_y is not None:
                x_sigmoid, y_sigmoid = sigmoid_interpolation(previous_x, x, previous_y, y)
                axs[1].plot(x_sigmoid, y_sigmoid, c=colors[group])
            previous_x, previous_y = x, y

    axs[1].set_ylim([-0.5, 8.5])
    axs[1].set_yticklabels([])
    axs[1].set_xticks([2, 3, 4, 5, 6])
    axs[1].set_xticklabels([2, 3, 4, 5, 6])
    axs[1].set_xlabel('Wave')
    axs[1].set_title('Drivens')

    legend = axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for legend_handle in legend.legend_handles:
        legend_handle.set_sizes([50])

    plt.savefig(f'plots/fig5.pdf', bbox_inches='tight')


def figSEIR(results, plotname):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1, top=0.90, bottom=0.05)

    # Incidence curves
    data = results['data'].reset_index(drop=True)

    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    # Plot each line with the corresponding color from COLORSG
    lines = []
    for age_group in age_groups:
        line, = axs[0].plot(data[age_group] * 10000, c=COLORSG[age_group], label=age_group)
        lines.append(line)

    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel('Incidence per 10,000 inhabitants')
    axs[0].legend(lines,
                     ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])


    # TE without GT
    transfer = results['results']
    transfer = np.clip(transfer, 0, None)

    print(f'max figSEIR bot-right = {np.max(transfer)}')
    c1 = axs[1].imshow(transfer, vmin=0, vmax=0.6, cmap='YlGnBu')

    # Axis
    axs[1].set_title('Transfer entropy with GT aggregation')
    axs[1].set_xticks(list(range(9)))
    axs[1].set_xticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)
    axs[1].set_yticks(list(range(9)))
    axs[1].set_yticklabels(
        ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'], rotation=30)

    # Color bar options
    cax = fig.add_axes([axs[1].get_position().x1 + 0.01,
                        axs[1].get_position().y0 - 0.009,
                        0.02,
                        axs[1].get_position().height])
    cbar = fig.colorbar(c1, cax=cax)
    cbar.set_label('transfer entropy')

    # Change font sizes
    axs[0].xaxis.label.set_fontsize(16)
    axs[0].yaxis.label.set_fontsize(16)
    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Padding
    fig.subplots_adjust(hspace=0.3)

    # Labels
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axs[0].text(-0.2, 1.1, '(a)', transform=axs[0].transAxes + trans, verticalalignment='top')
    axs[1].text(-0.2, 1.1, '(b)', transform=axs[1].transAxes + trans, verticalalignment='top')

    plt.savefig(f'plots/{plotname}.pdf')

def total_te(matrices, n):
    for idx, matrix in enumerate(matrices):
        print(f'Total TE Wave {idx + 2} = {matrix.sum()/math.log2(n)}')


if __name__ == "__main__":
    # Static data
    fig1()
    fig2()

    # Fig3
    results = pickle.load(open('results/results_simulation.pkl', 'rb'))
    results_gt = pickle.load(open('results/results_simulation_gt.pkl', 'rb'))

    fig3(results, results_gt, 'fig3')

    # Fig4
    waves = [pickle.load(open(f'results/results_wave{wave}_gt.pkl', 'rb'))['results']
             for wave in range(2, 7)]
    incidences = [
        pickle.load(open(f'results/results_wave{wave}_gt.pkl', 'rb'))['data'].reset_index(drop=True)
        for wave in range(2, 7)]

    fig4(waves, incidences, 'fig4')

    # Fig5
    fig5()

    # SM
    results = pickle.load(open('results/results_simulation_null.pkl', 'rb'))
    results_gt = pickle.load(open('results/results_simulation_null_gt.pkl', 'rb'))

    fig3(results, results_gt, 'figS3')

    waves = [pickle.load(open(f'results/results_wave{wave}.pkl', 'rb'))['results']
             for wave in range(2, 7)]
    incidences = [pickle.load(open(f'results/results_wave{wave}.pkl', 'rb'))['data'].reset_index(drop=True)
                  for wave in range(2, 7)]

    fig4(waves, incidences, 'figS2')

    # Measure total TE
    matrices = []
    for wave in range(2, 7):
        matrix = pickle.load(open(f'results/results_wave{wave}.pkl', 'rb'))['results']
        matrices.append(np.clip(matrix, 0, None))

    print('Micro level')
    total_te(matrices, 21)

    matrices = []
    for wave in range(2, 7):
        matrix = pickle.load(open(f'results/results_wave{wave}_gt.pkl', 'rb'))['results']
        matrices.append(np.clip(matrix, 0, None))

    print('Macro level')
    total_te(matrices, 3)

    # FigSEIR
    # print('SEIR model')
    # results = pickle.load(open('results/results_seir_gt.pkl', 'rb'))

    # figSEIR(results, 'figSEIR')

