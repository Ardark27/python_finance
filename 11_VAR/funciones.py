import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def montecarlo(data, time_window='5y'):
    '''
    montecarlo(data, time_window='5y')
    data = dataframe with the log returns
    time_window = time window for the simulation default '5y'

    return data_montecarlo

    this function is used to obtain the montecarlo simulation of the
    data provided.

    '''
    date_init = data.iloc[-1].name + pd.Timedelta('1d')
    time_index = pd.date_range(
        start=date_init, end=date_init + pd.Timedelta(time_window), freq='B')
    mean_data = data.mean()
    std_data = data.std()
    data_montecarlo = pd.DataFrame(np.random.normal(
        loc=mean_data,
        scale=std_data,
        size=(len(time_index), 1000)),
        index=time_index)
    return data_montecarlo


def bootstrapping(data, time_window='5y'):
    '''
    bootstrapping(data, time_window='5y')
    data = dataframe with the price
    time_window = time window for the simulation default '5y'

    return data_boots

    this function is used to obtain the bootstrapping simulation of the
    data provided.
    '''
    date_init = data.iloc[-1].name + pd.Timedelta('1d')
    time_index = pd.date_range(start=date_init,
                               end=date_init+pd.Timedelta(time_window),
                               freq='B')
    data_boots = pd.DataFrame(index=time_index)

    for i in range(1000):
        data_index = np.random.randint(
            0, high=data.shape[0], size=len(time_index), dtype=int)
        data_boots_i = pd.DataFrame(
            data.iloc[data_index].values, index=time_index)
        data_boots = pd.concat([data_boots, data_boots_i], axis=1)

    return data_boots


def quantile_plot(data):
    '''
    quantile_plot(data)
    data = dataframe with the quantiles simulation

    returns the plot of the quantiles 0.05, 0.5, 0.95
    '''
    plt.fill_between(data.index,
                     data.iloc[:, 0],
                     data.iloc[:, 1],
                     color='gainsboro')
    plt.fill_between(data.index,
                     data.iloc[:, 1],
                     data.iloc[:, 2],
                     color='gainsboro')
    plt.plot(data.iloc[:, 0], label='0.05',
             linewidth=2, color='tab:blue')
    plt.plot(data.iloc[:, 1], label='0.5',
             linewidth=2, color='tab:orange')
    plt.plot(data.iloc[:, 2], label='0.95',
             linewidth=2, color='tab:green')
    plt.legend()
    plt.xlim(data.index[0], data.index[-1])
    return 0