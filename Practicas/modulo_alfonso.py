import numpy as np

def añade_bollinger(data, avg_size, std_size, k_b=2, k_l=2):

    mv_avg = data["close"].rolling(avg_size).mean()
    mv_std = data["close"].rolling(std_size).std()
    up_band = mv_avg + k_b * mv_std
    low_band = mv_avg - k_l * mv_std
    data["upper_band"] = up_band
    data["lower_band"] = low_band
    data = data.dropna()
    return data


def añade_cruces(data):
    '''    Añade una columna a data con los cruces con las bandas de Bolllinger
    '''

    upper_cross = np.diff(data.loc[:, "close"] > data.loc[:, "upper_band"],
                          prepend=False)
    lower_cross = np.diff(data.loc[:, "close"] < data.loc[:, "lower_band"],
                          prepend=False)
    data["upper_cross"] = upper_cross
    data["lower_cross"] = lower_cross
    return data