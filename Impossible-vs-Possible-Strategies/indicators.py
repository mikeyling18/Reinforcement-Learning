'''
This file contains methods to calculate technical indicators that will be used to
create the best manual strategy
'''


import pandas as pd
import matplotlib.pyplot as plt
from util import get_prices


def get_rolling_mean(values, window):
    return pd.Series(values).rolling(window=window).mean()


def get_rolling_std(values, window):
    return pd.Series(values).rolling(window=window).std()


def get_momentum(values):
    return values/values.shift(1) - 1


def get_bollinger_bands(rm, rstd):
    upper_bollinger_band = rm + 2*rstd
    lower_bollinger_band = rm - 2*rstd
    bollinger_band = pd.DataFrame(columns=['Upper', 'Lower', 'SMA'])
    bollinger_band['Upper'] = upper_bollinger_band
    bollinger_band['Lower'] = lower_bollinger_band
    bollinger_band['SMA'] = rm
    return bollinger_band


def get_indicators(symbols, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    df_prices = get_prices(symbols,dates)

    # Get the Rolling Mean
    rm = get_rolling_mean(df_prices['JPM'], window = 20)

    # Get the Rolling Standard Deviation
    rstd = get_rolling_std(df_prices['JPM'], window = 20)

    # Get the Momentum
    mom = get_momentum(df_prices['JPM'])

    # Get the Upper and Lower Bollinger Bands
    df_bollinger_bands = get_bollinger_bands(rm, rstd)

    # Plot Raw Prices and Bollinger Bands
    plt.figure(1)
    plt.suptitle('Bollinger Bands', fontsize =20)
    plt.plot(df_prices.index, df_prices['JPM'], color = 'r', label = 'Raw Price')
    plt.plot(df_prices.index, df_bollinger_bands['Upper'], color = 'b', label='_nolegend_')
    plt.plot(df_prices.index, df_bollinger_bands['Lower'], color = 'b', label = 'Bollinger Bands')
    plt.legend(loc='best')
    plt.xticks(rotation=45)

    # Plot Raw Prices and Moving Average
    plt.figure(2)
    plt.suptitle('Moving Average', fontsize = 20)
    plt.plot(df_prices.index, df_prices['JPM'], color = 'r', label= 'Raw Price')
    plt.plot(df_prices.index, rm, color = 'b', label = 'Moving Average')
    plt.legend(loc='best')
    plt.xticks(rotation=90)

    # Plot Raw Price and Momentum

    f, axarr = plt.subplots(2, sharex = 'all')
    axarr[0].plot(df_prices.index, df_prices['JPM'], color = 'r')
    axarr[0].set_title('Raw Price')
    axarr[1].set_title('Momentum')
    axarr[1].plot(df_prices.index, mom, color = 'b')
    plt.suptitle('Raw Price and Momentum', fontsize = 20)
    plt.xticks(rotation=90)

    # plt.show(block=False)
    # plt.gcf().clear()

    return



