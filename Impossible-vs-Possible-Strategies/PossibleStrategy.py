# Author: Mikey Ling
# mling7 - GTID: 903278121

import matplotlib.pyplot as plt
from indicators import get_bollinger_bands, get_rolling_std, get_rolling_mean
from util import get_portfolio_value, get_stats
import datetime as dt
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def usePossibleStrategy(df_prices, symbols, sd, ed, start_val):

    orders = []
    short_Xs = []
    long_Xs = []
    trade_num = 1000
    shares_min = -1000
    shares_max = 1000

    # rm = get_rolling_mean(df_prices['SPY'], window=20)
    # rstd = get_rolling_std(df_prices['SPY'], window=20)
    #
    # market = get_bollinger_bands(rm, rstd)
    # market['SPY'] = df_prices['SPY']
    # market = market.fillna(method='bfill')
    # market = market.fillna(method='ffill')

    # Get the Rolling Mean
    rm = get_rolling_mean(df_prices['JPM'], window=10)

    # Get the Rolling Standard Deviation
    rstd = get_rolling_std(df_prices['JPM'], window=10)
    bband = get_bollinger_bands(rm, rstd)
    bband[symbols]=df_prices[symbols]
    bband = bband.fillna(method = 'bfill')
    current_number_of_shares = 0

    for i in range(1, bband.shape[0]):
        if (bband.iloc[i - 1][symbols] > bband.iloc[i - 1]['Upper'])[0] and (bband.iloc[i][symbols] < bband.iloc[i]['Upper'])[0] and current_number_of_shares - trade_num >= shares_min:
            short_Xs.append([bband.index[i]])
            orders.append([bband.index[i], symbols, 'SELL', trade_num])
            current_number_of_shares -= trade_num
        elif (bband.iloc[i - 1][symbols] < bband.iloc[i - 1]['Lower'])[0] and (bband.iloc[i][symbols] > bband.iloc[i]['Lower'])[0] and current_number_of_shares + trade_num <= shares_max:
            long_Xs.append([bband.index[i]])
            orders.append([bband.index[i], symbols, 'BUY', trade_num])
            current_number_of_shares += trade_num

    orders.append([ed-dt.timedelta(days=1), symbols, 'BUY', 0])
    df_trades = pd.DataFrame(orders, columns=['Date', 'Symbol', 'Order', 'Shares'])
    df_shortXs = pd.DataFrame(short_Xs, columns=['Shorts'])
    df_longXs = pd.DataFrame(long_Xs, columns=['Longs'])

    df_benchmark_orders = pd.DataFrame([[min(df_prices.index), symbols, 'BUY', 1000],
                                        [max(df_prices.index), symbols, 'BUY', 0]],
                                        columns=['Date', 'Symbol', 'Order', 'Shares'])
    df_ms_value = get_portfolio_value(df_prices, df_trades, start_val, commission=9.95, impact=0.005)
    df_benchmark_value = get_portfolio_value(df_prices, df_benchmark_orders, start_val, commission=9.95, impact=0.005)


    fig, ax = plt.subplots()
    ax.set_title('Manual vs Benchmark Strategy', fontsize=20)
    ax.plot(df_ms_value.index, df_ms_value / df_ms_value.ix[0], color='black', label='Manual Strategy')
    ax.plot(df_benchmark_value.index, df_benchmark_value / df_benchmark_value.ix[0], color='blue', label='Benchmark Strategy')
    ax.legend(loc='best')
    plt.xticks(rotation=45)
    for i in range(0, df_shortXs.shape[0]):
        plt.axvline(x=pd.to_datetime(df_shortXs.iloc[i]['Shorts']), color ='red')
    for i in range(0, df_longXs.shape[0]):
        plt.axvline(x=pd.to_datetime(df_longXs.iloc[i]['Longs']), color ='green')
    plt.show()

    avg_daily_ret, std_daily_ret, sharpe_ratio, cum_ret = get_stats(df_ms_value)

    # Comparative Analysis Stuff
    print('Manual Strategy Performance Data; ')
    print('Cumulative Return of Fund: {}'.format(cum_ret))
    print('Standard Deviation of Fund: {}'.format(std_daily_ret))
    print('Average Daily Return of Fund: {}\n'.format(avg_daily_ret))

    return df_trades, df_shortXs, df_longXs, bband

