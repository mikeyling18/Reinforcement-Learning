import pandas as pd
import matplotlib.pyplot as plt

from util import get_stats, get_portfolio_value


def useImpossibleStrategy(df_prices, symbols, sd, ed, start_val):

    # fill back in time, then fill forward
    df_prices = df_prices.fillna(method='bfill')
    df_prices = df_prices.fillna(method='ffill')

    # create an empty order set
    orders = []
    shares_max = 1000
    shares_min = -1000

    current_shares = 0

    for count, (date, row) in enumerate(df_prices.iloc[:-1].iterrows()):
        current_price = df_prices['JPM'][count]
        next_price = df_prices['JPM'][count+1]

        if (next_price > current_price) & (current_shares < shares_max):
            shares_to_buy = shares_max - current_shares
            orders.append([date, symbols, 'BUY', shares_to_buy])
            current_shares += shares_to_buy

        elif (next_price < current_price) & (current_shares > shares_min):
            shares_to_sell = current_shares - shares_min
            orders.append([date, symbols, 'SELL', shares_to_sell])
            current_shares -= shares_to_sell

    df_orders = pd.DataFrame(orders, columns=['Date', 'Symbol', 'Order', 'Shares'])



    df_benchmark_orders = pd.DataFrame([[min(df_prices.index), symbols, 'BUY', 1000],
                                        [max(df_prices.index), symbols, 'BUY',0]],
                                        columns=['Date','Symbol','Order','Shares'])

    df_benchmark_value = get_portfolio_value(df_prices, df_benchmark_orders, start_val, commission=9.95, impact=0.005)
    df_bps = get_portfolio_value(df_prices, df_orders, start_val, commission=0, impact=0)

    #
    # avg_daily_ret, std_daily_ret, sharpe_ratio, cum_ret = get_stats(df_benchmark_value)
    # print('Benchmark Stats')
    # print('Cumulative Return of Fund: {} '.format(cum_ret))
    # print('Standard Deviation of Fund: {} '.format(std_daily_ret))
    # print('Average Daily Return of Fund: {} \n'.format(avg_daily_ret))

    avg_daily_ret, std_daily_ret, sharpe_ratio, cum_ret = get_stats(df_bps)
    print('Best Possible Strategy Stats')
    print('Cumulative Return of Fund: {}'.format(cum_ret))
    print('Standard Deviation of Fund: {}'.format(std_daily_ret))
    print('Average Daily Return of Fund: {}\n'.format(avg_daily_ret))

    # Plot Benchmark Dataframe
    fig, ax = plt.subplots()
    ax.set_title('JPM Benchmark vs Best Possible Strategy', fontsize = 20)
    ax.plot(df_benchmark_value.index, df_benchmark_value/df_benchmark_value.ix[0], color='blue', label='SPY')
    ax.plot()
    ax.plot(df_bps.index, df_bps/df_bps.ix[0], color='black', label='Best Strategy')
    ax.legend(loc='best')
    plt.xticks(rotation=45)
    plt.show()

    return df_orders


