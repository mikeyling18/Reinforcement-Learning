from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
API_KEY = 'GFTVBJ9OZBU5IXA1'

def get_portfolio_value(df_prices, trades, start_val, commission, impact):

    df = trades
    dates = pd.date_range(min(df['Date']), max(df['Date']))
    symbols = df.iloc[0]['Symbol']

    # Initialize Cash column
    df_prices['Cash'] = 1

    # Fill forwards, then fill back :)
    df_prices.fillna(method='ffill')
    df_prices.fillna(method='bfill')

    # make df_trades a copy of df_prices with same index and columns
    df_trades = pd.DataFrame(index=df_prices.index, columns=df_prices.columns)

    # make sure df_trades is in chronological order from top to bottom
    df_trades.sort_index()

    # initialize everything that is NOT a zero to zero
    df_trades[df_trades != 0] = 0

    # day 1 portfolio value is the start value
    df_trades.iloc[0]['Cash'] = start_val

    # iterate through each row in the 'df' dataframe...
    for index, row in df.iterrows():

        num_shares = row['Shares']  # get the number of shares involved in the sell or buy
        share = row['Symbol']  # get the symbol involved in the sell or buy
        date = row['Date']  # get the date of the sell or buy order
        order_type = row['Order']  # get the order type ('BUY' or 'SELL')

        # if the order type is a sell...
        if order_type == 'SELL':
            # the transaction incurs a 0.5% fee of the exit price...so the new price is 99.5% of the original
            price =  float(df_prices.loc[date][share] * (1 - impact))

            # the number of shares of that company is decreased by the number of shares in the order
            df_trades.loc[date][share] = df_trades.loc[date][share] - num_shares

            # and the amount of available cash increases by the price * number of shares being sold
            df_trades.loc[date]['Cash'] = df_trades.loc[date]['Cash'] + (price * num_shares)
        # if the order type is a buy...
        elif order_type == 'BUY':
            # the transaction incurs a 0.5% fee of the entrance price...so the new price is 100.5% of the original
            price = float(df_prices.loc[date][share] * (1 + impact))

            # the number of shares of that company is increased by the number of shares in the order
            df_trades.loc[date][share] = df_trades.loc[date][share] + num_shares

            # and the amount of available cash decreases by the price * number of shares being purchased
            df_trades.loc[date]['Cash'] = df_trades.loc[date]['Cash'] - (price * num_shares)

        # finally, a flat fee of $(commission) is applied to each sell and buy order...
        df_trades.loc[date]['Cash'] -= commission
    # the final values "holdings" from the video is the cumulaive sum of the number trades * prices
    values = df_trades.cumsum() * df_prices

    # get the sum of the columns to get the values data frame in the correct format
    return pd.DataFrame(values.sum(axis=1))


def get_stats(portvals):
    daily_ret = portvals.copy()
    daily_ret[1:] = (daily_ret[1:] / daily_ret[:-1].values) - 1
    daily_ret = daily_ret[1:]

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    adr = daily_ret.mean().values[0]
    sddr = daily_ret.std().values[0]

    # calculate Sharpe Ratio
    sf = 252.0
    rfr = 0.0
    sr = pow(sf, .5) * (daily_ret - rfr).mean() / sddr

    return adr, sddr, float(sr), float(cr)


def get_prices(symbols, date_range, addMarket=True):
    # Get closing prices for all symbols
    all_prices = pd.DataFrame(index=date_range)
    # if addMarket and 'SPY' not in symbols:
    #     symbols = ['SPY'] + symbols
    for sym in symbols:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=sym, outputsize='full')
        data.index = pd.to_datetime(data.index)
        df = data.reindex(date_range)
        y = df['4. close'].to_frame(name=sym)
        all_prices = all_prices.join(y)
        all_prices.dropna(inplace=True)
    return all_prices