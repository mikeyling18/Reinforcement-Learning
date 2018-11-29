import pandas as pd
import ManualStrategy as ms
import BestStrategy as bs
import datetime as dt
from util import get_prices


def main():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    symbols = ['JPM']
    start_val = 100000
    benchmark_days = get_prices(symbols, pd.date_range(start_date, end_date))

    ms.useManualStrategy(benchmark_days, symbols, start_date, end_date, start_val)
    bs.useBestStrategy(benchmark_days, symbols, start_date, end_date, start_val)

if __name__ == "__main__":
    main()

