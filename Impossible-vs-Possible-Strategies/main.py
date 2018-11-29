import pandas as pd
import PossibleStrategy as ps
import ImpossibleStrategy as imps
import datetime as dt
from util import get_prices


def main():
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    symbols = ['JPM']
    start_val = 100000
    benchmark_days = get_prices(symbols, pd.date_range(start_date, end_date))

    ps.usePossibleStrategy(benchmark_days, symbols, start_date, end_date, start_val)
    imps.useImpossibleStrategy(benchmark_days, symbols, start_date, end_date, start_val)

if __name__ == "__main__":
    main()

