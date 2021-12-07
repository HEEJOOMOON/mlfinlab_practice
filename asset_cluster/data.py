import pandas as pd
import FinanceDataReader as fdr
from typing import Optional

def choose_stocks(stocks: Optional[list],
                  close: bool=True,
                  volume: bool=False,
                  start_date='2017-01-01',
                  end_date='2019-12-31'):

    if close:
        df_close = pd.DataFrame()
        data_close = load_data(stocks, start_date, end_date,
                               df_close, 'Close')
        data_close.to_csv('close.csv')

    if volume:
        df_volume = pd.DataFrame()
        data_volume = load_data(stocks, start_date, end_date,
                               df_volume, 'Volme')
        data_volume.to_csv('volume.csv')


def load_data(list_, start_, end_, idx_, df, col):
    for t in list_['Symbol'].iloc[idx_]:
        tmp = fdr.DataReader(t, start=start_, end=end_)
        df[t] = tmp[col]

    return df