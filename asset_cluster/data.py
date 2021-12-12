import pandas as pd
import FinanceDataReader as fdr

def load_data(stocks: list,
                  close: bool=True,
                  volume: bool=False,
                  start_date='2017-01-01',
                  end_date='2019-12-31'):
    '''

    :param stocks: (list) stock tickers list
    :param close: (bool)
    :param volume: (bool)
    :param start_date:
    :param end_date:
    :return: (None) save csv file
    '''

    if close:
        df_close = pd.DataFrame()
        _load_edata(stocks, start_date, end_date, df_close, 'Close')
        df_close.to_csv('Datasets/close.csv')

    if volume:
        df_volume = pd.DataFrame()
        _load_edata(stocks, start_date, end_date, df_volume, 'Volme')
        df_volume.to_csv('Datasets/volume.csv')


def _load_edata(list_, start_, end_, df, col):
    '''

    :param list_: stock ticker list
    :param start_: start date
    :param end_: end date
    :param df: empty dataframe
    :param col: 'Close' or 'Volume'
    :return:
    '''
    for t in list_:
        tmp = fdr.DataReader(t, start=start_, end=end_)
        df[t] = tmp[col]