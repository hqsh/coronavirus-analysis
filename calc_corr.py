from collections import OrderedDict
from coronavirus_analyzer import CoronavirusAnalyzer
from huiyan_crawler import HuiyanCrawler
from util.util import Util
import numpy as np
import pandas as pd
import datetime


def calc_corr(last_date=None, n=3, consider_population=False, shift_one_day=False, sample_cnt=0):
    if last_date is None:
        last_date = datetime.date.today() - datetime.timedelta(days=1)
    elif isinstance(last_date, str):
        last_date = Util().str_date_to_date(last_date)
    h5_file_path = HuiyanCrawler.get_df_move_inc_corr_path(
        consider_population, n=n, shift_one_day=shift_one_day, sample_cnt=sample_cnt)
    try:
        df = pd.read_hdf(h5_file_path, 'huiyan')
        # print(df)
        y, m, d = df.index[-1].split('-')
        date = datetime.date(int(y), int(m), int(d)) + datetime.timedelta(days=1)
    except (FileNotFoundError, IndexError):
        df = None
        date = datetime.date(2020, 1, 17)
    data = OrderedDict()
    updated = False
    print('计算范围：{} ～ {}'.format(date, last_date))
    while date <= last_date:
        str_date = str(date)
        if str_date not in data:
            data[str_date] = {}
        analyzer = CoronavirusAnalyzer(str_date, first_date='2020-01-17')
        s_origin_corr = None
        for shift in range(11):
            for window in range(1, 11):
                s_corr = analyzer.get_move_in_injured_corr(
                    n=n, shift=shift, window=window, consider_population=consider_population,
                    shift_one_day=shift_one_day, sample_cnt=sample_cnt)
                if shift == 0 and window == 1:
                    s_origin_corr = s_corr
                for region, val in zip(s_corr.index, s_corr.values):
                    if not np.isnan(val) and (region not in data[str_date] or data[str_date][region]['corr'] < val):
                        data[str_date][region] = {'shift': shift, 'window': window, 'corr': val,
                                                  'corr_inc': val - s_origin_corr[region]}
        print('{} 处理完毕'.format(str_date))
        updated = True
        date += datetime.timedelta(days=1)
    if updated:
        dfs = []
        for date, info in data.items():
            _df = pd.DataFrame(data[date])
            _df.index = pd.MultiIndex.from_product([[date], _df.index])
            _df = _df.unstack(1)
            dfs.append(_df)
        _df = pd.concat(dfs)
        _df = _df.reindex(list(data.keys()))
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
        if df.size == 0:
            updated = False
    if updated:
        df.to_hdf(h5_file_path, 'huiyan')
        excel_file_path = HuiyanCrawler.get_df_move_inc_corr_path(
            consider_population, 'xlsx', n=n, shift_one_day=shift_one_day, sample_cnt=sample_cnt)
        df.to_excel(excel_file_path)
        print('数据更新完毕')
    else:
        print('数据无需更新')


if __name__ == '__main__':
    for n in range(3, 4):
        for sample_cnt in [0]:  # , 7, 14, 21, 28, 35
            last_date = datetime.date(2020, 2, 29)
            print('n = {}，sample_cnt = {}，处理开始'.format(n, sample_cnt))
            calc_corr(last_date, n=n, sample_cnt=sample_cnt)
            print('n = {}，sample_cnt = {}，处理完毕'.format(n, sample_cnt))
