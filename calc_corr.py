from collections import OrderedDict
from coronavirus_analyzer import CoronavirusAnalyzer
from huiyan_crawler import HuiyanCrawler
import numpy as np
import pandas as pd
import datetime


def calc_corr(last_date, consider_population=False):
    h5_file_path = HuiyanCrawler.get_df_move_inc_corr_path(consider_population)
    try:
        df = pd.read_hdf(h5_file_path, 'huiyan')
        print(df)
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
        analyzer = CoronavirusAnalyzer(str_date)
        for shift in range(11):
            for n in range(1, 11):
                s_corr = analyzer.get_move_in_injured_corr(n=n, shift=shift, consider_population=consider_population)
                for region, val in zip(s_corr.index, s_corr.values):
                    if not np.isnan(val) and (region not in data[str_date] or data[str_date][region]['corr'] < val):
                        data[str_date][region] = {'shift': shift, 'window': n, 'corr': val}
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
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
        if df.size == 0:
            updated = False
    if updated:
        df.to_hdf(h5_file_path, 'huiyan')
        excel_file_path = HuiyanCrawler.get_df_move_inc_corr_path(consider_population, 'xlsx')
        df.to_excel(excel_file_path)
        print('数据更新完毕')
    else:
        print('数据无需更新')


if __name__ == '__main__':
    last_date = datetime.date(2020, 2, 12)
    calc_corr(last_date)
