from collections import OrderedDict
from dxy_crawler import DxyCrawler
from dxy_daily_crawler import DxyDailyCrawler
from huiyan_crawler import HuiyanCrawler
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from util.util import Util, with_logger
from weather_crawler import WeatherCrawler
import numpy as np
import pandas as pd
# import xgboost as xgb
xgb = None
import datetime
import math


@with_logger
class CoronavirusAnalyzer:
    '''
    冠状病毒分析
    '''

    def __init__(self, last_date=None, is_predict=False, first_date=None, sample_cnt=0, in_english=False,
                 use_dxy_new_version_data=True):
        '''
        :param str or datetime.date last_date: 最后一天，对日频数据有效，用于模拟历史，会把最后一天数据用前一天数据填充，格式是 yyyy-mm-dd
        :param bool is_predict: 是否用于预测，如果是的话，在取数据时候会做 index 偏移 1 位的处理
        :param str or datetime.date last_date: 最早一天
        :param int sample_cnt: 计算相关性的最大样本数，值若 <= 0，代表从 first_date 起
        :param in_english: 是否使用英文输出
        :param use_dxy_new_version_data: 丁香园在 2 月底开始公布历史日频数据
        :return:
        '''
        self.__is_predict = is_predict
        self.__util = Util()
        self.__huiyan_crawler = HuiyanCrawler()
        self.__weather_crawler = WeatherCrawler()
        self.__region_to_english = self.__util.get_region_info('region', 'english')
        self.__region_to_chinese = {v: k for k, v in self.__region_to_english.items()}
        if last_date is None:
            last_date = datetime.date.today()
        self.__last_date = str(last_date)
        if first_date is None:
            self.__first_date = '2020-01-21' if is_predict else '2020-01-11'  # 有效数据的第一天
        else:
            self.__first_date = str(first_date)
        self.__df_move_inc_corr = self.get_df_move_inc_corr(sample_cnt=sample_cnt, in_english=in_english)
        self.__df_move_inc_corr_cp = self.get_df_move_inc_corr(True, sample_cnt=sample_cnt, in_english=in_english)
        self.__in_english = in_english
        self.__use_dxy_new_version_data = use_dxy_new_version_data
        self.__dxy_daily_crawler = DxyDailyCrawler()
        df = self.df_virus_daily_inc_injured
        no_inc_injured_regions = df.columns[df.iloc[-1] == 0]
        df = self.df_virus_daily_inc
        no_inc_regions = Util.get_multi_col_0(df)[(df.iloc[-1].values.reshape(-1, 4) != 0).sum(axis=1) == 0]
        if str(datetime.date.today()) == self.__last_date:
            self.logger.warning('在最后一天（{}），如下这些地区没有新增的确诊人数：{}，如下这些地区没有任何疫情数据变化：{}。'
                                '请确保这些地区已经公布了最后一天的数据（一般是后面一天上午公布），否则分析出来的结果可能不准确。'
                                .format(last_date, '、'.join(no_inc_injured_regions), '、'.join(no_inc_regions)))

    @property
    def plt(self):
        '''
        支持中文的 plt
        :return:
        '''
        return self.__util.plt

    def append_dates(self, df):
        '''
        对函数返回的 DataFrame 检查是否最后一天，并且如果最后一天是当天，需要清空
        :param df:
        :return:
        '''
        if self.__is_predict:
            if df.columns.size == 0:
                return df
            date = self.__util.str_date_to_date(df.index[-1])
            append_index = []
            date += datetime.timedelta(days=1)
            while date <= self.__util.str_date_to_date(self.__last_date):
                append_index.append(str(date))
                date += datetime.timedelta(days=1)
            if len(append_index) > 0:
                index = df.index.tolist()
                index += append_index
                df = df.reindex(index)
            else:
                df.iloc[-1] = np.nan
            df.fillna(method='pad', inplace=True)
        return df

    @property
    def df_virus(self):
        '''
        实时病毒感染数据
        :return:
        '''
        return DxyCrawler.load_dxy_data_frame('recent', 'h5').astype(np.int32)

    @property
    def df_virus_injured(self):
        '''
        实时病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus).astype(np.int32)

    @property
    def df_virus_daily(self):
        '''
        日频病毒感染数据
        :return:
        '''
        if self.__use_dxy_new_version_data:
            df = self.__dxy_daily_crawler.df_virus_daily
        else:
            df = DxyCrawler.load_dxy_data_frame('recent_daily', 'h5')
        if self.__last_date is not None:
            df = df.loc[self.__first_date: self.__last_date]
        df.fillna(method='pad', inplace=True)
        df = df.astype(np.int32)
        return self.append_dates(df)

    @property
    def df_virus_daily_injured(self):
        '''
        日频病毒感染数据中的确诊人数
        :return:
        '''
        df = self.get_injured(self.df_virus_daily).astype(np.int32)
        df = self.__data_frame_region_to_english_if_needed(df)
        return df

    @property
    def df_recent_daily_injured(self):
        '''
        最新日频病毒感染数据中的确诊人数
        :return:
        '''
        return self.df_virus_daily_injured.iloc[[-1], :]

    @property
    def df_virus_daily_inc(self):
        '''
        日频新增病毒感染数据
        :return:
        '''
        if self.__use_dxy_new_version_data:
            df = self.__dxy_daily_crawler.df_virus_daily_inc
        else:
            df = DxyCrawler.load_dxy_data_frame('recent_daily_inc', 'h5')
        if self.__last_date is not None:
            df = df.loc[self.__first_date: self.__last_date]
        df.fillna(0, inplace=True)
        return df.astype(np.int32)

    @property
    def df_virus_daily_inc_injured(self):
        '''
        日频新增病毒感染数据中的确诊人数
        :return:
        '''
        df = self.get_injured(self.df_virus_daily_inc)
        df = self.append_dates(df)
        return self.__data_frame_region_to_english_if_needed(df)

    def get_df_virus_n_days_inc_injured(self, n, shift_one_day=True):
        '''
        最近 n 天新增确诊人数（不含当天），2 级列索引
        :param n:
        :param shift_one_day: 是否忽略当天
        :return:
        '''
        assert n > 0
        df_daily = self.df_virus_daily_inc_injured
        df_values = df_daily.values
        values = np.zeros(shape=(df_daily.shape[0], df_daily.shape[1] * n), dtype=np.int32)
        for end_i in range(df_daily.shape[0]):
            start_i = end_i - n
            if start_i < 0:
                start_i = 0
            arr = df_values[start_i: end_i]
            need_add_row_cnt = n - arr.shape[0]
            if need_add_row_cnt > 0:
                add_arr = np.zeros(shape=(need_add_row_cnt, arr.shape[1]), dtype=np.int32)
                arr = np.vstack([add_arr, arr])
            arr = arr.T.reshape(-1)
            values[end_i, :] = arr
        cols = []
        if shift_one_day:
            for i in range(1, n + 1):
                cols.insert(0, '{}天前新增'.format(i))
        else:
            for i in range(0, n):
                if i == 0:
                    cols.insert(0, '当天新增')
                else:
                    cols.insert(0, '{}天前新增'.format(i))
        df_n_days = pd.DataFrame(values, index=df_daily.index, columns=pd.MultiIndex.from_product(
            [df_daily.columns, cols]))
        return df_n_days

    def get_df_virus_daily_inc_injured_cum_n(self, n, shift_one_day=True):
        '''
        shift_one_day 为 True，近 n 天累计新增确诊人数（不含当天的最近 n 天新增确诊人数和，即 1天前、2天前、......、n天前的确诊人数和）
        shift_one_day 为 False，近 n 天累计新增确诊人数（不含当天的最近 n 天新增确诊人数和，即 当天、1天前、2天前、......、n-1天前的确诊人数和）
        :param n:
        :param shift_one_day: 是否忽略当天
        :return:
        '''
        df_virus_daily_inc_injured = self.df_virus_daily_inc_injured
        arr = df_virus_daily_inc_injured.values
        arr_cum_n = np.zeros(shape=arr.shape, dtype=np.int32)
        for end_i in range(arr.shape[0]):
            _end_i = end_i
            if not shift_one_day:
                _end_i += 1
            start_i = 0 if _end_i < n else _end_i - n
            arr_cum_n[end_i, :] = arr[start_i: _end_i, :].sum(axis=0)
        return pd.DataFrame(arr_cum_n, index=df_virus_daily_inc_injured.index,
                            columns=df_virus_daily_inc_injured.columns).astype(np.int32)

    @staticmethod
    def get_injured(df):
        '''
        获取确诊人数，2 级列索引变成 1 级列索引
        :param df:
        :return:
        '''
        col_1_size = df.columns.levels[1].size
        try:
            regions = df.columns.levels[0][df.columns.codes[0][::col_1_size]]
        except AttributeError:
            regions = df.columns.levels[0][df.columns.labels[0][::col_1_size]]
        idx = 3 if col_1_size == 4 else 4
        df1 = pd.DataFrame(df.values[:, idx::col_1_size], index=df.index, columns=regions)

        # 测试正确性，待测试完后删除
        ss_injured = []
        for region in df.columns.levels[0]:
            s_injured = df[region]['确诊']
            s_injured.name = region
            ss_injured.append(s_injured)
        df2 = pd.DataFrame(ss_injured).T[regions]
        if ((df1 - df2).values != 0).sum() != 0 or ((df1.values - df2.values) != 0).sum() != 0:
            df1.to_csv('df_bak/exception/get_injured_1.csv')
            df1.to_excel('df_bak/exception/get_injured_1.xlsx')
            df2.to_csv('df_bak/exception/get_injured_2.csv')
            df2.to_excel('df_bak/exception/get_injured_2.xlsx')
            raise ValueError('get_injured 切片有问题')
        return df1

    def del_city_regions(self, df, selected_cols_1=None, del_special_regions=False, inplace=True,
                         omitted_provinces=None):
        '''
        删除 DataFrame 的普通城市，或特区列索引
        :param df:
        :param selected_cols_1: 保留的 1 级列索引（如果是多级列索引）
        :param del_special_regions: 是否删除特区
        :param inplace:
        :param omitted_provinces: 需要指定删除的省或直辖市
        :return:
        '''
        cols = ['香港', '澳门', '台湾'] if del_special_regions else []
        df_cols = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else df.columns
        for region in df_cols:
            if not self.__util.region_is_province(region):
                cols.append(region)
        if omitted_provinces is not None:
            assert isinstance(omitted_provinces, list)
            cols = cols + omitted_provinces
        return self.__util.del_cols(df, cols, selected_cols_1, inplace)

    def del_city_special_regions(self, df, selected_cols_1=None, inplace=True, omitted_provinces=None):
        return self.del_city_regions(df, selected_cols_1=selected_cols_1, del_special_regions=True, inplace=inplace,
                                     omitted_provinces=omitted_provinces)

    @property
    def df_distance(self):
        '''
        各地相互距离数据
        :return:
        '''
        return pd.read_csv('data/全国各地之间距离.csv', index_col=0)

    @property
    def df_region(self):
        '''
        各地信息数据
        :return:
        '''
        return pd.read_csv('data/全国各地信息.csv', index_col=0)

    @property
    def df_weather(self):
        '''
        各地天气数据
        :return:
        '''
        df = self.__weather_crawler.get_weather_df()
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df

    @property
    def df_weather_ma(self):
        '''
        滑动窗口加权平均日频率天气
        :return:
        '''
        return self.__weather_crawler.get_weather_ma_data()

    @property
    def df_weather_average(self):
        '''
        各地最近平均天气数据
        :return:
        '''
        return self.get_weather_average_data()

    def get_weather_average_data(self, start_shift=14, end_shift=3):
        '''
        获取天气的加权（权重值线性递增）平均数据，从地区有首例确诊病人往前找 start_shift 天，到 end_shift 天前，之间的数据取均值
        专家分析：从已有病例来看，这一新型冠状病毒的潜伏期平均7天左右，短的2到3天，长的12天
        :param start_shift: 默认取最大潜伏期的值
        :param end_shift: 默认取最小潜期的伏值
        :return:
        '''
        return self.__weather_crawler.get_weather_average_data(self.df_virus_daily, start_shift, end_shift)

    @property
    def df_curve_in(self):
        df = self.__huiyan_crawler.df_curve_in
        df = self.__data_frame_region_to_english_if_needed(df)
        return df.loc[self.__first_date: self.__last_date]

    def get_df_curve_in(self, shift=0):
        df_curve_in = self.__util.shift_date_index(self.df_curve_in, shift)
        return df_curve_in.loc[self.__first_date:]

    def get_df_curve_out(self, shift=0):
        df_curve_out = self.__util.shift_date_index(self.df_curve_out, shift)
        return df_curve_out.loc[self.__first_date:]

    @property
    def df_curve_out(self):
        df = self.__huiyan_crawler.df_curve_out
        df = self.__data_frame_region_to_english_if_needed(df)
        return df

    def get_df_curve_in_out_rate(self, shift=0):
        df = self.df_curve_in / self.df_curve_out
        return self.__util.shift_date_index(df, shift)

    @property
    def df_curve_in_out_rate(self):
        return self.get_df_curve_in_out_rate(1)

    @property
    def df_move_in_injured(self):
        return self.get_df_move_in_injured()

    @property
    def df_move_in_injured_cp(self):
        return self.get_df_move_in_injured(consider_population=True)

    @property
    def df_move_in_risk(self):
        return self.get_df_move_in_injured(shift_one_day=False)

    @property
    def df_move_in_risk_cp(self):
        return self.get_df_move_in_injured(shift_one_day=False, consider_population=True)

    @property
    def df_move_inc_corr(self):
        df = self.__df_move_inc_corr
        if self.__last_date in df.index:
            df = self.__df_move_inc_corr[:self.__last_date]
        dfs = []
        if self.__in_english:
            for region in df.columns.levels[0]:
                df_region = df[region]
                df_region.columns = pd.MultiIndex.from_product([[region], ['corr', 'corr delta', 'offset', 'window']])
                dfs.append(df_region)
            df = pd.concat(dfs, axis=1)
        return df

    @property
    def df_move_inc_corr_cp(self):
        return self.__df_move_inc_corr_cp

    def get_df_move_inc_corr(self, n=3, consider_population=False, shift_one_day=False, sample_cnt=0, in_english=False):
        path = HuiyanCrawler.get_df_move_inc_corr_path(n=n, consider_population=consider_population,
                                                       shift_one_day=shift_one_day, sample_cnt=sample_cnt)
        try:
            df = pd.read_hdf(path, 'huiyan')
        except FileNotFoundError:
            return pd.DataFrame([])
        if in_english:
            dfs = []
            for region in df.columns.levels[0]:
                _df = df[region].copy()
                _df.columns = pd.MultiIndex.from_product([[self.__region_to_english[region]], _df.columns])
                dfs.append(_df)
            df = pd.concat(dfs, axis=1)
        return df

    def get_df_move_in_injured(self, shift=0, window=1, n=3, consider_population=False, shift_one_day=True):
        '''
        进入某一地区的感染人流规模估算 = sigma ( 进入某一地区的各地人数规模 * 各来源地不含当天近 n 天的感染人数总和 ）
        :param shift: 偏移量，默认为 0，表示当天只能拿到前 shift 天的计算结果数据
        :param window: 最近 window 天进入该地区的人流风险系数的均值
        :param n: 最近 n 天（不含当天）人流来源地的累计确诊人数
        :param consider_population: 是否考虑人流来源地的人口数量
        :param shift_one_day: 是否不考虑当天
        :return:
        '''
        path = 'cache/{}/move_in_injured-{}-shift={}-window={}-n={}-cp={}.csv'.format(
            'shift_one_day' if shift_one_day else 'not_shift_one_day', self.__last_date, shift, window, n,
            consider_population)
        df = None
        try:
            df = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            pass
        if df is not None:
            df = df.loc[self.__first_date: self.__last_date]
            return self.__data_frame_region_to_english_if_needed(df)
        in_english = self.__in_english
        self.__in_english = False
        df_cum_n = self.get_df_virus_daily_inc_injured_cum_n(n=n, shift_one_day=shift_one_day)
        self.__in_english = in_english
        ss = []
        for region in self.__util.huiyan_region_id:
            df_rate = self.__huiyan_crawler.get_rate(region)
            if df_rate.size > 0:
                rate_col = '规模/人口' if consider_population else '规模'
                df_rate = df_rate[['省', rate_col]]
                df_rate.index = pd.MultiIndex.from_arrays([df_rate.index, df_rate['省']])
                del df_rate['省']
                s_cum_n = df_cum_n.stack(0)
                s_cum_n.index.names = ['日期', '省']
                s_rate = df_rate[rate_col]
                df = pd.DataFrame({'风险规模': s_rate * s_cum_n})
                df.fillna(0, inplace=True)
                df = df.unstack(1)
                s = df.sum(axis=1)
                s.name = region
                ss.append(s)
        df = pd.DataFrame(ss).T.sort_index(axis=1)
        if window > 1:
            arr = df.values
            values = arr.copy()
            for win in range(1, window):
                values[win:] += arr[:-win]
            values /= window
            df.values[:] = values
        df = self.__util.shift_date_index(df, shift)
        df = df.loc[:self.__last_date]
        df.to_csv(path)
        df = df.loc[self.__first_date:]
        df = self.__data_frame_region_to_english_if_needed(df)
        return df

    def get_df_move_in_injured_inc_rate(self, compare_shift=1, shift=3):
        '''
        获取进入地区人口风险系数的增长趋势，VAL[shift] / VAL[shift + compare_shift]
        :param compare_shift:
        :param shift: 默认为 3，因为 3 天前人流风险系数是 1 - 7 天前中重要性最强的特征
        :return:
        '''
        df_move_in = self.get_df_move_in_injured(shift=shift)
        df_move_in_compare = self.get_df_move_in_injured(shift=shift+compare_shift)
        return df_move_in / df_move_in_compare

    def get_move_in_injured_corr(self, n=8, shift=0, window=1, consider_population=False, shift_one_day=True,
                                 sample_cnt=0):
        '''
        获取进入地区人口风险系数和每日新增人数的相关性，返回相应的 Series
        :param n: 最近 n 天（不含当天）人流来源地的累计确诊人数，默认取 n 为 8，因经测试，n 为 8 时候相关性最大，或 8 以后相关性增幅明显变小
        :param shift:
        :param window:
        :param consider_population: 是否考虑人口（各来源地的风险系数除以来源地的人口数）
        :param shift_one_day: 是否忽略当天
        :param int sample_cnt: 计算相关性的最大样本数，值若 <= 0，代表从 first_date 起
        :return:
        '''
        df_move_in_injured = self.get_df_move_in_injured(
            shift=shift, window=window, n=n, consider_population=consider_population, shift_one_day=shift_one_day)
        df_inc_injured = self.df_virus_daily_inc_injured.sort_index(axis=1)
        corr = {}
        for region in df_move_in_injured.columns:
            if region in df_move_in_injured and region in df_inc_injured:
                index = set(df_move_in_injured.index.tolist()) & set(df_inc_injured.index.tolist())
                if len(index) < df_move_in_injured.index.size or len(index) < df_inc_injured.index.size:
                    index = sorted(list(index))
                    df_move_in_injured = df_move_in_injured.loc[index]
                    df_inc_injured = df_inc_injured.loc[index]
                arr_move_in_injured = df_move_in_injured.loc[self.__first_date:, region].values
                arr_inc_injured = df_inc_injured.loc[self.__first_date:, region].values
                if sample_cnt > 0:
                    arr_move_in_injured = arr_move_in_injured[-sample_cnt:]
                    arr_inc_injured = arr_inc_injured[-sample_cnt:]
                corr[region] = np.corrcoef(arr_move_in_injured, arr_inc_injured)[0, 1]
        return pd.Series(corr)

    def get_move_in_injured_corr_info(self, n=8, shift=0):
        '''
        获取进入地区人口风险系数和每日新增人数的相关性，并和累计人数一起组合成 DataFrame 返回
        :param n: 最近 n 天（不含当天）人流来源地的累计确诊人数，默认取 n 为 8，因经测试，n 为 8 时候相关性最大，或 8 以后相关性增幅明显变小
        :param shift:
        :return:
        '''
        s_virus_injured = self.df_virus_injured.iloc[-1]
        data = OrderedDict()
        df_move_in_injured = self.get_df_move_in_injured(shift=shift, n=n)
        df_inc_injured = self.df_virus_daily_inc_injured.sort_index(axis=1)
        for region in df_move_in_injured.columns:
            if region in df_move_in_injured and region in df_inc_injured:
                corr_val = np.corrcoef(df_move_in_injured[region], df_inc_injured[region])[0, 1]
                data[region] = [s_virus_injured[region], corr_val]
        df = pd.DataFrame(data).T
        df.columns = ['累计确诊', '每日新增确诊和近{}天内进入人口风险系数的相关性'.format(n)]
        df['累计确诊'] = df['累计确诊'].astype(np.int32)
        return df.sort_values(df.columns[1], ascending=False)

    @staticmethod
    def get_df_day_idx(index, columns):
        '''
        获取每列为 1、2、3、......、df.shape[0] 的，index、columns 和 df 一样的 DataFrame
        :param index:
        :param columns:
        :return:
        '''
        arr = np.zeros(shape=(len(index), len(columns)), dtype=np.int32)
        arr[:] = np.arange(1, arr.shape[0] + 1).reshape(arr.shape[0], 1)
        return pd.DataFrame(arr, index=index, columns=columns).copy()

    @staticmethod
    def moving_avg(df, window=3, shift=0, keep_shape=False):
        '''
        滑动平均
        :param df:
        :param window:
        :param shift:
        :param keep_shape:
        :return:
        '''
        df_values = df.values
        arr = np.zeros(shape=(df.shape[0] - window + 1, df.shape[1]), dtype=np.float64)
        for col_idx in range(df.shape[1]):
            col_value = df_values[:, col_idx]
            moving_avg_col = np.cumsum(col_value, dtype=float)
            moving_avg_col[window:] = moving_avg_col[window:] - moving_avg_col[:-window]
            moving_avg_col = moving_avg_col[window - 1:] / window
            arr[:, col_idx] = moving_avg_col
        index = df.index[window - 1:]
        if shift > 0:
            arr[shift:, :] = arr[:-shift, :]
            arr = arr[shift:]
            index = index[shift:]
        res_df = pd.DataFrame(arr, index=index, columns=df.columns)
        if keep_shape:
            res_df = res_df.reindex(df.index)
        return res_df

    @staticmethod
    def k_means(df, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                precompute_distances='auto', verbose=0, random_state=None, copy_x=True,
                n_jobs=1, algorithm='auto', try_times=1, insure_sorted=False, order=None):
        '''
        :param df: 一级列索引的 DataFrame
        :param df:
        :param n_clusters:
        :param init:
        :param n_init:
        :param max_iter:
        :param tol:
        :param precompute_distances:
        :param verbose:
        :param random_state:
        :param copy_x:
        :param n_jobs:
        :param algorithm:
        :param try_times: 反复调用 k mean 取最优的次数
        :param insure_sorted: 如果确保输入数据是排序好的，分类需要确保是连续的
        :param order: 重新排序簇 id
        :return:
        '''
        res_clf = res_col_in_cluster = res_cluster_centers = min_inertia = None
        for _ in range(try_times):
            while True:
                clf = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                             precompute_distances=precompute_distances, verbose=verbose,
                             random_state=random_state, copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
                clf.fit(df.values.T)
                need_continue = False
                label_to_cols = OrderedDict()
                if insure_sorted:
                    label_set = set()
                    last_label = None
                    for i, label in enumerate(clf.labels_):
                        if label not in label_to_cols:
                            label_to_cols[label] = []
                        label_to_cols[label].append(i)
                        if label in label_set:
                            if last_label is not None and last_label != label:
                                need_continue = True
                                break
                        else:
                            label_set.add(label)
                        last_label = label
                else:
                    for i, label in enumerate(clf.labels_):
                        if label not in label_to_cols:
                            label_to_cols[label] = []
                        label_to_cols[label].append(i)
                if not need_continue:
                    break
            col_in_cluster = [df.columns.values[idx] for idx in label_to_cols.values()]
            cluster_centers = clf.cluster_centers_[list(label_to_cols.keys())]
            if res_clf is None or clf.inertia_ < min_inertia:
                min_inertia = clf.inertia_
                res_clf = clf
                res_col_in_cluster = col_in_cluster
                res_cluster_centers = cluster_centers
        if order is not None:
            res_col_in_cluster = np.array([res_col_in_cluster[_] for _ in order])
            res_cluster_centers = np.array([res_cluster_centers[_] for _ in order])
        return res_clf, res_col_in_cluster, res_cluster_centers

    def subplots(self, df, ncols=4):
        '''
        对 df 每列画一个折线图
        :param df: 1 级行、列索引的 DataFrame
        :param ncols:
        :return:
        '''
        try:
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
        except ImportError:
            pass
        assert not isinstance(df.index, pd.MultiIndex)
        assert not isinstance(df.columns, pd.MultiIndex)
        nrows = math.ceil(df.shape[1] / ncols)
        ncols = 4
        fig, ax = self.plt.subplots(figsize=(ncols * 5, nrows * 3), nrows=nrows, ncols=ncols)
        x_data = Util.str_dates_to_dates(df.index)
        k = 0
        if ax.ndim == 1:
            ax = [ax]
        for i in range(nrows):
            for j in range(ncols):
                if k == df.columns.size:
                    break
                region = df.columns[k]
                y_data = df[region]
                ax[i][j].plot(x_data, y_data)
                ax[i][j].set_title(region)
                k += 1
        self.plt.show()

    def __region_to_english_if_needed(self, region):
        if self.__in_english and region in self.__region_to_english:
            return self.__region_to_english[region]
        return region

    def __series_region_to_english_if_needed(self, s):
        if self.__in_english:
            index = []
            not_found_regions = []
            for region in s.index:
                if region in self.__region_to_english:
                    index.append(self.__region_to_english[region])
                else:
                    not_found_regions.append(region)
            if 0 < len(not_found_regions) < s.size:
                raise ValueError('index 长度不正确，可能是中文索引有不支持的地区，需要添加：{}'
                                 .format('、'.join(not_found_regions)))
            if len(index) > 0:
                s.index = index
            if s.index.name == '日期':
                s.index.name = 'date'
        return s

    def __data_frame_region_to_english_if_needed(self, df):
        if self.__in_english:
            columns = []
            not_found_regions = []
            for region in df.columns:
                if region in self.__region_to_english:
                    columns.append(self.__region_to_english[region])
                else:
                    not_found_regions.append(region)
            if 0 < len(not_found_regions) < df.shape[1]:
                raise ValueError('index 长度不正确，可能是中文索引有不支持的地区，需要添加：{}'
                                 .format('、'.join(not_found_regions)))
            if len(columns) > 0:
                df.columns = columns
            if df.index.name in ['日期', None]:
                df.index.name = 'date'
            for region in ['Wuhan', 'Wenzhou']:
                if region in df.columns:
                    del df[region]
        return df

    def plot_move_inc_corr(self, region, date=None, consider_population=False, n=3, offset=None, window=None,
                           resize=True, shift_one_day=False, sample_cnt=0, use_best_fit=True):
        '''
        画出进入人流风险系数和每日新增确诊人数的折线图，并输出相关系数
        :param region:
        :param date:
        :param consider_population:
        :param n:
        :param offset:
        :param window:
        :param resize: 是否将风险系数调整为每日新增人数相同的纵坐标(最高风险系数调整为最高每日新增确诊人数的值)
        :param shift_one_day:
        :param int sample_cnt: 计算相关性的最大样本数，值若 <= 0，代表从 first_date 起
        :param use_best_fit:
        '''
        if not use_best_fit and offset is None and window is None:
            offset = 0
            window = 1
        region = self.__region_to_english_if_needed(region)
        if date is None:
            date = self.__last_date
        date = str(date)
        if offset is None or window is None:
            if n == 3 and not consider_population and not shift_one_day:
                df_corr = self.__df_move_inc_corr_cp if consider_population else self.__df_move_inc_corr
            else:
                df_corr = self.get_df_move_inc_corr(n=n, consider_population=consider_population,
                                                    shift_one_day=shift_one_day)
            if df_corr.size == 0:
                raise ValueError('无数据，需要先构造进入人流风险系数和每日新增确诊人数的相关系数表')
            if date > df_corr.index[-1]:
                date = df_corr.index[-1]
            # region_in_chinese = self.__region_to_chinese[region] if self.__in_english else region
            s = df_corr.loc[date, region]
            offset = int(s['shift'])
            window = int(s['window'])
        s_corr = self.get_move_in_injured_corr(
            n=n, shift=offset, window=window, consider_population=consider_population, shift_one_day=shift_one_day,
            sample_cnt=sample_cnt)
        s_corr = self.__series_region_to_english_if_needed(s_corr)
        print('{} = {}, window = {}, corr = {}'.format('offset' if self.__in_english else 'shift',
                                                       offset, window, s_corr[region]))
        s_daily_inc = self.df_virus_daily_inc_injured[region]
        s_daily_inc = s_daily_inc.loc[:date]
        s_move_in = self.get_df_move_in_injured(
            offset, window, n, consider_population, shift_one_day)[region].loc[self.__first_date:]
        s_move_in = s_move_in.loc[:date]
        if resize:
            s_move_in = s_move_in / s_move_in.max() * s_daily_inc.max()
        # figsize = (6, 3.6)
        figsize = (8, 5)
        # label = 'daily new diagnosed' if self.__in_english else '每日新增确诊人数'
        if self.__in_english:
            label = 'NEW'
        else:
            label = '每日新增确诊人数'
        if self.__in_english:
            for s in [s_daily_inc, s_move_in]:
                s.index.name = 'date'
                s.index = ['{}/{}/{}'.format(date[5:7], date[-2:], date[:4]) for date in s.index]
        max_val = s_daily_inc.max()
        s_daily_inc /= max_val
        s_daily_inc.plot(color='red', label=label, figsize=figsize)
        # label = 'daily immigration risk' if self.__in_english else '进入人流风险系数'
        if self.__in_english:
            if use_best_fit:
                label = 'PROCESSED RISK'
            else:
                label = 'RISK'
        else:
            label = '每日新增确诊人数'
        s_move_in /= max_val
        s_move_in.plot(color='blue', label=label, figsize=figsize)
        self.plt.title(region)
        self.plt.legend(loc='upper left')

    def plot_region_map(self, region_cluster_ids, title):
        '''
        画中国各地区疫情图
        :param Series region_cluster_ids: index 为各地名、values 为各地所在簇 id
        :param title: 图片标题
        :return:
        '''
        def __get_color(_cluster_id, _cluster_cnt):
            return cmap(int(200 - _cluster_id * 50 * 4 / (_cluster_cnt - 1)))

        try:
            from mpl_toolkits.basemap import Basemap
        except ImportError:
            raise ImportError('需要安装 basemap，并在 jupyter 中使用该函数')

        self.plt.figure(figsize=(10, 10))
        map = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51,
                      projection='lcc', lat_1=33, lat_2=45, lon_0=100)
        map.readshapefile('data/basemap/china/gadm36_CHN_shp/gadm36_CHN_1', 'states', drawbounds=True)
        map.readshapefile('data/basemap/china/gadm36_TWN_shp/gadm36_TWN_1', 'taiwan', drawbounds=True)
        statenames = []
        colors = {}
        cmap = self.plt.cm.YlOrRd
        cluster_cnt = np.unique(region_cluster_ids.values).size
        for shapedict in map.states_info:
            statename = shapedict['NL_NAME_1']
            p = statename.split('|')
            if len(p) > 1:
                s = p[1]
            else:
                s = p[0]
            s = s[:2]
            if s == '黑龍':
                s = '黑龙江'
            elif s == '内蒙':
                s = '内蒙古'
            statenames.append(s)
            c_id = region_cluster_ids[s]
            if s not in colors:
                colors[s] = __get_color(c_id, cluster_cnt)
        tw_c_id = region_cluster_ids['台湾']
        tw_color = __get_color(tw_c_id, cluster_cnt)
        ax = self.plt.gca()
        for idx, seg in enumerate(map.states):
            color = rgb2hex(colors[statenames[idx]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)
        for seg in map.taiwan:
            color = rgb2hex(tw_color)
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)
        self.plt.title(title, fontsize=24)
        self.plt.show()

    @property
    def region_risk_level(self):
        _region_risk_level = {
            4: ['湖北'],
            3: ['广东', '浙江', '河南', '湖南', '安徽', '江西'],
            2: ['江苏', '重庆', '山东', '四川', '北京', '黑龙江', '上海', '福建', '陕西', '河北', '广西'],
            1: ['云南', '海南', '山西', '辽宁', '贵州', '天津', '甘肃', '吉林', '内蒙古', '宁夏', '新疆', '青海', '西藏',
                '香港', '澳门', '台湾']
        }
        region_risk_level = {}
        for level, regions in _region_risk_level.items():
            for region in regions:
                region_risk_level[region] = level
        return region_risk_level

    def get_df_region_risk_level(self, index, columns):
        region_risk_level = self.region_risk_level
        values = np.zeros(shape=(len(index), len(columns)), dtype=np.int32)
        for j, region in enumerate(columns):
            values[:, j] = region_risk_level[region]
        return pd.DataFrame(values, index=index, columns=columns)

    def get_df_region_cluster_one_hot(self, index, columns):
        region_risk_level = self.region_risk_level
        levels = set()
        for j, region in enumerate(columns):
            levels.add(region_risk_level[region])
        levels = sorted(list(levels))
        level_cnt = len(levels)
        values = np.zeros(shape=(len(index), len(columns) * level_cnt), dtype=np.bool)
        for j, region in enumerate(columns):
            level = region_risk_level[region]
            idx = levels.index(level)
            values[:, j * level_cnt + idx] = True
        cluster_names = ['是否地区簇{}'.format(_) for _ in levels]
        columns = pd.MultiIndex.from_product([columns, cluster_names])
        return pd.DataFrame(values, index=index, columns=columns)

    def get_next_ma(self, df, n, slice_last_rows=False):
        '''
        将 df 的每行的值改为当前行和 n-1 个后续行的均值。
        如果 slice_last_rows 为 True，则去掉最后 n - 1 行，否则先将最后缺失的 n - 1 行以原有的最后一行补上，然后取平均
        '''
        assert isinstance(n, int) and n > 0
        if n == 1:
            return df
        values = df.values
        if not slice_last_rows:
            append_values = np.zeros(shape=(n-1, values.shape[1]), dtype=values.dtype)
            append_values[:] = values[-1, :]
            values = np.vstack([values, append_values])
        row_cnt = df.shape[0] - n + 1 if slice_last_rows else df.shape[0]
        arr = np.zeros(shape=(row_cnt, df.shape[1]), dtype=values.dtype)
        for shift in range(n):
            arr += values[shift: shift + row_cnt, :]
        arr /= n
        index = df.index
        if slice_last_rows and n > 1:
            index = index[: -(n-1)]
        return pd.DataFrame(arr, index=index, columns=df.columns)

    def get_data(self, predict_day_cnt=1, n_days=7, n_move_days=7, cum_n_days=10000, use_move_in_corr=False,
                 use_move_in=False, use_move_in_injured=True, use_best_fit_move_in_injured=False,
                 move_in_injured_consider_population=False, use_move_out=False, use_day_idx=True,
                 use_in_out_rate=True, use_move_in_inc_rate=False, use_ma3=False, use_ma5=False,
                 use_ma7=False, use_last_ma3=True, use_last_ma3_rank=False, use_region_risk_level=False,
                 use_region_cluster_one_hot=False, use_weather=False, sample_weight_type=1,
                 sample_weight_power=0.5, omit_hubei=True, selected_regions=None, train_omit_zero_y=False,
                 sample_cnt=0):
        '''
        获取训练和预测数据
        predict_day_cnt             预测的天数，值代表预测后面连续的多天每日新增均值
        n_days                      使用不含当天的最近 n_days 天每日新增确诊人数
        n_move_days                 使用不含当天的最近 n_move_days 天的人流风险系数
        cum_n_days                  人流数据风险系数使用不含人流当天的 cum_n_days 天内的累计确诊人数，作为来源地区风险性评估，
                                    默认取足够大的数字，表示累计的所有天
        use_move_in_corr            是否使用进入人流风险系数和地区每日新增相关性的数据
        use_move_in                 是否使用进地区人流规模数据
        use_move_in_injured         是否使用人流风险系数数据
        use_best_fit_move_in_injured    使用最佳 shift、window（n）组合的人流风险系数
        move_in_injured_consider_population    人流风险系数是否考虑来源地人口数量
        use_move_out                是否使用出地区人流规模数据
        use_day_idx                 是否使用天数
        use_in_out_rate             各地 1天前的人流进出比例
        use_move_in_inc_rate        人流数据风险系数趋势
        use_ma3                     是否使用前3天（不含当天）新增确诊人数均值，函数中 use_data 是算法使用的数据
        use_ma5                     是否使用前5天（不含当天）新增确诊人数均值，函数中 use_data 是算法使用的数据
        use_ma7                     是否使用前7天（不含当天）新增确诊人数均值，函数中 use_data 是算法使用的数据
        use_last_ma3                是否使用 last_date（不含 last_date）的最近 3 天新增确诊人数均值
        use_last_ma3_rank           是否使用 last_date（不含 last_date）的最近 3 天新增确诊人数均值的排序
        use_region_risk_level       是否使用地区风险级别
        use_region_cluster_one_hot  是否使用地区风险级别的独热编码
        use_weather                 是否使用天气数据，函数中 use_data 是算法使用的数据
        sample_weight_type          0 不使用 sample_weight，1 使用最近 3 天新增均值，2 使用最近 3 天新增均值排名
        sample_weight_power         sample_weight 使用乘方的次数
        omit_hubei                  是否忽略湖北
        selected_regions            只学习和预测某些地区，如果为 None，表示都学习和预测
        train_omit_zero_y           训练集忽略预测值为 0 的
        sample_cnt                  函数 get_move_in_injured_corr 的参数，计算相关性的最大样本数，值若 <= 0，代表从 first_date 起
        '''
        if use_move_in and (use_move_in_injured or use_best_fit_move_in_injured):
            use_move_in_injured = use_best_fit_move_in_injured = False
            use_last_ma3 = use_last_ma3_rank = False
        if use_best_fit_move_in_injured:
            use_move_in_injured = False
        if use_last_ma3 and use_last_ma3_rank:
            use_last_ma3 = False
        omitted_provinces = ['湖北'] if omit_hubei else []
        df_virus_daily_inc_injured = self.del_city_special_regions(self.df_virus_daily_inc_injured,
                                                                   omitted_provinces=omitted_provinces)
        df_virus_daily_inc_injured = self.get_next_ma(df_virus_daily_inc_injured, predict_day_cnt)
        index = df_virus_daily_inc_injured.index.copy()
        columns = df_virus_daily_inc_injured.columns.copy()
        dfs = []
        use_data = []  # 所有使用的数据 list
        if use_move_in_corr:
            s_corr = self.get_move_in_injured_corr(sample_cnt=sample_cnt)[columns]
            df_corr = pd.DataFrame(np.zeros(shape=(index.size, columns.size), dtype=np.float64),
                                   index=index, columns=columns)
            df_corr.values[:] = s_corr.values
            df_corr.columns = pd.MultiIndex.from_product([df_corr.columns, ['进入人流相关性']])
            use_data.append(df_corr)
        if use_move_in:
            df_curve_in_1 = self.del_city_special_regions(self.get_df_curve_in(1), omitted_provinces=omitted_provinces)
            df_curve_in_2 = self.del_city_special_regions(self.get_df_curve_in(2), omitted_provinces=omitted_provinces)
            df_curve_in_3 = self.del_city_special_regions(self.get_df_curve_in(3), omitted_provinces=omitted_provinces)
            df_curve_in_4 = self.del_city_special_regions(self.get_df_curve_in(4), omitted_provinces=omitted_provinces)
            df_curve_in_5 = self.del_city_special_regions(self.get_df_curve_in(5), omitted_provinces=omitted_provinces)
            df_curve_in_6 = self.del_city_special_regions(self.get_df_curve_in(6), omitted_provinces=omitted_provinces)
            df_curve_in_7 = self.del_city_special_regions(self.get_df_curve_in(7), omitted_provinces=omitted_provinces)
            df_curve_in_1.columns = pd.MultiIndex.from_product([df_curve_in_1.columns, ['1天前人流进量']])
            df_curve_in_2.columns = pd.MultiIndex.from_product([df_curve_in_2.columns, ['2天前人流进量']])
            df_curve_in_3.columns = pd.MultiIndex.from_product([df_curve_in_3.columns, ['3天前人流进量']])
            df_curve_in_4.columns = pd.MultiIndex.from_product([df_curve_in_4.columns, ['4天前人流进量']])
            df_curve_in_5.columns = pd.MultiIndex.from_product([df_curve_in_5.columns, ['5天前人流进量']])
            df_curve_in_6.columns = pd.MultiIndex.from_product([df_curve_in_6.columns, ['6天前人流进量']])
            df_curve_in_7.columns = pd.MultiIndex.from_product([df_curve_in_7.columns, ['7天前人流进量']])
            use_data += [df_curve_in_1, df_curve_in_2, df_curve_in_3, df_curve_in_4,
                         df_curve_in_5, df_curve_in_6, df_curve_in_7]
        if use_move_out:
            df_curve_out_1 = self.del_city_special_regions(self.get_df_curve_out(1), omitted_provinces=omitted_provinces)
            df_curve_out_2 = self.del_city_special_regions(self.get_df_curve_out(2), omitted_provinces=omitted_provinces)
            df_curve_out_3 = self.del_city_special_regions(self.get_df_curve_out(3), omitted_provinces=omitted_provinces)
            df_curve_out_4 = self.del_city_special_regions(self.get_df_curve_out(4), omitted_provinces=omitted_provinces)
            df_curve_out_5 = self.del_city_special_regions(self.get_df_curve_out(5), omitted_provinces=omitted_provinces)
            df_curve_out_6 = self.del_city_special_regions(self.get_df_curve_out(6), omitted_provinces=omitted_provinces)
            df_curve_out_7 = self.del_city_special_regions(self.get_df_curve_out(7), omitted_provinces=omitted_provinces)
            df_curve_out_1.columns = pd.MultiIndex.from_product([df_curve_out_1.columns, ['1天前人流出量']])
            df_curve_out_2.columns = pd.MultiIndex.from_product([df_curve_out_2.columns, ['2天前人流出量']])
            df_curve_out_3.columns = pd.MultiIndex.from_product([df_curve_out_3.columns, ['3天前人流出量']])
            df_curve_out_4.columns = pd.MultiIndex.from_product([df_curve_out_4.columns, ['4天前人流出量']])
            df_curve_out_5.columns = pd.MultiIndex.from_product([df_curve_out_5.columns, ['5天前人流出量']])
            df_curve_out_6.columns = pd.MultiIndex.from_product([df_curve_out_6.columns, ['6天前人流出量']])
            df_curve_out_7.columns = pd.MultiIndex.from_product([df_curve_out_7.columns, ['7天前人流出量']])
            use_data += [df_curve_out_1 + df_curve_out_2 + df_curve_out_3 + df_curve_out_4,
                         df_curve_out_5, df_curve_out_6, df_curve_out_7]
        if use_move_in_injured:
            # 各地 7 天内（不含当天，即 1天前到7天前，的所有确诊人数作为权重 * 进入该地区的人流规模）
            _shift = 1
            df_move_in_injured_1 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+0, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_2 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+1, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_3 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+2, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_4 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+3, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_5 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+4, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_6 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+5, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_7 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+6, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_8 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+7, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_9 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+8, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_10 = self.del_city_special_regions(self.get_df_move_in_injured(
                _shift+9, cum_n_days, move_in_injured_consider_population), omitted_provinces=omitted_provinces)
            df_move_in_injured_1.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_1.columns, ['1天前人流风险系数']])
            df_move_in_injured_2.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_2.columns, ['2天前人流风险系数']])
            df_move_in_injured_3.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_3.columns, ['3天前人流风险系数']])
            df_move_in_injured_4.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_4.columns, ['4天前人流风险系数']])
            df_move_in_injured_5.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_5.columns, ['5天前人流风险系数']])
            df_move_in_injured_6.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_6.columns, ['6天前人流风险系数']])
            df_move_in_injured_7.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_7.columns, ['7天前人流风险系数']])
            df_move_in_injured_8.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_8.columns, ['8天前人流风险系数']])
            df_move_in_injured_9.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_9.columns, ['9天前人流风险系数']])
            df_move_in_injured_10.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_10.columns, ['10天前人流风险系数']])
            if not use_move_in_inc_rate:
                use_data.append(df_move_in_injured_1)
                use_data.append(df_move_in_injured_2)
            use_data.append(df_move_in_injured_3)
            if not use_move_in_inc_rate:
                for i in range(4, n_move_days + 1):
                    local_vals = locals()
                    val = local_vals.get('df_move_in_injured_{}'.format(i))
                    if val is not None:
                        use_data.append(val)
        if use_best_fit_move_in_injured:
            data = []
            for region in columns:
                s = self.__df_move_inc_corr.loc[self.__last_date, region]
                df_move_in_injured = self.get_df_move_in_injured(
                    int(s['shift']), int(s['window']), move_in_injured_consider_population)
                data.append(df_move_in_injured[region])
            df_move_in_injured = pd.DataFrame(data).T
            s_max = df_move_in_injured.max(axis=0)
            df_move_in_injured = df_move_in_injured / s_max
            df_move_in_injured.columns = pd.MultiIndex.from_product([df_move_in_injured.columns, ['人流风险系数']])
            use_data.append(df_move_in_injured)
        if use_weather:
            # 各地 14天前、13天前、......、3天前的天气滑动加权平均数据，权重值为 1,2,3,4,5,6,7,8,9,10,9,8
            df_weather_ma = self.del_city_special_regions(self.df_weather_ma, omitted_provinces=omitted_provinces)
            use_data.append(df_weather_ma)
        if use_ma3 + use_ma5 + use_ma7 < 2:
            # 使用如果均线数量小于 2，则使用各地 1天前、2天前、......、n 天前的每日新增确诊人数
            df_virus_n_days = self.del_city_special_regions(self.get_df_virus_n_days_inc_injured(n_days),
                                                            omitted_provinces=omitted_provinces)
            use_data.append(df_virus_n_days)
        df_daily_inc_ma3 = self.del_city_special_regions(
            self.moving_avg(self.df_virus_daily_inc_injured, window=3, shift=1, keep_shape=True).fillna(0),
            omitted_provinces=omitted_provinces)
        df_daily_inc_last_ma3_rank = None
        if use_last_ma3 or use_last_ma3_rank or sample_weight_type == 2:
            # 各地 last_date（不含 last_date）的最近 3 天新增确诊人数均值
            df_daily_inc_last_ma3 = df_daily_inc_ma3.copy()
            df_daily_inc_last_ma3.values[:] = df_daily_inc_last_ma3.iloc[-1]
            if use_last_ma3_rank or sample_weight_type == 2:
                # 各地 last_date（不含 last_date）的最近 3 天新增确诊人数均值排序
                df_daily_inc_last_ma3_rank = df_daily_inc_last_ma3.copy().astype(np.int32)
                df_daily_inc_last_ma3_rank.values[:] = np.argsort(df_daily_inc_last_ma3.iloc[-1].values)
                df_daily_inc_last_ma3_rank.columns = pd.MultiIndex.from_product(
                    [df_daily_inc_last_ma3_rank.columns, ['地区疫情严重程度排序']])
                if use_last_ma3_rank:
                    use_data.append(df_daily_inc_last_ma3_rank)
            if use_last_ma3:
                # 各地 last_date（不含 last_date）的最近 3 天新增确诊人数均值
                df_daily_inc_last_ma3.columns = pd.MultiIndex.from_product(
                    [df_daily_inc_last_ma3.columns, ['地区疫情严重程度']])
                use_data.append(df_daily_inc_last_ma3)
        if use_ma3 or sample_weight_type == 1:
            # 各地 1 - 3 天前每日新增均值
            df_daily_inc_ma3.columns = pd.MultiIndex.from_product(
                [df_daily_inc_ma3.columns, ['3日新增均值']])
            if use_ma3:
                use_data.append(df_daily_inc_ma3)
        if use_ma5:
            # 各地 1 - 5 天前每日新增均值
            df_daily_inc_ma5 = self.del_city_special_regions(
                self.moving_avg(self.df_virus_daily_inc_injured, window=5, shift=1, keep_shape=True).fillna(0),
                omitted_provinces=omitted_provinces)
            df_daily_inc_ma5.columns = pd.MultiIndex.from_product(
                [df_daily_inc_ma5.columns, ['5日新增均值']])
            use_data.append(df_daily_inc_ma5)
        if use_ma7:
            df_daily_inc_ma7 = self.del_city_special_regions(
                self.moving_avg(self.df_virus_daily_inc_injured, window=7, shift=1, keep_shape=True).fillna(0),
                omitted_provinces=omitted_provinces)
            df_daily_inc_ma7.columns = pd.MultiIndex.from_product(
                [df_daily_inc_ma7.columns, ['7日新增均值']])
            # 各地 1 - 7 天前每日新增均值
            use_data.append(df_daily_inc_ma7)
        if use_in_out_rate:
            # 各地 n 天前或平均的人流进出比例
            df_curve_in_out_rate = self.del_city_special_regions(self.get_df_curve_in_out_rate(1),
                                                                 omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_2 = self.del_city_special_regions(self.get_df_curve_in_out_rate(2),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_3 = self.del_city_special_regions(self.get_df_curve_in_out_rate(3),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_4 = self.del_city_special_regions(self.get_df_curve_in_out_rate(4),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_5 = self.del_city_special_regions(self.get_df_curve_in_out_rate(5),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_6 = self.del_city_special_regions(self.get_df_curve_in_out_rate(6),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_7 = self.del_city_special_regions(self.get_df_curve_in_out_rate(7),
                                                                   omitted_provinces=omitted_provinces)
            df_curve_in_out_rate_avg = df_curve_in_out_rate + df_curve_in_out_rate_2 + df_curve_in_out_rate_3 + \
                df_curve_in_out_rate_4 + df_curve_in_out_rate_5 + df_curve_in_out_rate_6 + df_curve_in_out_rate_7
            df_curve_in_out_rate_avg /= 7
            df_curve_in_out_rate.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate.columns, ['1天前进出比例']])
            df_curve_in_out_rate_2.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_2.columns, ['2天前进出比例']])
            df_curve_in_out_rate_3.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_3.columns, ['3天前进出比例']])
            df_curve_in_out_rate_4.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_4.columns, ['4天前进出比例']])
            df_curve_in_out_rate_5.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_5.columns, ['5天前进出比例']])
            df_curve_in_out_rate_6.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_6.columns, ['6天前进出比例']])
            df_curve_in_out_rate_7.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_7.columns, ['7天前进出比例']])
            df_curve_in_out_rate_avg.columns = pd.MultiIndex.from_product(
                [df_curve_in_out_rate_avg.columns, ['近期平均进出比例']])
            # use_data.append(df_curve_in_out_rate)
            # use_data.append(df_curve_in_out_rate_2)
            # use_data.append(df_curve_in_out_rate_3)
            # use_data.append(df_curve_in_out_rate_4)
            # use_data.append(df_curve_in_out_rate_5)
            # use_data.append(df_curve_in_out_rate_6)
            # use_data.append(df_curve_in_out_rate_7)
            use_data.append(df_curve_in_out_rate_avg)
        if use_move_in_inc_rate:
            # 各地人流数据风险系数趋势
            df_move_in_injured_inc_rate_1 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(1),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_2 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(2),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_3 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(3),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_4 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(4),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_5 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(5),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_6 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(6),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_7 = self.del_city_special_regions(self.get_df_move_in_injured_inc_rate(7),
                                                                          omitted_provinces=omitted_provinces)
            df_move_in_injured_inc_rate_1.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_1.columns, ['风险系数1日趋势']])
            df_move_in_injured_inc_rate_2.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_2.columns, ['风险系数2日趋势']])
            df_move_in_injured_inc_rate_3.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_3.columns, ['风险系数3日趋势']])
            df_move_in_injured_inc_rate_4.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_4.columns, ['风险系数4日趋势']])
            df_move_in_injured_inc_rate_5.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_5.columns, ['风险系数5日趋势']])
            df_move_in_injured_inc_rate_6.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_6.columns, ['风险系数6日趋势']])
            df_move_in_injured_inc_rate_7.columns = pd.MultiIndex.from_product(
                [df_move_in_injured_inc_rate_7.columns, ['风险系数7日趋势']])
            use_data += [df_move_in_injured_inc_rate_1 + df_move_in_injured_inc_rate_2 +
                         df_move_in_injured_inc_rate_3 + df_move_in_injured_inc_rate_4 +
                         df_move_in_injured_inc_rate_5 + df_move_in_injured_inc_rate_6 + df_move_in_injured_inc_rate_7]
        if use_day_idx:
            # 自然数日期
            df_day_idx = self.del_city_special_regions(self.get_df_day_idx(index, columns),
                                                       omitted_provinces=omitted_provinces)
            df_day_idx.columns = pd.MultiIndex.from_product(
                [df_day_idx.columns, ['疫情天数']])
            use_data.append(df_day_idx)
        if use_region_risk_level:
            # 地区风险级别
            df_region_risk_level = self.get_df_region_risk_level(index, columns)
            df_region_risk_level.columns = pd.MultiIndex.from_product(
                [df_region_risk_level.columns, ['地区风险级别']])
            use_data.append(df_region_risk_level)
        if use_region_cluster_one_hot:
            # 地区簇独热编码
            df_region_cluster_one_hot = self.get_df_region_cluster_one_hot(index, columns)
            use_data.append(df_region_cluster_one_hot)

        # 合并数据
        for df in use_data:
            for region in omitted_provinces:
                if region in df:
                    raise Exception('函数内部代码有误')
            if df.shape[0] != index.size:
                df = df.reindex(index)
            dfs.append(df)
        df_trait = pd.concat(dfs, axis=1)
        df_trait = df_trait.sort_index(axis=1)
        df_X_train = df_trait.iloc[:-1]
        df_y_train = df_virus_daily_inc_injured.iloc[:df_X_train.shape[0]]
        df_X_test = df_trait.iloc[-1:]
        df_y_test = df_virus_daily_inc_injured.iloc[-1:]
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        sample_weight = None
        regions = []
        for region in df_y_train.columns:
            if selected_regions is not None and region not in selected_regions:
                continue
            if region not in omitted_provinces:
                regions.append(region)
                arr_X_train = df_X_train[region].values
                arr_y_train = df_y_train[region].values
                if train_omit_zero_y:
                    not_zero_mask = arr_y_train != 0
                    arr_X_train = arr_X_train[not_zero_mask]
                    arr_y_train = arr_y_train[not_zero_mask]
                arr_X_test = df_X_test[region].values
                arr_y_test = df_y_test[region].values
                if sample_weight_type == 1:
                    # 训练集3日新增均值
                    arr_sample_weight = df_daily_inc_ma3[region]['3日新增均值'].values[:-1]
                elif sample_weight_type == 2:
                    # 训练集地区疫情严重程度排序
                    arr_sample_weight = df_daily_inc_last_ma3_rank[region]['地区疫情严重程度排序'].values[:-1]
                elif sample_weight_type == 3:
                    # 训练集真实值
                    arr_sample_weight = df_y_train[region].values
                elif sample_weight_type == 0:
                    arr_sample_weight = None
                else:
                    raise ValueError('arr_sample_weight 值错误：{}'.format(sample_weight_type))
                if arr_sample_weight is not None and sample_weight_power != 1:
                    arr_sample_weight = arr_sample_weight ** sample_weight_power
                # arr_y_train_sqrt = np.sqrt(arr_y_train)
                if X_train is None:
                    X_train = arr_X_train
                    y_train = arr_y_train
                    X_test = arr_X_test
                    y_test = arr_y_test
                    if arr_sample_weight is not None:
                        sample_weight = arr_sample_weight
                else:
                    X_train = np.vstack([X_train, arr_X_train])
                    y_train = np.hstack([y_train, arr_y_train])
                    X_test = np.vstack([X_test, arr_X_test])
                    y_test = np.hstack([y_test, arr_y_test])
                    if arr_sample_weight is not None:
                        sample_weight = np.hstack([sample_weight, arr_sample_weight])
        # print(X_train.shape, y_train.shape)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtrain.feature_names = df_X_train.columns.levels[1]
        dtest = xgb.DMatrix(X_test)
        dtest.feature_names = df_X_test.columns.levels[1]
        return {
            'df_trait': df_trait, 'df_X_train': df_X_train, 'df_y_train': df_y_train, 'df_X_test': df_X_test,
            'df_y_test': df_y_test, 'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
            'regions': regions, 'dtrain': dtrain, 'dtest': dtest, 'sample_weight': sample_weight,
            'sample_weight_eval_set': sample_weight
        }

    @staticmethod
    def objective(preds, dtrain):
        pass

    @staticmethod
    def evaluation(preds, dtrain):
        labels = dtrain.get_label()
        weight = labels / labels.sum()
        mask = labels > 0
        labels_mask = labels[mask]
        preds_mask = preds[mask]
        weight_mask = weight[mask]
        val = sum(abs(labels_mask - preds_mask) / labels_mask * weight_mask)
        return 'evaluation', val

    def get_model(self, data, learning_rate=None, n_estimators=None, max_depth=None, min_child_weight=None,
                  gamma=None, subsample=None, colsample_bytree=None, reg_alpha=None, objective='reg:gamma',
                  nfold=5, use_sklearn_model=True, print_log=False, **kwargs):
        '''
        调参、训练模型，并返回模型
        :param data:
        :param learning_rate:
        :param n_estimators:
        :param max_depth:
        :param min_child_weight:
        :param gamma:
        :param subsample:
        :param colsample_bytree:
        :param reg_alpha:
        :param objective: reg:squarederror, reg:gamma, reg:logistic
        :param nfold:
        :param use_sklearn_model:
        :param print_log: 是否输出相关日志
        :return:
        '''
        dtrain = data['dtrain']
        local_params = locals()
        if learning_rate is None:
            if objective == 'reg:gamma':
                _learning_rate = 0.1
            else:
                _learning_rate = 0.1
        else:
            _learning_rate = learning_rate
        model = xgb.XGBRegressor(
            max_depth=4 if max_depth is None else max_depth,
            learning_rate=_learning_rate,
            objective=objective)
        xgb_params = model.get_xgb_params()
        if n_estimators is None:
            cv_result = xgb.cv(
                model.get_xgb_params(), dtrain,
                num_boost_round=model.get_params()['n_estimators'],
                nfold=nfold,
                # metrics='rmse'
            )
            if print_log:
                print('init n_estimators: {}'.format(cv_result.shape[0]))
            xgb_params['n_estimators'] = cv_result.shape[0]
        learning_rate_range = range(1, 8) if objective == 'reg:gamma' else range(5, 21)
        for param_info in [
            [('max_depth', range(3, 16)), ('min_child_weight', range(1, 6))],
            [('gamma', [_ / 10.0 for _ in range(0, 5)])],
            [('subsample', [_ / 10.0 for _ in range(6, 11)]), ('colsample_bytree', [_ / 10.0 for _ in range(6, 11)])],
            [('reg_alpha', [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100])],
            [('n_estimators', [_ for _ in range(50, 160, 10)])],
            [('learning_rate', [_ / 100.0 for _ in learning_rate_range])],
            # [('max_depth', range(3, 16)), ('min_child_weight', range(1, 6))],
        ]:
            need_run_cv = False
            for name, search_range in param_info:
                val = local_params.get(name)
                if val is None:
                    need_run_cv = True
                    break
            if need_run_cv:
                search = GridSearchCV(estimator=xgb.XGBRegressor(**xgb_params),
                                      param_grid={name: search_range for name, search_range in param_info})
                # sample_weight_eval_set=data['sample_weight_eval_set']
                search.fit(data['X_train'], data['y_train'], sample_weight=data['sample_weight'])
                if print_log:
                    names = ' & '.join([name for name, _ in param_info])
                    # print('search {}, cv_results_: {}'.format(names, search.cv_results_))
                    print('search {}, best_score_: {}'.format(names, search.best_score_))
                    print('search {}, best_params_: {}'.format(names, search.best_params_))
                for name, _ in param_info:
                    xgb_params[name] = search.best_params_[name]
            else:
                for name, _ in param_info:
                    xgb_params[name] = local_params[name]
        # 训练
        if print_log:
            print('xgb_params: {}'.format(xgb_params))
        if use_sklearn_model:
            model.set_params(**xgb_params)
            # sample_weight_eval_set = data['sample_weight_eval_set']
            model.fit(data['X_train'], data['y_train'], sample_weight=data['sample_weight'])
        else:
            model = xgb.train(xgb_params, dtrain)
        if isinstance(model, xgb.XGBRegressor):
            # sklearn 模型有，xgb 模型没有
            score = model.score(data['X_train'], data['y_train'])
            print('模型 score：{}'.format(score))
            if print_log:
                # 特征重要性
                index = [_ for _ in data['df_trait'].columns.levels[1]]
                s_trait_importance = pd.Series(model.feature_importances_, index=index)
                s_trait_importance.plot(kind='bar')
                self.plt.title('特征重要性')
                self.plt.show()
        return model

    def get_df_predict_real(self, predicts, desc=None):
        '''
            使用 train_and_predict 的结果整理预测和实际的比对
        '''
        df_predict = pd.DataFrame(predicts).T
        regions = df_predict.columns.tolist()
        df_real = self.df_virus_daily_inc_injured.loc[df_predict.index, df_predict.columns]
        df_real.fillna(0, inplace=True)
        df_real = df_real.astype(np.int32)
        col = '预测'
        if desc is not None:
            col = '{}{}'.format(col, desc)
        df_predict.columns = pd.MultiIndex.from_product([['{}'.format(c) for c in df_predict.columns], [col]])
        df_real.columns = pd.MultiIndex.from_product([df_real.columns, ['实际']])
        df_predict = df_predict // 0.1 / 10
        df_predict_real = pd.concat([df_predict, df_real], axis=1)[regions]
        arr = df_predict_real.iloc[:, 0:df_predict_real.shape[1]:2].values - \
              df_predict_real.iloc[:, 1:df_predict_real.shape[1]:2].values
        df_compare = pd.DataFrame(arr, index=df_predict_real.index, columns=regions)
        return df_predict_real, df_compare

    @staticmethod
    def predict_compare(df_compare1, df_compare2):
        '''
        使用 2 次 get_df_predict_real 的返回值 df_compare，比较并返回第 2 次预测值跟接近第 1 次正确值的概率
        '''
        df_compare2 = df_compare2[df_compare1.columns]
        arr = np.zeros(shape=df_compare1.shape, dtype=np.int8)
        arr1 = abs(df_compare1.values)
        arr2 = abs(df_compare2.values)
        arr[arr1 < arr2] = 1
        arr[arr1 > arr2] = 2
        win_rate = []
        for j in range(arr.shape[1]):
            win_rate.append(((arr[:, j] == 2).sum() - (arr[:, j] == 1).sum()) / arr.shape[0])
        return pd.Series(win_rate, df_compare1.columns), pd.DataFrame(
            arr, index=df_compare1.index, columns=df_compare1.columns)

    @property
    def df_risk_and_size(self):
        df_move_in_risk = self.df_move_in_risk
        s_total = df_move_in_risk.mean().sort_values(ascending=False)
        s_total.name = 'mean "risk"'
        s_to_home = df_move_in_risk.loc[:'2020-01-24'].mean().sort_values(ascending=False)
        s_to_work = df_move_in_risk.loc['2020-01-25':].mean().sort_values(ascending=False)
        s_home_work = s_to_home / s_to_work
        s_to_home.name = 'to home "risk"'
        s_to_work.name = 'to work "risk"'
        s_home_work.name = '"risk" home work rate'
        df_risk = pd.DataFrame([s_total, s_to_home, s_to_work, s_home_work]).T.sort_values(
            by='mean "risk"', ascending=False)
        df_curve_in = self.df_curve_in
        s_total = df_curve_in.mean().sort_values(ascending=False)
        s_total.name = 'mean "size"'
        s_to_home = df_curve_in.loc[:'2020-01-24'].mean().sort_values(ascending=False)
        s_to_home.name = 'to home "size"'
        s_to_work = df_curve_in.loc['2020-01-25':].mean().sort_values(ascending=False)
        s_to_work.name = 'to work "size"'
        s_home_work = s_to_home / s_to_work
        s_home_work.name = '"size" home work rate'
        df_move = pd.DataFrame([s_total, s_to_home, s_to_work, s_home_work]).T.sort_values(
            by='mean "size"', ascending=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df_rank = pd.DataFrame([np.arange(df_risk.shape[0]) + 1]).T
        df_rank.index = df_risk.index
        df_rank.columns = ['"risk" rank']
        df = pd.concat([df_rank, df_risk, df_move], axis=1, sort=False)
        return df

    @property
    def df_half_day_corr_offset_window(self):
        df_new_risk = self.df_new_risk
        df = df_new_risk[
            [('new', 'half date idx'), ('new', 'half date'), ('new', 'half date corr'), ('new', 'half date offset'),
             ('new', 'half date window')]]
        df.columns = ['half date idx', 'half date', 'corr', 'offset', 'window']
        df = df.copy()
        df['offset + window'] = df['offset'] + df['window']
        df = df.sort_values(by='half date')
        return df

    @property
    def df_new_risk(self):
        df_move_inc_corr = self.df_move_inc_corr
        df_move_in_risk = self.df_move_in_risk
        df_curve_in = self.df_curve_in.loc[self.__first_date: self.__last_date]
        df_virus_daily_inc_injured = self.df_virus_daily_inc_injured
        if self.__in_english:
            del df_curve_in['Hubei']
            if 'Wuhan' in df_curve_in.columns:
                del df_curve_in['Wuhan']
        else:
            del df_curve_in['湖北']
            if '武汉' in df_curve_in.columns:
                del df_curve_in['武汉']
        s_to_home = df_curve_in.loc[:'2020-01-24'].mean().sort_values(ascending=False)
        s_to_home.name = 'to_home'
        s_to_work = df_curve_in.loc['2020-01-25':].mean().sort_values(ascending=False)
        s_to_work.name = 'to_work'
        s_home_work = s_to_home / s_to_work
        s_home_work.name = 'home work rate'
        df_move = pd.DataFrame([s_to_home, s_to_work, s_home_work]).T.sort_values(
            by='home work rate', ascending=True)

        # 山东省 2020-02-20 日新增的 202 是监狱中累计的人数，并非一天的人数，需要去掉
        region = 'Shandong' if self.__in_english else '山东'
        df_virus_daily_inc_injured.loc['2020-02-20', region] = 0
        if self.__in_english:
            del df_virus_daily_inc_injured['Hubei']
        else:
            del df_virus_daily_inc_injured['湖北']

        dfs_risk_new = []
        for df_data, col_name in zip([df_curve_in, df_move_in_risk, df_virus_daily_inc_injured],
                                     ['move', 'risk', 'new']):
            s_max_val = df_data.max()
            s_half_val = s_max_val / 2.0
            top_dates = {}
            half_dates = {}
            corrs = {}
            offsets = {}
            windows = {}
            index = df_data.index.tolist()
            for region in df_data.columns:
                if region in ['武汉', 'Wuhan', '温州', 'Wenzhou', '湖北', 'Hubei']:
                    continue
                top_date = None
                max_val = s_max_val[region]
                for date in index:
                    val = df_data.loc[date, region]
                    if val == max_val:
                        top_date = date
                        break
                assert top_date is not None
                top_dates[region] = top_date
                half_val = s_half_val[region]
                for row_idx in range(len(index))[::-1]:
                    last_val = df_data[region].values[row_idx - 1]
                    this_val = df_data[region].values[row_idx]
                    if last_val > half_val and this_val <= half_val:
                        half_date = index[row_idx]
                        half_dates[region] = half_date
                        s_corr = df_move_inc_corr.loc[half_date, region]
                        corrs[region] = s_corr['corr']
                        offsets[region] = s_corr['shift']
                        windows[region] = s_corr['window']
                        break

            s_top_date = pd.Series(top_dates)
            s_half_date = pd.Series(half_dates)
            s_top_date.values[:] = Util.str_dates_to_dates(s_top_date)
            s_half_date.values[:] = Util.str_dates_to_dates(s_half_date)
            s_turn_half_day_cnt = (s_half_date - s_top_date) / 86400 / 1e9
            s_corr = pd.Series(corrs)
            s_offset = pd.Series(offsets)
            s_window = pd.Series(windows)

            s_max_val.name = 'top value'
            s_top_date.name = 'top date'
            s_half_date.name = 'half date'
            s_half_val.name = 'half value'
            s_turn_half_day_cnt.name = 'turn half day cnt'
            s_corr.name = 'half date corr'
            s_offset.name = 'half date offset'
            s_window.name = 'half date window'

            df = pd.DataFrame([s_top_date, s_max_val, s_half_date, s_turn_half_day_cnt, s_corr,
                               s_offset, s_window]).T.sort_values(by='half date', ascending=True)
            for col, new_col in zip(['top date', 'half date'], ['top date idx', 'half date idx']):
                df[new_col] = [index.index(str(date)) if isinstance(date, datetime.date) else 0 for date in df[col]]
                dates = ['-'.join(str(date).split('-')[1:]) for date in df[col]]
                df[col].values[:] = dates
            df.columns = pd.MultiIndex.from_product([[col_name], df.columns])
            dfs_risk_new.append(df)

        df_diagnosed = self.df_virus_daily_injured.iloc[[-1]]
        if self.__in_english:
            del df_diagnosed['Hubei']
        else:
            del df_diagnosed['湖北']
        df_diagnosed = df_diagnosed.T
        df_diagnosed.columns = pd.MultiIndex.from_product([[''], ['diagnosed']])
        df_diagnosed = df_diagnosed.sort_values(by=('', 'diagnosed'), ascending=False)
        df_home_work = df_move[['home work rate']]
        df_home_work = df_home_work.sort_values(by=df_home_work.columns[0], ascending=True)
        df_home_work.columns = pd.MultiIndex.from_product([[''], df_home_work.columns])

        min_max_scaler = preprocessing.MinMaxScaler()
        X_min_max = min_max_scaler.fit_transform(df_diagnosed)
        df_diagnosed_normalized = pd.DataFrame(X_min_max, index=df_diagnosed.index, columns=df_diagnosed.columns)
        df_diagnosed_normalized.columns = pd.MultiIndex.from_product([[''], ['normalized diagnosed']])
        X_min_max = min_max_scaler.fit_transform(df_home_work)
        df_home_work_normalized = pd.DataFrame(X_min_max, index=df_home_work.index, columns=df_home_work.columns)
        df_home_work_normalized.columns = pd.MultiIndex.from_product([[''], ['normalized home work rate']])

        # df_risk_new = pd.concat([df_diagnosed, df_home_work] + dfs_risk_new, axis=1, sort=False)
        df_risk_new = pd.concat(
            [df_home_work, df_home_work_normalized, df_diagnosed, df_diagnosed_normalized] + dfs_risk_new,
            axis=1, sort=False)
        df_risk_new = df_risk_new.dropna(how='any', axis=0)
        df_risk_new[('move', 'top date idx')] = df_risk_new[('move', 'top date idx')].astype(np.int32)
        df_risk_new[('move', 'half date idx')] = df_risk_new[('move', 'half date idx')].astype(np.int32)
        df_risk_new[('new', 'top date idx')] = df_risk_new[('new', 'top date idx')].astype(np.int32)
        df_risk_new[('new', 'half date idx')] = df_risk_new[('new', 'half date idx')].astype(np.int32)
        df_risk_new[('', 'diagnosed')] = df_risk_new[('', 'diagnosed')].astype(np.int32)
        # df_risk_new[[('risk', 'turn half day cnt'), ('new', 'turn half day cnt')]]
        order_types = {}
        for region in df_risk_new.index:
            risk_top_date = df_risk_new.loc[region, ('risk', 'top date')]
            new_top_date = df_risk_new.loc[region, ('new', 'top date')]
            new_half_date = df_risk_new.loc[region, ('new', 'half date')]
            risk_half_date = df_risk_new.loc[region, ('risk', 'half date')]
            if risk_top_date <= new_top_date <= risk_half_date <= new_half_date:
                order_types[region] = 1
            elif risk_top_date <= new_top_date <= new_half_date <= risk_half_date:
                order_types[region] = 2
            elif risk_top_date <= risk_half_date <= new_top_date <= new_half_date:
                order_types[region] = 3
            elif new_top_date <= risk_top_date <= risk_half_date <= new_half_date:
                order_types[region] = 4
            elif new_top_date <= risk_top_date <= new_half_date <= risk_half_date:
                order_types[region] = 5
            elif new_top_date <= new_half_date <= risk_top_date <= risk_half_date:
                order_types[region] = 6
        df_risk_new['', 'ordered'] = pd.Series(order_types)
        return df_risk_new


def train_and_predict(start_date='2020-02-01', end_date=None, get_data_params=None, xgb_params=None, print_log=False):
    '''
    训练和预测
    '''
    util = Util()
    if end_date is None:
        end_date = datetime.date.today()
    else:
        end_date = util.str_date_to_date(end_date)
    predicts = OrderedDict()
    if get_data_params is None:
        get_data_params = {}
    assert isinstance(get_data_params, dict)
    if xgb_params is None:
        xgb_params = {}
    assert isinstance(xgb_params, dict)
    date = util.str_date_to_date(start_date)
    while date <= end_date:
        str_date = str(date)
        date += datetime.timedelta(days=1)
        analyzer = CoronavirusAnalyzer(str_date, True)
        data = analyzer.get_data(**get_data_params)
        model = analyzer.get_model(data, print_log=print_log, **xgb_params)
        s_X = data['df_X_test'].loc[str_date]
        predict = OrderedDict()
        for region in data['regions']:
            predict[region] = model.predict(s_X[region].values.reshape(1, -1))[0]
        predicts[str_date] = pd.Series(predict)
    return predicts
