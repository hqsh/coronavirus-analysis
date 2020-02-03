from collections import OrderedDict
from dxy_crawler import DxyCrawler
from huiyan_crawler import HuiyanCrawler
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans
from util.util import Util, with_logger
from weather_crawler import WeatherCrawler
import numpy as np
import pandas as pd
import datetime
import math


@with_logger
class CoronavirusAnalyzer:
    '''
    冠状病毒分析
    '''
    def __init__(self, last_date=None):
        '''
        :param str last_date: 最后一天，对日频数据有效，用于测试集，或者如果在凌晨，有地区当天数据还没公布的时候，需要去掉当天的数据，
                              格式是 yyyy-mm-dd
        :return:
        '''
        self.__util = Util()
        self.__huiyan_crawler = HuiyanCrawler()
        self.__weather_crawler = WeatherCrawler()
        if last_date is None:
            last_date = str(datetime.date.today() - datetime.timedelta(days=1))
        self.__last_date = last_date
        df = self.df_virus_daily_inc_injured
        no_inc_injured_regions = df.columns[df.iloc[-1] == 0]
        df = self.df_virus_daily_inc
        no_inc_regions = Util.get_multi_col_0(df)[(df.iloc[-1].values.reshape(-1, 4) != 0).sum(axis=1) == 0]
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
        df = DxyCrawler.load_dxy_data_frame('recent_daily', 'h5')
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df.astype(np.int32)

    @property
    def df_virus_daily_injured(self):
        '''
        日频病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus_daily).astype(np.int32)

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
        df = DxyCrawler.load_dxy_data_frame('recent_daily_inc', 'h5')
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df.astype(np.int32)

    @property
    def df_virus_daily_inc_injured(self):
        '''
        日频新增病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus_daily_inc)

    @property
    def df_virus_7_days_inc_injured(self):
        '''
        最近7天新增确诊人数（不含当天），2 级列索引
        :return:
        '''
        df_daily = self.df_virus_daily_inc_injured
        df_values = df_daily.values
        values = np.zeros(shape=(df_daily.shape[0], df_daily.shape[1] * 7), dtype=np.int32)
        for end_i in range(df_daily.shape[0]):
            start_i = end_i - 7
            if start_i < 0:
                start_i = 0
            arr = df_values[start_i: end_i]
            need_add_row_cnt = 7 - arr.shape[0]
            if need_add_row_cnt > 0:
                add_arr = np.zeros(shape=(need_add_row_cnt, arr.shape[1]), dtype=np.int32)
                arr = np.vstack([add_arr, arr])
            arr = arr.T.reshape(-1)
            values[end_i, :] = arr
        df_7_days = pd.DataFrame(values, index=df_daily.index, columns=pd.MultiIndex.from_product(
            [df_daily.columns, ['7天前', '6天前', '5天前', '4天前', '3天前', '2天前', '1天前']]))
        return df_7_days

    @property
    def df_virus_daily_inc_injured_cum_7(self):
        '''
        近 7 天累计新增确诊人数（不含当天的最近 7 天新增确诊人数和，即 1天前、2天前、......、7天前的确诊人数和）
        :return:
        '''
        df_virus_daily_inc_injured = self.df_virus_daily_inc_injured
        arr = df_virus_daily_inc_injured.values
        arr_cum_7 = np.zeros(shape=arr.shape, dtype=np.int32)
        for end_i in range(arr.shape[0]):
            start_i = 0 if end_i < 7 else end_i - 7
            arr_cum_7[end_i, :] = arr[start_i: end_i, :].sum(axis=0)
        return pd.DataFrame(arr_cum_7, index=df_virus_daily_inc_injured.index,
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
        return pd.DataFrame(df.values[:, idx::col_1_size], index=df.index, columns=regions)

    def del_city_regions(self, df, selected_cols_1=None, del_special_regions=False, inplace=True):
        '''
        删除 DataFrame 的普通城市，或特区列索引
        :param df:
        :param selected_cols_1: 保留的 1 级列索引（如果是多级列索引）
        :param del_special_regions: 是否删除特区
        :param inplace:
        :return:
        '''
        cols = ['香港', '澳门', '台湾'] if del_special_regions else []
        df_cols = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else df.columns
        for region in df_cols:
            if not self.__util.region_is_province(region):
                cols.append(region)
        return self.__util.del_cols(df, cols, selected_cols_1, inplace)

    def del_city_special_regions(self, df, selected_cols_1=None, inplace=True):
        return self.del_city_regions(df, selected_cols_1=selected_cols_1, del_special_regions=True, inplace=inplace)

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
    def df_move_in_injured(self):
        return self.get_df_move_in_injured()

    def get_df_move_in_injured(self, shift=1):
        '''
        进入某一地区的感染人流规模估算 = sigma ( 进入某一地区的各地人数规模 * 各来源地不含当天近 7 天的感染人数总和 ）
        :param shift: 偏移量，默认为 1，表示当天只能拿到前 shift 天的结算结果数据
        :return:
        '''
        df_cum_7 = self.df_virus_daily_inc_injured_cum_7
        ss = []
        for region in self.__util.huiyan_region_id:
            df_rate = self.__huiyan_crawler.get_rate(region)
            if df_rate.size > 0:
                df_rate = self.__huiyan_crawler.get_rate(region)[['省', '规模']]
                df_rate.index = pd.MultiIndex.from_arrays([df_rate.index, df_rate['省']])
                del df_rate['省']
                s_cum_7 = df_cum_7.stack(0)
                s_cum_7.index.names = ['日期', '省']
                df_rate = df_rate.loc['2020-01-10':]
                s_rate = df_rate['规模']
                df = pd.DataFrame({'风险规模': s_rate * s_cum_7})
                df.fillna(0, inplace=True)
                df = df.unstack(1)
                s = df.sum(axis=1)
                s.name = region
                ss.append(s)
        df = pd.DataFrame(ss).T.sort_index(axis=1)
        df = df.loc[: self.__last_date]
        if shift > 1:
            df.values[shift:, :] = df.values[:-shift, :]
            df.values[:shift, :] = 0
        df = df.loc['2020-01-11':]
        return df

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
            label_to_cols = None
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
