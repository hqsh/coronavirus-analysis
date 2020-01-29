from collections import OrderedDict
from dxy_crawler import DxyCrawler
from sklearn.cluster import KMeans
from util.util import Util, with_logger
from weather_crawler import WeatherCrawler
import matplotlib.pyplot as plt
import pandas as pd
import math


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


@with_logger
class CoronavirusAnalyzer:
    '''
    冠状病毒分析
    '''
    def __init__(self, last_date=None):
        '''
        :param str last_date: 最后一天，用于测试集，或者如果在凌晨，有地区当天数据还没公布的时候，需要去掉当天的数据，
                              格式是 yyyy-mm-dd
        :return:
        '''
        self.__weather_crawler = WeatherCrawler()
        self.__last_date = last_date

    @property
    def df_virus(self):
        '''
        实时病毒感染数据
        :return:
        '''
        df = DxyCrawler.load_dxy_data_frame('recent', 'h5')
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df

    @property
    def df_virus_injured(self):
        '''
        实时病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus)

    @property
    def df_virus_daily(self):
        '''
        日频病毒感染数据
        :return:
        '''
        df = DxyCrawler.load_dxy_data_frame('recent_daily', 'h5')
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df

    @property
    def df_virus_daily_injured(self):
        '''
        日频病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus_daily)

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
        return df

    @property
    def df_virus_daily_inc_injured(self):
        '''
        日频新增病毒感染数据中的确诊人数
        :return:
        '''
        return self.get_injured(self.df_virus_daily_inc)

    @staticmethod
    def get_injured(df):
        '''
        获取确诊人数列，2 级列索引变成 1 级列索引
        :param df:
        :return:
        '''
        regions = df.columns.levels[0][df.columns.codes[0][::4]]
        return pd.DataFrame(df.values[:, 3::4], index=df.index, columns=regions)

    @property
    def df_distance(self):
        '''
        各地相互距离数据
        :return:
        '''
        return pd.read_csv('data/全国各地之间距离.csv', index_col=0)

    @property
    def df_info(self):
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
    def df_weather_average(self):
        '''
        各地最近平均天气数据
        :return:
        '''
        df = self.get_weather_average_data()
        if self.__last_date is not None:
            df = df.loc[: self.__last_date]
        return df

    def get_weather_average_data(self, start_shift=14, end_shift=3):
        '''
        获取天气的加权（权重值线性递增）平均数据，从地区有首例确诊病人往前找 start_shift 天，到 end_shift 天前，之间的数据取均值
        专家分析：从已有病例来看，这一新型冠状病毒的潜伏期平均7天左右，短的2到3天，长的12天
        :param start_shift: 默认取最大潜伏期的值
        :param end_shift: 默认取最小潜期的伏值
        :return:
        '''
        return self.__weather_crawler.get_weather_average_data(self.df_virus_daily, start_shift, end_shift)

    @staticmethod
    def k_means(df, n_clusters=8, init='k-means++', n_init=10,
                max_iter=300, tol=1e-4, precompute_distances='auto',
                verbose=0, random_state=None, copy_x=True,
                n_jobs=1, algorithm='auto', try_times=1, insure_sorted=False):
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
        :return:
        '''

        res_clf = res_col_in_cluster = res_cluster_centers = min_inertia = None
        for _ in range(try_times):
            label_to_cols = None
            while True:
                clf = KMeans(n_clusters=n_clusters, max_iter=n_clusters, n_jobs=n_jobs)
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
        return res_clf, res_col_in_cluster, res_cluster_centers

    @staticmethod
    def subplots(df, ncols=4):
        '''
        对 df 每列画一个折线图
        :param df: 1 级行、列索引的 DataFrame
        :param ncols:
        :return:
        '''
        assert not isinstance(df.index, pd.MultiIndex)
        assert not isinstance(df.columns, pd.MultiIndex)
        nrows = math.ceil(df.shape[1] / ncols)
        ncols = 4
        fig, ax = plt.subplots(figsize=(ncols * 5, nrows * 3), nrows=nrows, ncols=ncols)
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
        plt.show()
