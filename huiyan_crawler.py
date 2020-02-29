from util.util import Util, with_logger
import numpy as np
import pandas as pd
import datetime


@with_logger
class HuiyanCrawler:
    # 迁徙规模指数：反映迁入或迁出人口规模，城市间可横向对比
    # 城市迁徙边界采用该城市行政区划，包含该城市管辖的区、县、乡、村
    # 2020-02-19 增加百度迁徙新支持的按省/直辖市统计（省份级别），原按城市统计（城市级别）并聚类为省份级别的方法和数据保留，但不再使用（只能获取top 100城市，精度略低）
    __curve_url_template = 'https://huiyan.baidu.com/migration/historycurve.json?' \
                                  'dt={}&id={}&type=move_{}&startDate={}&endDate={}'
    # 迁徙各地的比例
    __rank_url_template = 'https://huiyan.baidu.com/migration/{}rank.json?dt={}&id={}&type=move_{}&date={}'

    def __init__(self, end_date=None):
        self.__util = Util()
        self.__start_date = datetime.date(2020, 1, 1)  # 该网址能查询到到最早统计日期是 2020-01-01
        if end_date is None:
            self.__end_date = datetime.date.today() - datetime.timedelta(days=1)  # 不包含当天数据
        else:
            self.__end_date = self.__util.str_date_to_date(str(end_date))
        self.__df_curve_in = None
        self.__df_curve_out = None
        self.__df_rank_in = None
        self.__df_rank_out = None

    def get_curve_url(self, region, move_type, start_date, end_date):
        '''
        获取迁徙规模指数的 url
        :param region:
        :param move_type: 'in' 或 'out'，表示进入或出去
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return:
        '''
        dt = 'province' if self.__util.region_is_province(region) else 'city'
        region_id = self.__util.get_huiyan_region_id(region)
        move_type = move_type
        start_date = str(start_date).replace('-', '')
        end_date = str(end_date).replace('-', '')
        return self.__curve_url_template.format(dt, region_id, move_type, start_date, end_date)

    def get_rank_url(self, region, move_type, date, is_city=True):
        '''
        获取迁徙流向的 url
        :param region:
        :param move_type: 'in' 或 'out'，表示进入或出去
        :param date: 日期
        :param is_city:
        :return:
        '''
        dt = 'province' if self.__util.region_is_province(region) else 'city'
        region_id = self.__util.get_huiyan_region_id(region)
        move_type = move_type
        date = str(date).replace('-', '')
        return self.__rank_url_template.format('city' if is_city else 'province', dt, region_id, move_type, date)

    def get_path(self, file_type, move_type, region=None, file_append='csv', is_city=True):
        '''
        :param file_type: curve 或 rank 或 rate
        :param move_type:
        :param region:
        :param file_append: csv 或 excel
        :param is_city:
        :return:
        '''
        if region is None:
            return 'data/huiyan/{}-{}.{}'.format(file_type, move_type, file_append)
        if file_type in ['rank', 'rate'] and not is_city:
            return 'data/huiyan/{}_province-{}-{}.{}'.format(file_type, region, move_type, file_append)
        return 'data/huiyan/{}-{}-{}.{}'.format(file_type, region, move_type, file_append)

    def __load_rank_df(self, region, move_type, need_index=True, is_city=True):
        path = self.get_path('rank', move_type, region, 'csv', is_city=is_city)
        try:
            if need_index:
                return pd.read_csv(path, index_col=0)
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame([])

    def df_rank_in(self, region, is_city=True):
        return self.__load_rank_df(region, 'in', is_city=is_city)

    def df_rank_out(self, region, is_city=True):
        return self.__load_rank_df(region, 'out', is_city=is_city)

    def __load_curve_df(self, move_type):
        path = self.get_path('curve', move_type, file_append='csv')
        try:
            return pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            return pd.DataFrame([])

    @property
    def df_curve_in(self):
        return self.__load_curve_df('in')

    @property
    def df_curve_out(self):
        return self.__load_curve_df('out')

    @staticmethod
    def get_df_move_inc_corr_path(consider_population=False, file_type='h5', n=3, shift_one_day=False, sample_cnt=0):
        return 'data/huiyan/人流风险和新增确诊/人流风险和新增确诊{}{}-n={}-sample_cnt={}.{}'.format(
            '-考虑人口' if consider_population else '', '' if shift_one_day else '-考虑当天', n, sample_cnt, file_type)

    def get_rate(self, region, move_type='in', is_city=False):
        '''
        region:
        move_type:
        is_city: True：按城市区分的 rank 算出的 rate；False：按省/直辖市区分的 rank 算出的 rate
        '''
        path = self.get_path('rate', move_type, region, is_city=is_city)
        try:
            return pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            return pd.DataFrame([])

    def crawl(self, is_city=True):
        data_rank_in = {}
        data_rank_out = {}
        df_curve_in = pd.DataFrame([])
        df_curve_out = pd.DataFrame([])
        dfs_rank = {}
        for region, region_id in self.__util.huiyan_region_id.items():
            data_rank_in[region] = []
            data_rank_out[region] = []
            for move_type, data_rank, df_curve in zip(['in', 'out'], [data_rank_in[region], data_rank_out[region]],
                                                      [df_curve_in, df_curve_out]):
                if is_city:
                    # get http curve
                    url = self.get_curve_url(region, move_type, self.__start_date, self.__end_date)
                    res = self.__util.http_request(url, '有异常，region：{}，url：{}'.format(region, url))
                    if res['errno'] == 0:
                        s = pd.Series(res['data']['list'])
                        if s.size == 0:
                            self.logger.warning('region：{}，url：{}，无数据'.format(region, url))
                        df_curve[region] = s
                    else:
                        self.logger.error('有异常，region：{}，url：{}, errmsg：{}'.format(region, url, res['errmsg']))
                # get http rank
                date = self.__start_date
                df = self.__load_rank_df(region, move_type, is_city=is_city)
                dfs_rank[(region, move_type)] = df
                if df.size > 0:
                    date = self.__util.str_date_to_date(df.index[-1]) + datetime.timedelta(days=1)
                self.logger.info('开始统计[{}]的[{}]人流比例数据（{}），{}～{}'.format(
                        region, '进入' if move_type == 'in' else '外出', '按[{}]统计'
                        .format('城市' if is_city else '省/直辖市'), date, self.__end_date))
                while date <= self.__end_date:
                    str_date = str(date)
                    url = self.get_rank_url(region, move_type, date, is_city=is_city)
                    print(url)
                    res = self.__util.http_request(url, '有异常，region：{}，url：{}'.format(region, url))
                    if res['errno'] == 0:
                        for d in res['data']['list']:
                            d['date'] = str_date
                            if is_city:
                                city_name = d['city_name']
                                if city_name.endswith('市'):
                                    d['city_name'] = city_name[:-1]
                            province_name = d['province_name']
                            d['province_name'] = province_name[:2] if self.__util.region_is_province(province_name[:2])\
                                else province_name[:3]
                        data_rank += res['data']['list']
                    else:
                        self.logger.error('有异常，region：{}，url：{}, errmsg：{}'.format(region, url, res['errmsg']))
                    date += datetime.timedelta(days=1)
            self.logger.info('{} 处理完成'.format(region))
        # save rank
        col_en_to_cn = {'city_name': '市', 'province_name': '省', 'value': '比例'}
        if is_city:
            col_en_to_cn['city_name'] = '市'
        for region, region_id in self.__util.huiyan_region_id.items():
            for move_type, data in zip(['in', 'out'], [data_rank_in[region], data_rank_out[region]]):
                df = pd.DataFrame(data)
                if df.size > 0:
                    df = df.set_index('date')
                    df.index.name = '日期'
                    df['value'] = df['value'].values / 100
                    df.columns = [col_en_to_cn[col] for col in df.columns]
                    old_df = dfs_rank[(region, move_type)]
                    df = old_df.append(df)
                    df.to_csv(self.get_path('rank', move_type, region, is_city=is_city))
        if is_city:
            # save curve
            for move_type, df_curve in zip(['in', 'out'], [df_curve_in, df_curve_out]):
                if df_curve.size > 0:
                    df_curve = df_curve.sort_index()
                    dates = []
                    for date in df_curve.index:
                        date = '{}-{}-{}'.format(date[:4], date[4: 6], date[6:])
                        dates.append(date)
                    df_curve.index = pd.Series(dates, name='日期')
                    df_curve = df_curve.sort_index(axis=0)
                    df_curve = df_curve.sort_index(axis=1)
                    path = self.get_path('curve', move_type)
                    df_curve.to_csv(path)
        self.logger.info('数据爬取完毕')
        self.calc_rate_by_province(is_city)

    def calc_rate_by_province(self, is_city=True):
        '''
        对 rank 原始数据，按省、直辖市归类计算进出人流规模
        :return:
        '''
        s_population = pd.read_csv('data/全国各地信息.csv', index_col=0)['人口'] / 1e8
        for move_type in ['in', 'out']:
            df_curve = self.__load_curve_df(move_type)
            for region in self.__util.huiyan_region_id:
                s_curve = df_curve[region]
                df = self.__load_rank_df(region, move_type, need_index=False, is_city=is_city)
                if df.size > 0:
                    if is_city:
                        del df['市']
                        df = df.groupby(['日期', '省']).sum()
                        df['规模'] = df['比例'] * s_curve
                        arr_scope = df['规模'].values
                        df_index_province = df.index.to_frame()['省'].values.astype('U')
                        unique_provinces = np.unique(df_index_province)
                        arr_scope_by_population = np.zeros(shape=(df.shape[0], ), dtype=np.float64)
                        for province in unique_provinces:
                            mask = df_index_province == province
                            population = s_population[province]
                            arr_scope_by_population[mask] = arr_scope[mask] / population
                        df['规模/人口'] = arr_scope_by_population
                    else:
                        region_curve = np.array([s_curve[date] for date in df['日期']])
                        df['规模'] = df['比例'].values * region_curve
                        df['规模/人口'] = df['规模'].values / s_population[df['省']].values
                    path = self.get_path('rate', move_type, region, is_city=is_city)
                    if is_city:
                        df.to_csv(path)
                    else:
                        df.to_csv(path, index=None)
        self.logger.info('按省、直辖市归类计算进出人流规模')

    def run(self):
        self.crawl()
        self.crawl(False)


if __name__ == '__main__':
    crawler = HuiyanCrawler('2020-02-27')
    crawler.run()
