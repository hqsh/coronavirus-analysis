from collections import OrderedDict
from lxml import etree
from util.util import Util, with_logger
import numpy as np
import pandas as pd
import datetime
import json
import os
import requests
import time


@with_logger
class DxyCrawler:
    '''
    丁香园爬虫
    '''
    # pandas 读写 h5 文件的 key
    __h5_key = 'dxy_data'
    # 疫情地图
    __data_url = 'https://3g.dxy.cn/newh5/view/pneumonia?sf=1&dn=2&from=singlemessage'
    # 实时播报
    __news_url = 'https://3g.dxy.cn/newh5/view/pneumonia/timeline'
    # 保存的文件前缀
    __file_name_perfix = 'dxy_data'

    def __init__(self, run_mode='live'):
        '''
        :param run_mode: 执行模式：live 代表从当前最新的数据起，实时更新；init 代表从 init_data 开始，通过历史 html 文件构造出最新数据
        :return:
        '''
        self.__key_cities = Util().key_cities
        self.__retry_sleep_seconds = 60
        self.__run_mode = run_mode
        try:
            if self.__run_mode == 'live':
                path = self.get_dxy_file_path('recent')
            elif self.__run_mode == 'init':
                path = self.get_file_path('init_data')
            else:
                raise ValueError('run_mode')
            self.__recent_df = pd.read_hdf(path, self.__h5_key)
            self.__recent_update_date_time = ' '.join(list(self.__recent_df.index.values[-1]))
        except FileNotFoundError:
            self.logger.warning('没有读取到历史数据')
            self.__recent_df = None
            self.__recent_update_date_time = '无历史数据'
        self.__recent_daily_df = None
        self.__recent_daily_inc_df = None
        self.__sorted_provinces = None
        self.logger.info('初始化完成，最近一次统计时间：{}，额外统计的城市：{}'
                         .format(self.__recent_update_date_time, '、'.join(self.__key_cities)))

    @classmethod
    def get_file_path(cls, name, file_name_append='h5', file_name_perfix=None):
        '''
        获取存取的文件路径
        :param name: 名称
        :param file_name_append: 文件后缀名
        :param file_name_perfix: 文件前缀名
        :return:
        '''
        dir = 'data/virus' if file_name_append in ['h5', 'xlsx'] else 'data'
        if file_name_append == 'html':
            dir = '{}/html'.format(dir)
        if name == 'init_data' or file_name_perfix is None:
            return '{}/{}.{}'.format(dir, name, file_name_append)
        return '{}/{}_{}.{}'.format(dir, cls.__file_name_perfix, name, file_name_append)

    @classmethod
    def get_dxy_file_path(cls, name, file_name_append='h5'):
        '''
        获取存取的文件路径
        :param name: 名称
        :param file_name_append: 文件后缀名
        :return:
        '''
        return cls.get_file_path(name, file_name_append, cls.__file_name_perfix)

    @classmethod
    def load_data_frame(cls, name, file_name_append='h5', file_name_perfix=None):
        '''
        加载 pandas DataFrame
        :param name: 名称
        :param file_name_append: 文件后缀名
        :param file_name_perfix: 文件前缀名
        :return:
        '''
        path = cls.get_file_path(name, file_name_append, file_name_perfix)
        if file_name_append == 'h5':
            return pd.read_hdf(path, cls.__h5_key)
        if file_name_append == 'xlsx':
            return pd.read_excel(path)
        if file_name_append == 'csv':
            return pd.read_csv(path)
        raise ValueError('file_name_append 错误')

    @classmethod
    def load_dxy_data_frame(cls, name, file_name_append='h5'):
        '''
        加载 pandas DataFrame
        :param name: 名称
        :param file_name_append: 文件后缀名
        :return:
        '''
        return cls.load_data_frame(name, file_name_append, cls.__file_name_perfix)

    @property
    def html_file_paths(self):
        '''
        html 文件路径的迭代器，按文件名排序
        :return:
        '''
        file_paths = []
        for file_name in os.listdir('data/html'):
            if file_name.endswith('.html'):
                file_paths.append('data/html/{}'.format(file_name))
        file_paths.sort()
        for path in file_paths:
            yield path

    def __save_recent_files(self, html_text=None):
        '''
        保存最新全量爬取的数据，excel 数据只用来看，pandas 读取后需要额外调整格式，麻烦，所以存取历史数据用 h5 文件；以及 html 文件
        :param html_text: html 原始文本，live 模式执行，要保存 html，需要填此参数
        :return:
        '''
        if self.__run_mode == 'live' and html_text is not None:
            # 备份 html 数据
            html_path = self.get_dxy_file_path(self.__recent_update_date_time, 'html')
            file = open(html_path, 'w')
            file.writelines(html_text)
            file.close()
        self.__calc_daily()
        for df, name in zip([self.__recent_df, self.__recent_daily_df, self.__recent_daily_inc_df],
                            ['recent', 'recent_daily', 'recent_daily_inc']):
            df.to_hdf(self.get_dxy_file_path(name), self.__h5_key)
            df.to_excel(self.get_dxy_file_path(name, 'xlsx'))

    def __calc_daily(self):
        '''
        将数据转换成日频和日频增量数据，如果当天有更新，每列取当天第一条更新的数据
        :return:
        '''
        first_date = self.__recent_df.index.levels[0][0]
        last_date = self.__recent_df.index.levels[0][-1]
        year, month, day = first_date.split('-')
        date = datetime.date(year=int(year), month=int(month), day=int(day))
        year, month, day = last_date.split('-')
        last_date = datetime.date(year=int(year), month=int(month), day=int(day))
        index = []
        while date <= last_date:
            index.append(str(date))
            date += datetime.timedelta(days=1)
        arr = np.zeros(shape=(len(index), int(self.__recent_df.shape[1] * 4 / 5)), dtype=np.float64)
        arr[:] = np.nan
        df = pd.DataFrame(arr)
        df.index = pd.Index(index, name='日期')
        cols = self.__recent_df.columns.levels[1].tolist()
        cols.remove('是否更新')
        df.columns = pd.MultiIndex.from_product([self.__recent_df.columns.levels[0].tolist(), cols])
        for idx in index:
            if idx in self.__recent_df.index.levels[0]:
                df_all_sliced = self.__recent_df.loc[idx]
                for region in df_all_sliced.columns.levels[0]:
                    df_sliced = df_all_sliced[region]
                    i = None
                    for i, is_updated in enumerate(df_sliced['是否更新']):
                        if is_updated:
                            break
                    if i is not None:
                        for col in cols:
                            df.loc[idx, (region, col)] = df_sliced[col][i]
        df.fillna(0, inplace=True)
        df = df.loc['2020-01-11':]  # 2020-01-11 之前的数据有大量缺失和不准，去掉
        sorted_provinces = ['全国'] + self.__sorted_provinces
        self.__recent_daily_df = df[sorted_provinces]
        arr = self.__recent_daily_df.values
        arr = arr[1:] - arr[:-1]
        df = pd.DataFrame(arr, index=self.__recent_daily_df.index[1:], columns=self.__recent_daily_df.columns)
        self.__recent_daily_inc_df = df[sorted_provinces]

    def run(self):
        '''
        持续爬取更新的数据或重演历史数据，合并和更新数据，保存到文件中
        :return:
        '''
        is_first_loop = True
        html_file_paths = self.html_file_paths
        while True:
            try:
                if is_first_loop:
                    is_first_loop = False
                elif self.__run_mode == 'live':
                    time.sleep(self.__retry_sleep_seconds)
                if self.__run_mode == 'init':
                    try:
                        file_path = html_file_paths.__next__()
                    except StopIteration:
                        self.logger.info('历史数据处理完，开始写入到文件')
                        break
                    file = open(file_path)
                    lines = file.readlines()
                    html_text = '\n'.join(lines)
                else:
                    res = requests.get(self.__data_url)
                    if res.status_code != 200:
                        self.logger.error('http status code: {}, {} 秒后重试'
                                          .format(res.status_code, self.__retry_sleep_seconds))
                        continue
                    res.encoding = 'utf-8'
                    html_text = res.text
                tree = etree.HTML(html_text)
                # print(html_text)
                # print(etree.tostring(tree, encoding="utf-8", pretty_print=True).decode("utf-8"))
                nodes = tree.xpath('//script[@id="getAreaStat"]')
                if len(nodes) != 1:
                    self.logger.error('nodes 数量不为 1，为：{}, {} 秒后重试'
                                      .format(len(nodes), self.__retry_sleep_seconds))
                    continue
                update_date_time = tree.xpath('//p[@class="mapTitle___2QtRg"]/span/text()')
                if self.__run_mode == 'init' and len(update_date_time) == 0:  # 老版本的 html
                    update_date_time = tree.xpath('//p[@class="mapTitle___2QtRg"]')[0].text
                    _, update_date, update_time, _ = update_date_time.split(' ')
                    update_date_time = '{} {}'.format(update_date, update_time)
                else:
                    if len(update_date_time) != 1:
                        self.logger.error('update_date_time 数量不为 1，为：{}, {} 秒后重试'
                                          .format(len(nodes), self.__retry_sleep_seconds))
                        continue
                    update_date_time = update_date_time[0].split('截至 ')[1].split('（北京时间）')[0].split(' ')[:2]
                    update_date = update_date_time[0]
                    update_time = update_date_time[1]
                    update_date_time = ' '.join(update_date_time)
                if self.__recent_update_date_time == update_date_time:
                    self.logger.info('和最近一次更新时间 {} 相同，等待 {} 秒后重试'
                                     .format(self.__recent_update_date_time, self.__retry_sleep_seconds))
                    continue
                infos_text = nodes[0].text
                infos_text = infos_text[len('try { window.getAreaStat = '):]
                infos_text = infos_text.split('}catch(e)')[0]
                infos = json.loads(infos_text)
                data = OrderedDict()
                self.__sorted_provinces = []
                for info in infos:
                    province = info['provinceShortName']
                    if province == '湖北':
                        for city_info in info['cities']:
                            city = city_info['cityName']
                            if city in self.__key_cities:
                                data[city] = OrderedDict()
                                data[city]['确诊'] = city_info['confirmedCount']
                                data[city]['疑似'] = city_info['suspectedCount']
                                data[city]['死亡'] = city_info['deadCount']
                                data[city]['治愈'] = city_info['curedCount']
                                data[city]['是否更新'] = False
                                self.__sorted_provinces.append(city)
                    self.__sorted_provinces.append(province)
                    data[province] = OrderedDict()
                    data[province]['确诊'] = info['confirmedCount']
                    data[province]['疑似'] = info['suspectedCount']
                    data[province]['死亡'] = info['deadCount']
                    data[province]['治愈'] = info['curedCount']
                    data[province]['是否更新'] = False
                    try:
                        for comment in info['comment'].split('，'):
                            for key in ['死亡', '治愈']:
                                if key in comment:
                                    for word in comment.split(' '):
                                        cnt = None
                                        try:
                                            cnt = int(word)
                                            break
                                        except ValueError:
                                            continue
                                    if cnt is not None and cnt > data[province][key]:
                                        self.logger.warning('{}{}人数在 comment 中（{}人）比 info 中（{}人）有更多'
                                                            .format(province, key, cnt, data[province][key]))
                                        data[province][key] = cnt
                    except Exception as e:
                        self.logger.error('comment：{}，解析有未知错误：{}，忽略'.format(info['comment'], e))
                df = pd.DataFrame(data)
                df.index = pd.MultiIndex.from_product([[update_date], [update_time], df.index])
                df = df.unstack()
                df.index.names = ['日期', '时间']
                if self.__recent_df is not None:
                    df = self.__recent_df.append(df)
                df = df[self.__sorted_provinces]
                # 处理空数据
                df.fillna(method='pad', inplace=True)
                df.fillna(0, inplace=True)
                df = pd.DataFrame(df.values.astype(np.int32), index=df.index, columns=df.columns)
                # 计算全国数据
                total_data = OrderedDict()
                for key in df.columns.levels[1]:
                    for province in self.__sorted_provinces:
                        if province not in self.__key_cities:
                            if key not in total_data:
                                total_data[key] = df[province][key].values
                            else:
                                if key == '是否更新':
                                    total_data[key] |= df[province][key].values
                                else:
                                    total_data[key] += df[province][key].values
                total_df = pd.DataFrame(total_data, index=df.index)
                total_df.columns = pd.MultiIndex.from_product([['全国'], total_df.columns.values])
                df = pd.concat([total_df, df], axis=1)
                # 设置是否更新字段
                index_1 = df.index[-1]
                index_2 = df.index[-2]
                for region in df.columns.levels[0]:
                    for col in ['确诊']:
                        if df.loc[index_1, (region, col)] != df.loc[index_2, (region, col)]:
                            df.loc[index_1, (region, '是否更新')] = 1
                            break
                self.__recent_df = df
                self.__recent_update_date_time = update_date_time
                # 保存数据
                if self.__run_mode == 'live':
                    self.__save_recent_files(html_text)
                self.logger.info('数据已更新，更新日期时间：{}'.format(update_date_time))
            except Exception as e:
                self.logger.error('未知异常：{}，{} 秒后重试'.format(e, self.__retry_sleep_seconds))
        self.__save_recent_files()
        self.logger.info('历史数据构造完成')


if __name__ == '__main__':
    run_modes = {1: 'init', 2: 'live'}
    crawler = DxyCrawler(run_modes[2])
    crawler.run()
