from collections import OrderedDict
from lxml import etree
import numpy as np
import pandas as pd
import json
import logging
import os
import requests
import time


# 丁香园爬虫
class DxyCrawler:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def __init__(self, run_mode='live', cities=None):
        '''
        :param run_mode: 执行模式：live 代表从当前最新的数据起，实时更新；init 代表从 init_data 开始，通过历史 html 文件构造出最新数据
        :param cities: 全国省、直辖市、港澳台都会统计，如果需要额外统计的城市，以列表形式列出
        :return:
        '''
        # 疫情地图
        self.__data_url = 'https://3g.dxy.cn/newh5/view/pneumonia?sf=1&dn=2&from=singlemessage'
        # 实时播报
        self.__news_url = 'https://3g.dxy.cn/newh5/view/pneumonia/timeline'
        # 保存的文件前缀
        self.__file_name_perfix = 'dxy_data'
        # pandas 读写 h5 文件的 key
        self.__h5_key = 'dxy_data'
        self.__retry_sleep_seconds = 60
        self.logger = logging.getLogger(__name__)
        self.__run_mode = run_mode
        self.__cities = [] if cities is None else cities
        try:
            if self.__run_mode == 'live':
                path = self.get_file_path('recent')
            elif self.__run_mode == 'init':
                path = self.get_file_path('init_data')
            else:
                raise ValueError('run_mode')
            self.__recent_df = pd.read_hdf(path, self.__h5_key)
            self.__recent_update_date_time = ' '.join(list(self.__recent_df.index.values[-1]))
        except FileNotFoundError:
            self.logger.warning('没有读取到历史数据')
            self.__recent_df = self.__recent_update_date_time = None
        self.logger.info('初始化完成，最近一次统计时间：{}，额外统计的城市：{}'
                         .format(self.__recent_update_date_time, '、'.join(self.__cities)))

    def get_file_path(self, update_date_time, file_name_append='h5'):
        '''
        获取存取的文件路径
        :param update_date_time: 网址更新的日期时间
        :param file_name_append: 文件后缀名
        :return:
        '''
        if update_date_time == 'init_data':
            return 'data/{}.{}'.format(update_date_time, file_name_append)
        return 'data/{}_{}.{}'.format(self.__file_name_perfix, update_date_time, file_name_append)

    @property
    def html_file_paths(self):
        '''
        html 文件路径的迭代器，按文件名排序
        :return:
        '''
        file_paths = []
        for file_name in os.listdir('data'):
            if file_name.endswith('.html'):
                file_paths.append('data/{}'.format(file_name))
        file_paths.sort()
        for path in file_paths:
            yield path

    def crawl_data(self):
        '''
        持续爬取更新的数据，并合并历史数据，保存到文件中
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
                        self.logger.info('历史数据构造完成')
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
                provinces = []
                for info in infos:
                    province = info['provinceShortName']
                    if province == '湖北':
                        for city_info in info['cities']:
                            city = city_info['cityName']
                            if city in self.__cities:
                                data[city] = OrderedDict()
                                data[city]['确诊'] = city_info['confirmedCount']
                                data[city]['疑似'] = city_info['suspectedCount']
                                data[city]['死亡'] = city_info['deadCount']
                                data[city]['治愈'] = city_info['curedCount']
                                provinces.append(city)
                    provinces.append(province)
                    data[province] = OrderedDict()
                    data[province]['确诊'] = info['confirmedCount']
                    data[province]['疑似'] = info['suspectedCount']
                    data[province]['死亡'] = info['deadCount']
                    data[province]['治愈'] = info['curedCount']
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
                df = df[provinces]
                # 处理空数据
                df.fillna(method='pad', inplace=True)
                df.fillna(0, inplace=True)
                df = pd.DataFrame(df.values.astype(np.int32), index=df.index, columns=df.columns)
                # 计算全国数据
                total_data = OrderedDict()
                for key in df.columns.levels[1]:
                    for province in provinces:
                        if key not in total_data:
                            total_data[key] = df[province][key].values
                        else:
                            total_data[key] += df[province][key].values
                total_df = pd.DataFrame(total_data, index=df.index)
                total_df.columns = pd.MultiIndex.from_product([['全国'], total_df.columns.values])
                df = pd.concat([total_df, df], axis=1)
                # excel 数据只用来看，pandas 读取后需要额外调整格式，麻烦，所以存取历史数据用 h5 文件
                if self.__run_mode == 'init':
                    df.to_hdf(self.get_file_path('recent'), self.__h5_key)
                    df.to_excel(self.get_file_path('recent', 'xlsx'))
                else:
                    df.to_hdf(self.get_file_path('recent'), self.__h5_key)
                    df.to_excel(self.get_file_path('recent', 'xlsx'))
                    # df.to_hdf(self.get_file_path(update_date_time), self.__h5_key)
                    # df.to_excel(self.get_file_path(update_date_time, 'xlsx'))
                    # 备份 html 数据
                    html_path = self.get_file_path(update_date_time, 'html')
                    file = open(html_path, 'w')
                    file.writelines(html_text)
                    file.close()
                self.__recent_df = df
                self.__recent_update_date_time = update_date_time
                self.logger.info('数据已更新，更新日期时间：{}'.format(update_date_time))
            except RuntimeError as e:
                self.logger.error('未知异常：{}，{} 秒后重试'.format(e, self.__retry_sleep_seconds))


if __name__ == '__main__':
    # run_mode = 'init'
    run_mode = 'live'
    crawler = DxyCrawler(run_mode, ['武汉'])
    crawler.crawl_data()
