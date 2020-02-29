from bs4 import BeautifulSoup
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
class DxyDailyCrawler:
    '''
    丁香园日频数据爬虫
    '''
    __data_url_template = 'https://ncov.dxy.cn/ncovh5/view/pneumonia_area?aid={}&from=dxy&link=&share=&source='
    __html_dir_path = 'data/html/dxy_daily_data'
    __pandas_df_dir_path = 'data/virus/dxy_daily_data'
    __pandas_df_path_template = __pandas_df_dir_path + '/{}.{}'
    __pandas_df_inc_dir_path = 'data/virus/dxy_daily_inc_data'
    __pandas_df_inc_path_template = __pandas_df_inc_dir_path + '/{}.{}'
    __h5_key = 'dxy_data'  # pandas 读写 h5 文件的 key

    def __init__(self):
        self.__util = Util()
        self.__region_ids = self.__util.huiyan_region_id

    def get_recent_html_dir_path(self):
        return '{}/{}'.format(self.__html_dir_path, sorted(os.listdir(self.__html_dir_path))[-1])

    @property
    def df_virus_daily(self):
        name = sorted(os.listdir(self.__pandas_df_dir_path))[-1].split('.')[0]
        path = self.__pandas_df_path_template.format(name, 'h5')
        return pd.read_hdf(path, self.__h5_key)

    @property
    def df_virus_daily_inc(self):
        name = sorted(os.listdir(self.__pandas_df_inc_dir_path))[-1].split('.')[0]
        path = self.__pandas_df_inc_path_template.format(name, 'h5')
        return pd.read_hdf(path, self.__h5_key)

    def crawl(self):
        date_time = str(datetime.datetime.now()).split('.')[0]
        dir_path = '{}/{}'.format(self.__html_dir_path, date_time)
        os.mkdir(dir_path)
        dfs = []
        dfs_inc = []
        for region, region_id in self.__region_ids.items():
            if int(region_id) % 1000 == 0:  # 省/直辖市才有数据
                self.logger.info('{} {} 数据爬取开始'.format(region, region_id))
                url = self.__data_url_template.format(region_id)
                html_text = None
                max_try_time = 3
                for try_time in range(1, max_try_time + 1):
                    try:
                        res = requests.get(url)
                        if res.status_code != 200:
                            raise ValueError('status_code 错误')
                        res.encoding = res.apparent_encoding
                        html_text = res.text
                        soup = BeautifulSoup(html_text, 'html.parser')
                        html_text = soup.prettify()
                        tree = etree.HTML(html_text)
                        nodes = tree.xpath('//html/body/script[@id="getAreaStat"]')
                        assert len(nodes) == 1
                        json_url = nodes[0].text.split('"locationId":{},"statisticsData":"'
                                                       .format(region_id))[-1].split('"')[0]
                        self.logger.info('{} {} json url: {}'.format(region, region_id, json_url))
                        res = json.loads(requests.get(json_url).text)
                        # print(res)
                        if not res['success']:
                            raise ValueError('success 为 False')
                        if len(res['data']) == 0:
                            raise ValueError('无数据')
                        for is_inc in [True, False]:
                            data = []
                            for one_data in res['data']:
                                _one_data = {}
                                if is_inc:
                                    cols = ['日期', '死亡', '治愈', '疑似', '确诊']
                                    dxy_cols = ['dateId', 'deadIncr', 'curedIncr', None, 'confirmedIncr']
                                else:
                                    cols = ['日期', '死亡', '治愈', '疑似', '确诊']
                                    dxy_cols = ['dateId', 'deadCount', 'curedCount', None, 'confirmedCount']
                                for col, dxy_col in zip(cols, dxy_cols):
                                    val = 0 if dxy_col is None else one_data[dxy_col]
                                    if col == '日期':
                                        date = str(val)
                                        val = '{}-{}-{}'.format(date[:4], date[4:6], date[6:])
                                    _one_data[col] = val
                                data.append(_one_data)
                            df = pd.DataFrame(data)
                            df = df.set_index('日期')
                            dates = df.index.tolist()
                            first_date = self.__util.str_date_to_date(dates[0])
                            while first_date > datetime.date(2020, 1, 11):
                                first_date -= datetime.timedelta(days=1)
                                dates.insert(0, str(first_date))
                            df = df.reindex(dates)
                            df.fillna(0, inplace=True)
                            # for col in df.columns:
                            #     df[col] = df[col].astype(np.int32)
                            df = df.astype(np.int32)
                            if region == '湖北':
                                idx = dates.index('2020-01-16')
                                vals = [4, 17, 59, 77, 72] if is_inc else [45, 62, 121, 198, 270]
                                for shift, val in enumerate(vals):
                                    df['确诊'].values[idx + shift] = val
                            df.columns = pd.MultiIndex.from_product([[region], df.columns])
                            if is_inc:
                                dfs_inc.append(df)
                            else:
                                dfs.append(df)
                        break
                    except Exception as e:
                        if try_time == max_try_time:
                            self.logger.error('{} {} 数据爬取出错，尝试了 {} 次都出错'
                                              .format(region, region_id, max_try_time))
                            raise e
                        self.logger.warning('{} {} 数据爬取出错，第 {} 次'.format(region, region_id, try_time))
                        time.sleep(1)
                f = open('{}/{}.html'.format(dir_path, region), 'w')
                f.writelines(html_text)
                f.close()
                self.logger.info('{} {} 数据爬取完成'.format(region, region_id))
        date = date_time.split(' ')[0]
        # 新增
        df_inc = pd.concat(dfs_inc, axis=1)
        path = self.__pandas_df_inc_path_template.format(date, 'h5')
        df_inc.to_hdf(path, self.__h5_key)
        path = self.__pandas_df_inc_path_template.format(date, 'xlsx')
        df_inc.to_excel(path)
        # 累计
        df = pd.concat(dfs, axis=1)
        path = self.__pandas_df_path_template.format(date, 'h5')
        df.to_hdf(path, self.__h5_key)
        path = self.__pandas_df_path_template.format(date, 'xlsx')
        df.to_excel(path)


if __name__ == '__main__':
    crawler = DxyDailyCrawler()
    crawler.crawl()
