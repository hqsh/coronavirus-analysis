from collections import OrderedDict
from util.util import Util, with_logger
from lxml import etree
import numpy as np
import pandas as pd
import datetime
import os
import requests
import time


@with_logger
class WeatherCrawler:
    '''
    下载历史天气数据
    '''
    # pandas 读写 h5 文件的 key
    __h5_key = 'weather'
    # 保存的文件前缀
    __file_name_perfix = 'weather'
    # url
    __url_template = 'http://www.tianqihoubao.com/lishi/{}/month/{}.html'
    # file path
    __file_path = 'data/weather'
    # 合法的天气和对应的数字数据
    __weather_info = {
        '晴': {'晴朗度': 3, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 0},
        '多云': {'晴朗度': 2, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 0},
        '阴': {'晴朗度': 1, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 0},
        '雾': {'晴朗度': 0, '雾度': 1, '霾度': 0, '降雨量': 0, '降雪量': 0},
        '大雾': {'晴朗度': 0, '雾度': 2, '霾度': 0, '降雨量': 0, '降雪量': 0},
        '霾': {'晴朗度': 0, '雾度': 0, '霾度': 1, '降雨量': 0, '降雪量': 0},
        '中度霾': {'晴朗度': 0, '雾度': 0, '霾度': 2, '降雨量': 0, '降雪量': 0},
        '重度霾': {'晴朗度': 0, '雾度': 0, '霾度': 3, '降雨量': 0, '降雪量': 0},
        '阵雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 1, '降雪量': 0},
        '小雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 2, '降雪量': 0},
        '小到中雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 3, '降雪量': 0},
        '中雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 4, '降雪量': 0},
        '中到大雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 5, '降雪量': 0},
        '大雨': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 6, '降雪量': 0},
        '雨夹雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 3, '降雪量': 3},
        '阵雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 1},
        '小雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 2},
        '小到中雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 3},
        '中雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 4},
        '中到大雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 5},
        '大雪': {'晴朗度': 0, '雾度': 0, '霾度': 0, '降雨量': 0, '降雪量': 6}
    }
    # 合法的风向
    __wind_directions = ['无持续风向', '东风', '南风', '西风', '北风', '东南风', '东北风', '西南风', '西北风']

    @property
    def weather_info(self):
        return pd.DataFrame(self.__weather_info)

    @classmethod
    def get_weather_df(cls):
        dfs = []
        for file_name in sorted(os.listdir(cls.__file_path)):
            if file_name.endswith('.h5'):
                dfs.append(pd.read_hdf('data/weather/{}'.format(file_name), cls.__h5_key))
        return pd.concat(dfs)

    def __init__(self, first_date=None):
        self.__util = Util()
        self.__first_date = self.__first_date = datetime.date(year=2019, month=12, day=1) \
            if first_date is None else first_date

    def run(self):
        '''
        下载天气数据
        :param region:
        :param date:
        :return:
        '''
        capital_province = self.__util.get_region_info('region', 'capital_province')
        key_cities = self.__util.key_cities
        date = datetime.date.today()
        while date >= self.__first_date:
            date = datetime.date(year=date.year, month=date.month, day=1)
            year_month = str(date)[:7]
            weather_by_region = {}
            for region, pinyin in self.__util.get_region_info('region', 'pinyin').items():
                if region in capital_province or region in key_cities:
                    url = self.__url_template.format(pinyin, str(date).replace('-', '')[:-2])
                    self.logger.info('下载并处理 {} {} 的天气，网址：{}'.format(year_month, region, url))
                    html_text = None
                    for try_time in range(1, 6):
                        try:
                            res = requests.get(url)
                            html_text = res.text
                        except Exception as e:
                            self.logger.warning('第 {} 次获取天气数据 {} {}，失败，{} 秒后重试，错误信息：{}'
                                                .format(try_time, region, date, try_time * 3, e))
                            time.sleep(try_time * 3)
                            continue
                        if res.status_code != 200:
                            self.logger.warning('第 {} 次获取天气数据 {} {}，失败，{} 秒后重试，http status code: {}'
                                                .format(try_time, region, date, try_time * 3, res.status_code))
                            time.sleep(try_time * 3)
                            continue
                    if html_text is None:
                        self.logger.error('尝试失败次数过多，程序退出')
                        return
                    tree = etree.HTML(html_text)
                    nodes = tree.xpath('//table/tr/td')
                    idx = 0
                    weather_by_region[region] = OrderedDict()
                    for key in ['日期', '上午天气', '下午天气', '上午温度', '下午温度', '风向', '最低风速', '最高风速',
                                '上午晴朗度', '上午雾度', '上午霾度', '上午降雨量', '上午降雪量', '下午晴朗度', '下午雾度',
                                '下午霾度', '下午降雨量', '下午降雪量']:
                        weather_by_region[region][key] = []
                    for node in nodes:
                        text = self.__util.str_replace_to_single(node.text).strip()
                        if len(text) == 0:
                            for sub_node in node.xpath('./a'):
                                text = self.__util.str_replace_to_single(sub_node.text).strip()
                                if len(text) > 1:
                                    break
                        if len(text) > 1:
                            idx += 1
                            if idx == 1:
                                # 日期
                                _date = self.__util.str_replace_to_single(text, ['年', '月', '日'], '-')[:-1]
                                weather_by_region[region]['日期'].append(_date)
                            elif idx == 2:
                                # 天气
                                str_am_weather, str_pm_weather = text.split(' /')
                                weather_by_region[region]['上午天气'].append(str_am_weather)
                                weather_by_region[region]['下午天气'].append(str_pm_weather)
                                # 对天气数据进行处理，根据上午/下午的天气，转换并增加上午/下午的晴朗度、降雨量、降雪量 3 个字段
                                for str_weather, str_apm in zip([str_am_weather, str_pm_weather], ['上午', '下午']):
                                    weather_info = self.__weather_info[str_weather]
                                    for factor in ['晴朗度', '雾度', '霾度', '降雨量', '降雪量']:
                                        weather_by_region[region]['{}{}'.format(str_apm, factor)]\
                                            .append(weather_info[factor])
                            elif idx == 3:
                                # 温度
                                am_temperature, pm_temperature = text.replace('℃', '').split(' / ')
                                am_temperature = int(am_temperature)
                                pm_temperature = int(pm_temperature)
                                weather_by_region[region]['上午温度'].append(am_temperature)
                                weather_by_region[region]['下午温度'].append(pm_temperature)
                            else:
                                idx = 0
                                # 风向、风速
                                str_am_wind, str_pm_wind = text.split(' /')
                                str_am_direction, wind = str_am_wind.split(' ')
                                am_wind_low_level, am_wind_high_level = wind.replace('级', '').split('-')
                                am_wind_low_level = int(am_wind_low_level)
                                am_wind_high_level = int(am_wind_high_level)
                                # weather_by_region[region]['上午风向'].append(str_am_direction)
                                # weather_by_region[region]['上午最低风速'].append(am_wind_low_level)
                                # weather_by_region[region]['上午最高风速'].append(am_wind_high_level)
                                str_pm_direction, wind = str_pm_wind.split(' ')
                                pm_wind_low_level, pm_wind_high_level = wind.replace('级', '').split('-')
                                pm_wind_low_level = int(pm_wind_low_level)
                                pm_wind_high_level = int(pm_wind_high_level)
                                # weather_by_region[region]['下午风向'].append(str_pm_direction)
                                # weather_by_region[region]['下午最低风速'].append(pm_wind_low_level)
                                # weather_by_region[region]['下午最高风速'].append(pm_wind_high_level)
                                if str_am_direction != str_pm_direction:
                                    # 目前看，这两个字段值总是相等，所以合并成一个字段
                                    raise ValueError('{} {} 上午和下午风向不同：'.format(
                                            year_month, region, str_am_direction, str_pm_direction))
                                if am_wind_low_level != pm_wind_low_level:
                                    # 目前看，这两个字段值总是相等，所以合并成一个字段
                                    raise ValueError('{} {} 上午和下午最低风速不同：'.format(
                                            year_month, region, am_wind_low_level, pm_wind_low_level))
                                if am_wind_high_level != pm_wind_high_level:
                                    # 目前看，这两个字段值总是相等，所以合并成一个字段
                                    raise ValueError('{} {} 上午和下午最高风速不同：'.format(
                                            year_month, region, am_wind_high_level, pm_wind_high_level))
                                weather_by_region[region]['风向'].append(str_am_direction)
                                weather_by_region[region]['最低风速'].append(am_wind_low_level)
                                weather_by_region[region]['最高风速'].append(am_wind_high_level)
            # 城市天气存文件
            dfs = []
            regions = sorted(list(weather_by_region.keys()))
            for region in regions:
                df = pd.DataFrame(weather_by_region[region])
                df = df.set_index('日期')
                df.columns = pd.MultiIndex.from_product([[region], df.columns.values])
                dfs.append(df)
            df = pd.concat(dfs, axis=1)
            # df.to_excel('{}/{}_city_{}.xlsx'.format(self.__file_path, self.__file_name_perfix, year_month))
            # df.to_hdf('{}/{}_city_{}.h5'.format(self.__file_path, self.__file_name_perfix, year_month), self.__h5_key)
            # 保存重点关注城市天气
            key_city_dfs = [df[[city]] for city in self.__util.key_cities]
            key_city_df = pd.concat(key_city_dfs, axis=1)
            for i, city in enumerate(df.columns.levels[0].values):
                # 城市转省、直辖市
                col_0 = capital_province.get(city)
                if col_0 is not None:
                    df.columns.levels[0].values[i] = col_0
            # df.to_excel('{}/{}_region_{}.xlsx'.format(self.__file_path, self.__file_name_perfix, year_month))
            # df.to_hdf('{}/{}_region_{}.h5'.format(self.__file_path, self.__file_name_perfix, year_month), self.__h5_key)
            df = pd.concat([key_city_df, df], axis=1)
            df.to_excel('{}/{}_{}.xlsx'.format(self.__file_path, self.__file_name_perfix, year_month))
            df.to_hdf('{}/{}_{}.h5'.format(self.__file_path, self.__file_name_perfix, year_month), self.__h5_key)
            self.logger.info('{} 的天气处理完毕'.format(year_month))
            date -= datetime.timedelta(days=1)

    def get_weather_average_data(self, df_virus_daily, start_shift=14, end_shift=3):
        '''
        获取天气的加权（权重值线性递增）平均数据，从地区有首例确诊病人往前找 start_shift 天，到 end_shift 天前，之间的数据取均值
        专家分析：从已有病例来看，这一新型冠状病毒的潜伏期平均7天左右，短的2到3天，长的12天
        :param df_virus_daily: 各地疫情日频数据
        :param start_shift: 默认取最大潜伏期的值
        :param end_shift: 默认取最小潜期的伏值
        :return:
        '''
        regions = df_virus_daily.columns.levels[0].tolist()
        regions.remove('全国')  # 去掉全国
        df_virus_daily = df_virus_daily[regions]
        df_weather = self.get_weather_df()
        regions = df_weather.columns.levels[0].tolist()
        df_all = pd.concat([df_virus_daily, df_weather], axis=1)
        end_idx = df_all.index.values.searchsorted(df_all.index[-1]) - end_shift + 1
        ss = []
        for region in regions:
            df_region = df_all[region]
            if region in ['武汉', '湖北'] or '确诊' not in df_region.columns:
                start_date = '2019-12-15' if region in ['武汉', '湖北'] else '2020-01-01'
                start_idx = df_region.index.values.searchsorted(start_date)
            else:
                arr = df_region['确诊'].values
                arr[np.isnan(arr)] = 0
                start_idx = (arr != 0).argmax() - start_shift
            df_region = df_region.iloc[start_idx: end_idx]
            for col in ['温度', '风速', '晴朗度', '雾度', '霾度', '降雨量', '降雪量']:
                prefix = ['最低', '最高'] if col == '风速' else ['上午', '下午']
                val = None
                for pre in prefix:
                    if val is None:
                        val = df_region['{}{}'.format(pre, col)].values
                    else:
                        val += df_region['{}{}'.format(pre, col)].values
                df_region['日均{}'.format(col)] = val / 2
            s = pd.Series([])
            for col in ['上午温度', '下午温度', '日均温度',
                        '最低风速', '最高风速', '日均风速',
                        '上午晴朗度', '下午晴朗度', '日均晴朗度',
                        '上午雾度', '下午雾度', '日均雾度',
                        '上午霾度', '下午霾度', '日均霾度',
                        '上午降雨量', '下午降雨量', '日均降雨量',
                        '上午降雪量', '下午降雪量', '日均降雪量',
                        ]:
                mean_col = '平均{}'.format(col)
                s[mean_col] = df_region[col].mean()
                weight_col = '加权平均{}'.format(col)
                weights = np.arange(1, df_region.shape[0] + 1)  # 线性递增权重
                down_cnt = 7 - end_shift  # 平均潜伏期后的 down_cnt 天的权重线性递减
                for i, down_val in zip(np.arange(1, down_cnt + 1), np.arange(1, down_cnt + 1)[::-1]):
                    weights[-i] -= down_val * 2
                # up_cnt = df_region.shape[0] - down_cnt  # 线性递增天数
                # print('线性递增天数：{}，线性递减天数：{}，{} 天气数据的加权平均权重：{}'
                #       .format(up_cnt, down_cnt, region, weights))
                s[weight_col] = np.average(df_region[col].values, weights=weights)
            s.name = region
            ss.append(s)
        df = pd.concat(ss, axis=1).T
        df.index.name = '地区'
        return df


if __name__ == '__main__':
    crawler = WeatherCrawler(datetime.date(year=2019, month=12, day=1))
    crawler.run()
