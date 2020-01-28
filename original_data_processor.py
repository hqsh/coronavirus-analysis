try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None
from math import asin, cos, radians, sin, sqrt
from util.util import Util, with_logger
import numpy as np
import pandas as pd


@with_logger
class OriginalDataProcessor:
    '''
    各种原始数据处理
    '''
    def __init__(self):
        self.__util = Util()
        self.__key_cities = self.__util.key_cities
        self.__processed_df = None  # 其他处理后的数据，行索引是地区，列索引是各数据名
        self.logger.info('初始化完成，重点关注城市：{}'.format('、'.join(self.__key_cities)))

    @staticmethod
    def geodistance(lng1, lat1, lng2, lat2):
        '''
        通过经纬度非精确计算，参考：https://www.cnblogs.com/andylhc/p/9481636.html，不安装额外包即可使用
        :param lng1: 经度1
        :param lat1: 纬度1
        :param lng2: 经度2
        :param lat2: 纬度2
        :return:
        '''
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
        distance = round(distance / 1000, 3)
        return distance

    def process_all(self):
        self.logger.info('处理开始')
        # (1) 处理全国各地距离的原始数据
        data = pd.read_csv('data/original/全国各地经纬度.csv', ' ')
        self.__util.add_to_city_to_region(data['省会'].values, data['地区'].values)
        data = data.set_index('地区')
        for city in self.__key_cities:
            # 增加重点关注城市经纬度为所在省会经纬度
            region = self.__util.get_region_by_city(city)
            s = data.loc[region]
            s.name = city
            data = data.append(s)
        position_df = pd.DataFrame([])
        position_df['省会'] = data['省会'].values
        position_df['地区'] = data.index.values
        east = []
        north = []
        for col in ['东经', '北纬']:
            for str_value in data[col]:
                str_value = str_value.replace('′', '')
                str_value = str_value.replace('E', '')
                str_value = str_value.replace('N', '')
                e, n = str_value.split('°')
                e = float(e)
                n = 0.0 if len(n) == 0 else float(n)
                value = e + n / 60.0
                if col == '东经':
                    east.append(value)
                else:
                    north.append(value)
        position_df['东经'] = east
        position_df['北纬'] = north
        position_df = position_df.set_index('地区')
        # df = df.sort_index(axis=0)
        cities_cnt = position_df.shape[0]
        distance = np.zeros(shape=(cities_cnt, cities_cnt), dtype=np.float64)
        for i in range(cities_cnt):
            for j in range(cities_cnt):
                if i < j:
                    if geodesic is None:
                        self.logger.warning('geopy 包没有安装，建议安装，否则经纬度计算结果不一定精确')
                        distance[i, j] = distance[j, i] = self.geodistance(east[i], north[i], east[j], north[j])
                    else:
                        distance[i, j] = distance[j, i] = geodesic((north[i], east[i]), (north[j], east[j])).km
        provinces = position_df.index.values
        distance_df = pd.DataFrame(distance, index=provinces, columns=provinces)
        distance_df.to_csv('data/全国各地之间距离.csv')
        # (2) 处理各地人口、GDP 数据
        data = pd.read_csv('data/original/全国各地人口GDP.csv', '\t')
        df = pd.DataFrame([])
        df['人口'] = data['2019年人口（万）'].values * 10000
        df['GDP'] = data['2018年GDP（亿）'].values * 100000000
        df['人均GDP'] = df['GDP'].values / df['人口'].values
        df.index = data['地区'].values
        df.index.name = '地区'
        # (3) 处理各地流动人口数据
        data = pd.read_csv('data/original/全国各地流动人口.csv', '\t')
        self.__util.add_to_city_to_region(data['城市'].values, data['地区'].values)
        col = '流动人口'
        df[col] = np.zeros(shape=(df.shape[0], ), dtype=np.float64)
        for city, population, region in zip(data['城市'].values, data['常住流动人口数量（万人）'].values,
                                            data['地区'].values):
            population *= 10000
            df.loc[region, col] = population
            if city in self.__key_cities:
                df.loc[city, col] += population
        df['流动人口占比'] = df['流动人口'].values / df['人口'].values
        df.loc['全国', col] = df[col].values.sum() - df[col][self.__key_cities].values.sum()
        # (4) 处理各地面积数据
        data = pd.read_csv('data/original/全国各地面积.csv', ' ')
        data = data.set_index('地区')
        df['面积'] = data['面积（万平方千米）']
        df['人均面积'] = df['面积'].values / df['人口'].values * 10000 * 1000 * 1000  # 单位：平方米
        df.loc['全国', '面积'] = 0
        df.loc['全国', '面积'] = df['面积'].values.sum() - df['面积'][self.__key_cities].values.sum()
        # (5) 处理各地海拔数据
        data = pd.read_csv('data/original/全国各地海拔.csv')
        data['地区'] = [self.__util.get_region_by_city(city) for city in data['城市']]
        data = data.set_index('地区')
        df['海拔'] = data['海拔']
        for city in self.__key_cities:
            # 重点关注城市海拔设置为所在地区海拔
            df.loc[city, '海拔'] = df.loc[self.__util.city_to_region[city], '海拔']
        # (6) 处理经纬度和距离武汉
        df['北纬'] = position_df['北纬']
        df['东经'] = position_df['东经']
        df['距离武汉'] = distance_df['武汉']
        df.to_csv('data/全国各地信息.csv')
        self.logger.info('处理完成')

if __name__ == '__main__':
    processor = OriginalDataProcessor()
    processor.process_all()
