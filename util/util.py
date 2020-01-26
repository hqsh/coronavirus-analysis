from configparser import ConfigParser
from copy import deepcopy
import logging
import re


class Singleton(object):
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls, *args, **kwargs)
        return cls.instance


def with_logger(cls):
    if not hasattr(cls, 'logger') or cls.__name__ != cls.logger.logger_name:
        logger = logging.getLogger(cls.__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        setattr(cls, 'logger_name', cls.__name__)
        setattr(cls, 'logger', logger)
    return cls


class Util(Singleton):
    def __init__(self):
        self.__config = ConfigParser()
        self.__config.read('config.ini')
        # 城市转地区（地区有：所有省会、所有直辖市、全国、重点关注城市），特殊城市转换成所在省
        self.__city_to_region = {}
        # 重点关注城市
        self.__key_cities = []
        key_cities_regions = self.get_config('basic', 'key_cities')
        if key_cities_regions is not None and key_cities_regions != '':
            for city_region in key_cities_regions.split(','):
                city, region = city_region.split(':')
                self.__city_to_region[city] = region
                self.__key_cities.append(city)
        self.__init_region_info()

    def __init_region_info(self):
        self.__region_info = {}  # key / value 形式的地区信息，value 也是个 dict
        for section, option in zip(['region', 'region'], ['pinyin', 'province_capital']):
            config_data = self.get_config(section, option)
            key = '{}-{}'.format(section, option)
            self.__region_info[key] = {}  # key / value 形式的地区信息
            for rp in config_data.split(','):
                k, v = rp.split(':')
                self.__region_info[key][k] = v
            if option in ['province_capital']:
                reversed_option = option.split('_')
                reversed_option.reverse()
                reversed_option = '_'.join(reversed_option)
                reversed_key = '{}-{}'.format(section, reversed_option)
                self.__region_info[reversed_key] = {capital: province for province, capital in
                                                    self.get_region_info(section, option, copy=False).items()}

    def get_config(self, section, option):
        '''
        获取某个配置项
        :param section:
        :param option:
        :return:
        '''
        return self.__config.get(section, option)

    @property
    def key_cities(self):
        '''
        获取重点关注城市列表
        :return:
        '''
        return deepcopy(self.__key_cities)

    @property
    def city_to_region(self, copy=False):
        '''
        获取 city_to_region 对应关系
        :param copy:
        :return:
        '''
        if copy:
            return deepcopy(self.__city_to_region)
        return self.__city_to_region

    def add_to_city_to_region(self, cities, regions):
        '''
        增加到 city_to_region 对应关系
        :param cities:
        :param regions:
        :return:
        '''
        for city, region in zip(cities, regions):
            if city in self.__city_to_region:
                if region != self.__city_to_region[city]:
                    raise ValueError('city：{}，对应的 2 各 region 不同，分别为：{}、{}'
                                     .format(city, region, self.__city_to_region[city]))
            else:
                self.__city_to_region[city] = region

    def get_region_by_city(self, city, raise_exception=True):
        region = self.__city_to_region.get(city)
        if raise_exception and region is None:
            raise KeyError('没有 city ({})，对应的 region'.format(city))
        return region

    def get_region_info(self, section, option, region='all', copy=True):
        '''
        获取地区信息
        :param section: 同配置文件 section
        :param option: 同配置文件 option
        :param region: 地区，all 代表返回 dict 格式的所有
        :param copy: 是否 copy
        :return:
        '''
        category = '{}-{}'.format(section, option)
        if category in self.__region_info:
            if region == 'all':
                if copy:
                    return deepcopy(self.__region_info[category])
                return self.__region_info[category]
            if region in self.__region_info[category]:
                return self.__region_info[category][region]
            raise KeyError('region 不存在')
        raise KeyError('kind 不存在')

    @staticmethod
    def str_replace_to_single(string, old_sub_strings=None, new_sub_string=None):
        '''
        连续多个字符串替换成一个字符串，默认参数为：字符串多个空格转一个空格
        :param list old_sub_strings: 原多个字符串
        :param str new_sub_string: 替换后的一个字符串
        :return:
        '''
        if old_sub_strings is None:
            old_sub_strings = [' ', '\r', '\n']
        if new_sub_string is None:
            new_sub_string = ' '
        re_old_sub_strings = r'[{}]+'.format(''.join(old_sub_strings))
        return re.sub(re_old_sub_strings, new_sub_string, string)
