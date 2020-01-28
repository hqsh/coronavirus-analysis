from dxy_crawler import DxyCrawler
from util.util import with_logger
from weather_crawler import WeatherCrawler
import pandas as pd


@with_logger
class CoronavirusAnalyzer:
    '''
    冠状病毒分析
    '''
    def __init__(self):
        self.df_virus = DxyCrawler.load_dxy_data_frame('recent', 'h5')
        self.df_virus_daily = DxyCrawler.load_dxy_data_frame('recent_daily', 'h5')
        self.df_virus_daily_inc = DxyCrawler.load_dxy_data_frame('recent_daily_inc', 'h5')
        self.df_distance = pd.read_csv('data/全国各地之间距离.csv', index_col=0)
        self.df_info = pd.read_csv('data/全国各地信息.csv', index_col=0)
        self.__weather_crawler = WeatherCrawler()
        self.df_weather = self.__weather_crawler.get_weather_df()
        self.df_weather_average = self.get_weather_average_data()

    def get_weather_average_data(self, start_shift=14, end_shift=3):
        '''
        获取天气的加权（权重值线性递增）平均数据，从地区有首例确诊病人往前找 start_shift 天，到 end_shift 天前，之间的数据取均值
        专家分析：从已有病例来看，这一新型冠状病毒的潜伏期平均7天左右，短的2到3天，长的12天
        :param start_shift: 默认取最大潜伏期的值
        :param end_shift: 默认取最小潜期的伏值
        :return:
        '''
        return self.__weather_crawler.get_weather_average_data(self.df_virus_daily, start_shift, end_shift)

