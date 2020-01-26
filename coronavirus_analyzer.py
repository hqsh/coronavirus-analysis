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
        self.df_distance = pd.read_csv('data/全国各地之间距离.csv')
        self.df_info = pd.read_csv('data/全国各地信息.csv')
        self.df_weather = WeatherCrawler.get_weather_df()
