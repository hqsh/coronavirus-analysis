from calc_corr import calc_corr
from dxy_daily_crawler import DxyDailyCrawler
from huiyan_crawler import HuiyanCrawler
from weather_crawler import WeatherCrawler
import datetime


if __name__ == '__main__':
    crawler = DxyDailyCrawler()
    crawler.crawl()

    crawler = HuiyanCrawler()
    crawler.run()

    for n in range(3, 4):
        calc_corr(n=n)
        print('n = {}，处理完毕'.format(n))

    crawler = WeatherCrawler(datetime.date(year=2020, month=2, day=1))
    crawler.run()
