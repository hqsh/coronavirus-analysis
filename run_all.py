from dxy_daily_crawler import DxyDailyCrawler
from huiyan_crawler import HuiyanCrawler
from weather_crawler import WeatherCrawler
import datetime


if __name__ == '__main__':
    crawler = DxyDailyCrawler()
    crawler.crawl()

    crawler = HuiyanCrawler()
    crawler.run()

    today = datetime.date.today()
    crawler = WeatherCrawler(datetime.date(year=today.year, month=today.month, day=1))
    crawler.run()
