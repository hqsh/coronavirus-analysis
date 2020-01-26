# coronavirus-analysis

### 主要文件和使用说明

文件 | 说明
-|-
dxy_crawler.py | 爬取疫情实时数据，需要不停开着电脑爬，否则会丢失数据，爬取的最新数据在 data/dxy_data_recent.* 中，以及原始 html 数据；若有数据丢失、在补充全 html 后，或需要通过原始数据重新构造疫情数据，将 run_mode 改为 init 可以重新构造数据
data 目录 | html 目录下存放爬虫的原始页面、original 目录下存放原始的地区信息数据
original_data_processor.py | 将 original 目录下的原始的地区信息数据处理成便于分析的数据，目前有输出到 data 目录下“全国各地*.csv”的几个文件
util.py | 公共方法类 Util 等
config.ini | 全局配置
coronavirus_analyzer.py | 疫情分析类
