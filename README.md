# coronavirus-analysis

### 主要文件和使用说明

文件 | 说明
-|-
dxy_crawler.py | 爬取疫情实时数据，需要不停开着电脑爬，否则会丢失数据，爬取的最新数据在 data/dxy_data_recent.* 中，以及原始 html 数据；若有数据丢失、在补充全 html 后，或需要通过原始数据重新构造疫情数据，将 run_mode 改为 init 可以重新构造数据
huiyan_crawler.py | 爬取百度迁徙数据，每天执行一次，可爬取上一天的全国各地人口迁徙数据
calc_corr.py | 通过每日新增确诊人数和人流数据计算论文中的相关系数，需要修改代码中的日期，并确保所需数据已经爬取到
weather_crawler.py | 爬取和处理历史天气数据，每天执行一次，可爬取上一天的全国各地天气
run_all.py | 依次执行 huiyan_crawler.py、calc_corr.py、weather_crawler.py 的爬虫或计算程序，需要修改代码中的日期，并确保所需的每日新增数据已经获取到、百度迁徙数据能够爬取到
data 目录 | html 目录下存放爬虫的原始页面、original 目录下存放原始的地区信息数据
original_data_processor.py | 将 original 目录下的原始的地区信息数据处理成便于分析的数据，目前有输出到 data 目录下“全国各地*.csv”的几个文件
util.py | 公共方法类 Util 等
config.ini | 全局配置
coronavirus_analyzer.py | 疫情分析类
cache/not_shift_one_day 目录 | 目录下的是人流风险系数的计算结果（不额外偏移1天，用于数据分析）（实时计算速度慢，相应代码有变化需要删除缓存文件）
cache/shift_one_day 目录 | 目录下的是人流风险系数的计算结果（额外偏移1天，用于疫情预测）（实时计算速度慢，相应代码有变化需要删除缓存文件）
论文 目录 | 相关论文，发表版：https://publichealth.jmir.org/2020/2/e18638/
