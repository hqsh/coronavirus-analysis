import pandas as pd
import numpy as np

# 将旧版手工输入数据数据转换成新版数据

old_df = pd.read_excel('./old_version.xlsx', sheet_name='人民日报数据')
dfs = []
df = None
pro = None

for i, col in enumerate(old_df.columns):
    s = old_df[col]
    col_1 = s.values[0]
    if col_1 == '确诊':
        pro = col
        df = pd.DataFrame([])
        df[col_1] = s.values[1:]
        index = s.index.values[1:]
        new_index = []
        for date, time in index:
            if time.hour < 10:
                hour = '0{}'.format(time.hour)
            else:
                hour = time.hour
            if time.minute < 10:
                minute = '0{}'.format(time.minute)
            else:
                minute = time.minute
            month = '0{}'.format(date.month) if date.month < 10 else str(date.month)
            day = '0{}'.format(date.day) if date.day < 10 else str(date.day)
            new_index.append(('{}-{}-{}'.format(date.year, month, day), '{}:{}'.format(hour, minute)))
        new_index = pd.MultiIndex.from_tuples(new_index)
        df.index = new_index
        df.index.name = ['日期', '时间']
    else:
        df[col_1] = s.values[1:]
    if pro in ['武汉', '湖北'] and col_1 == '治愈' or pro not in ['武汉', '湖北'] and col_1 == '疑似':
        for col in ['死亡', '治愈']:
            if False:
                arr = np.zeros(shape=(df.shape[0], ), dtype=np.float64)
                arr[:] = np.nan
                df[col] = arr
        df.columns = pd.MultiIndex.from_product([[pro], df.columns])
        dfs.append(df)
df = pd.concat(dfs, axis=1)
df2 = pd.read_hdf('./dxy_data_2020-01-24 14:33.h5', 'dxy_data')
df = df.append(df2)
cols = ['武汉', '湖北', '广东', '浙江', '北京', '重庆', '湖南', '上海', '四川', '山东', '安徽', '广西', '福建', '河南', '江苏',
        '海南', '天津', '江西', '云南', '陕西', '黑龙江', '辽宁', '贵州', '吉林', '宁夏', '香港', '澳门', '河北', '甘肃', '新疆',
        '台湾', '山西', '内蒙古', '青海']
new_cols = []
for col in cols:
    if col in df.columns.levels[0]:
        new_cols.append(col)
df = df[new_cols]
df.fillna(method='pad', inplace=True)
# 增加是否更新字段
dfs = []
for region in df.columns.levels[0]:
    df_region = df[region].copy()
    arr_is_updated = np.zeros(shape=df_region.shape[0], dtype=np.bool)
    for col in df_region.columns:
        if col == '确诊':
            arr = df_region[col].values
            arr_is_updated[1:] |= arr[1:] != arr[:-1]
    df_region['是否更新'] = arr_is_updated
    df_region.columns = pd.MultiIndex.from_product([[region], df_region.columns.values])
    dfs.append(df_region)
df = pd.concat(dfs, axis=1)
for col in ['确诊', '死亡', '治愈', '疑似']:
    for idx in df.index:
        if np.isnan(df.loc[idx, ('湖北', col)]):
            df.loc[idx, ('湖北', col)] = df.loc[idx, ('武汉', col)]
df.to_excel('../data/init_data.xlsx')
df.to_hdf('../data/init_data.h5', 'dxy_data')
