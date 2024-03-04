import os
import pandas as pd
from datetime import datetime

# 网吧信息数据处理
def cafproc():
    intcaf = pd.read_csv(
        'data/网吧信息.csv', usecols=['SITEID', 'TITLE', 'lng', 'lat'])
    # 清除含有空值的行
    intcaf = intcaf.dropna()
    # 将经纬度转换为数值类型，非数值的转换为NaN
    intcaf['lng'] = pd.to_numeric(intcaf['lng'], errors='coerce')
    intcaf['lat'] = pd.to_numeric(intcaf['lat'], errors='coerce')
    
    intcaf = intcaf.dropna()
    intcaf.to_csv('data/intcafe.csv', index=False)

# 上网记录信息处理
global df_intcaf, df_prov, df_city, df_area
df_intcaf = pd.read_csv('newdata/intcafe.csv')
df_prov = pd.read_csv('newdata/provinces.csv')
df_city = pd.read_csv('newdata/cities.csv')
df_area = pd.read_csv('newdata/areas.csv')
df_prov['code'] = df_prov['code'].astype(str)
df_city['code'] = df_city['code'].astype(str)
df_area['code'] = df_area['code'].astype(str)

def clean_hydata(hy_path):
    df_hy = pd.read_csv(hy_path)
    print("未清洗数据情况：", df_hy.shape)
    
    df_hy = df_hy.dropna()
    
    # 将AREAID转换为字符串类型
    df_hy['AREAID'] = df_hy['AREAID'].astype(str)

    # 转换日期时间格式
    df_hy['ONLINETIME'] = pd.to_datetime(df_hy['ONLINETIME'].astype(
        str), format='%Y%m%d%H%M%S', errors='coerce')
    df_hy['OFFLINETIME'] = pd.to_datetime(df_hy['OFFLINETIME'].astype(
        str), format='%Y%m%d%H%M%S', errors='coerce')
    df_hy['BIRTHDAY'] = pd.to_datetime(df_hy['BIRTHDAY'].astype(
        str), format='%Y%m%d', errors='coerce')
    
    # 清除CUSTOMERNAME不符合规范的行
    df_hy = df_hy[~df_hy['CUSTOMERNAME'].str.startswith('`')]

    # 清除XB不是“男”或“女”的行
    df_hy = df_hy[df_hy['XB'].isin(['男', '女'])]

    # 清除ONLINETIME和OFFLINETIME格式不符的行
    df_hy = df_hy.dropna(subset=['ONLINETIME', 'OFFLINETIME', 'BIRTHDAY'])
    
    # 清除AREAID不符合规范的行
    df_hy = df_hy[~df_hy['AREAID'].str.startswith('`')]

    # 计算上网时的年龄
    df_hy['AGE'] = 2016 - df_hy['BIRTHDAY'].dt.year
    # 清除年龄不在0-100岁之间的数据
    df_hy = df_hy[(df_hy['AGE'] > 0) & (df_hy['AGE'] <= 100)]

    # 计算上网时长（以小时为单位）
    df_hy['DURATION'] = ((
        df_hy['OFFLINETIME'] - df_hy['ONLINETIME']).dt.total_seconds() / 3600).round(2)
    # 清除上网时长为负数的数据
    df_hy = df_hy[df_hy['DURATION'] >= 0]

    # print("清洗后数据：", df_hy.shape)

    # 清除不在网吧表中的网吧ID
    df_hy = df_hy[df_hy['SITEID'].isin(df_intcaf['SITEID'])]
    

    # 将AREAID对应的籍贯名称添加到数据中
    # 提取AREAID的前两位和前四位
    df_hy['AREAID_2'] = df_hy['AREAID'].str[:2]
    df_hy['AREAID_4'] = df_hy['AREAID'].str[:4]

    # 查询对应的省份名称
    df_hy = df_hy.merge(df_prov[['code', 'name']], left_on='AREAID_2', right_on='code', how='left')
    df_hy.rename(columns={'name': 'PROVINCE'}, inplace=True)

    # 查询对应的城市名称
    df_hy = df_hy.merge(df_city[['code', 'name']], left_on='AREAID_4', right_on='code', how='left')
    df_hy.rename(columns={'name': 'CITY'}, inplace=True)
    
    # 查询对应的地区名称
    df_hy = df_hy.merge(df_area[['code', 'name']], left_on='AREAID', right_on='code', how='left')
    df_hy.rename(columns={'name': 'AREA'}, inplace=True)


    # 设置ISLOCAL列
    df_hy['ISLOCAL'] = 0
    df_hy.loc[df_hy['AREAID_2'] == '50', 'ISLOCAL'] = 1
    df_hy.loc[df_hy['AREAID_4'] == '5102', 'ISLOCAL'] = 1

    # 清理不再需要的列
    df_hy.drop(['AREAID_2', 'AREAID_4', 'code_x',
            'code_y', 'code'], axis=1, inplace=True)
    
    df_hy = df_hy[df_hy['PROVINCE'].notna()]
    # 显示一些行以确认结果
    # print(df_hy.head())

    # 保存清洗后的数据到新文件
    base_name = os.path.basename(hy_path).split('.')[0]
    new_file_path = f'cleaned/{base_name}.csv'
    df_hy.to_csv(new_file_path, index=False)
    
    print("清洗后数据：", df_hy.shape)


# 文件列表
files = [
    'data/hydata_swjl_0.csv',
    'data/hydata_swjl_1.csv',
    'data/hydata_swjl_3.csv',
    'data/hydata_swjl_4.csv',
    'data/hydata_swjl_5.csv',
    'data/hydata_swjl_6.csv',
    'data/hydata_swjl_7.csv',
    'data/hydata_swjl_8.csv',
    'data/hydata_swjl_9.csv',
    'data/hydata_swjl_10.csv',
    'data/hydata_swjl_11.csv',
    'data/hydata_swjl_12.csv',
    'data/hydata_swjl_13.csv',
    'data/hydata_swjl_14.csv',
    'data/hydata_swjl_15.csv',
    'data/hydata_swjl_16.csv'
]

# 对每个文件执行清洗操作
for file in files:
    clean_hydata(file)
