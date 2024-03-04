import pandas as pd

files = [
    'cleaned/hydata_swjl_0.csv',
    'cleaned/hydata_swjl_1.csv',
    'cleaned/hydata_swjl_3.csv',
    'cleaned/hydata_swjl_4.csv',
    'cleaned/hydata_swjl_5.csv',
    'cleaned/hydata_swjl_6.csv',
    'cleaned/hydata_swjl_7.csv',
    'cleaned/hydata_swjl_8.csv',
    'cleaned/hydata_swjl_9.csv',
    'cleaned/hydata_swjl_10.csv',
    'cleaned/hydata_swjl_11.csv',
    'cleaned/hydata_swjl_12.csv',
    'cleaned/hydata_swjl_13.csv',
    'cleaned/hydata_swjl_14.csv',
    'cleaned/hydata_swjl_15.csv',
    'cleaned/hydata_swjl_16.csv'
]

data = pd.concat([pd.read_csv(file) for file in files])
data = data[data['ISLOCAL'] == 1]
data['ONLINETIME'] = pd.to_datetime(data['ONLINETIME'])
# 每天的总上网人数
daily_counts = data.groupby(data['ONLINETIME'].dt.date).size().reset_index(name='count')

# 每小时每天的上网人数
data['hour'] = data['ONLINETIME'].dt.hour
data['weekday'] = data['ONLINETIME'].dt.weekday
hourly_counts = data.groupby(['hour', 'weekday']).size().reset_index(name='count')

daily_counts.to_csv('dailycounts.csv', index=False)
hourly_counts.to_csv('hourlycounts.csv', index=False)