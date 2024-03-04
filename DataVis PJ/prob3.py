import pandas as pd
import numpy as np
import json

files = [
    'newdata/hydata_swjl_0.csv',
    'newdata/hydata_swjl_1.csv',
    'newdata/hydata_swjl_3.csv',
    'newdata/hydata_swjl_4.csv',
    'newdata/hydata_swjl_5.csv',
    'newdata/hydata_swjl_6.csv',
    'newdata/hydata_swjl_7.csv',
    'newdata/hydata_swjl_8.csv',
    'newdata/hydata_swjl_9.csv',
    'newdata/hydata_swjl_10.csv',
    'newdata/hydata_swjl_11.csv',
    'newdata/hydata_swjl_12.csv',
    'newdata/hydata_swjl_13.csv',
    'newdata/hydata_swjl_14.csv',
    'newdata/hydata_swjl_15.csv',
    'newdata/hydata_swjl_16.csv'
]

df = pd.concat([pd.read_csv(file) for file in files])

# 筛选年龄在14到35岁之间的记录
teens = df[df['AGE'].between(14, 35)]
# 计算每个PERSONID的上网记录数
teens_count = teens['PERSONID'].value_counts()
# 筛选上网记录数大于6的PERSONID
suspects_count = teens_count[teens_count > 6]
# 保存嫌疑PERSONID和上网记录数
df_suspects_count = suspects_count.sort_values(ascending=False).reset_index()
df_suspects_count.columns = ['PERSONID', 'count']
df_suspects_count.to_csv('newdata/suspects.csv', index=False)
# 保存嫌疑记录
suspects = suspects_count.index
df_suspects = teens[teens['PERSONID'].isin(suspects)]
df_suspects.loc[:, 'ONLINETIME'] = pd.to_datetime(df_suspects['ONLINETIME'])
df_suspects.loc[:, 'OFFLINETIME'] = pd.to_datetime(df_suspects['OFFLINETIME'])
# 按照上线时间排序
df_suspects = df_suspects.sort_values(by='ONLINETIME')

# 建立邻接矩阵
adj_matrix = np.zeros((len(suspects), len(suspects)))

# 定义计算相似度/距离的函数
def similarity(record1, record2):
    time_diff1 = abs(record1['OFFLINETIME'] - record2['ONLINETIME'])
    time_diff2 = abs(record2['OFFLINETIME'] - record1['ONLINETIME'])
    time_diff = min(time_diff1, time_diff2)
    time_max = max(time_diff1, time_diff2)
    similarity = 0
    if time_diff <= pd.Timedelta(minutes=15):
        similarity += 0.5
        if time_max <= pd.Timedelta(minutes=15):
            similarity += 0.5
        if record1['SITEID'] == record2['SITEID']:
            similarity += 1
    return similarity


# 滑动窗口计算相似度
window_size = 100
for i in range(len(df_suspects)):
    for j in range(i+1, min(i+window_size, len(df_suspects))):
        record1 = df_suspects.iloc[i]
        record2 = df_suspects.iloc[j]
        dist = similarity(record1, record2)
        adj_matrix[suspects.get_loc(record1['PERSONID']), suspects.get_loc(
            record2['PERSONID'])] = dist


# 创建节点列表
nodes = [{'id': personid} for personid in suspects]

# 创建边列表
links = []
for i in range(len(suspects)):
    for j in range(i + 1, len(suspects)):
        if adj_matrix[i, j] > 0:  # 仅在相似度大于0时添加边
            links.append({
                'source': suspects[i],
                'target': suspects[j],
                'value': adj_matrix[i, j]
            })

graph_data = {'nodes': nodes, 'links': links}

with open('newdata/suspectsimi.json', 'w') as file:
    json.dump(graph_data, file)
