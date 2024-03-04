import pandas as pd

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

# 判定未成年人上网记录
def filerundeage():
    df['UNDERAGE']=0
    df.loc[df['AGE'] < 18, 'UNDERAGE'] = 1
    df.loc[df['DURATION'] > 24, 'UNDERAGE'] = 2
    df.loc[(df['AGE'] > 50) & (df['DURATION'] > 12), 'UNDERAGE'] = 2
    df_filtered = df[df['UNDERAGE'] != 0]
    df_filtered.to_csv('newdata/underage.csv', index=False)

# 统计各网吧未成年人上网记录
def intcount():
    underage = pd.read_csv('newdata/underage.csv')
    intcafe = pd.read_csv('newdata/intcafe.csv')
    allrc = df.groupby('SITEID').size().reset_index(name='count')
    # Counting UNDERAGE==1 and UNDERAGE==2 occurrences per SITEID in underage
    underage1_counts = underage[underage['UNDERAGE'] == 1].groupby(
        'SITEID').size().reset_index(name='UNDERAGE1')
    underage2_counts = underage[underage['UNDERAGE'] == 2].groupby(
        'SITEID').size().reset_index(name='UNDERAGE2')

    allrc = allrc.merge(underage1_counts, on='SITEID', how='left')
    allrc = allrc.merge(underage2_counts, on='SITEID', how='left')

    allrc.fillna(0, inplace=True)
    allrc['IL'] = 0
    allrc.loc[(allrc['UNDERAGE1'] > 0) | (
        allrc['UNDERAGE2'] > 20), 'IL'] = 1
    allrc = allrc.merge(intcafe, on='SITEID', how='right')
    allrc.fillna(0, inplace=True)
    allrc['UNDERAGE1'] = allrc['UNDERAGE1'].astype(int)
    allrc['UNDERAGE2'] = allrc['UNDERAGE2'].astype(int)
    allrc['count'] = allrc['count'].astype(int)
    allrc['IL'] = allrc['IL'].astype(int)
    allrc.to_csv('newdata/intcafecount.csv', index=False)

# 获得未成年人上网记录的唯一PERSONID，以及未成年人中非本地人的记录
def filterunique():
    df = pd.read_csv('newdata/underage.csv')
    df_underage = df[df['UNDERAGE'] == 1]
    # 删除重复的 PERSONID，只保留第一次出现的记录
    unique_underage = df_underage.drop_duplicates(subset='PERSONID')
    # 在 unique_underage 基础上，进一步筛选出 ISLOCAL=0 的记录
    underage_notloc = unique_underage[unique_underage['ISLOCAL'] == 0]
    unique_underage.to_csv('newdata/uni_under.csv', index=False)
    underage_notloc.to_csv('newdata/under_notlocal.csv', index=False)

