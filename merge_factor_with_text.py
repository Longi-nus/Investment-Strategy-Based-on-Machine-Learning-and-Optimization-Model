# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:11:25 2021
@author: YUAN YE
"""

import pandas as pd

def merge_factor(path1=r'./project_datasets/test_result.xlsx', path2 = r'./project_datasets/data_backtest_date_excess.xlsx',
                 out_path='r./backtest_data_excess.parquet.gzip'):

    result = pd.read_excel(path1, dtype='object')

    group_result = result.groupby(['Symbol', 'DeclareDate']).apply(lambda x: x['label'].sum() / x['label'].count())

    group_result = group_result.reset_index()

    group_result.rename(columns={'DeclareDate': 'date', 'Symbol': 'code', 0: 'count'}, inplace=True)

    origin = pd.read_excel(path2, dtype='object')

    merge = pd.merge(origin, group_result, on=['date', 'code'], how='left')

    merge.rename(columns={'count': 'opinion'}, inplace=True)

    merge['opinion'] = merge['opinion'].fillna(0)

    merge = merge[merge['G_S_YTD'] != ' ']
    merge = merge[merge['G_OCF_YTD'] != ' ']

    date_df = pd.read_excel(r'./project数据/因子数据/上市日期.xlsx', dtype='object')
    date_df['上市日期'] = [str(i)[:7] for i in date_df['上市日期'].tolist()]
    date_df.rename(columns={"代码": 'code', "上市日期": 'date'}, inplace=True)
    date_df['flag'] = 1
    # train_data = pd.read_excel(r'./project_code分类/project数据/model_data/train_data_final.xlsx', dtype='object')
    merge2 = pd.merge(merge, date_df, on=['date', 'code'], how='left')

    merge2 = merge2[merge2['flag'] != 1]
    del merge2['flag']

    tuishi = pd.read_excel(r'./project数据/因子数据/已经退市的股票.xlsx')
    tuishi_list = [i.split('.')[0] for i in tuishi['证券代码'].tolist()]

    merge2 = merge2[~merge2['code'].isin(tuishi_list)]

    df2 = merge2.groupby(['code']).count()['label']
    df2 = df2.reset_index()
    df2.sort_values('label', inplace=True)

    label_df = pd.read_excel(r'./project数据/因子数据/标签_排序.xlsx', dtype='object')
    label2_df = pd.read_excel(r'./project数据/因子数据/月度收益率_排序.xlsx', dtype='object')
    label_df.set_index('Code', drop=True, inplace=True)
    label2_df.set_index('Code', drop=True, inplace=True)

    test_all = merge2[merge2[['label2', 'label']].isnull().all(1)]

    for index, row in test_all.iterrows():
        test_all.loc[index, 'label'] = label_df.loc[row['code'], row['date']]
        test_all.loc[index, 'label2'] = label2_df.loc[row['code'], row['date']]

    test_all['label2'] = test_all['label2'] * 100
    merge2 = merge2[~merge2.index.isin(test_all.index)]

    merge2 = merge2.append(test_all)
    merge2.dropna(subset=['label'], inplace=True)
    merge2.dropna(subset=['label2'], inplace=True)

    for index, row in merge2.iterrows():
        if row['label2'] <= -5:
            merge2.loc[index, 'label'] = 0
        elif row['label2'] > -5 and row['label2'] < 5:
            merge2.loc[index, 'label'] = 1
        elif row['label2'] >= 5:
            merge2.loc[index, 'label'] = 2

    # 训练数据中的噪声
    merge2 = merge2[merge2['G_S_YTD'] != 742970040383100032]
    merge2 = merge2[merge2['G_S_YTD'] != 53696203625508192]
    merge2 = merge2[merge2['PM_EX_TTM'] != -60421489448616496]
    merge2 = merge2[merge2['PM_EX_TTM'] != -3130691354564114944]
    merge2 = merge2[merge2['PM_EX_YTD'] != -3130691354564114944]

    merge2.to_parquet(out_path, engine='pyarrow', compression='gzip')


if __name__ == '__main__':
    merge_factor(path1=r'./project数据/舆情数据/test_result_ori.xlsx',
                 path2=r'./project数据/因子数据/data_train_date.xlsx',
                 out_path=r'./project数据/model_data/train_data.parquet.gzip')

    merge_factor(path1=r'./project数据/舆情数据/test_result.xlsx',
                 path2=r'./project数据/因子数据/data_backtest_date.xlsx',
                 out_path=r'./project数据/model_data/backtest_data.parquet.gzip')