"""
Created on Wed Mar 24 12:58:38 2021
@author: YUAN YE
"""
import pandas as pd
from zipfile import ZipFile
import re


def clear_table(_year_start, _year_end):
    """
    切分训练和验证集数据
    :return:
    """
    title_content_dict = {2006: "2005-2008", 2007: "2005-2008", 2008: "2005-2008",
                          2009: "2009-2012", 2010: "2009-2012", 2011: "2009-2012",
                          2012: "2009-2012", 2013: "2013-2016", 2014: "2013-2016",
                          2015: "2013-2016", 2016: "2013-2016", 2017: "2017-2020",
                          2018: "2017-2020", 2019: "2017-2020", 2020: "2017-2020"}

    df_train = pd.DataFrame()
    # 读取正文数据
    for year in range(_year_start, _year_end):
        # with ZipFile(r'./project数据/舆情数据/新闻标题link正文/{}.zip'.format(str(year)), 'r') as my_zip:  # 打开zip
        #     for each_info in my_zip.infolist():
        #         if '.xlsx' in str(each_info):
        #             with my_zip.open(each_info, 'r') as x:  # 打开zip里面的一个文件（xlsx）
        #                 data_df = pd.read_excel(x, dtype='object')[['DeclareDate', 'Title']].loc[2:]
        with ZipFile(r'./project数据/舆情数据/新闻标题link证券/{}.zip'.format(title_content_dict[year]), 'r') as my_zip:  # 打开zip
            for each_info in my_zip.infolist():
                if '.xlsx' in str(each_info):
                    with my_zip.open(each_info, 'r') as x:  # 打开zip里面的一个文件（xlsx）
                        title_df = pd.read_excel(x, dtype='object').loc[2:]

        del title_df['NewsID']
        title_df = title_df[title_df['SecurityType'] == 'A股']

        # data_merge = pd.merge(data_df, title_df, on=['DeclareDate', 'Title'], how='inner')

        df_train = df_train.append(title_df)

    return df_train


def process_data(text_df):
    """
    :param dataframe:
    :return:
    """
    text_df = text_df[['Symbol', 'ShortName', 'DeclareDate', 'Title']]
    text_df.dropna(subset=['Title'], inplace=True)
    text_df.drop_duplicates(subset=['Title'], inplace=True)
    text_df = text_df[~text_df['Title'].str.contains('行业|行情')]
    text_df = text_df[~text_df['Title'].str.contains('涨停|跌停|简报')]

    flag_list = []
    for index, row in text_df.iterrows():

        if re.search(r'股份有限公司股票(\d{4})年', row['Title']) is None:

            name = row['Title'].split('：')[0]
            if "：" in row['Title'] and len(name) <= 5:
                if name == row['ShortName']:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            else:
                flag_list.append(True)

        else:
            flag_list.append(False)

    text_df['flag'] = flag_list

    text_df = text_df[text_df['flag'] == True]

    text_df.reset_index(drop=True, inplace=True)
    text_df.rename(columns={'Title': 'policy_point'}, inplace=True)
    text_df[['DeclareDate', 'Symbol', 'ShortName', 'policy_point']].to_csv(r'./project_数据/舆情数据/text_data.csv', index=None)

    return text_df


if __name__ == '__main__':

    year_start = 2016
    year_end = 2021

    data = clear_table(year_start, year_end)
    text_df = process_data(data)

    text_df = text_df[(text_df['DeclareDate'] >= year_start) & (text_df['DeclareDate'] < year_end)]
    # 切分成几个小文件，方便放入GPU预测
    i = 0
    for n in range(0, len(text_df), 5000):
        df_each = text_df.iloc[n:n + 5000, :]
        df_each.to_csv(r'./project数据/舆情数据/text_data/text_data_{}.csv'.format(str(i)), index=None)
        i += 1
