"""
Created on Wed Mar 24 12:58:38 2021
@author: YUAN YE
"""
import pandas as pd
import os
from datetime import datetime
import time


def timer(func):
    def call_func(*args, **kwargs):
        print("start counting")
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print("end counting")
        print(f"using time{int(total_time // 60)}分{total_time % 60:.2f}秒")
        # print("using time" + str(int(total_time // 60)) + 'minutes')
    return call_func


def zp_data_process(path):
    file_list = os.listdir(path)
    for each_excel in file_list:
        dframe = pd.read_excel(path + '/' + each_excel)
        del dframe['Unnamed: 1']
        new_code_list = []
        for each_code in dframe['Unnamed: 0'].tolist():
            new_code_list.append(each_code.split('.')[0])
        dframe['Unnamed: 0'] = new_code_list
        dframe.set_index('Unnamed: 0', inplace=True)
        dframe = dframe[dframe.columns[dframe.columns > datetime.strptime('2006-1-1', '%Y-%m-%d')]]
        date_list = []
        for each in dframe.columns:
            date_list.append('-'.join(str(each).split('-')[0:2]))
        dframe.columns = date_list
        dframe.index.names = ['']
        dframe.to_excel(path + '/' + each_excel)

    return dframe


@timer
def read_data(path, _date_list):
    with open(r'./project数据/code.txt', "r") as f:
        code_list = eval(f.read())
    
    date_list = _date_list

    file_list = os.listdir(path)
    file_list.remove('label.xlsx')
    file_list.remove('label2.xlsx')
    factor_list = [i.split('.')[0] for i in file_list]

    for each_date in date_list:

        column_list = []
        for index in range(len(file_list)):
            filename = file_list[index]
            print(filename)
            column_list.append(pd.read_excel(path + '/' + filename)[each_date])

        dframe = pd.DataFrame(column_list)
        dframe = dframe.T
        dframe.columns = factor_list
        dframe['code'] = code_list
        dframe.set_index('code', drop=True)
        dframe.to_excel('./project数据/因子数据/日期数据划分/' + each_date + '.xlsx')


def add_label(path, _date_list):

    date_list = _date_list
    df_label = pd.read_excel(path + '/label.xlsx')
    df_label2 = pd.read_excel(path + '/label2_超额.xlsx')
    for index in range(0, len(date_list)):
        df = pd.read_excel('./project数据/日期数据划分/' + date_list[index] + '.xlsx', dtype='object')
        df['label'] = df_label[date_list[index]].tolist()
        df['label2'] = df_label2[date_list[index]].tolist()
        df['date'] = date_list[index]
        df.to_excel('./project数据/因子数据/日期数据划分_label_date/' + date_list[index] + '.xlsx', index=None)


def data_clean(path, _date_list):
    for each_date in _date_list:
        # if each_date == '2020-12':
        #     break
        df = pd.read_excel(path + each_date + '.xlsx', dtype='object')
        st_df = pd.read_excel(r'./project数据/ST股.xlsx', header=None)
        # 去掉st股
        st_list = list(set([i.split('.')[0] for i in st_df[0].tolist()]))
        df = df[~df['code'].isin(st_list)]
        # 去掉平安银行股
        df = df[df['code'] != '000001']
        # 根据log(市值)等于0删掉当时未上市公司
        df = df[df['LOG10 Market Value'] != 0]
        
        # 删除为空的行，用于其他模型
        # df.dropna(inplace=True)

        # 用均值填充空值，用于lstm模型
        s1 = df['label']
        s2 = df['label2']
        s3 = df['code']
        s4 = df['date']
        df.drop(columns=['label', 'label2', 'code', 'date'], inplace=True)

        values = dict([(col_name, col_mean) for col_name, col_mean in zip(df.columns.tolist(), df.mean().tolist())])
        df.fillna(value=values, inplace=True)
        
        df['label'] = s1
        df['label2'] = s2
        df['code'] = s3
        df['date'] = s4

        df.to_excel(r'./project数据/因子数据/日期数据划分_label_date_去空_lstm/' + each_date + '.xlsx', index=None)
    return 


def merge_data(path, single_date_list, file_name):

    df = pd.read_excel(path + single_date_list[0] + '.xlsx', dtype='object')
    for index in range(1, len(single_date_list)):
        df = df.append(pd.read_excel(path + single_date_list[index] + '.xlsx', dtype='object'))
    df.to_excel(r'./project数据/因子数据/data_{}_date.xlsx'.format(file_name), index=None)


if __name__ == '__main__':
    with open(r'./project数据/date_list.txt', "r") as f:
        date_list = eval(f.read())

    # dataframe = zp_data_process(r'./project数据/仲鹏数据')
    dataframe = read_data(path=r'./project数据/原始数据', _date_list=date_list)
    add_label(path=r'./project数据/原始数据', _date_list=date_list)
    data_clean(path='./project数据/因子数据/日期数据划分_label_date/', _date_list=date_list)
    
    
    train_date_list = []
    for i in date_list[:-1]:
        if int(i.split('-')[0]) <= 2015:
            train_date_list.append(i)

    backTest_date_list = []
    for i in date_list[:-1]:
        if int(i.split('-')[0]) > 2015:
            backTest_date_list.append(i)

    merge_data('./project数据/因子数据/日期数据划分_label_date_去空_lstm/', train_date_list, "train")
    merge_data('./project数据/因子数据/日期数据划分_label_date_去空_lstm/', backTest_date_list, "backtest")

