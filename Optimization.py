"""
Created on Wed April 24 12:58:38 2021
@author: YUAN YE
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras_self_attention import SeqSelfAttention
from keras.models import load_model
import numpy as np
import datetime
from dateutil.relativedelta import *
import json


def data_clean(df):
    # 一些数据清理
    df.dropna(inplace=True)
    del df['Unnamed: 0']
    df['label2'] = df['label2'] / 100
    df.rename(columns={'label2': 'stock_yield'}, inplace=True)
    df = df[df['G_S_YTD'] != ' ']
    df = df[df['G_OCF_YTD'] != ' ']

    y = df['label']
    code_list = df['code']
    date_list = df['date']

    # 选择txt文件中的因子
    with open(r'./project数据/selected_factors.txt', 'r') as file:
        factors_list = file.read().splitlines()
    df = df[factors_list]

    # 因子名称
    feat_names = df.columns
    # 特征缩放
    stdsc = StandardScaler()
    df = pd.DataFrame(stdsc.fit_transform(df), columns=feat_names)

    df['code'] = code_list
    df['label'] = y
    df['date'] = date_list
    df.dropna(inplace=True)

    return df

def select_stock(_date, df, _time_stamp=10):
    """
    :param _date: 当前月份
    :param df: 输入数据
    :param _time_stamp: 滚动日期，固定为10
    :return: list[股票代码] 在当月应该买哪些股票
    """

    x_test = []
    # 测试集生成
    code_list_uni = list(set(df['code'].tolist()))

    test_code_list = []
    for each_code in code_list_uni:
        each_df = df[df['code'] == each_code]
        each_df.sort_values(by='date', inplace=True)
        each_df.reset_index(drop=True, inplace=True)
        each_df = each_df[each_df['date'] <= _date].tail(_time_stamp)
        if len(each_df) < 10:
            continue

        each_df.set_index('date', drop=True, inplace=True)
        del each_df['code']
        x_test.append(each_df.values)
        test_code_list.append(each_code)

    x_test = np.array(x_test).astype('float')

    # 导入模型预测该买哪些股票
    model = load_model(r"./project数据/model/model_cnn_lstm_3_class.h5",
                       custom_objects={'SeqSelfAttention': SeqSelfAttention})
    result_df = pd.DataFrame(np.argmax(model.predict(x_test), axis=1), columns=['label_pre'])
    result_df['code'] = test_code_list
    stock_for_buy_list = result_df[result_df['label_pre'] == 2]['code'].tolist()

    return stock_for_buy_list

def optimization(stock_pool_code_list, date):
    """
    :param dataframe: 股票代码和预测的收益率
    :param date: 日期
    :return: 不同权重对应w的dict
    """
    ## 根据标准差求出股票组合内部权重
    df = pd.read_parquet(r'./project数据/model_data/daily5yield.parquet.gzip')
    df = df[stock_pool_code_list]
    df = df[df.index < datetime.datetime.strptime(date, '%Y-%m') + relativedelta(months=+1)].tail(6)
    # 删除最后一个5日收益率为0的
    column_list = []
    for each_code in df.columns:
        if df[each_code][-1] != 0:
            column_list.append(each_code)
    df = df[column_list]

    # 求出标准差倒数的和
    std_reverse_sum = 0
    for each in df.columns:
        std_reverse_sum += 1/df[each].std()

    s_portfolio_w_list = []
    for each in df.columns:
        s_portfolio_w_list.append((1/df[each].std())/std_reverse_sum)

    ## 股票组合和债券的权重分配
    # if 13 <= risk_score <= 19:
    #     w_stock_bond = 0.9
    # elif 20 <= risk_score <= 27:
    #     w_stock_bond = 0.8
    # elif 28 <= risk_score <= 35:
    #     w_stock_bond = 0.7
    # elif 36 <= risk_score <= 43:
    #     w_stock_bond = 0.6
    # elif 44 <= risk_score <= 51:
    #     w_stock_bond = 0.5

    # 给code加上后缀
    with open(r'./project数据/model_data/houzhui_code.json', 'r') as f:
        code_dict = json.load(f)

    houzui_code_list = []
    for each in df.columns.tolist():
        houzui_code_list.append(code_dict[each])

    # houzui_code_list.append('511010.XSHG')

    # W_list2 = [i * w_stock_bond for i in s_portfolio_w_list]
    # W_list2.append(1 - w_stock_bond)

    _output_dict = {}
    date = datetime.datetime.strftime(datetime.datetime.strptime(date, '%Y-%m').date() + relativedelta(months=+1),
                               '%Y-%m')
    _output_dict[date] = {'codes': houzui_code_list, 'weights': s_portfolio_w_list}

    return _output_dict


def main(round):

    backtest_date_list = \
        ['2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07',
         '2017-08', '2017-09', '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05',
         '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03',
         '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01',
         '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11']

    if round == 1:

        select_stock_data = pd.read_parquet(r'./project数据/model_data/backtest_data.parquet.gzip')
        select_stock_data = data_clean(select_stock_data)

        output_dict = {}
        # 给code加上后缀
        with open(r'./project数据/model_data/houzhui_code.json', 'r') as f:
            code_dict = json.load(f)

        for each_date in backtest_date_list:
            stock_pool_list = select_stock(_date=each_date, df=select_stock_data)
            date = datetime.datetime.strftime(datetime.datetime.strptime(each_date, '%Y-%m').date() + relativedelta(months=+1),
                                              '%Y-%m')
            houzui_code_list = []
            for each in stock_pool_list:
                houzui_code_list.append(code_dict[each])

            output_dict[date] = houzui_code_list

        with open('output_code.json', 'w') as json_file:
            json.dump(output_dict, json_file)

    if round == 2:

        with open(r'./stock_choosen_10.json', 'r') as f:
            selected_dict = json.load(f)

        output_dict = {}
        for each_date in backtest_date_list:
            print(each_date)
            yuechu_date = datetime.datetime.strftime(
                datetime.datetime.strptime(each_date, '%Y-%m').date() + relativedelta(months=+1),
                '%Y-%m')
            stock_pool_list = [i.split('.')[0] for i in selected_dict[yuechu_date]]
            output_dict.update(optimization(stock_pool_list, date=each_date))

        with open('weight_code3.json', 'w') as json_file:
            json.dump(output_dict, json_file)

    return output_dict


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    output_dict = main(round=2)
