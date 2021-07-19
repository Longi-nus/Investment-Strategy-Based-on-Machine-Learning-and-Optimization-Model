# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:58:38 2021
@author: YUAN YE
"""
import pandas as pd
from pylab import *
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mr

def timer(func):
    def call_func(*args, **kwargs):
        print("计时开始")
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print("计时结束")
        print(f"程序用时{int(total_time // 60)}分{total_time % 60:.2f}秒")
    return call_func


def data_process(path):

    """
    :param path
    :return: dataframe
    """

    df = pd.read_parquet(path)

    del df['code']
    del df['date']
    del df['Unnamed: 0']
    df = df[df['G_S_YTD'] != ' ']
    del df['label']
    df['label2'] = df['label2']/100

    # 产生X, y
    X = df.drop('label2', axis=1)
    y = df['label2']

    # 因子名称
    feat_names = df.drop(['label2'], axis=1).columns

    # 特征缩放
    stdsc = StandardScaler()
    X = pd.DataFrame(stdsc.fit_transform(X), columns=feat_names)
    print(X, y, feat_names)

    return X, y, feat_names


def random_forest_selection(_X, _y, _feat_names):

    """
    随机森林
    :param X, y, feat_names
    :return:
    """

    rf = RandomForestRegressor(
        criterion='mse',
        n_estimators=1000,
        max_depth=None,  # 定义树的深度, 可以用来防止过拟合
        min_samples_split=50,  # 定义至少多少个样本的情况下才继续分叉
        # min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
    rf.fit(_X, _y)
    print("\n\n ---随机森林---")

    # 因子重要性
    importance = rf.feature_importances_
    feat_names = _feat_names
    indices = np.argsort(importance)[::-1]
    print('重要性排名为:' + '->'.join(list(feat_names[indices])))
    # 画出重要的因子
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure(figsize=(12, 8))
    # plt.title("Feature importances by RandomForest", fontsize='xx-large')
    # plt.bar(range(len(indices)), importance[indices], color='lightblue', align="center")
    # plt.step(range(len(indices)), np.cumsum(importance[indices]), where='mid', label='Cumulative')
    # plt.xticks(range(len(indices)), feat_names[indices], rotation=45, fontsize=12)
    # plt.xlim([-1, len(indices)])
    # plt.show()
    # y = importance.tolist()
    # y.sort(reverse=True)
    # x = feat_names[indices].tolist()
    # x = x[:10]
    # y = y[:10]
    # plt.figure(figsize=(8, 6))
    # plt.title("Feature importances by RandomForest", fontsize='xx-large')
    # plt.bar(x, y)
    # plt.xticks(rotation='vertical')
    
    # for a, b in zip(x, y):
    #     b = round(b,2)
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # plt.savefig(r'./rf.png')
    # plt.show()



    return list(feat_names[indices])


def xgboost_selection(_X, _y, _feat_names):
    """
    xgboost
    :param X, y feat_names
    :return:
    """

    xg = XGBRegressor(
        learning_rate=0.05,
        n_estimators=1000,  # 树的个数--1000棵树建立xgboost
        min_child_weight=6,
                   )

    xg.fit(_X, _y)
    print("\n\n ---xgboost---")

    # 因子重要性
    importance = xg.feature_importances_
    feat_names = _feat_names
    indices = np.argsort(importance)[::-1]
    print('重要性排名为:' + '->'.join(list(feat_names[indices])))

    # 画出重要的因子
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure(figsize=(12, 8))
    # plt.title("Feature importances by Xgboost", fontsize='xx-large')
    # plt.bar(range(len(indices)), importance[indices], color='lightblue', align="center")
    # plt.step(range(len(indices)), np.cumsum(importance[indices]), where='mid', label='Cumulative')
    # plt.xticks(range(len(indices)), feat_names[indices], rotation=45, fontsize=12)
    # plt.xlim([-1, len(indices)])
    # plt.show()
    
    # y = importance.tolist()
    # y.sort(reverse=True)
    # x = feat_names[indices].tolist()
    # x= x[:10]
    # y = y[:10]
    # plt.figure(figsize=(8, 6))
    # plt.title("Feature importances by Xgboost", fontsize='xx-large')
    # plt.bar(x, y)
    # plt.xticks(rotation='vertical')
    
    # for a, b in zip(x, y):
    #     b = round(b,2)
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # plt.savefig(r'./xg.png')
    # plt.show()

    return list(feat_names[indices])


def mutual_info_selection(_X, _y):

    """
    互信息
    :param X, y:
    :return:
    """

    mutual_info_dict = {}
    for index, row in _X.iteritems():
        mutual_info_dict[index] = mr.normalized_mutual_info_score(row, _y)

    sorted_dict = sorted(mutual_info_dict.items(), key=lambda x: x[1], reverse=True)

    key_list = []
    value_list = []
    for each_item in sorted_dict:
        key_list.append(each_item[0])
        value_list.append(each_item[1])
        
    # y = value_list[:10]
    # x = key_list[:10]
    # plt.figure(figsize=(8, 6))
    # plt.title("Feature relevance by Mutual Information", fontsize='xx-large')
    # plt.bar(x, y)
    # plt.xticks(rotation='vertical')
    #
    # for a, b in zip(x, y):
    #     b = round(b,2)
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # plt.savefig(r'./mi.png')
    # plt.show()
    return key_list


def f_selection(_X, _y, _feat_names):
    """
    :param X, y, k:
    :return:
    """
    # selector = SelectKBest(score_func=f_classif)
    fv, pv = f_regression(_X, _y)
    indices = np.argsort(fv)[::-1]
    fv_list = fv.tolist()
    fv_list.sort(reverse=True)
    
    # x = _feat_names[indices].tolist()[:10]
    # y = fv_list[:10]
    # plt.figure(figsize=(8, 6))
    # plt.title("Feature relevance by F stastiscs", fontsize='xx-large')
    # plt.bar(x, y)
    # plt.xticks(rotation='vertical')
    # plt.savefig(r'./fs.png')
    # plt.show()
    
    

    return _feat_names[indices].tolist()


    # selected_feature = []
    # feature_idx = selector.get_support()
    # for index, value in enumerate(list(feature_idx)):
    #     if value == True:
    #         selected_feature.append(X.columns[index])
    #
    # return selected_feature


@ timer
def main():

    X, y, feat_names = data_process(path=r'./project数据/model_data/train_data.parquet.gzip')
    l1 = random_forest_selection(X, y, feat_names)
    l2 = xgboost_selection(X, y, feat_names)
    l3 = mutual_info_selection(X, y)
    l4 = f_selection(X, y, feat_names)

    result_dict = {
        'random_forest': l1,
        'xgboost': l2,
        'mutual_information': l3,
        'f_value': l4
    }

    df_result = pd.DataFrame(result_dict)
    df_result.to_excel(r'./project数据/因子数据/factorFilter_result.xlsx')


if __name__ == '__main__':
    main()