# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:01:25 2021
@author: YUAN YE
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical


# 数据清理
def data_process(path):
    """
    :param path
    :return: dataframe
    """

    df = pd.read_parquet(path)

    # 一些数据清理
    df.dropna(inplace=True)
    del df['Unnamed: 0']
    df['label2'] = df['label2'] / 100
    df.rename(columns={'label2': 'stock_yield'}, inplace=True)
    df['label'] = df['label'].shift(-1)
    df = df[df['G_S_YTD'] != ' ']
    df = df[df['G_OCF_YTD'] != ' ']
    df.dropna(inplace=True)

    # # df['label'] = df['label'].apply(str)
    # le = LabelEncoder()
    # le = le.fit([0, 1])
    # df['label'] = le.transform(df['label'])  # 使用训练好的LabelEncoder对原数据进行编码

    y = df['label']
    code_list = df['code']
    date_list = df['date']

    # 选择txt文件中的因子
    with open(r'./project数据/selected_factors.txt', 'r') as file:
        factors_list = file.read().splitlines()[:-1]
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
    # 训练集验证集划分
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=168)

    df_train.drop(['code', 'date'], axis=1, inplace=True)
    x_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    # 找一支股票看看效果
    # df_test = df_test[df_test['code'] == '000419']

    del df_test['code']

    date_plot = df_test['date']
    x_test = df_test.drop(['label', 'date'], axis=1)
    y_test = df_test['label']


    return x_train, x_test, y_train, y_test, date_plot

# 非线性模型
class NonLinearModel:

    def __init__(self, x_train, x_test, y_train, y_test, date):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train.astype('int')
        self.y_test = y_test.astype('int')
        self.date = date

    def report_to_excel(self, report_, name):
        pd.DataFrame(report_).transpose().to_excel('./{}_report.xlsx'.format(name))

    def random_forest(self, trainable):
        """
        xgboost
        :param X_train, X_test, y_train, y_test
        :return:
        """
        if trainable == True:
            rf = RandomForestClassifier(
                criterion='entropy',
                n_estimators=500,
                max_depth=None,  # 定义树的深度, 可以用来防止过拟合
                min_samples_split=10,  # 定义至少多少个样本的情况下才继续分叉
                # min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
            )
            rf.fit(self.x_train, self.y_train)
            print("\n\n ---随机森林---")

            joblib.dump(rf, r'./project数据/model/randomForest.pkl')
        rf = joblib.load(r'./project数据/model/randomForest.pkl')
        print("\n\n ---random forest---")
        print(classification_report(self.y_test, rf.predict(self.x_test)))
        report = classification_report(self.y_test, rf.predict(self.x_test), output_dict=True)
        NonLinearModel.report_to_excel(self, report, "Random Forest")

    def xgboost(self, trainable):
        """
        xgboost
        :param X_train, X_test, y_train, y_test
        :return:
        """
        if trainable==True:
            xg = XGBClassifier(
                learning_rate=0.01,
                n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                max_depth=None,  # 树的深度
                objective='binary:logitraw',  # 指定损失函数
                use_label_encoder=False
            )
            xg.fit(self.x_train, self.y_train)
            joblib.dump(xg, r'./project数据/model/xgboost.pkl')

        xg = joblib.load(r'./project数据/model/xgboost.pkl')
        print("\n\n ---xgboost---")
        print(classification_report(self.y_test, xg.predict(self.x_test)))
        report = classification_report(self.y_test, xg.predict(self.x_test), output_dict=True)
        NonLinearModel.report_to_excel(self, report, "xgboost")

    def svc(self, trainable):
        """

        :return:
        """
        if trainable==True:

            pca = PCA(n_components=15, whiten=True, random_state=42)
            svc = SVC(C=10, kernel='rbf', class_weight='balanced', max_iter=50000)
            model = make_pipeline(pca, svc)

            # param_grid = {'svr__C': [1, 5, 10, 50, 100, 1000]}
            # grid = GridSearchCV(model, param_grid)
            # grid.fit(self.x_train, self.y_train)
            #
            # final_model = grid.best_estimator_
            model.fit(self.x_train, self.y_train)

            joblib.dump(model, r'./project数据/model/svm.pkl')

        final_model = joblib.load(r'./project数据/model/svm.pkl')
        print("\n\n ---pca+svm---")
        print(classification_report(self.y_test, final_model.predict(self.x_test)))
        report = classification_report(self.y_test, final_model.predict(self.x_test), output_dict=True)
        NonLinearModel.report_to_excel(self, report, "SVM")

    def nn(self, trainable):

        if trainable == True:
            model = Sequential()  # 初始化，很重要！
            model.add(Dense(units=64,  # 输出大小
                            activation='relu',  # 激励函数
                            input_shape=(self.x_train.shape[1],)  # 输入大小, 也就是列的大小
                            )
                      )

            model.add(Dense(units=64,
                            #                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
                            #                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                            activation='relu'  # 激励函数
                            # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
                            )
                      )

            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))  # 丢弃神经元链接概率

            model.add(Dense(units=3,
                            activation='softmax',
                            )
                      )

            print(model.summary())  # 打印网络层次结构

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',  # 优化器
                          metrics=['accuracy']
                          )

            history = model.fit(self.x_train, to_categorical(y_train.astype('int'), num_classes=3),
                                epochs=50,  # 迭代次数
                                batch_size=64,  # 每次用来梯度下降的批处理数据大小
                                verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                                validation_data=(self.x_test, to_categorical(self.y_test.astype('int'), num_classes=3))
                                # 验证集
                                )


            # 保存模型
            model.save('./project数据/model/model_nn_3_class.h5')  # creates a HDF5 file 'my_model.h5'

        # 加载模型
        model = load_model('./project数据/model/model_nn_3_class.h5')
        print(model.summary())  # 打印网络层次结构

        print("\n\n ---nn---")
        result = model.predict(self.x_test)
        report = classification_report(self.y_test, np.argmax(result, axis=1), output_dict=True)
        print(classification_report(self.y_test, np.argmax(result, axis=1)))
        NonLinearModel.report_to_excel(self, report, "NN")



# 程序入口
if __name__ == '__main__':
    x_train, x_test, y_train, y_test, date = data_process(r'./project数据/model_data/train_data.parquet.gzip')

    non_linear_model = NonLinearModel(x_train, x_test, y_train, y_test, date)
    # non_linear_model.random_forest(trainable=False)
    # non_linear_model.xgboost(trainable=False)
    # non_linear_model.svc(trainable=True)
    # non_linear_model.nn(trainable=False)


