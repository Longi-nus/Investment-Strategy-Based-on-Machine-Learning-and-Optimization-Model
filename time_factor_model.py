"""
Created on April 2021
@author: YUAN YE
"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, Convolution1D, Flatten, GlobalMaxPooling1D
from keras.models import load_model
import math
import numpy as np
from keras.utils.np_utils import to_categorical


def data_process(path, _time_stamp):
    """
    :param _time_stamp:
    :param path
    :return: dataframe
    """

    df = pd.read_parquet(path)

    # 选择time_stamp,最高为119
    time_stamp = _time_stamp
    df_count = df.groupby(['code']).count()['label']
    df_count = df_count.reset_index()
    df_count.sort_values('label', inplace=True)
    df_count = df_count[df_count['label'] > time_stamp]
    df = df[df['code'].isin(list(set(df_count['code'].tolist())))]

    # 一些数据清理
    df.dropna(inplace=True)
    del df['Unnamed: 0']
    df['label2'] = df['label2']/100
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

    x_train, y_train, x_test, y_test = [], [], [], []
    # 生成训练集验证集
    code_list_uni = list(set(df['code'].tolist()))
    train_code_list = code_list_uni[: math.ceil(len(code_list_uni)/100 * 85)]
    test_code_list = code_list_uni[math.ceil(len(code_list_uni)/100 * 85):]

    # 训练集生成
    for each_code in train_code_list:
        each_df = df[df['code'] == each_code]
        each_df.sort_values(by='date', inplace=True)
        each_df.reset_index(drop=True, inplace=True)

        pointer_start = 0
        pointer_end = time_stamp

        while pointer_end < len(each_df):
            dff = each_df[pointer_start: pointer_end]
            dff.set_index('date', drop=True, inplace=True)
            del dff['code']
            x_train.append(dff.values)
            y_train.append(each_df.iloc[pointer_end, -2])
            pointer_start += 1
            pointer_end += 1

    x_train, y_train = np.array(x_train), np.array(y_train)

    # 验证集生成
    for each_code in test_code_list:
        each_df = df[df['code'] == each_code]
        each_df.sort_values(by='date', inplace=True)
        each_df.reset_index(drop=True, inplace=True)

        pointer_start = 0
        pointer_end = time_stamp

        while pointer_end < len(each_df):
            dff = each_df[pointer_start: pointer_end]
            dff.set_index('date', drop=True, inplace=True)
            del dff['code']
            x_test.append(dff.values)
            y_test.append(each_df.iloc[pointer_end, -2])
            pointer_start += 1
            pointer_end += 1

    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, x_test, y_train, y_test

class NonLinearModel:

    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        self.x_train = x_train.astype('float64')
        self.x_test = x_test.astype('float64')
        self.y_train = to_categorical(y_train.astype('int'), num_classes=3)
        self.y_test = y_test
        self.epochs = epochs

    def plot_loss_acc(self, history, name):
        # 绘制损失图
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title(name, fontsize='12')
        plt.ylabel('loss', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()

        # 绘制accuracy图
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.title(name, fontsize='12')
        plt.ylabel('accuracy', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()

    def report_to_excel(self, report_, name):
        pd.DataFrame(report_).transpose().to_excel('./{}_report.xlsx'.format(name))

    def rnn(self, trainable):

        if trainable == True:
            model = Sequential()
            model.add(SimpleRNN(64, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(units=3,
                            activation='softmax',
                            )
                      )

            print(model.summary())  # 打印网络层次结构

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',  # 优化器
                          metrics=['accuracy']
                          )

            history = model.fit(self.x_train, self.y_train,
                                epochs=self.epochs,  # 迭代次数
                                batch_size=64,  # 每次用来梯度下降的批处理数据大小
                                verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                                validation_data=(self.x_test, to_categorical(self.y_test.astype('int'), num_classes=3))
                                )

            # 绘制loss, accuracy图
            NonLinearModel.plot_loss_acc(self, history, "RNN")

            # 保存模型
            model.save('./project数据/model/model_rnn_3_class.h5')  # creates a HDF5 file 'my_model.h5'

        model = load_model('./project数据/model/model_rnn_3_class.h5')
        print(model.summary())  # 打印网络层次结构

        print("\n\n ---rnn---")
        result = model.predict(self.x_test)
        report = classification_report(self.y_test, np.argmax(result, axis=1), output_dict=True)
        print(classification_report(self.y_test, np.argmax(result, axis=1)))
        NonLinearModel.report_to_excel(self, report, "RNN")

    def lstm(self, trainable):

        if trainable == True:
            model = Sequential()
            model.add(LSTM(64, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=3,
                            activation='softmax',
                            )
                      )

            print(model.summary())  # 打印网络层次结构

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',  # 优化器
                          metrics=['accuracy']
                          )

            history = model.fit(self.x_train, self.y_train,
                                epochs=self.epochs,  # 迭代次数
                                batch_size=64,  # 每次用来梯度下降的批处理数据大小
                                verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                                validation_data=(self.x_test, to_categorical(self.y_test.astype('int'), num_classes=3))
                                # 验证集
                                )

            # 绘制loss, accuracy图
            NonLinearModel.plot_loss_acc(self, history, "LSTM")

            model.save(r"./project数据/model/model_lstm_3_class.h5")

        model = load_model(r"./project数据/model/model_lstm_3_class.h5",
                           custom_objects={'SeqSelfAttention': SeqSelfAttention})
        print(model.summary())  # 打印网络层次结构

        # model = load_model(r"./project数据/model/model_cnn_lstm_time.h5")
        print("\n\n ---lstm---")
        result = model.predict(self.x_test)
        # print(np.argmax(result, axis=1).tolist())
        report = classification_report(self.y_test, np.argmax(result, axis=1), output_dict=True)
        print(classification_report(self.y_test, np.argmax(result, axis=1)))
        NonLinearModel.report_to_excel(self, report, "LSTM")

    def cnn(self, trainable):

        if trainable == True:
            model = Sequential()
            model.add(Convolution1D(64, 3, input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                                    padding='same',
                                    activation='relu'),
                      )
            model.add(Convolution1D(64, 3, padding='same', activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Flatten())
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=3,
                            activation='softmax',
                            )
                      )

            print(model.summary())  # 打印网络层次结构

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',  # 优化器
                          metrics=['accuracy']
                          )

            history = model.fit(self.x_train, self.y_train,
                                epochs=self.epochs,  # 迭代次数
                                batch_size=64,  # 每次用来梯度下降的批处理数据大小
                                verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                                validation_data=(self.x_test, to_categorical(self.y_test.astype('int'), num_classes=3))
                                # 验证集
                                )

            # 绘制loss, accuracy图
            NonLinearModel.plot_loss_acc(self, history, "CNN")

            model.save(r"./project数据/model/model_cnn_3_class.h5")

        model = load_model(r"./project数据/model/model_cnn_3_class.h5")
        print(model.summary())  # 打印网络层次结构
        print("\n\n ---cnn---")
        result = model.predict(self.x_test)
        report = classification_report(self.y_test, np.argmax(result, axis=1), output_dict=True)
        print(classification_report(self.y_test, np.argmax(result, axis=1)))
        NonLinearModel.report_to_excel(self, report, "CNN")

    def cnn_bilstm_attention(self, trainable):

        if trainable == True:
            model = Sequential()

            model.add(Convolution1D(64, 3, input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                                    padding='same',
                                    activation='relu'),
                      )
            model.add(Convolution1D(64, 3, padding='same', activation='relu'))
            # model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
            model.add(LSTM(64, dropout=0.2, return_sequences=True))
            model.add(SeqSelfAttention())
            model.add(GlobalMaxPooling1D())

            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=3,
                            activation='softmax',
                            )
                      )

            print(model.summary())  # 打印网络层次结构

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',  # 优化器
                          metrics=['accuracy']
                          )

            history = model.fit(self.x_train, self.y_train,
                                epochs=self.epochs,  # 迭代次数
                                batch_size=64,  # 每次用来梯度下降的批处理数据大小
                                verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                                validation_data=(self.x_test, to_categorical(self.y_test.astype('int'), num_classes=3))  # 验证集
                                )

            # 绘制loss, accuracy图
            NonLinearModel.plot_loss_acc(self, history, "CNN + Bi-LSTM + Attention")

            model.save(r"./project数据/model/model_cnn_lstm_3_class.h5")

        model = load_model(r"./project数据/model/model_cnn_lstm_3_class.h5",
                           custom_objects={'SeqSelfAttention': SeqSelfAttention})
        print(model.summary())  # 打印网络层次结构

        # model = load_model(r"./project数据/model/model_cnn_lstm_time.h5")
        print("\n\n ---cnn+lstm+attention---")
        result = model.predict(self.x_test)
        report = classification_report(self.y_test, np.argmax(result, axis=1), output_dict=True)
        print(classification_report(self.y_test, np.argmax(result, axis=1)))
        NonLinearModel.report_to_excel(self, report, "CNN+LSTM+ATTENTION")


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_process(path=r'./project数据/model_data/train_data.parquet.gzip',
                                                    _time_stamp=10)

    non_linear_model = NonLinearModel(x_train, x_test, y_train, y_test, epochs=15)
    # non_linear_model.rnn(trainable=False)
    # non_linear_model.lstm(trainable=False)
    non_linear_model.cnn(trainable=False)
    non_linear_model.cnn_bilstm_attention(trainable=False)
