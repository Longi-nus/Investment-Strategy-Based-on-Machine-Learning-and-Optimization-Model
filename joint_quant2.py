"""
Created on April 2021
@author: Zhao Zhongpeng
"""
from jqfactor import get_factor_values
from scipy import stats
# import warnings
from jqdata import finance
from jqlib.technical_analysis import *
from jqdata import *
import pandas as pd
import json

# 读取json文件
df2 = read_file('weight_code.json')
position_data = json.loads(df2)


## 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 设定指数
    # g.stockindex = '000002.XSHG'
    # 设定所有A股作为基准
    set_benchmark('000002.XSHG')
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True)
    # 设定成交量比例

    set_option('order_volume_ratio', 1)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, \
                             open_commission=0.0003, close_commission=0.0003, \
                             close_today_commission=0, min_commission=5), type='stock')
    # 最大持仓数量
    # g.stocknum = 4000

    ## 自动设定调仓月份（如需使用自动，注销下段）
    # f = 2  # 调仓频率
    log.info(list((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))
    # g.Transfer_date = list(range(1,13,12//f))

    ## 手动设定调仓月份（如需使用手动，注释掉上段）
    g.Transfer_date = list((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

    # 根据大盘止损，如不想加入大盘止损，注释下句即可
    # run_daily(dapan_stoploss, time='open')
    ## 按月调用程序
    run_monthly(trade, monthday=1, time='close')


## 交易函数
def trade(context):
    month = context.current_dt.month;
    year = context.current_dt.year;
    day = context.current_dt.day
    current_date = datetime.date(year, month, day)
    str_date = str(current_date)[:7]
    # 如果当前月为交易月
    if month in g.Transfer_date:
        ## 获得Buylist
        stock_list = position_data[str_date]['codes']
        Buylist = ['511010.XSHG'] + stock_list

        # 分配资金
        Num_buy = min(len(Buylist), 11)
        print(Num_buy)
        weight_for_stock = 0.4
        # Cash = context.portfolio.total_value*weight_for_stock/(Num_buy-1)

        ## 卖出
        positions = context.portfolio.positions
        if len(context.portfolio.positions) > 0:
            for stock in context.portfolio.positions.keys():
                if stock not in Buylist[:Num_buy]:
                    order_target(stock, 0)
                else:
                    if stock == '511010.XSHG':
                        order_value(stock, context.portfolio.total_value * (1 - weight_for_stock) -
                                    context.portfolio.long_positions[stock].value)
                    else:
                        index = stock_list.index(stock)
                        weight = position_data[str_date]['weights'][index] * weight_for_stock
                        order_value(stock, context.portfolio.total_value * weight - context.portfolio.long_positions[
                            stock].value)

        ## 买入
        if len(Buylist) > 0:
            for stock in Buylist[0:Num_buy]:
                if stock not in context.portfolio.positions.keys():
                    if stock == '511010.XSHG':
                        order_value(stock, context.portfolio.total_value * (1 - weight_for_stock))
                    else:
                        index = stock_list.index(stock)
                        weight = position_data[str_date]['weights'][index] * weight_for_stock
                        order_value(stock, context.portfolio.total_value * weight)


    else:
        return

