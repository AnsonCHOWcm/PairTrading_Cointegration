from __future__ import (absolute_import, division, print_function, unicode_literals)
from Cointegration_Pair_Trading_backtrader_Strategy_class_PS import *
from PerformanceTracker import *
import glob
import pandas as pd
import datetime
import backtrader as bt

class CommInfoFractional(bt.CommissionInfo):

    def getsize(self, price, cash):

        return self.p.leverage * (cash / price)

if __name__ == '__main__':

    # Flag to indicating whether it is optimization process

    print_log = True
    cross_zero_flag = True
    ADF_thres = -2
    cointegration_look_back_period = 30
    PS_thres = 2
    long_bias_flag = True
    pair_selection_freq = 'month'


    cerebro = bt.Cerebro()

    cerebro.addstrategy(CointegrationStrat_PS, printlog=print_log, SL_rate=-0.05, cross_zero=cross_zero_flag,
                        ADF_threshold=ADF_thres, coint_look_back_period=cointegration_look_back_period,
                        PS_threshold=PS_thres, long_bias=long_bias_flag, selection_freq=pair_selection_freq)

    cerebro.broker.setcash(3000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_slippage_perc(0.001)
    cerebro.broker.addcommissioninfo(CommInfoFractional())

    path = '/Volumes/SSD/Trading/Algo/Data/Crypto/CCXT Data/Daily/*'

    for fname in glob.glob(path):
        datafile = pd.read_csv(fname, index_col=0, parse_dates=True, infer_datetime_format=True)

        data = bt.feeds.PandasData(
            dataname=datafile,
            timeframe=bt.TimeFrame.Days,
            compression=1,
            fromdate=datetime.datetime(2018, 6, 1),
            todate=datetime.datetime(2022, 4, 30)
        )

        name = datafile['symbol'][0]

        cerebro.adddata(data, name=name)

    # for data in cerebro.datas:
    #     data.plotinfo.plot = False

    cerebro.addobserver(bt.observers.Broker)


    cerebro.addanalyzer(PortfolioPerformance, _name='portfolioperformance', annualizedperiod=365)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    strat = cerebro.run(maxcpus=1)

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    print('*******Strategy Performance**********')

    performance = strat[0].analyzers.portfolioperformance.get_analysis()

    print('Total Period (Log) Return : %f' % performance['periodreturn'])

    print('Annualized (Log) Return : %f' % performance['annualizedreturn'])

    print('Sharpe Ratio: %f' % performance['sharperatio'])

    print('MDD: %f' % performance['MDD'])

    print('Calmar Ratio : %f' % (performance['annualizedreturn']/performance['MDD']))

    tradeanalysis = strat[0].analyzers.tradeanalyzer.get_analysis()

    print('Total Trade : %f' % tradeanalysis.total.total)
    print('Total PnL : %f' % tradeanalysis.pnl.net.total)
    print('Win Rate : %f' % (int(tradeanalysis.won.total) / int(tradeanalysis.total.total)))
    print('Average Win Trade Profit : %f' % tradeanalysis.won.pnl.average)
    print('Average Loss Trade Profit : %f' % tradeanalysis.lost.pnl.average)
    print('Average Trading Period : %f' % tradeanalysis.len.average)

    print('********End**********')


    cerebro.plot()