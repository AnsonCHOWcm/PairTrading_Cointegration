import backtrader as bt
import numpy as np
import statsmodels.api as sm
import scipy

class CointegrationStrat_PS(bt.Strategy):

    params = (
        ('printlog', True),
        ('SL_rate', -0.05),
        ('cross_zero', False),
        ('ADF_threshold', -2),
        ('coint_look_back_period', 365),
        ('PS_threshold', -2),
        ('long_bias', True),
        ('selection_freq', 'month'),
    )

    def log(self ,txt ,doprint = None):

        dt = self.datas[0].datetime.date(0)
        tm = self.datas[0].datetime.time(0)

        if doprint == None :

            doprint = self.params.printlog

        if doprint :

            print('%s - %s, %s' % (dt.isoformat(), tm.isoformat(), txt))

    def OLS_beta(self, X, Y):

        try:
            beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
        except:
            return None

        return beta

    def OLS_sigma(self, X, Y):

        beta = self.OLS_beta(X, Y)

        if beta is None:
            return None

        RSS = np.matmul((Y - np.matmul(X, beta)).T, (Y - np.matmul(X, beta)))[0][0]

        n = X.shape[0]

        p = X.shape[1]

        sigma_sq = RSS/(n-p-1)

        return np.sqrt(sigma_sq)

    def OLS_beta_t_val(self, target_feature_position, X, Y):

        beta = self.OLS_beta(X, Y)

        if beta is None :
            return None

        sigma = self.OLS_sigma(X, Y)
        V_k = np.linalg.inv(np.matmul(X.T, X))[target_feature_position, target_feature_position]

        return beta[target_feature_position][0]/(sigma * np.sqrt(V_k))

    def OLS_llf(self, X, Y):

        beta = self.OLS_beta(X, Y)

        if beta is None :
            return None

        n = X.shape[0]

        sigma = self.OLS_sigma(X, Y)

        llf = -1 * n /2 * np.log(2*np.pi*sigma**2) - 1/(2*sigma**2) * np.matmul((Y - np.matmul(X, beta)).T, (Y - np.matmul(X, beta)))[0][0]

        return llf

    def cointegration_coefficient_TLS(self, scaled_log_price, sign, colname):

        beta_1 = []
        beta_0 = []

        X = scaled_log_price[:, 0]
        Y = scaled_log_price[:, 1]

        C_0 = sum((X - np.mean(X)) * (Y - np.mean(Y)))
        C_1 = sum((X - np.mean(X)) ** 2 - (Y - np.mean(Y)) ** 2)
        C_2 = -1 * C_0

        beta_1.append((-1 * C_1 + sign * np.sqrt(C_1 ** 2 - 4 * C_0 * C_2)) / (2 * C_0))
        beta_0.append(np.mean(Y) - beta_1[-1] * np.mean(X))

        X = scaled_log_price[:, 1]
        Y = scaled_log_price[:, 0]

        C_0 = sum((X - np.mean(X)) * (Y - np.mean(Y)))
        C_1 = sum((X - np.mean(X)) ** 2 - (Y - np.mean(Y)) ** 2)
        C_2 = -1 * C_0

        beta_1.append((-1 * C_1 + sign * np.sqrt(C_1 ** 2 - 4 * C_0 * C_2)) / (2 * C_0))
        beta_0.append(np.mean(Y) - beta_1[-1] * np.mean(X))

        if (beta_1[0] > beta_1[1]):

            first_asset = colname[1]
            second_asset = colname[0]

            return beta_0[0], beta_1[0], first_asset, second_asset

        else:

            first_asset = colname[0]
            second_asset = colname[1]

            return beta_0[1], beta_1[1], first_asset, second_asset

    def spread_lag_order(self, spread, min_lag, max_lag):

        spread_diff = np.array(spread[1:]) - np.array(spread[:-1])

        best_bic = 100000
        best_lag_order = -1

        for p in range(min_lag, max_lag + 1):

            Y = []
            X = []

            if p == 0:

                Y = spread_diff
                X = spread[:-1]

            else:

                for i in range(p, len(spread_diff)):
                    Y.append(spread_diff[i - 1])
                    X.append(np.append(np.array(spread[i - 1]), spread_diff[(i - p):(i - 1)][::-1]))

            Y = np.array([Y]).reshape((len(Y), 1))
            X = np.array(X)
            X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)

            n = X.shape[0]

            llf = self.OLS_llf(X, Y)

            if llf is None :
                continue

            bic = (p+2)*np.log(n)-2*llf

            if (bic < best_bic):
                best_bic = bic
                best_lag_order = p

        return best_lag_order

    def ADF_test(self, spread, lag_order):

        spread_diff = np.array(spread[1:]) - np.array(spread[:-1])

        Y = []
        X = []

        if lag_order == 0:

            Y = spread_diff
            X = spread[:-1]

        else:

            for i in range(lag_order, len(spread_diff)):
                Y.append(spread_diff[i - 1])
                X.append(np.append(np.array(spread[i - 1]), spread_diff[(i - lag_order):(i - 1)][::-1]))

        Y = np.array([Y]).reshape((len(Y), 1))
        X = np.array(X)
        X = np.append(np.ones(shape = (X.shape[0],1)), X, axis = 1)

        t_value = self.OLS_beta_t_val(1, X, Y)

        return t_value

    def ADF_Measure(self, asset_sample, prev_sample_dict):

        n = len(asset_sample)
        best_ADF_test_value = self.params.ADF_threshold
        selected_first_asset = None
        selected_second_asset = None
        selected_beta_1 = None
        selected_beta_0 = None
        selected_best_lag_order = None

        for i in range(n - 1):

            first_asset = asset_sample[i]

            for j in range(i + 1, n):

                second_asset = asset_sample[j]

                first_asset_sample = prev_sample_dict[first_asset]
                second_asset_sample = prev_sample_dict[second_asset]

                log_scaled_first_asset_data = np.log(first_asset_sample) - np.log(first_asset_sample[0])
                log_scaled_second_asset_data = np.log(second_asset_sample) - np.log(second_asset_sample[0])

                test_sample_data = np.array([log_scaled_first_asset_data, log_scaled_second_asset_data]).T


                sign = np.sign(np.cov(test_sample_data.T)[1, 0])
                beta_0, beta_1, first_asset, second_asset = \
                    self.cointegration_coefficient_TLS(test_sample_data, sign, [first_asset, second_asset])

                first_asset_sample = prev_sample_dict[first_asset]
                second_asset_sample = prev_sample_dict[second_asset]

                log_scaled_first_asset_data = np.log(first_asset_sample) - np.log(first_asset_sample[0])
                log_scaled_second_asset_data = np.log(second_asset_sample) - np.log(second_asset_sample[0])

                test_sample_data = np.array([log_scaled_first_asset_data, log_scaled_second_asset_data]).T

                spread = (test_sample_data[:, 0] - beta_0 - beta_1 * test_sample_data[:, 1]) / (1 + beta_1 ** 2)

                best_lag_order = self.spread_lag_order(spread, 1, 10)

                if best_lag_order == -1:
                    continue

                test_value = self.ADF_test(spread, best_lag_order)

                if test_value is None:
                    continue

                if test_value < best_ADF_test_value:
                    best_ADF_test_value = test_value
                    selected_first_asset = first_asset
                    selected_second_asset = second_asset
                    selected_beta_1 = beta_1
                    selected_beta_0 = beta_0
                    selected_best_lag_order = best_lag_order

        return selected_beta_0, selected_beta_1, selected_first_asset, selected_second_asset, best_ADF_test_value, selected_best_lag_order


    def __init__(self):

        self.trading_days = 0
        self.pair_selection_flag = False
        self.current_time = None
        self.prev_time = None
        # Storing the selected pair and its sample log return
        self.beta_0 = None
        self.beta_1 = None
        self.selected_first_asset = None
        self.selected_second_asset = None
        self.selected_first_asset_log_price_ref = None
        self.selected_second_asset_log_price_ref = None
        self.spread_optimal_lag_order = None
        self.trading_flag = 0
        self.asset_sample = []
        self.prev_sample_dict = {}
        # Storing the portfilio spending
        self.max_pos = 0.98
        self.SL_rate = self.params.SL_rate
        self.initial_asset_value = None
        # Trading Signal related
        self.curr_PS_power = 0
        self.curr_PS_val = 0


    def next(self):

        self.trading_days += 1

        if self.trading_days <= self.params.coint_look_back_period:
            return

        if self.current_month is None:
            self.current_month = self.datas[0].datetime.date(0).month

        # Entering Next Month/week, renew the selected pairs
        if self.params.selection_freq == 'month':
            if self.datas[0].datetime.date(0).month != self.current_time:
                # Amending the reference month
                self.prev_time = self.current_time
                self.current_time = self.datas[0].datetime.date(0).month
                self.pair_selection_flag = True
        elif self.params.selection_freq == 'week':
            if self.datas[0].datetime.date(0).weekday() == 0:
                self.prev_time = self.current_time
                self.current_time = self.datas[0].datetime.date(0).month
                # Amending the reference month
                self.pair_selection_flag = True


        if self.pair_selection_flag:

            # Storing the close price of the asset in the previous month

            self.asset_sample = []

            for data in self.datas:

                self.asset_sample.append(data._name)

                asseet_price_sample = []

                for day in range(-1 * self.params.coint_look_back_period, 0):
                    asseet_price_sample.append(data.close[day])

                self.prev_sample_dict[data._name] = asseet_price_sample

            # Selecting the Trading Pairs for next month based on TLS regrssion and checking the ADF test value

            beta_0, beta_1, selected_first_asset, selected_second_asset, test_value, optimal_lag_order = \
            self.ADF_Measure(self.asset_sample, self.prev_sample_dict)

            self.log('Selected pair %s & %s' %(selected_first_asset, selected_second_asset))
            self.log('Selected ADF value : %f' % test_value)

            # Unwind all the trading position of previous pairs

            if self.selected_first_asset != selected_first_asset or self.selected_second_asset != selected_second_asset:
                if self.trading_flag != 0:
                    self.trading_flag = 0
                    self.log('Close Position :  %s & %s' % (self.selected_first_asset ,self.selected_second_asset))

                    # back to even position
                    if self.params.long_bias:
                        number_of_asset = len(self.asset_sample)
                        for data in self.datas:
                            self.order_target_percent(data=data, target=1 / number_of_asset)
                    else:
                        for data in self.datas:
                            if data._name == self.selected_first_asset or data._name == self.selected_second_asset:
                                self.close(data, exectype=bt.Order.Market)

            self.selected_first_asset = selected_first_asset
            self.selected_second_asset = selected_second_asset
            self.beta_0 = beta_0
            self.beta_1 = beta_1
            self.spread_optimal_lag_order = optimal_lag_order

            if self.selected_first_asset is None :
                return

            for data in self.datas:

                 if data._name == self.selected_first_asset:
                     self.selected_first_asset_log_price_ref = np.log(data.close[0])

                 elif data._name == self.selected_second_asset:
                     self.selected_second_asset_log_price_ref = np.log(data.close[0])

            self.log('Finish Spreaed Threshold Searching')

            return

        if self.prev_time is None:
            return

        if self.selected_first_asset is None :
            return

        for data in self.datas:

            if data._name == self.selected_first_asset:
                first_asset = data
                curr_first_asset_scaled_price = np.log(data.close[0]) - self.selected_first_asset_log_price_ref

            elif data._name == self.selected_second_asset:
                second_asset = data
                curr_second_asset_scaled_price = np.log(data.close[0]) - self.selected_second_asset_log_price_ref

        curr_spread = (curr_first_asset_scaled_price - self.beta_1 * curr_second_asset_scaled_price - self.beta_0)\
                      /(1+self.beta_1**2)

        # checking whether pair is still conintegrated
        cointegrated_flag = True
        prev_first_asset_data = []
        prev_second_asset_data = []

        for data in self.datas:

            if data._name == self.selected_first_asset:
                for day in range(-1 * self.params.coint_look_back_period, 0):
                    prev_first_asset_data.append(data.close[day])

            elif data._name == self.selected_second_asset:
                for day in range(-1 * self.params.coint_look_back_period, 0):
                    prev_second_asset_data.append(data.close[day])

        prev_log_scaled_first_asset_data = np.log(prev_first_asset_data) - np.log(prev_first_asset_data[0])
        prev_log_scaled_second_asset_data = np.log(prev_second_asset_data) - np.log(prev_second_asset_data[0])

        prev_spread = (prev_log_scaled_first_asset_data - self.beta_0 - self.beta_1 * prev_log_scaled_second_asset_data)\
                      /(1 + self.beta_1 ** 2)

        curr_spread_vol = np.std(prev_spread)

        curr_ADF_test_val = self.ADF_test(prev_spread, self.spread_optimal_lag_order)

        if curr_ADF_test_val > self.params.ADF_threshold:
            cointegrated_flag = False
        else :
            self.curr_PS_power = self.params.ADF_threshold - curr_ADF_test_val
            self.curr_PS_val = np.abs(curr_spread/curr_spread_vol) ** self.curr_PS_power

        if self.getposition(first_asset).size == 0:

            if not cointegrated_flag:
                return

            first_asset_proportion = 1 / (1+np.abs(self.beta_1)) * self.max_pos

            second_asset_proportion = self.beta_1 / (1+np.abs(self.beta_1)) * self.max_pos

            # Do Long and Short based on the divergence of the equilibrium of two correlated asset

            if (self.curr_PS_val > self.params.PS_threshold and curr_spread < 0):

                # close the even pos
                if self.params.long_bias:
                    for data in self.datas:
                        if data._name != self.selected_first_asset and data._name != self.selected_second_asset:
                            self.close(data, exectype=bt.Order.Market)

                self.order_target_percent(data = first_asset, target = first_asset_proportion)
                self.order_target_percent(data = second_asset, target = -1*second_asset_proportion)
                self.trading_flag = -1
                self.log('Long : %s & Short : %s' % (first_asset._name ,second_asset._name))
                self.initial_asset_value = self.stats.broker.value[0]

            elif (self.curr_PS_val > self.params.PS_threshold and curr_spread > 0):

                # close the even pos
                if self.params.long_bias:
                    for data in self.datas:
                        if data._name != self.selected_first_asset and data._name != self.selected_second_asset:
                            self.close(data, exectype=bt.Order.Market)

                self.order_target_percent(data = first_asset, target = -1*first_asset_proportion)
                self.order_target_percent(data = second_asset, target = second_asset_proportion)
                self.trading_flag = 1
                self.log('Long : %s & Short : %s' % (second_asset._name, first_asset._name))
                self.initial_asset_value = self.stats.broker.value[0]

        else:

            # Close the Position if the divergenece disappear

            curr_pnl = (self.stats.broker.value[0] - self.initial_asset_value)/self.initial_asset_value

            SL_flag = False

            if curr_pnl < self.SL_rate:
                SL_flag = True

            if self.params.cross_zero:

                if (curr_spread >= 0 and self.trading_flag == -1) or (curr_spread <= 0 and self.trading_flag == 1) or SL_flag\
                        or not cointegrated_flag:
                    self.log('Close Position :  %s & %s' % (first_asset._name ,second_asset._name))

                    # back to even position

                    if self.params.long_bias:
                        number_of_asset = len(self.asset_sample)
                        for data in self.datas:
                            self.order_target_percent(data=data, target=1 / number_of_asset)
                    else:
                        self.close(first_asset, exectype=bt.Order.Market)
                        self.close(second_asset, exectype=bt.Order.Market)


            else:

                if (curr_spread > -1 * self.spread_threshold and self.trading_flag == -1) or \
                        (curr_spread < self.spread_threshold and self.trading_flag == 1) or SL_flag\
                        or not cointegrated_flag:
                    self.log('Close Position :  %s & %s' % (first_asset._name ,second_asset._name))

                    # back to even position

                    if self.params.long_bias:
                        number_of_asset = len(self.asset_sample)
                        for data in self.datas:
                            self.order_target_percent(data=data, target=1 / number_of_asset)
                    else:
                        self.close(first_asset, exectype=bt.Order.Market)
                        self.close(second_asset, exectype=bt.Order.Market)


    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:

            if order.isbuy():
                self.log('BUY EXECUTED, Price : %.2f, Size : %.2f ,Cost : %.2f, Comm %.2f' % (
                         order.executed.price,
                         order.executed.size,
                         order.executed.value,
                         order.executed.comm))

            elif order.issell():
                self.log('SELL EXECUTED, Price : %.2f, Size : %.2f ,Cost : %.2f, Comm %.2f' % (
                         order.executed.price,
                         order.executed.size,
                         order.executed.value,
                         order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):

        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f , NET %.2f' % (trade.pnl, trade.pnlcomm))













