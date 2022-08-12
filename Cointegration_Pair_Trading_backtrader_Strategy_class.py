import backtrader as bt
import numpy as np
import statsmodels.api as sm
import scipy

class CointegrationStrat(bt.Strategy):

    params = (
        ('printlog', True),
        ('SL_rate', -0.05),
        ('cross_zero', False),
    )

    def log(self ,txt ,doprint = None):

        dt = self.datas[0].datetime.date(0)
        tm = self.datas[0].datetime.time(0)

        if doprint == None :

            doprint = self.params.printlog

        if doprint :

            print('%s - %s, %s' % (dt.isoformat(), tm.isoformat(), txt))

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
        best_lag_order = 1

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

            X = sm.add_constant(X)

            OLS_model = sm.OLS(Y, X)
            res = OLS_model.fit()

            if (res.info_criteria('bic') < best_bic):
                best_bic = res.info_criteria('bic')
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

        OLS_model = sm.OLS(Y, X)
        res = OLS_model.fit()

        return res.tvalues[0]

    def ADF_Measure(self, asset_sample, prev_sample_dict):

        n = len(asset_sample)
        best_ADF_test_value = 100000
        selected_first_asset = None
        selected_second_asset = None
        selected_beta_1 = None
        selected_beta_0 = None

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


                spread = (test_sample_data[:, 0] - beta_0 - beta_1 * test_sample_data[:, 1]) / np.sqrt(
                    1 + beta_1 ** 2)

                best_lag_order = self.spread_lag_order(spread, 1, 10)


                test_value = self.ADF_test(spread, best_lag_order)

                if test_value < best_ADF_test_value:
                    best_ADF_test_value = test_value
                    selected_first_asset = first_asset
                    selected_second_asset = second_asset
                    selected_beta_1 = beta_1
                    selected_beta_0 = beta_0

        return selected_beta_0, selected_beta_1, selected_first_asset, selected_second_asset, best_ADF_test_value

    def previous_performance(self,selected_first_asset_prev_open,selected_first_asset_prev_close,
                             selected_second_asset_prev_open, selected_second_asset_prev_close,
                             beta_0, beta_1, threshold, SL_rate, cross_zero_flag):

        trading_flag = 0
        num_trading_days = len(selected_first_asset_prev_open)
        total_trade_count = 0
        total_PnL = 0


        for day in range(1, num_trading_days - 1):

            current_first_asset_close_price = selected_first_asset_prev_close[day]
            current_second_asset_close_price = selected_second_asset_prev_close[day]
            scaled_current_first_asset_close_price = np.log(selected_first_asset_prev_close[day]) - np.log(selected_first_asset_prev_close[0])
            scaled_current_second_asset_close_price = np.log(selected_second_asset_prev_close[day]) - np.log(selected_second_asset_prev_close[0])
            next_first_asset_open_price = selected_first_asset_prev_open[day+1]
            next_second_asset_open_price = selected_second_asset_prev_open[day+1]

            current_std_spread_noise = (scaled_current_first_asset_close_price - beta_0 - beta_1 *
                                    scaled_current_second_asset_close_price) / np.sqrt(1 + beta_1 ** 2)

            if trading_flag == 0:

                if abs(current_std_spread_noise) < threshold:
                    continue

                elif current_std_spread_noise > threshold:

                    trading_flag = 1
                    long_cost = next_second_asset_open_price
                    short_cost = next_first_asset_open_price
                    total_trade_count += 1

                else:

                    trading_flag = -1
                    long_cost = next_first_asset_open_price
                    short_cost = next_second_asset_open_price
                    total_trade_count += 1

            else:

                if trading_flag == 1:

                    trade_PnL = (beta_1 / (1 + beta_1)) * (
                                current_second_asset_close_price - long_cost) / long_cost - (
                                            1 / (1 + beta_1)) * (
                                            current_first_asset_close_price - short_cost) / short_cost

                    if cross_zero_flag:

                        if trade_PnL > SL_rate and current_std_spread_noise < 0 and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_second_asset_open_price
                            short_close = next_first_asset_open_price
                            PnL = (beta_1 / (1 + beta_1)) * (long_close - long_cost) / long_cost - (
                                        1 / (1 + beta_1)) * (short_close - short_cost) / short_cost


                    else :

                        if trade_PnL > SL_rate and current_std_spread_noise < threshold and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_second_asset_open_price
                            short_close = next_first_asset_open_price
                            PnL = (beta_1 / (1 + beta_1)) * (long_close - long_cost) / long_cost - (
                                    1 / (1 + beta_1)) * (short_close - short_cost) / short_cost

                    total_PnL += PnL

                else:

                    trade_PnL = (1 / (1 + beta_1)) * (
                                current_first_asset_close_price - long_cost) / long_cost - (
                                            beta_1 / (1 + beta_1)) * (
                                            current_second_asset_close_price - short_cost) / short_cost

                    if cross_zero_flag :

                        if trade_PnL > SL_rate and current_std_spread_noise > 0 and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_first_asset_open_price
                            short_close = next_second_asset_open_price
                            PnL = (1 / (1 + beta_1)) * (long_close - long_cost) / long_cost - (
                                        beta_1 / (1 + beta_1)) * (short_close - short_cost) / short_cost

                    else:

                        if trade_PnL > SL_rate and current_std_spread_noise > -1 * threshold and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_first_asset_open_price
                            short_close = next_second_asset_open_price
                            PnL = (1 / (1 + beta_1)) * (long_close - long_cost) / long_cost - (
                                    beta_1 / (1 + beta_1)) * (short_close - short_cost) / short_cost

                    total_PnL += PnL

        if total_trade_count == 0 :
            avg_PnL = 0
        else :
            avg_PnL = float(total_PnL/total_trade_count)


        return total_trade_count, avg_PnL

    def regularization_cost(self, regularized_value, trade_off_rate, Target_value):

        accuracy_cost = sum((Target_value - regularized_value) ** 2)
        smoonthness_cost = sum((regularized_value[1:] - regularized_value[:-1]) ** 2)

        return accuracy_cost + trade_off_rate * smoonthness_cost

    def threshold_optimization(self, threshold_set, Total_count, Avg_PnL):

        trade_off_rate_set = np.array(range(-12, 13))
        Total_count_reg_cost = []
        Avg_PnL_reg_cost = []

        for trade_off_rate in trade_off_rate_set:
            Total_count_res = scipy.optimize.minimize(self.regularization_cost, np.zeros(len(Total_count)),
                                                      args=(np.exp(trade_off_rate), Total_count))
            Avg_PnL_res = scipy.optimize.minimize(self.regularization_cost, np.zeros(len(Avg_PnL)),
                                                  args=(np.exp(trade_off_rate), Avg_PnL))

            Total_count_cost = self.regularization_cost(Total_count_res.x, np.exp(trade_off_rate), Total_count)
            Avg_PnL_cost = self.regularization_cost(Avg_PnL_res.x, np.exp(trade_off_rate), Avg_PnL)

            Total_count_reg_cost.append(Total_count_cost)
            Avg_PnL_reg_cost.append(Avg_PnL_cost)

        Total_count_distance = 10000
        Total_count_mean = np.mean(Total_count_reg_cost)
        Total_count_target = -1

        Avg_PnL_distance = 10000
        Avg_PnL_mean = np.mean(Avg_PnL_reg_cost)
        Avg_PnL_target = -1

        for i in range(len(Avg_PnL_reg_cost)):

            if abs(Total_count_reg_cost[i] - Total_count_mean) < Total_count_distance:
                Total_count_distance = abs(Total_count_reg_cost[i] - Total_count_mean)

                Total_count_target = i

            if abs(Avg_PnL_reg_cost[i] - Avg_PnL_mean) < Avg_PnL_distance:
                Avg_PnL_distance = abs(Avg_PnL_reg_cost[i] - Avg_PnL_mean)

                Avg_PnL_target = i

        Total_count_best_trade_off_rate = trade_off_rate_set[Total_count_target]
        Avg_PnL_best_trade_off_rate = trade_off_rate_set[Avg_PnL_target]

        Total_count_optimal_res = scipy.optimize.minimize(self.regularization_cost, np.zeros(len(Total_count)),
                                                          args=(np.exp(Total_count_best_trade_off_rate), Total_count))
        Avg_PnL_optimal_res = scipy.optimize.minimize(self.regularization_cost, np.zeros(len(Total_count)),
                                                      args=(np.exp(Avg_PnL_best_trade_off_rate), Avg_PnL))

        smoothed_Total_PnL = Total_count_optimal_res.x * Avg_PnL_optimal_res.x

        selected_threshold = threshold_set[np.where(smoothed_Total_PnL == max(smoothed_Total_PnL))[0][0]]

        return (selected_threshold)

    def optimal_threshold(self, selected_first_asset_prev_open, selected_first_asset_prev_close,
                          selected_second_asset_prev_open, selected_second_asset_prev_close,
                          beta_0, beta_1, threshold_set, SL_rate):

        Total_count = []
        Avg_PnL = []

        for threshold in threshold_set:

            curr_trade_count, curr_avg_pnl = self.previous_performance(selected_first_asset_prev_open,
                                                                  selected_first_asset_prev_close,
                                                                  selected_second_asset_prev_open,
                                                                  selected_second_asset_prev_close,
                                                                  beta_0, beta_1, threshold, SL_rate, self.params.cross_zero)

            Total_count.append(curr_trade_count)
            Avg_PnL.append(curr_avg_pnl)

        if len(np.where(Total_count == 0)) == len(Total_count):
            return 1

        print('Start threshold optimization')

        best_threshold = self.threshold_optimization(threshold_set, Total_count, Avg_PnL)

        print('Finished threshold')

        return best_threshold


    def __init__(self):

        self.current_month = None
        self.prev_month = None
        # Storing the selected pair and its sample log return
        self.beta_0 = None
        self.beta_1 = None
        self.selected_first_asset = None
        self.selected_second_asset = None
        self.selected_first_asset_log_price_ref = None
        self.selected_second_asset_log_price_ref = None
        self.spread_threshold = None
        self.trading_flag = None
        self.asset_sample = []
        self.prev_sample_dict = {}
        # Storing the portfilio spending
        self.max_pos = 0.98
        self.SL_rate = self.params.SL_rate
        self.threshold_set = np.array(range(1 , 80)) * 0.025
        self.initial_asset_value = None


    def next(self):

        if self.current_month is None:
            self.current_month = self.datas[0].datetime.date(0).month

        # Entering Next Month, renew the selected pairs
        if self.datas[0].datetime.date(0).month != self.current_month:

            # Amending the reference month
            self.prev_month = self.current_month
            self.current_month = self.datas[0].datetime.date(0).month

            # Counting the number of trading days in the prev month
            trading_date_count = 0
            while self.datas[0].datetime.date(trading_date_count-1).month == self.prev_month:
                trading_date_count -= 1

            # Storing the close price of the asset in the previous month

            for data in self.datas:

                self.asset_sample.append(data._name)

                asseet_price_sample = []

                for day in range(trading_date_count, 0):
                    asseet_price_sample.append(data.close[day])

                self.prev_sample_dict[data._name] = asseet_price_sample

            # Selecting the Trading Pairs for next month based on TLS regrssion and checking the ADF test value

            self.beta_0, self.beta_1, selected_first_asset, selected_second_asset, test_value = \
            self.ADF_Measure(self.asset_sample, self.prev_sample_dict)

            self.log('Selected pair %s & %s' %(selected_first_asset, selected_second_asset))
            self.log('Selected ADF value : %f' % test_value)

            if self.selected_first_asset != selected_first_asset or self.selected_second_asset != selected_second_asset:
                for data in self.datas:
                    if data._name == self.selected_first_asset or data._name == self.selected_second_asset:
                        self.close(data)
                        self.log('Close Position : %s' % data._name)

            self.selected_first_asset = selected_first_asset
            self.selected_second_asset = selected_second_asset

            # Saving the log price reference for this month and selecting the threshold
            selected_first_asset_prev_open = []
            selected_first_asset_prev_close = []
            selected_second_asset_prev_open = []
            selected_second_asset_prev_close = []

            for data in self.datas:

                 if data._name == self.selected_first_asset:
                     self.selected_first_asset_log_price_ref = np.log(data.close[0])
                     for day in range(trading_date_count, 0):
                         selected_first_asset_prev_open.append(data.open[day])
                         selected_first_asset_prev_close.append(data.close[day])

                 elif data._name == self.selected_second_asset:
                     self.selected_second_asset_log_price_ref = np.log(data.close[0])
                     for day in range(trading_date_count, 0):
                         selected_second_asset_prev_open.append(data.open[day])
                         selected_second_asset_prev_close.append(data.close[day])

            self.spread_threshold = self.optimal_threshold(selected_first_asset_prev_open, selected_first_asset_prev_close,
                          selected_second_asset_prev_open, selected_second_asset_prev_close,
                          self.beta_0, self.beta_1, self.threshold_set, self.SL_rate)

            self.log('Finish Spreaed Threshold Searching')

            return

        if self.prev_month is None:
            return

        for data in self.datas:

            if data._name == self.selected_first_asset:
                first_asset = data
                curr_first_asset_scaled_price =  np.log(data.close[0]) - self.selected_first_asset_log_price_ref

            elif data._name == self.selected_second_asset:
                second_asset = data
                curr_second_asset_scaled_price = np.log(data.close[0]) - self.selected_second_asset_log_price_ref

        curr_spread = (curr_first_asset_scaled_price - self.beta_1 * curr_second_asset_scaled_price - self.beta_0)\
                      /np.sqrt(1+self.beta_1**2)


        if self.getposition(first_asset).size == 0:

            first_asset_proportion = 1 / (1+self.beta_1) * self.max_pos

            second_asset_proportion = self.beta_1 / (1+self.beta_1) * self.max_pos

            # Do Long and Short based on the divergence of the equilibrium of two correlated asset

            if (curr_spread < -1 * self.spread_threshold):

                self.order_target_percent(data = first_asset, target = first_asset_proportion)
                self.order_target_percent(data = second_asset, target = -1*second_asset_proportion)
                self.trading_flag = -1
                self.log('Long : %s & Short : %s' % (first_asset._name ,second_asset._name))
                self.initial_asset_value = self.stats.broker.value[0]

            elif (curr_spread > self.spread_threshold):

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

                if (curr_spread >= 0 and self.trading_flag == -1) or (curr_spread <= 0 and self.trading_flag == 1) or SL_flag:
                    self.close(first_asset, exectype=bt.Order.Market)
                    self.close(second_asset, exectype=bt.Order.Market)
                    self.log('Close Position :  %s & %s' % (first_asset._name ,second_asset._name))

            else:

                if (curr_spread > -1 * self.spread_threshold and self.trading_flag == -1) or \
                        (curr_spread < self.spread_threshold and self.trading_flag == 1) or SL_flag:
                    self.close(first_asset, exectype=bt.Order.Market)
                    self.close(second_asset, exectype=bt.Order.Market)
                    self.log('Close Position :  %s & %s' % (first_asset._name ,second_asset._name))

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













