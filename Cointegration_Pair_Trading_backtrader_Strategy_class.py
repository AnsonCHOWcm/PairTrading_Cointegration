import backtrader as bt
import numpy as np
import statsmodels.api as sm
import scipy

class CointegrationStrat(bt.Strategy):

    params = (
        ('printlog', True),
        ('SL_rate', -0.05),
        ('cross_zero', False),
        ('ADF_threshold', -2),
        ('coint_look_back_period', 365),
        ('long_bias', True),
        ('selection_freq', 'month'),
        ('selected_pairs_number', 1),
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

    def ADF_Measure(self, asset_sample, prev_sample_dict, num_pairs):

        n = len(asset_sample)

        selected_asset_pairs = []
        selected_betas = []
        selected_best_lag_order = []
        selected_best_ADF_test_value = []
        for i in range(num_pairs):
            selected_asset_pairs.append(None)
            selected_betas.append(None)
            selected_best_lag_order.append(None)
            selected_best_ADF_test_value.append(self.params.ADF_threshold)

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

                pos = 4
                best_position_flag = False

                while pos != 0 and not best_position_flag:

                    if test_value < selected_best_ADF_test_value[pos-1]:
                        pos -= 1
                        continue
                    else :
                        best_position_flag = True

                if pos < 4:
                    if pos < 3:
                        # adjust the selected pair
                        selected_asset_pairs[pos+1 : ] = selected_asset_pairs[pos:-1]
                        selected_betas[pos + 1:] = selected_betas[pos:-1]
                        selected_best_lag_order[pos + 1:] = selected_best_lag_order[pos:-1]
                        selected_best_ADF_test_value[pos + 1:] = selected_best_ADF_test_value[pos:-1]

                    selected_asset_pairs[pos] = [first_asset, second_asset]
                    selected_betas[pos] = [beta_0, beta_1]
                    selected_best_lag_order[pos] = best_lag_order
                    selected_best_ADF_test_value[pos] = test_value

        return selected_asset_pairs, selected_betas, selected_best_lag_order, selected_best_ADF_test_value

    def previous_performance(self, coint_look_back_period, selected_first_asset_prev_open, selected_first_asset_prev_close,
                             selected_second_asset_prev_open, selected_second_asset_prev_close,
                             beta_0, beta_1, threshold, SL_rate, cross_zero_flag):

        trading_flag = 0
        num_trading_days = len(selected_first_asset_prev_open)
        total_trade_count = 0
        total_PnL = 0


        for day in range(coint_look_back_period, num_trading_days - 1):

            current_first_asset_close_price = selected_first_asset_prev_close[day]
            current_second_asset_close_price = selected_second_asset_prev_close[day]
            scaled_current_first_asset_close_price = np.log(selected_first_asset_prev_close[day]) - np.log(selected_first_asset_prev_close[coint_look_back_period-1])
            scaled_current_second_asset_close_price = np.log(selected_second_asset_prev_close[day]) - np.log(selected_second_asset_prev_close[coint_look_back_period-1])
            next_first_asset_open_price = selected_first_asset_prev_open[day+1]
            next_second_asset_open_price = selected_second_asset_prev_open[day+1]

            prev_first_asset_close_price_ls = selected_first_asset_prev_close[day - coint_look_back_period:day]
            prev_second_asset_close_price_ls = selected_second_asset_prev_close[day - coint_look_back_period:day]

            scaled_prev_first_asset_close_price_ls = np.log(prev_first_asset_close_price_ls) - np.log(prev_first_asset_close_price_ls[0])
            scaled_prev_second_asset_close_price_ls = np.log(prev_second_asset_close_price_ls) - np.log(prev_first_asset_close_price_ls[0])

            current_spread_noise = (scaled_current_first_asset_close_price - beta_0 - beta_1 *
                                    scaled_current_second_asset_close_price) / (1 + beta_1 ** 2)

            prev_std_spread_noise_ls = (scaled_prev_first_asset_close_price_ls - beta_0 - beta_1 *
                                    scaled_prev_second_asset_close_price_ls) / (1 + beta_1 ** 2)

            spread_noise_std = np.std(prev_std_spread_noise_ls)

            current_std_spread_noise = current_spread_noise/spread_noise_std

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

                    trade_PnL = (beta_1 / (1 + np.abs(beta_1))) * (
                                current_second_asset_close_price - long_cost) / long_cost - (
                                            1 / (1 + np.abs(beta_1))) * (
                                            current_first_asset_close_price - short_cost) / short_cost

                    if cross_zero_flag:

                        if trade_PnL > SL_rate and current_std_spread_noise > 0 and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_second_asset_open_price
                            short_close = next_first_asset_open_price
                            PnL = (beta_1 / (1 + np.abs(beta_1))) * (long_close - long_cost) / long_cost - (
                                        1 / (1 + np.abs(beta_1))) * (short_close - short_cost) / short_cost


                    else :

                        if trade_PnL > SL_rate and current_std_spread_noise > threshold and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_second_asset_open_price
                            short_close = next_first_asset_open_price
                            PnL = (beta_1 / (1 + np.abs(beta_1))) * (long_close - long_cost) / long_cost - (
                                        1 / (1 + np.abs(beta_1))) * (short_close - short_cost) / short_cost

                    total_PnL += PnL

                else:

                    trade_PnL = (1 / (1 + np.abs(beta_1))) * (
                                current_first_asset_close_price - long_cost) / long_cost - (
                                            beta_1 / (1 + np.abs(beta_1))) * (
                                            current_second_asset_close_price - short_cost) / short_cost

                    if cross_zero_flag :

                        if trade_PnL > SL_rate and current_std_spread_noise < 0 and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_first_asset_open_price
                            short_close = next_second_asset_open_price
                            PnL = (1 / (1 + np.abs(beta_1))) * (long_close - long_cost) / long_cost - (
                                        beta_1 / (1 + np.abs(beta_1))) * (short_close - short_cost) / short_cost

                    else:

                        if trade_PnL > SL_rate and current_std_spread_noise < -1 * threshold and day != num_trading_days - 2:
                            continue

                        else:

                            trading_flag = 0
                            long_close = next_first_asset_open_price
                            short_close = next_second_asset_open_price
                            PnL = (1 / (1 + np.abs(beta_1))) * (long_close - long_cost) / long_cost - (
                                        beta_1 / (1 + np.abs(beta_1))) * (short_close - short_cost) / short_cost

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

    def optimal_threshold(self, coint_look_back_period, selected_first_asset_prev_open, selected_first_asset_prev_close,
                          selected_second_asset_prev_open, selected_second_asset_prev_close,
                          beta_0, beta_1, threshold_set, SL_rate):

        Total_count = []
        Avg_PnL = []

        for threshold in threshold_set:

            curr_trade_count, curr_avg_pnl = self.previous_performance(coint_look_back_period,
                                                                       selected_first_asset_prev_open,
                                                                       selected_first_asset_prev_close,
                                                                       selected_second_asset_prev_open,
                                                                       selected_second_asset_prev_close,
                                                                       beta_0, beta_1, threshold, SL_rate,
                                                                       self.params.cross_zero)

            Total_count.append(curr_trade_count)
            Avg_PnL.append(curr_avg_pnl)

        if len(np.where(Total_count == 0)) == len(Total_count):
            return 1

        print('Start threshold optimization')

        best_threshold = self.threshold_optimization(threshold_set, Total_count, Avg_PnL)

        print('Finished threshold')

        return best_threshold

    def flow_management(self, trading_flag, trading_pairs, trading_betas, long_only_flag, max_pos, asset_universe):

        number_pending_trade_pairs = 0
        asset_position_dict = {}
        for asset_name in asset_universe:
            asset_position_dict[asset_name] = float(0)

        for flag in trading_flag:
            if flag != 0:
                number_pending_trade_pairs += 1

        if number_pending_trade_pairs == 0:

            if long_only_flag:
                for asset_name in asset_universe:
                    asset_position_dict[asset_name] = float(1/len(asset_universe))

        else:

            each_pair_prop = max_pos / number_pending_trade_pairs

            for pos, pending_pair in enumerate(trading_pairs):

                if trading_flag[pos] == 0 :
                    continue

                else:

                    first_asset_name = pending_pair[0]
                    second_asset_name = pending_pair[1]
                    beta_1 = trading_betas[pos][1]

                    first_asset_prop = 1/(1 + np.abs(beta_1)) * each_pair_prop
                    second_asset_prop = beta_1/(1 + np.abs(beta_1)) * each_pair_prop

                    if trading_flag[pos] == 1:

                        asset_position_dict[first_asset_name] = asset_position_dict[first_asset_name] - first_asset_prop
                        asset_position_dict[second_asset_name] = asset_position_dict[second_asset_name] + second_asset_prop

                    else:

                        asset_position_dict[first_asset_name] = asset_position_dict[first_asset_name] + first_asset_prop
                        asset_position_dict[second_asset_name] = asset_position_dict[second_asset_name] - second_asset_prop


        return asset_position_dict

    def __init__(self):

        self.trading_days = 0
        self.pair_selection_flag = False
        self.current_time = None
        self.prev_time = None
        # Storing the selected pairs and its sample log return
        self.selected_asset_pairs = []
        self.selected_asset_pair_log_price_ref = []
        for i in range(self.params.selected_pairs_number):
            self.selected_asset_pair_log_price_ref.append(None)
        self.selected_betas = []
        self.selected_best_lag_order = []
        self.selected_spread_thresholds = []
        for i in range(self.params.selected_pairs_number):
            self.spread_thresholds.append(None)
        self.trading_flag = []
        self.asset_sample = []
        self.prev_sample_dict = {}
        # Storing the portfilio spending
        self.max_pos = 0.98
        self.SL_rate = self.params.SL_rate
        self.threshold_set = np.array(range(1, 80)) * 0.025
        self.initial_asset_value = None
        # storing the trading pairs info
        self.trading_pairs = []
        self.trading_pair_log_price_ref = []
        self.trading_betas = []
        self.trading_best_lag_order = []
        self.trading_spread_thresholds = []
        # managing asset position
        self.asset_universe = []
        self.asset_position_dict = {}

    def next(self):

        if self.trading_days == 0:
            for data in self.datas:
                self.asset_position_dict[data._name] = 0
                self.asset_universe.append(data._name)

        self.trading_days += 1

        if self.trading_days <= self.params.coint_look_back_period:
            return

        if self.current_time is None:
            self.current_time = self.datas[0].datetime.date(0).month

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

            self.pair_selection_flag = False
            # Storing the close price of the asset in the previous month

            self.asset_sample = []

            for data in self.datas:

                self.asset_sample.append(data._name)

                asseet_price_sample = []

                for day in range(-1 * self.params.coint_look_back_period, 0):
                    asseet_price_sample.append(data.close[day])

                self.prev_sample_dict[data._name] = asseet_price_sample

            # Selecting the Trading Pairs for next month based on TLS regrssion and checking the ADF test value

            selected_asset_pairs, selected_betas, selected_best_lag_order, selected_best_ADF_test_value = \
            self.ADF_Measure(self.asset_sample, self.prev_sample_dict, self.params.selected_pairs_number)

            # record the ref price for all selected pairs
            for pos, curr_pairs in enumerate(selected_asset_pairs):
                if curr_pairs is None:
                    continue
                else:
                    for data in self.datas:
                        if data._name == curr_pairs[0]:
                            first_asset_log_price_ref = np.log(data.close[0])
                        elif data._name == curr_pairs[1]:
                            second_asset_log_price_ref = np.log(data.close[0])

                    self.selected_asset_pair_log_price_ref[pos] = [first_asset_log_price_ref, second_asset_log_price_ref]

            # renew the selected pair information

            self.selected_asset_pairs = selected_asset_pairs
            self.selected_betas = selected_betas
            self.selected_best_lag_order = selected_best_lag_order

            if self.selected_asset_pairs[0] is None :
                return

            # Saving the log price reference for this month and selecting the threshold

            for pos, curr_pairs in enumerate(selected_asset_pairs):

                if curr_pairs is None:
                    continue

                selected_first_asset = curr_pairs[0]
                selected_second_asset = curr_pairs[1]

                selected_first_asset_prev_open = []
                selected_first_asset_prev_close = []
                selected_second_asset_prev_open = []
                selected_second_asset_prev_close = []

                for data in self.datas:

                     if data._name == selected_first_asset:
                         for day in range(-2 * self.params.coint_look_back_period, 0):
                             selected_first_asset_prev_open.append(data.open[day])
                             selected_first_asset_prev_close.append(data.close[day])

                     elif data._name == selected_second_asset:
                         for day in range(-2 * self.params.coint_look_back_period, 0):
                             selected_second_asset_prev_open.append(data.open[day])
                             selected_second_asset_prev_close.append(data.close[day])

                curr_pair_spread_threshold = self.optimal_threshold(self.params.coint_look_back_period,
                                                                    selected_first_asset_prev_open, selected_first_asset_prev_close,
                                                                    selected_second_asset_prev_open, selected_second_asset_prev_close,
                                                                    self.beta_0, self.beta_1, self.threshold_set, self.SL_rate)

                self.selected_spread_thresholds[pos] = curr_pair_spread_threshold

            return

        if self.prev_time is None:
            return

        if self.selected_first_asset is None :
            return

        # computing the signal for existing traded pairs

        for pos, trading_pair in enumerate(self.trading_pairs):

            selected_first_asset = trading_pair[0]
            selected_second_asset = trading_pair[1]
            selected_first_asset_log_price_ref = self.trading_pair_log_price_ref[pos][0]
            selected_second_asset_log_price_ref = self.trading_pair_log_price_ref[pos][1]
            beta_0 = self.trading_betas[pos][0]
            beta_1 = self.trading_betas[pos][1]
            spread_optimal_lag_order = self.trading_best_lag_order[pos]

            prev_first_asset_data = []
            prev_second_asset_data = []

            for data in self.datas:

                if data._name == selected_first_asset:
                    first_asset = data
                    curr_first_asset_scaled_price = np.log(data.close[0]) - selected_first_asset_log_price_ref
                    for day in range(-1 * self.params.coint_look_back_period, 0):
                        prev_first_asset_data.append(data.close[day])


                elif data._name == selected_second_asset:
                    second_asset = data
                    curr_second_asset_scaled_price = np.log(data.close[0]) - selected_second_asset_log_price_ref
                    for day in range(-1 * self.params.coint_look_back_period, 0):
                        prev_first_asset_data.append(data.close[day])

            curr_spread = (curr_first_asset_scaled_price - beta_1 * curr_second_asset_scaled_price - beta_0)/(1 + beta_1 ** 2)

            prev_log_scaled_first_asset_data = np.log(prev_first_asset_data) - np.log(prev_first_asset_data[0])
            prev_log_scaled_second_asset_data = np.log(prev_second_asset_data) - np.log(prev_second_asset_data[0])

            prev_spread = (prev_log_scaled_first_asset_data - beta_0 - beta_1 * prev_log_scaled_second_asset_data)/(1 + beta_1 ** 2)

            curr_ADF_test_val = self.ADF_test(prev_spread, spread_optimal_lag_order)

            if curr_ADF_test_val > self.params.ADF_threshold:
                self.trading_flag[pos] = 0

            else :
                curr_spread_vol = np.std(prev_spread)
                curr_std_spread = curr_spread / curr_spread_vol

                if self.params.cross_zero:
                    exit_threshold = 0
                else :
                    exit_threshold = self.trading_spread_thresholds[pos]

                if (self.trading_flag[pos] == 1 and curr_std_spread < exit_threshold) or \
                        (self.trading_flag[pos] == -1 and curr_std_spread > -1 * exit_threshold):
                    self.trading_flag[pos] = 0

        # computing signal for selected pairs

        for pos, curr_pair in enumerate(self.selected_asset_pairs):

            if curr_pair in self.trading_pairs:
                continue

            selected_first_asset = curr_pair[0]
            selected_second_asset = curr_pair[1]
            selected_first_asset_log_price_ref = self.selected_asset_pair_log_price_ref[pos][0]
            selected_second_asset_log_price_ref = self.selected_asset_pair_log_price_ref[pos][1]
            beta_0 = self.selected_betas[pos][0]
            beta_1 = self.selected_betas[pos][1]
            spread_optimal_lag_order = self.selected_best_lag_order[pos]

            prev_first_asset_data = []
            prev_second_asset_data = []

            for data in self.datas:

                if data._name == selected_first_asset:
                    first_asset = data
                    curr_first_asset_scaled_price = np.log(data.close[0]) - selected_first_asset_log_price_ref
                    for day in range(-1 * self.params.coint_look_back_period, 0):
                        prev_first_asset_data.append(data.close[day])


                elif data._name == selected_second_asset:
                    second_asset = data
                    curr_second_asset_scaled_price = np.log(data.close[0]) - selected_second_asset_log_price_ref
                    for day in range(-1 * self.params.coint_look_back_period, 0):
                        prev_first_asset_data.append(data.close[day])

            curr_spread = (curr_first_asset_scaled_price - beta_1 * curr_second_asset_scaled_price - beta_0)/(1 + beta_1 ** 2)

            prev_log_scaled_first_asset_data = np.log(prev_first_asset_data) - np.log(prev_first_asset_data[0])
            prev_log_scaled_second_asset_data = np.log(prev_second_asset_data) - np.log(prev_second_asset_data[0])

            prev_spread = (prev_log_scaled_first_asset_data - beta_0 - beta_1 * prev_log_scaled_second_asset_data)/(1 + beta_1 ** 2)

            curr_ADF_test_val = self.ADF_test(prev_spread, spread_optimal_lag_order)

            if curr_ADF_test_val > self.params.ADF_threshold:
                continue

            else :
                curr_spread_vol = np.std(prev_spread)
                curr_std_spread = curr_spread / curr_spread_vol

                trading_threshold = self.selected_spread_thresholds[pos]

                if (curr_std_spread > trading_threshold):
                    self.trading_pairs.append(curr_pair)
                    self.trading_pair_log_price_ref.append(self.selected_asset_pair_log_price_ref[pos])
                    self.trading_betas.apppend(self.selected_betas[pos])
                    self.trading_best_lag_order.append(self.selected_best_lag_order[pos])
                    self.trading_spread_thresholds.append(self.selected_spread_thresholds[pos])
                    self.trading_flag.append(1)
                elif (curr_std_spread < -1 * trading_threshold):
                    self.trading_pairs.append(curr_pair)
                    self.trading_pair_log_price_ref.append(self.selected_asset_pair_log_price_ref[pos])
                    self.trading_betas.apppend(self.selected_betas[pos])
                    self.trading_best_lag_order.append(self.selected_best_lag_order[pos])
                    self.trading_spread_thresholds.append(self.selected_spread_thresholds[pos])
                    self.trading_flag.append(-1)

        # Flow Management

        self.asset_position_dict = self.flow_management(self.trading_flag, self.trading_pairs, self.trading_betas,
                                                        self.params.long_bias, self.max_pos, self.asset_universe)

        # Adjusting the asset position according to the flow

        for data in self.datas:
            curr_asset = data
            curr_asset_name = data._name
            asset_pos = self.asset_position_dict[curr_asset_name]
            self.order_target_percent(data=curr_asset, target=asset_pos)
            self.log('Asset Name : %s, Position : %f' %(curr_asset_name, asset_pos))

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













