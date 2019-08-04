import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, roll_max_drawdown, alpha_beta, max_drawdown
from termcolor import colored
import csv 
from sklearn.preprocessing import scale
import wandb
import empyrical
import re
import pyarrow.parquet as pq 
import pyarrow as pa
import logging
import random
from env.SwapTradingGraph import SwapTradingGraph
import json

if __name__ == '__main__':
    from util import *
    from constants import *
else:
    from env.util import *
    from env.constants import *


FeatureRow = collections.namedtuple(
    'FeatureRow',
    list(Account._fields)+clean_names(FEATURES)
)

class SwapTradingEnv(gym.Env):
    """
    0-4x the total token amount available in 
    the trading pair of your margin account
    """
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    randomize_values = {
        "leverage": [10,20,30,40,50,60,70,80,90],
        "initial_balance": [0.1, 0.5, 1.0, 1.5, 2],
        "commission": [0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002],
        "execution_certainty": [0.25, 0.5, 0.75, 1]
    }

    def __init__(
        self,
        training=False,
        **kwargs
    ):
        super(SwapTradingEnv, self).__init__()
        
        self.training = training
        self.data_file = kwargs.get('data_file', '../data/test.parquet')
        self.leverage = kwargs.get('leverage', 30)
        self.initial_balance = kwargs.get('initial_balance', 10)
        self.reward_func = kwargs.get('reward_func', 'B')
        self.commission = kwargs.get('commission', 0.0002)
        self.decay = kwargs.get('decay', 0.5)
        self.annualization = kwargs.get('annualization', 3153600)
        self.state_buffer_size = kwargs.get('state_buffer_size', 300)
        self.evaluation_history_size = kwargs.get('evaluation_history_size', 300)
        self.account_history_size = kwargs.get('account_history_size', 300)
        self.scaler_high = kwargs.get('scaler_high', 255)
        self.face_value = kwargs.get('face_value', 100)
        self.maint_margin_ratio = kwargs.get('min_margin_ratio', 0.02)
        self.price_field = kwargs.get('price_field', 'price')
        self.scaler_filename = kwargs.get('scaler_filename', 'BTC-USD-SWAP-250-scaler')
    
        # Account history and net worth dequeue's
        self.state_buffer = collections.deque(maxlen=self.state_buffer_size)
        self.net_worths = collections.deque(maxlen=self.evaluation_history_size)
        self.account_history = collections.deque(maxlen=self.account_history_size)

        # Action space size is 2: one for full long and one for full short
        self.action_space = spaces.Discrete(2)

        self.ind_df, self.obs_df = self._load()

        self.obs_shape = (1, len(FeatureRow._fields))
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=self.obs_shape)

        self.scaler = preprocessing.MinMaxScaler(feature_range=(0,self.scaler_high))
        # self.scaler.fit(self.obs_df)

        if self.leverage >= 99:
            raise ValueError("Can't have leverage greater than 99X")

        self.reset()

    def _load(self):
        df = pq.read_table(self.data_file).to_pandas()
        ind = df[[self.price_field]]
        obs = df[FEATURES]
        obs.columns = clean_names(obs.columns)
        return ind, obs

    def _current_price(self):
        return self.ind_df[self.price_field].values[self.current_step]

    @property
    def _rand(self):
        return random.uniform(0, self.scaler_high)

    def _next_observation(self):
        features = self.obs_df.iloc[self.current_step].to_dict()
        account = self.account_history[-1]._asdict()
        features.update(account)
        row = FeatureRow(**features)
        self.state_buffer.append(row)

        obs = self.scaler.fit_transform(self.state_buffer)[-1]    
        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        obs[obs==-np.inf] = 0
        obs[obs==np.nan] = 0
        obs[obs==np.inf] = 0

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        return obs

    def _take_action(self, action):
        results = []
        has_position = False

        cost = 0
        short_unrealized_pnl = 0 
        long_unrealized_pnl = 0
        current_short_value = 0 
        current_long_value = 0
        margin_ratio = 0
        initial_short_value = 0
        initial_long_value = 0
        short_settlement_price = 0
        long_settlement_price = 0

        current_price = self._current_price()
        account = self.account_history[-1]

        price_per_contract = self.face_value/current_price
        contract_price = current_price/self.face_value
        
        # If a short position exists
        if account.short_position > 0:
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            current_short_value = price_per_contract * account.short_position
            initial_short_value = (self.face_value/account.short_settlement_price) * account.short_position
            short_unrealized_pnl = current_short_value - initial_short_value

        # If a long position exists
        if account.long_position > 0:
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            current_long_value = price_per_contract * account.long_position    
            initial_long_value = (self.face_value/account.long_settlement_price) * account.long_position    
            long_unrealized_pnl = initial_long_value - current_long_value
        
        #
        unrealized_pnl = short_unrealized_pnl + long_unrealized_pnl
        equity = account.total_available_balance + (account.long_position/self.leverage)*price_per_contract + (account.short_position/self.leverage)*price_per_contract
        
        
        current_position_value = current_short_value + current_long_value
        margin_available = equity - (equity*self.maint_margin_ratio)
        
        # Margin ratio = (Balance + RPL + UPL)/Position Value
        if current_position_value > 0:
            margin_ratio = equity/current_position_value
        
        if action != self.prev_action:
            if action == 0:
                # Take a full short position
                next_short = int((margin_available*self.leverage) / price_per_contract)
                next_long = 0
                next_balance = equity - (next_short/self.leverage) * price_per_contract
                
                short_settlement_price = current_price
                
            elif action == 1:
                # Take a full long position 
                next_short = 0 
                next_long = int((margin_available*self.leverage) / price_per_contract)
                next_balance = equity - (next_long/self.leverage) * price_per_contract
                
                long_settlement_price = current_price
                
            next_short_value = next_short*price_per_contract
            next_long_value = next_long*price_per_contract

            prev_dist = np.array([initial_short_value, initial_long_value])
            next_dist = np.array([next_short_value, next_long_value])
            
            cost = sum(abs((next_dist - prev_dist))*self.commission)
            next_balance = next_balance - cost
            next_equity = next_balance + next_short_value + next_long_value
            
            long_position = next_long
            short_position = next_short
            balance = next_balance
            
            long_realized_pnl = long_unrealized_pnl
            short_realized_pnl = short_unrealized_pnl
            realized_pnl = long_realized_pnl + short_realized_pnl

            new_account = Account(
                equity=next_equity,
                margin=margin_available,
                margin_frozen=0,
                margin_ratio=margin_ratio,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_available_balance=balance,
                long_avail_position=0, 
                long_avg_cost=0,
                long_leverage=self.leverage,
                long_liquidation_price=0,
                long_margin=0,
                long_position=long_position,
                long_realized_pnl=0,
                long_settlement_price=long_settlement_price,
                short_avail_position=0,
                short_avg_cost=0,
                short_leverage=self.leverage,
                short_liquidation_price=0,
                short_margin=0,
                short_position=short_position,
                short_realized_pnl=short_realized_pnl,
                short_settlement_price=short_settlement_price,
                rate_limited=0,
                cost=cost
            )

        else:
            self.consec_steps += 1

            new_account = Account(
                equity=equity,
                margin=margin_available,
                margin_frozen=0,
                margin_ratio=margin_ratio,
                realized_pnl=account.realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_available_balance=account.total_available_balance,
                long_avail_position=0, 
                long_avg_cost=0,
                long_leverage=self.leverage,
                long_liquidation_price=0,
                long_margin=0,
                long_position=account.long_position,
                long_realized_pnl=account.long_realized_pnl,
                long_settlement_price=account.long_settlement_price,
                short_avail_position=0,
                short_avg_cost=0,
                short_leverage=self.leverage,
                short_liquidation_price=0,
                short_margin=0,
                short_position=account.short_position,
                short_realized_pnl=account.short_realized_pnl,
                short_settlement_price=account.short_settlement_price,
                rate_limited=0,
                cost=0
            )

        self.net_worths.append(equity)
        self.account_history.append(new_account)
        self.prev_action = action
        self.margin_ratio = margin_ratio
        return new_account

    def _done(self):
        if  self.current_step == len(self.obs_df) - 1:
            return True
        if self.net_worths[-1] < self.initial_balance / 10:
            return True
        if self.margin_ratio < self.maint_margin_ratio and self.margin_ratio > 0:
            return True
        
    def reset(self):
        self.state_buffer.clear()
        self.net_worths.clear()
        self.current_step = 0
        self.trades = []
        self.prev_action = 1
        self.consec_steps = 0
        self.margin_ratio = 0

        self.account_history.append(Account(
                equity=self.initial_balance,
                margin=0,
                margin_frozen=0,
                margin_ratio=0,
                realized_pnl=0,
                unrealized_pnl=0,
                total_available_balance=self.initial_balance,
                long_avail_position=0, # positions available to be closed
                long_avg_cost=0,
                long_leverage=self.leverage,
                long_liquidation_price=0,
                long_margin=0,
                long_position=0,
                long_realized_pnl=0,
                long_settlement_price=0,
                short_avail_position=0,
                short_avg_cost=0,
                short_leverage=self.leverage,
                short_liquidation_price=0,
                short_margin=0,
                short_position=0,
                short_realized_pnl=0,
                short_settlement_price=0,
                rate_limited=0,
                cost=0
        ))

        self.net_worths.append(self.initial_balance)
        return self._next_observation()

    def _reward(self):


        # print("current step: " +str(self.current_step))
        returns = np.diff(self.net_worths)

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(returns, annualization=self.annualization)
        else:
            reward = returns[-1]

        #  # Add decay incentive against 
        # # stat hold position
        # hold_decay = 0
        # margin_dist = 0
        # if self.consec_steps > 60:
        #     hold_decay = self.consec_steps * self.decay
        #     # print("hold_decay: " +str(hold_decay))
        #         # print(hold_decay)
        #     # else:
        #     #     hold_decay = self.consec_steps * self.decay
        #     # hold_decay = 0

        #     # margin_dist = -(self.margin_ratio - self.maint_margin_ratio)

        #     # print("margin_frac: " +str(margin_dist))

        #     # print(margin_frac)
        #     # print("-" *90)
        #     # print("equity before: " +str(equity))
        #     # equity = equity - (margin_frac+hold_decay)
        #     # print("equity after: " +str(equity))

        # reward = reward - (hold_decay + margin_dist)

        return reward if np.isfinite(reward) else 0 

    def step(self, action):
        account = self._take_action(action)
        self.current_step += 1
        obs = self._next_observation()
        reward = self._reward()
        done = self._done()
        return obs, float(reward), done, dict(account._asdict())

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = SwapTradingGraph(
                self.ind_df, 
                self.account_history
            )

        self.viewer.render(
            self.current_step, 
            self.net_worths, 
            self.account_history
        )

if __name__ == "__main__":
    env = SwapTradingEnv()
    for x in range(400):
        obs, reward, done, _ = env.step(random.choice([0,1,2]))