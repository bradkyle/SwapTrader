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
import random

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

FeatureRow2 = collections.namedtuple(
    'FeatureRow',
    clean_names(FEATURES)
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
        "commission": [-0.00025, 0.0, 0.00025, 0.0005, 0.00075],
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
        self.enable_decay = kwargs.get('enable_decay', False)
        self.randomize = kwargs.get('randomize', False)
        self.enable_hold_decay = kwargs.get('enable_hold_decay', True)
        self.decay = kwargs.get('decay', 0.05)
        self.threshold_max_steps = kwargs.get('threshold_max_steps', 30)
        self.annualization = kwargs.get('annualization', 3153600)
        self.state_buffer_size = kwargs.get('state_buffer_size', 300)
        self.evaluation_history_size = kwargs.get('evaluation_history_size', 300)
        self.account_history_size = kwargs.get('account_history_size', 300)
        self.scaler_high = kwargs.get('scaler_high', 255)
        self.min_consec_steps = kwargs.get('min_consec_steps', 1)
        self.face_value = kwargs.get('face_value', 100)
        self.maint_margin_ratio = kwargs.get('min_margin_ratio', 0.02)        
        self.num_actions = kwargs.get('num_actions', 2)
        self.price_field = kwargs.get('price_field', 'price')
        self.obs_type = kwargs.get('obs_type', 'obs_1')
        self.scaler_filename = kwargs.get('scaler_filename', 'BTC-USD-SWAP-250-scaler')

        if self.obs_type == 'obs_1':
            self.next_obs = self._obs_1
            self.init_row = self._obs_init_1
            self.obs_shape = (1, len(FeatureRow._fields))
            
        elif self.obs_type == 'obs_2':
            self.next_obs = self._obs_2
            self.init_row = self._obs_init_2
            self.obs_shape = (1, len(FEATURES))

        self.observation_space = spaces.Box(low=0, high=self.scaler_high, dtype=np.uint8, shape=self.obs_shape)
    
        # Account history and net worth dequeue's
        self.state_buffer = collections.deque(maxlen=self.state_buffer_size)
        self.net_worths = collections.deque(maxlen=self.evaluation_history_size)
        self.account_history = collections.deque(maxlen=self.account_history_size)

        # Action space size is 2: one for full long and one for full short
        self.action_space = spaces.Discrete(self.num_actions)

        self.ind_df, self.obs_df = self._load()

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

    def _obs_init_1(self, step):
        features = self.obs_df.iloc[step].to_dict()
        account = self.account_history[-1]._asdict()
        features.update(account)
        row = FeatureRow(**features)
        self.state_buffer.append(row)

    def _obs_1(self):
        features = self.obs_df.iloc[self.current_step].to_dict()
        account = self.account_history[-1]._asdict()
        features.update(account)
        row = FeatureRow(**features)
        self.state_buffer.append(row)

        obs = np.array(self.state_buffer)

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        obs = self.scaler.fit_transform(obs)[-1]    
        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        obs[obs==-np.inf] = 0
        obs[obs==np.nan] = 0
        obs[obs==np.inf] = 0

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        return [obs]

    def _obs_init_2(self, step):
        features = self.obs_df.iloc[step].to_dict()
        row = FeatureRow2(**features)
        self.state_buffer.append(row)

    def _obs_2(self):
        features = self.obs_df.iloc[self.current_step].to_dict()
        row = FeatureRow2(**features)
        self.state_buffer.append(row)

        obs = np.array(self.state_buffer)

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        obs = self.scaler.fit_transform(obs)[-1]    
        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        obs[obs==-np.inf] = 0
        obs[obs==np.nan] = 0
        obs[obs==np.inf] = 0

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        return [obs]

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
        current_position_value = current_short_value + current_long_value
        equity = account.total_available_balance + (initial_long_value + initial_short_value + unrealized_pnl)/self.leverage
        
        margin_available = equity - (equity*self.maint_margin_ratio)

        # long_margin = self.face_value * account.long_position / (account.long_settlement_price * self.leverage)
        # short_margin = self.face_value * account.short_position / (account.short_settlement_price * self.leverage)

        # Margin ratio = (Balance + RPL + UPL)/Position Value
        if current_position_value > 0:
            margin_ratio = equity/current_position_value
        
        actual_action = action
        if self.consec_steps <= self.min_consec_steps:
            action = self.prev_action

        # TODO add min position 
        if action != self.prev_action:
            
            if self.num_actions == 5:
                if action == 0:
                    # Take a full short position
                    next_short = int((margin_available*self.leverage) / price_per_contract)
                    next_long = 0
                    next_balance = equity - (next_short/self.leverage) * price_per_contract
                    
                    short_settlement_price = current_price

                elif action == 1:
                    # Take a full short position
                    next_short = int(((margin_available/2)*self.leverage) / price_per_contract)
                    next_long = 0
                    next_balance = equity - (next_short/self.leverage) * price_per_contract
                    
                    short_settlement_price = current_price

                elif action == 2:
                    # Take a flat position 
                    next_short = 0 
                    next_long = 0
                    next_balance = equity
                    
                elif action == 3:
                    # Take a full long position 
                    next_short = 0 
                    next_long = int(((margin_available/2)*self.leverage) / price_per_contract)
                    next_balance = equity - (next_long/self.leverage) * price_per_contract
                    
                    long_settlement_price = current_price

                elif action == 4:
                    # Take a full long position 
                    next_short = 0 
                    next_long = int((margin_available*self.leverage) / price_per_contract)
                    next_balance = equity - (next_long/self.leverage) * price_per_contract
                    
                    long_settlement_price = current_price

                else:
                    raise ValueError("Invalid action: " +str(action))
            
            elif self.num_actions == 4:
                if action == 0:
                    # Take a full short position
                    next_short = int((margin_available*self.leverage) / price_per_contract)
                    next_long = 0
                    next_balance = equity - (next_short/self.leverage) * price_per_contract
                    
                    short_settlement_price = current_price

                elif action == 1:
                    # Take a full short position
                    next_short = int(((margin_available/2)*self.leverage) / price_per_contract)
                    next_long = 0
                    next_balance = equity - (next_short/self.leverage) * price_per_contract
                    
                    short_settlement_price = current_price
                    
                elif action == 2:
                    # Take a full long position 
                    next_short = 0 
                    next_long = int(((margin_available/2)*self.leverage) / price_per_contract)
                    next_balance = equity - (next_long/self.leverage) * price_per_contract
                    
                    long_settlement_price = current_price

                elif action == 3:
                    # Take a full long position 
                    next_short = 0 
                    next_long = int((margin_available*self.leverage) / price_per_contract)
                    next_balance = equity - (next_long/self.leverage) * price_per_contract
                    
                    long_settlement_price = current_price

                else:
                    raise ValueError("Invalid action: " +str(action))

            elif self.num_actions == 3:
                if action == 0:
                    # Take a full short position
                    next_short = int((margin_available*self.leverage) / price_per_contract)
                    next_long = 0
                    next_balance = equity - (next_short/self.leverage) * price_per_contract
                    
                    short_settlement_price = current_price

                elif action == 1:
                    # Take a flat position 
                    next_short = 0 
                    next_long = 0
                    next_balance = equity
                    
                elif action == 2:
                    # Take a full long position 
                    next_short = 0 
                    next_long = int((margin_available*self.leverage) / price_per_contract)
                    next_balance = equity - (next_long/self.leverage) * price_per_contract
                    
                    long_settlement_price = current_price

                else:
                    raise ValueError("Invalid action: " +str(action))

            elif self.num_actions == 2:
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

                else:
                    raise ValueError("Invalid action: " +str(action))

                
            next_short_value = next_short*price_per_contract
            next_long_value = next_long*price_per_contract

            prev_dist = np.array([initial_short_value, initial_long_value])
            next_dist = np.array([next_short_value, next_long_value])
            
            delta = abs(next_dist - prev_dist)
            cost = sum(delta)*self.commission
            next_balance = next_balance - cost
            next_equity = next_balance + next_short_value/self.leverage + next_long_value/self.leverage
            
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
                cost=cost,
                delta=sum(delta)
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
                cost=0,
                delta=0
            )

        extra = {
            "current_price": current_price,
            'actual_action': actual_action
        }

        self.net_worths.append(equity)
        self.account_history.append(new_account)
        self.prev_action = action
        self.margin_ratio = margin_ratio
        return new_account, extra

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
        self.current_step = self.state_buffer_size + 1
        self.trades = []
        self.prev_action = None
        self.consec_steps = 0
        self.margin_ratio = 0

        if self.randomize and self.training:
            self.leverage = random.choice(self.randomize_values['commission'])
            self.initial_balance = random.choice(self.randomize_values['commission'])
            self.commission = random.choice(self.randomize_values['commission'])
            self.execution_certainty = random.choice(self.randomize_values['execution_certainty'])
            self.balance_entropy = random.choice(self.randomize_values['balance_entropy'])

        # TODO state buffer initialization
        default_account = Account(
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
                cost=0,
                delta=0
        )

        [self.account_history.append(default_account) for _ in range(self.account_history_size)]
        [self.init_row(i) for i in range(self.state_buffer_size)]
        self.net_worths.append(self.initial_balance)

        return self.next_obs()

    def _reward(self):

        # print("current step: " +str(self.current_step))
        returns = np.diff(self.net_worths)

        if np.count_nonzero(returns) < 1:
            return 0, 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(returns, annualization=self.annualization)
        else:
            reward = returns[-1]

        # Add decay incentive against 
        # stat hold position
        consec_decay = 0
        if self.training and self.enable_hold_decay:
            if self.prev_action == 1:
                consec_decay = self.decay*(self.consec_steps**2)


        if self.training and self.enable_decay:
            if self.consec_steps > self.threshold_max_steps:
                consec_decay = self.decay * self.consec_steps
        
        reward = reward - consec_decay
        reward = reward if np.isfinite(reward) else 0

        return reward, consec_decay

    def step(self, action):
        account, extra = self._take_action(action)
        self.current_step += 1
        obs = self.next_obs()
        reward, consec_decay = self._reward()
        done = self._done()

        info = {}
        info.update(account._asdict())
        info.update(extra)
        info.update({
            'action':action,
            'consec_decay':consec_decay,
            'reward':reward
        })

        return obs, float(reward), done, info

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

# def make_env(data_provider: BaseDataProvider, rank: int = 0, seed: int = 0):
#     def _init():
#         env = TradingEnv(data_provider)
#         env.seed(seed + rank)
#         return env

#     set_global_seeds(seed)
#     return _init


if __name__ == "__main__":
    env = SwapTradingEnv()
    for x in range(400):
        obs, reward, done, _ = env.step(random.choice([0,1,2]))


    