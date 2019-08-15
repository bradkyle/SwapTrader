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
import json
import random
import collections

if __name__ == '__main__':
    from util import *
    from constants import *
    from Position import *
else:
    from env.util import *
    from env.constants import *
    from env.Position import *

FeatureRow = collections.namedtuple(
    'FeatureRow',
    clean_names(FEATURES)
)

class MarketMakingEnv():
    def __init__(
        self, 
        training=False,
        *args, 
        **kwargs
    ):
        super(MarketMakingEnv, self).__init__()

        self.training = training
        self.data_file = kwargs.get('data_file', '../data/NODIFF_TEST.parquet')
        self.leverage = kwargs.get('leverage', 30)
        self.initial_balance = kwargs.get('initial_balance', 10)
        self.reward_func = kwargs.get('reward_func', 'B')
        self.taker_fee = kwargs.get('taker_fee', 0.00075)
        self.maker_fee = kwargs.get('maker_fee', -0.00025)
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

        self.observation_space = spaces.Box(
            low=0, 
            high=self.scaler_high, 
            dtype=np.uint8, 
            shape=self.obs_shape
        )
    
        # Account history and net worth dequeue's
        self.state_buffer = collections.deque(maxlen=self.state_buffer_size)
        self.net_worths = collections.deque(maxlen=self.evaluation_history_size)
        self.account_history = collections.deque(maxlen=self.account_history_size)

        # Action space size is 2: one for full long and one for full short
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.ind_df, self.obs_df = self._load()

        self.scaler = preprocessing.MinMaxScaler(feature_range=(0,self.scaler_high))
        # self.scaler.fit(self.obs_df)

        if self.leverage >= 99:
            raise ValueError("Can't have leverage greater than 99X")

        self.reset()
    
    def _load(self):
        df = pq.read_table(self.data_file).to_pandas()
        ind = df[STATE_FIELDS.keys()]
        obs = df[FEATURES]
        obs.columns = clean_names(obs.columns)
        ind.rename(columns=STATE_FIELDS, inplace=True)
        return ind, obs

    def _current_state(self):
        return self.ind_df[STATE_FIELDS.values()].iloc[self.current_step]
    
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

    def _derive_action(self, action):
        return action - self.action_space.high/2

    # TODO random funding rate + make testable
    def _take_action(self, action):
        state = self._current_state()
        action = self._derive_action(action)

        # The order price is taken to be limit seen as though market
        # orders will execute immediately, post only orders
        # Limit Order processing #TODO if order > quant + level_delta>level_size
        if len(self.bids) >= 1:
            for order in self.bids:
                if state.bid_price < order.price \
                    or order.quantity > state.bid_qty \
                    or state.bid_qty/sum(np.diff(self.bid_price_history)) > state.bid_qty/order.quantity:
                        self.position.buy(order, self.maker_fee)
                        self.bid_price_history = []
                elif order.price >= state.ask_price:
                        self.bids = [] # simulate post only
                        self.bid_price_history = []

        if len(self.asks) >= 1:
            for order in self.asks:
                if state.ask_price < order.price \
                    or order.quantity > state.ask_qty \
                    or state.ask_qty/sum(np.diff(self.ask_price_history)) > state.ask_qty/order.quantity:
                        self.position.sell(order, self.maker_fee)
                        self.ask_price_history = []
                elif order.price <= state.bid_price:
                        self.asks = [] # simulate post only
                        self.ask_price_history = []


        # Simultate a synthetic high leverage by using stop loss that places market order
        # instead of liquidating
        # Place a market/taker order crossing the bid ask spread
        if self.position.is_liquidated and self.position.is_short:
            self.position.liquidate(price=state.bid_price, fee=self.taker_fee)

        if self.position.is_liquidated and self.position.is_long:
            self.position.liquidate(price=state.ask_price, fee=self.taker_fee)

        # Position Value = Face Value X Number of swaps/Latest Mark Price
        current_value = state.mark_price * self.currentQty
        initial_value = self.avgEntryPrice * self.currentQty
        unrealized_pnl = current_value + initial_value

        if action != self.prev_action:

            # cancel orders
            self.orders = []

            if action > 0:
                next_long = abs(action) * self.position.available
                next_short = 0 
            elif action < 0:
                next_long = 0
                next_short = abs(action) * self.position.available
            elif action == 0:
                next_short = 0
                next_long = 0

            short_delta = next_short - self.position.current_short
            long_delta = next_long - self.position.current_long

            # TODO make sure > threshold order contract size
            # Create Limit orders for maker rebates
            if short_delta > 0:
                # open_short ask price
                self.asks.append(Order(
                    price=state.ask_price,
                    size=abs(int(short_delta/state.ask_price)),
                    side=side.SELL
                ))
                
            elif short_delta < 0:
                # close_short bid price
                self.bids.append(Order(
                    price=state.bid_price,
                    size=abs(int(short_delta/state.bid_price)),
                    side=side.BUY
                ))
            
            if long_delta > 0:
                # open long ask price
                self.bids.append(Order(
                    price=state.bid_price,
                    size=abs(int(long_delta/state.bid_price)),
                    side=side.BUY
                ))

            elif long_delta < 0:
                # open short bid price
                self.asks.append(Order(
                    price=state.ask_price,
                    size=abs(int(long_delta/state.ask_price)),
                    side=side.SELL
                ))      

        self.net_worths.append(self.position.equity)
        self.position_history.append(self.position.simple)

    def reset(self):
        self.current_step = 0

        self.short_position = None
        self.long_position = None
        self.balance = None

        self.bids = []
        self.asks = []
        self.bid_hitory = []
        self.ask_history = []

        self.position = Position(
            account=self.initial_balance,
            leverage=self.leverage,
            
        )
    
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



if __name__ == "__main__":
    env = MarketMakingEnv()
    for x in range(400):
        obs, reward, done, _ = env.step(random.choice([0,1,2]))



    