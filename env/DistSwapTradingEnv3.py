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
import sys

if __name__ == '__main__':
    from util import *
    from constants import *
else:
    from env.util import *
    from env.constants import *

class Account():
    def __init__(
        self,
        **kwargs
    ):
        self.equity=kwargs.get('equity',0)
        self.margin=kwargs.get('margin',0)
        self.margin_frozen=kwargs.get('margin_frozen',0)
        self.margin_ratio=kwargs.get('margin_ratio',0)
        self.realized_pnl=kwargs.get('realized_pnl',0)
        self.unrealized_pnl=kwargs.get('unrealized_pnl',0)
        self.balance=kwargs.get('balance',0)
        self.long_avail_position=kwargs.get('long_avail_position',0)
        self.long_unrealized_pnl=kwargs.get('long_unrealized_pnl',0)
        self.initial_long_value=kwargs.get('initial_long_value',0)
        self.current_long_value=kwargs.get('current_long_value',0)
        self.long_avg_cost=kwargs.get('long_avg_cost',0)
        self.long_leverage=kwargs.get('long_leverage',0)
        self.long_liquidation_price=kwargs.get('long_liquidation_price',0)
        self.long_margin=kwargs.get('long_margin',0)
        self.long_position=kwargs.get('long_position',0)
        self.long_realized_pnl=kwargs.get('long_realized_pnl',0)
        self.long_settlement_price=kwargs.get('long_settlement_price',0)
        self.short_avail_position=kwargs.get('short_avail_position',0)
        self.short_unrealized_pnl=kwargs.get('short_unrealized_pnl',0)
        self.initial_short_value=kwargs.get('initial_short_value',0)
        self.current_short_value=kwargs.get('current_short_value',0)
        self.short_avg_cost=kwargs.get('short_avg_cost',0)
        self.short_leverage=kwargs.get('short_leverage',0)
        self.short_liquidation_price=kwargs.get('short_liquidation_price',0)
        self.short_margin=kwargs.get('short_margin',0)
        self.short_position=kwargs.get('short_position',0)
        self.short_realized_pnl=kwargs.get('short_realized_pnl',0)
        self.short_settlement_price=kwargs.get('short_settlement_price',0)
        self.rate_limited=kwargs.get('rate_limited',0)
        self.cost=kwargs.get('cost',0)
        self.delta=kwargs.get('delta',0)

    def _asdict(self):
        return self.__dict__

    @property
    def fields(self):
        return list(self.__dict__.keys())


FeatureRow = collections.namedtuple(
    'FeatureRow',
    Account().fields+clean_names(FEATURES)
)

class orderType(Enum):
    OPEN_SHORT = 1
    CLOSE_SHORT = 2
    OPEN_LONG = 3 
    CLOSE_LONG = 4

    def __int__(self):
        return '%s' % self.value
        
    def __str__(self):
        return '%s' % self.name

Order = collections.namedtuple(
    'Order',
    [
        'price',
        'quantity',
        'type'
    ]
)

# TODO bid price changes then execute else don't 
# TODO implement maker and taker fees

class DistSwapTradingEnv(gym.Env):
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
        super(DistSwapTradingEnv, self).__init__()
        
        self.training = training
        self.data_file = kwargs.get('data_file', '../data/test.parquet')
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

    def _derive_action(self, action):
        return action - self.action_space.high/2

    def _take_action(self, action):
        state = self._current_state()
        account = self.account_history[-1]
        action = self._derive_action(action)

        # Fill orders if possible
        # ----------------------------------------------------------->

        def _maker_qty(order):
            qty = (order.quantity/self.leverage) * self.face_value/order.price
            return qty, -(qty*self.maker_fee)

        # TODO lock up available balance on order
        # TODO add randomization and partial fills i.e. if 
        # TODO if order.quantity > state.bid_qty order.qty = state.bid_qty
        if len(self.bids) >= 1:
            bid_churn = sum(np.diff(self.bid_price_history))
            for i, order in enumerate(self.bids):
                if True:
                        if order.type == orderType.CLOSE_SHORT:
                            qty, rebate = _maker_qty(order)
                            account.short_position -= order.quantity
                            account.balance += (qty+rebate)
                            account.short_settlement_price = order.price

                        elif order.type == orderType.OPEN_LONG:
                            qty, rebate = _maker_qty(order)
                            if account.balance > qty:
                                account.long_position += order.quantity
                                account.balance -= (qty-rebate)
                                account.long_settlement_price = order.price

                        self.bid_price_history = []
                        del self.bids[i]

                elif order.price >= state.ask_price:
                        del self.bids[i] # simulate post only by removing order
                        self.bid_price_history = [] #TODO add bid history to
                else:
                    self.bid_price_history.append(state.bid_price) 

        if len(self.asks) >= 1:
            ask_churn = sum(np.diff(self.ask_price_history))
            for i, order in enumerate(self.asks):
                if True:
                        if order.type == orderType.OPEN_SHORT:
                            qty, rebate = _maker_qty(order)
                            if account.balance > qty:
                                account.short_position += order.quantity
                                account.balance -= (qty-rebate)
                                account.short_settlement_price = order.price

                        elif order.type == orderType.CLOSE_LONG:
                            qty, rebate = _maker_qty(order)
                            
                            account.long_position -= order.quantity
                            account.balance += (qty+rebate)
                            account.long_settlement_price = order.price
                       
                        self.ask_price_history = []
                        del self.asks[i]
                        
                elif order.price <= state.bid_price:
                        del self.asks[i] # simulate post only by removing order
                        self.ask_price_history = [] #TODO add ask history
                else:
                        self.ask_price_history.append(state.ask_price)

        # Logic
        # ------------------------------------------------------------>

        #TODO change to mark price and check unrealized pnl calc
        price_per_contract = self.face_value/state.price
        
        # If a short position exists
        if account.short_position > 0: #TODO change short_sttlment price to avgExec
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            account.current_short_value = price_per_contract * account.short_position
            account.initial_short_value = (self.face_value/account.short_settlement_price) * account.short_position
            account.short_unrealized_pnl = account.current_short_value - account.initial_short_value

            # Simulate synthetic high leverage by placing a market order if 
            # short_margin_ratio > maint_margin_ratio
            short_margin_ratio = account.equity/account.current_short_value
            if short_margin_ratio < self.maint_margin_ratio and short_margin_ratio > 0:
                short_value = (self.face_value/state.bid_price) * account.short_position
                short_exit_value = short_value/self.leverage - (short_value * self.taker_fee)
                account.balance += short_exit_value 
                account.short_position = 0
        else:
            account.current_short_value = 0
            account.initial_short_value = 0
            account.short_unrealized_pnl = 0

        # If a long position exists
        if account.long_position > 0:
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            account.current_long_value = price_per_contract * account.long_position
            account.initial_long_value = (self.face_value/account.long_settlement_price) * account.long_position    
            account.long_unrealized_pnl = account.initial_long_value - account.current_long_value

            # Simulate synthetic high leverage by placing a market order if 
            # long_margin_ratio > maint_margin_ratio
            long_margin_ratio = account.equity/account.current_long_value
            if long_margin_ratio < self.maint_margin_ratio and long_margin_ratio > 0:
                long_value = (self.face_value/state.ask_price) * account.long_position
                long_exit_value = long_value/self.leverage - (long_value * self.taker_fee)
                account.balance += long_exit_value
                account.long_position = 0
        else:
            account.current_long_value = 0
            account.initial_long_value = 0
            account.long_unrealized_pnl = 0

        account.unrealized_pnl = account.short_unrealized_pnl + account.long_unrealized_pnl
        current_position_value = account.current_short_value + account.current_long_value
        account.equity = account.balance + (account.initial_long_value + account.initial_short_value + account.unrealized_pnl)/self.leverage
        
        #TODO check
        margin_available = account.equity - (account.equity*self.maint_margin_ratio)

        # Place orders 
        # ------------------------------------------------------------>
        long_delta = 0
        short_delta = 0

        if action != self.prev_action:

            # Close open orders
            self.asks = []
            self.bids = []

            if action > 0:
                long_dist = margin_available * abs(action)
                short_dist = 0
            elif action <0:
                short_dist = margin_available * abs(action)
                long_dist = 0
            elif action == 0:
                short_dist = 0
                long_dist = 0

            next_short = int((short_dist*self.leverage) / price_per_contract) #TODO change
            next_long = int((long_dist*self.leverage) / price_per_contract)
            
            short_delta = next_short - account.short_position
            long_delta = next_long - account.long_position

            #TODO check order types
            if short_delta > 0:
                # open_short ask price
                self.asks.append(Order(
                    price=state.ask_price,
                    quantity=abs(short_delta),
                    type=orderType.OPEN_SHORT
                ))
                
            elif short_delta < 0:
                # close_short bid price
                self.bids.append(Order(
                    price=state.bid_price,
                    quantity=abs(short_delta),
                    type=orderType.CLOSE_SHORT
                ))
            
            if long_delta > 0:
                # open long ask price
                self.bids.append(Order(
                    price=state.bid_price,
                    quantity=abs(long_delta),
                    type=orderType.OPEN_LONG
                ))

            elif long_delta < 0:
                # open short bid price
                self.asks.append(Order(
                    price=state.ask_price,
                    quantity=abs(long_delta),
                    type=orderType.CLOSE_LONG
                )) 

        extra = {
            "current_price": state.price,
            "actual_action": action,
            "long_delta": long_delta,
            "short_delta": short_delta
        }

        self.net_worths.append(account.equity)
        self.account_history.append(account)
        self.prev_action = action

        return account, extra


    def _done(self):
        if  self.current_step == len(self.obs_df) - 1:
            return True
        if self.net_worths[-1] < self.initial_balance / 10:
            return True
        
        
    def reset(self):
        self.state_buffer.clear()
        self.net_worths.clear()
        self.current_step = self.state_buffer_size + 1
        self.trades = []
        self.prev_action = None
        self.consec_steps = 0
        self.margin_ratio = 0
        self.asks = []
        self.bids = []
        self.ask_price_history = []
        self.bid_price_history = []

        if self.randomize and self.training:
            self.leverage = random.choice(self.randomize_values['commission'])
            self.initial_balance = random.choice(self.randomize_values['commission'])
            self.commission = random.choice(self.randomize_values['commission'])
            self.execution_certainty = random.choice(self.randomize_values['execution_certainty'])
            self.balance_entropy = random.choice(self.randomize_values['balance_entropy'])

        # TODO state buffer initialization
        default_account = Account(
                equity=self.initial_balance,
                balance=self.initial_balance,
                long_leverage=self.leverage,
                short_leverage=self.leverage
        )

        [self.account_history.append(default_account) for _ in range(self.account_history_size)]
        [self.init_row(i) for i in range(self.state_buffer_size)]
        self.net_worths.append(self.initial_balance)

        return self.next_obs()

    def _reward(self):

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


    
