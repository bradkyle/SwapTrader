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
else:
    from env.util import *
    from env.constants import *

class State():
    pass

FeatureRow = collections.namedtuple(
    'FeatureRow',
    clean_names(FEATURES)
)

Price = collections.namedtuple(
    'Price',
    [
        'bid_qty',
        'bid_price',
        'last',
        'ask_price',
        'ask_qty'
    ]
)

Order = collections.namedtuple(
    'Order',
    [
        'price',
        'quantity'
    ]
)

class orderType(Enum):
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4

    def __str__(self):
        return '%s' % self.value

class Account():

    def __init__(
        self,
    ):
        
        self.account = 0
        self.avgEntryPrice = 0
        self.commission = 0
        self.leverage = 0
        self.liquidationPrice = 0
        self.lastPrice = 0
        
        [
            {
                "account": 235781,
                "avgCostPrice": 11405.1095,
                "avgEntryPrice": 11405.1095,
                "bankruptPrice": 13592,
                "breakEvenPrice": 11219.5,
                "commission": 0.00075,
                "crossMargin": true,
                "currency": "XBt",
                "currentComm": 18957,
                "currentCost": 3546131,
                "currentQty": -400,
                "currentTimestamp": "2019-08-12T21:37:30.096Z",
                "deleveragePercentile": 1,
                "execBuyCost": 0,
                "execBuyQty": 0,
                "execComm": 8530,
                "execCost": 11389191,
                "execQty": -1299,
                "execSellCost": 11389191,
                "execSellQty": 1299,
                "foreignNotional": 400,
                "grossExecCost": 3507065,
                "grossOpenCost": 33405958,
                "grossOpenPremium": 0,
                "homeNotional": -0.035096,
                "indicativeTax": 0,
                "indicativeTaxRate": 0,
                "initMargin": 384420,
                "initMarginReq": 0.01,
                "isOpen": true,
                "lastPrice": 11397.7,
                "lastValue": 3509600,
                "leverage": 100,
                "liquidationPrice": 13511.5,
                "longBankrupt": 0,
                "maintMargin": 40264,
                "maintMarginReq": 0.005,
                "marginCallPrice": 13511.5,
                "markPrice": 11397.7,
                "markValue": 3509600,
                "openOrderBuyCost": -18726100,
                "openOrderBuyPremium": 0,
                "openOrderBuyQty": 2100,
                "openOrderSellCost": -33010868,
                "openOrderSellPremium": 0,
                "openOrderSellQty": 3801,
                "openingComm": 10427,
                "openingCost": -7843060,
                "openingQty": 899,
                "openingTimestamp": "2019-08-12T21:00:00.000Z",
                "posAllowance": 0,
                "posComm": 2657,
                "posCost": 3507064,
                "posCost2": 3507064,
                "posCross": 0,
                "posInit": 35071,
                "posLoss": 0,
                "posMaint": 20193,
                "posMargin": 37728,
                "posState": "",
                "prevClosePrice": 11421.36,
                "prevRealisedPnl": 0,
                "prevUnrealisedPnl": 0,
                "quoteCurrency": "USD",
                "realisedCost": 39067,
                "realisedGrossPnl": -39067,
                "realisedPnl": -58024,
                "realisedTax": 0,
                "rebalancedPnl": 0,
                "riskLimit": 20000000000,
                "riskValue": 36915558,
                "sessionMargin": 0,
                "shortBankrupt": 0,
                "simpleCost": null,
                "simplePnl": null,
                "simplePnlPcnt": null,
                "simpleQty": null,
                "simpleValue": null,
                "symbol": "XBTUSD",
                "targetExcessMargin": 0,
                "taxBase": 0,
                "taxableMargin": 0,
                "timestamp": "2019-08-12T21:37:31.892Z",
                "underlying": "XBT",
                "unrealisedCost": 3507064,
                "unrealisedGrossPnl": 2536,
                "unrealisedPnl": 2536,
                "unrealisedPnlPcnt": 0.0007,
                "unrealisedRoePcnt": 0.0723,
                "unrealisedTax": 0,
                "varMargin": 0
            }
        ]
        
        {
        "account": 235781,
        "avgCostPrice": 11434,
        "avgEntryPrice": 11434,
        "bankruptPrice": 10440,
        "breakEvenPrice": 11478.5,
        "commission": 0.00075,
        "crossMargin": true,
        "currency": "XBt",
        "currentComm": 10429,
        "currentCost": -7834310,
        "currentQty": 898,
        "currentTimestamp": "2019-08-12T20:35:40.082Z",
        "deleveragePercentile": 1,
        "execBuyCost": 28896297,
        "execBuyQty": 3304,
        "execComm": 17677,
        "execCost": -28012946,
        "execQty": 3203,
        "execSellCost": 883351,
        "execSellQty": 101,
        "foreignNotional": -898,
        "grossExecCost": 7853776,
        "grossOpenCost": 18702850,
        "grossOpenPremium": 0,
        "homeNotional": 0.07858398,
        "indicativeTax": 0,
        "indicativeTaxRate": 0,
        "initMargin": 215224,
        "initMarginReq": 0.01,
        "isOpen": true,
        "lastPrice": 11426.99,
        "lastValue": -7858398,
        "leverage": 100,
        "liquidationPrice": 10489,
        "longBankrupt": 0,
        "maintMargin": 84494,
        "maintMarginReq": 0.005,
        "marginCallPrice": 10489,
        "markPrice": 11426.99,
        "markValue": -7858398,
        "openOrderBuyCost": -18702850,
        "openOrderBuyPremium": 0,
        "openOrderBuyQty": 2101,
        "openOrderSellCost": -17189000,
        "openOrderSellPremium": 0,
        "openOrderSellQty": 2000,
        "openingComm": -7248,
        "openingCost": 20178636,
        "openingQty": -2305,
        "openingTimestamp": "2019-08-12T20:00:00.000Z",
        "posAllowance": 0,
        "posComm": 5954,
        "posCost": -7853908,
        "posCost2": -7853908,
        "posCross": 4490,
        "posInit": 78540,
        "posLoss": 0,
        "posMaint": 46009,
        "posMargin": 88984,
        "posState": "",
        "prevClosePrice": 11433.77,
        "prevRealisedPnl": 0,
        "prevUnrealisedPnl": 0,
        "quoteCurrency": "USD",
        "realisedCost": 19598,
        "realisedGrossPnl": -19598,
        "realisedPnl": -30027,
        "realisedTax": 0,
        "rebalancedPnl": 0,
        "riskLimit": 20000000000,
        "riskValue": 26561248,
        "sessionMargin": 0,
        "shortBankrupt": 0,
        "simpleCost": null,
        "simplePnl": null,
        "simplePnlPcnt": null,
        "simpleQty": null,
        "simpleValue": null,
        "symbol": "XBTUSD",
        "targetExcessMargin": 0,
        "taxBase": 0,
        "taxableMargin": 0,
        "timestamp": "2019-08-12T20:35:40.082Z",
        "underlying": "XBT",
        "unrealisedCost": -7853908,
        "unrealisedGrossPnl": -4490,
        "unrealisedPnl": -4490,
        "unrealisedPnlPcnt": -0.0006,
        "unrealisedRoePcnt": -0.0572,
        "unrealisedTax": 0,
        "varMargin": 0
        }
        # Cross Liquidation Price =1/(1/Entry Price+Minimum Margin Balance/Position Size)

        # Minimum Margin Balance = Account Balance *(1- (Maintenance Margin + Taker Fees + Funding Rate))

    def update_long(self):
        pass

    def update_short(self):
        pass 

    def update_balance(self):
        pass
    


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

        self.observation_space = spaces.Box(low=0, high=self.scaler_high, dtype=np.uint8, shape=self.obs_shape)
    
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

    # TODO random funding rate + make testable
    def _take_action(self, action):
        state = self._current_state()

        def _maker(order):
            return order.quantity - order.quantity*self.maker_fee

        def _taker(order):
            return order.quantity - order.quantity*self.taker_fee

        # Order processing #TODO if order > quant + level_delta>level_size
        if len(self.orders) >= 1:
            for order in self.orders:
                if state.bid_price < order.price:
                    if order.type == orderType.OPEN_SHORT: 
                        self.account.long_position += _maker(order)
                    elif order.type == orderType.CLOSE_SHORT:
                        self.account.short_position -= order.quantity
                    elif order.type == orderType.OPEN_LONG:
                        self.account.short_position -= order.quantity
                    elif order.type == orderType.CLOSE_LONG:
                        self.account.short_position -= order.quantity

                if state.ask_price > order.price: #or quantity > order.qquantity
                    if order.type == orderType.OPEN_SHORT: 
                        self.account.long_position += _maker(order)
                    elif order.type == orderType.CLOSE_SHORT:
                        self.account.short_position -= order.quantity
                    elif order.type == orderType.OPEN_LONG:
                        self.account.short_position -= order.quantity
                    elif order.type == orderType.CLOSE_LONG:
                        self.account.short_position -= order.quantity
        

        # Simultate a synthetic high leverage by using stop loss that places market order
        # instead of liquidating
        # Place a market/taker order crossing the bid ask spread
        if state.price >= self.account.short_liquidation_price:
            short_value = self.account.short_position*state.ask_price
            self.account.balance +=  short_value - short_value * self.taker_fee
            self.account.short_position = 0

        if state.price <= self.account.long_liquidation_price:
            long_value = self.account.short_position*state.ask_price
            self.account.balance +=  long_value - long_value * self.taker_fee
            self.account.long_position = 0

        # If a short position exists
        if self.account.short_position > 0:
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            current_short_value = state.price * self.account.short_position
            initial_short_value = self.account.short_avg_entry_price * self.account.short_position
            short_unrealized_pnl = current_short_value - initial_short_value

        # If a long position exists
        if self.account.long_position > 0:
            # Position Value = Face Value X Number of swaps/Latest Mark Price
            current_long_value = state.price * self.account.long_position    
            initial_long_value = self.account.long_avg_entry_price  * self.account.long_position    
            long_unrealized_pnl = initial_long_value - current_long_value


        unrealized_pnl = short_unrealized_pnl + long_unrealized_pnl
        current_position_value = current_short_value + current_long_value
        equity = self.account.balance + (initial_long_value + initial_short_value + unrealized_pnl)
        
        available = equity - (equity*self.maint_margin_ratio)

        if action != self.prev_action:

            # cancel orders
            self.orders = []
            
            long_action = action
            short_action = 1-action

            next_short = (available * short_action)*self.leverage
            next_long = (available * long_action)*self.leverage

            short_delta = next_short - current_short_value
            long_delta = next_long - current_long_value

            # TODO make sure > threshold order contract size
            # Create Limit orders for maker rebates
            if short_delta > 0:
                # open_short ask price
                self.orders.append(Order(
                    price=state.ask_price,
                    size=abs(int(short_delta/state.ask_price)),
                    type=orderType.OPEN_SHORT
                ))
                
            elif short_delta < 0:
                # close_short bid price
                self.orders.append(Order(
                    price=state.bid_price,
                    size=abs(int(short_delta/state.bid_price)),
                    type=orderType.CLOSE_SHORT
                ))
            
            if long_delta > 0:
                # open long ask price
                self.orders.append(Order(
                    price=state.bid_price,
                    size=abs(int(long_delta/state.bid_price)),
                    type=orderType.OPEN_LONG
                ))

            elif long_delta < 0:
                # open short bid price
                self.orders.append(Order(
                    price=state.ask_price,
                    size=abs(int(long_delta/state.ask_price)),
                    type=orderType.CLOSE_LONG
                ))
            else:
                pass


        self.net_worths.append(equity)
        self.account_history.append(self.account.to_list())

    def reset(self):
        self.current_step = 0

        self.short_position = None
        self.long_position = None
        self.balance = None

        self.orders = []

        self.account = Account(
            equity=0,
            margin=0,
            margin_frozen=0,
            margin_ratio=0,
            short_position=0,
            long_position=0
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



    