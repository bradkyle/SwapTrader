import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
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

# \
# or order.quantity > state.bid_qty \
# or bid_churn>0 and state.bid_qty/bid_churn > state.bid_qty/order.quantity

# if state.ask_price < order.price \
#     or order.quantity > state.ask_qty \
#     or ask_churn>0 and state.ask_qty/ask_churn > state.ask_qty/order.quantity:

if __name__ == '__main__':
    from util import *
    from constants import *
else:
    from env.util import *
    from env.constants import *

def _fill_order(self, state_history, order):
    if state_history[-1].bid_price < order.price:
        pass
    elif state_history[1:]:
        pass

def _take_action(self, action):
    action = self._derive_action(action)
    state = self._current_state()
    account = self.account_history[-1]
        
    def _maker_qty(order):
        qty = (order.quantity/self.leverage) * self.face_value/order.price
        return qty, -(qty*self.maker_fee)

    # Fill Bids
    # ----------------------------------------------------------->
    
    def fill_bid(order, fill_frac, ):
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

    # If there are open orders in the bids
    if len(self.bids) >=1:
        for i, order in enumerate(self.bids):

            if state.bid_price < order.price:
                pass # full execution with noise
            elif order.quantity > state.bid_qty:
                pass # calculate ratio of change apply to order fill with noise            
            elif [state.bid_qty for state in self.state_history]:
                pass # calculate ratio of volatility with respect to order size fill order amount therin based on this
            elif order.price >= state.ask_price:
                pass # Simulate post only order and cancel order

    # Fill Asks
    # ------------------------------------------------------------>

    def fill_ask(order, fill_frac, ):
        if order.type == orderType.CLOSE_LONG:
            qty, rebate = _maker_qty(order)
            
            account.long_position -= order.quantity
            account.balance += (qty+rebate)
            account.long_settlement_price = order.price 

        elif order.type == orderType.OPEN_SHORT:
            qty, rebate = _maker_qty(order)
            if account.balance > qty:
                account.short_position += order.quantity
                account.balance -= (qty-rebate)
                account.short_settlement_price = order.price
 

    # If there are open orders in the asks
    if len(self.asks) >=1:
        for i, order in enumerate(self.asks):
            
            if state.ask_price > order.price:
                pass # full execution with noise
            elif order.quantity > state.bid_qty:
                pass # calculate ratio of change apply to order fill with noise            
            elif [state.bid_qty for state in self.state_history]:
                pass # calculate ratio of volatility with respect to order size fill order amount therin based on this
            elif order.price >= state.ask_price:
                pass # Simulate post only order and cancel order


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

        # TODO minimum order size
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
    self.state_history.append(state)
    self.prev_action = action

    return account, extra