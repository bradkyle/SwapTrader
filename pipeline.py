import faust 
import pandas as pd 
import pyarrow as pa 
import numpy as np
import logging 
import json 
import rethinkdb as rdb
from lomond.persist import persist
from lomond import WebSocket
import zlib
import requests
import dateutil.parser as dp
import hmac
import base64
import multiprocessing
from env.DistSwapTradingEnv import *

r = rdb.RethinkDB()


class orderType(Enum):
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4

    def __str__(self):
        return '%s' % self.value

class Runner():
    def __init__(
        self
    ):
        
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

        self.conn = r.connect(self.host, self.port)
        self.queue = multiprocessing.Queue()


    def _exec(features): 

        features = FeatureRow(**features)

        self.state_buffer.append(features)

        df = pd.DataFrame(self.state_buffer)
        df.drop_duplicates(subset=TIME, keep="last",inplace=True)
        df.set_index(TIME, inplace=True)
        df.sort_index(inplace=True, ascending=True)
        clean(df)
        df = add_candle_indicators(
            df, 
            'ind_', 
            'OkexSensor_swap/candle60s:BTC-USD-SWAP_close',
            'OkexSensor_swap/candle60s:BTC-USD-SWAP_high',
            'OkexSensor_swap/candle60s:BTC-USD-SWAP_low',
            'OkexSensor_swap/candle60s:BTC-USD-SWAP_volume'
        )

        obs.columns = clean_names(obs.columns)

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))

        obs = self.scaler.fit_transform(obs)[-1]    
        obs = np.reshape(obs.astype('float16'), self.obs_shape)

        obs[obs==-np.inf] = 0
        obs[obs==np.nan] = 0
        obs[obs==np.inf] = 0

        if not np.isfinite(obs.any()):
            raise ValueError("Non finite values in: " + str(obs))
        
        # Get prediction from model
        action = self.model.predict([obs])

        if action != self.prev_action:

            short_action = 1-action 
            long_action = action

            next_short = available_balance * short_action 
            next_long = available_balance * long_action

            short_delta = next_short - current_short 
            long_delta = next_long - current_long

            orders = []

            if short_delta > 0:
                # Open short
                orders.append({
                    "client_oid": "open_short_",
                    "price": "5",
                    "size": "2",
                    "type": str(orderType.OPEN_SHORT),
                    "match_price": "1"
                })
            else:
                # Close Short
                orders.append({
                    "client_oid": "close_short_",
                    "price": "5",
                    "size": "2",
                    "type": str(orderType.CLOSE_SHORT),
                    "match_price": "1"
                })

            if long_delta > 0:
                # Open Long
                orders.append({
                    "client_oid": "open_long_",
                    "price": "5",
                    "size": "2",
                    "type": str(orderType.OPEN_LONG),
                    "match_price": "1"
                })
            else:
                # Close Long
                orders.append({
                    "client_oid": "close_long_",
                    "price": "5",
                    "size": "2",
                    "type": str(orderType.CLOSE_LONG),
                    "match_price": "1"
                })

            result = swapAPI.take_orders(
                orders, 
                self.instrument_id
            )



    def subscribe(self, callback):
        if self._ready:
            feed = r.db(self.dbname).table(self.table).changes().run(self.conn)
            for change in feed:
                callback(self.get())
        else:
            logging.warn("Buffer not ready: "+ str(self._length))

    
    def run(self):
        run_process = multiprocessing.Process(target=self._run) 
        run_process.daemon = True
        run_process.start() 

        publish_process = multiprocessing.Process(target=self._persist)
        publish_process.daemon = True
        publish_process.start()

        run_process.join() 
        publish_process.join()
