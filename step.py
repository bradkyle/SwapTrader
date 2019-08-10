import requests
import pandas as pd 
import pyarrow.parquet as pq
import pyarrow as pa

import okex.account_api as account
import okex.ett_api as ett
import okex.futures_api as future
import okex.lever_api as lever
import okex.spot_api as spot
import okex.swap_api as swap
import json
import time
from enum import Enum

class orderType(Enum):
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4

    def __str__(self):
        return '%s' % self.value

api_key = '8888175d-d023-44ab-b096-330d8a61a317'
secret_key = '7541C30C7D10CEB44C4F4D03A82C421F'
passphrase = "Ga1!4FQC^u"
# available = 2

action = 0.5
long_action = action 
short_action = 1-action

swapAPI = swap.SwapAPI(api_key, secret_key, passphrase, True)

"""
{'info': {'equity': '0.1630',
  'fixed_balance': '0.0000',
  'instrument_id': 'BTC-USD-SWAP',
  'maint_margin_ratio': '0.0050',
  'margin': '0.0175',
  'margin_frozen': '0.0000',
  'margin_mode': 'crossed',
  'margin_ratio': '9.3069',
  'realized_pnl': '-0.0001',
  'timestamp': '2019-08-10T12:30:07.865Z',
  'total_avail_balance': '0.1631',
  'unrealized_pnl': '0.0000'}}
"""
account = swapAPI.get_coin_account('BTC-USD-SWAP')['info']

available = float(account['equity'])

# TODO transform account into available

# Position
def get_position():
    position = swapAPI.get_position()

    def get_avail_position(side, inst):
        if len(position)>0:
            pos = [d for d in position[0]['holding'] if d['side'] == side and d['instrument_id'] == inst]
            if len(pos) > 0 :
                return float(pos[0]['avail_position'])
            else:
                return 0
        else:
            return 0

    long_position = get_avail_position('long', 'BTC-USD-SWAP')
    short_position = get_avail_position('short', 'BTC-USD-SWAP')

    return (long_position, short_position)

position = get_position()

print(position)

next_short = available * short_action
next_long = available * long_action

short_delta = next_short - position[1]
long_delta = next_long - position[0]

depth = swapAPI.get_depth('BTC-USD-SWAP', '5')
orders = []
current_time = int(time.time()*1000)

depth = swapAPI.get_depth('BTC-USD-SWAP', '5')
ask_price = float(depth['asks'][0][0])
bid_price = float(depth['bids'][0][0])

print('---------------------: asks')
print([d[0] for d in depth['asks']])
print('---------------------: bids')
print([d[0] for d in depth['bids']])

# Close orders that aren't at best price
orders = swapAPI.get_order_list('4', 'BTC-USD-SWAP', '', '', '')['order_info']
close_order_ids = [o['order_id'] for o in orders if o['price'] != best_price]
if len(close_order_ids) > 0:
    result = swapAPI.revoke_orders(close_order_ids, 'BTC-USD-SWAP')

price_per_contract = 100/bid_price

if short_delta > 0:
    # Open short
    orders.append({
        "client_oid": "openshort"+str(current_time),
        "price": str(ask_price),
        "size": str(int(short_delta)),
        "type": str(orderType.OPEN_SHORT),
        "match_price": "0"
    })
elif long_delta < 0:
    # Close Short
    orders.append({
        "client_oid": "closeshort"+str(current_time),
        "price": str(bid_price),
        "size": str(int(short_delta)),
        "type": str(orderType.CLOSE_SHORT),
        "match_price": "0"
    })

if long_delta > 0:
    # Open Long
    orders.append({
        "client_oid": "openlong"+str(current_time),
        "price": str(bid_price),
        "size": str(int(long_delta)),
        "type": str(orderType.OPEN_LONG),
        "match_price": "0"
    })
elif long_delta < 0:
    # Close Long
    orders.append({
        "client_oid": "closelong"+str(current_time),
        "price": str(ask_price),
        "size": str(int(long_delta)),
        "type": str(orderType.CLOSE_LONG),
        "match_price": "0"
    })

    
print(orders)
print('---------------------')
# result = swapAPI.take_orders(
#     orders, 
#     'BTC-USD-SWAP'
# )
# print(result)