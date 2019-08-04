import sys
import json
import pandas as pd
from env.constants import *
import argparse
import os
import time
from env.util import *
import rethinkdb as rdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import logging
import pyarrow.parquet as pq
import pyarrow as pa

r = rdb.RethinkDB()

num = 88000
cutoff = 1500
host='0.0.0.0'
test_portion = 0.1
port=28015
dbname='axiom'
table='features'

conn = r.connect(host, port).repl()
cursor = r.db(dbname).table(table).order_by(index=r.desc(TIME)).limit(num+cutoff+10).run(conn)

df = pd.DataFrame(cursor)
df.drop_duplicates(subset=TIME, keep="last",inplace=True)
df.set_index(TIME, inplace=True)
df.sort_index(inplace=True, ascending=True)
clean(df)

indf = df[[PRICE]]

indf.rename(
    columns={
        PRICE: 'price'
    }, 
    inplace=True
)

print (df.head())

df = add_candle_indicators(
    df, 
    'ind_', 
    'OkexSensor_swap/candle60s:BTC-USD-SWAP_close',
    'OkexSensor_swap/candle60s:BTC-USD-SWAP_high',
    'OkexSensor_swap/candle60s:BTC-USD-SWAP_low',
    'OkexSensor_swap/candle60s:BTC-USD-SWAP_volume'
)
clean(df)
df = df[FEATURES]

# df = apply_fracdiff(df)
# clean(df)

df = df.merge(
        indf, 
        how='outer', 
        left_index=True, 
        right_index=True
)

ldf = int(len(df) * 0.8)

table = pa.Table.from_pandas(df.iloc[:ldf])
pq.write_table(table, './data/DATA.parquet')

table = pa.Table.from_pandas(df.iloc[ldf:])
pq.write_table(table, './data/TEST.parquet')
