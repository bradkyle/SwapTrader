import json 
import pyarrow as pa 
import pyarrow.parquet as pq 
import pandas as pd 
import os
import fileinput
import sys
import numpy as np
from matplotlib.pyplot import figure

os.system('gunzip ./features/*')

d = './features/'
files = os.listdir(d)

for z in files:
     with open(d+z, 'r') as original: data = original.read()
     with open(d+z, 'w') as modified: modified.write("[" + data + "{}]")

     with fileinput.FileInput(d+z, inplace=True, backup='.bak') as file:
         for line in file:
             print(line.replace('}', '},'), end='')

for z in files:
     with fileinput.FileInput(d+z, inplace=True, backup='.bak') as file:
         for line in file:
             print(line.replace('{},', '{}'), end='')

os.system('rm -rf ./features/*.bak')

recs = []
for z in files:
    with open(d+z) as f:
        recs += json.loads(f.read())


df = pd.DataFrame(recs)

df.set_index('window', inplace=True)
df.sort_index(inplace=True)

table = pa.Table.from_pandas(df)
pq.write_table(table, './data.parquet')