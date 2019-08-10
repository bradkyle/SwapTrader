from ingress import Ingress
import pyarrow as pa 
import pyarrow.parquet as pq 
import pandas as pd 
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a pubsub subscription')
    parser.add_argument(
        '-pid', 
        '--project_id', 
        default=os.getenv('PROJECT_ID'), 
        type=str, 
        help='Name of the google cloud project'
    )
    parser.add_argument(
        '-t', 
        '--topic', 
        default=os.getenv('TOPIC'), 
        type=str, 
        help='Google cloud pubsub topic name'
    )
    parser.add_argument(
        '-ho', 
        '--host', 
        default=os.getenv('HOST'), 
        type=str, 
        help='Rethinkdb host to connect to'
    )

    args = parser.parse_args()

    df = pq.read_table('./data.parquet').to_pandas()

    df.sort_index(inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.reset_index(inplace=True)
    df.dropna(subset=['window'], inplace=True)

    records = df.to_dict('records')

    runner = Ingress(
        topic=args.topic, 
        pid=args.project_id,
        host=args.host
    )
    runner.insert(records)