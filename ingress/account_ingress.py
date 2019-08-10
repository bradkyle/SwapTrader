import sys
import json
import pandas as pd
import argparse
import os
import time
import rethinkdb as rdb
import logging
import multiprocessing
from lomond.persist import persist
from lomond import WebSocket
import json
from datetime import datetime
from util import *
from constants import *

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

r = rdb.RethinkDB()

class Ingress():
    def __init__(
        self,
        config_file,
        host='rethinkdb',
        port=28015,
        dbname='axiom',
        table='features',
    ):
        self.dbname=dbname
        self.host=host 
        self.port=port
        self.table=table
        self.conf_file = config_file
        self.logged_in = False

        self.conn = r.connect(self.host, self.port)
        if not r.db_list().contains(self.dbname).run(self.conn):
            r.db_create(self.dbname).run(self.conn)

        if not r.db(self.dbname).table_list().contains('account').run(self.conn):
            r.db(self.dbname).table_create('account').run(self.conn)
            r.db(self.dbname).table('account').index_create('time').run(self.conn)

        if not r.db(self.dbname).table_list().contains('order').run(self.conn):
            r.db(self.dbname).table_create('order').run(self.conn)
            r.db(self.dbname).table('order').index_create('time').run(self.conn)

        if not r.db(self.dbname).table_list().contains('position').run(self.conn):
            r.db(self.dbname).table_create('position').run(self.conn)
            r.db(self.dbname).table('position').index_create('time').run(self.conn)

        self.websocket = WebSocket('wss://real.okex.com:10442/ws/v3')
        self.queue = multiprocessing.Queue()

        if os.path.exists(self.conf_file):
            with open(self.conf_file, 'r') as f:
                data = json.load(f)
                self.api_key = data['api_key']
                self.api_secret = data['api_secret']
                self.passphrase = data['passphrase']
        else:
            raise FileNotFoundError("Okex key file not found")

    def _persist(self):
        while True:
            table, msg, time = self.queue.get()
            try:
                msg.update({'time': time})
                res = r.db(self.dbname).table(table).insert([msg], conflict="replace").run(self.conn)
                logging.info(res)
            except Exception as e:
                logging.error(e)
                logging.error(msg)
                self.conn = r.connect(self.host, self.port)

    def publish(self, msg, table, time):
        self.queue.put((table, msg, time))
        
    def _run(self):
        for event in persist(self.websocket, ping_rate=30):
                if event.name == 'poll':
                    if not self.logged_in:
                        login_str = login_params(str(str(server_timestamp())), self.api_key, self.passphrase, self.api_secret)
                        self.websocket.send_text(login_str)
                    else:
                        sub_param = {"op": "subscribe", "args": [
                            "swap/position:BTC-USD-SWAP",
                            "swap/account:BTC-USD-SWAP",
                            "swap/order:BTC-USD-SWAP"
                        ]}
                        sub_str = json.dumps(sub_param)
                        self.websocket.send_text(sub_str)
                elif event.name == 'binary':
                    try:
                        res = json.loads(inflate(event.data))
                        if "table" in res:
                            if res["table"] == "swap/position":
                                self.publish(res['data'][0], "position", self.parse_dt(res['data'][0]["timestamp"]))
                            elif res["table"] == "swap/account":
                                self.publish(res['data'][0], "account", self.parse_dt(res['data'][0]['timestamp']))
                            elif res["table"] == "swap/order":
                                self.publish(res['data'][0], "order", self.parse_dt(res['data'][0]['timestamp']))
                        elif "event" in res:
                            if res["event"] == "login":
                                self.logged_in=True
                                logging.info("Successfully logged in!")
                            elif res['event'] == "subscribe":
                                pass
                            elif res['event'] == 'error':
                                if res['errorCode'] == 30041:
                                    self.logged_in = False
                                    logging.info("Logged out")
                            else:
                                logging.warn(res)
                        else:
                            logging.warn(res)
                    except Exception as e:
                        logging.error(e)
                elif event.name == "text":
                        logging.warn(event.text)

    def parse_dt(self, time):
        return r.epoch_time(datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())

    def run(self):
        run_process = multiprocessing.Process(target=self._run) 
        run_process.daemon = True
        run_process.start() 

        publish_process = multiprocessing.Process(target=self._persist)
        publish_process.daemon = True
        publish_process.start()

        run_process.join() 
        publish_process.join()

    def insert(self, records):
            res = r.db(self.dbname).table(self.table).insert(records, conflict="replace").run(self.conn)
            logging.info(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a pubsub subscription')

    parser.add_argument(
        '-ho', 
        '--host', 
        default=os.getenv('HOST'), 
        type=str, 
        help='Rethinkdb host to connect to'
    )

    parser.add_argument(
        '-c', 
        '--conf', 
        default=os.getenv('OKEX_CONFIG_FILE'), 
        type=str, 
        help='Path to config file for okex'
    )

    args = parser.parse_args()

    runner = Ingress(
        host=args.host,
        config_file=args.conf
    )
    runner.run()