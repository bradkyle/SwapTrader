import sys
import json
from subscriber import Subscriber
import pandas as pd
from constants import *
import argparse
import os
import time
from util import *
import rethinkdb as rdb
import logging
import multiprocessing

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

r = rdb.RethinkDB()

class Ingress():
    def __init__(
        self,
        topic,
        pid,
        host='rethinkdb',
        port=28015,
        dbname='axiom',
        table='features'
    ):
        self.topic = topic
        self.pid = pid
        self.dbname=dbname
        self.host=host 
        self.port=port
        self.table=table

        self.conn = r.connect(self.host, self.port)
        if not r.db_list().contains(self.dbname).run(self.conn):
            r.db_create(self.dbname).run(self.conn)
        if not r.db(self.dbname).table_list().contains(self.table).run(self.conn):
            r.db(self.dbname).table_create(self.table, primary_key=TIME).run(self.conn)

        self.queue = multiprocessing.Queue()

    def _persist(self):
        while True:
            tick = self.queue.get()
            try:
                logging.info(tick["window"])
                res = r.db(self.dbname).table(self.table).insert([tick], conflict="replace").run(self.conn)
                count = r.db(self.dbname).table(self.table).count().run(self.conn)
                logging.info(res)
                logging.info(count)
            except Exception as e:
                logging.error(e)
                logging.error(tick)
                self.conn = r.connect(self.host, self.port)

    def publish(self, msg):
        self.queue.put(json.loads(msg.data.decode("utf-8")))
        msg.ack()
        
    def _run(self):
        worker = Subscriber(
            topic_name=self.topic,
            project_id=self.pid
        )
        worker.run(callback=self.publish)

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

    runner = Ingress(
        topic=args.topic, 
        pid=args.project_id,
        host=args.host
    )
    runner.run()