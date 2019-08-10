import os
from google.cloud import pubsub_v1
import json
import argparse
import logging
import pandas as pd
import zlib

def inflate(data):
    decompress = zlib.decompressobj(
            -zlib.MAX_WBITS  # see above
    )
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated

class Subscriber():

    def __init__(
        self,
        project_id,
        topic_name,
        compressed=True
    ):
        if topic_name is None:
            raise ValueError("Please specify topic name")
        
        if project_id is None:
            raise ValueError("Please specify project id")

        self.compressed = compressed
        self.subscriber = pubsub_v1.SubscriberClient()
        self.topic_path = 'projects/{project_id}/topics/{topic}'.format(
            project_id=project_id,
            topic=topic_name,
        )
        self.subscription_path = 'projects/{project_id}/subscriptions/{sub}'.format(
            project_id=project_id,
            sub='test_subscripton_'+topic_name
        )

        project_path = self.subscriber.project_path(project_id)
        all_subscriptions = [s.name for s in self.subscriber.list_subscriptions(project_path)]

        if self.subscription_path not in all_subscriptions:
            self.subscriber.create_subscription(
                name=self.subscription_path, 
                topic=self.topic_path
            )

    def run(self, callback):
        future = self.subscriber.subscribe(
            self.subscription_path, 
            callback
        )

        try:
            future.result()
        except Exception as ex:
            self.subscription.close()
            logging.error(ex)
            raise


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
        default="prices", 
        type=str, 
        help='Google cloud pubsub topic name'
    )

    args = parser.parse_args()
    
    print("="*90)
    print(args.project_id)
    print(args.topic)
    print("="*90)

    msgs = []

    import sys 
    def callback(msg):
        x = json.loads(msg.data)
        df = pd.DataFrame([x])
        print(df.head())
        msg.ack()


    worker = Subscriber(
        topic_name=args.topic,
        project_id=args.project_id
    )
    worker.run(callback=callback)
