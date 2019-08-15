
import collections

class OrderBook():
    def __init__(
        self,
        state_history_size=50
    ):
        self.bids = []
        self.asks = []
        self.state_history = collections.deque(
            maxlen=self.state_history_size
        )

    def add_bid(self):
        pass

    def add_ask(self):
        pass
    
    def update(self, state):
        pass

    