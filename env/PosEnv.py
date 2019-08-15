
import collections

Position = collections.namedtuple(
    'Position',
    []
)

class Account():
    def __init__(self, *args, **kwargs):
        pass

        self.amount = None
        self.availableMargin = None
        


class PosEnv():
    def __init__(
        self,
        state_history_size=50
    ):
        pass

    def _take_action(self):
        state = self._current_state()

        # Update order book and position with latest state
        # ---------------------------------------------------------->
        self.orderbook.update(state)
        self.account.update(state)
        self.position.update(state)

        # Check if position can be liquidated
        # ---------------------------------------------------------->
        if self.position.is_liquidated and self.position.is_short:
            self.position.liquidate(price=state.bid_price, fee=self.taker_fee)

        if self.position.is_liquidated and self.position.is_long:
            self.position.liquidate(price=state.ask_price, fee=self.taker_fee)

        # Conduct check on orderbook to see if 
        # any orders can be filled/partially filled
        # then add amounts therin to the position  
        # ---------------------------------------------------------->

        # Update position values with respect to the 
        # current state
        # ---------------------------------------------------------->
        # unrealized profit
        # realized profit
        

        # If the action has changed:

            # Cancel all orders
            self.orderbook.cancel_all()

            # Get target distribution of value

            # Calculate Deltas

            # Add orders to order book for the next step


        # Update historic values

        pass

    def reset(self):
        pass
    
    def step(self, action):
        pass