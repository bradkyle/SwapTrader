import numpy as np

BASE_URL = "https://testnet.bitmex.com/api/v1/"
# BASE_URL = "https://www.bitmex.com/api/v1/" # Once you're ready, uncomment this.

API_KEY = "lAUW_RgS5oft_nM9tRf2-T-w"
API_SECRET = "TAn3meA6g8n1zMP-PCWdXEVKL8fB34nnVB4mnhXzjJxLwWDB"

# Amount of leverage to use 
LEVERAGE = 1

# Instrument to market make on BitMEX.
SYMBOL = "XBTUSD"

# If True, will only send orders that rest in the book (ExecInst: ParticipateDoNotInitiate).
# Use to guarantee a maker rebate.
# However -- orders that would have matched immediately will instead cancel, and you may end up with
# unexpected delta. Be careful.
POST_ONLY = True

# If true, don't set up any orders, just say what we would do
# DRY_RUN = True
DRY_RUN = False

# Wait times between orders / errors
API_REST_INTERVAL = 1
API_ERROR_INTERVAL = 10
TIMEOUT = 7

# If we're doing a dry run, use these numbers for BTC balances
DRY_BTC = 50

# Available levels: logging.(DEBUG|INFO|WARN|ERROR)
LOG_LEVEL = logging.INFO

# Specify the contracts that you hold. These will be used in portfolio calculations.
CONTRACTS = ['XBTUSD']


class Runner():
    def __init__(self, **kwargs):
        pass

    def _ingress(self):
        pass

    def cancel_all_orders(self):
        if self.dry_run:
            return

        logger.info("Resetting current position. Canceling all existing orders.")
        tickLog = self.get_instrument()['tickLog']

        # In certain cases, a WS update might not make it through before we call this.
        # For that reason, we grab via HTTP to ensure we grab them all.
        orders = self.bitmex.http_open_orders()

        for order in orders:
            logger.info("Canceling: %s %d @ %.*f" % (order['side'], order['orderQty'], tickLog, order['price']))

        if len(orders):
            self.bitmex.cancel([order['orderID'] for order in orders])

        sleep(settings.API_REST_INTERVAL)

    def sanity_check(self):
        """Perform checks before placing orders."""

        # Check if OB is empty - if so, can't quote.
        self.exchange.check_if_orderbook_empty()

        # Ensure market is still open.
        self.exchange.check_market_open()

    
    def exit(self):
        logger.info("Shutting down. All open orders will be cancelled.")
        try:
            self.exchange.cancel_all_orders()
            self.exchange.bitmex.exit()
        except errors.AuthenticationError as e:
            logger.info("Was not authenticated; could not cancel orders.")
        except Exception as e:
            logger.info("Unable to cancel orders: %s" % e)

        sys.exit()

    def _exec(self):
        obs = self.get_obs()
        action = self.model.predict([obs])


        if action != self.prev_action:
            self.cancel_all_orders()
            available_balance = self.available_balance()

            if action > 0:
                long_dist = available_balance * abs(action)
                short_dist = 0
            elif action <0:
                short_dist = available_balance * abs(action)
                long_dist = 0
            elif action == 0:
                short_dist = 0
                long_dist = 0

            next_short = int()
            next_long = int()

            short_delta = next_short - self.current_short()
            long_delta = next_short - self.current_long()

            orders = []

            #TODO check order types
            if short_delta > 0:
                # open_short ask price
                self.asks.append(Order(
                    price=state.ask_price,
                    quantity=abs(short_delta),
                    type=orderType.OPEN_SHORT
                ))
                
            elif short_delta < 0:
                # close_short bid price
                self.bids.append(Order(
                    price=state.bid_price,
                    quantity=abs(short_delta),
                    type=orderType.CLOSE_SHORT
                ))
            
            if long_delta > 0:
                # open long ask price
                self.bids.append(Order(
                    price=state.bid_price,
                    quantity=abs(long_delta),
                    type=orderType.OPEN_LONG
                ))

            elif long_delta < 0:
                # open short bid price
                self.asks.append(Order(
                    price=state.ask_price,
                    quantity=abs(long_delta),
                    type=orderType.CLOSE_LONG
                )) 

            self.create_bulk_orders(orders)