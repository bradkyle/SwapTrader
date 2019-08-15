
if __name__ == '__main__':
    from constants import *
else:
    from env.constants import *

class Position():

    def __init__(
        self,
        account=0,
        avgCostPrice=0,
        avgEntryPrice=0,
        bankruptPrice=0,
        breakEvenPrice=0,
        currentComm=0,
        currentCost=0,
        currentQty=0,
        leverage=0,
        liquidationPrice=0,
        lastPrice=0,     
        maintMarginReq=0,


    ):
        #TODO position available
        #TODO buy
        #TODO sell
        #TODO current_short
        #TODO current_long
        #TODO liquidation price
        pass

    @property
    def is_short(self):
        return self.currentQty < 0

    @property
    def is_long(self):
        return self.currentQty > 0

    @property
    def margin_ratio(self):
        return self.account/abs(self.currentQty)

    @property
    def is_liquidated(self):
        return self.margin_ratio < self.maintMarginReq and self.margin_ratio > 0

    @property
    def unrealized_pnl(self):
        pass

    @property
    def average_entry_price(self):
        pass

    def liquidate(self, price, fee):
        qty = abs(self.currentQty) * price
        cost = qty*fee
        qty = qty - cost 
        self.currentQty = 0
        self.account += qty
        self.lastPrice = price


    def sell(self, order, fee):
        qty = (1/self.avgEntryPrice - 1/order.price) * order.quantity

        if self.
        self.currentQty -= (qty - qty*fee)

        self.lastPrice = order.price


    def buy(self, order, fee):
        qty = (1/self.avgEntryPrice - 1/order.price) * order.quantity

        self.currentQty += (qty - qty*fee)

        self.lastPrice = order.price