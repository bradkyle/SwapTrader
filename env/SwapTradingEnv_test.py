

import pytest
import mock
from env.SwapTradingEnv import SwapTradingEnv
from mock import patch, Mock
import pandas as pd
import json
import numpy as np
import pyarrow.parquet as pq

class TestSwapTradingEnv(SwapTradingEnv):
    def __init__(self, current_price=10000, data_file='./data/test.parquet'):
        super(TestSwapTradingEnv, self).__init__(data_file=data_file)
        self.current_price = current_price

    def _current_price(self):
        return self.current_price


@pytest.mark.parametrize(
    "prev_action, next_action, prev_short, prev_long, prev_balance, exp_short, exp_long, exp_balance", 
    [
        (
            1,
            2, 
            0, 
            0, 
            1, 
            0, 
            0,
            0 
        ),
    ]
)
def test_take_action_without_commission(
    prev_action, 
    next_action, 
    prev_short,
    prev_long,
    prev_balance,
    exp_short,
    exp_long,
    exp_balance
):
   
    env = TestSwapTradingEnv()


    account = env._take_action(next_action)

    assert(account.long_position == exp_long)