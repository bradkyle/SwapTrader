import wandb
from wandb.tensorflow import WandbHook
wandb.init(project="swaptrader")

import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import gym
import optuna
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import random
from stable_baselines.common.policies import MlpLnLstmPolicy, CnnPolicy, MlpPolicy, MlpLstmPolicy,CnnLnLstmPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, DDPG, ACER
from stable_baselines.ddpg.policies import LnMlpPolicy, LnCnnPolicy
from env.SwapTradingEnv2 import SwapTradingEnv
import random
from random import shuffle
import csv
import time
import json
import random
import sys 
np.set_printoptions(threshold=sys.maxsize)

env = DummyVecEnv(
        [lambda: SwapTradingEnv(
            data_file='./data/DATA.parquet',
            training=True
        )]
    )

test_env = DummyVecEnv(
        [lambda: SwapTradingEnv(
            data_file='./data/TEST.parquet',
            training=True
        )]
    )

# env = SubprocVecEnv([lambda: SwapTradingEnv(
#             data_file='/home/thorad/Core/Projects/SwapTrader/data/BTC-USD-SWAP-FRACDIFF.parquet'
#         ) for i in range(2)])

model = PPO2(
    MlpPolicy, 
    env, 
    verbose=1,
    tensorboard_log="./tensorboard"
)

for x in range(5):
    model.learn(50000)
    model.save('./agents/agent_'+str(x)+'.pkl')

    obs = test_env.reset()
    done=False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        print(reward)
        print(info)
        wandb.log(info[0])