import wandb
from wandb.tensorflow import WandbHook
# wandb.init(project="margintrader")

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
from env.SwapTradingEnv import SwapTradingEnv
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
            data_file='./data/TEST.parquet'
        )]
    )

# Load Model
# ========================================================>
model = PPO2.load('./agents/agent.pkl', env=env)

test_env = model.get_env()
obs = env.reset()
done=False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render(mode="human")