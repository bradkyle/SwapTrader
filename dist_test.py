import wandb
from wandb.tensorflow import WandbHook
wandb.init(project="distswaptrader")

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
from env.DistSwapTradingEnv import DistSwapTradingEnv
import random
from random import shuffle
import csv
import time
import json
import random
import sys 
np.set_printoptions(threshold=sys.maxsize)

wandb.config.leverage = 50
wandb.config.initial_balance = 10
wandb.config.commission = 0.000
wandb.config.reward_func = None
wandb.config.state_buffer_size = 300
wandb.config.evaluation_history_size = 300
wandb.config.account_history_size = 300
wandb.config.scaler_high = 255
wandb.config.min_margin_ratio = 0.02
wandb.config.enable_decay = False
wandb.config.enable_hold_decay = False
wandb.config.decay = 0.05
wandb.config.threshold_max_steps = 30
wandb.config.learning_rate = 1e-4
wandb.config.randomize = False
wandb.config.has_hold = False
wandb.config.include_account = False
wandb.config.min_consec_steps = 0
wandb.config.num_actions = 2
wandb.config.obs_type = 'obs_1'
wandb.config.train_data_file = './data/FRACDIFF_DATA.parquet'
wandb.config.test_data_file = './data/FRACDIFF_TEST.parquet'
wandb.config.agent_id = 'ebfae67f-9e55-4c9d-b471-589d38227d5a'
wandb.config.env_name = 'DistSwapTradingEnv'

env = DummyVecEnv(
        [lambda: DistSwapTradingEnv(
            data_file=wandb.config.train_data_file,
            training=True,
            **dict(wandb.config)
        )]
    )

model = PPO2.load('./agents/'+wandb.config.agent_id+'.pkl', env=env)

# Load Model
# ========================================================>

obs = model.env.reset()
done=False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(reward)
    print(info)
    wandb.log(info[0])
    # test_env.render(mode="human")