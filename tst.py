import wandb
from wandb.tensorflow import WandbHook
wandb.init(project="testing")
import uuid
import os
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
from env.DistSwapTradingEnv3 import DistSwapTradingEnv
import random
from random import shuffle
import csv
import time
import json
import random
import sys 
np.set_printoptions(threshold=sys.maxsize)

agent_id = str(uuid.uuid4())

wandb.config.leverage = 70
wandb.config.initial_balance = 10
wandb.config.maker_commission = -0.00025
wandb.config.taker_commission = 0.00075
wandb.config.reward_func = 'sortino'
wandb.config.state_buffer_size = 300
wandb.config.evaluation_history_size = 300
wandb.config.account_history_size = 3
wandb.config.scaler_high = 255
wandb.config.min_margin_ratio = 0.005
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
wandb.config.train_data_file = './data/NODIFF_TEST.parquet'
wandb.config.test_data_file = './data/NODIFF_DATA.parquet'
wandb.config.env_name = 'DistSwapTradingEnv3'
wandb.config.train_steps = 60000
wandb.config.agent_id = agent_id

env = DummyVecEnv(
        [lambda: DistSwapTradingEnv(
            data_file=wandb.config.train_data_file,
            training=True,
            **dict(wandb.config)
        )]
    )

test_env = DummyVecEnv(
        [lambda: DistSwapTradingEnv(
            data_file=wandb.config.test_data_file,
            training=False,
            **dict(wandb.config)
        )]
    )

model = PPO2(
    MlpLnLstmPolicy, 
    env,
    learning_rate=wandb.config.learning_rate,
    verbose=1,
    nminibatches=1,
    tensorboard_log="./tensorboard"
)

model.learn(wandb.config.train_steps)
model.save(os.path.join(wandb.run.dir, "model.h5"))
model.save('./agents/'+ agent_id + '.pkl')

obs = test_env.reset()
done=False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    print(reward)
    print(info)
    wandb.log(info[0])

print(agent_id)
