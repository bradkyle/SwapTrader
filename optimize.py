import optuna

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import random
from random import shuffle
from nets.CustomCnnPolicy import CustomCnnPolicy
from nets.CustomLstmPolicy import CustomLstmPolicy

from env.MarginTradingEnv import MarginTradingEnv

reward_strategy = 'sortino'
db_name = 'paramsdb'

# number of parallel jobs
n_jobs = 24
# maximum number of trials for finding the best hyperparams
n_trials = 1000
# number of evaluations for pruning per trial
n_evaluations = 3

# Config
# =============================================================>

def optimize_envs(trial):
    return {
        'initial_balance':1, 
        'action_type': 'discrete_5',
        'action_granularity': 1,
        'threshold': 0.5,
        'window_size': trial.suggest_categorical('window_size', [1, 5, 10])
    }

def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

def optimize_policy(trial):
    return {
        'policy': int(trial.suggest_loguniform('n_steps', 16, 2048)),
    }

def optimize_agent(trial):
    env_params = optimize_envs(trial)
   
    policy = MlpLnLstmPolicy

    model_params = optimize_ppo2(trial)
    model = PPO2(
        policy, 
        train_env, 
        verbose=0, 
        nminibatches=1,
        tensorboard_log="./tensorboard", 
        **model_params
    )

    for eval_idx in range(n_evaluations-1):
        try:
            model.learn(total_timesteps=train_len)
        except AssertionError:
            raise

        reward_sum = 0.0
        done = False
        obs = test_env.reset()
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            reward_sum += reward

        trial.report(-1 * reward_sum, eval_idx)

        if trial.should_prune(eval_idx):
            raise optuna.structs.TrialPruned()

        eval_idx+=1
        test_env, train_env = create_envs(env_params, eval_idx)
        model.set_env(train_env)

    return -1 * reward_sum


def optimize():
    study_name = 'ppo2_' + reward_strategy
    study = optuna.create_study(
        study_name=study_name, storage=params_db_file, load_if_exists=True)

    try:
        study.optimize(optimize_agent, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


if __name__ == '__main__':
    optimize()
