import gym
import json
import datetime as dt
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import argparse
from config import get_config
from env.TradingEnv import TradingEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['stock', 'FX', 'options','bitcoin'])



if __name__ == '__main__':
    args = parser.parse_args()
    if args.env_name == 'stock':
        df = pd.read_csv('./data/AAPL.csv')
        df = df.sort_values('Date')
        graph_title = 'Apple Stock'
    elif args.env_name == 'bitcoin':
        df = pd.read_csv('./data/coinbaseUSD.csv')
        df = df.sort_values('Timestamp')
        graph_title = 'coinbase'

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: TradingEnv(df,args.env_name)])

    model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=50)

    obs = env.reset()
    
    for _ in range(1000):
        env.render(title=graph_title) 
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        
        if done: obs = env.reset()


    env.close()
    
