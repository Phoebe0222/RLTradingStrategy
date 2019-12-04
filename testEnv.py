import gym
import json
import datetime as dt
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import argparse
from config import get_config
from envs.TradeEnv import TradeEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['stock', 'FX', 'option','bitcoin'])



if __name__ == '__main__':
    args = parser.parse_args()
    if args.env_name == 'stock':
        df = pd.read_csv('./data/stock/AAPL.csv')
        graph_title = 'Apple Stock'
    elif args.env_name == 'bitcoin':
        df = pd.read_csv('./data/bitcoin/coinbaseUSD.csv')
        graph_title = 'coinbase'
    elif args.env_name == 'FX':
        '''TODO'''
        pass
    elif args.env_name == 'option':
        '''TODO'''
        pass

    # spliting train/test env
    training_size = int(0.8*len(df))
    train_df = df.iloc[0:training_size]
    test_df = df.iloc[training_size+1:len(df)]
    train_df = train_df.sort_values(by=[train_df.columns[0]])
    test_df = test_df.sort_values(by=[test_df.columns[0]])

    # The algorithms require a vectorized environment to run
    train_env = DummyVecEnv([lambda: TradeEnv(train_df,args.env_name)])
    test_env = DummyVecEnv([lambda: TradeEnv(test_df,args.env_name)])
    # Define a model, doc:  https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html#logging-more-values
    model = PPO2(MlpPolicy, train_env, verbose=1,tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=10000)
    model.save('./models/ppo2_{}'.format(args.env_name))

    # init testing env
    obs = test_env.reset()
    
    # start agent 
    for _ in range(1000):
        train_env.render(title=graph_title) 
        action, _states = model.predict(obs)
        obs, rewards, done, info = train_env.step(action)
        
        if done: obs = test_env.reset()

    test_env.close()
    
