import gym
import json
import datetime as dt
from stable_baselines.common.vec_env import DummyVecEnv
import pandas as pd

from env.TradingEnv import TradingEnv



df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: TradingEnv(df,'stock')])


obs = env.reset()
for _ in range(1000):
    env.render() 
    action = env.action_space.sample() # takes a random action
    obs, rewards, done, info = env.step(action)
    
    if done: obs = env.reset()
env.close()
    
