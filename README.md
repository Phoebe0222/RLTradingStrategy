# RLTradingStrategy
A trading agent with reinforcement learning in OpenAI's customised environment 

## How to run
`python testEnv.py --env_name <asset-to-be-traded>`
For example,
`python testEnv.py --env_name stock`


## A screenshot of the trading process
Stock example
<img src='apple.png'>
Bbitcoin example
<img src='coinbase.png'>

## TODO:
- visualise using tensorboard
- split train/test dateset 
- [better reward function](https://medium.com/@SOGorman35/now-that-i-had-a-chance-to-read-your-article-in-a-bit-more-depth-ill-add-some-more-input-beyond-b71e442bb8a)
- multi-agent
- add word embedding layer for news data 
- use RNN, CNN 


## Citing
[Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)[[repo](https://github.com/notadamking/Stock-Trading-Environment)]

[Rendering elegant stock trading agents using Matplotlib and Gym](https://towardsdatascience.com/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4)[[repo](https://github.com/notadamking/Stock-Trading-Visualization)]

[Creating Bitcoin trading bots don’t lose money](https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29)[[repo](https://github.com/notadamking/RLTrader)]

[Optimizing deep learning trading bots using state-of-the-art techniques](https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b)

