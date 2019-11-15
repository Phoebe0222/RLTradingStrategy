import gym
from gym import spaces
import numpy as np
from render.StockTradingGraph import StockTradingGraph
# config
max_share_num = 2147483647
max_share_price = 5000
max_steps = 1000
init_bal = 10000
lookback_window_size = 40

class TradingEnv(gym.Env):
    """Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, assetType, dayRange = 5):
        '''
        define action and observaton space as gym.spaces objects 
        input:
            assetType: 
            a string; type of assets that the agent trades with; 
            choose between 'stock', 'FX', 'options'
            
            df:
            a pandas dataframe containing data 
            
            dayRange:
            an integet; specifies the number of days that agent uses to determine actions
        '''
        super(TradingEnv, self).__init__()
        self.df = df
        self.dayRange = dayRange
        
        if assetType == 'stock':
            # actions include buy, sell, or hold x% 
            self.action_space = spaces.Box(low=np.array([0, 0]), 
                                           high=np.array([3, 1]), 
                                           dtype=np.float16)
            # observations include prices containing the open-high-low-close (OHLC) values for the last five days
            self.observation_space = spaces.Box(low=0, high=1, 
                                                shape=(self.dayRange+1, self.dayRange+1), 
                                                dtype=np.float16)

    
  
    def reset(self):
        
        '''
        initialise or reset an existing environment's state:
            balance, net_worth and max net worth are initial account balance;
            intial timestep is randomly chosen within the data timeframe
        
        '''
            
        self.balance = init_bal
        self.max_bal = init_bal
        self.net_worth = init_bal
        self.max_net_worth = init_bal
        self.shares_held = 0
        self.avg_cost = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0 #np.random.randint(0, self.df.shape[0] - self.dayRange)
        return self._next_observation()
    
    def _take_action(self, action):
        '''
        execute a given action for one time step within the environment 
        input:
            action:
            an np.array of shape 2;
            action[0] is a float between 0 to 3 that specifies whether buy (0~1), sell(1~2) or hold(2~3);
            action[1] is a float between 0 to 1 that specifies the bbuy/sell amount in %
            
        '''
        
        # set the current price to be a random price between open and close price within the timestep
        current_price = np.random.uniform(
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "Close"])
        
        action_type = action[0] 
        amount = action[1]
        
        if action_type < 1:
            '''
            buy x% of balance in shares:
            1. calculate total possible number of shares from the current balance and current price
            2. buy x% of the total possible number of shares
            3. calculate cost of buying the shares
            4. update balance with buying cost
            5. update previous average buying cost with the new buying cost 
            6. update number of shares held 
            
            '''
           
            total_possible = self.balance / current_price 
            shares_bought = total_possible * amount
            buying_cost = shares_bought * current_price
            self.balance -= buying_cost
            
            prev_cost = self.avg_cost * self.shares_held
            self.avg_cost = (prev_cost + buying_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            
            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_bought, 'total': buying_cost,
                                    'type': "buy"})
        
        elif action_type < 2:
            '''
            sell x% of shares held:
            1. sell x% of the total number of shares held
            2. update balance with selling revenue
            3. update number of shares held 
            4. update total number of shares sold  
            5. update total value of share sold
            '''

            shares_sold = self.shares_held * amount 
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
            
            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})
        
        
        # update portfolio net worth 
        self.net_worth = self.balance + self.shares_held * current_price
        
        # update maximum net worth
        if self.net_worth > self.max_net_worth: self.max_net_worth = self.net_worth
            
        # update maximum balance 
        if self.balance > self.max_bal: self.max_bal = self.balance 
            
    
    def _next_observation(self):
        
        '''
        given the current timestep and action, get the OHLC data from the last 5 days, normalise it, and appends portfolio's status
        
        '''

        
        OHCL = np.array([
            self.df.loc[self.current_step: self.current_step + self.dayRange, 'Open'].values / max_share_price,
            self.df.loc[self.current_step: self.current_step + self.dayRange, 'High'].values / max_share_price,
            self.df.loc[self.current_step: self.current_step + self.dayRange, 'Low'].values / max_share_price,
            self.df.loc[self.current_step: self.current_step + self.dayRange, 'Close'].values / max_share_price,
            self.df.loc[self.current_step: self.current_step + self.dayRange, 'Volume'].values / max_share_num
        ])
        
        portfolio = np.array([[
            self.balance / self.max_bal,
            self.max_net_worth / self.max_net_worth,
            self.shares_held / max_share_num,
            self.avg_cost / max_share_price,
            self.total_shares_sold / max_share_num,
            self.total_sales_value / (max_share_num * max_share_price)
        ]])

        obs = np.append(OHCL, portfolio, axis=0)
        return obs
    
    
    
    def step(self, action):
        '''
        given the action taken, update rewards, observations, and whether it's terminated, i.e. negative balance
        '''
        self._take_action(action)
        
        self.current_step += 1
        
        # discount the account balance 
        delay_modifier = (self.current_step / max_steps)
        reward = self.balance * delay_modifier
        
        # terminate when net worh is below 0 or timestep exceeds timeframe
        done = self.net_worth <= 0 or self.current_step > self.df.shape[0] - self.dayRange
        
        obs = self._next_observation()
        return obs, reward, done, {}
        
        
    def _render_to_file(self, filename = 'render.txt'):
        # Render the environment to the screen
        profit = self.net_worth - init_bal
        
        file = open(filename, 'a+')
             
        file.write(f'Step: {self.current_step}\n')    
        file.write(f'Balance: {self.balance}\n')      
        file.write(f'Shares held: {self.shares_held}(Total sold: {self.total_shares_sold})\n')      
        file.write(f'Avg cost for held shares: {self.avg_cost}(Total sales value: {self.total_sales_value})\n')        
        file.write(f'Net worth: {self.net_worth}(Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')
        
        file.close()
        
        
        
    def render(self, mode='live', title=None, **kwargs):
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(self.df, title)
                
            if self.current_step > lookback_window_size:
                self.visualization.render(self.current_step, self.net_worth, 
                                          self.trades, window_size=lookback_window_size)
                
    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
