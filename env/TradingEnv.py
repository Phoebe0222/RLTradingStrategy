import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
from render.TradingGraph import TradingGraph
from config import get_config


class TradingEnv(gym.Env):
    '''Trading Environment that follows gym interface'''
    metadata = {'render.modes': ['live', 'file', 'none']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, assetType):
        '''
        define action and observaton space as gym.spaces objects 
        input:
            assetType: 
            a string; type of assets that the agent trades with; 
            choose between 'stock', 'FX', 'options', 'bitcoin'
            
            df:
            a pandas dataframe containing data 
        '''
        super(TradingEnv, self).__init__()
        self.df = df.dropna().reset_index()
        self.assetType = assetType
        
        self.config = get_config(self.assetType) 
        
        # actions include buy, sell, or hold x% 
        self.action_space = spaces.Box(low=np.array([0, 0]), 
                                           high=np.array([3, 1]), 
                                           dtype=np.float16)
        # observations include prices containing the open-high-low-close (OHLC) values for the last five days
        self.observation_space = spaces.Box(low=0, high=1, 
                                                shape=(df.shape[1] - 1 + 5, self.config.lookback_range+1), 
                                                dtype=np.float16)
            


    
  
    def reset(self):
        
        '''
        initialise or reset an existing environment's state:
            balance, net_worth are initial account balance;
            intial timestep is randomly chosen within the data timeframe
        
        '''
        self.balance = self.config.init_bal
        self.net_worth = self.config.init_bal
        self.assets_held = 0
        self.avg_cost = 0
        self.total_assets_sold = 0
        self.total_sales_value = 0

        self._reset_session()
        # repeat these for the number of times same as lookback_range + 1
        self.account_history = np.repeat([
                [self.net_worth],
                [self.assets_held],
                [self.avg_cost],
                [self.total_assets_sold],
                [self.total_sales_value]
            ], self.config.lookback_range + 1, axis=1)

        self.trades = []
        return self._next_observation()

    def _reset_session(self):

        '''
        reset the trading session, including resetting current_step to 0, 
        defining the steps left to traverse, defining the start of traverse point,
        extracting the data from the whole dataframe  
        '''
        self.current_step = 0
        # traversing data in serial style 
        if self.config.serial: 
            self.steps_left = len(self.df) - self.config.lookback_range - 1
            self.frame_start = self.config.lookback_range
        # traversing data randomly
        else:
            self.steps_left = np.random.randint(int(len(self.df)/2), len(self.df))
            self.frame_start = np.random.randint(
                self.config.lookback_range, len(self.df) - self.steps_left)
        # extract active dataframe
        self.active_df = self.df[self.frame_start -   
            self.config.lookback_range:self.frame_start + self.steps_left]

    def _next_observation(self):
        
        '''
        given the current timestep and action, get the OHLC data from the last 5 days, normalise it, and appends portfolio's status
        
        '''
        start = self.current_step
        end = self.current_step + self.config.lookback_range + 1 
        df = self.active_df
        
        # excluding the first column which is date or time
        OHCL = [df[df.columns[2]][start:end].values]
        for col in df.columns[3:]:
            OHCL = np.append(OHCL,[df[col][start:end].values],axis=0)
        
        # normalisation 
        OHCL = self.scaler.fit_transform(OHCL)
        scaled_history = self.scaler.fit_transform(self.account_history)
        
        # append abservation from market data and account history in the same date range
        obs = np.append(OHCL, scaled_history[:, -(self.config.lookback_range + 1):], axis=0)
        #print(obs)
        return obs

    
    def step(self, action):
        '''
        given the action taken, update rewards, observations, and whether it's terminated, i.e. negative balance
        '''
        self._take_action(action)
        
        self.current_step += 1
        self.steps_left -= 1


        # determination of rewards depends on the asset type
        if self.assetType == 'stock':
            # discount the account balance 
            delay_modifier = (self.current_step / self.config.max_steps)
            reward = self.balance * delay_modifier
        elif self.assetType == 'bitcoin':
            reward = self.net_worth

        # restart a trading session when the shole dataframe is traversed 
        if self.steps_left == 0:
            self.balance += self.assets_held * self.current_price 
            self.assets_held = 0
            self._reset_session()
        
        # terminate when net worh is below 0 or timestep exceeds timeframe
        done = self.net_worth <= 0 or self.steps_left == 0
        # load observations 
        obs = self._next_observation()
        return obs, reward, done, {}
    

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
        self.current_price = np.random.uniform(
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "Close"])
        
        action_type = action[0] 
        amount = action[1]
        
        if action_type < 1:
            '''
            buy x% of balance in assets:
            1. calculate total possible number of assets from the current balance and current price
            2. buy x% of the total possible number of assets
            3. calculate cost of buying the assets
            4. update balance with buying cost
            5. update previous average buying cost with the new buying cost 
            6. update number of assets held 
            
            '''
           
            total_possible = int(self.balance / self.current_price) 
            assets_bought = int(total_possible * amount)
            buying_cost = assets_bought * self.current_price * (1 + self.config.tran_cost)
            self.balance -= buying_cost
            
            prev_cost = self.avg_cost * self.assets_held
            self.avg_cost = (prev_cost + buying_cost) / (self.assets_held + assets_bought) if self.assets_held + assets_bought > 0 else 0
            self.assets_held += assets_bought
            
            if assets_bought > 0:
                self.trades.append({'step': self.frame_start + self.current_step,
                                    'assets': assets_bought, 'total': buying_cost,
                                    'type': "buy"})
        
        elif action_type < 2:
            '''
            sell x% of assets held:
            1. sell x% of the total number of assets held
            2. update balance with selling revenue
            3. update number of assets held 
            4. update total number of assets sold  
            5. update total value of share sold
            '''

            assets_sold = self.assets_held * amount 
            sales = assets_sold * self.current_price * (1 - self.config.tran_cost)

            self.balance += sales
            self.assets_held -= assets_sold
            self.total_assets_sold += assets_sold
            self.total_sales_value += sales
            
            if assets_sold > 0:
                self.trades.append({'step': self.frame_start + self.current_step,
                                    'assets': assets_sold, 'total': sales,
                                    'type': "sell"})
        
        
        # update portfolio net worth 
        self.net_worth = self.balance + self.assets_held * self.current_price
        # update account hisotry 
        self.account_history = np.append(self.account_history,
                                        [[self.net_worth],
                                        [self.assets_held],
                                        [self.avg_cost],
                                        [self.total_assets_sold],
                                        [self.total_sales_value]], axis = 1)
      
            
    def _render_to_file(self, filename = 'render.txt'):
        # Render the environment to the screen
        profit = self.net_worth - self.config.init_bal
        
        file = open(filename, 'a+')
             
        file.write(f'Step: {self.current_step}\n')    
        file.write(f'Balance: {self.balance}\n')      
        file.write(f'Shares held: {self.assets_held}(Total sold: {self.total_assets_sold})\n')      
        file.write(f'Avg cost for held assets: {self.avg_cost}(Total sales value: {self.total_sales_value})\n')        
        file.write(f'Net worth: {self.net_worth}\n')
        file.write(f'Profit: {profit}\n\n')
        
        file.close()
        
        
        
    def render(self, mode='live', title=None, **kwargs):
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
        elif mode == 'live':
            if self.viewer == None:
                self.viewer = TradingGraph(self.df, title)
                
            if self.current_step > self.config.lookback_range:
                self.viewer.render(self.current_step, self.net_worth, 
                                          self.trades, window_size=self.config.lookback_range)
                
    def close(self):
        if self.viewer != None:
            self.viewer.close()
            self.viewer = None
