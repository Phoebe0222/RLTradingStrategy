class config_stock:
    def __init__(self):
        # env config 
        self.max_steps = 1000
        self.init_bal = 10000
        self.lookback_range = 5
        self.tran_cost = 0.003
        self.serial = True
        self.num_portfolio_status = 5

class config_bitcoin:
    def __init__(self):
        # env config 
        self.max_steps = 1000
        self.init_bal = 10000
        self.lookback_range = 50
        self.tran_cost = 0.00075
        self.serial = True
        self.num_portfolio_status = 5

class FX:
    def __init__(self):
        # env config 
        self.max_steps = 1000
        self.init_bal = 10000
        self.lookback_range = 50
        self.tran_cost = 0.00075
        self.serial = True
        self.num_portfolio_status = 5

class options:
    def __init__(self):
        # env config 
        self.max_steps = 1000
        self.init_bal = 10000
        self.lookback_range = 50
        self.tran_cost = 0.00075
        self.serial = True
        self.num_portfolio_status = 5


def get_config(env_name):
    if env_name == 'stock':
        return config_stock()
    elif env_name == 'FX':
        return config_FX()
    elif env_name == 'options':
        return config_options()
    elif env_name == 'bitcoin':
        return config_bitcoin()