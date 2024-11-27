from typing import List, Dict


from math import exp
from scipy.stats import bernoulli

class LossFIFO():
    def __init__(self, 
                 max_len: int,
                 m: int):
        self.max_len = max_len
        self.m = m
        self.hist_losses = [[] for _ in range(m)]
    
    def update(self, losses: List[float]):
        for i, loss in enumerate(losses):
            self.hist_losses[i].append(loss)
            if len(self.hist_losses[i]) > self.max_len:
                self.hist_losses[i] = self.hist_losses[i][1:]
    
    def get(self):
        return self.hist_losses
    
    def get_first(self):
        return [fifo[0] for fifo in self.hist_losses]

class ReLoBPeCh():
    def __init__(self,
                 expected_saudade: float = 0.99,   
                 alpha: float = 0.99,
                 beta: float = 0.5,
                 temperature: float = 100,
                 wait: int = 200,
                 max_len: int = 100
                 ):
        
        self.saudade = expected_saudade
        self.alpha = alpha
        self.temperature = temperature
        # self.intrisic_losses = intrisic_losses
        self.beta = beta
        self.peak_losses = [0.0, 0.0]
        self.peak_epoch = 0
        self.wait = wait
        self.max_len = max_len
    

    def update(self,
               epoch,
               losses: List[float],
               ):
        
        if epoch == 0:
            self.inital_losses = losses
            self.m = len(losses)
            self.factors = [0.0, 1.0]
            self.hist_losses = LossFIFO(self.max_len, self.m)
            
        elif epoch < self.wait:
            self.hist_losses.update(losses)            

        else:
            for i in range(self.m):
                if losses[i] > self.peak_losses[i]:
                    self.peak_losses[i] = losses[i]
            
                peak_loss_factor = self.beta * self.balance_factor(losses[i], losses, self.peak_losses[i], self.peak_losses) + (1 - self.beta) * self.balance_factor(losses[i], losses, self.hist_losses.get_first()[i], self.hist_losses.get_first())

                self.factors[i] = self.alpha * self.factors[i] + (1 - self.alpha) * peak_loss_factor
        
            self.hist_losses.update(losses)

        return self.factors
    
    def balance_factor(self,
                       target_loss: float,
                       all_losses: List[float],
                       hist_target_loss: float,
                       hist_all_losses: List[float]):
        
        assert len(all_losses) == self.m
        # a = exp(target_loss / hist_target_loss / self.temperature)
        a = exp(self.temperature * (target_loss - hist_target_loss))
        # b = sum([exp(all_losses[i] / hist_all_losses[i] / self.temperature) for i in range(self.m)]) + 1e-8
        b = sum([exp(self.temperature * (all_losses[i] - hist_all_losses[i])) for i in range(self.m)]) + 1e-8
        
        return a / b

    def state_dict(self):

        return {'m': self.m, 'factors': self.factors, 'inital_losses': self.inital_losses, 'hist_losses': self.hist_losses.get(), 'peak_losses': self.peak_losses, }

    def load_state_dict(self, state_dict: Dict):
        self.m = state_dict['m']
        self.factors = state_dict['factors']
        self.inital_losses = state_dict['inital_losses']
        self.hist_losses = LossFIFO(self.max_len, self.m)
        self.hist_losses.hist_losses = state_dict['hist_losses']
        self.peak_losses = state_dict['peak_losses']
