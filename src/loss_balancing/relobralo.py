
from typing import List, Dict


from math import exp
from scipy.stats import bernoulli

import torch
# An Implementation of ReLoBRaLo (Relative Loss Balancing with Random Lookback)
# Bischof, Rafael, and Michael Kraus. "Multi-objective loss balancing for physics-informed deep learning." arXiv preprint arXiv:2110.09813 (2021).

class ReLoBRaLo():

    def __init__(self,
                 expected_saudade: float = 0.99,   
                 alpha: float = 0.99,
                 temperature: float = 1e-2,
                 ):
        
        self.saudade = expected_saudade
        self.alpha = alpha
        self.temperature = temperature
        # self.intrisic_losses = intrisic_losses
        

    def update(self,
               epoch,
               losses: List[float],
               ):
        
        if epoch == 0:
            self.inital_losses = losses
            self.m = len(losses)
            self.factors = [1.0 for _ in range(self.m)]
            
        else:
            for i in range(self.m):
                p = bernoulli.rvs(self.saudade, size=1)[0]
                hist_factor = p * self.factors[i] + (1 - p) * self.balance_factor(losses[i], losses, self.inital_losses[i], self.inital_losses)
                self.factors[i] = self.alpha * hist_factor + (1 - self.alpha) * self.balance_factor(losses[i], losses, self.hist_losses[i], self.hist_losses)
        
        self.hist_losses = losses
        return self.factors

    def balance_factor(self,
                       target_loss: float,
                       all_losses: List[float],
                       hist_target_loss: float,
                       hist_all_losses: List[float]):
        
        assert len(all_losses) == self.m
        a = exp(target_loss / hist_target_loss / self.temperature)
        b = sum([exp(all_losses[i] / hist_all_losses[i] / self.temperature) for i in range(self.m)])
        return a / b
    
    def state_dict(self):
        return {'m': self.m, 'factors': self.factors, 'inital_losses': self.inital_losses, 'hist_losses': self.hist_losses}

    def load_state_dict(self, state_dict: Dict):
        self.m = state_dict['m']
        self.factors = state_dict['factors']
        self.inital_losses = state_dict['inital_losses']
        self.hist_losses = state_dict['hist_losses']
        