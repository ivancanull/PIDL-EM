import numpy as np
from source import *
class Grid1D:
    def __init__(self,
                 size: int,
                 max_time: int,
                 # time_step: float,
                 cdtds: float,
                 ):
        
        self.size = size
        self.max_time = max_time
        # self.dt = time_step
        self.ez = np.zeros([self.max_time, self.size])
        self.hy = np.zeros([self.max_time, self.size])
        
        self.chyh = np.ones([self.size])
        self.chye = np.ones([self.size])
        self.ceze = np.ones([self.size])
        self.cezh = np.ones([self.size])

        self.cdtds = cdtds
        
    def updateE(self,
                t: int):

        self.ez[t, 1:] = self.ceze[1:] * self.ez[t-1, 1:] + \
                         self.cezh[1:] * (self.hy[t, 1:] - self.hy[t, :-1])
        return
    
    def updateH(self,
                t: int):
        self.hy[t, :-1] = self.chyh[:-1] * self.hy[t-1, :-1] + \
                          self.chye[:-1] * (self.ez[t-1, 1:] - self.ez[t-1, :-1]) 
        return

class Grid2D_TM:
    def __init__(self,
                 x_size: int,
                 y_size: int,
                 max_time: int,
                 # time_step: float,
                 cdtds: float,
                 ):
        
        self.x_size = x_size
        self.y_size = y_size
        self.max_time = max_time
        # self.dt = time_step
        self.ez = np.zeros([self.max_time, self.x_size, self.y_size])
        self.hx = np.zeros([self.max_time, self.x_size, self.y_size - 1])
        self.hy = np.zeros([self.max_time, self.x_size - 1, self.y_size])
        
        self.chxh = np.ones([self.x_size, self.y_size - 1])
        self.chxe = np.ones([self.x_size, self.y_size - 1])
        self.chyh = np.ones([self.x_size - 1, self.y_size])
        self.chye = np.ones([self.x_size - 1, self.y_size])
        self.ceze = np.ones([self.x_size, self.y_size])
        self.cezh = np.ones([self.x_size, self.y_size])

        self.cdtds = cdtds
        
    def updateE(self,
                t: int):
        self.ez[t, 1:-1, 1:-1] = self.ceze[1:-1, 1:-1] * self.ez[t-1, 1:-1, 1:-1] + \
                                 self.cezh[1:-1, 1:-1] * ((self.hy[t, 1:, 1:-1] - self.hy[t, :-1, 1:-1]) - (self.hx[t, 1:-1, 1:] - self.hx[t, 1:-1, :-1]))
        return
    
    def updateH(self,
                t: int):
        self.hx[t, :, :] = self.chxh[:, :] * self.hx[t-1, :, :] - \
                           self.chxe[:, :] * (self.ez[t-1, :, 1:] - self.ez[t-1, :, :-1])
        self.hy[t, :, :] = self.chyh[:, :] * self.hy[t-1, :, :] + \
                           self.chye[:, :] * (self.ez[t-1, 1:, :] - self.ez[t-1, :-1, :])
        return
        
