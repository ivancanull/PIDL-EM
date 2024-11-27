import numpy as np
from grid import *

class BC1D:
    def __init__(self,
                 left: int,
                 right: int):
        self.left = left
        self.right = right

class ABC1D(BC1D):
    def updateH(self,
                time: int,
                grid: Grid1D):
        grid.hy[time, self.left] = grid.hy[time-1, self.right]
        return
    
    def updateE(self,
                time: int,
                grid: Grid1D):
        
        grid.ez[time, self.left] = grid.ez[time-1, self.right]
        return
    