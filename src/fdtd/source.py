import torch
import math
import numpy as np
from grid import *

class Source1D:

    def __init__(self,
                 ):
        return

class Source2D:

    def __init__(self,
                 ):
        return
    
class GaussExcitation1D(Source1D):

    def __init__(self,
                 delay: int,
                 width: int,
                 cdtds: float):
        super().__init__()
        self.delay = delay
        self.width = width
        self.cdtds = cdtds

    def excite(self,
               time: float,
               location: float):
        return np.exp(-((time - self.delay - location / self.cdtds) / self.width) ** 2)

class HarmonicSource1D(Source1D):

    def __init__(self,
                 ppw: float,
                 cdtds: float):
        super().__init__()
        self.ppw = ppw # points per wavelength
        self.cdtds = cdtds

    def excite(self,
               time: float,
               location: float):
        return np.sin(2.0 * math.pi / self.ppw * (self.cdtds * time - location))
    
class RickerWaveletSource2D(Source2D):

    def __init__(self,
                 ppw: float,
                 cdtds: float):
        super().__init__()
        self.ppw = ppw # points per wavelength
        self.cdtds = cdtds
    
    def excite(self,
               time: float,
               location: float):
        arg = (math.pi * ((self.cdtds * time - location) / self.ppw - 1.0)) ** 2
        return (1.0 - 2.0 * arg) * np.exp(-arg)