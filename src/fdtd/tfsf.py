from grid import *
from source import *

class TFSF1D:
    def __init__(self,
                 boundary_location: int,
                 source: Source1D):
        self.boundary_location = boundary_location
        self.source = source

    def updateH(self,
               time: int,
               grid: Grid1D):
        grid.hy[time, self.boundary_location] -= self.source.excite(time, 0) * grid.chye[self.boundary_location]
        return

    def updateE(self,
               time: int,
               grid: Grid1D):
        grid.ez[time, self.boundary_location + 1] += self.source.excite(time + 0.5, -0.5)

        return