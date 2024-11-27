
import torch
import torch.nn as nn

class PAPE(nn.Module):
    def __init__(self,
                 max_min):
        pass
class Maxwell_EzHxHy(nn.Module):
    def __init__(self,
                 dx: float,
                 dy: float,
                 dz: float,
                 dte: float,
                 delta_t: float,
                 murz: float = 1.0,
                 nz: int = 2,):
        super(Maxwell_EzHxHy, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.murz = murz
        self.dte = dte
        self.delta_t = delta_t
        self.nz = nz
        # print(f"delta dte: {self.dte}, delta t: {self.delta_t}")

    def forward(self,
                prediction: torch.Tensor,
                coefficients: torch.Tensor,
                source: torch.Tensor,
                ):
        
        # Ez
        Ez = prediction[:, :, 0:self.nz, :, :]
        Hx = prediction[:, :, self.nz:self.nz*2, :, :]
        Hy = prediction[:, :, self.nz*2:self.nz*3, :, :]
        sigez = coefficients[:, self.nz*5:self.nz*6, :, :]
        epsrz = coefficients[:, self.nz*2:self.nz*3, :, :]

        
        dhydx = (Hy[:, 1:, :, 1:, :] - Hy[:, 1:, :, :-1, :]) / self.dx # (nx, ny - 1, nz, nt)
        dhxdy = (Hx[:, 1:, :, :, 1:] - Hx[:, 1:, :, :, :-1]) / self.dy # (nx - 1, ny, nz, nt)
        # choose (nx - 1, ny - 1, nz)
        curnt_Ez = Ez[:, 1:, :, :, :] * 100
        prev_Ez = Ez[:, :-1, :, :, :] * 100
        
        lz = (sigez[:, None, ...] * curnt_Ez + epsrz[:, None, ...] * (curnt_Ez - prev_Ez)/ self.dte / self.delta_t)[..., :-1, :-1] # lz shape: (b, nt - 1, nz, nx, ny)
        rz = dhydx[..., :-1] - dhxdy[..., :-1, :] - source[:, 1:, :, :-1, :-1] / self.dz

        # apply mask
        # lz [..., 0:1] = 0
        # lz [..., -1:] = 0
        # rz [..., 0:1] = 0
        # rz [..., -1:] = 0
        lz [..., 0:1, :] = 0
        lz [..., -1:, :] = 0
        rz [..., 0:1, :] = 0
        rz [..., -1:, :] = 0

        # print("lz: ", torch.mean(lz / 1000).item(), "rz: ", torch.mean(rz / 1000).item())
        return lz / 1e9, rz / 1e9

class CustomLossEzHxHy(nn.Module):
    def __init__(self,
                 loss_fn: nn.Module = nn.MSELoss()):
        
        super(CustomLossEzHxHy, self).__init__()
        self.loss = loss_fn

    def loss_ph(self,
                lz: torch.Tensor,
                rz: torch.Tensor,
                ):
        return self.loss(lz, rz)

    def loss_data(self,
                  prediction: torch.Tensor,
                  truth: torch.Tensor):
        return self.loss(prediction, truth)
    
    def forward(self,
                prediction: torch.Tensor,
                truth: torch.Tensor,
                lz: torch.Tensor,
                rz: torch.Tensor,
                ):
        # boundary loss
        
        loss_ph = self.loss_ph(lz, rz)
        loss_data = self.loss_data(prediction, truth)
        return loss_ph, loss_data
    