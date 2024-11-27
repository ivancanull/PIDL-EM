from typing import List, Sequence, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["GradientChecker"]

class GradientChecker():
    """
    Check the gradients of Maxwell Equations in the structure is zero or not.
    """
    def __init__(self,
                 shape: Tuple[int],
                 npml: int,
                 dx: float,
                 dy: float,
                 dz: float,
                 dte: float | torch.Tensor,
                 dtm: float | torch.Tensor,
                 nt: int,
                 dt: int):
        self.nt = nt
        self.shape = shape
        self.npml = npml
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dte = dte
        self.dtm = dtm
        self.sampled_t = np.arange(self.nt)[::dt]
        # print(f"dx: {dx}, dy: {dy}, dz: {dz}, dte: {dte.item()}, dtm: {dtm.item()}")
        # self.prev_ex = self.curnt_ex = torch.zeros(1, 1, shape[0], shape[1]+1, shape[2]+1)
        # self.prev_ey = self.curnt_ey = torch.zeros(1, 1, shape[0]+1, shape[1], shape[2]+1)
        # self.prev_ez = self.curnt_ez = torch.zeros(1, 1, shape[0]+1, shape[1]+1, shape[2])
        # self.prev_hx = self.curnt_hx = torch.zeros(1, 1, shape[0]+1, shape[1], shape[2])
        # self.prev_hy = self.curnt_hy = torch.zeros(1, 1, shape[0], shape[1]+1, shape[2])
        # self.prev_hz = self.curnt_hz = torch.zeros(1, 1, shape[0], shape[1], shape[2]+1)

        self.pad_x = (0, 0, 0, 0, 1, 1) # to pad the last 3 dimensions, use (padding_left, padding_right, padding_top, padding_bottom, padding_top, padding_bottom, padding_front, padding_back)
        self.pad_y = (0, 0, 1, 1, 0, 0)
        self.pad_z = (1, 1, 0, 0, 0, 0)

        self.inner_field = lambda x: x[:, :, self.npml:-self.npml, self.npml:-self.npml, self.npml:-self.npml]

    def update_epsr_sige(self, epsrx, epsry, epsrz, sigex, sigey, sigez):
        self.epsrx, self.epsry, self.epsrz = epsrx, epsry, epsrz
        self.sigex, self.sigey, self.sigez = sigex, sigey, sigez

    def update_mur_sigm(self, murx, mury, murz, sigmx, sigmy, sigmz):
        self.murx, self.mury, self.murz = murx, mury, murz
        self.sigmx, self.sigmy, self.sigmz = sigmx, sigmy, sigmz

    def record_current_field(self, t, ex, ey, ez, hx, hy, hz):
        self.curnt_t = t
        self.curnt_ex, self.curnt_ey, self.curnt_ez = ex, ey, ez
        self.curnt_hx, self.curnt_hy, self.curnt_hz = hx, hy, hz
    
    def left_shift_prev_field(self):
        self.prev_t = self.curnt_t
        self.prev_ex, self.prev_ey, self.prev_ez = self.curnt_ex, self.curnt_ey, self.curnt_ez
        self.prev_hx, self.prev_hy, self.prev_hz = self.curnt_hx, self.curnt_hy, self.curnt_hz
    
    def plot_error_map(self, lx, rx, dx, name):

        lx, rx, dx = torch.mean(torch.abs(lx), dim=-1), torch.mean(torch.abs(rx), dim=-1), torch.mean(torch.abs(dx), dim=-1)

        vmin = min(torch.min(lx).item(), torch.min(rx).item())
        vmax = max(torch.max(lx).item(), torch.max(rx).item())

        kw = {
            'vmin': vmin,
            'vmax': vmax,
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
        def plot_contourf(ax, data, title, kw):
            # Set limits of the plot from coord limits
            pc = ax.imshow(data, vmin = kw['vmin'], vmax = kw['vmax'])
            ax.set_title(title)
            return pc
        
        pc = plot_contourf(axes[0], lx.cpu().numpy(), f'Left Part of the Equation', kw)
        pc = plot_contourf(axes[1], rx.cpu().numpy(), f'Right Part of the Equation', kw)
        pc = plot_contourf(axes[2], dx.cpu().numpy(), f'Residual of the Equation', kw)

        fig.suptitle(f'Absolute Values of Maxwell Equations of {name.upper()}')
        fig.colorbar(pc, ax=axes, location='bottom')
        plt.savefig(f"error_map_{name}_{int(self.prev_t)}_{int(self.curnt_t)}.png")
        plt.savefig(f"error_map_{name}_{int(self.prev_t)}_{int(self.curnt_t)}.pdf")
        plt.close()

    def calculate_gradient(self, t, ex, ey, ez, hx, hy, hz, jx, jy, jz):
        if t == 0:
            self.record_current_field(t, ex, ey, ez, hx, hy, hz)
            return 
        elif t in self.sampled_t:
            self.left_shift_prev_field()
            self.record_current_field(t, ex, ey, ez, hx, hy, hz)
            delta_t = self.curnt_t - self.prev_t

            dezdy = (self.curnt_ez[:, :, :, 1:, :] - self.curnt_ez[:, :, :, :-1, :]) / self.dy
            deydz = (self.curnt_ey[:, :, :, :, 1:] - self.curnt_ey[:, :, :, :, :-1]) / self.dz
            lx = - self.sigmx * self.curnt_hx - self.murx * (self.curnt_hx - self.prev_hx) / self.dtm / delta_t
            rx = dezdy - deydz
            dmx = lx - rx
            dmx = self.inner_field(dmx)

            dexdz = (self.curnt_ex[:, :, :, :, 1:] - self.curnt_ex[:, :, :, :, :-1]) / self.dz
            dezdx = (self.curnt_ez[:, :, 1:, :, :] - self.curnt_ez[:, :, :-1, :, :]) / self.dx
            ly = - self.sigmy * self.curnt_hy - self.mury * (self.curnt_hy - self.prev_hy) / self.dtm / delta_t
            ry = dexdz - dezdx
            dmy = ly - ry
            dmy = self.inner_field(dmy)

            deydx = (self.curnt_ey[:, :, 1:, :, :] - self.curnt_ey[:, :, :-1, :, :]) / self.dx
            dexdy = (self.curnt_ex[:, :, :, 1:, :] - self.curnt_ex[:, :, :, :-1, :]) / self.dy
            lz = - self.sigmz * self.curnt_hz - self.murz * (self.curnt_hz - self.prev_hz) / self.dtm / delta_t
            rz = deydx - dexdy
            dmz = lz - rz
            dmz = self.inner_field(dmz)

            # if t == self.nt // 2:
            if True:
                lx, ly, lz = self.inner_field(lx), self.inner_field(ly), self.inner_field(lz)
                rx, ry, rz = self.inner_field(rx), self.inner_field(ry), self.inner_field(rz)  
                self.plot_error_map(lx[0,0], rx[0,0], dmx[0,0], "hx")
                self.plot_error_map(ly[0,0], ry[0,0], dmy[0,0], "hy")
                self.plot_error_map(lz[0,0], rz[0,0], dmz[0,0], "hz")

            dhzdy = F.pad(self.curnt_hz[:, :, :, 1:, :] - self.curnt_hz[:, :, :, :-1, :], self.pad_y) / self.dy
            dhydz = F.pad(self.curnt_hy[:, :, :, :, 1:] - self.curnt_hy[:, :, :, :, :-1], self.pad_z) / self.dz
            lx = self.sigex * self.curnt_ex + self.epsrx * (self.curnt_ex - self.prev_ex) / self.dte / delta_t
            rx = dhzdy - dhydz - jx / self.dx
            dex = lx - rx
            dex = self.inner_field(dex)

            dhxdz = F.pad(self.curnt_hx[:, :, :, :, 1:] - self.curnt_hx[:, :, :, :, :-1], self.pad_z) / self.dz
            dhzdx = F.pad(self.curnt_hz[:, :, 1:, :, :] - self.curnt_hz[:, :, :-1, :, :], self.pad_x) / self.dx
            ly = self.sigey * self.curnt_ey + self.epsry * (self.curnt_ey - self.prev_ey) / self.dte / delta_t
            ry = dhxdz - dhzdx - jy / self.dy
            dey = ly - ry
            dey = self.inner_field(dey)

            dhydx = F.pad(self.curnt_hy[:, :, 1:, :, :] - self.curnt_hy[:, :, :-1, :, :], self.pad_x) / self.dx
            dhxdy = F.pad(self.curnt_hx[:, :, :, 1:, :] - self.curnt_hx[:, :, :, :-1, :], self.pad_y) / self.dy
            lz = self.sigez * self.curnt_ez + self.epsrz * (self.curnt_ez - self.prev_ez) / self.dte / delta_t
            rz = dhydx - dhxdy - jz / self.dz
            dez = lz - rz 
            dez = self.inner_field(dez)

            
            # mean_lx, mean_ly, mean_lz = torch.mean(torch.abs(lx)).item(), torch.mean(torch.abs(ly)).item(), torch.mean(torch.abs(lz)).item()
            # print(f"mean_lx: {mean_lx:.4e}, mean_ly: {mean_ly:.4e}, mean_lz: {mean_lz:.4e}")
            # mean_rx, mean_ry, mean_rz = torch.mean(torch.abs(rx)).item(), torch.mean(torch.abs(ry)).item(), torch.mean(torch.abs(rz)).item()
            # print(f"mean_rx: {mean_rx:.4e}, mean_ry: {mean_ry:.4e}, mean_rz: {mean_rz:.4e}")        

            mean_dex, mean_dey, mean_dez = torch.mean(torch.abs(dex)).item(), torch.mean(torch.abs(dey)).item(), torch.mean(torch.abs(dez)).item()
            # print(f"mean_dex: {mean_dex:.4e}, mean_dey: {mean_dey:.4e}, mean_dez: {mean_dez:.4e}")
            
            max_dex, max_dey, max_dez = torch.max(torch.abs(dex)).item(), torch.max(torch.abs(dey)).item(), torch.max(torch.abs(dez)).item()
            # print(f"max_dex: {max_dex:.4e}, max_dy: {max_dey:.4e}, max_dez: {max_dez:.4e}")


            # if t == self.nt // 2:
            if True:
                lx, ly, lz = self.inner_field(lx), self.inner_field(ly), self.inner_field(lz)
                rx, ry, rz = self.inner_field(rx), self.inner_field(ry), self.inner_field(rz)  
                self.plot_error_map(lx[0,0], rx[0,0], dex[0,0], "ex")
                self.plot_error_map(ly[0,0], ry[0,0], dey[0,0], "ey")
                self.plot_error_map(lz[0,0], rz[0,0], dez[0,0], "ez")
            return
        else:
            return
    
