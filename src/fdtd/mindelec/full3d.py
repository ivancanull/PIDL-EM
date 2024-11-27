# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=W0613
"""
3D Differentiable FDTD
"""
# from mindspore import nn, ops
import os
from typing import List, Sequence, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .record import SourceRecorder
from .visualizer import *
from .record import *
from .constants import epsilon0, mu0
from .utils import *
from .gradient_checker import *

def get_random_indices(start, end, num, rtype="torch"):
    """
    Create random indices from start to end.
    """
    if rtype == "torch":
        return torch.tensor(np.sort(np.random.choice(range(start, end), num, replace=False)))
    elif rtype == "list":
        return np.sort(np.random.choice(range(start, end), num, replace=False)).tolist()
    else:
        raise ValueError(f"Invalid return type: {rtype}")

class FDTDLayer():
    """
    One-step 3D FDTD.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        cpmlx_e (Tensor): Updating coefficients for electric fields in the x-direction CPML.
        cpmlx_m (Tensor): Updating coefficients for magnetic fields in the x-direction CPML.
        cpmly_e (Tensor): Updating coefficients for electric fields in the y-direction CPML.
        cpmly_m (Tensor): Updating coefficients for magnetic fields in the y-direction CPML.
        cpmlz_e (Tensor): Updating coefficients for electric fields in the z-direction CPML.
        cpmlz_m (Tensor): Updating coefficients for magnetic fields in the z-direction CPML.
    """

    def __init__(self,
                 cell_lengths,
                 cpmlx_e, cpmlx_m,
                 cpmly_e, cpmly_m,
                 cpmlz_e, cpmlz_m,
                 device,
                 ):
        super(FDTDLayer, self).__init__()

        dx = cell_lengths[0]
        dy = cell_lengths[1]
        dz = cell_lengths[2]

        self.cpmlx_e = cpmlx_e
        self.cpmlx_m = cpmlx_m
        self.cpmly_e = cpmly_e
        self.cpmly_m = cpmly_m
        self.cpmlz_e = cpmlz_e
        self.cpmlz_m = cpmlz_m
        self.device = device

        # operators
        self.dx_wghts = (tensor([-1., 1.]).reshape((1, 1, 2, 1, 1)) / dx).to(self.device)
        self.dy_wghts = (tensor([-1., 1.]).reshape((1, 1, 1, 2, 1)) / dy).to(self.device)
        self.dz_wghts = (tensor([-1., 1.]).reshape((1, 1, 1, 1, 2)) / dz).to(self.device)

        self.pad_x = (0, 0, 0, 0, 1, 1) # to pad the last 3 dimensions, use (padding_left, padding_right, padding_top, padding_bottom, padding_top, padding_bottom, padding_front, padding_back)
        self.pad_y = (0, 0, 1, 1, 0, 0)
        self.pad_z = (1, 1, 0, 0, 0, 0)

        # Padding Implementation on MindSpore
        # self.pad_x = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
        # self.pad_y = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
        # self.pad_z = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0), (1, 1)))

    def __call__(self, jx_t, jy_t, jz_t,
                 ex, ey, ez, hx, hy, hz,
                 pexy, pexz, peyx, peyz, pezx, pezy,
                 phxy, phxz, phyx, phyz, phzx, phzy,
                 cexe, cexh, ceye, ceyh, ceze, cezh,
                 chxh, chxe, chyh, chye, chzh, chze):
        """One-step forward propagation

        Args:
            jx_t (Tensor): Source at time t + 0.5 * dt.
            jy_t (Tensor): Source at time t + 0.5 * dt.
            jz_t (Tensor): Source at time t + 0.5 * dt.
            ex, ey, ez, hx, hy, hz (Tensor): E and H fields.
            pexy, pexz, peyx, peyz, pezx, pezy (Tensor): CPML auxiliary fields.
            phxy, phxz, phyx, phyz, phzx, phzy (Tensor): CPML auxiliary fields.
            cexe, cexh, ceye, ceyh, ceze, cezh (Tensor): Updating coefficients.
            chxh, chxe, chyh, chye, chzh, chze (Tensor): Updating coefficients.

        Returns:
            hidden_states (tuple)
        """

        # -------------------------------------------------
        # Step 1: Update H's at n+1/2 step
        # -------------------------------------------------
        # compute curl E
        deydx = F.conv3d(ey, self.dx_wghts) / self.cpmlx_m[2]
        dezdx = F.conv3d(ez, self.dx_wghts) / self.cpmlx_m[2]
        dexdy = F.conv3d(ex, self.dy_wghts) / self.cpmly_m[2]
        dezdy = F.conv3d(ez, self.dy_wghts) / self.cpmly_m[2]
        dexdz = F.conv3d(ex, self.dz_wghts) / self.cpmlz_m[2]
        deydz = F.conv3d(ey, self.dz_wghts) / self.cpmlz_m[2]
        
        # previous implementation on Mindspore
        # deydx = self.dx_oper(ey, self.dx_wghts) / self.cpmlx_m[2]
        # dezdx = self.dx_oper(ez, self.dx_wghts) / self.cpmlx_m[2]
        # dexdy = self.dy_oper(ex, self.dy_wghts) / self.cpmly_m[2]
        # dezdy = self.dy_oper(ez, self.dy_wghts) / self.cpmly_m[2]
        # dexdz = self.dz_oper(ex, self.dz_wghts) / self.cpmlz_m[2]
        # deydz = self.dz_oper(ey, self.dz_wghts) / self.cpmlz_m[2]

        # update auxiliary fields in CFS-PML
        phyx = self.cpmlx_m[0] * phyx + self.cpmlx_m[1] * dezdx
        phzx = self.cpmlx_m[0] * phzx + self.cpmlx_m[1] * deydx
        phxy = self.cpmly_m[0] * phxy + self.cpmly_m[1] * dezdy
        phzy = self.cpmly_m[0] * phzy + self.cpmly_m[1] * dexdy
        phxz = self.cpmlz_m[0] * phxz + self.cpmlz_m[1] * deydz
        phyz = self.cpmlz_m[0] * phyz + self.cpmlz_m[1] * dexdz

        # update H
        hx = chxh * hx + chxe * ((deydz + phxz) - (dezdy + phxy))
        hy = chyh * hy + chye * ((dezdx + phyx) - (dexdz + phyz))
        hz = chzh * hz + chze * ((dexdy + phzy) - (deydx + phzx))

        # -------------------------------------------------
        # Step 2: Update E's at n+1 step
        # -------------------------------------------------
        # compute curl H
        dhydx = F.pad(F.conv3d(hy, self.dx_wghts), self.pad_x) / self.cpmlx_e[2]
        dhzdx = F.pad(F.conv3d(hz, self.dx_wghts), self.pad_x) / self.cpmlx_e[2]
        dhxdy = F.pad(F.conv3d(hx, self.dy_wghts), self.pad_y) / self.cpmly_e[2]
        dhzdy = F.pad(F.conv3d(hz, self.dy_wghts), self.pad_y) / self.cpmly_e[2]
        dhxdz = F.pad(F.conv3d(hx, self.dz_wghts), self.pad_z) / self.cpmlz_e[2]
        dhydz = F.pad(F.conv3d(hy, self.dz_wghts), self.pad_z) / self.cpmlz_e[2]

        # previous implementation on Mindspore
        # dhydx = self.pad_x(self.dx_oper(hy, self.dx_wghts)) / self.cpmlx_e[2]
        # dhzdx = self.pad_x(self.dx_oper(hz, self.dx_wghts)) / self.cpmlx_e[2]
        # dhxdy = self.pad_y(self.dy_oper(hx, self.dy_wghts)) / self.cpmly_e[2]
        # dhzdy = self.pad_y(self.dy_oper(hz, self.dy_wghts)) / self.cpmly_e[2]
        # dhxdz = self.pad_z(self.dz_oper(hx, self.dz_wghts)) / self.cpmlz_e[2]
        # dhydz = self.pad_z(self.dz_oper(hy, self.dz_wghts)) / self.cpmlz_e[2]

        # update auxiliary fields in CFS-PML
        peyx = self.cpmlx_e[0] * peyx + self.cpmlx_e[1] * dhzdx
        pezx = self.cpmlx_e[0] * pezx + self.cpmlx_e[1] * dhydx
        pexy = self.cpmly_e[0] * pexy + self.cpmly_e[1] * dhzdy
        pezy = self.cpmly_e[0] * pezy + self.cpmly_e[1] * dhxdy
        pexz = self.cpmlz_e[0] * pexz + self.cpmlz_e[1] * dhydz
        peyz = self.cpmlz_e[0] * peyz + self.cpmlz_e[1] * dhxdz

        # update E
        ex = cexe * ex + cexh * ((dhzdy + pexy) - (dhydz + pexz) - jx_t)
        ey = ceye * ey + ceyh * ((dhxdz + peyz) - (dhzdx + peyx) - jy_t)
        ez = ceze * ez + cezh * ((dhydx + pezx) - (dhxdy + pezy) - jz_t)

        hidden_states = (ex, ey, ez, hx, hy, hz, pexy, pexz, peyx,
                         peyz, pezx, pezy, phxy, phxz, phyx, phyz, phzx, phzy)

        return hidden_states

class ADFDTD:
    """3D Differentiable FDTD Network.

    Args:
        cell_numbers (tuple): Number of Yee cells in (x, y, z) directions.
        cell_lengths (tuple): Lengths of Yee cells.
        nt (int): Number of time steps.
        dt (float): Time interval.
        ns (int): Number of sources.
        designer (BaseTopologyDesigner): Customized Topology designer.
        cfs_pml (CFSParameters): CFS parameter class.
        init_weights (Tensor): Initial weights, Default: ``None``.

    Returns:
        outputs (Tensor): Customized outputs.
    """

    def __init__(self,
                 cell_numbers,
                 cell_lengths,
                 nt, dt,
                 ns,
                 designer,
                 cfs_pml,
                 source_record_sampler:SourceRecorder | None = None,
                 field_record_sampler: FieldRecorder | None = None,
                 coefficient_recorder: CoefficientRecorder | None = None,
                 visualizer: Visualizer | None = None,
                 device=torch.device("cpu"),
                 check_gradient=False,
                 ):

        self.nx = cell_numbers[0]
        self.ny = cell_numbers[1]
        self.nz = cell_numbers[2]
        self.dx = cell_lengths[0]
        self.dy = cell_lengths[1]
        self.dz = cell_lengths[2]
        self.nt = nt
        self.ns = ns
        self.dt = tensor(dt)
        self.device = device
        self.designer = designer
        self.cfs_pml = cfs_pml
        self.check_gradient = check_gradient

        # auxiliary variables
        self.dte = tensor(dt / epsilon0).to(self.device)
        self.dtm = tensor(dt / mu0).to(self.device)
        
        if self.check_gradient:
            self.gradient_checker = GradientChecker(
                cell_numbers, self.cfs_pml.npml, self.dx, self.dy, self.dz, self.dte, self.dtm, nt, dt=10)
    
        self.source_record_sampler = source_record_sampler
        if self.source_record_sampler is not None:
            self.source_record_sampler.initiate_sources(nt, cell_numbers, self.cfs_pml.npml)

        self.field_record_sampler = field_record_sampler
        if self.field_record_sampler is not None:
            self.field_record_sampler.initiate_fields(nt, cell_numbers, self.cfs_pml.npml)
        
        self.coefficient_recorder = coefficient_recorder
        if self.coefficient_recorder is not None:
            self.coefficient_recorder.initiate_coefficients(self.cfs_pml.npml)

        self.visualizer = visualizer
       
        self.mur = tensor(1.)
        self.sigm = tensor(0.)

        if self.cfs_pml is not None:
            # CFS-PML Coefficients
            cpmlx_e, cpmlx_m = self.cfs_pml.get_update_coefficients(
                self.nx, self.dx, self.dt, self.designer.background_epsr)
            cpmly_e, cpmly_m = self.cfs_pml.get_update_coefficients(
                self.ny, self.dy, self.dt, self.designer.background_epsr)
            cpmlz_e, cpmlz_m = self.cfs_pml.get_update_coefficients(
                self.nz, self.dz, self.dt, self.designer.background_epsr)

            cpmlx_e = tensor(cpmlx_e.reshape((3, 1, 1, -1, 1, 1)))
            cpmlx_m = tensor(cpmlx_m.reshape((3, 1, 1, -1, 1, 1)))
            cpmly_e = tensor(cpmly_e.reshape((3, 1, 1, 1, -1, 1)))
            cpmly_m = tensor(cpmly_m.reshape((3, 1, 1, 1, -1, 1)))
            cpmlz_e = tensor(cpmlz_e.reshape((3, 1, 1, 1, 1, -1)))
            cpmlz_m = tensor(cpmlz_m.reshape((3, 1, 1, 1, 1, -1)))

        else:
            # PEC boundary
            cpmlx_e = cpmlx_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmly_e = cpmly_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmlz_e = cpmlz_m = tensor([0., 0., 1.]).reshape((3, 1))

        # To Device
        cpmlx_e, cpmlx_m, cpmly_e, cpmly_m, cpmlz_e, cpmlz_m = \
            cpmlx_e.to(self.device), cpmlx_m.to(self.device), cpmly_e.to(self.device), \
            cpmly_m.to(self.device), cpmlz_e.to(self.device), cpmlz_m.to(self.device)

        # FDTD layer
        self.fdtd_layer = FDTDLayer(
            cell_lengths, cpmlx_e, cpmlx_m, cpmly_e, cpmly_m,
            cpmlz_e, cpmlz_m, self.device)

        # define material parameters smoother
        self.smooth_yz = 0.25 * ones((1, 1, 1, 2, 2)).to(self.device)
        self.smooth_xz = 0.25 * ones((1, 1, 2, 1, 2)).to(self.device)
        self.smooth_xy = 0.25 * ones((1, 1, 2, 2, 1)).to(self.device)

        self.pad_yz = (1, 1, 1, 1, 0, 0) # to pad the last 3 dimensions, use (padding_left, padding_right, padding_top, padding_bottom, padding_top, padding_bottom, padding_front, padding_back)
        self.pad_xz = (1, 1, 0, 0, 1, 1)
        self.pad_xy = (0, 0, 1, 1, 1, 1)

        # previous implementation on Mindspore
        # self.smooth_yz_oper = nn.Conv3d(
        #     in_channel=1, out_channel=1, kernel_size=(1, 2, 2), pad_mode='pad', pad=(0, 0, 1, 1, 1, 1))
        # self.smooth_xz_oper = nn.Conv3D(
        #     in_channel=1, out_channel=1, kernel_size=(2, 1, 2), pad_mode='pad', pad=(1, 1, 0, 0, 1, 1))
        # self.smooth_xy_oper = nn.Conv3D(
        #     in_channel=1, out_channel=1, kernel_size=(2, 2, 1), pad_mode='pad', pad=(1, 1, 1, 1, 0, 0))

    def __call__(self, waveform_t, time_estimation):
        """
        ADFDTD-based forward propagation.

        Args:
            waveform_t (Tensor, shape=(nt,)): Time-domain waveforms.

        Returns:
            outputs (Tensor): Customized outputs.
        """
        # ----------------------------------------
        # Initialization
        # ----------------------------------------
        # constants
        nx, ny, nz, ns, nt = self.nx, self.ny, self.nz, self.ns, self.nt
        dt = self.dt

        epsr, sige = self.designer.generate_object()
        epsr, sige = epsr.to(self.device), sige.to(self.device)

        # delectric smoothing
        epsrx = F.conv3d(F.pad(epsr[None, None], self.pad_yz), self.smooth_yz) # (1, 1, x, y+1, z+1) same as ex
        sigex = F.conv3d(F.pad(sige[None, None], self.pad_yz), self.smooth_yz) # (1, 1, x, y+1, z+1)
        epsry = F.conv3d(F.pad(epsr[None, None], self.pad_xz), self.smooth_xz) # (1, 1, x+1, y, z+1) same as ey
        sigey = F.conv3d(F.pad(sige[None, None], self.pad_xz), self.smooth_xz) # (1, 1, x+1, y, z+1)
        epsrz = F.conv3d(F.pad(epsr[None, None], self.pad_xy), self.smooth_xy) # (1, 1, x+1, y+1, z) same as ez
        sigez = F.conv3d(F.pad(sige[None, None], self.pad_xy), self.smooth_xy) # (1, 1, x+1, y+1, z)

        (epsrx, epsry, epsrz, sigex, sigey, sigez) = self.designer.modify_object(
            (epsrx, epsry, epsrz, sigex, sigey, sigez)
        )

        # non-magnetic & magnetically lossless material
        murx = mury = murz = self.mur
        sigmx = sigmy = sigmz = self.sigm

        # updating coefficients
        cexe, cexh = fcmpt(self.dte, epsrx, sigex)
        ceye, ceyh = fcmpt(self.dte, epsry, sigey)
        ceze, cezh = fcmpt(self.dte, epsrz, sigez)
        chxh, chxe = fcmpt(self.dtm, murx, sigmx)
        chyh, chye = fcmpt(self.dtm, mury, sigmy)
        chzh, chze = fcmpt(self.dtm, murz, sigmz)

        # record epsr, mur, sige, sigm
        if self.check_gradient:
            self.gradient_checker.update_epsr_sige(epsrx, epsry, epsrz, sigex, sigey, sigez)
            self.gradient_checker.update_mur_sigm(murx, mury, murz, sigmx, sigmy, sigmz)

        if self.coefficient_recorder is not None:
            self.coefficient_recorder.update(epsrx, epsry, epsrz, sigex, sigey, sigez, murx, mury, murz, sigmx, sigmy, sigmz)

        # hidden states
        ex = torch.zeros((ns, 1, nx, ny + 1, nz + 1)).to(self.device)
        ey = torch.zeros((ns, 1, nx + 1, ny, nz + 1)).to(self.device)
        ez = torch.zeros((ns, 1, nx + 1, ny + 1, nz)).to(self.device)
        hx = torch.zeros((ns, 1, nx + 1, ny, nz)).to(self.device)
        hy = torch.zeros((ns, 1, nx, ny + 1, nz)).to(self.device)
        hz = torch.zeros((ns, 1, nx, ny, nz + 1)).to(self.device)

        # CFS-PML auxiliary fields
        pexy = torch.zeros_like(ex).to(self.device)
        pexz = torch.zeros_like(ex).to(self.device)
        peyx = torch.zeros_like(ey).to(self.device)
        peyz = torch.zeros_like(ey).to(self.device)
        pezx = torch.zeros_like(ez).to(self.device)
        pezy = torch.zeros_like(ez).to(self.device)
        phxy = torch.zeros_like(hx).to(self.device)
        phxz = torch.zeros_like(hx).to(self.device)
        phyx = torch.zeros_like(hy).to(self.device)
        phyz = torch.zeros_like(hy).to(self.device)
        phzx = torch.zeros_like(hz).to(self.device)
        phzy = torch.zeros_like(hz).to(self.device)

        # set source location
        jx_t = torch.zeros_like(ex).to(self.device)
        jy_t = torch.zeros_like(ey).to(self.device)
        jz_t = torch.zeros_like(ez).to(self.device)

        # ----------------------------------------
        # Update
        # ----------------------------------------
        outputs = []

        if time_estimation:
            iterator = tqdm(range(nt))
        else:
            iterator = range(nt)

        for t in iterator:
            jx_t, jy_t, jz_t = self.designer.update_sources(
                (jx_t, jy_t, jz_t), (ex, ey, ez),
                waveform_t[t], dt)

            # RNN-Style Update
            ex, ey, ez, hx, hy, hz, \
            pexy, pexz, peyx, peyz, pezx, pezy, \
            phxy, phxz, phyx, phyz, phzx, phzy = \
                self.fdtd_layer(jx_t, jy_t, jz_t,
                                ex, ey, ez, hx, hy, hz,
                                pexy, pexz, peyx, peyz, pezx, pezy,
                                phxy, phxz, phyx, phyz, phzx, phzy,
                                cexe, cexh, ceye, ceyh, ceze, cezh,
                                chxh, chxe, chyh, chye, chzh, chze)
            
            # self.field_recorder.update(t, ex, ey, ez, hx, hy, hz)
            if self.field_record_sampler is not None:
                self.field_record_sampler.update(t, ex, ey, ez, hx, hy, hz)

            if self.source_record_sampler is not None:
                self.source_record_sampler.update(t, jx_t, jy_t, jz_t)
            
            # check the gradietnts
            if self.check_gradient:
                self.gradient_checker.calculate_gradient(t, ex, ey, ez, hx, hy, hz, jx_t, jy_t, jz_t)
            
            # compute outputs
            outputs.append(self.designer.get_outputs_at_each_step(
                (ex, ey, ez, hx, hy, hz)))
            # print(f"iter: {t} / {nt}")
        
        if self.visualizer is not None:
            self.visualizer.plot_max_map()
            self.visualizer.plot_min_map()
            self.visualizer.plot_max_animation()
            self.visualizer.plot_min_animation()

        outputs = torch.stack(outputs, dim=0) # (nt, ns, nr, 2)
        return outputs
