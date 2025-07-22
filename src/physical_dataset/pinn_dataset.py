import torch
import pandas as pd
import numpy as np
from typing import Sequence, List
import torch.nn.functional as F

class CustomPinnDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 air_buffers: Sequence[int],
                 device: torch.device,
                 field_target: str,
                 img_input: bool = False,
                 ):
        self.indices = []
        for i in dataset.index: # debug for example
            l1 = dataset.loc[i, 'l1']
            l2 = dataset.loc[i, 'l2']
            d = dataset.loc[i, 'd']
            self.indices.append(f"{l1}_{l2}_{d}")
        self.air_buffers = air_buffers
        self.device = device
        self.img_input = img_input
        self.field_target = field_target
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        label = self.indices[index]
        surfaces = np.load(f'./result/surfaces/{label}_surfaces.npz')["img"][None, :, :] # X: [1, nx, ny] 
        fields = np.load(f'./result/fields/{label}_fields.npz')

        # only read jz because in the previous setting, only jz is recorded
        # only keep the jz at height = air_buffers[2]
        sources = np.load(f'./result/surfaces/{label}_sources.npz')["jz"][:,self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]]  # X: [nt, nx, ny, nz]
        sources = torch.tensor(sources, dtype=torch.float32).to(self.device)[:, None, ...]


        # repeat surfaces at dim = 0 and stack with sources at dim = 1
        if not self.img_input:
            surfaces = np.repeat(surfaces, sources.shape[0], axis=0) # [nt, nx, ny]
            surfaces = np.stack([surfaces, sources / 16000], axis=1) # [nt, 2, nx, ny]
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device)
        else:
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device) # surfaces[1, nx, ny]
        
        # convert to torch.tensor first for padding
        if self.field_target == "all":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ex", "ey", "ez", "hx", "hy", "hz"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        elif self.field_target == "ezhxhy":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ez", "hx", "hy"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        else:
            field = torch.tensor(fields[self.field_target], dtype=torch.float32).to(self.device)
            # pad to align the fields and remove air buffers
            field = self.pad_fields(field, self.field_target) # [nt, nz, nx, ny]
            # field = self.sample_time(field)

        # load coefficients
        coefficients = np.load(f'./result/coefficients/{label}_coefficients.npz')
        # print(f"epsrx: {coefficients['epsrx'].shape}") # (nx - 1, ny, nz)
        # print(f"epsry: {coefficients['epsry'].shape}") # (nx, ny - 1, nz)
        # print(f"epsrz: {coefficients['epsrz'].shape}") # (nx, ny, nz - 1)
        # print(f"sigex: {coefficients['sigex'].shape}") # (nx - 1, ny, nz)
        # print(f"sigey: {coefficients['sigey'].shape}") # (nx, ny - 1, nz)
         #print(f"sigez: {coefficients['sigez'].shape}") # (nx, ny, nz - 1)
        coefficients = torch.concat([self.pad_fields(torch.tensor(coefficients[target], dtype=torch.float32).to(self.device), target) for target in ["epsrx", "epsry", "epsrz", "sigex", "sigey", "sigez"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        # coefficients = coefficients.swapaxes(-1, -3).swapaxes(-1, -2) # [6, nz, nx, ny]

        # print(f"coefficients: {coefficients.shape}")

        return surfaces, field, sources, coefficients
    
    def pad_fields(self,
                   field: torch.Tensor,
                   target: str):
        if target == "ex" or target == "sigex" or target == "epsrx":
            

            field = F.pad(field, (0, 0, 0, 0, 0, 1), mode='constant', value=0) 
            if target == "ex":
                field = field / 100 # temporal normalization
        elif target == "ey" or target == "sigey" or target == "epsry": 
            field = F.pad(field, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            if target == "ey":
                field = field / 100
        elif target == "ez" or target == "sigez"or target == "epsrz":
            field = F.pad(field, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            if target == "ez":
                field = field / 100
        elif target == "hx":
            field = F.pad(field, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
        elif target == "hy":
            field = F.pad(field, (0, 1, 0, 0, 0, 1), mode='constant', value=0)
        elif target == "hz":
            field = F.pad(field, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        else:
            raise ValueError('Unknown field target.')
        
        # remove air buffers
        return field[..., self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]:-self.air_buffers[2]].swapaxes(-1, -3).swapaxes(-1, -2)

    def get_max_min(self):
        max_min = {}
        for index in range(len(self)):
            label = self.indices[index]
            fields = np.load(f'./result/fields/{label}_fields.npz')
            for target in ["ex", "ey", "ez", "hx", "hy", "hz"]:
                if target not in max_min:
                    max_min[target] = [fields[target].max(), fields[target].min()]
                else:
                    max_min[target][0] = max(max_min[target][0], fields[target].max())
                    max_min[target][1] = min(max_min[target][1], fields[target].min())
        return max_min
    
    
    # def sample_time(self,
    #                 field: torch.Tensor):
    #     sampled_indices = np.linspace(0, field.shape[0]-1, SAMPLE_TIME).astype(int)
    #     return field[:, sampled_indices, ...]
    
class AntennaPinnDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 air_buffers: Sequence[int],
                 device: torch.device,
                 field_target: str,
                 img_input: bool = False,
                 ):
        self.indices = []
        for i in dataset.index: # debug for example
            l1 = dataset.loc[i, 'l1']
            l2 = dataset.loc[i, 'l2']
            l3 = dataset.loc[i, 'l3']
            l4 = dataset.loc[i, 'l4']
            self.indices.append(f"{l1}_{l2}_{l3}_{l4}")
        self.air_buffers = air_buffers
        self.device = device
        self.img_input = img_input
        self.field_target = field_target
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        label = self.indices[index]
        surfaces = np.load(f'./result/surfaces/{label}_surfaces.npz')["img"][None, :, :] # X: [1, nx, ny] 
        fields = np.load(f'./result/fields/{label}_fields.npz')

        # only read jz because in the previous setting, only jz is recorded
        # only keep the jz at height = air_buffers[2]
        sources = np.load(f'./result/surfaces/{label}_sources.npz')["jz"][:,self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]]  # X: [nt, nx, ny, nz]
        sources = torch.tensor(sources, dtype=torch.float32).to(self.device)[:, None, ...]


        # repeat surfaces at dim = 0 and stack with sources at dim = 1
        if not self.img_input:
            surfaces = np.repeat(surfaces, sources.shape[0], axis=0) # [nt, nx, ny]
            surfaces = np.stack([surfaces, sources / 16000], axis=1) # [nt, 2, nx, ny]
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device)
        else:
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device) # surfaces[1, nx, ny]
        
        # convert to torch.tensor first for padding
        if self.field_target == "all":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ex", "ey", "ez", "hx", "hy", "hz"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        elif self.field_target == "ezhxhy":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ez", "hx", "hy"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        else:
            field = torch.tensor(fields[self.field_target], dtype=torch.float32).to(self.device)
            # pad to align the fields and remove air buffers
            field = self.pad_fields(field, self.field_target) # [nt, nz, nx, ny]
            # field = self.sample_time(field)

        # load coefficients
        coefficients = np.load(f'./result/coefficients/{label}_coefficients.npz')
        # print(f"epsrx: {coefficients['epsrx'].shape}") # (nx - 1, ny, nz)
        # print(f"epsry: {coefficients['epsry'].shape}") # (nx, ny - 1, nz)
        # print(f"epsrz: {coefficients['epsrz'].shape}") # (nx, ny, nz - 1)
        # print(f"sigex: {coefficients['sigex'].shape}") # (nx - 1, ny, nz)
        # print(f"sigey: {coefficients['sigey'].shape}") # (nx, ny - 1, nz)
         #print(f"sigez: {coefficients['sigez'].shape}") # (nx, ny, nz - 1)
        coefficients = torch.concat([self.pad_fields(torch.tensor(coefficients[target], dtype=torch.float32).to(self.device), target) for target in ["epsrx", "epsry", "epsrz", "sigex", "sigey", "sigez"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        # coefficients = coefficients.swapaxes(-1, -3).swapaxes(-1, -2) # [6, nz, nx, ny]

        # print(f"coefficients: {coefficients.shape}")

        return surfaces, field, sources, coefficients
    
    def pad_fields(self,
                   field: torch.Tensor,
                   target: str):
        if target == "ex" or target == "sigex" or target == "epsrx":
            field = F.pad(field, (0, 0, 0, 0, 0, 1), mode='constant', value=0) 
            if target == "ex":
                field = field / 100 # temporal normalization
        elif target == "ey" or target == "sigey" or target == "epsry": 
            field = F.pad(field, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            if target == "ey":
                field = field / 100
        elif target == "ez" or target == "sigez"or target == "epsrz":
            field = F.pad(field, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            if target == "ez":
                field = field / 100
            
        elif target == "hx":
            field = F.pad(field, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
        elif target == "hy":
            field = F.pad(field, (0, 1, 0, 0, 0, 1), mode='constant', value=0)
        elif target == "hz":
            field = F.pad(field, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        else:
            raise ValueError('Unknown field target.')
        
        # remove air buffers
        return field[..., self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]:-self.air_buffers[2]].swapaxes(-1, -3).swapaxes(-1, -2)

    def get_max_min(self):
        max_min = {}
        for index in range(len(self)):
            label = self.indices[index]
            fields = np.load(f'./result/fields/{label}_fields.npz')
            for target in ["ex", "ey", "ez", "hx", "hy", "hz"]:
                if target not in max_min:
                    max_min[target] = [fields[target].max(), fields[target].min()]
                else:
                    max_min[target][0] = max(max_min[target][0], fields[target].max())
                    max_min[target][1] = min(max_min[target][1], fields[target].min())
        return max_min
    
    
    # def sample_time(self,
    #                 field: torch.Tensor):
    #     sampled_indices = np.linspace(0, field.shape[0]-1, SAMPLE_TIME).astype(int)
    #     return field[:, sampled_indices, ...]

class FNOExpDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 air_buffers: Sequence[int],
                 device: torch.device,
                 field_target: str,
                 variables: List[str],
                 seq_len: int = 5,
                 img_input: bool = False,
                 i2f: bool = False,
                 result_dir: str = "./result"
                 ):
        self.indices = []
        for i in dataset.index: # debug for example
            data_comb = ""
            for var in variables:
                data_comb += str(dataset.loc[i, var]) + "_"
            self.indices.append(data_comb[:-1])
        self.air_buffers = air_buffers
        self.device = device
        self.img_input = img_input
        self.field_target = field_target  
        self.seq_len = seq_len  
        self.i2f = i2f
        self.result_dir = result_dir
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        label = self.indices[index]
        surfaces = np.load(f'{self.result_dir}/surfaces/{label}_surfaces.npz')["img"][None, :, :] # X: [1, nx, ny] 
        fields = np.load(f'{self.result_dir}/fields/{label}_fields.npz')

        # only read jz because in the previous setting, only jz is recorded
        # only keep the jz at height = air_buffers[2]
        sources = np.load(f'{self.result_dir}/surfaces/{label}_sources.npz')["jz"][:,self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]]  # X: [nt, nx, ny, nz]
        sources = torch.tensor(sources, dtype=torch.float32).to(self.device)[:, None, ...]

        # repeat surfaces at dim = 0 and stack with sources at dim = 1
        surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device) # surfaces[1, nx, ny]
        
        # convert to torch.tensor first for padding
        if self.field_target == "all":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ex", "ey", "ez", "hx", "hy", "hz"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        elif self.field_target == "ezhxhy":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ez", "hx", "hy"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        else:
            field = torch.tensor(fields[self.field_target], dtype=torch.float32).to(self.device)
            # pad to align the fields and remove air buffers
            field = self.pad_fields(field, self.field_target) # [nt, nz, nx, ny]
            # field = self.sample_time(field)

        # load coefficients
        coefficients = np.load(f'{self.result_dir}/coefficients/{label}_coefficients.npz')
        # print(f"epsrx: {coefficients['epsrx'].shape}") # (nx - 1, ny, nz)
        # print(f"epsry: {coefficients['epsry'].shape}") # (nx, ny - 1, nz)
        # print(f"epsrz: {coefficients['epsrz'].shape}") # (nx, ny, nz - 1)
        # print(f"sigex: {coefficients['sigex'].shape}") # (nx - 1, ny, nz)
        # print(f"sigey: {coefficients['sigey'].shape}") # (nx, ny - 1, nz)
        # print(f"sigez: {coefficients['sigez'].shape}") # (nx, ny, nz - 1)
        coefficients = torch.concat([self.pad_fields(torch.tensor(coefficients[target], dtype=torch.float32).to(self.device), target) for target in ["epsrx", "epsry", "epsrz", "sigex", "sigey", "sigez"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        # coefficients = coefficients.swapaxes(-1, -3).swapaxes(-1, -2) # [6, nz, nx, ny]

        # print(f"coefficients: {coefficients.shape}")

        if self.i2f:
            return surfaces, field
        else:
            return field[0], field[1:1+self.seq_len], sources, coefficients
    
    def pad_fields(self,
                   field: torch.Tensor,
                   target: str):
        if target == "ex" or target == "sigex" or target == "epsrx":
            

            field = F.pad(field, (0, 0, 0, 0, 0, 1), mode='constant', value=0) 
            if target == "ex":
                field = field / 100 # temporal normalization
        elif target == "ey" or target == "sigey" or target == "epsry": 
            field = F.pad(field, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            if target == "ey":
                field = field / 100
        elif target == "ez" or target == "sigez"or target == "epsrz":
            field = F.pad(field, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            if target == "ez":
                field = field / 100
        elif target == "hx":
            field = F.pad(field, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
        elif target == "hy":
            field = F.pad(field, (0, 1, 0, 0, 0, 1), mode='constant', value=0)
        elif target == "hz":
            field = F.pad(field, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        else:
            raise ValueError('Unknown field target.')
        
        # remove air buffers
        return field[..., self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]:-self.air_buffers[2]].swapaxes(-1, -3).swapaxes(-1, -2)

class FNOAntennaDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 air_buffers: Sequence[int],
                 device: torch.device,
                 field_target: str,
                 seq_len: int = 5,
                 img_input: bool = False,
                 i2f: bool = False
                 ):
        self.indices = []
        for i in dataset.index: # debug for example
            l1 = dataset.loc[i, 'l1']
            l2 = dataset.loc[i, 'l2']
            l3 = dataset.loc[i, 'l3']
            l4 = dataset.loc[i, 'l4']
            self.indices.append(f"{l1}_{l2}_{l3}_{l4}")
        self.air_buffers = air_buffers
        self.device = device
        self.img_input = img_input
        self.field_target = field_target  
        self.seq_len = seq_len  
        self.i2f = i2f
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        label = self.indices[index]
        surfaces = np.load(f'./result/surfaces/{label}_surfaces.npz')["img"][None, :, :] # X: [1, nx, ny] 
        fields = np.load(f'./result/fields/{label}_fields.npz')

        # only read jz because in the previous setting, only jz is recorded
        # only keep the jz at height = air_buffers[2]
        sources = np.load(f'./result/surfaces/{label}_sources.npz')["jz"][:,self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]]  # X: [nt, nx, ny, nz]
        sources = torch.tensor(sources, dtype=torch.float32).to(self.device)[:, None, ...]


        # repeat surfaces at dim = 0 and stack with sources at dim = 1
        surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device) # surfaces[1, nx, ny]
        
        # convert to torch.tensor first for padding
        if self.field_target == "all":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ex", "ey", "ez", "hx", "hy", "hz"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        elif self.field_target == "ezhxhy":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ez", "hx", "hy"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        else:
            field = torch.tensor(fields[self.field_target], dtype=torch.float32).to(self.device)
            # pad to align the fields and remove air buffers
            field = self.pad_fields(field, self.field_target) # [nt, nz, nx, ny]
            # field = self.sample_time(field)

        # load coefficients
        coefficients = np.load(f'./result/coefficients/{label}_coefficients.npz')
        # print(f"epsrx: {coefficients['epsrx'].shape}") # (nx - 1, ny, nz)
        # print(f"epsry: {coefficients['epsry'].shape}") # (nx, ny - 1, nz)
        # print(f"epsrz: {coefficients['epsrz'].shape}") # (nx, ny, nz - 1)
        # print(f"sigex: {coefficients['sigex'].shape}") # (nx - 1, ny, nz)
        # print(f"sigey: {coefficients['sigey'].shape}") # (nx, ny - 1, nz)
        # print(f"sigez: {coefficients['sigez'].shape}") # (nx, ny, nz - 1)
        coefficients = torch.concat([self.pad_fields(torch.tensor(coefficients[target], dtype=torch.float32).to(self.device), target) for target in ["epsrx", "epsry", "epsrz", "sigex", "sigey", "sigez"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        # coefficients = coefficients.swapaxes(-1, -3).swapaxes(-1, -2) # [6, nz, nx, ny]

        # print(f"coefficients: {coefficients.shape}")

        if self.i2f:
            return surfaces, field
        else:
            return field[0], field[1:1+self.seq_len], sources, coefficients
    
    def pad_fields(self,
                   field: torch.Tensor,
                   target: str):
        if target == "ex" or target == "sigex" or target == "epsrx":
            

            field = F.pad(field, (0, 0, 0, 0, 0, 1), mode='constant', value=0) 
            if target == "ex":
                field = field / 100 # temporal normalization
        elif target == "ey" or target == "sigey" or target == "epsry": 
            field = F.pad(field, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            if target == "ey":
                field = field / 100
        elif target == "ez" or target == "sigez"or target == "epsrz":
            field = F.pad(field, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            if target == "ez":
                field = field / 100
        elif target == "hx":
            field = F.pad(field, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
        elif target == "hy":
            field = F.pad(field, (0, 1, 0, 0, 0, 1), mode='constant', value=0)
        elif target == "hz":
            field = F.pad(field, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        else:
            raise ValueError('Unknown field target.')
        
        # remove air buffers
        return field[..., self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]:-self.air_buffers[2]].swapaxes(-1, -3).swapaxes(-1, -2)

class PinnDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: pd.DataFrame,
                 air_buffers: Sequence[int],
                 variables: List[str],
                 device: torch.device,
                 field_target: str,
                 seq_len: int,
                 img_input: bool = False,
                 result_dir: str = "./result",
                 return_mode: str = "regular"
                 ):
        self.indices = []
        for i in dataset.index: # debug for example
            data_comb = ""
            for var in variables:
                data_comb += str(dataset.loc[i, var]) + "_"
            self.indices.append(data_comb[:-1])
        self.air_buffers = air_buffers
        self.device = device
        self.img_input = img_input
        self.field_target = field_target
        self.seq_len = seq_len  
        self.result_dir = result_dir
        self.return_mode = return_mode
        if self.return_mode not in ["regular", "fno", "fno_i2f"]:
            raise ValueError('Unknown return mode.')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        label = self.indices[index]
        surfaces = np.load(f'{self.result_dir}/surfaces/{label}_surfaces.npz')["img"][None, :, :] # X: [1, nx, ny] 
        fields = np.load(f'{self.result_dir}/fields/{label}_fields.npz')

        # only read jz because in the previous setting, only jz is recorded
        # only keep the jz at height = air_buffers[2]
        sources = np.load(f'{self.result_dir}/surfaces/{label}_sources.npz')["jz"][:,self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]]  # X: [nt, nx, ny, nz]
        sources = torch.tensor(sources, dtype=torch.float32).to(self.device)[:, None, ...]

        # repeat surfaces at dim = 0 and stack with sources at dim = 1
        if not self.img_input:
            surfaces = np.repeat(surfaces, sources.shape[0], axis=0) # [nt, nx, ny]
            surfaces = np.stack([surfaces, sources / 16000], axis=1) # [nt, 2, nx, ny]
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device)
        else:
            surfaces = torch.tensor(surfaces, dtype=torch.float32).to(self.device) # surfaces[1, nx, ny]
        
        # convert to torch.tensor first for padding
        if self.field_target == "all":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ex", "ey", "ez", "hx", "hy", "hz"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        elif self.field_target == "ezhxhy":
            field = torch.concat([self.pad_fields(torch.tensor(fields[target], dtype=torch.float32).to(self.device), target) for target in ["ez", "hx", "hy"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        else:
            field = torch.tensor(fields[self.field_target], dtype=torch.float32).to(self.device)
            # pad to align the fields and remove air buffers
            field = self.pad_fields(field, self.field_target) # [nt, nz, nx, ny]
            # field = self.sample_time(field)

        # load coefficients
        coefficients = np.load(f'{self.result_dir}/coefficients/{label}_coefficients.npz')
        # print(f"epsrx: {coefficients['epsrx'].shape}") # (nx - 1, ny, nz)
        # print(f"epsry: {coefficients['epsry'].shape}") # (nx, ny - 1, nz)
        # print(f"epsrz: {coefficients['epsrz'].shape}") # (nx, ny, nz - 1)
        # print(f"sigex: {coefficients['sigex'].shape}") # (nx - 1, ny, nz)
        # print(f"sigey: {coefficients['sigey'].shape}") # (nx, ny - 1, nz)
         #print(f"sigez: {coefficients['sigez'].shape}") # (nx, ny, nz - 1)
        coefficients = torch.concat([self.pad_fields(torch.tensor(coefficients[target], dtype=torch.float32).to(self.device), target) for target in ["epsrx", "epsry", "epsrz", "sigex", "sigey", "sigez"]], dim=-3) # X: [nt, 6 * nz, nx, ny]
        # coefficients = coefficients.swapaxes(-1, -3).swapaxes(-1, -2) # [6, nz, nx, ny]

        # print(f"coefficients: {coefficients.shape}")
        if self.return_mode == "fno":
            return field[0], field[1:1+self.seq_len], sources, coefficients
        elif self.return_mode == "fno_i2f":
            return surfaces, field
        else:
            return surfaces, field, sources, coefficients

    def pad_fields(self,
                   field: torch.Tensor,
                   target: str):
        if target == "ex" or target == "sigex" or target == "epsrx":
            

            field = F.pad(field, (0, 0, 0, 0, 0, 1), mode='constant', value=0) 
            if target == "ex":
                field = field / 100 # temporal normalization
        elif target == "ey" or target == "sigey" or target == "epsry": 
            field = F.pad(field, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            if target == "ey":
                field = field / 100
        elif target == "ez" or target == "sigez"or target == "epsrz":
            field = F.pad(field, (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            if target == "ez":
                field = field / 100
        elif target == "hx":
            field = F.pad(field, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
        elif target == "hy":
            field = F.pad(field, (0, 1, 0, 0, 0, 1), mode='constant', value=0)
        elif target == "hz":
            field = F.pad(field, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
        else:
            raise ValueError('Unknown field target.')
        
        # remove air buffers
        return field[..., self.air_buffers[0]:-self.air_buffers[0], self.air_buffers[1]:-self.air_buffers[1], self.air_buffers[2]:-self.air_buffers[2]].swapaxes(-1, -3).swapaxes(-1, -2)

    def get_max_min(self):
        max_min = {}
        for index in range(len(self)):
            label = self.indices[index]
            fields = np.load(f'./result/fields/{label}_fields.npz')
            for target in ["ex", "ey", "ez", "hx", "hy", "hz"]:
                if target not in max_min:
                    max_min[target] = [fields[target].max(), fields[target].min()]
                else:
                    max_min[target][0] = max(max_min[target][0], fields[target].max())
                    max_min[target][1] = min(max_min[target][1], fields[target].min())
        return max_min
    
    
    # def sample_time(self,
    #                 field: torch.Tensor):
    #     sampled_indices = np.linspace(0, field.shape[0]-1, SAMPLE_TIME).astype(int)
    #     return field[:, sampled_indices, ...]