from typing import List, Tuple

import os
import numpy as np
import torch

__all__ = ["SourceRecorder", "FieldRecorder", "SourceRecordSampler", "FieldRecordSampler", "CoefficientRecorder"]

Supported_Sample_Methods = ["time_random", "time_interval", "time_continuous"]
def get_random_indices(start, end, num, rtype="torch", repeat=False):
    """
    Create random indices from start to end.
    """
    if repeat:
        indices = np.sort(np.random.randint(start, end, num, dtype=int))
    else:
        indices = np.sort(np.random.choice(range(start, end), num, replace=False))

    if rtype == "torch":
        return torch.tensor(indices)    
    elif rtype == "list":
        return indices.tolist()
    else:
        raise ValueError(f"Invalid return type: {rtype}")

class SourceRecorder():
    """
    Record the full-time Source J.
    """
    def __init__(self,
                 name: str,):
        self.name = name
    
    def initiate_sources(self,
                         timestep: int,
                         shape: Tuple[int],
                         npml: int,):
        
        self.timestep = timestep
        self.shape = shape
        self.npml = npml
        self.inner_field = lambda x: x[self.npml:-self.npml, self.npml:-self.npml, self.npml:-self.npml]
        self.jx = np.ndarray((timestep, shape[0]-2*self.npml, shape[1]+1-2*self.npml, shape[2]+1-2*self.npml))
        self.jy = np.ndarray((timestep, shape[0]+1-2*self.npml, shape[1]-2*self.npml, shape[2]+1-2*self.npml))
        self.jz = np.ndarray((timestep, shape[0]+1-2*self.npml, shape[1]+1-2*self.npml, shape[2]-2*self.npml))

    def update(self, t, jx, jy, jz):
        self.jx[t] = self.inner_field(jx.cpu().numpy()[0, 0])
        self.jy[t] = self.inner_field(jy.cpu().numpy()[0, 0])
        self.jz[t] = self.inner_field(jz.cpu().numpy()[0, 0])

    def save_numpy(self, save_dir="./"):
        np.savez_compressed(os.path.join(save_dir, f"{self.name}_sources.npz"), jx=self.jx, jy=self.jy, jz=self.jz)

class FieldRecorder():
    """
    Record the full-time EM fields.
    """
    def __init__(self,
                 name,
                 ):
        self.name = name
        
    def initiate_fields(self,
                        timestep: int,
                        shape: Tuple[int],
                        npml: int,):
        self.timestep = timestep
        self.shape = shape
        self.npml = npml
        self.inner_field = lambda x: x[self.npml:-self.npml, self.npml:-self.npml, self.npml:-self.npml]
        self.ex = np.ndarray((timestep, shape[0]-2*self.npml, shape[1]+1-2*self.npml, shape[2]+1-2*self.npml))
        self.ey = np.ndarray((timestep, shape[0]+1-2*self.npml, shape[1]-2*self.npml, shape[2]+1-2*self.npml))
        self.ez = np.ndarray((timestep, shape[0]+1-2*self.npml, shape[1]+1-2*self.npml, shape[2]-2*self.npml))
        self.hx = np.ndarray((timestep, shape[0]+1-2*self.npml, shape[1]-2*self.npml, shape[2]-2*self.npml))
        self.hy = np.ndarray((timestep, shape[0]-2*self.npml, shape[1]+1-2*self.npml, shape[2]-2*self.npml))
        self.hz = np.ndarray((timestep, shape[0]-2*self.npml, shape[1]-2*self.npml, shape[2]+1-2*self.npml))
    
    def update(self, t, ex, ey, ez, hx, hy, hz):
        self.ex[t] = self.inner_field(ex.cpu().numpy()[0, 0])
        self.ey[t] = self.inner_field(ey.cpu().numpy()[0, 0])
        self.ez[t] = self.inner_field(ez.cpu().numpy()[0, 0])
        self.hx[t] = self.inner_field(hx.cpu().numpy()[0, 0])
        self.hy[t] = self.inner_field(hy.cpu().numpy()[0, 0])
        self.hz[t] = self.inner_field(hz.cpu().numpy()[0, 0])

    def save_numpy(self, save_dir="./"):
        np.savez_compressed(os.path.join(save_dir, f"{self.name}_fields.npz"), ex=self.ex, ey=self.ey, ez=self.ez, hx=self.hx, hy=self.hy, hz=self.hz)
        # np.save(os.path.join(save_dir, f"{self.name}_ex.npy"), self.ex)
        # np.save(os.path.join(save_dir, f"{self.name}_ey.npy"), self.ey)
        # np.save(os.path.join(save_dir, f"{self.name}_ez.npy"), self.ez)
        # np.save(os.path.join(save_dir, f"{self.name}_hx.npy"), self.hx)
        # np.save(os.path.join(save_dir, f"{self.name}_hy.npy"), self.hy)
        # np.save(os.path.join(save_dir, f"{self.name}_hz.npy"), self.hz)
    
    def check_numpy_exists(self, save_dir="./"):
        return os.path.exists(os.path.join(save_dir, f"{self.name}_fields.npz"))

class SourceRecordSampler(SourceRecorder):
    """
    Sample the sources.
    """
    def __init__(self,
                 name,
                 sample_method: str,
                 sample_num: dict,
                 nt: int,
                 dt: int = 1,
                 ):
        super().__init__(name)
        """
        Record the fields at sampled time indices.

        Args:
            sample_method (str): Sampling method. Supported methods: "time_random", "time_interval"
            sample_num (dict): Number of samples in each direction.
            ori_dim (dict): Original dimensions of the fields.
        """
        self.sample_method = sample_method
        self.sample_num = sample_num

        self.sampled_time_indices = self.get_time_samples(self.sample_num["t"], nt, dt)

    def initiate_sources(self, 
                         timestep: int, 
                         shape: Tuple[int], 
                         npml: int):
        return super().initiate_sources(len(self.sampled_time_indices), shape, npml)
    
    def get_time_samples(self,
                         sample_num: int,
                         nt: int,
                         dt: int = 1,
                         ):
        if self.sample_method == "time_random":
            sample_indices = get_random_indices(0, nt, sample_num, rtype="list")
        elif self.sample_method == "time_interval":
            sample_indices = np.linspace(0, nt, sample_num, dtype=int).tolist()
        elif self.sample_method == "time_continuous":
            start = max(0, nt // 2 - (sample_num - 1) * dt // 2)
            end = start +  (sample_num - 1) * dt + 1
            sample_indices = np.arange(start, end, dt, dtype=int).tolist()
            assert len(sample_indices) == sample_num
            # sample_indices = np.arange(nt//2, nt//2+sample_num, dtype=int).tolist()
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")
        return sample_indices
    
    def update(self, t, jx, jy, jz):
               
        if self.sample_method in Supported_Sample_Methods:
            if t in self.sampled_time_indices:
                super().update(self.sampled_time_indices.index(t), jx, jy, jz)
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")
    
    def check_existance(self, save_dir="./"):
        if os.path.exists(os.path.join(save_dir, f"{self.name}_sources.npz")):
            return True

    def save_numpy(self, save_dir="./"):
        np.savez_compressed(os.path.join(save_dir, f"{self.name}_sources.npz"),
                            t=self.sampled_time_indices,
                            jx=self.jx, jy=self.jy, jz=self.jz)
        return
    
class FieldRecordSampler(FieldRecorder):
    """
    Sample the fields.
    """
    def __init__(self,
                 name,
                 sample_method: str,
                 sample_num: dict,
                 nt: int,
                 dt: int = 1
                 ):
        super().__init__(name)
        """
        Record the fields at sampled time indices.

        Args:
            sample_method (str): Sampling method. Supported methods: "time_random", "time_interval"
            sample_num (dict): Number of samples in each direction.
            ori_dim (dict): Original dimensions of the fields.
        """
        self.sample_method = sample_method
        self.sample_num = sample_num

        self.sampled_time_indices = self.get_time_samples(self.sample_num["t"], nt, dt)

    def initiate_fields(self, 
                        timestep: int, 
                        shape: Tuple[int], 
                        npml: int):
        return super().initiate_fields(len(self.sampled_time_indices), shape, npml)
    
    def get_time_samples(self,
                         sample_num: int,
                         nt: int,
                         dt: int = 1,
                         ) -> List[int]:
        if self.sample_method == "time_random":
            sample_indices = get_random_indices(0, nt, sample_num, rtype="list")
        elif self.sample_method == "time_interval":
            sample_indices = np.linspace(0, nt, sample_num, dtype=int).tolist()
        elif self.sample_method == "time_continuous":
            start = max(0, nt // 2 - (sample_num - 1) * dt // 2)
            end = start +  (sample_num - 1) * dt + 1
            sample_indices = np.arange(start, end, dt, dtype=int).tolist()
            assert len(sample_indices) == sample_num
            # sample_indices = np.arange(nt//2, nt//2+sample_num, dtype=int).tolist()
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")
        return sample_indices
    
    def update(self,
               t, ex, ey, ez, hx, hy, hz):
               
        """
        Args:
            sample_num (dict): Number of samples in each direction.
            ori_dim (dict): Original dimensions of the fields.
        """
        if self.sample_method in Supported_Sample_Methods:
            if t in self.sampled_time_indices:
                super().update(self.sampled_time_indices.index(t), ex, ey, ez, hx, hy, hz)
        
        else:
            raise ValueError(f"Invalid sample method: {self.sample_method}")
    
    def check_existance(self, save_dir="./"):
        if os.path.exists(os.path.join(save_dir, f"{self.name}_fields.npz")):
            return True

    def save_numpy(self, save_dir="./"):
        np.savez_compressed(os.path.join(save_dir, f"{self.name}_fields.npz"),
                            t=self.sampled_time_indices, 
                            ex=self.ex, ey=self.ey, ez=self.ez, 
                            hx=self.hx, hy=self.hy, hz=self.hz)
        return
    
class CoefficientRecorder:
    """
    Record the full-time EM fields coefficients.
    """
    def __init__(self,
                 name,
                 ):
        self.name = name
    
    def initiate_coefficients(self,
                              npml: int,):
     
        self.npml = npml
        self.inner_field = lambda x: x[self.npml:-self.npml, self.npml:-self.npml, self.npml:-self.npml]
        

    def update(self,
               epsrx, epsry, epsrz, sigex, sigey, sigez,
               murx, mury, murz, sigmx, sigmy, sigmz):
        self.epsrx, self.epsry, self.epsrz = self.inner_field(epsrx.cpu().numpy()[0, 0]), self.inner_field(epsry.cpu().numpy()[0, 0]), self.inner_field(epsrz.cpu().numpy()[0, 0])
        self.sigex, self.sigey, self.sigez = self.inner_field(sigex.cpu().numpy()[0, 0]), self.inner_field(sigey.cpu().numpy()[0, 0]), self.inner_field(sigez.cpu().numpy()[0, 0])
        self.murx, self.mury, self.murz = murx.item(), mury.item(), murz.item()
        self.sigmx, self.sigmy, self.sigmz = sigmx.item(), sigmy.item(), sigmz.item()

    def save_numpy(self, save_dir="./"):
        np.savez_compressed(os.path.join(save_dir, f"{self.name}_coefficients.npz"),
                            epsrx=self.epsrx, epsry=self.epsry, epsrz=self.epsrz,
                            sigex=self.sigex, sigey=self.sigey, sigez=self.sigez,
                            murx=self.murx, mury=self.mury, murz=self.murz,
                            sigmx=self.sigmx, sigmy=self.sigmy, sigmz=self.sigmz)
        return

