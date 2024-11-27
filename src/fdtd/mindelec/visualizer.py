from matplotlib import animation
from .record import *
import os
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    """
    A visualizer and statistics helper for em fields.
    """

    def __init__(self,
                 recorder: FieldRecorder,
                 fig_dir: str,
                 ):
        self.recorder = recorder
        self.name = self.recorder.name
        self.fig_dir = fig_dir
        self.titles = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    def count_max_min(self):
        self.max = [np.max(self.recorder.ex), np.max(self.recorder.ey), np.max(self.recorder.ez), np.max(self.recorder.hx), np.max(self.recorder.hy), np.max(self.recorder.hz)]
        self.min = [np.min(self.recorder.ex), np.min(self.recorder.ey), np.min(self.recorder.ez), np.min(self.recorder.hx), np.min(self.recorder.hy), np.min(self.recorder.hz)]

    def plot_histogram(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Histogram of all-time EM Fields")

        axes[0, 0].hist(self.recorder.ex.flatten(), bins=100, color='r', alpha=0.7)
        axes[0, 1].hist(self.recorder.ey.flatten(), bins=100, color='g', alpha=0.7)
        axes[0, 2].hist(self.recorder.ez.flatten(), bins=100, color='b', alpha=0.7)
        axes[1, 0].hist(self.recorder.hx.flatten(), bins=100, color='r', alpha=0.7)
        axes[1, 1].hist(self.recorder.hy.flatten(), bins=100, color='g', alpha=0.7)
        axes[1, 2].hist(self.recorder.hz.flatten(), bins=100, color='b', alpha=0.7)

        for i in range(2):
            for j in range(3):
                axes[i, j].set_title(self.titles[3*i+j])
        
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_histogram.pdf"))
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_histogram.png"))
        plt.close()

    def locate_max_min(self, argfunc, field, all_time=False):

        idx = np.unravel_index(argfunc(field), field.shape)
        return idx

    def plot_single_field_at_time_t(self, 
                                    fig: plt.Figure,
                                    ax: plt.Axes,
                                    field: np.ndarray, 
                                    t: int, 
                                    z: int,
                                    title: str,
                                    animated: bool = False):
        """
        Plot a single field surface of z at time t.
        """
        im = ax.imshow(field[t, :, :, z], cmap='jet', animated=animated)
        ax.set_title(f"{title} at t = {t}, z = {z}")
        fig.colorbar(im, ax=ax, location="bottom")
        return im

    def plot_max_map(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Max Map of EM Fields")

        for i in range(2):
            for j in range(3):
                field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                max_idx = self.locate_max_min(np.argmax, field)
                self.plot_single_field_at_time_t(fig, axes[i, j], field, max_idx[0], max_idx[-1], f"Max of {self.titles[i*3+j]}")
        
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_max_map.pdf"))
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_max_map.png"))
        plt.close()

    def plot_min_map(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Min Map of EM Fields")

        for i in range(2):
            for j in range(3):
                field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                min_idx = self.locate_max_min(np.argmin, field)
                self.plot_single_field_at_time_t(fig, axes[i, j], field, min_idx[0], min_idx[-1], f"Min of {self.titles[i*3+j]}")
        
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_min_map.pdf"))
        plt.savefig(os.path.join(self.fig_dir, f"{self.name}_min_map.png"))
        plt.close()

    def plot_max_animation(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        max_idx = []
        ims = []
        
        for i in range(2):
            for j in range(3):
                idx = i*3+j
                field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                max_idx.append(self.locate_max_min(np.argmax, field)) 
                ims.append(self.plot_single_field_at_time_t(fig, axes[i, j], field, max_idx[idx][0], max_idx[idx][-1], f"Max of {self.titles[idx]}", animated=True))

        def animate_func(frame, ims):

            for i in range(2):
                for j in range(3):
                    idx = i*3+j
                    field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                    ims[idx].set_array(field[int(frame), :, :, max_idx[idx][-1]])
                    ims[idx].set_title(f"Max of {self.titles[idx]} at t = {int(frame)}, z = {max_idx[idx][-1]}")
        
        nframe = 60
        t = np.linspace(0, self.recorder.timestep, nframe, endpoint=False).tolist()
        ani = animation.FuncAnimation(fig, animate_func, len(t), fargs=(ims,), interval=100, blit=False)
        ani.save(os.path.join(self.fig_dir, f"{self.name}_max_animation.gif"))
        plt.close()
    
    def plot_min_animation(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        min_idx = []
        ims = []
        
        for i in range(2):
            for j in range(3):
                idx = i*3+j
                field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                min_idx.append(self.locate_max_min(np.argmin, field)) 
                ims.append(self.plot_single_field_at_time_t(fig, axes[i, j], field, 0, min_idx[idx][-1], f"Min of {self.titles[idx]}", animated=True))

        def animate_func(frame, ims):

            for i in range(2):
                for j in range(3):
                    idx = i*3+j
                    field = self.recorder.__dict__[self.titles[3*i+j].lower()]
                    ims[idx].set_array(field[int(frame), :, :, min_idx[idx][-1]])
                    ims[idx].set_title(f"Min of {self.titles[idx]} at t = {int(frame)}, z = {min_idx[idx][-1]}")
        
        nframe = 60
        t = np.linspace(0, self.recorder.timestep, nframe, endpoint=False)[1:].tolist()
        ani = animation.FuncAnimation(fig, animate_func, t, fargs=(ims,), interval=100, blit=False)
        ani.save(os.path.join(self.fig_dir, f"{self.name}_min_animation.gif"))
        plt.close()
