from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
class Compressor(ABC):
    """
    Abstract class for compressors.
    """
    
    def __init__(self):
        pass

    @abstractmethod
    def compress(self, field):
        pass

    def decompress(self, compressed_field):
        pass

class DFTCompressor(Compressor):

    def __init__(self,
                 n_mode_1: int,
                 n_mode_2: int,
                 n_mode_3: int):
        super().__init__()
        self.n_mode_1 = n_mode_1
        self.n_mode_2 = n_mode_2
        self.n_mode_3 = n_mode_3
        return
    
    def compress(self, field: torch.Tensor):
        
        x_ft = torch.fft.rfftn(field, dim=(-3, -2, -1), norm="ortho")

        # 80: :40 -39:
        # 81: :41 -40:
        n_mode_1 = min(self.n_mode_1, (field.size(-3) - 1) // 2)
        n_mode_2 = min(self.n_mode_2, (field.size(-2) - 1) // 2)
        n_mode_3 = min(self.n_mode_3, (field.size(-1) - 1) // 2)

        x_ft[..., 1+n_mode_1:-n_mode_1, :, :] = 0
        x_ft[..., :, 1+n_mode_2:-n_mode_2, :] = 0
        x_ft[..., :, :, 1+n_mode_3:] = 0

        compressed_field = torch.fft.irfftn(x_ft, s=(field.size(-3), field.size(-2), field.size(-1)), dim=(-3, -2, -1), norm="ortho")

        fig, axes = plt.subplots(1, 3)

        vmin, vmax = min(torch.min(compressed_field).item(), torch.min(field).item()), max(torch.max(compressed_field).item(), torch.max(field).item())
        pc = axes[1].imshow(torch.mean(compressed_field[0, 0], dim=-1, keepdim=False).cpu().numpy(), vmin=vmin, vmax=vmax)
        pc = axes[0].imshow(torch.mean(field[0, 0], dim=-1, keepdim=False).cpu().numpy(), vmin=vmin, vmax=vmax)
        pc = axes[2].imshow(torch.mean(torch.abs(compressed_field[0, 0] - field[0, 0]), dim=-1, keepdim=False).cpu().numpy(), vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=axes, location='bottom')
        plt.savefig(f"dft_{n_mode_1}_{n_mode_2}_{n_mode_3}.png")
        plt.savefig(f"dft_{n_mode_1}_{n_mode_2}_{n_mode_3}.pdf")
        
        return x_ft

if __name__ == "__main__":
    test_x = torch.rand((1, 1, 72, 68, 16))
    compressor = DFTCompressor(30, 30, 8)
    compressor.compress(test_x)
    
