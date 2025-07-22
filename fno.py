# test distributed data on slurm
from calendar import c
import sys
import os
from typing import Sequence

import torch.distributed

from src import *
from src.build_model import build_model

# custom library ^_^
import planetzoo as pnz

from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

PLOT_EXAMPLES = 3
NUM_EXAMPLES = 1
FIELD_TARGET = "ez" # ez | all | ezhxhy
SEQ_LEN = 25
MODEL_NAME = "FNO2D"
IMG_INPUT = True
DDP = False
LABELS = ["ex", "ey", "ez", "hx", "hy", "hz"]
train_ratio = 0.2
PH_FACTOR = 0
SAMPLE_DT = 10
if FIELD_TARGET == "ezhxhy":
    LABELS = ["ez", "hx", "hy"]

cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
cfl_number = 0.9
dt = estimate_time_interval(cell_lengths, cfl_number, epsr_min=1., mur_min=1.)

dte = dt / epsilon0
dtm = dt / mu0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    parser.add_argument('-f', '--force', action='store_true', help='retrain')
    return parser.parse_args()

def plot_lz_rz(lz: torch.Tensor, # shape: (nt - 1, nz, nx - 1, ny - 1)
               rz: torch.Tensor,
               epoch: int,
               postfix: str,
               channel: int = 0):
    
    lz = lz[::lz.shape[0]//PLOT_EXAMPLES, ...]
    rz = rz[::rz.shape[0]//PLOT_EXAMPLES, ...]
    fig, axes = plt.subplots(3, PLOT_EXAMPLES, figsize=(21, 9), )

    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    for i in range(PLOT_EXAMPLES):
        lzi = lz[i, channel,].detach().cpu().numpy()
        rzi = rz[i, channel,].detach().cpu().numpy()

        vmax = np.max([np.max(np.abs(lzi)), np.max(np.abs(rzi))])
        vmin = -vmax
        pc = axes[0, i].imshow(lzi, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Left of Maxwell Equation')
        pc = axes[1, i].imshow(rzi, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Right of Maxwell Equation')
        pc = axes[2, i].imshow(np.abs(lzi - rzi), cmap='jet', vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f'Error')
    
    fig.suptitle(f"Comparison of Maxwell Equation")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_epoch_{epoch}_{postfix}_Maxwell.pdf")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_epoch_{epoch}_{postfix}_Maxwell.png")
    plt.close("all")
        
def plot(prediction: torch.Tensor,  # shape: (nt, nz, nx, ny)
         truth: torch.Tensor, # shape: (nt, nz, nx, ny)
         epoch: int,
         postfix: str,
         channel: int = 1,
         channel_name: str = "ez"):
    
    # select linespace time steps
    # prediction = prediction[::prediction.shape[0]//PLOT_EXAMPLES, ...]
    # truth = truth[::truth.shape[0]//PLOT_EXAMPLES, ...]
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    
    pred = prediction[channel, :, :].detach().cpu().numpy()
    true = truth[channel, :, :].detach().cpu().numpy()
    vmax = np.max([np.max(np.abs(pred)), np.max(np.abs(true))])
    vmin = -vmax
    pc = axes[0].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Prediction {FIELD_TARGET}')
    pc = axes[1].imshow(true, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Truth {FIELD_TARGET}')
    pc = axes[2].imshow(np.abs(pred - true), cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Error {FIELD_TARGET}')

    # for j in range(3):
    #     fig.colorbar(pc, ax=axes[j, PLOT_EXAMPLES-1], location='right')

    fig.suptitle(f"Comparison at epoch = {epoch} of {channel_name}")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_exp_epoch_{epoch}_{FIELD_TARGET}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_{postfix}_{channel_name}_pinn.pdf")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_exp_epoch_{epoch}_{FIELD_TARGET}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_{postfix}_{channel_name}_pinn.png")
    plt.close("all")
    return

def plot_one_frame(prediction: torch.Tensor,  # shape: (nz, nx, ny)
                    truth: torch.Tensor, # shape: (nz, nx, ny)
                    epoch: int,
                    postfix: str,
                    channel_name: str = "ezhxhy"):
    
    # select linespace time steps

    columns = prediction.shape[0]

    fig, axes = plt.subplots(3, columns, figsize=(16, 9))
    for i in range(columns):
        pred = prediction[i, :, :].detach().cpu().numpy()
        true = truth[i, :, :].detach().cpu().numpy()
        vmax = np.max([np.max(np.abs(pred[i])), np.max(np.abs(true[i]))])
        vmin = -vmax
        pc = axes[0, i].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Prediction {FIELD_TARGET}')
        pc = axes[1, i].imshow(true, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Truth {FIELD_TARGET}')
        pc = axes[2, i].imshow(np.abs(pred - true), cmap='jet', vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f'Error {FIELD_TARGET}')

    # for j in range(3):
    #     fig.colorbar(pc, ax=axes[j, PLOT_EXAMPLES-1], location='right')

    fig.suptitle(f"Comparison at epoch = {epoch} of {channel_name}")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_exp_epoch_{epoch}_{FIELD_TARGET}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_{postfix}_{channel_name}_pinn.pdf")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_exp_epoch_{epoch}_{FIELD_TARGET}_{str(train_ratio).replace('.', 'd')}_ph_{str(PH_FACTOR).replace('.', 'd')}_{postfix}_{channel_name}_pinn.png")
    plt.close("all")
    return


def train():

    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    air_buffers = configs.air_buffers
    dataset_dir = configs.dir.data
    train_ratio = configs.train_ratio
    seq_len = configs.seq_len

    # concat l1, l2, d from configs.vars and combine with '_'
    vars_str = vars2str(configs.vars)

    # read dataset indices from generated csv
    if os.path.exists(f'{dataset_dir}/{vars_str}.csv'):
        df = pd.read_csv(f'{dataset_dir}/{vars_str}.csv')
        train, test = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
        if os.path.exists(f"{dataset_dir}/{vars_str}_train_indices_{str(train_ratio).replace('.', 'd')}.txt") and os.path.exists(f"{dataset_dir}/{vars_str}_test_indices_{str(train_ratio).replace('.', 'd')}.txt"):
            with open(f"{dataset_dir}/{vars_str}_train_indices_{str(train_ratio).replace('.', 'd')}.txt", "r") as f:
                train_indices = [int(idx) for idx in f.read().split("\n")]
            with open(f"{dataset_dir}/{vars_str}_test_indices_{str(train_ratio).replace('.', 'd')}.txt", "r") as f:
                test_indices = [int(idx) for idx in f.read().split("\n")]
            train = df.loc[train_indices]
            test = df.loc[test_indices]
        else:
            with open(f"{dataset_dir}/{vars_str}_train_indices_{str(train_ratio).replace('.', 'd')}.txt", "w") as f:
                f.write("\n".join(train.index.astype(str).values))
            with open(f"{dataset_dir}/{vars_str}_test_indices_{str(train_ratio).replace('.', 'd')}.txt", "w") as f:
                f.write("\n".join(test.index.astype(str).values))
    else:
        raise FileNotFoundError(f"{vars_str}.csv not found.")
    
    
    field_target = configs.field_target
    ph_factor = configs.ph_factor

    train_dataset = PinnDataset(dataset=train, 
                                air_buffers=air_buffers, 
                                variables=configs.vars,
                                device=device, 
                                field_target=field_target, 
                                seq_len=seq_len, 
                                img_input=True,
                                result_dir=f'{dataset_dir}/result',
                                return_mode="fno_i2f")
    test_dataset = PinnDataset(dataset=test,
                               air_buffers=air_buffers,
                               variables=configs.vars,
                               device=device,
                               field_target=field_target,
                               seq_len=seq_len,
                               img_input=True,
                               result_dir=f'{dataset_dir}/result',
                               return_mode="fno_i2f")

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    model_name = configs.model.name.lower()

    # Build model using the new build_model function
    net = build_model(
        model_type=model_name,
        in_channels=configs.model.in_channels,
        out_channels=configs.model.out_channels,
        device=device,
        seq_len=seq_len if model_name in ['fno2drnn', 'fno2dgru'] else None,
        dim=configs.model.dim,
        kernel_list=configs.model.kernel_list,
        kernel_size_list=configs.model.kernel_size_list,
        padding_list=configs.model.padding_list,
        hidden_list=configs.model.hidden_list,
        mode_list=[tuple(mode) for mode in configs.model.mode_list],
        act_func=configs.model.act_func,
        unet=configs.model.unet if hasattr(configs.model, 'unet') else False
    )
    
    net.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                           func_args={"mode": "fan_in", "nonlinearity": "relu"})
        
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)

    custom_loss_fn = torch.nn.MSELoss()
    
    not_read_ckpt = args.force
    num_epochs = 5000
    epoch_continue = 0

    # Create necessary folders in advance
    os.makedirs(configs.dir.model, exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    ckpt_path = f"./{configs.dir.model}/{model_name.lower()}_exp_{field_target}_{str(train_ratio).replace('.', 'd')}_ph_{str(ph_factor).replace('.', 'd')}_pinn_ckpt.pt"
    train_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_exp_{field_target}_{str(train_ratio).replace('.', 'd')}_ph_{str(ph_factor).replace('.', 'd')}_pinn_train_loss.csv"
    test_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_exp_{field_target}_{str(train_ratio).replace('.', 'd')}_ph_{str(ph_factor).replace('.', 'd')}_pinn_test_loss.csv"
    train_loss_df = pd.DataFrame(columns=['epoch', 'loss']).astype({'epoch': 'int64', 'loss': 'float64'})
    test_loss_df = pd.DataFrame(columns=['epoch', 'loss']).astype({'epoch': 'int64', 'loss': 'float64'})

    if not not_read_ckpt:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt['epoch']
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # processor = torch.load(ckpt['processor'])
        if os.path.exists(train_loss_df_path):
            train_loss_df = pd.read_csv(train_loss_df_path, index_col=False)
        if os.path.exists(test_loss_df_path):
            test_loss_df = pd.read_csv(test_loss_df_path, index_col=False)

    net.train()
    epoch_pbar = tqdm(range(epoch_continue, num_epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        
        train_loss = train_data_loss = train_ph_loss = 0.0
        train_size = 0
        
        # Training loop without inner progress bar
        for i, (X_train, y_train) in enumerate(train_dataloader):
            y_pred_train = net(X_train)
            loss = custom_loss_fn(y_pred_train, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * X_train.shape[0]
            train_size += X_train.shape[0]
        
        loss = train_loss / train_size
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({'train_loss': f'{loss:.6f}'})

        new_row = pd.DataFrame({'epoch': [epoch + 1], 'loss': [loss]})
        train_loss_df = pd.concat([train_loss_df, new_row], ignore_index=True)

        if epoch % configs.epoch_to_validate == (configs.epoch_to_validate - 1):
            with torch.no_grad():
                test_loss = test_data_loss = test_ph_loss = 0.0
                test_size = 0
                for i, (X_test, y_test) in enumerate(test_dataloader):
                    y_pred_test = net(X_test)
                    loss  = custom_loss_fn(y_pred_test, y_test)

                    test_loss += loss.item() * X_test.shape[0]
                    test_size += X_test.shape[0] 

                test_loss = test_loss / test_size
                
                # Update epoch progress bar with test loss
                epoch_pbar.set_postfix({'train_loss': f'{train_loss/train_size:.6f}', 'test_loss': f'{test_loss:.6f}'})

                new_row = pd.DataFrame({'epoch': [epoch + 1], 'loss': [test_loss]})
                test_loss_df = pd.concat([test_loss_df, new_row], ignore_index=True)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                train_loss_df.to_csv(train_loss_df_path, index=False) 
                test_loss_df.to_csv(test_loss_df_path, index=False)

    net.eval()
    with torch.no_grad():
        for i, (X_test, y_test) in enumerate(test_dataloader):
            print(X_test.shape)
            print(y_test.shape)
            y_pred_test = net(X_test)            
            
            loss  = custom_loss_fn(y_pred_test, y_test)
            print(f"Final Loss: {loss.item()}")   
            
            
if __name__ == "__main__":
    train()