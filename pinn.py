# test distributed data on slurm
import sys
import os
from typing import Sequence

import torch.distributed

sys.path.append(os.path.abspath('..'))

from src import *

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

PLOT_EXAMPLES = 6
NUM_EXAMPLES = 1
FIELD_TARGET = "ezhxhy" # ez | all | ezhxhy
SEQ_LEN = 25
MODEL_NAME = "SimI2V"
IMG_INPUT = True
DDP = True
LABELS = ["ex", "ey", "ez", "hx", "hy", "hz"]
TRAIN_RATIO = 0.2
PH_FACTOR = 0
SAMPLE_DT = 10
if FIELD_TARGET == "ezhxhy":
    LABELS = ["ez", "hx", "hy"]

# LOSS_BALANCING = "RELOBRALO"
LOSS_BALANCING = "RELOBPECH"
BETA = 0.1
LOSS_PH_THRESHOLD = 0.02
WAIT = 10
cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
cfl_number = 0.9
dt = estimate_time_interval(cell_lengths, cfl_number, epsr_min=1., mur_min=1.)

dte = dt / epsilon0
dtm = dt / mu0

def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        description='FDTD-Based Electromagnetics Forward-Problem Solver')
    parser.add_argument('--force', action='store_true', help='force to retrain')
    options = parser.parse_args()
    return options

def plot_lz_rz(lz: torch.Tensor, # shape: (nt - 1, nz, nx - 1, ny - 1)
               rz: torch.Tensor,
               epoch: int,
               postfix: str,
               channel: int = 0):
    
    lz = lz[::lz.shape[0]//PLOT_EXAMPLES, ...]
    rz = rz[::rz.shape[0]//PLOT_EXAMPLES, ...]
    fig, axes = plt.subplots(3, PLOT_EXAMPLES, figsize=(16, 9))
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
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_{str(TRAIN_RATIO).replace('.', 'd')}_{LOSS_BALANCING}_nowait_beta_{str(BETA).replace('.', 'd')}_epoch_{epoch}_{postfix}_Maxwell.pdf")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_{str(TRAIN_RATIO).replace('.', 'd')}_{LOSS_BALANCING}_nowait_beta_{str(BETA).replace('.', 'd')}_epoch_{epoch}_{postfix}_Maxwell.png")
    plt.close("all")
        
def plot(prediction: torch.Tensor,  # shape: (nt, nz, nx, ny)
         truth: torch.Tensor, # shape: (nt, nz, nx, ny)
         epoch: int,
         postfix: str,
         channel: int = 1,
         channel_name: str = "ez"):
    
    # select linespace time steps
    prediction = prediction[::prediction.shape[0]//PLOT_EXAMPLES, ...]
    truth = truth[::truth.shape[0]//PLOT_EXAMPLES, ...]
    fig, axes = plt.subplots(3, PLOT_EXAMPLES, figsize=(16, 9))
    for i in range(PLOT_EXAMPLES):
        pred = prediction[i, channel, :, :].detach().cpu().numpy()
        true = truth[i, channel, :, :].detach().cpu().numpy()
        vmax = np.max([np.max(np.abs(pred)), np.max(np.abs(true))])
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
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_epoch_{epoch}_{FIELD_TARGET}_{str(TRAIN_RATIO).replace('.', 'd')}_{LOSS_BALANCING}_{postfix}_{channel_name}_pinn.pdf")
    plt.savefig(f"./figures/{MODEL_NAME.lower()}_epoch_{epoch}_{FIELD_TARGET}_{str(TRAIN_RATIO).replace('.', 'd')}_{LOSS_BALANCING}_{postfix}_{channel_name}_pinn.png")
    plt.close("all")
    return

def read_snp(dataset: pd.DataFrame,
             device: torch.device):
    """
    Read snp files and return torch tensor of X and y.
    """
    Xs, ys = [], []
    for i in tqdm(dataset.index, desc="Reading Snp Files"):
        l1 = dataset.loc[i, 'l1']
        l2 = dataset.loc[i, 'l2']
        d = dataset.loc[i, 'd']
        snp = np.load(f'./result/s_parameters/microstrip_filter_s_parameters_{l1}_{l2}_{d}.npz')
        s_parameters = snp["s_parameters"] # shape: (nf, 2, 1)
        s11 = s_parameters[:, 0, 0]
        s21 = s_parameters[:, 1, 0]
        s11 = np.stack([s11.real, s11.imag], axis=0)
        s21 = np.stack([s21.real, s21.imag], axis=0)
        y = np.concatenate([s11, s21], axis=0)
        X = np.array([l1, l2, d])
        Xs.append(X)
        ys.append(y)
    return torch.tensor(np.stack(Xs, axis=0), dtype=torch.float32).to(device), torch.tensor(np.stack(ys, axis=0), dtype=torch.float32).to(device)

def init_distribution(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
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
    
    # TODO: use custom pytorch dataset to load data
    air_buffers=(3, 3, 3)

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


    train_dataset = CustomPinnDataset(dataset=train, 
                                      air_buffers=air_buffers, 
                                      device=device, 
                                      field_target=field_target, 
                                      img_input=True)
    
    test_dataset = CustomPinnDataset(dataset=test, 
                                     air_buffers=air_buffers, 
                                     device=device, 
                                     field_target=field_target, 
                                     img_input=True)

    train_dataloader = DataLoader(train_dataset, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    out_channels = 2

    if config.loss_balancing.name == "RELOBRALO":
        loss_balancing = ReLoBRaLo()
    elif config.loss_balancing.name == "RELOBPECH":
        loss_balancing = ReLoBPeCh(beta=config.loss_balancing.beta, wait=config.loss_balancing.wait)
    beta = config.loss_balancing.beta


    # unet lstm
    # net = pnz.model.UNetLSTM(in_channels=in_channels, hidden_channels=[32, 64, 128, 256], out_channels=out_channels, norm_layer=nn.BatchNorm2d).to(device)

    # net = pnz.model.ConvLSTM(in_channels=in_channels, hidden_channels=[32, 32, 32, out_channels], batch_first=True, return_all_layers=False).to(device)
    
    model_name = configs.model.name.lower()

    if model_name == "simi2vsepdecnoskip":
        if field_target == "all":
            net = pnz.model.SimI2VSepDecNoSkip(
                in_channels=configs.model.in_channels, 
                hidden_channels=configs.model.hidden_channels, 
                out_channels=configs.model.out_channels, 
                seq_len=seq_len * 6).to(device)
        elif field_target == "ezhxhy":
            net = pnz.model.SimI2VSepDecNoSkip(
                in_channels=configs.model.in_channels, 
                hidden_channels=configs.model.hidden_channels, 
                out_channels=configs.model.out_channels, 
                seq_len=seq_len * 3).to(device)
        else:
            net = pnz.model.SimI2VSepDecNoSkip(
                in_channels=configs.model.in_channels, 
                hidden_channels=configs.model.hidden_channels, 
                out_channels=configs.model.out_channels, 
                seq_len=seq_len).to(device)
    elif model_name == "simi2vconnectdecnoskip":
        if field_target == "ezhxhy":
            net = pnz.model.SimI2VConnectDecNoSkip(
                in_channels=configs.model.in_channels, 
                hidden_channels=configs.model.hidden_channels, 
                groups=configs.model.groups, 
                out_channels=configs.model.out_channels, 
                seq_len=seq_len).to(device)
        else:
            raise NotImplementedError("Not implemented yet.")

    elif model_name == "simi2v":
        if field_target == "ezhxhy":
            net = pnz.model.SimI2V(
                in_channels=configs.model.in_channels, 
                hidden_channels=configs.model.hidden_channels, 
                groups=configs.model.groups, 
                out_channels=configs.model.out_channels, 
                seq_len=seq_len).to(device)
        else:
            raise NotImplementedError("Not implemented yet.")
        
    net.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                           func_args={"mode": "fan_in", "nonlinearity": "relu"})
        
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)

    maxwell_ezhxhy = Maxwell_EzHxHy(dx=cell_lengths[0], 
                                    dy=cell_lengths[1], 
                                    dz=cell_lengths[2], 
                                    dte=dte, 
                                    delta_t=SAMPLE_DT)
    
    
    custom_loss_fn = CustomLossEzHxHy(loss_fn=torch.nn.MSELoss())

    not_read_ckpt = args.force
    num_epochs = 10000

    ckpt_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{config.loss_balancing.name}_nowait_beta_{str(beta).replace('.', 'd')}_pinn_ckpt.pt"


    train_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{config.loss_balancing.name}_nowait_beta_{str(beta).replace('.', 'd')}_pinn_train_loss.csv"
    test_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{config.loss_balancing.name}_nowait_beta_{str(beta).replace('.', 'd')}_pinn_test_loss.csv"
    train_loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    test_loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    epoch_continue = 0

    if not not_read_ckpt:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt['epoch']
            loss_balancing.load_state_dict(ckpt['loss_balancing'])
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # processor = torch.load(ckpt['processor'])
        if os.path.exists(train_loss_df_path):
            train_loss_df = pd.read_csv(train_loss_df_path, index_col=False)
        if os.path.exists(test_loss_df_path):
            test_loss_df = pd.read_csv(test_loss_df_path, index_col=False)
    
    net.train()
    
    for epoch in range(epoch_continue, num_epochs):
        
        train_loss = train_data_loss = train_ph_loss = 0.0
        train_size = 0
        for i, (X_train, y_train, sources_train, coefficients_train) in enumerate(train_dataloader):
            batch_size = X_train.size()[0]
            y_pred_train = net(X_train)
            
            if field_target == "all" and model_name == "SimI2VSepDecNoSkip":
                y_pred_train = y_pred_train.view(batch_size, seq_len, 6 * out_channels, *y_pred_train.size()[-2:])
            elif field_target == "ezhxhy" and model_name == "SimI2VSepDecNoSkip":
                y_pred_train = y_pred_train.view(batch_size, seq_len, 3 * out_channels, *y_pred_train.size()[-2:])

            lz, rz = maxwell_ezhxhy(y_pred_train, coefficients_train, sources_train)
            loss_ph, loss_data = custom_loss_fn(y_pred_train, y_train, lz, rz)

            losses = [loss_ph.item() * 10, loss_data.item()]
            factors = loss_balancing.update(epoch, losses)

            loss = loss_ph * factors[0] + loss_data * factors[1] * 10
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * X_train.shape[0]
            train_data_loss += loss_data.item() * X_train.shape[0]
            train_ph_loss += loss_ph.item() * X_train.shape[0]
            train_size += X_train.shape[0]
        
        loss = train_loss / train_size
        data_loss = train_data_loss / train_size
        ph_loss = train_ph_loss / train_size

        train_loss_df = pd.concat([train_loss_df, pd.DataFrame({'epoch': epoch + 1, 'loss': loss, 'data_loss': data_loss, 'ph_loss': ph_loss, 'ph_factor': factors[0], 'data_factor': factors[1]}, index=[0])], ignore_index=True)

        if epoch % 10 == 9:

            # loss_info = f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {loss}'
            # print(loss_info)
            with torch.no_grad():
                test_loss = test_data_loss = test_ph_loss = 0.0
                test_size = 0
                for i, (X_test, y_test, sources_test, coefficients_test) in enumerate(test_dataloader):
                    batch_size = X_test.size()[0]
                    y_pred_test = net(X_test)
                    if field_target == "all" and model_name == "SimI2VSepDecNoSkip":
                        y_pred_test = y_pred_test.view(batch_size, seq_len, 6 * out_channels, *y_pred_test.size()[-2:])
                    elif field_target == "ezhxhy" and model_name == "SimI2VSepDecNoSkip":
                        y_pred_test = y_pred_test.view(batch_size, seq_len, 3 * out_channels, *y_pred_test.size()[-2:])
                    lz, rz = maxwell_ezhxhy(y_pred_test, coefficients_test, sources_test)
                    loss_ph, loss_data = custom_loss_fn(y_pred_test, y_test, lz, rz)
                    loss = loss_ph * PH_FACTOR + loss_data

                    test_loss += loss.item() * X_test.shape[0]
                    test_data_loss += loss_data.item() * X_test.shape[0]
                    test_ph_loss += loss_ph.item() * X_test.shape[0]
                    test_size += X_test.shape[0] 

                test_loss = test_loss / test_size
                data_loss = test_data_loss / test_size
                ph_loss = test_ph_loss / test_size

                test_loss_df = pd.concat([test_loss_df, pd.DataFrame({'epoch': epoch + 1, 'loss': test_loss, 'data_loss': data_loss, 'ph_loss': ph_loss}, index=[0])], ignore_index=True)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_balancing': loss_balancing.state_dict(),
                    # 'processor': processor
                }, ckpt_path)
                train_loss_df.to_csv(train_loss_df_path, index=False) 
                test_loss_df.to_csv(test_loss_df_path, index=False)

    net.eval()
    with torch.no_grad():
        for i, (X_test, y_test, sources_test, coefficients_test) in enumerate(test_dataloader):
            batch_size = X_test.size()[0]
            y_pred_test = net(X_test)
            if field_target == "all" and model_name == "SimI2VSepDecNoSkip":
                y_pred_test = y_pred_test.view(batch_size, seq_len, 6 * out_channels, *y_pred_test.size()[-2:])
            elif field_target == "ezhxhy" and model_name == "SimI2VSepDecNoSkip":
                y_pred_test = y_pred_test.view(batch_size, seq_len, 3 * out_channels, *y_pred_test.size()[-2:])
            lz, rz = maxwell_ezhxhy(y_pred_test, coefficients_test, sources_test)
            loss_ph, loss_data = custom_loss_fn(y_pred_test, y_test, lz, rz)
            loss = loss_ph * PH_FACTOR + loss_data
            print(f"Final Loss: {loss.item()}")   
            if field_target == "all":
                for j in range(6 * out_channels):
                    plot(y_pred_test[0], y_test[0], num_epochs, "test_final", channel=j, channel_name=LABELS[j // 2])
            elif field_target == "ezhxhy":
                for j in range(3 * out_channels):
                    plot(y_pred_test[0], y_test[0], num_epochs, "test_final", channel=j, channel_name=LABELS[j // 2])
                # plot lz rz
                plot_lz_rz(lz[0], rz[0], num_epochs, "train", channel=0)
            else:
                for j in range(6 * out_channels if field_target == "all" else out_channels):
                    plot(y_pred_test[0], y_test[0], num_epochs, "test_final", channel=j, channel_name=field_target)
            
if __name__ == "__main__":
    train()