# test distributed data on slurm
import sys
import os
from typing import Sequence

import torch.distributed

sys.path.append(os.path.abspath('..'))

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

# PLOT_EXAMPLES = 6
# NUM_EXAMPLES = 1
# FIELD_TARGET = "ezhxhy" # ez | all | ezhxhy
# SEQ_LEN = 25
# MODEL_NAME = "SimI2V"
# IMG_INPUT = True
# DDP = True
# LABELS = ["ex", "ey", "ez", "hx", "hy", "hz"]
# TRAIN_RATIO = 0.2
# PH_FACTOR = 0
# SAMPLE_DT = 10
# if FIELD_TARGET == "ezhxhy":
#     LABELS = ["ez", "hx", "hy"]

# # LOSS_BALANCING = "RELOBRALO"
# LOSS_BALANCING = "RELOBPECH"
# BETA = 0.1
# LOSS_PH_THRESHOLD = 0.02
# WAIT = 10
# cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
# cfl_number = 0.9
# dt = estimate_time_interval(cell_lengths, cfl_number, epsr_min=1., mur_min=1.)

# dte = dt / epsilon0
# dtm = dt / mu0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    parser.add_argument('-f', '--force', action='store_true', help='retrain')
    return parser.parse_args()

def train():
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cell_lengths = configs.cell_lengths
    cfl_number = configs.cfl_number
    dt = estimate_time_interval(cell_lengths, cfl_number, epsr_min=1., mur_min=1.)
    dte = dt / epsilon0
    dtm = dt / mu0

    air_buffers = configs.air_buffers
    dataset_dir = configs.dir.data
    train_ratio = configs.train_ratio
    seq_len = configs.seq_len
    air_buffers=configs.air_buffers

    # concat l1, l2, d from configs.vars and combine with '_'
    vars_str = vars2str(configs.vars)
    
    # TODO: use custom pytorch dataset to load data

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
                                return_mode="regular")
    test_dataset = PinnDataset(dataset=test,
                               air_buffers=air_buffers,
                               variables=configs.vars,
                               device=device,
                               field_target=field_target,
                               seq_len=seq_len,
                               img_input=True,
                               result_dir=f'{dataset_dir}/result',
                               return_mode="regular")
    
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    out_channels = 2
    if hasattr(configs, 'loss_balancing'):
        if configs.loss_balancing.name == "RELOBRALO":
            loss_balancing = ReLoBRaLo()
            loss_balancing_mark = f'{configs.loss_balancing.name}'
        elif configs.loss_balancing.name == "RELOBPECH":
            loss_balancing = ReLoBPeCh(beta=configs.loss_balancing.beta, wait=configs.loss_balancing.wait)
            loss_balancing_mark = f'{configs.loss_balancing.name}_beta_{str(configs.loss_balancing.beta).replace(".", "d")}'
        else:
            raise NotImplementedError(f"Loss balancing {configs.loss_balancing.name} is not implemented.")
    else:
        loss_balancing = None
        loss_balancing_mark = f'ph_{str(ph_factor).replace(".", "d")}'

    model_name = configs.model.name.lower()

    # Calculate seq_len multiplier based on field_target
    seq_len_multiplier = 1
    if field_target == "all":
        seq_len_multiplier = 6
    elif field_target == "ezhxhy":
        seq_len_multiplier = 3

    # Build model using centralized function
    if model_name == "simi2vsepdecnoskip":
        net = build_model(
            model_type="simi2vsepdecnoskip",
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            device=device,
            seq_len=seq_len * seq_len_multiplier,
            hidden_channels=configs.model.hidden_channels
        )
    elif model_name == "simi2vconnectdecnoskip":
        if field_target == "ezhxhy":
            net = build_model(
                model_type="simi2vconnectdecnoskip",
                in_channels=configs.model.in_channels,
                out_channels=configs.model.out_channels,
                device=device,
                seq_len=seq_len,
                hidden_channels=configs.model.hidden_channels,
                groups=configs.model.groups
            )
        else:
            raise NotImplementedError("Not implemented yet.")
    elif model_name == "simi2v":
        if field_target == "ezhxhy":
            net = build_model(
                model_type="simi2v",
                in_channels=configs.model.in_channels,
                out_channels=configs.model.out_channels,
                device=device,
                seq_len=seq_len,
                hidden_channels=configs.model.hidden_channels,
                groups=configs.model.groups
            )  
        else:
            raise NotImplementedError("Not implemented yet.")
        
    net.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                           func_args={"mode": "fan_in", "nonlinearity": "relu"})
        
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)

    maxwell_ezhxhy = Maxwell_EzHxHy(dx=cell_lengths[0], 
                                    dy=cell_lengths[1], 
                                    dz=cell_lengths[2], 
                                    dte=dte, 
                                    delta_t=configs.sample_dt,
                                    nz=out_channels)
    
    
    custom_loss_fn = CustomLossEzHxHy(loss_fn=torch.nn.MSELoss())

    not_read_ckpt = args.force
    num_epochs = 10000
        
    ckpt_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{loss_balancing_mark}_pinn_ckpt.pt"
    train_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{loss_balancing_mark}_pinn_train_loss.csv"
    test_loss_df_path = f"./{configs.dir.model}/{model_name.lower()}_{field_target}_{str(train_ratio).replace('.', 'd')}_{loss_balancing_mark}_pinn_test_loss.csv"
    train_loss_df = pd.DataFrame()
    test_loss_df = pd.DataFrame()
    epoch_continue = 0

    if not not_read_ckpt:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt['epoch']
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if loss_balancing is not None and 'loss_balancing' in ckpt:
                loss_balancing.load_state_dict(ckpt['loss_balancing'])
            # processor = torch.load(ckpt['processor'])
            print(f"Continue training from epoch {epoch_continue + 1}.")
        if os.path.exists(train_loss_df_path):
            train_loss_df = pd.read_csv(train_loss_df_path, index_col=False)
        if os.path.exists(test_loss_df_path):
            test_loss_df = pd.read_csv(test_loss_df_path, index_col=False)
    
    net.train()
    
    epoch_pbar = tqdm(range(epoch_continue, num_epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        
        train_loss = train_data_loss = train_ph_loss = 0.0
        train_size = 0
        for i, (X_train, y_train, sources_train, coefficients_train) in enumerate(train_dataloader):
            batch_size = X_train.size()[0]
            y_pred_train = net(X_train)
            
            if model_name == "simi2vsepdecnoskip":
                y_pred_train = y_pred_train.view(batch_size, seq_len, seq_len_multiplier * out_channels, *y_pred_train.size()[-2:])

            lz, rz = maxwell_ezhxhy(y_pred_train, coefficients_train, sources_train)
            loss_ph, loss_data = custom_loss_fn(y_pred_train, y_train, lz, rz)

            if loss_balancing:
                losses = [loss_ph.item() * 10, loss_data.item()] # 10 is a magic number
                factors = loss_balancing.update(epoch, losses)
                loss = loss_ph * factors[0] + loss_data * factors[1] * 10
            else:
                loss = loss_ph * ph_factor + loss_data

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
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({'train_loss': f'{loss:.6f}', 'data_loss': f'{data_loss:.6f}', 'ph_loss': f'{ph_loss:.6f}'})

        if loss_balancing:
            new_row = {'epoch': epoch + 1, 'loss': loss, 'data_loss': data_loss, 'ph_loss': ph_loss, 'ph_factor': factors[0], 'data_factor': factors[1]}
        else:
            new_row = {'epoch': epoch + 1, 'loss': loss, 'data_loss': data_loss, 'ph_loss': ph_loss}
        
        if train_loss_df.empty:
            train_loss_df = pd.DataFrame([new_row])
        else:
            train_loss_df = pd.concat([train_loss_df, pd.DataFrame([new_row])], ignore_index=True)

        if epoch % 10 == 9:

            # loss_info = f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {loss}'
            # print(loss_info)
            with torch.no_grad():
                test_loss = test_data_loss = test_ph_loss = 0.0
                test_size = 0
                for i, (X_test, y_test, sources_test, coefficients_test) in enumerate(test_dataloader):
                    batch_size = X_test.size()[0]
                    y_pred_test = net(X_test)
                    if model_name == "simi2vsepdecnoskip":
                        y_pred_test = y_pred_test.view(batch_size, seq_len, seq_len_multiplier * out_channels, *y_pred_test.size()[-2:])
                    lz, rz = maxwell_ezhxhy(y_pred_test, coefficients_test, sources_test)
                    loss_ph, loss_data = custom_loss_fn(y_pred_test, y_test, lz, rz)
                    
                    loss = loss_ph + loss_data
                    test_loss += loss.item() * X_test.shape[0]
                    test_data_loss += loss_data.item() * X_test.shape[0]
                    test_ph_loss += loss_ph.item() * X_test.shape[0]
                    test_size += X_test.shape[0] 

                test_loss = test_loss / test_size
                data_loss = test_data_loss / test_size
                ph_loss = test_ph_loss / test_size
                new_test_row = {'epoch': epoch + 1, 'loss': test_loss, 'data_loss': data_loss, 'ph_loss': ph_loss}
                if test_loss_df.empty:
                    test_loss_df = pd.DataFrame([new_test_row])
                else:
                    test_loss_df = pd.concat([test_loss_df, pd.DataFrame([new_test_row])], ignore_index=True)
                # Update epoch progress bar with test loss
                epoch_pbar.set_postfix({'train_loss': f'{train_loss/train_size:.6f}', 'test_loss': f'{test_loss:.6f}', 'test_data_loss': f'{data_loss:.6f}', 'test_ph_loss': f'{ph_loss:.6f}'})

                test_loss_df = pd.concat([test_loss_df, pd.DataFrame({'epoch': epoch + 1, 'loss': test_loss, 'data_loss': data_loss, 'ph_loss': ph_loss}, index=[0])], ignore_index=True)

                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_balancing': loss_balancing.state_dict() if loss_balancing else None,
                    # 'processor': processor
                }, ckpt_path)
                train_loss_df.to_csv(train_loss_df_path, index=False) 
                test_loss_df.to_csv(test_loss_df_path, index=False)

    net.eval()
    with torch.no_grad():
        for i, (X_test, y_test, sources_test, coefficients_test) in enumerate(test_dataloader):
            batch_size = X_test.size()[0]
            y_pred_test = net(X_test)
            if model_name == "simi2vsepdecnoskip":
                y_pred_test = y_pred_test.view(batch_size, seq_len, seq_len_multiplier * out_channels, *y_pred_test.size()[-2:])
            lz, rz = maxwell_ezhxhy(y_pred_test, coefficients_test, sources_test)
            loss_ph, loss_data = custom_loss_fn(y_pred_test, y_test, lz, rz)
            loss = loss_ph + loss_data
            print(f"Final Loss: {loss.item()}")   

if __name__ == "__main__":
    train()