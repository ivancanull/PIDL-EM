import torch.nn as nn
import planetzoo as pnz

def build_fno2d_rnn(in_channels, out_channels, seq_len, device, dim=128, 
                    kernel_list=[64, 64, 128, 128, 128], kernel_size_list=[1, 1, 1, 1, 1], 
                    padding_list=[0, 0, 0, 0, 0], hidden_list=[128], 
                    mode_list=[(16, 16), (16, 16), (16, 16), (16, 16), (16, 16)], 
                    act_func="ReLU"):
    """Build FNO2DRNN model"""
    return pnz.model.neuralop.FNO2DRNN(
        in_channels=in_channels, out_channels=out_channels, seq_len=seq_len, 
        dim=dim, kernel_list=kernel_list, kernel_size_list=kernel_size_list, 
        padding_list=padding_list, hidden_list=hidden_list, mode_list=mode_list, 
        act_func=act_func
    ).to(device)

def build_fno2d_gru(in_channels, out_channels, seq_len, device, dim=64,
                    kernel_list=[64, 64, 64, 64, 64], kernel_size_list=[1, 1, 1, 1, 1],
                    padding_list=[0, 0, 0, 0, 0], hidden_list=[64],
                    mode_list=[(16, 16), (16, 16), (16, 16), (16, 16), (16, 16)],
                    act_func="ReLU", unet=False):
    """Build FNO2DGRU model"""
    return pnz.model.neuralop.FNO2DGRU(
        in_channels=in_channels, out_channels=out_channels, seq_len=seq_len,
        dim=dim, kernel_list=kernel_list, kernel_size_list=kernel_size_list,
        padding_list=padding_list, hidden_list=hidden_list, mode_list=mode_list,
        act_func=act_func, unet=unet
    ).to(device)

def build_fno2d(in_channels, out_channels, device, dim=128,
                kernel_list=[64, 64, 128, 128, 128], kernel_size_list=[1, 1, 1, 1, 1],
                padding_list=[0, 0, 0, 0, 0], hidden_list=[128],
                mode_list=[(16, 16), (16, 16), (16, 16), (16, 16), (16, 16)],
                act_func="ReLU"):
    """Build FNO2D model"""
    return pnz.model.neuralop.FNO2D(
        in_channels=in_channels, out_channels=out_channels,
        dim=dim, kernel_list=kernel_list, kernel_size_list=kernel_size_list,
        padding_list=padding_list, hidden_list=hidden_list, mode_list=mode_list,
        act_func=act_func
    ).to(device)

def build_unet_lstm(in_channels, out_channels, device, 
                    hidden_channels=[32, 64, 128, 256], norm_layer=nn.BatchNorm2d):
    """Build UNetLSTM model"""
    return pnz.model.UNetLSTM(
        in_channels=in_channels, hidden_channels=hidden_channels,
        out_channels=out_channels, norm_layer=norm_layer
    ).to(device)

def build_conv_lstm(in_channels, out_channels, device, 
                    hidden_channels=None, batch_first=True, return_all_layers=False):
    """Build ConvLSTM model"""
    if hidden_channels is None:
        hidden_channels = [32, 32, 32, out_channels]
    return pnz.model.ConvLSTM(
        in_channels=in_channels, hidden_channels=hidden_channels,
        batch_first=batch_first, return_all_layers=return_all_layers
    ).to(device)

def build_simi2v_sep_dec_no_skip(in_channels, out_channels, seq_len, device, hidden_channels):
    """Build SimI2VSepDecNoSkip model"""
    return pnz.model.SimI2VSepDecNoSkip(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        seq_len=seq_len
    ).to(device)

def build_simi2v_connect_dec_no_skip(in_channels, out_channels, seq_len, device, 
                                     hidden_channels, groups):
    """Build SimI2VConnectDecNoSkip model"""
    return pnz.model.SimI2VConnectDecNoSkip(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        groups=groups,
        out_channels=out_channels,
        seq_len=seq_len
    ).to(device)

def build_simi2v(in_channels, out_channels, seq_len, device, hidden_channels, groups):
    """Build SimI2V model"""
    return pnz.model.SimI2V(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        groups=groups,
        out_channels=out_channels,
        seq_len=seq_len
    ).to(device)

def build_model(model_type, in_channels, out_channels, device, seq_len=None, **kwargs):
    """
    Build a neural network model based on the specified type.
    
    Args:
        model_type (str): Type of model to build ('fno2d', 'fno2d_rnn', 'fno2d_gru', 'unet_lstm', 'conv_lstm', 'simi2v_sep_dec_no_skip', 'simi2v_connect_dec_no_skip', 'simi2v')
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels  
        device: Device to place the model on
        seq_len (int): Sequence length (required for FNO models and SimI2V models)
        **kwargs: Additional model-specific parameters
    
    Returns:
        Neural network model
    """
    if model_type == 'fno2d':
        return build_fno2d(in_channels, out_channels, device, **kwargs)
    elif model_type == 'fno2drnn':
        return build_fno2d_rnn(in_channels, out_channels, seq_len, device, **kwargs)
    elif model_type == 'fno2dgru':
        return build_fno2d_gru(in_channels, out_channels, seq_len, device, **kwargs)
    elif model_type == 'unetlstm':
        return build_unet_lstm(in_channels, out_channels, device, **kwargs)
    elif model_type == 'convlstm':
        return build_conv_lstm(in_channels, out_channels, device, **kwargs)
    elif model_type == 'simi2vsepdecnoskip':
        return build_simi2v_sep_dec_no_skip(in_channels, out_channels, seq_len, device, **kwargs)
    elif model_type == 'simi2vconnectdecnoskip':
        return build_simi2v_connect_dec_no_skip(in_channels, out_channels, seq_len, device, **kwargs)
    elif model_type == 'simi2v':
        return build_simi2v(in_channels, out_channels, seq_len, device, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
