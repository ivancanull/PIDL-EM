import sys
import os
import time
sys.path.append(os.path.abspath('..'))

from src import *

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        description='FDTD-Based Electromagnetics Forward-Problem Solver')
    parser.add_argument('--device_target', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--nt', type=int, default=1000,
                        help='Number of time steps.')
    parser.add_argument('--max_call_depth', type=int, default=1000)
    # parser.add_argument('--dataset_dir', type=str,
    #                     default='./dataset', help='dataset directory')
    parser.add_argument('--result_dir', type=str,
                        default='./result', help='result directory')
    parser.add_argument('--cfl_number', type=float, default=0.9, help='CFL number')
    parser.add_argument('--fmax', type=float, default=20e9,
                        help='highest frequency (Hz)')
    parser.add_argument('--force', action='store_true', help='force to avoid skip')
    options = parser.parse_args()
    return options

def get_waveform_t(nt, dt, fmax):
    """
    Compute waveforms at time t.

    Args:
        nt (int): Number of time steps.
        dt (float): Time interval.
        fmax (float): Maximum freuqency of Gaussian wave

    Returns:
        waveform_t (Tensor, shape=(nt,)): Waveforms.
    """

    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t, t

def create_parameterized_microstrip_filter(air_buffers, npml, l1, l2, d=6):
    """ microstrip filter """
    cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
    obj_lengths = (50 * cell_lengths[0],
                   46 * cell_lengths[1],
                   1 * cell_lengths[2])
    cell_numbers = (
        2 * npml + 2 * air_buffers[0] + int(obj_lengths[0] / cell_lengths[0]),
        2 * npml + 2 * air_buffers[1] + int(obj_lengths[1] / cell_lengths[1]),
        2 * npml + 2 * air_buffers[2] + int(obj_lengths[2] / cell_lengths[2]),
    )

    origin = (
        npml + air_buffers[0],
        npml + air_buffers[1],
        npml + air_buffers[2],
    )
    grid = GridHelper(cell_numbers, cell_lengths, origin=origin)

    assert l1 >= 0 & l2 - d >= 0
    assert l1 + d <= 50 & l2 <= 50

    # Define antenna
    grid[0:50, 0:46, 0:1] = UniformBrick(epsr=2.2)
    grid[l1:l1+d, 0:20, 1] = PECPlate('z')
    grid[l2-d:l2, 26:46, 1] = PECPlate('z')
    grid[0:50, 20:26, 1] = PECPlate('z')
    grid[0:50, 0:46, 0] = PECPlate('z')

    # Define sources
    grid[l1:l1+d, 0, 0:1] = VoltageSource(1., 50., 'zp')

    # Define load
    grid[l2-d:l2, 46, 0:1] = Resistor(50., 'z')

    # Define monitors
    grid[l1:l1+d, 10, 0:1] = VoltageMonitor('zp')
    grid[l1:l1+d, 10, 1] = CurrentMonitor('yp')
    grid[l2-d:l2, 36, 0:1] = VoltageMonitor('zp')
    grid[l2-d:l2, 36, 1] = CurrentMonitor('yn')

    return grid, origin

def solve(args):
    """solve process"""

    # set up problem
    nt = args.nt
    fmax = args.fmax
    cfl_number = args.cfl_number
    air_buffers = (3, 3, 3)
    npml = 8

    l1 = args.l1
    l2 = args.l2
    d = args.d

    name = f"{num2str(l1)}_{num2str(l2)}_{num2str(d)}"

    s_para_dir = os.path.join(args.result_dir, "s_parameters")
    fields_dir = os.path.join(args.result_dir, "fields")
    surfaces_dir = os.path.join(args.result_dir, "surfaces")
    coefficents_dir = os.path.join(args.result_dir, "coefficients")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(s_para_dir):
        os.makedirs(s_para_dir)
    if not os.path.exists(fields_dir):
        os.makedirs(fields_dir)
    if not os.path.exists(surfaces_dir):
        os.makedirs(surfaces_dir)
    if not os.path.exists(coefficents_dir):
        os.makedirs(coefficents_dir)

    grid_helper, origin = create_parameterized_microstrip_filter(air_buffers, npml, l1, l2, d)
    antenna = Antenna(grid_helper)

    # plot surface example
    # img = antenna.generate_surface_img(origin)
    # if not os.path.exists("example_surface.png"):
    #     fig, ax = plt.subplots()
    #     ax.imshow(img, cmap="gray", vmin=0, vmax=2)
    #     fig.savefig(f"example_surface.png")
    #     fig.savefig(f"example_surface.pdf")

    np.savez_compressed(os.path.join(surfaces_dir, f"{name}_surfaces.npz"), img=img)

    epsr, sige = antenna.construct() # (x, y, z) 

    ns = len(grid_helper.sources_on_edges)

    cpml = CFSParameters(npml=npml, alpha_max=0.05, sigma_factor=1.3, kappa_max=7, order=3)

    dt = estimate_time_interval(grid_helper.cell_lengths, cfl_number, epsr_min=1., mur_min=1.)
        
    waveform_t, t = get_waveform_t(nt, dt, fmax)

    # sampling frequencies
    fs = np.linspace(0., fmax, 1001, endpoint=True)

    # define fdtd network
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    sample_num = {"t": 25}
    sample_dt = 10

    if sample_num is not None:
        field_record_sampler = FieldRecordSampler(name,
                                                    sample_method="time_continuous",
                                                    sample_num=sample_num,
                                                    nt=nt,
                                                    dt=sample_dt,)
        source_record_sampler = SourceRecordSampler(name,
                                                    sample_method="time_continuous",
                                                    sample_num=sample_num,
                                                    nt=nt,
                                                    dt=sample_dt)
    else:
        field_record_sampler = FieldRecorder(name)
        source_record_sampler = SourceRecorder(name)

    coefficient_recorder = CoefficientRecorder(name)

    fdtd_net = full3d.ADFDTD(cell_numbers=grid_helper.cell_numbers, 
                             cell_lengths=grid_helper.cell_lengths, 
                             nt=nt, 
                             dt=dt, 
                             ns=ns, 
                             designer=antenna, 
                             cfs_pml=cpml, 
                             source_record_sampler=source_record_sampler,
                             field_record_sampler=field_record_sampler, 
                             coefficient_recorder=coefficient_recorder,
                             device=device)
    
    # early stop if the file exists
    if not os.path.exists(os.path.join(args.result_dir, "s_parameters", f'microstrip_filter_s_parameters_{num2str(l1)}_{num2str(l2)}_{num2str(d)}.npz')) or args.force:
    # define solver
        solver = SParameterSolver(fdtd_net)

        # solve
        _ = solver.solve(waveform_t, time_estimation=False)

        # save field
        fdtd_net.field_record_sampler.save_numpy(fields_dir)

        # save sources
        fdtd_net.source_record_sampler.save_numpy(surfaces_dir)

        # save coefficients
        fdtd_net.coefficient_recorder.save_numpy(coefficents_dir)
        
        # eval
        # s_parameters = solver.eval(fs, t)

        # show results
        # s_parameters = s_parameters.cpu().numpy()
        # s_complex = s_parameters[..., 0] + 1j * s_parameters[..., 1]
        
        # save results
        # np.savez(os.path.join(args.result_dir, "s_parameters", f'microstrip_filter_s_parameters_{num2str(l1)}_{num2str(l2)}_{num2str(d)}.npz'), s_parameters=s_complex, frequency=fs)

    return

def create_dataset():
    if os.path.exists('l1_l2_d.csv'):
        df = pd.read_csv('l1_l2_d.csv')
        return df
    else:
        items = []
        for l1 in range(5, 15):
            for l2 in range(35, 45):
                for d in range(5, 7):
                    items.append({'l1': l1, 'l2': l2, 'd': d})
        df = pd.DataFrame(items)
        df.to_csv('l1_l2_d.csv', index=True)
        return df

if __name__ == '__main__':
    args_ = parse_args()
    para_df = create_dataset()

    start = time.time()    
    for i in tqdm(range(len(para_df))):

        args_.l1 = int(para_df.loc[i, 'l1'])
        args_.l2 = int(para_df.loc[i, 'l2'])
        args_.d = int(para_df.loc[i, 'd'])

        solve(args_)
    end = time.time()
    print(f"Total time: {end - start} s")
