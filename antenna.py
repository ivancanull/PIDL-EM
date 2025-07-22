import sys
import os
import time
from turtle import fd

from src.fdtd import *
from src.utils import *
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
    parser.add_argument('--fmax', type=float, default=10e9,
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

def create_invert_f_antenna(air_buffers, npml, l1, l2, l3, l4):
    """ Get grid for IFA. """
    # Define FDTD grid
    cell_lengths = (0.4e-3, 0.4e-3, 0.262e-3)
    obj_lengths = (100 * cell_lengths[0],
                   100 * cell_lengths[1],
                   3 * cell_lengths[2])
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
    # Define antenna
    grid[0:100, 0:100, 0:3] = UniformBrick(epsr=2.2) 
    grid[0:46+l2, 60:66, 3] = PECPlate('z') 
    grid[40:46+l2, 75:81, 3] = PECPlate('z')
    grid[40+l2:46+l2, l4:81, 3] = PECPlate('z')
    grid[40+l1:46+l1, l3:81, 3] = PECPlate('z')
    grid[40, 75:81, 0:3] = PECPlate('x')
    grid[0:40, 0:100, 0] = PECPlate('z')
    grid[0, 60:66, 0:3] = VoltageSource(1., 50., 'zp')

    grid[0, 61:66, 0:3] = VoltageMonitor('zp')
    grid[0, 60:66, 2] = CurrentMonitor('zp')
    
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
    l3 = args.l3
    l4 = args.l4

    

    name = f"{num2str(l1)}_{num2str(l2)}_{num2str(l3)}_{num2str(l4)}"

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

    grid_helper, origin = create_invert_f_antenna(air_buffers, npml, l1, l2, l3, l4)

    antenna = Antenna(grid_helper)

    if args.plot:
        img = np.zeros(grid_helper.cell_numbers)
        for obj in grid_helper.objects_on_faces:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            img[i_s:i_e+1, j_s:j_e+1, k_s:k_e+1] = 1
        
        img = img[origin[0]:-origin[0]+1, origin[1]:-origin[1]+1, -origin[2]]
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0, vmax=2)
        fig.savefig(f"antenna.png")
        fig.savefig(f"antenna.pdf")

    # plot surface example
    img = antenna.generate_surface_img(origin)
    if not os.path.exists("example_surface.png"):
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0, vmax=2)
        fig.savefig(f"example_surface.png")
        fig.savefig(f"example_surface.pdf")

    np.savez_compressed(os.path.join(surfaces_dir, f"{name}_surfaces.npz"), img=img)

    epsr, sige = antenna.construct() # (x, y, z) 

    ns = len(grid_helper.sources_on_edges)

    cpml = CFSParameters(npml=npml, alpha_max=0.05, sigma_factor=1.3, kappa_max=7, order=3)

    dt = estimate_time_interval(grid_helper.cell_lengths, cfl_number, epsr_min=1., mur_min=1.)
    
    # print(f"Estimated time interval: {dt:4e} s")
    
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
    # if not os.path.exists(os.path.join(args.result_dir, "s_parameters", f'microstrip_filter_s_parameters_{num2str(l1)}_{num2str(l2)}_{num2str(l3)}_{num2str(l4)}.npz')) or args.force:
    if not os.path.exists(os.path.join(fields_dir, f"{name}_fields.npz")) or args.force:
    # define solver
        solver = SParameterSolver(fdtd_net)

        # solve
        _ = solver.solve(waveform_t, time_estimation=False)

        # save field
        if fdtd_net.field_record_sampler is not None:
            fdtd_net.field_record_sampler.save_numpy(fields_dir)

        # save sources
        if fdtd_net.source_record_sampler is not None:
            fdtd_net.source_record_sampler.save_numpy(surfaces_dir)

        # save coefficients
        if fdtd_net.coefficient_recorder is not None:
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
    if os.path.exists('l1_l2_l3_l4.csv'):
        df = pd.read_csv('l1_l2_l3_l4.csv')
        return df
    else:
        items = []
        i = 0
        for l1 in range(10, 14):
            for l2 in range(24, 28):
                for l3 in range(40, 44):
                    for l4 in range(20, 24):
                        items.append({'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4})
                        
        df = pd.DataFrame(items)
        df.to_csv('l1_l2_l3_l4.csv', index=True)
        return df

if __name__ == '__main__':
    args_ = parse_args()
    para_df = create_dataset()

    start = time.time()    
    for i in tqdm(range(len(para_df))):

        args_.l1 = int(para_df.loc[i, 'l1']) # type: ignore
        args_.l2 = int(para_df.loc[i, 'l2']) # type: ignore
        args_.l3 = int(para_df.loc[i, 'l3']) # type: ignore
        args_.l4 = int(para_df.loc[i, 'l4']) # type: ignore
        args_.plot = True if i == 0 else False
        
        solve(args_)
        

    end = time.time()
    print(f"Total time: {end - start} s")
