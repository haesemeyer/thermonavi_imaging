"""
Script to fit MINE on Suite2P segmented imaging data from Kaarthik's flow setup
"""

import argparse
import os
from os import path
import numpy as np
import h5py
from tifffile import imread
from rwave_data_fit import interpolate_rows
import mine
import re
from utilities import safe_standardize
from typing import Tuple
from scipy.ndimage import gaussian_filter1d


def get_frame_duration(file_name: str) -> float:
    info_file = open(file_name, 'rt')
    all_contents = info_file.readlines()
    info_file.close()
    for line in all_contents:
        if "Frame duration" in line:
            v = line.split(':')[-1].strip()
            return float(v)
    return np.nan


def process_plane(exp_s2p_dir: str, ct: float, weight_group: h5py.Group) -> (
        Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Process data from one plane
    :param exp_s2p_dir: The suite2p base directory of the experimental plane
    :param ct: The test correlation threshold for mine
    :param weight_group: HDF5 group to store model weights in
    :return:
        [0]: n_neurons x time matrix of neurons that were identified as heat responsive
        [1]: n_neurons vector of indices of identified neurons
        [2]: n_neurons x time matrix of deconvolved spike data of neurons that were identified as heat responsive
    """

    root_dir, s2p_name = path.split(exp_s2p_dir)
    # some experiments (TG) will have mask files to identify valid regions (Pixel > 0) for neurons to be analyzed
    mask_file = path.join(root_dir, s2p_name + ".mask")
    has_mask = path.exists(mask_file)
    # find actual experiment name - the folder name comes from analyzing a single plane and therefore should have
    # a suffix with the structure: _Z_#_0 where # is an arbitrary number of unknown number of digits
    suffix = re.search(r'_Z_\d+_0$', s2p_name)
    if suffix is None:
        raise Exception(f"Could not identify file name based on suffix search of suite2p folder {s2p_name}")
    exp_name = s2p_name[:suffix.start()]
    # the name of the plane should be the simply s2p_name without the last _0
    plane_name = s2p_name[:-2]
    # get frame-duration from info file
    info_file = path.join(root_dir, exp_name + ".info")
    frame_duration = get_frame_duration(info_file)
    if np.isnan(frame_duration):
        raise Exception(f"Could not determine frame duration of experiment {exp_name}")
    # load temperature data
    temperature_file = path.join(root_dir, plane_name + ".temp")
    temperature = np.genfromtxt(temperature_file)
    # filter temperature to remove some of the measurement noise
    temperature = gaussian_filter1d(temperature, 2)
    m_temp = np.mean(temperature)
    s_temp = np.std(temperature)
    # load calcium data of suite2p identified neurons
    s2p_out_folder = path.join(path.join(exp_s2p_dir, "suite2p"), "plane0")
    if not has_mask:
        is_cell = np.load(path.join(s2p_out_folder, "iscell.npy"))[:, 1] > .5  # boolean vector of cells
    else:
        is_cell = np.load(path.join(s2p_out_folder, "iscell.npy"))[:, 1] >= .4  # lower cell ident threshold in masked
    if not np.any(is_cell):
        print()
        print(f"Suite2p did not identify any neurons in {exp_name}")
        print()
        return None, None, None, None, None, None
    ca = np.load(path.join(s2p_out_folder, "F.npy"))[is_cell]
    spk = np.load(path.join(s2p_out_folder, "spks.npy"))[is_cell]
    ca_frame_times = np.arange(ca.shape[1]) * frame_duration
    interp_times = np.arange((ca_frame_times[-1] * 1000) // 200) * 0.2
    i_ca = interpolate_rows(interp_times, ca_frame_times, ca)
    i_spk = interpolate_rows(interp_times, ca_frame_times, spk)
    # process stimulus data
    stim_times = np.arange(temperature.size) / 20  # temperature measurements are stored at 20Hz
    i_stim_temp = np.interp(interp_times, stim_times, temperature)

    i_stim_temp = (i_stim_temp - m_temp) / s_temp
    i_ca = safe_standardize(i_ca, 1)

    # if a mask is present, prune cells based on mask values
    if has_mask:
        s2p_stats = np.load(path.join(s2p_out_folder, "stat.npy"), allow_pickle=True)[is_cell]
        mask_tif = imread(mask_file)
        inside_mask = np.full(s2p_stats.shape[0], False)
        for put_cell in range(s2p_stats.shape[0]):
            my, mx = s2p_stats[put_cell]["med"]
            inside_mask[put_cell] = mask_tif[my, mx] > 0
        print()
        print(f"Mask present for {exp_name}")
        print(f"Out of {s2p_stats.shape[0]} neurons {inside_mask.sum()} neurons are inside the mask.")
        print()
        i_ca = i_ca[inside_mask]
        i_spk = i_spk[inside_mask]
    else:
        print()
        print(f"No mask present for {exp_name}")
        print()

    # run MINE
    # miner = mine.Mine(2/3, 50, ct, False, False, 25, 5)
    # Only allow for 2 seconds model history to make this more compatible with behavioral timescales
    miner = mine.Mine(2/3, 50, ct, False, False, 25, 5)
    miner.verbose = True
    miner.n_epochs = 150
    miner.model_weight_store = weight_group
    # mdata = miner.analyze_data([i_stim_temp], i_ca, generate_validator(m_temp, s_temp, hot))
    mdata = miner.analyze_data([i_stim_temp], i_ca)
    neurons_fit = mdata.correlations_test >= ct
    print(f"In experiment {exp_name}, identified {np.sum(neurons_fit)} neurons.")
    # taylor_scores_raw = mdata.taylor_scores
    # min_significance = 1 - 0.05 / np.sum(neurons_fit)
    # normal_quantiles_by_sigma = np.array([0.682689492137, 0.954499736104, 0.997300203937, 0.999936657516,
    #                                       0.999999426697, 0.999999998027])
    # n_sigma = np.where((min_significance - normal_quantiles_by_sigma) < 0)[0][0] + 1
    # taylor_sig = taylor_scores_raw[:, :, 0] - n_sigma * taylor_scores_raw[:, :, 1] - st
    # taylor_scores = taylor_scores_raw[:, :, 0]
    # taylor_scores[taylor_sig <= 0] = 0
    # is_temp_responsive = np.logical_and(neurons_fit, taylor_scores[:, 0] > 0)
    # print(f"In experiment {exp_name} under condition {'hot' if hot else 'cold'}, identified {np.sum(neurons_fit)} temperature responsive neurons.")
    return i_ca[neurons_fit], np.arange(i_ca.shape[0])[neurons_fit].astype(int), i_spk[neurons_fit], mdata.correlations_test[neurons_fit], m_temp, s_temp


if __name__ == '__main__':
    # the following will prevent tensorflow from using the GPU - as the used models have very low complexity
    # they will generally be fit faster on the CPU - furthermore parallelization currently used
    # will not work if tensorflow is run on the GPU!!
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="collect_HtColdFlow_S2P_TempRes",
                                       description="Uses MINE to identify heat responsive neurons "
                                                   "after Suite2P segmentation")
    a_parser.add_argument("-d", "--directory", help="Path to 2P date folder", type=str, required=True)
    a_parser.add_argument("-ct", "--corr_thresh", help="Correlation threshold to consider fit",
                          type=float, default=np.sqrt(0.5))
    a_parser.add_argument("-si", "--sigthresh", help="The threshold for taylor metric significance",
                          type=float, default=0.1)

    args = a_parser.parse_args()

    date_dir = args.directory
    c_thresh = args.corr_thresh
    sig_thresh = args.sigthresh

    # collect subfolders containing suite-2p analysis directories
    all_items = os.listdir(date_dir)
    suite2p_dirs = [path.join(date_dir, d) for d in all_items if os.path.isdir(path.join(date_dir, d))]
    # only keep those that actually have a suite2p folder
    suite2p_dirs = [d for d in suite2p_dirs if path.isdir(path.join(d, "suite2p"))]
    # process experiments
    with h5py.File(path.join(date_dir, path.split(date_dir)[-1]+'.hdf5'), 'w') as dfile:
        for s2pd in suite2p_dirs:
            qual_name = path.split(s2pd)[-1]
            # Full
            hot_weight_group = dfile.create_group(qual_name+"_full_weights")
            heat_res_neurons, heat_res_indices, heat_res_spikes, heat_res_tcorr, temp_mean, tmp_std = process_plane(
                s2pd, c_thresh, hot_weight_group)
            if heat_res_neurons is None:
                continue
            if heat_res_neurons.size > 0:
                grp = dfile.create_group(qual_name+"_full")
                grp.create_dataset('heat_res_neurons', data=heat_res_neurons)
                grp.create_dataset('heat_res_indices', data=heat_res_indices)
                grp.create_dataset('heat_res_spikes', data=heat_res_spikes)
                grp.create_dataset('heat_res_testcorr', data=heat_res_tcorr)
                grp.create_dataset('temp_mean', data=temp_mean)
                grp.create_dataset('tmp_std', data=tmp_std)
            # # Hot
            # hot_weight_group = dfile.create_group(qual_name + "_hot_weights")
            # heat_res_neurons, heat_res_indices = process_experiment(s2pd, c_thresh, sig_thresh, True,
            #                                                         hot_weight_group)
            # if heat_res_neurons.size > 0:
            #     grp = dfile.create_group(qual_name + "_hot")
            #     grp.create_dataset('heat_res_neurons', data=heat_res_neurons)
            #     grp.create_dataset('heat_res_indices', data=heat_res_indices)
            # # Cold
            # cold_weight_group = dfile.create_group(qual_name+"_cold_weights")
            # heat_res_neurons, heat_res_indices = process_experiment(s2pd, c_thresh, sig_thresh, False, cold_weight_group)
            # if heat_res_neurons.size > 0:
            #     grp = dfile.create_group(qual_name+"_cold")
            #     grp.create_dataset('heat_res_neurons', data=heat_res_neurons)
            #     grp.create_dataset('heat_res_indices', data=heat_res_indices)
