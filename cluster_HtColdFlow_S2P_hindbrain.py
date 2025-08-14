"""
Script to cluster hot-cold flow experiment temperature responses and model how these responses could encode
delta-deviation.
"""

import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib as mpl
import argparse
import os
from os import path
from sklearn.linear_model import RidgeCV, LinearRegression, LogisticRegressionCV
import seaborn as sns
import scipy.stats as sts
import mine
import utilities
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import model


def compute_mm_delta_temps(temps: np.ndarray, frequency_hz: int) -> np.ndarray:
    """
    Tries to approximate the "previous bout delta-T" in the Markov Model by assigning the change across
    the previous second to each timepoint
    :param temps: The temperatures for which to compute delta-T
    :param frequency_hz: The samples per second for the temps recording
    :return: Matched prev-second delta-T
    """
    delta_temps = np.full_like(temps, np.nan)
    for ix in range(delta_temps.size):
        if ix == 0:
            delta_temps[ix] = 0
        elif ix < frequency_hz:
            delta_temps[ix] = (temps[ix] - temps[0]) * frequency_hz / ix  # extrapolate
        else:
            delta_temps[ix] = temps[ix] - temps[ix - frequency_hz]
    return delta_temps


def tm_from_glm(t: float, dt: float, glm_orig_coef: np.ndarray, glm_icept: np.ndarray) -> pd.DataFrame:
    """
    From temperature and delta-temp inputs uses the glm coefficients to compute the transition matrix
    :param t: The temperature at which to compute the transition matrix
    :param dt: The delta-temperature at which to compute the transition matrix
    :param glm_orig_coef: The glm coefficients of the Markov Model transition
    :param glm_icept: The glm intercepts of the Markov Model transition
    :return: The transition matrix
    """
    trans_mat = np.zeros((3, 3))
    lp = np.zeros(3)

    design = np.array([t, dt, np.abs(dt), np.abs(t), t*dt, t*np.abs(dt), np.abs(t)*dt])
    # Transitions from reversal
    lp[0] = glm_orig_coef[0, :, 0] @ design + glm_icept[0, 0]
    lp[1] = glm_orig_coef[0, :, 1] @ design + glm_icept[0, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[0, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from general
    lp[0] = glm_orig_coef[1, :, 0] @ design + glm_icept[1, 0]
    lp[1] = glm_orig_coef[1, :, 1] @ design + glm_icept[1, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[1, k] = exp_lp[k] / np.sum(exp_lp)
    # Transitions from persistent
    lp[0] = glm_orig_coef[2, :, 0] @ design + glm_icept[2, 0]
    lp[1] = glm_orig_coef[2, :, 1] @ design + glm_icept[2, 1]
    lp[2] = 0
    exp_lp = np.exp(lp)
    for k in range(3):
        trans_mat[2, k] = exp_lp[k] / np.sum(exp_lp)
    return pd.DataFrame(trans_mat, index=["R(o)", "X(o)", "P(o)"], columns=["R(i)", "X(i)", "P(i)"])


def set_journal_style(plot_width_mm=30, plot_height_mm=30, margin_mm=10):
    """
    Set Matplotlib style for journal publication with:
    - Only x and y axes visible.
    - A legend without a bounding box and with elements having only a fill (no stroke).
    - An actual plot area (excluding labels) of at least `plot_width_mm` × `plot_height_mm`.

    Parameters:
    - plot_width_mm (float): Minimum plot area width in mm (default: 30 mm).
    - plot_height_mm (float): Minimum plot area height in mm (default: 30 mm).
    - margin_mm (float): Extra margin for labels and titles (default: 10 mm).
    """
    # Convert mm to inches (1 inch = 25.4 mm)
    fig_width_in = (plot_width_mm + 2 * margin_mm) / 25.4
    fig_height_in = (plot_height_mm + 2 * margin_mm) / 25.4

    pl.rcParams.update({
        'font.size': 7,
        'font.family': 'Arial',
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 1.984,  # 0.7 mm tick length
        'ytick.major.size': 1.984,
        'xtick.minor.size': 1.984,
        'ytick.minor.size': 1.984,
        'savefig.dpi': 300,  # High resolution
        'figure.figsize': (fig_width_in, fig_height_in),
        'savefig.transparent': True,  # transparent background
        'figure.constrained_layout.use': True  # Ensure proper layout
    })


if __name__ == '__main__':

    mpl.rcParams['pdf.fonttype'] = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="cluster_HtColdFlow_S2P",
                                       description="Clusters flow setup temperature responses after Suite2P and MINE")
    a_parser.add_argument("-d", "--directory", help="Path to folder with response Hdf5 file", type=str, required=True)

    args = a_parser.parse_args()

    hdf_dir = args.directory

    hdf_files = [path.join(hdf_dir, f) for f in os.listdir(hdf_dir) if ".hdf5" in f and "cluster" not in f]
    exp_folders = [f[:-5]+"/" for f in hdf_files]
    temp_files = [path.join(ef, f) for ef in exp_folders for f in os.listdir(ef) if ".temp" in f and "cluster" not in f]

    all_temperatures = [np.genfromtxt(tf) for tf in temp_files]
    all_temperatures = [at for at in all_temperatures if at.size > 20000]
    min_len = min([len(at) for at in all_temperatures])
    temperatures = np.mean([at[:min_len] for at in all_temperatures], axis=0)

    # collect and concatenate all responses
    all_temperature_neurons = []
    all_temperature_spikes = []
    all_ids = []
    all_test_correlations = []
    all_plane_ids = []  # simple set of ids where all neurons from one plane receive the same consecutive number
    all_fish_ids = []
    plane_counter = 0
    for fid, f in enumerate(hdf_files):
        with h5py.File(f, "r") as dfile:
            for k in dfile:
                if "weights" in k:
                    continue
                grp = dfile[k]
                responses = grp["heat_res_neurons"][()]
                try:
                    spks = grp["heat_res_spikes"][()]
                except KeyError:
                    spks = responses.copy()
                indices = grp["heat_res_indices"][()]
                test_corrs = grp["heat_res_testcorr"][()]
                all_test_correlations.append(test_corrs)
                # filter by test correlation - rather loosely, let clustering figure out the rest
                responses = responses[test_corrs > 0.6]
                spks = spks[test_corrs > 0.6]
                indices = indices[test_corrs > 0.6]
                all_temperature_neurons.append(responses)
                all_temperature_spikes.append(spks)
                all_ids += [(f, k, ix) for ix in indices]
                all_plane_ids.append(np.full(responses.shape[0], plane_counter))
                all_fish_ids.append(np.full(responses.shape[0], fid))
                plane_counter += 1
    all_temperature_neurons = np.vstack(all_temperature_neurons)[:, :-1]
    all_temperature_spikes = np.vstack(all_temperature_spikes)[:, :-1]
    all_temperature_neurons = utilities.safe_standardize(all_temperature_neurons, 1)
    all_plane_ids = np.hstack(all_plane_ids)
    all_test_correlations = np.hstack(all_test_correlations)
    all_fish_ids = np.hstack(all_fish_ids)

    if (path.exists(path.join(hdf_dir, "cluster_membership.npy")) and
            np.load(path.join(hdf_dir, "cluster_membership.npy")).size==all_temperature_neurons.shape[0]):
        cluster_membership = np.load(path.join(hdf_dir, "cluster_membership.npy"))
    else:
        cluster_membership = utilities.greedy_corr_cluster(all_temperature_neurons, 25, np.sqrt(0.5))
        np.save(path.join(hdf_dir, "cluster_membership.npy"), cluster_membership)

    # filter clusters: Each cluster has to be present in at least 25 imaging planes
    prelim = np.unique(cluster_membership)
    prelim = prelim[prelim > -1]
    for p in prelim:
        planes = all_plane_ids[cluster_membership == p]
        if np.unique(planes).size < 25:
            cluster_membership[cluster_membership == p] = -1

    clus_nums = np.unique(cluster_membership)
    clus_nums = clus_nums[clus_nums > -1]

    temp_times = np.arange(temperatures.size) / 20

    plot_prob = np.hstack([50 / np.sum(cluster_membership == c) for c in cluster_membership])
    plot_prob[cluster_membership == -1] = 0
    plot_rand = np.random.rand(plot_prob.size)

    plot_act = all_temperature_neurons[plot_prob > plot_rand]
    sort_clusters = np.argsort(cluster_membership[plot_prob > plot_rand])

    time = np.arange(all_temperature_neurons.shape[1]) / 5

    df_plot_act = pd.DataFrame(plot_act, index=np.arange(plot_act.shape[0]), columns=time)

    df_sorted_act = pd.DataFrame(plot_act[sort_clusters], index=np.arange(plot_act.shape[0])[sort_clusters], columns=time)

    pl.figure()
    sns.heatmap(df_plot_act, vmax=5, rasterized=True, cmap='viridis', xticklabels=300)
    pl.xlabel("Time [s]")
    pl.ylabel("Temperature encoding neurons")

    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42

    fig = pl.figure()
    sns.heatmap(df_sorted_act, vmax=5, rasterized=True, cmap='viridis', xticklabels=300)
    pl.xlabel("Time [s]")
    pl.ylabel("Sorted neurons")
    fig.savefig(path.join(hdf_dir, "REVISION_3B_clustered_heatmap.pdf"), dpi=300)

    # Plot per-fish cluster counts
    cluster_abbrev_map = {0: "CA",
                           1: "H",
                           2: "HC",
                           3: "C",
                           5: "CC",
                           6: "HH",
                           7: "CH"}
    cluster_counts = {"Response type": [], "Count": []}
    hot_vs_cold_counts ={"Response type": [], "Count": []}
    for fid in range(np.max(all_fish_ids)+1):
        hc = 0
        cc = 0
        for cid in clus_nums:
            cluster_counts["Response type"].append(cluster_abbrev_map[cid])
            cluster_counts["Count"].append(np.sum(cluster_membership[all_fish_ids==fid] == cid))
            if cid in [0, 3, 5, 7]:
                cc += np.sum(cluster_membership[all_fish_ids==fid] == cid)
            else:
                hc += np.sum(cluster_membership[all_fish_ids==fid] == cid)
        hot_vs_cold_counts["Response type"].append("Cold")
        hot_vs_cold_counts["Count"].append(cc)
        hot_vs_cold_counts["Response type"].append("Hot")
        hot_vs_cold_counts["Count"].append(hc)

    fig = pl.figure()
    sns.boxplot(data=cluster_counts, x="Response type", y="Count", whis=np.inf)
    sns.despine()
    pl.yscale('log')
    fig.savefig(path.join(hdf_dir, "REVISION_3C_fish_cluster_membership_count.pdf"))

    hot_vs_cold_counts = pd.DataFrame(hot_vs_cold_counts)
    cold_counts = hot_vs_cold_counts['Count'][hot_vs_cold_counts["Response type"]=="Cold"]
    hot_counts = hot_vs_cold_counts['Count'][hot_vs_cold_counts["Response type"]=="Hot"]
    print(f"Average number of cold neurons: {np.mean(cold_counts)}")
    print(f"Average number of hot neurons: {np.mean(hot_counts)}")
    print(f"Ranksum p-value: {sts.ranksums(hot_counts, cold_counts)}")

    # Plot fraction of fish with given cluster
    cluster_fraction = {"Response type": [], "Fraction": []}
    fish_count = np.max(all_fish_ids)+1
    for cid in clus_nums:
        cluster_fraction["Response type"].append(cluster_abbrev_map[cid])
        cluster_fraction["Fraction"].append(np.unique(all_fish_ids[cluster_membership==cid]).size / fish_count)

    fig = pl.figure()
    sns.barplot(data=cluster_fraction, x="Response type", y="Fraction")
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_S3E_fish_cluster_fraction.pdf"))

    # plot stimulus characterization - delta-T [C/s] vs. T [delta-T at 20Hz hence mult by 20]
    fig = pl.figure()
    pl.scatter(temperatures[1:], np.diff(temperatures) * 20, s=1, alpha=0.3, rasterized=True)
    pl.xlabel("Temperature [C]")
    pl.ylabel("Temperature change [C/s]")
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_S3B_temperature_stim_characterization.pdf"), dpi=600)

    aligned_temps = temperatures[::4][:5399]
    aligned_dtemps = np.diff(temperatures[::4])[:5399]
    cluster_avgs = {}
    cluster_errs = {}
    for i, cn in enumerate(clus_nums):
        fig = pl.figure()
        cluster_data = gaussian_filter1d(all_temperature_spikes[cluster_membership == cn, :], 3, axis=1)
        bs_vars = utilities.bootstrap(cluster_data, 100, np.mean)
        m = np.mean(bs_vars, 0)
        e = np.std(bs_vars, axis=0)
        cluster_errs[cn] = e
        cluster_avgs[cn] = m
        pl.fill_between(time, m-e, m+e, alpha=0.3, color='C1')
        pl.plot(time, m, color='C1')
        pl.xlabel("Time [s]")
        pl.ylabel("Activity [AU]")
        ax_op = pl.twinx()
        ax_op.plot(temp_times, temperatures, 'k--')
        ax_op.set_ylabel("Temperature [C]")

    # Collect average cluster responses in an array that can be later used for fitting
    clusters_for_prediction = np.hstack([cluster_avgs[k][:, None] for k in cluster_avgs])

    # pairwise correlations of cluster averages
    df_cluster_corrs = pd.DataFrame(np.corrcoef(np.vstack([cluster_avgs[k] for k in cluster_avgs])),
                                    index=[
        "CA",
        "H",
        "HC",
        "C",
        "CC",
        "HH",
        "CH"
    ],
                                    columns=[
        "CA",
        "H",
        "HC",
        "C",
        "CC",
        "HH",
        "CH"
    ])

    fig = pl.figure()
    sns.heatmap(df_cluster_corrs, vmin=-1, vmax=1, center=0)
    fig.savefig(path.join(hdf_dir, "REVISION_S3D_Pairwise_Cluster_Correlations.pdf"))

    # distribution of pairwise activity correlations in unclustered group
    pw_unclust_corr = np.corrcoef(all_temperature_neurons[cluster_membership==-1])
    pw_uc_unique = pw_unclust_corr[np.triu_indices(pw_unclust_corr.shape[0], 1)].ravel()
    fig = pl.figure()
    max_count = pl.hist(pw_uc_unique, 250, density=True)[0].max()
    pl.plot([np.sqrt(0.5), np.sqrt(0.5)], [0, max_count], 'k--')
    pl.xlim(-1, 1)
    pl.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    pl.xlabel("Pairwise correlation")
    pl.ylabel("Density")
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_S3C_pairwise_act_corr_unclustered.pdf"))


    # combined activity plots
    fig = pl.figure()
    pl.fill_between(time, cluster_avgs[0] - cluster_errs[0], cluster_avgs[0] + cluster_errs[0], alpha=0.3, color='C1')
    pl.plot(time, cluster_avgs[0], color='C1', label="Cold adapting")
    pl.fill_between(time, cluster_avgs[3] - cluster_errs[3], cluster_avgs[3] + cluster_errs[3], alpha=0.3, color='C4')
    pl.plot(time, cluster_avgs[3], color='C4', label="Cold")
    pl.xlabel("Time [s]")
    pl.ylabel("Activity [AU]")
    pl.legend()
    ax_op = pl.twinx()
    ax_op.plot(temp_times, temperatures, 'k--')
    ax_op.set_ylabel("Temperature [C]")
    fig.savefig(path.join(hdf_dir, "REVISION_3D_cluster_response_CA_C.pdf"))

    fig = pl.figure()
    pl.fill_between(time, cluster_avgs[5] - cluster_errs[5], cluster_avgs[5] + cluster_errs[5], alpha=0.3, color='C5')
    pl.plot(time, cluster_avgs[5], color='C5', label="Cold and Cooling")
    pl.fill_between(time, cluster_avgs[6] - cluster_errs[6], cluster_avgs[6] + cluster_errs[6], alpha=0.3, color='C6')
    pl.plot(time, cluster_avgs[6], color='C6', label="Hot and Heating")
    pl.fill_between(time, cluster_avgs[7] - cluster_errs[7], cluster_avgs[7] + cluster_errs[7], alpha=0.3, color='C7')
    pl.plot(time, cluster_avgs[7], color='C7', label="Cold and Heating")
    pl.xlabel("Time [s]")
    pl.ylabel("Activity [AU]")
    pl.legend()
    ax_op = pl.twinx()
    ax_op.plot(temp_times, temperatures, 'k--')
    ax_op.set_ylabel("Temperature [C]")
    fig.savefig(path.join(hdf_dir, "REVISION_3E_cluster_response_CC_HH_CH.pdf"))

    fig = pl.figure()
    pl.fill_between(time, cluster_avgs[1] - cluster_errs[1], cluster_avgs[1] + cluster_errs[1], alpha=0.3, color='C2')
    pl.plot(time, cluster_avgs[1], color='C2', label="Hot")
    pl.fill_between(time, cluster_avgs[2] - cluster_errs[2], cluster_avgs[2] + cluster_errs[2], alpha=0.3, color='C3')
    pl.plot(time, cluster_avgs[2], color='C3', label="Hot and Cooling")
    pl.xlabel("Time [s]")
    pl.ylabel("Activity [AU]")
    pl.legend()
    ax_op = pl.twinx()
    ax_op.plot(temp_times, temperatures, 'k--')
    ax_op.set_ylabel("Temperature [C]")
    fig.savefig(path.join(hdf_dir, "REVISION_3F_cluster_response_H_HC.pdf"))

    temp_in = np.interp(time, temp_times, temperatures)
    temp_in = utilities.safe_standardize(temp_in)

    responses = np.r_[
        [utilities.safe_standardize(np.mean(gaussian_filter1d(all_temperature_spikes[cluster_membership == cn, :], 3, axis=1), 0)) for cn in
         clus_nums]]
    init_in = np.random.randn(1, 10, 1).astype(np.float32)
    if not path.exists(path.join(hdf_dir, "cluster_models.hdf5")):
        with h5py.File(path.join(hdf_dir, "cluster_models.hdf5"), 'w') as dfile:
            miner = mine.Mine(2/3, 10, np.sqrt(0.5), True, False,
                              5, 25)
            miner.n_epochs = 1000
            miner.model_weight_store = dfile
            mdata = miner.analyze_data([temp_in], responses)
            cluster_test_corrs = mdata.correlations_test
    else:
        # models have been fit, load from file
        cluster_test_corrs = np.zeros(clus_nums.size)
        with h5py.File(path.join(hdf_dir, "cluster_models.hdf5"), 'r') as dfile:
            for i, cn in enumerate(clus_nums):
                w_grp = dfile[f"cell_{i}_weights"]
                weights = utilities.modelweights_from_hdf5(w_grp)
                m = model.get_standard_model(10)
                m(init_in)
                m.set_weights(weights)
                # calculate test correlations
                test_start = int(temp_in.size*2.0/3.0)
                prediction = utilities.simulate_response(m, temp_in[test_start:][:, None])
                cluster_test_corrs[i] = np.corrcoef(prediction, responses[i][test_start+9:])[0,1]

    # plot test correlations by cluster
    fig = pl.figure()
    pl.plot([
        "CA",
        "H",
        "HC",
        "C",
        "CC",
        "HH",
        "CH"
    ], cluster_test_corrs, 'o')
    pl.ylabel("MINE test correlation")
    pl.ylim(0, 1)
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_S4B_Cluster_MINE_test_corrs.pdf"))

    fish_files = os.listdir("./../behavioral_fever/KAB_Gradient/")
    fish_files = [f for f in fish_files if ".pkl" in f and "fish" in f]
    bout_files = [f[:f.find("fish")]+"bout.pkl" for f in fish_files]

    mu = np.mean(temperatures[::4][:all_temperature_neurons.shape[1]])
    std = np.std(temperatures[::4][:all_temperature_neurons.shape[1]])

    all_states = []  # for each bout the state
    all_bout_act = []  # for each bout the activity of all neurons at the start

    rel_behav_path = "./../behavioral_fever/KAB_Gradient/"

    all_fish_data = []
    all_exp_neuron_responses = []

    for j, (ff, bf) in enumerate(zip(fish_files, bout_files)):

        print(f"Processing fish {j+1} out of {len(fish_files)} fish")

        bout_act_save_name = path.splitext(bf)[0] + "_activity.npy"
        full_act_save_name = path.splitext(bf)[0] + "_fullact.npy"

        if path.exists(path.join(rel_behav_path, bout_act_save_name)):
            # if prediction already exists, load it instead of recreating it
            exp_neuron_responses = np.load(path.join(rel_behav_path, full_act_save_name)).T  # this is transposed on saving!!
            all_exp_neuron_responses.append(exp_neuron_responses.T)
            bout_data = pd.read_pickle(path.join(rel_behav_path, bf))
            starts = np.array(bout_data["Start"]).astype(int)
            states = np.array(bout_data["State"])
            this_bout_act = np.zeros((states.size, len(exp_neuron_responses) * 3))
            avg_bout_act = np.zeros((states.size, len(exp_neuron_responses)))
            for i, nr in enumerate(exp_neuron_responses):
                activity_history = np.hstack([nr[starts][:, None], nr[starts - 50][:, None], nr[starts - 100][:, None]])
                avg_bout_act[:, i] = np.mean(activity_history, axis=1)
            all_bout_act.append(avg_bout_act)
            all_states.append(states)
            fish_data = pd.read_pickle(path.join(rel_behav_path, ff))
            all_fish_data.append(fish_data)
            continue

        fish_data = pd.read_pickle(path.join(rel_behav_path, ff))
        all_fish_data.append(fish_data)
        bout_data = pd.read_pickle(path.join(rel_behav_path, bf))

        exp_temps = np.array(fish_data["Temperature"])  # at 100 Hz
        exp_time = np.arange(exp_temps.size) / 100
        ds_exp_temps = exp_temps[::20]  # downsampled to 5 Hz, suitable as MINE input
        ds_exp_time = np.arange(ds_exp_temps.size) / 5
        ds_exp_time = ds_exp_time[9:]

        ds_exp_neuron_responses = []
        with h5py.File(path.join(hdf_dir, "cluster_models.hdf5"), "r") as dfile:
            for i, cn in enumerate(clus_nums):
                if cluster_test_corrs[i] < np.sqrt(0.5):
                    # Do not fit neurons for models that poorly generalize
                    continue
                w_grp = dfile[f"cell_{i}_weights"]
                weights = utilities.modelweights_from_hdf5(w_grp)
                m = model.get_standard_model(10)
                m(init_in)
                m.set_weights(weights)
                ds_exp_neuron_responses.append(utilities.simulate_response(m, (ds_exp_temps[:, None]-mu)/std))

        exp_neuron_responses = [np.interp(exp_time, ds_exp_time, denr) for denr in ds_exp_neuron_responses]
        all_exp_neuron_responses.append(np.vstack(exp_neuron_responses).T)

        starts = np.array(bout_data["Start"]).astype(int)
        states = np.array(bout_data["State"])

        all_states.append(states)
        # for each swim bout we store the activity of each cluster at the start, 500 ms before the start and 1 s before
        this_bout_act = np.zeros((states.size, len(exp_neuron_responses)*3))
        avg_bout_act = np.zeros((states.size, len(exp_neuron_responses)))
        for i, nr in enumerate(exp_neuron_responses):
            activity_history = np.hstack([nr[starts][:, None], nr[starts-50][:, None], nr[starts-100][:, None]])
            this_bout_act[:, i*3:i*3+3] = activity_history
            avg_bout_act[:, i] = np.mean(activity_history, axis=1)
        # save bout activity for this experiment

        np.save(path.join(rel_behav_path, bout_act_save_name), this_bout_act)
        np.save(path.join(rel_behav_path, full_act_save_name), np.vstack(exp_neuron_responses).T)
        all_bout_act.append(avg_bout_act)

    all_states = np.hstack(all_states)
    all_bout_act = np.vstack(all_bout_act)

    valid = np.sum(np.isnan(all_bout_act), 1) == 0
    all_states = all_states[valid]
    all_bout_act = all_bout_act[valid]

    def plot_activation(activity_data: np.ndarray, fish_dframe: pd.DataFrame, start_frame: int, end_frame: int, cluster_index:int, plotax, x_offset=0.0):
        """
        Plots a sub-trajectory such that points of maximal activity for different types are marked on a background
        swimming trajectory.
        :param activity_data: For each type the activity across the entire exeriment (n_frames x n_clusters)
        :param fish_dframe: The fish dataframe
        :param start_frame: The start of the sub-trajectory
        :param end_frame: The end of the sub-trajectory
        :param cluster_index: The cluster index for which to plot activity
        :param plotax: The figure axes
        :param x_offset: Can be used to shift the trajectory around on the x-axis since this position is arbitrary
        :return: The figure object
        """
        global all_bout_act
        if plotax is None:
            plotax = pl.subplots()[1]
        thresholds = np.nanmean(all_bout_act[:, cluster_index]) + 0.5*np.nanstd(all_bout_act[:, cluster_index])
        max_level = thresholds + 3*np.nanstd(all_bout_act[:, cluster_index])
        plotmap = pl.colormaps["plasma"]
        plotax.plot(fish_dframe["X Position"][start_frame:end_frame]*0.04 + x_offset,
                fish_dframe["Temperature"][start_frame:end_frame], color=[0.7, 0.7, 0.7])
        abt = (activity_data[start_frame:end_frame, cluster_index] > thresholds)
        x = fish_dframe["X Position"][start_frame:end_frame][abt]*0.04 + x_offset
        y = fish_dframe["Temperature"][start_frame:end_frame][abt]
        level = (activity_data[start_frame:end_frame, cluster_index][abt] - thresholds) / (max_level - thresholds)
        level[level > 1] = 1
        colors = plotmap(level)
        colors[:, -1] = 1  # alpha
        if x.size > 0:
            plotax.scatter(x, y, color=colors, s=10)
        # mark start of trajectory
        plotax.scatter(fish_dframe["X Position"][start_frame]*0.04+x_offset, fish_dframe["Temperature"][start_frame], color='k', s=40)

    for i in range(all_exp_neuron_responses[0].shape[1]):
        fig, ax = pl.subplots()
        plot_activation(all_exp_neuron_responses[12], all_fish_data[12], 114155, 115155, i, ax)
        plot_activation(all_exp_neuron_responses[29], all_fish_data[29], 66676, 67676, i, ax)
        plot_activation(all_exp_neuron_responses[13], all_fish_data[13], 78282, 79282, i, ax)
        plot_activation(all_exp_neuron_responses[29], all_fish_data[29], 144564, 145564, i, ax, -0.3)
        plot_activation(all_exp_neuron_responses[-1], all_fish_data[-1], 80000, 81000, i, ax, 1)
        plot_activation(all_exp_neuron_responses[5], all_fish_data[5], 131000, 135500, i, ax)
        plot_activation(all_exp_neuron_responses[-5], all_fish_data[-5], 133000, 135500, i, ax)
        plot_activation(all_exp_neuron_responses[-7], all_fish_data[-7], 149000, 150000, i, ax)
        pl.axis('scaled')
        pl.xlim(0, 2.1)
        pl.ylim(18, 32)
        pl.ylabel("Temperature [C]")
        pl.xlabel("Position [AU]")
        sns.despine(fig, ax)
        fig.savefig(path.join(hdf_dir, f"REVISION_4D_trajectory_activity_plot_cluster{clus_nums[i]}.pdf"))

    # Plot example activity traces for each type with same color map as traces above
    clus_names = ["Cold adapting", "Hot", "Hot and Cooling", "Cold", "Cold and Cooling", "Hot and Heating",
                  "Cold and Heating"]

    findex = 12
    fstart = 65000
    fend = 65000 + 40*100
    time = np.arange(fend - fstart) / 100
    fig = pl.figure()
    pl.plot(time, all_fish_data[findex]["Temperature"][fstart:fend])
    pl.ylabel("Gradient temperature [C]")
    pl.xlabel("Time")
    sns.despine()
    fig.savefig(path.join(hdf_dir, f"REVISION_S4C_example_behavior_temperature.pdf"))

    for cluster_index in [2, 6]:
        threshold = np.nanmean(all_bout_act[:, cluster_index]) + 0.5 * np.nanstd(all_bout_act[:, cluster_index])
        max_act = threshold + 3 * np.nanstd(all_bout_act[:, cluster_index])
        plot_act = all_exp_neuron_responses[findex][fstart:fend, cluster_index].copy()
        plot_act -= np.nanmean(all_bout_act[:, cluster_index])
        plot_act /= np.nanstd(all_bout_act[:, cluster_index])
        plotmap = pl.colormaps["plasma"]
        level = (all_exp_neuron_responses[findex][fstart:fend, cluster_index] - threshold) / (max_act - threshold)
        level[level > 1] = 1
        colors = plotmap(level)
        colors[:, -1] = 1
        fig = pl.figure()
        pl.plot(time, plot_act, color=[0.7, 0.7, 0.7], label=clus_names[cluster_index])
        pl.plot([0, time.max()], [0.5, 0.5], 'k--')
        pl.plot([0, time.max()], [3, 3], 'k--')
        if np.sum(level > 0) > 0:
            pl.scatter(time[level>0], plot_act[level>0], color=colors[level>0], s=1)
        pl.xlabel("Time [s]")
        pl.ylabel("Activity [zscore]")
        pl.legend()
        sns.despine()
        fig.savefig(path.join(hdf_dir, f"REVISION_S4D-E_example_behavior_time_cluster_act{clus_nums[cluster_index]}.pdf"))

    # Test how well neural activity can predict temperature as well as temperature-change - NOTE: In the following
    # the train and test sets are split at random, by sample, since there is no guarantee that fish will explore the
    # entire chamber during a "training part" of the experiment which would trivially limit generalization.
    train_acts = []
    train_temps = []
    train_dtemps = []
    test_acts = []
    test_temps = []
    test_dtemps = []
    p_train = 0.8

    per_exp_test_results = {"Input": [], "R2": []}

    for i in range(len(all_exp_neuron_responses)):
        these_responses = all_exp_neuron_responses[i].copy()
        these_temps = np.array(all_fish_data[i]["Temperature"])
        # compute delta-temperatures as weighted ~2s average in C/s
        these_dtemps = np.r_[np.nan, gaussian_filter1d(np.diff(these_temps), 60) * 100]
        valid_responses = np.sum(np.isnan(these_responses), 1) == 0
        valid_responses = np.logical_and(valid_responses, np.isfinite(these_temps))
        valid_responses = np.logical_and(valid_responses, np.isfinite(these_dtemps))
        these_responses = these_responses[valid_responses]
        these_responses -= np.mean(these_responses, 0, keepdims=True)
        these_responses /= np.std(these_responses, 0, keepdims=True)
        these_temps = these_temps[valid_responses]
        these_dtemps = these_dtemps[valid_responses]
        rand = np.random.rand(these_responses.shape[0])

        these_train_acts = these_responses[rand < p_train]
        these_test_acts = these_responses[rand >= p_train]
        these_train_temps = these_temps[rand < p_train]
        these_test_temps = these_temps[rand >= p_train]
        these_train_dtemps = these_dtemps[rand < p_train]
        these_test_dtemps = these_dtemps[rand >= p_train]
        # compute context: +/-1 C around 25 and +/- 0.01 C/s around 0 are no context
        these_context = np.zeros_like(these_temps)
        cold = these_temps < 24
        cooling = these_dtemps < -0.0052
        hot = these_temps > 26
        heating = these_dtemps > 0.0052
        these_context[np.logical_and(cold, cooling)] = -1
        these_context[np.logical_and(hot, heating)] = -1
        these_context[np.logical_and(hot, cooling)] = 1
        these_context[np.logical_and(cold, heating)] = 1
        these_train_context = these_context[rand < p_train]
        these_test_context = these_context[rand >= p_train]

        # calculate per-fish predictions
        # Temperature
        per_exp_test_results["Input"].append("Temperature")
        exp_pred_model = RidgeCV()
        exp_pred_model.fit(these_train_acts, these_train_temps)
        per_exp_test_results["R2"].append(exp_pred_model.score(these_test_acts, these_test_temps))

        # Temperature change
        per_exp_test_results["Input"].append("Temperature change")
        exp_pred_model = RidgeCV()
        exp_pred_model.fit(these_train_acts, these_train_dtemps)
        per_exp_test_results["R2"].append(exp_pred_model.score(these_test_acts, these_test_dtemps))

        # Context
        per_exp_test_results["Input"].append("Context")
        log_reg = LogisticRegressionCV()
        log_reg.fit(these_train_acts, these_train_context)
        per_exp_test_results["R2"].append(log_reg.score(these_test_acts, these_test_context))

        # store all for later use in "overall models"
        train_acts.append(these_train_acts)
        test_acts.append(these_test_acts)
        train_temps.append(these_train_temps)
        test_temps.append(these_test_temps)
        train_dtemps.append(these_train_dtemps)
        test_dtemps.append(these_test_dtemps)
    train_temps = np.hstack(train_temps)
    test_temps = np.hstack(test_temps)
    train_dtemps = np.hstack(train_dtemps)
    test_dtemps = np.hstack(test_dtemps)
    train_acts = np.vstack(train_acts)
    test_acts = np.vstack(test_acts)

    fig = pl.figure()
    sns.boxplot(data=per_exp_test_results, x="Input", y="R2", whis=np.inf)
    sns.despine()
    pl.ylim(0, 1)
    fig.savefig(path.join(hdf_dir, f"REVISION_4C_Activity_Prediction_Summary_Stats.pdf"))

    # The following shows the prediction confidence in the following way:
    # Test temperature changes within a bin come from different times across different experiments. Each of these timepoints
    # therefore belongs to a different set of activities across the seven response types. These are then used to predict
    # the actual temperature change at each timepoint. The average line in this plot therefore reflects the overall goodnes of
    # fit based on the distance to the identity, while the error is dominated by uncertainties in the neural response
    # prediction which generally depends on stimulus history rather than instantaneous values

    temp_ranges = {
        "18C - 22C": [18, 22],
        "22C - 28C": [22, 28],
        "28C - 32C": [28, 32]
    }

    delta_predictions = {}

    for k in temp_ranges.keys():
        train_sel = np.logical_and(train_temps >= temp_ranges[k][0], train_temps <= temp_ranges[k][1])
        test_sel = np.logical_and(test_temps >= temp_ranges[k][0], test_temps <= temp_ranges[k][1])
        dtemp_pred_model = RidgeCV()
        dtemp_pred_model.fit(train_acts[train_sel], train_dtemps[train_sel])
        dtemp_test_pred = dtemp_pred_model.predict(test_acts[test_sel])
        delta_predictions[k] = (test_dtemps[test_sel], dtemp_test_pred)

    dtemp_bins = np.linspace(-0.5, 0.5, 20)
    dtemp_bc = dtemp_bins[:-1] + np.diff(dtemp_bins)/2

    fig, axes = pl.subplots(nrows=3, sharex=True, sharey=True)
    for ix, k in enumerate(delta_predictions.keys()):
        lower = np.zeros(dtemp_bc.size)
        upper = lower.copy()
        avg = lower.copy()
        real, test = delta_predictions[k]
        for i in range(dtemp_bc.size):
            pred_in_bin = test[np.logical_and(real >= dtemp_bins[i], real <= dtemp_bins[i + 1])]
            lower[i] = np.percentile(pred_in_bin, 2.5)
            upper[i] = np.percentile(pred_in_bin, 97.5)
            avg[i] = np.mean(pred_in_bin)
        axes[ix].fill_between(dtemp_bc, lower, upper, alpha=0.2, color=f"C{ix}")
        axes[ix].plot(dtemp_bc, avg, color=f"C{ix}", label=k)
        axes[ix].plot([-0.5, 0.5], [0, 0], 'k--')
        axes[ix].plot([0, 0], [-0.5, 0.5], 'k--')
        axes[ix].set_ylabel("Prediction [C/s]")
        axes[ix].legend()
    axes[-1].set_xlabel("True temperature change [C/s]")
    sns.despine()
    fig.savefig(path.join(hdf_dir, f"REVISION_S4F_Activity_TempChange_Prediction.pdf"))

    aligned_1s_delta = compute_mm_delta_temps(aligned_temps, 5)

    # z-score clusters_for_prediction
    clusters_for_prediction -= np.mean(clusters_for_prediction, axis=0, keepdims=True)
    clusters_for_prediction /= np.std(clusters_for_prediction, axis=0, keepdims=True)

    # Check predictability of mode transition model inputs based on imaging data
    # Temperature - Distance to preference, i.e. abs(temperature-25) - Temperature change - abs(Temperature change)
    input_pred_scores = {"Variable": [], "R2": []}
    rand = np.random.rand(aligned_1s_delta.shape[0])
    pred_model = RidgeCV()
    input_pred_scores["Variable"].append("Temperature")
    pred_model.fit(clusters_for_prediction[rand<0.8], aligned_temps[rand<0.8, None])
    input_pred_scores["R2"].append(pred_model.score(clusters_for_prediction[rand>=0.8], aligned_temps[rand>=0.8, None]))
    pred_model = RidgeCV()
    input_pred_scores["Variable"].append("Preference distance")
    pred_model.fit(clusters_for_prediction[rand<0.8], np.abs(aligned_temps[rand<0.8, None]-25))
    input_pred_scores["R2"].append(pred_model.score(clusters_for_prediction[rand>=0.8], np.abs(aligned_temps[rand>=0.8, None]-25)))
    pred_model = RidgeCV()
    input_pred_scores["Variable"].append("Temp. change")
    pred_model.fit(clusters_for_prediction[rand<0.8], aligned_1s_delta[rand<0.8, None])
    input_pred_scores["R2"].append(pred_model.score(clusters_for_prediction[rand>=0.8], aligned_1s_delta[rand>=0.8, None]))
    pred_model = RidgeCV()
    input_pred_scores["Variable"].append("Abs. temp. change")
    pred_model.fit(clusters_for_prediction[rand<0.8], np.abs(aligned_1s_delta[rand<0.8, None]))
    input_pred_scores["R2"].append(pred_model.score(clusters_for_prediction[rand>=0.8], np.abs(aligned_1s_delta[rand>=0.8, None])))

    fig = pl.figure()
    sns.stripplot(data=input_pred_scores, x="Variable", y="R2", color='k')
    sns.despine()
    pl.ylim(0, 1)
    fig.savefig(path.join(hdf_dir, f"REVISION_S5A_ModelInput_Prediction_Scores.pdf"))

    # Predict transition probabilities, computed on the imaging traces, from the cluster activity as well
    # as the temperature stimulus itself as a control - since that stimulus "created" the activity of the response types
    # load stimulus driven model
    with h5py.File(".\\..\\stan_state_space\\sim_model_store.hdf5", 'r') as dfile:
        type_group = "Swim_mode_transit_predictable_higher_order"
        grp = dfile[type_group]
        mm_icept_stim = grp["mm_icept"][()]
        mm_orig_stim = grp["mm_orig"][()]

    n_samples = 100
    trans_probs = np.zeros((aligned_temps.size, 3, 3))
    n_draws = mm_icept_stim.shape[0]

    for i, (temp, dtemp) in enumerate(zip(aligned_temps, aligned_1s_delta)):
        for s in range(n_samples):
            draw = np.random.randint(n_draws)
            trans_mat = tm_from_glm(temp-25, dtemp, mm_orig_stim[draw], mm_icept_stim[draw])
            tm = np.array(trans_mat)
            for k in range(3):
                for kk in range(3):
                    trans_probs[i, k, kk] += tm[k, kk]
    trans_probs /= n_samples

    trans_probs = gaussian_filter1d(trans_probs, 5, axis=0)

    trans_prob_predictions = np.zeros_like(trans_probs)

    def logit(x):
        return np.log(x / (1 - x))

    def sigm(x):
        return 1 / (1 + np.exp(-x))

    # Fit the following quantities as these are also what we display on behavior:
    # 1. Probability to initiate reversal
    # 2. Probability of continuing persistent swims
    # 3. Probability of being in the general state
    p_rev_enter = np.zeros(trans_probs.shape[0])
    p_general = np.zeros_like(p_rev_enter)
    p_stay_persistent = np.zeros_like(p_rev_enter)
    for i in range(p_rev_enter.size):
        p_stay_persistent[i] = trans_probs[i, 2, 2]
        ss_probs = np.linalg.matrix_power(trans_probs[i], 100)
        p_general[i] = ss_probs[0, 1]
        p_persist = ss_probs[0, 2]
        p_rev_enter[i] = p_general[i] * trans_probs[i, 1, 0] + p_persist * trans_probs[i, 2, 0]

    prob_to_predict = [p_rev_enter, p_stay_persistent, p_general]
    pred_names = ["Reversal entry", "Maintain persist", "General mode"]
    cluster_scores = []
    cluster_predictions = []
    temperature_scores = []

    # Compute test score on all three models when using activity vs. when using temperature as inputs
    for i, p in enumerate(prob_to_predict):
        rand = np.random.rand(trans_probs.shape[0])
        train = rand < 0.8
        test = np.logical_not(train)
        lr = LinearRegression()
        lr.fit(clusters_for_prediction[train], logit(p[train]))
        cluster_scores.append(lr.score(clusters_for_prediction[test], logit(p[test])))
        cluster_predictions.append(lr.predict(clusters_for_prediction))
        lr = LinearRegression()
        lr.fit(aligned_temps[train, None], logit(p[train]))
        temperature_scores.append(lr.score(aligned_temps[test, None], logit(p[test])))

    time = np.arange(aligned_dtemps.size)/5
    fig, axes = pl.subplots(nrows=3, sharex=True)
    for i in range(3):
        axes[i].plot(time, prob_to_predict[i], 'k', label=pred_names[i])
        axes[i].plot(time, sigm(cluster_predictions[i]), f'C{i}', label=f"R2 = {np.round(cluster_scores[i], 2)}")
        axes[i].set_ylabel("Probability")
        axes[i].legend()
    axes[2].set_xticks([0, 200, 400, 600, 800, 1000])
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(hdf_dir, f"REVISION_7A_Activity_Prob_Predictions.pdf"))

    fig = pl.figure()
    pl.scatter(temperature_scores[0], cluster_scores[0], marker='o', label=pred_names[0])
    pl.scatter(temperature_scores[1], cluster_scores[1], marker='P', label=pred_names[1])
    pl.scatter(temperature_scores[2], cluster_scores[2], marker='s', label=pred_names[2])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.legend(loc='lower right')
    pl.xlabel("Stimulus prediction R2")
    pl.ylabel("Response type prediction R2")
    sns.despine()
    fig.savefig(path.join(hdf_dir, f"REVISION_7B_Activity_vs_Stim_Predictions.pdf"))

    # For each of the three models use bootstrapping to plot model parameters and confidence for influence of each response type
    def bootstrap_regression_coefficients(X: np.ndarray, y: np.ndarray, nboot=10_000) -> np.ndarray:
        """
        Uses bootstrapping to obtain confidence intervals on regression coefficients
        :param X: The independent variable
        :param y: The dependent variable
        :param nboot: The number of bootstrap samples
        :return: nboot x X.shape[1] matrix of coefficients
        """
        coef_boot = np.zeros((nboot, X.shape[1]))
        lr_boot = LinearRegression()
        data_ix = np.arange(X.shape[0])[::50]  # we subsample to remove the influence of auto-correlations which reduce the reported error
        for b in range(nboot):
            boot_ix = np.random.choice(data_ix, size=data_ix.size, replace=True)
            lr_boot.fit(X[boot_ix], y[boot_ix])
            coef_boot[b] = lr_boot.coef_.copy()
        return coef_boot

    boot_coef = []
    for i, p in enumerate(prob_to_predict):
        boot_coef.append(bootstrap_regression_coefficients(clusters_for_prediction, p))

    fig, axes = pl.subplots(nrows=3, sharex=True)
    for i in range(3):
        axes[i].bar(np.arange(7), np.mean(boot_coef[i], axis=0), color=f"C{i}")
        axes[i].vlines(np.arange(7), np.percentile(boot_coef[i], 2.5, axis=0), np.percentile(boot_coef[i], 97.5, axis=0), colors='k')
        axes[i].set_ylabel("Value")
    axes[2].set_xticks(np.arange(7), labels=["CA", "H", "HC", "C", "CC", "HH", "CH"])
    fig.tight_layout()
    sns.despine(fig)
    fig.savefig(path.join(hdf_dir, f"REVISION_7C_Activity_Prob_Prediction_Coefs.pdf"))
