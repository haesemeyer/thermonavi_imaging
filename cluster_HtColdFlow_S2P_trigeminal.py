"""
Script to cluster hot-cold flow experiment temperature responses  within the trigeminal ganglion, where expected
numbers of neurons are low
"""

import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib as mpl
import argparse
import os
from os import path
import seaborn as sns
import utilities
import pandas as pd
from cluster_HtColdFlow_S2P_hindbrain import set_journal_style

if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="cluster_HtColdFlow_S2P_trigeminal",
                                       description="Clusters flow setup temperature responses after Suite2P and MINE for regions with few neurons")
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
                # filter by test correlation - rather loosely, let clustering figure out the rest
                responses = responses[test_corrs > 0.5]
                spks = spks[test_corrs > 0.5]
                indices = indices[test_corrs > 0.5]
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
    all_fish_ids = np.hstack(all_fish_ids)

    if (path.exists(path.join(hdf_dir, "cluster_membership.npy")) and
            np.load(path.join(hdf_dir, "cluster_membership.npy")).size==all_temperature_neurons.shape[0]):
        cluster_membership = np.load(path.join(hdf_dir, "cluster_membership.npy"))
    else:
        cluster_membership = utilities.greedy_corr_cluster(all_temperature_neurons, 5, np.sqrt(0.5))
        np.save(path.join(hdf_dir, "cluster_membership.npy"), cluster_membership)

    # filter clusters: Each cluster has to be present in at least two imaging planes
    prelim = np.unique(cluster_membership)
    prelim = prelim[prelim > -1]
    for p in prelim:
        planes = all_plane_ids[cluster_membership == p]
        if np.unique(planes).size < 3:
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

    set_journal_style(23, 23)

    fig = pl.figure()
    sns.heatmap(df_sorted_act, vmax=5, rasterized=True, cmap='viridis', xticklabels=300)
    pl.xlabel("Time [s]")
    pl.ylabel("Sorted neurons")
    fig.savefig(path.join(hdf_dir, "REVISION_S3H_clustered_heatmap.pdf"), dpi=300)

    # plot per-fish cluster counts
    cluster_abbrev_map = {
        0: "CA",
        1: "C",
        2: "H"
    }

    cluster_counts = {"Response type": [], "Count": []}
    for fid in range(np.max(all_fish_ids)+1):
        for cid in range(3):
            cluster_counts["Response type"].append(cluster_abbrev_map[cid])
            cluster_counts["Count"].append(np.sum(cluster_membership[all_fish_ids==fid] == cid))

    fig = pl.figure()
    sns.swarmplot(data=cluster_counts, x="Response type", y="Count")
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_3H_fish_cluster_membership_count.pdf"))

    cluster_avgs = {}
    cluster_errs = {}

    for i, cn in enumerate(clus_nums):
        fig = pl.figure()
        cluster_data = all_temperature_neurons[cluster_membership == cn, :]
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

    # combined activity plots
    fig = pl.figure()
    pl.fill_between(time, cluster_avgs[0] - cluster_errs[0], cluster_avgs[0] + cluster_errs[0], alpha=0.3, color='C0')
    pl.plot(time, cluster_avgs[0], color='C0', label="Cold adapting")
    pl.fill_between(time, cluster_avgs[2] - cluster_errs[2], cluster_avgs[2] + cluster_errs[2], alpha=0.3, color='C1')
    pl.plot(time, cluster_avgs[2], color='C1', label="Hot")
    pl.fill_between(time, cluster_avgs[1] - cluster_errs[1], cluster_avgs[1] + cluster_errs[1], alpha=0.3, color='C2')
    pl.plot(time, cluster_avgs[1], color='C2', label="Cold", linestyle='dashed')
    pl.xlabel("Time [s]")
    pl.ylabel("Activity [AU]")
    pl.legend()
    ax_op = pl.twinx()
    ax_op.plot(temp_times, temperatures, 'k-.')
    ax_op.set_ylabel("Temperature [C]")
    fig.savefig(path.join(hdf_dir, "REVISION_3I_cluster_response_0_2_1.pdf"))

    # pairwise correlations of cluster averages
    df_cluster_corrs = pd.DataFrame(np.corrcoef(np.vstack([cluster_avgs[k] for k in cluster_avgs])),
                                    index=[
        "CA",
        "C",
        "H"
    ],
                                    columns=[
        "CA",
        "C",
        "H"
    ])

    fig = pl.figure()
    sns.heatmap(df_cluster_corrs, vmin=-1, vmax=1, center=0)
    fig.savefig(path.join(hdf_dir, "REVISION_S3J_TG_Pairwise_Cluster_Correlations.pdf"))

    # distribution of pairwise activity correlations in unclustered group
    pw_unclust_corr = np.corrcoef(all_temperature_neurons[cluster_membership==-1])
    pw_uc_unique = pw_unclust_corr[np.triu_indices(pw_unclust_corr.shape[0], 1)].ravel()
    fig = pl.figure()
    max_count = pl.hist(pw_uc_unique, 50, density=True)[0].max()
    pl.plot([np.sqrt(0.5), np.sqrt(0.5)], [0, max_count], 'k--')
    pl.xlim(-1, 1)
    pl.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    pl.xlabel("Pairwise correlation")
    pl.ylabel("Density")
    sns.despine()
    fig.savefig(path.join(hdf_dir, "REVISION_S3I_TG_pairwise_act_corr_unclustered.pdf"))
