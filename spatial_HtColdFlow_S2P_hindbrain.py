import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib as mpl
import argparse
import os
from os import path
import seaborn as sns
import re
import utilities
from sklearn.neighbors import NearestNeighbors
from cluster_HtColdFlow_S2P_hindbrain import set_journal_style


def load_anatomy_csv(file_path: str) -> np.ndarray:
    if not "_transformed.csv" in file_path:
        raise ValueError("Wrong file path")
    coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            if line == "":
                break
            content = str.strip(line[:line.find("\n")])
            location = content.split(' ')
            if len(location) == 4:
                coordinates.append(np.full(3, np.nan))
            elif len(location) == 3:
                coordinates.append(np.array([float(c) for c in location]))
            else:
                raise ValueError("Wrong format")
    return np.vstack(coordinates)


if __name__ == '__main__':

    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="spatial_HtColdFlow_S2P_hindbrain",
                                       description="Runs spatial analysis of clustered neurons in the medulla")
    a_parser.add_argument("-d", "--directory", help="Path to folder with response Hdf5 file", type=str, required=True)

    args = a_parser.parse_args()

    hdf_dir = args.directory

    plot_dir = "REVISION_spatial_hindbrain"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    hdf_files = [path.join(hdf_dir, f) for f in os.listdir(hdf_dir) if ".hdf5" in f and "cluster" not in f]
    exp_folders = [f[:-5]+"/" for f in hdf_files]
    temp_files = [path.join(ef, f) for ef in exp_folders for f in os.listdir(ef) if ".temp" in f and "cluster" not in f]

    all_temperatures = [np.genfromtxt(tf) for tf in temp_files]
    all_temperatures = [at for at in all_temperatures if at.size > 20000]
    min_len = min([len(at) for at in all_temperatures])
    temperatures = np.mean([at[:min_len] for at in all_temperatures], axis=0)

    # collect and concatenate all responses
    all_temperature_neurons = []
    all_ids = []
    all_test_correlations = []
    all_plane_ids = []  # simple set of ids where all neurons from one plane receive the same consecutive number
    plane_counter = 0
    neuron_origin = {}
    for f in hdf_files:
        first, last = path.split(f)
        raw_folder = path.join(first, path.splitext(last)[0])
        neuron_origin[raw_folder] = {}
        with h5py.File(f, "r") as dfile:
            for k in dfile:
                if "weights" in k:
                    continue
                plane_s2p_folder = path.join(path.join(path.join(raw_folder, k[:-5]), "suite2p"), "plane0")
                assert path.exists(plane_s2p_folder)
                grp = dfile[k]
                responses = grp["heat_res_neurons"][()]
                indices = grp["heat_res_indices"][()]
                test_corrs = grp["heat_res_testcorr"][()]
                all_test_correlations.append(test_corrs)
                # filter by test correlation - rather loosely, let clustering figure out the rest
                responses = responses[test_corrs > 0.6]
                indices = indices[test_corrs > 0.6]
                neuron_origin[raw_folder][plane_s2p_folder] = indices
                all_temperature_neurons.append(responses)
                all_ids += [(f, k, ix) for ix in indices]
                all_plane_ids.append(np.full(responses.shape[0], plane_counter))
                plane_counter += 1
    all_temperature_neurons = np.vstack(all_temperature_neurons)[:, :-1]
    all_temperature_neurons = utilities.safe_standardize(all_temperature_neurons, 1)
    all_plane_ids = np.hstack(all_plane_ids)
    all_test_correlations = np.hstack(all_test_correlations)

    if (path.exists(path.join(hdf_dir, "cluster_membership.npy")) and
            np.load(path.join(hdf_dir, "cluster_membership.npy")).size==all_temperature_neurons.shape[0]):
        cluster_membership = np.load(path.join(hdf_dir, "cluster_membership.npy"))
    else:
        raise NotImplementedError()  # clustering is not performed by here but in cluster_HtColdFlow_S2P_hindbrain.py

    # filter clusters: Each cluster has to be present in at least 25 imaging planes
    # NOTE: This is copy-pasted code from cluster_HtColdFlow_S2P_hindbrain.py - not a great idea
    prelim = np.unique(cluster_membership)
    prelim = prelim[prelim > -1]
    for p in prelim:
        planes = all_plane_ids[cluster_membership == p]
        if np.unique(planes).size < 25:
            cluster_membership[cluster_membership == p] = -1

    # assign cluster membership to each neuron in neuron_origin
    start_ix = 0
    for k in neuron_origin:
        for kk in neuron_origin[k]:
            orig_ix = neuron_origin[k][kk]
            orig_clus_mem = cluster_membership[start_ix:start_ix+orig_ix.size]
            start_ix += orig_ix.size
            neuron_origin[k][kk] = (orig_ix, orig_clus_mem)

    clus_nums = np.unique(cluster_membership)
    clus_nums = clus_nums[clus_nums > -1]

    # collect all plane images and neuron coordinates
    z_spacing = 5
    zoom = 1.25
    res = 500/512/zoom

    # to compute spatial clustering, we need to store not only the coordinates of selected
    # neurons, but all s2p identified neurons - to this end we give those neurons that are
    # not heat responsive a cluster-id of -2
    full_neuron_coords = []
    full_neuron_clix = []
    full_neuron_fid = []  # so we can do this per-fish

    # link registered coordinates and date fish ids
    registered_folder = path.join(hdf_dir, "All_coordinates")
    registered_files = [f for f in os.listdir(registered_folder) if "transformed.csv" in f]
    registered_coordinates = {}
    for rf in registered_files:
        datum = rf.split("_")[0]
        registered_coordinates[datum] = load_anatomy_csv(path.join(registered_folder, rf))

    tempres_registered_coords = []

    for fid, k in enumerate(neuron_origin):
        all_coordinates = []
        anat_folder = path.join(k, "anatomy")
        tempres_registered_coords.append(registered_coordinates[path.split(k)[-1]])
        for kk in neuron_origin[k]:
            plane = path.split(path.split(kk)[0])[0]
            orig_ix, orig_clus_mem = neuron_origin[k][kk]
            suffix = re.search(r'_Z_\d+_0$', plane)
            plane_id = int(re.search(r'_\d+_', suffix.group()).group()[1:-1])

            if not path.exists(anat_folder):
                os.makedirs(anat_folder)
            plane_name = path.split(plane)[1]

            if orig_ix.size > 0:
                # LIMIT BY THE SAME IS CELL CRITERION USED DURING FITTING
                is_cell = np.load(path.join(kk, "iscell.npy"))[:, 1] > .5
                s2pstats = np.load(path.join(kk, 'stat.npy'), allow_pickle=True)
                assert s2pstats.shape[0] > np.sum(is_cell)
                s2pstats = s2pstats[is_cell]
                # coordinates are stored as y followed by x
                # coords for heat responsive neurons
                plane_co = [np.hstack(s2s['med']) for s2s in s2pstats[orig_ix]]
                plane_co = np.vstack(plane_co)
                # coords for all neurons as well as augmented cluster index
                full_co = [np.hstack(s2s['med']) for s2s in s2pstats]
                full_co = np.vstack(full_co)
                full_clix = np.full(full_co.shape[0], -2)
                full_clix[orig_ix] = orig_clus_mem
                plane_co = np.c_[plane_co*res, np.ones((plane_co.shape[0], 1))*z_spacing*plane_id]
                full_co = np.c_[full_co*res, np.ones((full_co.shape[0], 1))*z_spacing*plane_id]
                all_coordinates.append(np.round(plane_co, 3))
                full_neuron_coords.append(full_co)
                full_neuron_clix.append(full_clix)
                full_neuron_fid.append(np.full(full_clix.size, fid))
        all_coordinates = np.vstack(all_coordinates)
        np.savetxt(path.join(anat_folder, f"{path.split(k)[-1]}_all_coordinates.csv"), all_coordinates, fmt='%.3f')

    full_neuron_coords = np.vstack(full_neuron_coords)
    full_neuron_clix = np.hstack(full_neuron_clix)
    full_neuron_fid = np.hstack(full_neuron_fid)
    tempres_registered_coords = np.vstack(tempres_registered_coords)

    hot_clus_nums = [1, 2, 6]
    cold_clus_nums = [0, 3, 5, 7]

    # for each cluster we plot a maximum of 250 neurons
    hot_clus_registered = []
    for ci in hot_clus_nums:
        ci_trc = tempres_registered_coords[cluster_membership==ci]
        p_include = 250 / ci_trc.shape[0]
        include = np.random.rand(ci_trc.shape[0]) < p_include
        hot_clus_registered.append(ci_trc[include])
    hot_clus_registered = np.vstack(hot_clus_registered)
    cold_clus_registered = []
    for ci in cold_clus_nums:
        ci_trc = tempres_registered_coords[cluster_membership == ci]
        p_include = 250 / ci_trc.shape[0]
        include = np.random.rand(ci_trc.shape[0]) < p_include
        cold_clus_registered.append(ci_trc[include])
    cold_clus_registered = np.vstack(cold_clus_registered)

    fig = pl.figure()
    pl.scatter(tempres_registered_coords[cluster_membership==-1, 0], tempres_registered_coords[cluster_membership==-1, 1], color='k', rasterized=True, edgecolors=None, alpha=0.05, s=1)
    pl.scatter(hot_clus_registered[:, 0], hot_clus_registered[:, 1], s=2, alpha=0.5, color="C3", rasterized=True, edgecolors='none')
    pl.scatter(cold_clus_registered[:, 0], cold_clus_registered[:, 1], s=2, alpha=0.5, color="C0", rasterized=True, edgecolors='none')
    pl.axis('equal')
    fig.savefig(path.join(plot_dir, "REVISION_S4A_Hot_and_Cold_neurons_top.pdf"))

    for ci in clus_nums:
        pl.figure()
        alpha_val = 1 / np.sum(cluster_membership==ci) * 500
        if alpha_val > 1:
            alpha_val = 1
        pl.scatter(tempres_registered_coords[cluster_membership==ci, 0], tempres_registered_coords[cluster_membership==ci, 1], s=2, alpha=alpha_val)

    # Compute and plot per-cluster Nearest-neighbor distances relative to shuffle controls (=expectation)
    n_shuffles = 100
    d_pwd = {"Type": [], "Normalized distance": [], "Normalized NN": []}
    clus_name = ["CA", "H", "HC", "C", "CC", "HH", "CH"]
    clus_name_by_number = {0: "CA",
                           1: "H",
                           2: "HC",
                           3: "C",
                           5: "CC",
                           6: "HH",
                           7: "CH"}
    full_fish = np.unique(full_neuron_fid)

    # Compute a spatial clustering metric which quantifies (compared to shuffles) how much closer within-cluster
    # nearest neighbors are vs. nearest neighbors in the outgroup. Note that for comparing two clusters, this will
    # lead to two distinct values, one for each cluster which do not have to be the same

    def cluster_metric(all_ident: np.ndarray, cat_name_dict: {}, n_shuffle: int, n_neighbors: int):
        """
        For clusters compute, relative to shuffles, the median nearest neighbor distance to other cluster neurons
        versus within-cluster neurons (i.e. metric > 1 means spatial clustering)
        :param all_ident: The cluster identities of all neurons - identity < 0 means not part of any cluster
        :param cat_name_dict: Dictionary relating category indices to their names
        :param n_shuffle: The number of shuffles to compute for expectation
        :param n_neighbors: The number of nearest neighbors to consider
        :return: Dictionary which for each fish indicates type names and cluster metric (for dataframe creation)
        """
        d_cluster_metric = {"Type": [], "Spatial clustering": []}
        for ff in full_fish:
            these_ids = all_ident[full_neuron_fid == ff]
            shuffle_ids = these_ids.copy()
            these_coords = full_neuron_coords[full_neuron_fid == ff]
            unique_ids = np.unique(these_ids[these_ids >= 0])
            if len(unique_ids) < 2 or min([np.sum(these_ids==u) for u in unique_ids]) < n_neighbors+1:
                # we cannot compute this metric if clusters do not have an outgroup
                continue
            for ID in unique_ids:
                in_cluster_coords = these_coords[these_ids == ID]
                out_cluster_coords = these_coords[np.logical_and(these_ids >= 0, these_ids != ID)]
                nbrs_in = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(in_cluster_coords)
                nbrs_out = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(out_cluster_coords)
                # The first neighbor is the point itself, so nearest neighbors are at indices 1:n
                all_in_dist = nbrs_in.kneighbors(in_cluster_coords)[0][:, 1:]
                assert all_in_dist.shape[1] == n_neighbors
                assert np.min(all_in_dist) > 0
                in_cluster_proximity = np.mean(np.mean(1/all_in_dist, axis=1))
                # The first neigbor is NOT the point itself
                all_out_dist = nbrs_out.kneighbors(in_cluster_coords)[0]
                assert all_out_dist.shape[0] == np.sum(these_ids == ID)
                assert all_out_dist.shape[1] == n_neighbors
                assert np.min(all_out_dist) > 0
                out_cluster_proximity = np.mean(np.mean(1/all_out_dist, axis=1))
                data_cluster_metric = in_cluster_proximity / out_cluster_proximity
                shuffle_cluster_metric = 0
                for s in range(n_shuffle):
                    np.random.shuffle(shuffle_ids)
                    sh_in_cluster_coords = these_coords[shuffle_ids == ID]
                    sh_out_cluster_coords = these_coords[np.logical_and(shuffle_ids >= 0, shuffle_ids != ID)]
                    nbrs_in = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(sh_in_cluster_coords)
                    nbrs_out = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(sh_out_cluster_coords)
                    sh_all_in_dist = nbrs_in.kneighbors(sh_in_cluster_coords)[0][:, 1:]
                    assert np.min(sh_all_in_dist) > 0
                    sh_all_out_dist =nbrs_out.kneighbors(sh_in_cluster_coords)[0]
                    assert np.min(sh_all_out_dist) > 0
                    sh_in_cluster_prox = np.mean(np.mean(1/sh_all_in_dist, axis=1))
                    sh_out_cluster_prox = np.mean(np.mean(1/sh_all_out_dist, axis=1))
                    shuffle_cluster_metric += sh_in_cluster_prox / sh_out_cluster_prox
                shuffle_cluster_metric /= n_shuffle
                print(shuffle_cluster_metric)
                d_cluster_metric["Type"].append(cat_name_dict[ID])
                d_cluster_metric["Spatial clustering"].append(data_cluster_metric / shuffle_cluster_metric)
        return d_cluster_metric

    hot_cold_coded = np.full_like(full_neuron_clix, np.nan)
    for i, clix in enumerate(full_neuron_clix):
        if clix in hot_clus_nums:
            hot_cold_coded[i] = 0
        elif clix in cold_clus_nums:
            hot_cold_coded[i] = 1
        elif clix < 0:
            hot_cold_coded[i] = -1
        else:
            assert False

    hot_cold_clustering = cluster_metric(hot_cold_coded, {0: "Hot", 1: "Cold"}, 1000, 5)

    fig, ax = pl.subplots()
    sns.pointplot(data=hot_cold_clustering, x="Type", y="Spatial clustering", join=False, color='black',
                  errorbar=("ci", 95), ax=ax)
    pl.plot([0, 2], [1, 1], "C3--")
    pl.ylabel("Spatial clustering [w. 95 % CI]")
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "REVISION_4A_spatial_htcold_clusterings.pdf"))

    tg_clus_nums = [0, 1, 3]
    monly_clus_nums = [2, 5, 6, 7]
    tg_medulla_coded = np.full_like(full_neuron_clix, np.nan)
    for i, clix in enumerate(full_neuron_clix):
        if clix in tg_clus_nums:
            tg_medulla_coded[i] = 0
        elif clix in monly_clus_nums:
            tg_medulla_coded[i] = 1
        elif clix < 0:
            tg_medulla_coded[i] = -1
        else:
            assert False

    tg_medulla_clustering = cluster_metric(tg_medulla_coded, {0: "TG Encoded", 1: "Medulla only"}, 1000, 5)

    fig, ax = pl.subplots()
    sns.pointplot(data=tg_medulla_clustering, x="Type", y="Spatial clustering", join=False, color='black',
                  errorbar=("ci", 95), ax=ax)
    pl.plot([0, 2], [1, 1], "C3--")
    pl.ylabel("Spatial clustering [w. 95 % CI]")
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "REVISION_4B_spatial_tgNontg_clusterings.pdf"))
