import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib as mpl
import argparse
import os
from os import path
from model import get_standard_model
import utilities
from taylorDecomp import complexity_scores, d2ca_dr2
import seaborn as sns
from cluster_HtColdFlow_S2P_hindbrain import set_journal_style


if __name__ == '__main__':
    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    a_parser = argparse.ArgumentParser(prog="spatial_HtColdFlow_S2P_hindbrain",
                                       description="Runs spatial analysis of clustered neurons in the medulla")
    a_parser.add_argument("-d", "--directory", help="Path to folder with response Hdf5 file", type=str, required=True)

    args = a_parser.parse_args()

    hdf_dir = args.directory

    plot_dir = "REVISION_nonlin_hindbrain"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    hdf_files = [path.join(hdf_dir, f) for f in os.listdir(hdf_dir) if ".hdf5" in f and "cluster" not in f]
    exp_folders = [f[:-5] + "/" for f in hdf_files]

    all_lin_scores = []
    all_sq_scores = []

    if path.exists(path.join(hdf_dir, "all_lin_scores.npy")) and path.exists(path.join(hdf_dir, "all_sq_scores.npy")):
        all_lin_scores = np.load(path.join(hdf_dir, "all_lin_scores.npy"))
        all_sq_scores = np.load(path.join(hdf_dir, "all_sq_scores.npy"))
    else:
        # collect and concatenate nonlinearity data of all neurons that underwent clustering (i.e. test_corr > 0.6)
        for i, f in enumerate(hdf_files):
            e_folder = exp_folders[i]
            m = get_standard_model(50)  # same history lenght as used during fitting
            m(np.random.randn(1, 50, 1).astype(np.float32))  # initialize to same shape
            x_bar = np.zeros((1, 50, 1))
            with h5py.File(f, "r") as dfile:
                for k in dfile:
                    if "weights" in k:
                        # we actuallly load the models based on the name of the non-weight group to later align test
                        # correlations and cluster numbers with neuron measures
                        continue
                    print(k)
                    temperature_file = path.join(e_folder, k[:-7] + ".temp")
                    temperature = np.genfromtxt(temperature_file)
                    temperature_regressor = temperature[:, None]
                    grp = dfile[k]
                    w_grp = dfile[k+"_weights"]
                    indices = grp["heat_res_indices"][()]
                    test_corrs = grp["heat_res_testcorr"][()]
                    temp_mean = grp["temp_mean"][()]
                    temp_std = grp["tmp_std"][()]
                    temperature_regressor -= temp_mean
                    temperature_regressor /= temp_std
                    # filter by test correlation the same we we do before clustering
                    indices = indices[test_corrs > 0.6]
                    # load models based on index
                    for neuron_ix in indices:
                        m_weights = utilities.modelweights_from_hdf5(w_grp[f"cell_{neuron_ix}_weights"])
                        m.set_weights(m_weights)
                        jac, hess = d2ca_dr2(m, x_bar)
                        lin, sq = complexity_scores(m, x_bar, jac, hess, temperature_regressor, 25)
                        all_lin_scores.append(lin)
                        all_sq_scores.append(sq)

        all_lin_scores = np.array(all_lin_scores)
        all_sq_scores = np.array(all_sq_scores)

        np.save(path.join(hdf_dir, "all_lin_scores.npy"), all_lin_scores)
        np.save(path.join(hdf_dir, "all_sq_scores.npy"), all_sq_scores)

    if (path.exists(path.join(hdf_dir, "cluster_membership.npy")) and
            np.load(path.join(hdf_dir, "cluster_membership.npy")).size==all_lin_scores.shape[0]):
        cluster_membership = np.load(path.join(hdf_dir, "cluster_membership.npy"))
    else:
        raise NotImplementedError()  # clustering is not performed by here but in cluster_HtColdFlow_S2P_hindbrain.py

    all_lin_scores = all_lin_scores[cluster_membership > -1]
    all_sq_scores = all_sq_scores[cluster_membership > -1]

    # For all clustered neurons plot the cumulative distribution of linear approximation scores
    fig = pl.figure()
    cumulative = np.zeros(101)
    for i, v in enumerate(np.linspace(-1,1, 101)):
        cumulative[i] = np.sum(all_lin_scores <= v)
    pl.plot(np.linspace(-1, 1, 101), cumulative/all_lin_scores.size)
    nonlin_frac = np.sum(all_lin_scores<=0.5)/all_lin_scores.size
    pl.plot([0.5, 0.5], [0, nonlin_frac], 'k--')
    pl.plot([-1, 0.5], [nonlin_frac, nonlin_frac], 'k--')
    pl.xlim(-1, 1)
    pl.ylim(0, 1.01)
    pl.ylabel("Cumulative fraction")
    pl.xlabel("Linear approximation R2")
    sns.despine()
    fig.savefig(path.join(plot_dir, "REVISION_S3F_Linear_model_Approx.pdf"))

    # For all nonlinear neurons (lin score < 0.5) plot the cumulative distribution of squared approximation scores
    nonlin_sq = all_sq_scores[all_lin_scores < 0.5]
    fig = pl.figure()
    cumulative = np.zeros(101)
    for i, v in enumerate(np.linspace(-1,1, 101)):
        cumulative[i] = np.sum(nonlin_sq <= v)
    pl.plot(np.linspace(-1, 1, 101), cumulative/nonlin_sq.size)
    cubic_frac = np.sum(nonlin_sq<=0.5)/nonlin_sq.size
    pl.plot([0.5, 0.5], [0, cubic_frac], 'k--')
    pl.plot([-1, 0.5], [cubic_frac, cubic_frac], 'k--')
    pl.xlim(-1, 1)
    pl.ylim(0, 1.01)
    pl.ylabel("Cumulative fraction")
    pl.xlabel("2nd order approximation R2")
    sns.despine()
    fig.savefig(path.join(plot_dir, "REVISION_S3G_2ndorder_model_Approx.pdf"))
