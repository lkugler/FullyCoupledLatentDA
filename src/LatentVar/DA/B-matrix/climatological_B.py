# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for computing the climatological B-matrix
"""

deterministic = False#True #False
use_compile = True

import argparse
import pickle
import psutil
import os

process = psutil.Process(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument('--AE_version', help="Version of AE", type=str, required=False, default='v63')
parser.add_argument("--forecast_len", help="Forecast length in hours", type=int, required=False, default=24)
parser.add_argument("--idx_reduction", help="Use every idx_reduction-th temporal instance to compute B", type=int, required=False, default=1)
parser.add_argument("--plot", help='Plot B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compute", help='Compute B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--persistence", help='Use persistence model instead of forward model', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--only_diag", help='Compute, plot, and store only the diagonaly elements', default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()


# # Osnovno
# import os
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
#%%
import sys
# input('imported clean state dict')
sys.path.append('../..')
from general_ae_info import ae_props, date_to_dataidx, standardise_destandardise
AE_props = ae_props(args.AE_version)
sys.path.append(str(AE_props.BASE_PATH))
from utilities import clean_state_dict
import os
if deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from scheduler import WarmupScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import importlib
import custom_loss



import gc
from datetime import datetime, timedelta




from pathlib import Path
FIGS = f"{AE_props.vPATH}/DA/B-matrix/figs"


tst = datetime.now()


# FWD
if args.persistence:
    FWD_model_name = 'persistence'
else:
    raise AttributeError("So far I've only implemented B-matrix computation for persistence forecast")

import datasets
importlib.reload(datasets)

# -----------------------------
# Device setup
# -----------------------------
free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
best_gpu = free_mem.index(max(free_mem))
device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Free memory on GPUs: {free_mem}")



# -----------------------------
# Model Initialization
# -----------------------------
# import models10

import subgraphs
importlib.reload(subgraphs)

from subgraphs import build_multiscale_graph, build_hierarchical_graph_v4, \
    build_hierarchical_graph_v5, create_pooling_matrices_v2, build_hierarchical_graph_v6

if AE_props.use_pooling:
    graphs = []
    for lev in range(AE_props.N_SUBGRAPHS+1):
        graphs.append(torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_edge_index_ae_{lev}.pt"))
    pooling_matrices, unpooling_matrices = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_pooling_matrices.pt")
    proper_level_connections = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_{AE_props.level_connections_type}.pt")

    graphs = [g.to(device) for g in graphs]
    pooling_matrices = [p.to(device) for p in pooling_matrices]
    unpooling_matrices = [u.to(device) for u in unpooling_matrices]

else:
    AE_edge_index = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_edge_index_ae.pt")
    AE_edge_index = AE_edge_index.to(device)

# importlib.reload(models10)
importlib.reload(AE_props.modelslib)

if deterministic:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

AE_model = AE_props.modelslib.ProgressiveGraphAutoencoder(
        in_dim=AE_props.IN_DIM, 
        encoder_hidden_dims=AE_props.HIDDEN_DIMS, 
        decoder_hidden_dims=AE_props.HIDDEN_DIMS[::-1], 
        latent_dim=AE_props.LATENT_DIM, 
        out_dim=AE_props.OUT_DIM,
        graphs = graphs, 
        level_connections = proper_level_connections,
        heads=AE_props.HEADS, 
        gat_dropout=AE_props.GAT_DROPOUT, feature_dropout=AE_props.FEATURE_DROPOUT, latent_dropout=AE_props.LATENT_DROPOUT, latent_noise_std=AE_props.LATENT_NOISE_STD,
        use_residuals=AE_props.use_residuals, use_residualsIO=AE_props.use_residualsIO).to(device)


print('\nThe path:', AE_props.best_model_pth, '\n')

best_AE_model = torch.load(AE_props.best_model_pth, map_location=device)  # use the latest or best
if (use_compile or use_distributed):# and args.AE_version != 'v55':
    AE_model.load_state_dict(clean_state_dict(best_AE_model))
else:
    AE_model.load_state_dict(best_AE_model)
AE_model = AE_model.to(device)
AE_model.eval()

# input('evaluated')


start_idx = 0
end_idx = AE_props.nt

# Dataset splits

# Now we only have 2 years of data, so we use it all
# idx_ic = range(start_idx + AE_props.n_train, start_idx + AE_props.n_train + AE_props.n_val, args.idx_reduction)
# idx_truth = range(start_idx + AE_props.n_train + args.forecast_len//AE_props.data_dt, start_idx + AE_props.n_train + AE_props.n_val + args.forecast_len//AE_props.data_dt, args.idx_reduction)
idx_ic = range(start_idx, end_idx - args.forecast_len//AE_props.data_dt, args.idx_reduction)
idx_truth = range(start_idx + args.forecast_len//AE_props.data_dt, end_idx, args.idx_reduction)

print([i for i in idx_ic])
print([i for i in idx_truth])

ic_dataset = datasets.ReallyLazyGraphDataset_v2(
    idx_ic, AE_props.contract.data_path, \
    AE_props.time_varying_variables, AE_props.static_variables, \
    AE_props.reconstructed_variables)

truth_dataset = datasets.ReallyLazyGraphDataset_v2(
    idx_truth, AE_props.contract.data_path, \
    AE_props.time_varying_variables, AE_props.static_variables, \
    AE_props.reconstructed_variables)

# free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
# best_gpu = free_mem.index(max(free_mem))
# device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(f"Free memory on GPUs: {free_mem}")
# print(f'\nCPU memory usage {process.memory_info().rss/1024**2} MB\n')



print(f'\nafter evaluating model CPU memory usage {process.memory_info().rss/1024**2} MB\n')
# --------------------------------------------------------------------------
# NASTAVITVE ZA SHRANJEVANJE
# --------------------------------------------------------------------------

B_matrix_savename = f'{AE_props.vPATH}/DA/B-matrix/matrices/{FWD_model_name}/{AE_props.EXP_ID}_climatological_prediction' +\
                    f'_len_{args.forecast_len}h'
if args.idx_reduction > 1:
    B_matrix_savename += f'_red_{args.idx_reduction}'
if args.only_diag:
    B_matrix_savename += '_diag'


# latent vector v bistvu ne bo v vektor, temvec "latentna matrika" dimenzije (nlatnode, latent_dim)
starting_vector = torch.zeros((AE_props.nlatnode, AE_props.latent_dim))
# Jb = torch.sum(torch.sum(v * B * v, axis=1)), B has shape (nlatnode, latent_dim)
# keep the 2D "diagonal" structure (this is equivalent to reshaping)
B_matrix_diag = starting_vector * starting_vector
B_matrix_diag = B_matrix_diag.to(device)

print(f'\ndiag CPU memory usage {process.memory_info().rss/1024**2} MB\n')

if not args.only_diag:
    # B_matrix_diag = torch.outer(starting_vector.reshape((nlatnode * latent_dim)), starting_vector.reshape((nlatnode * latent_dim)))
    # B_matrix_offdiag_indices = torch.triu_indices(nlatnode*latent_dim - 1, nlatnode*latent_dim - 1)
    print(f'\noffdiag indices CPU memory usage {process.memory_info().rss/1024**2} MB\n')
    B_matrix_offdiag_upper = torch.zeros(((AE_props.nlatnode*AE_props.latent_dim) * (AE_props.nlatnode*AE_props.latent_dim - 1) //2)) #torch.zeros(B_matrix_offdiag_indices.shape[1])
    print(f'\nupper CPU memory usage {process.memory_info().rss/1024**2} MB\n')
    # B_matrix_offdiag_upper.to(device)
    all_bg_errors = torch.zeros((len([i for i in idx_ic]), AE_props.nlatnode*AE_props.latent_dim))
    print(f'\nbg errors CPU memory usage {process.memory_info().rss/1024**2} MB\n')
    all_bg_errors.to(device)

B_matrix_len = 0
# B_matrix_len.to(device)
del starting_vector
gc.collect()



if args.compute:
    # --------------------------------------------------------------------------
    # IZRACUN
    # --------------------------------------------------------------------------

    print('Total temporal instances:', len([i for i in idx_ic]))
    pidx = 100

    with torch.no_grad():

        for i in range(len(ic_dataset)):
            if B_matrix_len % pidx == 0:
                et = datetime.now()
                try:
                    print(B_matrix_len, (et - st) / pidx)
                except:
                    print(B_matrix_len)
                st = datetime.now()
            # 1) Compute forecast
            sample_ic = ic_dataset[i].to(device)
            if args.persistence:
                # Persistence forecast: fc = ic
                encoded_forecast = AE_model.encode(sample_ic.x)#, batch_ic.edge_index)
                # ec1 = AE_model1.encode(sample_ic.x)#, batch_ic.edge_index)
                # ec2 = AE_model2.encode(sample_ic.x)#, batch_ic.edge_index)
                # print(torch.amax(ec2 - ec1))
                # input('..')
            else:
                raise AttributeError("Computation only implemented for persistence model")

            # 2) Encode truth
            sample_truth = truth_dataset[i].to(device)
            encoded_truth = AE_model.encode(sample_truth.x)#, batch_truth.edge_index)

            test_decode = AE_model.decode(encoded_truth)
            print(test_decode.shape)


            background_error = encoded_forecast - encoded_truth
            # B_matrix_diag.to(device)
            # print(device)
            # print(background_error.device, B_matrix_diag.device)
            # background_error.to(device)
            # bge2 = background_error.to(device)
            # B_matrix_diag = B_matrix_diag * (B_matrix_len / (B_matrix_len + 1)) + bge2 * bge2 * (1 / (B_matrix_len + 1))

            print(B_matrix_diag.shape)
            print(background_error.shape)
            B_matrix_diag = B_matrix_diag * (B_matrix_len / (B_matrix_len + 1)) + background_error * background_error * (1 / (B_matrix_len + 1))

            if not args.only_diag:
                all_bg_errors[B_matrix_len, :] = background_error.reshape(AE_props.nlatnode * AE_props.latent_dim).clone()

            B_matrix_len += 1

        if not args.only_diag:
            tot_offdiag = 0
            for irow in range(AE_props.nlatnode * AE_props.latent_dim - 1):
                if irow % 10000 == 0 or irow == 1 or irow == 2:
                    print(irow)
                B_matrix_offdiag_upper_row = torch.mean(all_bg_errors[:, irow:irow+1] * all_bg_errors[:, irow+1:], axis=0)
                len_these_offdiag = len(B_matrix_offdiag_upper_row)
                # print(len_these_offdiag, tot_offdiag)
                # input()
                # try:
                B_matrix_offdiag_upper[tot_offdiag:tot_offdiag+len_these_offdiag] = B_matrix_offdiag_upper_row.cpu()
                # except:
                #     print(irow, tot_offdiag, tot_offdiag+len_these_offdiag, B_matrix_offdiag_upper_row.cpu().shape)
                #     input()
                tot_offdiag += len_these_offdiag

            print('off diagonals with zero value', len(B_matrix_offdiag_upper) - torch.count_nonzero(B_matrix_offdiag_upper))

    if args.only_diag:
        torch.save(B_matrix_diag, B_matrix_savename + '.pt')
            

# v4


if args.plot:
    print('plotting')
    import matplotlib.pyplot as plt
    #B_matrix_diag = pickle.load(open(B_matrix_savename + '.pkl', 'rb'))
    B_matrix_diag = B_matrix_diag.cpu()
    log_diagonals = np.zeros(shape=(len(B_matrix_diag)))
    # log_off_diagonals = np.zeros(shape=((len(B_matrix_diag) - 1) * len(B_matrix_diag) // 2))
    off_diags_end_idx = 0

    # greater_than_1 = 0
    # for irow in range(len(B_matrix_diag) - 1):
    #     # print(irow, B_matrix_diag[irow])
    #     log_diagonals[irow] = np.log10(B_matrix_diag[irow][irow])
    #     off_diags_start_idx = off_diags_end_idx
    #     off_diags_end_idx += len(B_matrix_diag) - irow - 1
    #     log_off_diagonals[off_diags_start_idx:off_diags_end_idx] = np.log10(np.abs(B_matrix_diag[irow][irow + 1:]))
    #     if irow == 0:
    #         worst_ratio = 10**max(log_off_diagonals[off_diags_start_idx:off_diags_end_idx])/10**log_diagonals[irow]
    #         if worst_ratio > 1:
    #             greater_than_1 += 1
    #     else:
    #         worst_ratio_right = 10**max(log_off_diagonals[off_diags_start_idx:off_diags_end_idx])/10**log_diagonals[irow]
    #         worst_ratio_left = max(np.abs(B_matrix_diag[irow][:irow]))/10**log_diagonals[irow]
    #         if max(worst_ratio_left, worst_ratio_right) > 1:
    #             greater_than_1 += 1
    #         worst_ratio = max(worst_ratio, worst_ratio_right, worst_ratio_left)

    # log_diagonals[-1] = np.log10(B_matrix_diag[-1][-1])
    # worst_ratio_left = max(np.abs(B_matrix_diag[-1][:-1]) / 10 ** log_diagonals[-1])
    # worst_ratio = max(worst_ratio, worst_ratio_left)
    # print('WORST RATIO', worst_ratio)
    # print('Ratios greater that 1', greater_than_1)

    # def max_offdiag_diag_ratio(matrix):
    #     matrix = np.array(matrix)
    #     diag = np.abs(np.diag(matrix))
        
    #     # Mask for non-diagonal elements
    #     mask = ~np.eye(matrix.shape[0], dtype=bool)
        
    #     # Compute absolute matrix
    #     abs_matrix = np.abs(matrix)
        
    #     # Set diagonal elements to NaN temporarily
    #     abs_matrix[~mask] = np.nan
        
    #     # Max of off-diagonal elements per row
    #     max_offdiag = np.nanmax(abs_matrix, axis=1)
        
    #     # Compute ratio
    #     ratio = max_offdiag / diag
        
    #     return np.max(ratio)
    # print(max_offdiag_diag_ratio(B_matrix_diag))


    def n_next_to_diagonal(matrix, n):
        # Only counts those to the right of the diagonal (+ the ones on the diagonal),
        # thus only suitable for symmetrical matrices
        # print(f'Seeking {n} next to diagonal')
        def n_next_to_diagonal_one_row(row, irow, n):
            # Problem can arise, if row ends before n
            diag_position = irow
            off_diags_available = len(row) - diag_position - 1
            if off_diags_available > n:
                return row[diag_position:diag_position + n + 1]
            else:
                return row[diag_position:]

        next_to_diagonals = np.zeros(shape=((n + 1) * len(matrix) - n * (n + 1) // 2))
        append_end_idx = 0
        for irow in range(len(matrix)):
            append_start_idx = append_end_idx
            n_next_to_diagonal_this_row = n_next_to_diagonal_one_row(matrix[irow], irow, n)
            append_end_idx = append_start_idx + len(n_next_to_diagonal_this_row)
            next_to_diagonals[append_start_idx:append_end_idx] = n_next_to_diagonal_this_row


        # print(f'Sought {n} next to diagonal')
        return next_to_diagonals

    # # ns = [0, 5, 10, 50, 100, 22*45, 22*45*50-1]
    # ns = [22*45*50-1, 22*25, 100, 50, 10, 5, 0]
    # log_next_to_diags = [np.log10(np.abs(n_next_to_diagonal(B_matrix_diag, n))) for n in ns]
    colors = ['C0', 'C1', 'C2', 'C3', 'gray', 'C5', 'C6']

    # del B_matrix_diag
    # gc.collect()

    # threshold = -1.5
    # log_diagonals_below_threshold = sum(log_diagonals < threshold)
    # print(f'log diagonals below {threshold}: {log_diagonals_below_threshold}')

    # logstart = -10  # Proposition: logdist >= 5
    # logend = 5
    # nbin = 5 * (logend - logstart) + 1  # 5*logdist + 1
    # bins = np.linspace(logstart, logend, nbin)
    # plt.hist(log_diagonals, bins=bins, density=False, alpha=0.8, label='Diagonal elements')
    # print('got diagonals')
    # plt.hist(log_off_diagonals, bins=bins, density=False, alpha=0.8, label='Off-diagonal elements')
    # print('got off-diagonals')
    # plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z$ element))')
    # plt.xlim(min(bins), max(bins))
    # plt.ylabel('Number of elements')
    # plt.legend()
    # plt.yscale('log')
    # plt.ylim(bottom=0.5)
    # addname = ''
    # if not args.persistence:
    #     plt.savefig(
    #         'figures/' + f'hist_{AE_root_model[AE_root_model.rfind("/")+1:]}' +\
    #                     f'_climatological_prediction_{args.start_date}_to_{args.end_date}' +\
    #                     f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_len}' +\
    #                     f'{addname}.pdf', dpi=300)
    # else:
    #     plt.savefig(
    #         'a.pdf', dpi=300)

    # logstart = -10  # Proposition: logdist >= 5
    # logend = 5
    # nbin = 5 * (logend - logstart) + 1  # 5*logdist + 1
    # bins = np.linspace(logstart, logend, nbin)
    # for inext in range(len(ns)):
    #     plt.hist(log_next_to_diags[inext], bins=bins, color=colors[inext], density=False, alpha=1, label=f'Diagonal + {ns[inext]} elements')
    # plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z$ element))')
    # plt.xlim(min(bins), max(bins))
    # plt.ylabel('Number of elements')
    # plt.legend()
    # plt.yscale('log')
    # plt.ylim(bottom=0.5)
    #
    # addname = ''
    # if args.latent_precondition:
    #     addname += '_latent_precondition'
    # if not args.persistence:
    #     plt.savefig(
    #         'figures/' + f'hist_nnext_{AE_root_model[AE_root_model.rfind("/") + 1:]}' + \
    #         f'_climatological_prediction_{args.start_date}_to_{args.end_date}' + \
    #         f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_steps}' + \
    #         f'{addname}.pdf', dpi=300)
    # else:
    #     plt.savefig(
    #         'figures/' + f'hist_nnext_{AE_root_model[AE_root_model.rfind("/") + 1:]}' + \
    #         f'_climatological_persistence_prediction_{args.start_date}_to_{args.end_date}' + \
    #         f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_steps}' + \
    #         f'{addname}.pdf', dpi=300)

    plt.cla()
    plt.clf()
    plt.figure(figsize=(7, 6))
    import matplotlib
    matplotlib.rcParams.update({"font.size": 16})

    # nbin = 31
    minbin, maxbin = -5, 3
    nbin = (maxbin - minbin) * 5 + 1
    bins = np.linspace(minbin, maxbin, nbin)
    # counts_diag, _ = np.histogram(np.log10(np.diagonal(B_matrix_diag)), bins=bins, density=True)
    counts_diag, _ = np.histogram(np.log10(B_matrix_diag.flatten()), bins=bins, density=True)
    plt.bar((bins[:-1] + bins[1:])/2, counts_diag, width=np.diff(bins), alpha=0.8, label='Diagonal elements')
    print('got diags')
    if not args.only_diag:
        # counts_offdiag, _ = np.histogram(np.log10(np.abs(B_matrix_diag - B_matrix_diag * np.identity(len(B_matrix_diag))).flatten()), bins=bins, density=True)
        counts_offdiag, _ = np.histogram(np.log10(np.abs(B_matrix_offdiag_upper)), bins=bins, density=True)
        plt.bar((bins[:-1] + bins[1:])/2, counts_offdiag, width=np.diff(bins),alpha=0.5, label='Off-diagonal elements')
        print('got offdiags')
    #plt.hist(np.log10(np.diagonal(B_matrix_diag)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
    # plt.hist(np.log10(np.abs(B_matrix_diag - B_matrix_diag * np.identity(len(B_matrix_diag))).flatten()), bins=bins, density=True,
    #          alpha=0.5, label='Off-diagonal elements')
    plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z^{clim}$ element))')
    plt.xlim(min(bins), max(bins))
    plt.ylabel('Share')
    plt.legend(loc='upper left')
    plt.title(r'Distribution of $\mathbf{B}_z^{clim}$ elements' + f', {AE_props.EXP_ID}')
    plt.yticks(ticks=nbin / (max(bins) - min(bins)) * np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
               labels=[r'$0\,\%$', r'$10\,\%$', r'$20\,\%$', r'$30\,\%$', r'$40\,\%$', r'$50\,\%$'])  # to mapiranje postudiraj
    plt.tight_layout()

    plt.savefig(f'{FIGS}/{AE_props.EXP_ID}_hist_{FWD_model_name}_{B_matrix_savename.split("/")[-1]}_share.pdf', dpi=300)