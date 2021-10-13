import os
import random
import math

import torch
import numpy as np
import scipy.sparse as sp
import pickle as pk
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import numpy.random as rnd

from param_parser import parameter_parser
from utils import polarized_ssbm, SSBM, fix_network, spectral_adjacency_reg, signed_Laplacian_features

def to_dataset(args, A_p, A_n, label, save_path, feat_given=None, conflict_groups=None, load_only=False):
    labels = label
    N = A_p.shape[0]
    idx = np.arange(N)
    num_clusters =  int(np.max(labels) + 1)
    clusters_sizes = [int(sum(labels==i)) for i in range(num_clusters)]
    test_sizes = [math.ceil(clusters_sizes[i] * args.test_ratio) for i in range(num_clusters)]
    val_ratio = 1 - args.train_ratio - args.test_ratio
    val_sizes = [math.ceil(clusters_sizes[i] * val_ratio) for i in range(num_clusters)]
    
    masks = {}
    masks['train'], masks['val'], masks['test'], masks['seed'] = [], [] , [], []
    for _ in range(args.num_trials):
        idx_test = []
        idx_val = []
        for i in range(num_clusters):
            idx_test_ind = random.sample(range(clusters_sizes[i]), k=test_sizes[i])
            idx_test.extend((np.array(idx)[labels==i])[idx_test_ind])
        idx_remain = list(set(idx).difference(set(idx_test))) # the rest of the indices
        clusters_sizes_remain = [int(sum(labels[idx_remain]==i)) for i in range(num_clusters)]
        for i in range(num_clusters):
            idx_val_ind = random.sample(range(clusters_sizes_remain[i]), k=val_sizes[i])
            idx_val.extend((np.array(idx_remain)[labels[idx_remain]==i])[idx_val_ind])
        idx_train = list(set(idx_remain).difference(set(idx_val))) # the rest of the indices
        clusters_sizes_train = [int(sum(labels[idx_train]==i)) for i in range(num_clusters)]
        seed_sizes = [math.ceil(sum(labels[idx_train]==i)* args.seed_ratio) for i in range(num_clusters)]
        idx_seed = []
        if args.seed_ratio:
            for i in range(num_clusters):
                idx_seed_ind = random.sample(range(clusters_sizes_train[i]), k=seed_sizes[i])
                idx_seed.extend((np.array(idx_train)[labels[idx_train]==i])[idx_seed_ind])
        train_indices = idx_train
        val_indices = idx_val
        test_indices = idx_test
        seed_indices = idx_seed
        train_mask = np.zeros((labels.shape[0], 1), dtype=int)
        train_mask[train_indices, 0] = 1
        train_mask = np.squeeze(train_mask, 1)
        val_mask = np.zeros((labels.shape[0], 1), dtype=int)
        val_mask[val_indices, 0] = 1
        val_mask = np.squeeze(val_mask, 1)
        test_mask = np.zeros((labels.shape[0], 1), dtype=int)
        test_mask[test_indices, 0] = 1
        test_mask = np.squeeze(test_mask, 1)
        seed_mask = np.zeros((labels.shape[0], 1), dtype=int)
        seed_mask[seed_indices, 0] = 1
        seed_mask = np.squeeze(seed_mask, 1)
        mask = {}
        mask['train'] = train_mask
        mask['val'] = val_mask
        mask['test'] = test_mask
        mask['seed'] = seed_mask
        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()
        mask['seed'] = torch.from_numpy(mask['seed']).bool()
    
        masks['train'].append(mask['train'].unsqueeze(-1))
        masks['val'].append(mask['val'].unsqueeze(-1))
        masks['test'].append(mask['test'].unsqueeze(-1))
        masks['seed'].append(mask['seed'].unsqueeze(-1))
    
    label = label - np.amin(label)
    num_clusters = int(np.amax(label)+1)
    label = torch.from_numpy(label).long()

    feat_adj_reg = spectral_adjacency_reg(A_p,A_n, num_clusters)
    feat_L = signed_Laplacian_features(A_p, A_n, num_clusters)

    data = Data(y=label,A_p=A_p, A_n=A_n, feat_adj_reg=feat_adj_reg, feat_L=feat_L, 
    feat_given=feat_given, conflict_groups=conflict_groups)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)
    data.seed_mask  = torch.cat(masks['seed'], axis=-1)
    if not load_only:
        dir_name = os.path.dirname(save_path)
        if os.path.isdir(dir_name) == False:
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                print('Folder exists!')
        pk.dump(data, open(save_path, 'wb'))
    return data

def main():
    args = parameter_parser()
    rnd.seed(args.seed)
    random.seed(args.seed)
    if args.dataset[-1] != '/':
        args.dataset += '/'
    if args.dataset == 'SSBM/':
        (A_p, A_n), labels = SSBM(n=args.N, k=args.K, pin=args.p, pout=None, etain=args.eta, sizes='fix_ratio',size_ratio=args.size_ratio)
        (A_p, A_n), labels = fix_network(A_p,A_n,labels,eta=args.eta)
        conflict_groups = None
        default_values = [args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
    elif args.dataset == 'polarized/':
        (A_p, A_n), labels, conflict_groups = polarized_ssbm(total_n=args.total_n, num_com=args.num_com, N=args.N, K=args.K, p=args.p, eta=args.eta, size_ratio=args.size_ratio)
        default_values = [args.total_n, args.num_com, args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
    default_name_base = '_'.join([str(int(100*value)) for value in default_values])
    if args.seed != 31:
        default_name_base = 'Seed' + str(args.seed) + '_' + default_name_base
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/'+args.dataset+default_name_base+'.pk')
    _ = to_dataset(args,A_p,A_n, labels, save_path = save_path, conflict_groups=conflict_groups)
    return

if __name__ == "__main__":
    main()