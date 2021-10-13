# standard libaries
import os
import random
import csv
import math
import pickle as pk

# third-party libraries
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import WebKB, WikipediaNetwork
from torch_geometric.data import Data
import numpy.random as rnd
from sklearn.preprocessing import normalize, StandardScaler

# internel
from utils import polarized_ssbm, SSBM, fix_network, spectral_adjacency_reg, signed_Laplacian_features
from generate_data import to_dataset


def load_data_from_memory(root, name=None):
    data = pk.load(open(root, 'rb'))
    if os.path.isdir(root) == False:
        try:
            os.makedirs(root)
        except FileExistsError:
            pass
    return [data]


def load_sampson():
    A = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/Sampson/Sampson.npy'))
    N = A.shape[0]
    A_p = np.maximum(A,np.zeros((N,N)))
    A_n = np.maximum(-A,np.zeros((N,N)))
    A_p = sp.csr_matrix(A_p)
    A_n = sp.csr_matrix(A_n)
    features = np.array([[1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0]]).T
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    labels = np.array([0,0,0,0,0,1,1,3,0,2,2,2,1,0,4,2,4,2,1,4,1,1,1,3,3])
    return A_p, A_n, labels, features


def load_wikirfa():
    A_p = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/wikirfa/pruned_A_p.npz'))
    A_n = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/wikirfa/pruned_A_n.npz'))
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/wikirfa/pruned_labels_from_SPONGE3.npy'))
    return A_p, A_n, labels

def load_rainfall():
    A_p = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/rainfall/plus_cc.npz'))
    A_n = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/rainfall/minus_cc.npz'))
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/rainfall/labels6SPONGE.npy'))
    return A_p, A_n, labels

def load_corre_networks(dataset):
    A = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/corr_networks/adj_'+dataset+'.npz'))
    A_p = (abs(A) + A)/2
    A_n = (abs(A) - A)/2
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/corr_networks/labels_'+dataset+'.npy'))
    return A_p, A_n, labels



def load_PPI():
    A_p = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/PPI/adjacency_plus.npz'))
    A_n = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/PPI/adjacency_minus.npz'))
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)),'../data/PPI/labels10.npy'))
    return A_p, A_n, labels

def load_SP1500():
    A_p = sp.load_npz(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../data/SP1500/adjacency_plus_cc.npz'))
    A_n = sp.load_npz(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../data/SP1500/adjacency_minus_cc.npz'))
    labels = np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/SP1500/sector_labels.npy'))
    return A_p, A_n, labels


def load_data(args, load_only=False, random_seed=10):
    rnd.seed(random_seed)
    random.seed(random_seed)
    device = args.device
    if args.dataset[-1] != '/':
        args.dataset += '/'
    if args.dataset[:-1].lower() == 'ssbm':
            default_values = [args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
    elif args.dataset[:-1].lower() == 'polarized':
        default_values = [args.total_n, args.num_com, args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
    else:
        default_values = [args.K, args.seed_ratio,
                        args.train_ratio, args.test_ratio, args.num_trials]
    default_name_base = '_'.join([str(int(100*value)) for value in default_values])
    default_name_base = 'Seed' + str(random_seed) + '_' + default_name_base
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/'+args.dataset+default_name_base+'.pk')
    if (not args.regenerate_data) and os.path.exists(save_path):
        print('Loading existing data!')
        data = load_data_from_memory(save_path, name=None)[0]
    else:
        print('Generating new data or new data splits!')
        features_given = None
        conflict_groups = None
        if args.dataset[:-1].lower() == 'ssbm':
            (A_p, A_n), labels = SSBM(n=args.N, k=args.K, pin=args.p, pout=None, etain=args.eta, sizes='fix_ratio',size_ratio=args.size_ratio)
            (A_p, A_n), labels = fix_network(A_p,A_n,labels,eta=args.eta)
            default_values = [args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
        elif args.dataset[:-1].lower() == 'polarized':
            (A_p, A_n), labels, conflict_groups = polarized_ssbm(total_n=args.total_n, num_com=args.num_com, N=args.N, K=args.K, p=args.p, eta=args.eta, size_ratio=args.size_ratio)
        elif args.dataset[:-1].lower() ==  'sampson':
            A_p, A_n, labels, features_given = load_sampson()
        elif args.dataset[:-1].lower() ==  'sp1500':
            A_p, A_n, labels = load_SP1500()
        elif args.dataset[:-1].lower() ==  'wikirfa':
            A_p, A_n, labels = load_wikirfa()
        elif args.dataset[:-1].lower() ==  'ppi':
            A_p, A_n, labels = load_PPI()
        elif args.dataset[:-1].lower() ==  'rainfall':
            A_p, A_n, labels = load_rainfall()
        elif args.dataset[:9].lower() == 'mr_yearly':
            A_p, A_n, labels = load_corre_networks(args.dataset[:-1])
        else:
            raise NameError(
                'Please input the correct data set name ending with "/" instead of {}!'.format(args.dataset))
        data = to_dataset(args,A_p,A_n, labels, save_path = save_path, feat_given=features_given, 
        conflict_groups=conflict_groups, load_only=load_only)

    feat_adj_reg = torch.FloatTensor(data.feat_adj_reg).to(device)
    feat_L = torch.FloatTensor(data.feat_L).to(device)
    if data.feat_given is not None:
        feat_given = torch.FloatTensor(data.feat_given).to(device)
    else:
        feat_given = None

    comb = (feat_adj_reg, feat_L, feat_given, data.A_p, data.A_n)

    label = data.y.data.numpy().astype('int')
    train_mask = data.train_mask.data.numpy().astype('bool_')
    val_mask = data.val_mask.data.numpy().astype('bool_')
    test_mask = data.test_mask.data.numpy().astype('bool_')
    seed_mask = data.seed_mask.data.numpy().astype('bool_')

    return label, train_mask, val_mask, test_mask, seed_mask, comb
