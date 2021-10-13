import random
from typing import Union, Tuple, Optional

import torch
import scipy.sparse as sp
from texttable import Texttable
import latextable
import numpy as np
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from utils import scipy_sparse_to_torch_sparse

class Prob_Balanced_Ratio_Loss(torch.nn.Module):
    r"""An implementation of the probablistic balanced ratio cut loss function.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p, A_n):
        super(Prob_Balanced_Ratio_Loss, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)

    def forward(self, prob):
        """Making a forward pass of the probablistic balanced ratio cut loss function.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        mat = self.mat.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):       
            prob_vector_mat = prob[:, k, None]
            denominator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),prob_vector_mat) + 1)[0,0]    # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]

            result += numerator/denominator
        return result

class Prob_Balanced_Normalized_Loss(torch.nn.Module):
    r"""An implementation of the probablistic balanced normalized cut loss function.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p, A_n):
        super(Prob_Balanced_Normalized_Loss, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        D_n = sp.diags(A_n.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        self.D_bar = scipy_sparse_to_torch_sparse(D_p + D_n)
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)

    def forward(self, prob):
        """Making a forward pass of the probablistic balanced normalized cut loss function.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        epsilon = torch.FloatTensor([1e-6]).to(device)
        mat = self.mat.to(device)
        D_bar = self.D_bar.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):
            prob_vector_mat = prob[:, k, None]
            denominator = torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(D_bar,prob_vector_mat))[0,0] + epsilon    # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]

            result += numerator/denominator
        return result

def triplet_loss_InnerProduct(nclass, labels_split, labels_split_dif, logits,n_samples):
    n_sample = n_samples
   
    n_sample_class = max((int)(n_sample / nclass),32)
    thre = 0.1
    loss = 0
    for i in range(nclass):
        # python2: xrange, python3: range
        try:
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)
    
            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(labels_split_dif[i], k=n_sample_class)
    
            feats_dif = logits[randInds_dif]
        except IndexError:
            n_sample_class = 32
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)
    
            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(labels_split_dif[i], k=n_sample_class)
    
            feats_dif = logits[randInds_dif]
        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss

def triplet_loss_InnerProduct_alpha(nclass, labels_split, labels_split_dif, logits,n_samples, thre=0.1):
    n_sample = n_samples
   
    n_sample_class = max((int)(n_sample / nclass),32)
    # thre = 0.1
    loss = 0
    for i in range(nclass):
        # python2: xrange, python3: range
        try:
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)
    
            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(labels_split_dif[i], k=n_sample_class)
    
            feats_dif = logits[randInds_dif]
        except IndexError:
            n_sample_class = 32
            randInds1 = random.choices(labels_split[i], k=n_sample_class)
            randInds2 = random.choices(labels_split[i], k=n_sample_class)
    
            feats1 = logits[randInds1]
            feats2 = logits[randInds2]
            randInds_dif = random.choices(labels_split_dif[i], k=n_sample_class)
    
            feats_dif = logits[randInds_dif]
        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss

default_compare_names_all = ['A','L','L_sym','BNC','BRC','SPONGE','SPONGE_sym','SSSNET_L']
default_metric_names = ['BRC','BNC','unhappy ratio (\%)', 'size ratio', 'size std']


def print_performance_mean_std(dataset:str, results:np.array, compare_names_all:list=default_compare_names_all,
                               metric_names:list=default_metric_names, print_latex:bool=True, print_std:bool=False):
    r"""Prints performance table (and possibly with latex) with mean and standard deviations.
        The best two performing methods are highlighted in \red and \blue respectively.

    Args:
        dataset: (string) Name of the data set considered.
        results: (np.array) Results with shape (num_trials, num_methods, num_metrics).
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the methods are deemed better with smaller values).
        print_latex: (bool, optional) Whether to print latex table also. Default True.
        print_std: (bool, optinoal) Whether to print standard deviations or just mean. Default False.
    """
    t = Texttable(max_width=120)
    final_res_show = np.chararray(
        [len(metric_names)+1, len(compare_names_all)+1], itemsize=30)
    final_res_show[0, 0] = dataset+'Metric/Method'
    final_res_show[0, 1:] = compare_names_all
    final_res_show[1:, 0] = metric_names
    results_mean = np.transpose(np.around(results.mean(0), 2))
    final_res_show[1:, 1:] = results_mean
    if print_std:
        plus_minus = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        std = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        std[:] = np.transpose(np.around(results.std(0), 2))
        final_res_show[1:, 1:] = final_res_show[1:, 1:] + plus_minus + std
    else:
        plus_minus = np.chararray(
            [len(metric_names)-2, len(compare_names_all)], itemsize=20)
        plus_minus[:] = '$\pm$'
        std = np.chararray(
            [len(metric_names), len(compare_names_all)], itemsize=20)
        std[:] = np.transpose(np.around(results.std(0), 2))
        final_res_show[1:-2, 1:] = final_res_show[1:-2, 1:] + plus_minus + std[:-2]
    if len(compare_names_all)>1:
        red_start = np.chararray([1], itemsize=20)
        blue_start = np.chararray([1], itemsize=20)
        both_end = np.chararray([1], itemsize=20)
        red_start[:] = '\\red{'
        blue_start[:] = '\\blue{'
        both_end[:] = '}'
        for i in range(results_mean.shape[0]):
            if metric_names[i][-3:] in ['ARI', 'NMI']:
                best_values = -np.sort(-results_mean[i])[:2] # the bigger, the better
            else:
                best_values = np.sort(results_mean[i])[:2] # the smaller, the better
            final_res_show[i+1, 1:][results_mean[i]==best_values[0]] = red_start + final_res_show[i+1, 1:][results_mean[i]==best_values[0]] + both_end
            if best_values[0] != best_values[1]:
                final_res_show[i+1, 1:][results_mean[i]==best_values[1]] = blue_start + final_res_show[i+1, 1:][results_mean[i]==best_values[1]] + both_end

    t.add_rows(final_res_show)
    print(t.draw())
    if print_latex:
        print(latextable.draw_latex(t, caption=dataset +
                                    " performance.", label="table:"+dataset) + "\n")


def label_size_ratio(labels_distributions, return_std=False):
    num_trials, num_methods, _ = labels_distributions.shape
    size_ratio = np.zeros([num_trials, num_methods])
    if return_std:
        size_std = np.zeros([num_trials, num_methods])
    for i in range(num_trials):
        for j in range(num_methods):
            data = labels_distributions[i, j]
            data = data[data.nonzero()]
            size_ratio[i, j] = data.max()/data.min()
            if return_std:
                size_std[i, j] = np.nanstd(data)
    if return_std:
        return size_ratio, size_std
    else:
        return size_ratio

def get_cut_and_distribution(labels: Union[list, np.array, torch.LongTensor],
                                        num_clusters: int,
                                        A_p: sp.csr_matrix,
                                        A_n: sp.csr_matrix) -> Tuple[float, float, np.ndarray]:
    r"""Computes cut values and distribution of labels.

    Args:
        labels (list, np.array or torch.LongTensor): Predicted labels.
        num_clusters (int): Number of clusters.
        A_p (scipy sparse matrix): Positive part of the djacency matrix.
        A_n (scipy sparse matrix): Negative part of the djacency matrix.

    :rtype: 
        brc(float): Balanced ratio cut value.
        bnc (float): Balanced normalized cut value.
        labels_distribution(np.array): Array of distribution of labels.
    """
    P = torch.zeros(labels.shape[0], num_clusters)
    for k in range(num_clusters):
        P[labels == k, k] = 1
    labels_distribution = np.array(P.sum(0).numpy(), dtype=int)
    brc = Prob_Balanced_Ratio_Loss(A_p, A_n)(P).item()
    bnc = Prob_Balanced_Normalized_Loss(A_p, A_n)(P).item()
    unhappy_ratio = Unhappy_ratio(A_p, A_n)(P).item()
    return brc, bnc, unhappy_ratio, labels_distribution

def calculate_positive_embedding_loss(z, positive_edges):
    """
    Calculating the loss on the positive edge embedding distances
    :param z: Hidden vertex representation.
    :param positive_edges: Positive training edges.
    :return : Loss value on positive edge embedding.
    """
    i, j, k = structured_negative_sampling(positive_edges,z.shape[0])

    out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
    return torch.clamp(out, min=0).mean()

def calculate_negative_embedding_loss(z, negative_edges):
    """
    Calculating the loss on the negative edge embedding distances
    :param z: Hidden vertex representation.
    :param negative_edges: Negative training edges.
    :return : Loss value on negative edge embedding.
    """
    i, j, k = structured_negative_sampling(negative_edges,z.shape[0])

    out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
    return torch.clamp(out, min=0).mean()

def link_sign_loss_function(z, positive_edges, negative_edges):
    """
    Calculating the link sign loss.
    :param z: Hidden vertex representation.
    :param positive_edges: Positive training edges.
    :param negative_edges: Negative training edges.
    :return loss: Value of loss.
    """
    loss_term_1 = calculate_positive_embedding_loss(z, positive_edges)
    loss_term_2 = calculate_negative_embedding_loss(z, negative_edges)
    loss_term = loss_term_1+loss_term_2
    return loss_term

def real_data_analysis(dataset='wikirfa', result_folder_name='../result_arrays/wikirfa/',
                       name_base='100_200_50_0_0_1_3200_10_0_10_5000_L_SpectralSSSNET.npy',
                       new_name_base='100_200_50_0_0_1_3200_10_0_10_5000_L_SpectralSSSNET.npy',
                       pred_save_dir='../logs/wikirfa/200_0_80_10_1000/04-26-08:56:24/',
                       new_pred_save_dir='../logs/wikirfa/200_0_80_10_1000/04-26-08:56:24/',
                       A_p_path='../data/wikirfa/pruned_A_p.npz', A_n_path='../data/wikirfa/pruned_A_n.npz',
                       compare_names_all=default_compare_names_all, metric_names=default_metric_names):
    r"""Conducts real data analysis.

    Args:
        dataset: (string, optional) Name of the data set considered.
        result_folder_name: (str, optional) Directory to store ARI (wrt to labels), flow matrices, imbalance values.
        name_base: (str, optional) The invariant component in result array file names for spectral methods.
        new_name_base: (str, optional) The invariant component in result array file names for SSSNET.
        pred_save_dir: (str, optional) The folder to save spectral predictions.
        new_pred_save_dir: (str, optional) The folder to save SSSNET predictions.
        A_p_path: (str, optional) Path to the positive part of the adjacency matrix.
        A_n_path: (str, optional) Path to the negative part of the adjacency matrix.
        compare_names_all: (list of strings, optional) Methods names to compare.
        metric_names: (list of strings, optional) Metrics to use (the methods are deemed better with smaller values).
    """
    cut = np.load(result_folder_name+'balanced_cuts'+name_base)[:,:len(compare_names_all)-1, :2]
    dist = np.load(result_folder_name+'labels_distribution'+name_base)[:,:len(compare_names_all)-1]
    cut = np.concatenate((cut, np.load(result_folder_name+'balanced_cuts'+new_name_base)[:,-1:,:2]), axis=1)
    dist = np.concatenate((dist, np.load(result_folder_name+'labels_distribution'+new_name_base)[:,-1:]), axis=1)

    size_ratio, size_std = label_size_ratio(dist, True)
    num_trials = cut.shape[0]
    num_clusters = dist.shape[-1]
    try:
        unhappy_ratios = np.load(result_folder_name+'unhappy_ratios.npy')
    except FileNotFoundError:
        unhappy_ratios = np.zeros((num_trials, len(compare_names_all)))
        A_p = sp.load_npz(A_p_path)
        A_n = sp.load_npz(A_n_path)
        unhappy_ratio_func = Unhappy_ratio(A_p, A_n)
        if dataset[-4:] == 'semi':
            extra_str = '_latest'
        else:
            extra_str = ''
        for i in range(num_trials):
            spectral_pred = np.load(pred_save_dir+'spectral_pred'+str(i)+'.npy')
            for j in range(spectral_pred.shape[0]):
                pred = spectral_pred[j]
                P = torch.zeros(pred.shape[0], num_clusters)
                for k in range(num_clusters):
                    P[pred == k, k] = 1
                unhappy_ratios[i, j] =  unhappy_ratio_func(P)
            pred = np.load(new_pred_save_dir+'SSSNET_L_pred'+extra_str+str(i)+'.npy')
            P = torch.zeros(pred.shape[0], num_clusters)
            for k in range(num_clusters):
                P[pred == k, k] = 1
            unhappy_ratios[i, -1] =  unhappy_ratio_func(P)
            pairwise_ARI(dataset+str(i), np.concatenate((spectral_pred, pred[None, :]), axis=0), compare_names_all)

    print_performance_mean_std(dataset, np.concatenate(
        (cut, 100*unhappy_ratios[:, :, None], size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names)

class Unhappy_ratio(torch.nn.Module):
    r"""A calculation of the ratio of unhappy edges among all edges.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p, A_n):
        super(Unhappy_ratio, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)
        self.num_edges = len((A_p - A_n).nonzero()[0])

    def forward(self, prob):
        """Making a forward pass of the calculation of the ratio of unhappy edges among all edges.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        mat = self.mat.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):
            prob_vector_mat = prob[:, k, None]
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]
            result += numerator
        return result/self.num_edges

def pairwise_ARI(dataset, pred_all, compare_names_all=default_compare_names_all):
    method_num = len(compare_names_all)
    pairwise_ARI_mat = np.ones((method_num, method_num))
    for i in range(method_num-1):
        for j in range(i+1, method_num):
            pairwise_ARI_mat[i,j] = adjusted_rand_score(pred_all[i], pred_all[j])
            pairwise_ARI_mat[j,i] = pairwise_ARI_mat[i,j]

    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111)
    cax = ax.matshow(pairwise_ARI_mat, interpolation='nearest')
    cb = fig.colorbar(cax)
    cb.ax.tick_params(labelsize=26) 

    ax.set_xticklabels(['']+compare_names_all,rotation=45,fontsize=20)
    ax.set_yticklabels(['']+compare_names_all,fontsize=20)

    # ax.set_title('Pairwise ARI for Wiki-Rfa')
    # plt.savefig('Pairwise ARI for Wiki-Rfa.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.savefig('../comparison_plots/real_plots/Pairwise_ARI_{}.pdf'.format(dataset))
    ax.set_title('Pairwise ARI for {}'.format(dataset))
    plt.show()