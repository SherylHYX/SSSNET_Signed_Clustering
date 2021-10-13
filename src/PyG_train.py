from __future__ import division
from __future__ import print_function

import os
import math
import random
import time
import argparse
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from texttable import Texttable
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import get_powers_sparse, scipy_sparse_to_torch_sparse
from utils import  split_labels, getClassMean, write_log, extract_edges
from metrics import triplet_loss_InnerProduct_alpha, Prob_Balanced_Ratio_Loss, Prob_Balanced_Normalized_Loss
from metrics import label_size_ratio, print_performance_mean_std, get_cut_and_distribution, link_sign_loss_function
from PyG_models import SSSNET
from cluster import Cluster 
from preprocess import load_data
from param_parser import parameter_parser

args = parameter_parser()
torch.manual_seed(args.seed)
device = args.device
if args.dataset[-1] != '/':
    args.dataset += '/'
if args.cuda:
    torch.cuda.manual_seed(args.seed)
no_magnet = True
compare_names = []
if 'spectral' in args.all_methods:
    compare_names = ['A','sns','dns','L','L_sym','BNC','BRC','SPONGE','SPONGE_sym']
num_gnn = 0
if 'SSSNET' in args.all_methods:
    num_gnn += 1
    compare_names.append('SSSNET')
    compare_names_all = []
    compare_names_all.extend(compare_names[:-1])
    for feat_opt in args.feature_options:
        compare_names_all.append(
                compare_names[-1]+'_'+feat_opt)
else:
    compare_names_all = compare_names


class SSSNET_Trainer(object):
    """
    Object to train and score different models.
    """

    def __init__(self, args, random_seed):
        """
        Constructing the trainer instance.
        :param args: Arguments object.
        """
        self.args = args
        self.device = args.device

        label, self.train_mask, self.val_mask, self.test_mask, self.seed_mask, comb = load_data(args, args.load_only, random_seed)

        # normalize label, the minimum should be 0 as class index
        _label_ = label - np.amin(label)
        self.label = torch.from_numpy(_label_[np.newaxis]).to(device)
        self.cluster_dim = np.amax(_label_)+1
        self.num_clusters = self.cluster_dim

        self.feat_adj_reg, self.feat_L, self.feat_given, self.A_p_scipy, self.A_n_scipy = comb
        self.edge_index_p = torch.LongTensor(self.A_p_scipy.nonzero()).to(self.args.device)
        self.edge_weight_p = torch.FloatTensor(sp.csr_matrix(self.A_p_scipy).data).to(self.args.device)
        self.edge_index_n = torch.LongTensor(self.A_n_scipy.nonzero()).to(self.args.device)
        self.edge_weight_n = torch.FloatTensor(sp.csr_matrix(self.A_n_scipy).data).to(self.args.device)
        self.A_p = get_powers_sparse(self.A_p_scipy, hop=1, tau=self.args.tau)[1].to(self.args.device)
        self.A_n = get_powers_sparse(self.A_n_scipy, hop=1, tau=0)[1].to(self.args.device)
        self.A_pt = get_powers_sparse(self.A_p_scipy.transpose(), hop=1, tau=self.args.tau)[1].to(self.args.device)
        self.A_nt = get_powers_sparse(self.A_n_scipy.transpose(), hop=1, tau=0)[1].to(self.args.device)
        if self.args.dense:
            self.A_p = self.A_p.to_dense()
            self.A_n = self.A_n.to_dense()
            self.A_pt = self.A_pt.to_dense()
            self.A_nt = self.A_nt.to_dense()
        self.c = Cluster((0.5*(self.A_p_scipy+self.A_p_scipy.transpose()), 
        0.5*(self.A_n_scipy+self.A_n_scipy.transpose()), int(self.num_clusters)))

        date_time = datetime.now().strftime('%m-%d-%H:%M:%S')

        if args.dataset[:-1].lower() == 'ssbm':
            default_values = [args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
        elif args.dataset[:-1].lower() == 'polarized':
            default_values = [args.total_n, args.num_com, args.p, args.K,args.N,args.seed_ratio, args.train_ratio, args.test_ratio,args.size_ratio,args.eta, args.num_trials]
        else:
            default_values = [args.K, args.seed_ratio,
                            args.train_ratio, args.test_ratio, args.num_trials]
        save_name = '_'.join([str(int(100*value)) for value in default_values])
        save_name += 'Seed' + str(random_seed)
        self.log_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), args.log_root, args.dataset[:-1], save_name, date_time)

        if os.path.isdir(self.log_path) == False:
            try:
                os.makedirs(self.log_path)
            except FileExistsError:
                print('Folder exists!')

        self.splits = self.train_mask.shape[1]
        if len(self.test_mask.shape) == 1:
            #data.test_mask = test_mask.unsqueeze(1).repeat(1, splits)
            self.test_mask = np.repeat(
                self.test_mask[:, np.newaxis], self.splits, 1)
        write_log(vars(args), self.log_path)  # write the setting

    def SSSNET(self, feat_choice):
        #################################
        # SSSNET
        #################################
        if feat_choice == 'A_reg':
            self.features = self.feat_adj_reg
        elif feat_choice == 'L':
            self.features = self.feat_L
        elif feat_choice == 'given':
            self.features = self.feat_given
        elif feat_choice == 'None':
            self.features = torch.eye(self.A_p_scipy.shape[0]).to(self.args.device)
        res_full = np.zeros([self.splits, 1])
        res_full_latest = np.zeros([self.splits, 1])
        res_all_full = np.zeros([self.splits, 1])
        res_all_full_latest = np.zeros([self.splits, 1])
        NMI_full = np.zeros([self.splits, 1])
        NMI_full_latest = np.zeros([self.splits, 1])
        NMI_all_full = np.zeros([self.splits, 1])
        NMI_all_full_latest = np.zeros([self.splits, 1])
        balanced_cuts_full = np.zeros([self.splits, 1, 3])
        balanced_cuts_full_latest = balanced_cuts_full.copy()
        labels_distribution_full = np.zeros(
            [self.splits, 1, self.num_clusters])
        labels_distribution_full_latest = labels_distribution_full.copy()
        args = self.args
        labels = self.label.view(-1)
        for split in range(self.splits):
            graphmodel = SSSNET(nfeat=self.features.shape[1],
                                hidden=self.args.hidden,
                                nclass=self.num_clusters,
                                dropout=self.args.dropout,
                                hop=self.args.hop,
                                fill_value=self.args.tau,
                                directed=self.args.directed).to(self.args.device)

            model = graphmodel
            opt = optim.Adam(model.parameters(), lr=self.args.lr,
                             weight_decay=self.args.weight_decay)

            train_index = self.train_mask[:, split]
            val_index = self.val_mask[:, split]
            test_index = self.test_mask[:, split]
            if args.AllTrain:
                # to use all nodes for fair comparison with spectral methods
                train_index[:] = True
                # to use all nodes for fair comparison with spectral methods
                val_index[:] = True
            if self.args.seed_ratio:
                seed_index = self.seed_mask[:, split]
                nclass, labels_split, _, labels_split_dif = split_labels(
                    self.label.view(-1)[seed_index])

            #################################
            # Train/Validation/Test
            #################################
            best_val_loss = 1000.0
            early_stopping = 0
            log_str_full = ''
            
            loss_func_pbrc_train = Prob_Balanced_Ratio_Loss(A_p=self.A_p_scipy[train_index][:,train_index], A_n=self.A_n_scipy[train_index][:,train_index])
            loss_func_pbnc_train = Prob_Balanced_Normalized_Loss(A_p=self.A_p_scipy[train_index][:,train_index], A_n=self.A_n_scipy[train_index][:,train_index])
            if not self.args.no_validation:
                loss_func_pbrc_val = Prob_Balanced_Ratio_Loss(A_p=self.A_p_scipy[val_index][:,val_index], A_n=self.A_n_scipy[val_index][:,val_index])
                loss_func_pbnc_val = Prob_Balanced_Normalized_Loss(A_p=self.A_p_scipy[val_index][:,val_index], A_n=self.A_n_scipy[val_index][:,val_index])
            if self.args.link_sign_loss:
                positive_edges_train = extract_edges(self.A_p_scipy.nonzero(), np.arange(self.A_p_scipy.shape[0])[train_index]).to(self.args.device)
                negative_edges_train = extract_edges(self.A_n_scipy.nonzero(), np.arange(self.A_n_scipy.shape[0])[train_index]).to(self.args.device)
                if not self.args.no_validation:
                    positive_edges_val = extract_edges(self.A_p_scipy.nonzero(), np.arange(self.A_p_scipy.shape[0])[val_index]).to(self.args.device)
                    negative_edges_val = extract_edges(self.A_n_scipy.nonzero(), np.arange(self.A_n_scipy.shape[0])[val_index]).to(self.args.device)
            for epoch in range(args.epochs):
                start_time = time.time()
                ####################
                # Train
                ####################
                train_loss, train_ARI = 0.0, 0.0

                model.train()
                logits, output, pred_label, prob = model(self.edge_index_p, self.edge_weight_p,
                self.edge_index_n, self.edge_weight_n, self.features) 
                if epoch == 0:
                    log_str_full += '\n' + f'cuda memory allocated, {torch.cuda.memory_allocated()/1024/1024}, reserved, {torch.cuda.memory_reserved()/1024/1024}'
                loss_pbrc = loss_func_pbrc_train(prob=prob[train_index])
                loss_pbnc = loss_func_pbnc_train(prob=prob[train_index])
                if not self.args.no_self_supervised:
                    train_loss = args.w_pbnc * loss_pbnc + args.w_pbrc * loss_pbrc
                else:
                    train_loss = torch.zeros(1).to(self.args.device)
                if self.args.link_sign_loss:
                    loss_link_sign = self.args.link_sign_loss_ratio*link_sign_loss_function(logits, positive_edges_train, negative_edges_train)
                    train_loss += loss_link_sign
                if self.args.seed_ratio:
                    loss_ce = F.nll_loss(
                        output[seed_index], labels[seed_index])
                    loss_triplet = triplet_loss_InnerProduct_alpha(
                        nclass.item(), labels_split, labels_split_dif, logits, args.samples, thre=self.args.alpha)
                    train_loss += self.args.supervised_loss_ratio*(loss_ce +
                                    self.args.triplet_loss_ratio*loss_triplet)
                opt.zero_grad()
                train_loss.backward()
                opt.step()

                train_ARI = adjusted_rand_score(
                    pred_label.view(-1)[train_index].to('cpu'), self.label.view(-1).to('cpu')[train_index])

                outstrtrain = 'Train loss:, %.6f, ARI: ,%.3f,' % (
                    train_loss.detach().item(), train_ARI)
                if self.args.link_sign_loss:
                    outstrtrain += 'link sign loss:, {:.3f},'.format(loss_link_sign.item())
                if self.args.seed_ratio:
                    outstrtrain += 'ce loss:, {:.3f}, triplet loss:,{:.3f},'.format(loss_ce.item(), loss_triplet.item())
                # scheduler.step()
                if not self.args.no_validation:
                    ####################
                    # Validation
                    ####################
                    model.eval()
                    val_loss, val_ARI = 0.0, 0.0

                    logits, output, pred_label, prob = model(self.edge_index_p, self.edge_weight_p,
                    self.edge_index_n, self.edge_weight_n, self.features) 
                    loss_pbrc = loss_func_pbrc_val(prob=prob[val_index])
                    loss_pbnc = loss_func_pbnc_val(prob=prob[val_index])
                    val_loss = args.w_pbnc * loss_pbnc + args.w_pbrc * loss_pbrc
                    if self.args.link_sign_loss:
                        loss_link_sign = self.args.link_sign_loss_ratio*link_sign_loss_function(logits, positive_edges_val, negative_edges_val)
                        val_loss += loss_link_sign

                    val_ARI = adjusted_rand_score(
                        pred_label.view(-1).to('cpu')[val_index], self.label.view(-1).to('cpu')[val_index])

                    outstrval = ' Validation loss:, %.6f, ARI:, %.3f,' % (
                        val_loss.detach().item(), val_ARI)
                    if self.args.link_sign_loss:
                        outstrval += 'link sign loss:, {:.3f},'.format(loss_link_sign.item())


                    ####################
                    # Save weights
                    ####################
                    save_perform = val_loss.detach().item()
                    if save_perform <= best_val_loss:
                        early_stopping = 0
                        best_val_loss = save_perform
                        torch.save(model.state_dict(), self.log_path +
                                '/PyG_SSSNET_'+feat_choice+'_model'+str(split)+'.t7')
                    else:
                        early_stopping += 1
                duration = "---, %.4f, seconds ---" % (time.time() - start_time)
                if not self.args.no_validation:
                    log_str = ("%d, / %d epoch," % (epoch, args.epochs)) + \
                        outstrtrain+outstrval+duration
                else:
                    log_str = ("%d, / %d epoch," % (epoch, args.epochs)) + \
                        outstrtrain+duration
                log_str_full += log_str + '\n'
                print(log_str)

                if early_stopping > args.early_stopping or epoch == (args.epochs-1):
                    torch.save(model.state_dict(), self.log_path +
                               '/PyG_SSSNET_'+feat_choice+'_model_latest'+str(split)+'.t7')
                    break

            status = 'w'
            if os.path.isfile(self.log_path + '/PyG_SSSNET_'+feat_choice+'_log'+str(split)+'.csv'):
                status = 'a'
            with open(self.log_path + '/PyG_SSSNET_'+feat_choice+'_log'+str(split)+'.csv', status) as file:
                file.write(log_str_full)
                file.write('\n')
                status = 'a'

            ####################
            # Testing
            ####################
            logstr = ''
            if not self.args.no_validation:
                model.load_state_dict(torch.load(
                    self.log_path + '/PyG_SSSNET_'+feat_choice+'_model'+str(split)+'.t7'))
                model.eval()
                _, _, pred_label, prob = model(self.edge_index_p, self.edge_weight_p,
                self.edge_index_n, self.edge_weight_n, self.features) 
                if self.args.SavePred:
                    np.save(self.log_path + '/PyG_SSSNET_'+feat_choice +
                            '_pred'+str(split), pred_label.view(-1).to('cpu'))

                val_ARI = adjusted_rand_score(
                    pred_label.view(-1).to('cpu')[val_index], self.label.view(-1).to('cpu')[val_index])

                test_ARI = adjusted_rand_score(
                    pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])

                all_ARI = adjusted_rand_score(
                    pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))

                test_NMI = normalized_mutual_info_score(
                    pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])

                all_NMI = normalized_mutual_info_score(
                    pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))

                brc, bnc, unhappy_ratio, labels_distribution = get_cut_and_distribution(pred_label.view(-1).to('cpu'),
                                                                                        self.num_clusters, self.A_p_scipy, self.A_n_scipy)
                balanced_cuts_full[split, 0] = [brc, bnc, unhappy_ratio]
                labels_distribution_full[split, 0] = labels_distribution
                logstr = 'best brc = {:.3f}, bnc = {:.6f}'.format(brc, bnc)+'\n'
                logstr += 'labels distribution is,{}'.format(
                    labels_distribution)+'\n'

            model.load_state_dict(torch.load(
                self.log_path + '/PyG_SSSNET_'+feat_choice+'_model_latest'+str(split)+'.t7'))
            model.eval()
            _, _, pred_label, prob = model(self.edge_index_p, self.edge_weight_p,
            self.edge_index_n, self.edge_weight_n, self.features) 
            if self.args.SavePred:
                np.save(self.log_path + '/PyG_SSSNET_'+feat_choice +
                        '_pred_latest'+str(split), pred_label.view(-1).to('cpu'))

            val_ARI_latest = adjusted_rand_score(
                pred_label.view(-1).to('cpu')[val_index], self.label.view(-1).to('cpu')[val_index])

            test_ARI_latest = adjusted_rand_score(
                pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])

            all_ARI_latest = adjusted_rand_score(
                pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))

            test_NMI_latest = normalized_mutual_info_score(
                pred_label.view(-1).to('cpu')[test_index], self.label.view(-1).to('cpu')[test_index])

            all_NMI_latest = normalized_mutual_info_score(
                pred_label.view(-1).to('cpu'), self.label.view(-1).to('cpu'))

            brc, bnc, unhappy_ratio, labels_distribution = get_cut_and_distribution(pred_label.view(-1).to('cpu'),
                                                                                    self.num_clusters, self.A_p_scipy, self.A_n_scipy)
            balanced_cuts_full_latest[split, 0] = [brc, bnc, unhappy_ratio]
            labels_distribution_full_latest[split, 0] = labels_distribution
            logstr += 'latest brc = {:.3f}, bnc = {:.6f}'.format(brc, bnc)+'\n'
            logstr += 'labels distribution is,{}'.format(
                labels_distribution)+'\n'

            ####################
            # Save testing results
            ####################
            if not self.args.no_validation:
                logstr += 'val_ARI:, '+str(np.round(val_ARI, 3))+' ,test_ARI: ,'+str(np.round(test_ARI, 3))+' ,val_ARI_latest: ,'+str(
                    np.round(val_ARI_latest, 3))+' ,test_ARI_latest: ,'+str(np.round(test_ARI_latest, 3))
                logstr += ' ,all_ARI: ,' + \
                    str(np.round(all_ARI, 3))+', all_ARI_latest: ,' + \
                    str(np.round(all_ARI_latest, 3))
            else:
                logstr += 'test_ARI_latest: ,'+str(np.round(test_ARI_latest, 3))
                logstr += ' ,all_ARI_latest: ,' + str(np.round(all_ARI_latest, 3))
            logstr += '\n' + f'cuda memory allocated, {torch.cuda.memory_allocated()/1024/1024}, reserved, {torch.cuda.memory_reserved()/1024/1024}'
            print(feat_choice)
            print(logstr)
            if not self.args.no_validation:
                res_full[split], res_all_full[split] = test_ARI, all_ARI
                NMI_full[split], NMI_all_full[split] = test_NMI, all_NMI
            res_full_latest[split], res_all_full_latest[split] = test_ARI_latest, all_ARI_latest
            NMI_full_latest[split], NMI_all_full_latest[split] = test_NMI_latest, all_NMI_latest
            with open(self.log_path + '/PyG_SSSNET_'+feat_choice+'_log'+str(split)+'.csv', status) as file:
                file.write(logstr)
                file.write('\n')
            torch.cuda.empty_cache()
        if not self.args.no_validation:
            results = (res_full, res_all_full,
                    res_full_latest, res_all_full_latest)
            NMIs = (NMI_full, NMI_all_full,
                    NMI_full_latest, NMI_all_full_latest)
            cuts = (balanced_cuts_full, balanced_cuts_full_latest)
            labels_distributions = (labels_distribution_full,
                                    labels_distribution_full_latest)
        else:
            results = (res_full_latest, res_all_full_latest)
            NMIs = (NMI_full_latest, NMI_all_full_latest)
            cuts = (balanced_cuts_full_latest)
            labels_distributions = (labels_distribution_full_latest)
        return results, NMIs, cuts, labels_distributions

    def gen_results_spectral(self):
        num_trials = self.splits
        num_clusters = self.num_clusters
        res_full = np.zeros([num_trials, len(compare_names)-num_gnn])
        res_all_full = np.zeros([num_trials, len(compare_names)-num_gnn])
        NMI_full = np.zeros([num_trials, len(compare_names)-num_gnn])
        NMI_all_full = np.zeros([num_trials, len(compare_names)-num_gnn])
        balanced_cuts_full = np.zeros([self.splits, len(compare_names)-num_gnn, 3])
        labels_distribution_full = np.zeros(
            [self.splits, len(compare_names)-num_gnn, self.num_clusters])
        for split in range(self.splits):
            if self.args.SavePred:
                pred_all = np.zeros(
                    [len(compare_names)-num_gnn, len(self.label.view(-1).to('cpu'))])
            test_index = self.test_mask[:, split]
            res = []
            res_all = []
            NMI_test = []
            NMI_all = []
            labels_test_cpu = (self.label.view(-1)[test_index]).to('cpu')
            labels_cpu = self.label.view(-1).to('cpu')
            idx_test_cpu = test_index
            # now append results for comparison methods
            for i, pred in enumerate([self.c.spectral_cluster_adjacency_reg(k=num_clusters, normalisation='none'),
                                      self.c.spectral_cluster_sns(k=num_clusters), self.c.spectral_cluster_dns(k=num_clusters),
                                      self.c.spectral_cluster_laplacian(k=num_clusters, normalisation='none'),
                                      self.c.spectral_cluster_laplacian(k=num_clusters, normalisation='sym'),
                                      self.c.spectral_cluster_bnc(k=num_clusters, normalisation='sym'),
                                      self.c.spectral_cluster_bnc(k=num_clusters, normalisation='none'),
                                      self.c.SPONGE(k=num_clusters), self.c.SPONGE_sym(k=num_clusters)]):
                brc, bnc, unhappy_ratio, labels_distribution = get_cut_and_distribution(pred,
                                                                        self.num_clusters, self.A_p_scipy, self.A_n_scipy)
                balanced_cuts_full[split, i] = [brc, bnc, unhappy_ratio]
                labels_distribution_full[split, i] = labels_distribution
                res.append(adjusted_rand_score(
                    pred[idx_test_cpu], labels_test_cpu))
                res_all.append(adjusted_rand_score(pred, labels_cpu))
                NMI_test.append(normalized_mutual_info_score(
                    pred[idx_test_cpu], labels_test_cpu))
                NMI_all.append(normalized_mutual_info_score(pred, labels_cpu))
                if self.args.SavePred:
                    pred_all[i] = pred
            if self.args.SavePred:
                np.save(self.log_path + '/spectral_pred'+str(split), pred_all)
            res_full[split] = res
            res_all_full[split] = res_all
            NMI_full[split] = NMI_test
            NMI_all_full[split] = NMI_all
        print('Test ARI for methods to compare:{}'.format(res_full))
        print('All data ARI for methods to compare:{}'.format(res_all_full))
        return res_full, res_all_full, NMI_full, NMI_all_full, balanced_cuts_full, labels_distribution_full


# train and grap results
if args.dataset[-1] != '/':
    args.dataeset += '/'
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../PyG_result_arrays/'+args.dataset)
if os.path.isdir(dir_name) == False:
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        print('Folder exists!')


label, _, _, _, _, comb = load_data(args, args.load_only, args.seeds[0])
num_clusters = int(np.max(label) - np.min(label) + 1)

final_res = np.zeros([args.num_trials*len(args.seeds), len(compare_names_all)])
final_res_all = np.zeros([args.num_trials*len(args.seeds), len(compare_names_all)])
final_nmi = np.zeros([args.num_trials*len(args.seeds), len(compare_names_all)])
final_nmi_all = np.zeros([args.num_trials*len(args.seeds), len(compare_names_all)])
balanced_cuts_full = np.zeros([args.num_trials*len(args.seeds), len(compare_names_all), 3])
labels_distribution_full = np.zeros(
    [args.num_trials*len(args.seeds), len(compare_names_all), num_clusters])
method_str = ''


final_res_latest = final_res.copy()
final_res_all_latest = final_res_all.copy()
final_nmi_latest = final_nmi.copy()
final_nmi_all_latest = final_nmi_all.copy()
balanced_cuts_full_latest = balanced_cuts_full.copy()
labels_distribution_full_latest = labels_distribution_full.copy()


if 'spectral' in args.all_methods:
    method_str += 'Spectral'
if 'SSSNET' in args.all_methods:
    method_str += 'SSSNET'

current_seed_ind = 0
for random_seed in args.seeds:
    trainer = SSSNET_Trainer(args, random_seed)
    if 'spectral' in args.all_methods:
        # spetral methods
        final_res[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn], \
            final_res_all[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn], \
                final_nmi[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn], \
                    final_nmi_all[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn], \
                        balanced_cuts_full[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names) - num_gnn], \
                            labels_distribution_full[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = trainer.gen_results_spectral()
        final_res_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = final_res[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn]
        final_res_all_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = final_res_all[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn]
        final_nmi_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = final_nmi[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn]
        final_nmi_all_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = final_nmi_all[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn]
        balanced_cuts_full_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names) - num_gnn] = balanced_cuts_full[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names) - num_gnn]
        labels_distribution_full_latest[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn] = labels_distribution_full[current_seed_ind:current_seed_ind+args.num_trials, :len(compare_names)-num_gnn]
    if 'SSSNET' in args.all_methods:
        current_ind = len(compare_names)-num_gnn
        # SSSNET
        for feat_choice in args.feature_options:    
            results, NMIs, cuts, labels_distributions = trainer.SSSNET(
                feat_choice)
            if not args.no_validation:
                final_res[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                    final_res_all[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                        final_res_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                            final_res_all_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = results
                final_nmi[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                    final_nmi_all[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                        final_nmi_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                            final_nmi_all_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = NMIs
                balanced_cuts_full[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                    balanced_cuts_full_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = cuts
                labels_distribution_full[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind + 1], \
                    labels_distribution_full_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = labels_distributions
            else:
                final_res_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                     final_res_all_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = results
                final_nmi_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1], \
                     final_nmi_all_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = NMIs
                balanced_cuts_full_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = cuts
                labels_distribution_full_latest[current_seed_ind:current_seed_ind+args.num_trials, current_ind:current_ind+1] = labels_distributions
            current_ind = current_ind + 1
    current_seed_ind += args.num_trials
    

# print results and save results to arrays
feat_choices = '_'.join(args.feature_options)
t = Texttable(max_width=120)
if args.dataset[:-1].lower() == 'ssbm':
    param_values = [args.p,args.eta,args.K,args.N,args.hop,args.tau, args.size_ratio, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.link_sign_loss, args.link_sign_loss_ratio, args.supervised_loss_ratio]
    t.add_rows([["Parameter","p","eta","K","N","hop","tau","size ratio",
    "seed ratio", "alpha", "lr", "hidden","triplet loss ratio","link sign loss","link sign loss ratio","supervised loss ratio","features"],
    ["Values",args.p,args.eta,args.K,args.N,args.hop,args.tau, args.size_ratio, 
    args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.link_sign_loss, 
    args.link_sign_loss_ratio, args.supervised_loss_ratio, feat_choices]])
elif args.dataset[:-1].lower() == 'polarized':
    param_values = [args.total_n, args.num_com, args.p,args.eta,args.K,args.N,
    args.hop,args.tau, args.size_ratio, args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, 
    args.link_sign_loss, args.link_sign_loss_ratio, args.supervised_loss_ratio]
    t.add_rows([["Parameter","total n","num_com","p","eta","K","N","hop","tau",
    "size ratio","seed ratio", "alpha", "lr", "hidden","triplet loss ratio","link sign loss","link sign loss ratio","supervised loss ratio","features"],
    ["Values",args.total_n, args.num_com,args.p,args.eta,args.K,args.N,args.hop,
    args.tau, args.size_ratio, args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, 
    args.link_sign_loss, args.link_sign_loss_ratio, args.supervised_loss_ratio, feat_choices]])
else:
    param_values = [args.AllTrain, args.hop,args.tau,args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.link_sign_loss, args.link_sign_loss_ratio, args.supervised_loss_ratio]
    t.add_rows([["Parameter","AllTrain","hop","tau","seed ratio", "alpha", "lr", "hidden","triplet loss ratio","link sign loss","link sign loss ratio","supervised loss ratio","features"],
    ["Values",args.AllTrain,args.hop,args.tau, args.seed_ratio, args.alpha, args.lr, args.hidden, args.triplet_loss_ratio, args.link_sign_loss, args.link_sign_loss_ratio, args.supervised_loss_ratio, feat_choices]])

save_name = '_'.join([str(int(100*value)) for value in param_values]) + '_' + feat_choices + '_' + method_str
save_name += 'seeds' + '_'.join([str(value) for value in np.array(args.seeds).flatten()])
print(t.draw())
if args.no_self_supervised:
    save_name += 'no_self_supervised'

metric_names = ['test ARI', 'all ARI','test NMI','all NMI','BRC', 'BNC', 'unhappy ratio','size ratio', 'size std']
if not args.no_validation:
    np.save(dir_name+'test'+save_name, final_res)
    np.save(dir_name+'all'+save_name, final_res_all)
    np.save(dir_name+'test_NMI'+save_name, final_nmi)
    np.save(dir_name+'all_NMI'+save_name, final_nmi_all)
    np.save(dir_name+'balanced_cuts'+save_name, balanced_cuts_full)
    np.save(dir_name+'labels_distribution'+save_name, labels_distribution_full)
    size_ratio, size_std = label_size_ratio(labels_distribution_full, True)
    print_performance_mean_std(args.dataset[:-1]+'_best', np.concatenate((final_res[:, :, None], final_res_all[:, :, None], 
    final_nmi[:, :, None], final_nmi_all[:, :, None], balanced_cuts_full, 
    size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names, False)

size_ratio, size_std = label_size_ratio(labels_distribution_full_latest, True)


np.save(dir_name+'test_latest'+save_name, final_res_latest)
np.save(dir_name+'all_latest'+save_name, final_res_all_latest)
np.save(dir_name+'test_NMI_latest'+save_name, final_nmi_latest)
np.save(dir_name+'all_NMI_latest'+save_name, final_nmi_all_latest)
np.save(dir_name+'balanced_cuts_latest'+save_name, balanced_cuts_full_latest)
np.save(dir_name+'labels_distribution_latest' +
        save_name, labels_distribution_full_latest)    

print_performance_mean_std(args.dataset[:-1]+'_latest', np.concatenate((final_res_latest[:, :, None], final_res_all_latest[:, :, None], 
final_nmi_latest[:, :, None], final_nmi_all_latest[:, :, None], 
balanced_cuts_full_latest, size_ratio[:, :, None], size_std[:, :, None]), 2), compare_names_all, metric_names, False)