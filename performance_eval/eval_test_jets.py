import time

import torch
import uproot
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
import mpmath

from dataloaders.jets_loader import JetGraphDataset


def _get_rand_index(labels, predictions):
    n_items = len(labels)
    if (n_items < 2):
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs += 1

    return correct_pairs / n_pairs


def _error_count(labels, predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true):
                true_positives += 1
            if (not label_true) and pred_true:
                false_positive += 1
            if label_true and (not pred_true):
                false_negative += 1
    return true_positives, false_positive, false_negative


def _get_recall(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_negative == 0:
        return 0

    return true_positives / (true_positives + false_negative)


def _get_precision(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)


def _f_measure(labels, predictions):
    precision = _get_precision(labels, predictions)
    recall = _get_recall(labels, predictions)

    if precision == 0 or recall == 0:
        return 0

    return 2 * (precision * recall) / (recall + precision)


def eval_jets_on_test_set(model, sleeptime):
    
    print('Predicting to test on test set...')
    pred = _predict_on_test_set(model, sleeptime)
    print('Loading test set...')
    test_ds = uproot.open('data/test/test_data.root')
    print("Loading dataset...")
    jet_np_array = test_ds['tree'].arrays(library='np')
    print("Extracting jet_flav...")
    jet_flav = jet_np_array['jet_flav']
    print("Extracting trk_vtx_index...")
    target = jet_np_array['trk_vtx_index']
    print('Calculating scores on test set... ')
    if len(pred) != len(target):
        print("Error in shapes", len(pred), len(target))
        target = target[:len(pred)]
        jet_flav = jet_flav[:len(pred)]
        print("Shortened to", len(target), len(jet_flav), len(pred)) 
    start = datetime.now()
    model_scores = {}

    target = np.asarray(target, dtype=object)
    pred = np.asarray(pred, dtype=object)#

    RI_func = np.vectorize(_get_rand_index)
    ARI_func = np.vectorize(adjustedRI_onesided)
    P_func = np.vectorize(_get_precision)
    R_func = np.vectorize(_get_recall)
    F1_func = np.vectorize(_f_measure)
    print('... RI')
    model_scores['RI'] = RI_func(target, pred)
    print('... ARI')
    model_scores['ARI'] = ARI_func(target, pred)
    print('... P')
    model_scores['P'] = P_func(target, pred)
    print('... R')
    model_scores['R'] = R_func(target, pred)
    print('... F1')
    model_scores['F1'] = F1_func(target, pred)

    end = datetime.now()
    print(f': {str(end - start).split(".")[0]}')

    flavours = {5: 'b jets', 4: 'c jets', 0: 'light jets'}
    metrics_to_table = ['P', 'R', 'F1', 'RI', 'ARI']

    df = pd.DataFrame(index=flavours.values(), columns=metrics_to_table)

    for flav_n, flav in flavours.items():
        for metric in metrics_to_table:
            mean_metric = np.mean(model_scores[metric][jet_flav == flav_n])
            df.at[flav, metric] = mean_metric

    return df


def _predict_on_test_set(model, sleeptime):
    test_ds = JetGraphDataset('test')
    model.eval()

    n_tracks = [test_ds[i][0].shape[0] for i in range(len(test_ds))]

    torch.cuda.empty_cache()

    indx_list = []
    predictions = []

    for tracks_in_jet in range(2, np.amax(n_tracks)+1):
        trk_indxs = np.where(np.array(n_tracks) == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list += list(trk_indxs)

        input_batch = torch.stack([test_ds[i][0] for i in trk_indxs])  # shape (B, N_i, 10)

        del trk_indxs
        torch.cuda.empty_cache()
        #print("del trk_indxs")

        edge_vals = model(input_batch).squeeze(1)
        edge_scores = 0.5*(edge_vals + edge_vals.transpose(1, 2))

        del edge_vals, input_batch
        torch.cuda.empty_cache()
        #print("del edge_vals, input_batch")

        edge_scores = torch.sigmoid(edge_scores)
        B,N,_ = edge_scores.shape
        edge_scores[:, np.arange(N), np.arange(N)] = 1.

        del B,N
        torch.cuda.empty_cache()
        #print("del B,N")

        pred_matrices = compute_clusters_with_partition_score(edge_scores)

        del edge_scores
        torch.cuda.empty_cache()
        #print("del edge_scores")

        pred_clusters = compute_vertex_assignment(pred_matrices)

        del pred_matrices
        torch.cuda.empty_cache()
        #print("del pred_matrices")

        predictions += list(pred_clusters.cpu().data.numpy())  # Shape

        del pred_clusters
        torch.cuda.empty_cache()
        #print("del pred_clusters")
        
        if sleeptime != 0:
            print("One iteration in _predict_on_test_set finished, waiting {:.1f}s for clearing VRAM".format(sleeptime))
            time.sleep(sleeptime)

    sorted_predictions = [list(x) for _, x in sorted(zip(indx_list, predictions))]
    del predictions, indx_list
    torch.cuda.empty_cache()
    return sorted_predictions


def compute_partition_score(edge_scores,pred_matrix):

    #print(edge_scores, pred_matrix)
    edge_scores = edge_scores.to("cpu")
    #pred_matrix.to("cuda")
    score = -(pred_matrix*torch.log(edge_scores+0.0000000000001)+(1-pred_matrix)*torch.log(1-edge_scores+0.0000000000001))
    #print(score.shape)
    score = score.sum(dim=1).sum(dim=1)
    return score


def fill_gaps(edge_vals):

    b, n, _ = edge_vals.shape
    
    with torch.no_grad():
        
        pred_matrices = edge_vals.ge(0.5).float() 
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

    return pred_matrices

def compute_vertex_assignment(pred_matrix):
    b, n, _ = pred_matrix.shape
    pred_matrix[:, np.arange(n), np.arange(n)] = 1.
    clusters = -1 * torch.ones((b, n))
    tensor_1 = torch.tensor(1.)
    for i in range(n):
        clusters = torch.where(pred_matrix[:, i] == 1, i * tensor_1, clusters)
    del b, n, pred_matrix, tensor_1
    torch.cuda.empty_cache()
    return clusters.long()

def compute_clusters_with_partition_score(edge_scores):
    # assuming edge_scores has been symmetrized and sigmoid was applied to it
    # edge scores shape is B,N,N
    
    edge_scores = edge_scores.to("cpu")
    B,N,_ = edge_scores.shape
    Ne = int(N*(N-1)/2)
    
    r,c = np.triu_indices(N,1)
    r  = np.tile(r,B)
    c = np.tile(c,B)
    z = np.repeat( np.arange(B), Ne)
    flat_edge_scores = edge_scores[z,r,c].view(B,Ne) #shape = B,Ne

    sorted_values, indices = torch.sort(flat_edge_scores,descending=True)
    
    final_edge_decision = torch.zeros(B,N,N)
    final_edge_decision[:, np.arange(N), np.arange(N)] = 1
    flat_sorted_edge_decisions = torch.zeros(B,Ne)
    
    partition_scores = compute_partition_score(edge_scores,final_edge_decision)
    
    for edge_i in range(Ne):
        
        temp_edge_decision = flat_sorted_edge_decisions.clone()
        
        temp_edge_decision[:,edge_i] = torch.where(sorted_values[:,edge_i] > 0.5,torch.tensor(1),torch.tensor(0))
        
        #reverse the sorting,
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))
        temp_partition = torch.zeros(B,N,N)
        temp_partition[z,r,c] = temp_edge_decision_unsorted.flatten()
        temp_partition.transpose(2,1)[z,r,c] = temp_edge_decision_unsorted.flatten()
        temp_partition = fill_gaps(temp_partition)
        
        temp_partition_scores = compute_partition_score(edge_scores,temp_partition)
        
        
        temp_edge_decision[:,edge_i] = torch.where( (temp_partition_scores < partition_scores) & (sorted_values[:,edge_i] > 0.5) ,torch.tensor(1),torch.tensor(0) )
        
        flat_sorted_edge_decisions = temp_edge_decision
        
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))

        final_edge_decision[z,r,c] = temp_edge_decision_unsorted.flatten()
        final_edge_decision.transpose(2,1)[z,r,c] = temp_edge_decision_unsorted.flatten()
        
        partition_scores = compute_partition_score(edge_scores,fill_gaps(final_edge_decision))

        del temp_edge_decision, temp_edge_decision_unsorted, temp_partition, temp_partition_scores
        torch.cuda.empty_cache()

    final_edge_decision = fill_gaps(final_edge_decision)

    del edge_scores, B, N, Ne, r, c, z, flat_edge_scores, sorted_values, indices, flat_sorted_edge_decisions, partition_scores
    torch.cuda.empty_cache()

    return final_edge_decision



def infer_clusters(edge_vals):
    '''
    Infer the clusters. Enforce symmetry.
    :param edge_vals: predicted edge score values. shape (B, N, N)
    :return: long tensor shape (B, N) of the clusters.
    '''
    # deployment - infer chosen clusters:
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals + edge_vals.transpose(1, 2)  # to make symmetric
        pred_matrices = pred_matrices.ge(0.).float()  # adj matrix - 0 as threshold
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        tensor_1 = torch.tensor(1., device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * tensor_1, clusters)

    return clusters.long()

def rand_index(labels,predictions):
    
    n_items = len(labels)
    if(n_items<2):
        return 1
    n_pairs = (n_items*(n_items-1))/2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i+1,n_items):
            label_true = labels[item_i]==labels[item_j]
            pred_true = predictions[item_i]==predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs+=1
    
    return correct_pairs/n_pairs


def Expval(labels):
    labels = np.array(labels)
    n = len(labels)
    nchoose2 = (n*(n-1))/2.0
    k = len(list(set(labels)))
    g = [len(np.where(labels==x)[0]) for x in list(set(labels))]
    bn = float( mpmath.bell(n) )
    bnmin1 = float( mpmath.bell(n-1) )
    bnratio = bnmin1/bn
    q = np.sum([(gi*(gi-1))/2 for gi in g])
    return bnratio*(q/nchoose2)+(1-bnratio)*(1-(q/nchoose2))
    
    

def adjustedRI_onesided(labels,predictions):
    
    ri = rand_index(labels,predictions)
    expval = Expval(labels)
    ari = (ri - expval)/(1-expval)
    return ari

def Error_count(labels,predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0
    
    for item_i in range(n_items):
        for item_j in range(item_i+1,n_items):
            label_true = labels[item_i]==labels[item_j]
            pred_true = predictions[item_i]==predictions[item_j]
            if (label_true and pred_true):
                true_positives+=1
            if (not label_true) and pred_true:
                false_positive+=1
            if label_true and (not pred_true):
                false_negative+=1
    return true_positives,false_positive,false_negative

def Recall(labels,predictions):
    true_positives,false_positive,false_negative = Error_count(labels,predictions)
    
    if true_positives+false_negative == 0:
        return 0
    
    return true_positives/(true_positives+false_negative)
    
def Precision(labels,predictions):
    true_positives,false_positive,false_negative = Error_count(labels,predictions)
    
    if true_positives+false_positive == 0:
        return 0
    return true_positives/(true_positives+false_positive)

def f_measure(labels,predictions):
    
    precision = Precision(labels,predictions)
    recall = Recall(labels,predictions)
    
    if precision==0 or recall==0:
        return 0
    
    return 2*(precision*recall)/(recall+precision)
