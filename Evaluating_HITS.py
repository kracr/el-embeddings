import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-GPU-714a8f8e-44b2-93f1-e64d-d5f204c379de/5/0"
import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
import operator
from collections import Counter

class ELModel(nn.Module):
    
    def __init__(self, nb_classes, nb_relations, embedding_size, batch_size, margin, reg_norm):
        super(ELModel, self).__init__()
        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.margin = margin
        self.reg_norm = reg_norm
        width = 3
        
        self.inf = 5.0 # For top radius
        self.cls_embeddings = nn.Embedding( nb_classes, embedding_size + 1)
        self.rel_embeddings = nn.Embedding( nb_relations, embedding_size + 1)

def load_eval_data(data_file):
    data = []
    rel = f'SubClassOf'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            data.append((id1, id2))
    return data

def evaluate_hits(data,cls_embeds_file, embedding_size, batch_size, margin, reg_norm):
    with open(cls_embeds_file, 'rb') as f:
        cls_df = pkl.load(f)
    nb_classes = len(cls_df['cls'])
    nb_relations = len(cls_df['rel'])
    model = ELModel(nb_classes, nb_relations, embedding_size, batch_size, margin, reg_norm).cuda()
    model.load_state_dict(cls_df['embeddings'])   
    model.eval()

    embeds_list = model.cls_embeddings(torch.tensor(list(range(nb_classes))).cuda())
#     print(list(range(nb_classes)))
#     embeds_list = cls_df['embeddings'].values

#     classes = {v: k for k, v in enumerate(cls_df['classes'])}
    classes = cls_df['classes']
    rel = model.rel_embeddings(torch.tensor(0).cuda()).detach().cpu().numpy()
    rel = rel[:-1]
    
    embeds_list = embeds_list.detach().cpu().numpy()

    size = len(embeds_list[0])
#     embeds = np.zeros((nb_classes, size), dtype=np.float32)
#     for i, emb in enumerate(embeds_list):
#         embeds[i, :] = emb
    embeds =  embeds_list   
    embeds = embeds[:, :-1]
#     print(classes)
    
    top1 = 0
    top10 = 0
    top25 = 0
    top50 = 0
    top100 = 0
    mean_rank = 0
    rank_vals =[]
    count=0
    for test_pts in tqdm(data):
        c = test_pts[0]
        d = test_pts[1]
        if c not in classes or d not in classes:
            count+=1
            continue
        index_c = classes[c]
        index_d = classes[d]
        dist =  np.linalg.norm(embeds - embeds[index_d], axis=1) 
        dist_dict = {i: dist[i] for i in range(0, len(dist))} 
        s_dst = dict(sorted(dist_dict.items(), key=operator.itemgetter(1)))
        s_dst_keys = list(s_dst.keys())
        ranks_dict = { s_dst_keys[i]: i for i in range(0, len(s_dst_keys))}
        rank_c = ranks_dict[index_c]
        mean_rank += rank_c
        rank_vals.append(rank_c)
        if rank_c == 1:
            top1 += 1
        if rank_c <= 10:
            top10 += 1
        if rank_c <= 100:
            top100 += 1
        if rank_c <= 25:
            top25 += 1
        if rank_c <= 50:
            top50 += 1
    
    n = len(data)
    top1 /= n
    top10 /= n
    top100 /= n
    top25 /= n
    top50 /= n
    mean_rank /= n
    total_classes = len(embeds)
    print(count)
    return top1,top10,top25,top50,top100,mean_rank,rank_vals,total_classes   

def compute_rank_percentile(scores,x):
    scores.sort()
    per = np.percentile(scores,x)
    return per

import statistics
def compute_median_rank(rank_list):
    med = np.median(rank_list)
    return med    

def calculate_percentile_1000(scores):
    ranks_1000=[]
    for item in scores:
        if item < 1000:
            ranks_1000.append(item)
    n_1000 = len(ranks_1000)
    nt = len(scores)
    percentile = (n_1000/nt)*100
    return percentile

def compute_rank_roc(ranks, n):
    auc_lst = list(ranks.keys())
    auc_x = auc_lst[1:]
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x)/n
    return auc

def out_results(rks_vals):
    med_rank = compute_median_rank(rks_vals)
    print("Median Rank:",med_rank)
    per_rank_90 = compute_rank_percentile(rks_vals,90)
    print("90th percentile rank:",per_rank_90)
    percentile_below1000 = calculate_percentile_1000(rks_vals)
    print("Percentile for below 1000:",percentile_below1000)
    print("% Cases with rank greater than 1000:",(100 - percentile_below1000))
    return med_rank, per_rank_90, percentile_below1000, (100 - percentile_below1000)

def print_results(rks_vals,n):
    print("top1:",top1)
    print("top10:",top10)
    print("top25:",top25)
    print("top50:",top50)
    print("top100:",top100)
    print("Mean Rank:",mean_rank)
    rank_dicts = dict(Counter(rks_vals))
    auc = compute_rank_roc(rank_dicts,n)
    print("AUC:",auc)
    mr, pr, p, bp = out_results(rks_vals) 
    print("top1, top10, top25, top50, top100, Mean Rank, AUC, Median Rank, 90th percentile rank, Pecentile for below 1000, % Cases with rank greater than 1000")
    print(top1, top10, top25, top50, top100, mean_rank, auc, mr, pr, p, bp) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, action = 'store', help="Data", default="GALEN")
    parser.add_argument("--tag", type=str, action = 'store', help="Data split eq 1_1, 1_n, n_n", default="1_1")
    parser.add_argument("--margin", type=str, action = 'store', help="Data split eq 1_1, 1_n, n_n", default="-0.1")
    parser.add_argument("--dim", type=str, action = 'store', help="Data split eq 1_1, 1_n, n_n", default="100")
    parser.add_argument("--early_stop", type=bool, action = 'store', help="Data split eq 1_1, 1_n, n_n", default=False)
    args = parser.parse_args()
    tag=args.tag
    print(args.early_stop, type(args.early_stop))
    if not args.early_stop:
        AEL_dir = f'Experiments/results_no_early_stopping/{args.data}/'
    else:
        AEL_dir = f'Experiments/results/{args.data}/'
    test_file = f'Experiments/data/{args.data}/{args.data}_inferences.txt'
    # test_file = f'Experiments/data/{args.data}/'+tag+'/test.txt'
    test_data = load_eval_data(test_file)

    # margin = -0.1
    margin = eval(args.margin)
    # embedding_size = 100
    embedding_size = eval(args.dim)
    batch_size =  256
    reg_norm=1
    learning_rate=3e-4
    cls_embeds_file = AEL_dir+tag+'_{'+str(embedding_size)+'}_{'+str(margin)+'}_{1000}.pkl'

    print(tag, margin, embedding_size)

    print('start evaluation........')
    top1,top10,top25,top50,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size,batch_size,margin,reg_norm)


    print("EmEL Results on test data")
    print_results(rank_vals,n_cls)


# # In[98]:


# # tag='GALEN'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = 0
# # embedding_size = 50
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'


# # # In[88]:


# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[91]:


# # print("EL Results on test data")
# # print_results(rank_vals,n_cls)


# # # GALEN Evaluation on Inferences

# # In[15]:


# tag='GALEN'
# AEL_dir = f'{tag}/EmEL/'
# test_file = f'{tag}/{tag}_inferences.txt'
# test_data = load_eval_data(test_file)
# margin = 0
# embedding_size = 50
# batch_size =  256
# device='gpu:0'
# reg_norm=1
# learning_rate=3e-4
# cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_150_cls.pkl'
# top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size,batch_size,margin,reg_norm)


# # In[16]:


# ###50,0
# print("==========EmEL Results on Inferences data=========")
# print_results(rank_vals,n_cls)


# # In[102]:


# # tag='GALEN'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_inferences.txt'
# # test_data = load_eval_data(test_file)
# # margin = 0
# # embedding_size = 50
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[103]:


# # print("==========EL Results on Inferences data=========")
# # print_results(rank_vals,n_cls)


# # # GO Hits Evaluation on Test Data

# # In[17]:


# tag='GO'
# AEL_dir = f'{tag}/EmEL/'
# test_file = f'{tag}/{tag}_test.txt'
# test_data = load_eval_data(test_file)
# margin = -0.1
# embedding_size = 100
# batch_size =  256
# device='gpu:0'
# reg_norm=1
# learning_rate=3e-4
# cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_150_cls.pkl'
# top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size,batch_size,margin,reg_norm)


# # In[18]:


# ####100,-0.1
# print("==========EmEL Results on Test data=========")
# print_results(rank_vals,n_cls)


# # In[19]:


# # tag='GO'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 100
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[20]:


# # ##100,-0.1
# # print("==========EL Results on Test data=========")
# # print_results(rank_vals,n_cls)



# # # GO Evaluation on Inferences 

# # In[21]:


# tag='GO'
# AEL_dir = f'{tag}/EmEL/'
# test_file = f'{tag}/{tag}_inferences.txt'
# test_data = load_eval_data(test_file)
# margin = -0.1
# embedding_size = 100
# batch_size =  256
# device='gpu:0'
# reg_norm=1
# learning_rate=3e-4
# cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_150_cls.pkl'
# top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size,batch_size,margin,reg_norm)


# # In[22]:


# ###100,-0.1
# print("==========EmEL Results on Inferences data=========")
# print_results(rank_vals,n_cls)



# # In[23]:


# # tag='GO'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_inferences.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 100
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[24]:


# # ####100,-0.1
# # print("==========EL Results on Inferences data=========")
# # print_results(rank_vals,n_cls)




# # # # Anatomy on Test Data

# # # In[8]:


# # tag='ANATOMY'
# # AEL_dir = f'{tag}/EmEL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 200
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[9]:


# # ##200,-0.1
# # print("==========EmEL Results on Test data=========")
# # print_results(rank_vals,n_cls)


# # # In[10]:


# # tag='ANATOMY'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 200
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[11]:


# # ###200,-0.1
# # print("==========EL Results on Test data=========")
# # print_results(rank_vals,n_cls)


# # # # Anatomy on Inferences Data

# # # In[12]:


# # tag='ANATOMY'
# # AEL_dir = f'{tag}/EmEL/'
# # test_file = f'{tag}/{tag}_inferences.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 200
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[13]:


# # ##200,-0.1
# # print("==========EmEL Results on Inferences data=========")
# # print_results(rank_vals,n_cls)


# # # In[14]:


# # tag='ANATOMY'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_inferences.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 200
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[15]:


# # ###200,-0.1
# # print("==========EL Results on Inferences data=========")
# # print_results(rank_vals,n_cls)



# # # # SNOMED on Test Data

# # # In[8]:


# # tag='SNOMED'
# # AEL_dir = f'{tag}/EmEL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 100
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)
# # print("==========EmEL Results on Test data=========")
# # print_results(rank_vals,n_cls)

# # tag='SNOMED'
# # AEL_dir = f'{tag}/EL/'
# # test_file = f'{tag}/{tag}_test.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 100
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)



# # print("==========EL Results on Test data=========")
# # print_results(rank_vals,n_cls)


# # # # SNOMED on Inferences

# # tag='SNOMED'
# # AEL_dir = f'{tag}/EmEL/'
# # test_file = f'{tag}/{tag}_inferences.txt'
# # test_data = load_eval_data(test_file)
# # margin = -0.1
# # embedding_size = 100
# # batch_size =  256
# # device='gpu:0'
# # reg_norm=1
# # learning_rate=3e-4
# # cls_embeds_file = AEL_dir + f'{tag}_{embedding_size}_{margin}_1000_cls.pkl'
# # test_data = test_data[0:12590]
# # top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file)


# # # In[12]:


# # print("==========EmEL Results on Inferences data=========")
# # print_results(rank_vals,n_cls)



