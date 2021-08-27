import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        
        self.inf = 5.0 # For top radius
        self.cls_embeddings = nn.Embedding( nb_classes, embedding_size + 1)
        self.rel_embeddings = nn.Embedding( nb_classes, (embedding_size + 1)**2)
        
    def reg(self, x):
        res = torch.abs(torch.norm(x, dim=1) - self.reg_norm)
        return res
    
    def nf1_loss(self, input):
        c = input[:, 0]
        r = input[:,1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)

        rm = r.view(-1, embedding_size + 1, embedding_size + 1)
        cr = torch.matmul(rm, torch.unsqueeze(c, 2)).squeeze()
        dr = torch.matmul(rm, torch.unsqueeze(d, 2)).squeeze()

        rc = F.relu(cr[:, -1])
        rd = F.relu(dr[:, -1])
        
        x1 = cr[:, 0:-1]
        x2 = dr[:, 0:-1]
#         x3 = x1 + r
        
        euc = torch.norm(x2 - x1, dim=1)
        dst = F.relu(euc + rc - rd + self.margin)

        return dst + self.reg(c[:, 0:-1]) + self.reg(d[:, 0:-1])
    
    def nf2_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        e = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        e = self.cls_embeddings(e)
        rc = F.relu(c[:, -1])
        rd = F.relu(d[:, -1])
        re = F.relu(e[:, -1])
        sr = rc + rd
        
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = e[:, 0:-1]
        
        x = x2 - x1
        dst = torch.norm(x, dim=1)
        dst2 = torch.norm(x3 - x1, dim=1)
        dst3 = torch.norm(x3 - x2, dim=1)
        rdst = F.relu(torch.min(rc, rd) - re - self.margin)
        dst_loss = F.relu(dst - sr - self.margin)+ F.relu(dst2 - rc - self.margin) + F.relu(dst3 - rd - self.margin) + rdst
        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3)
    
    def nf3_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        
        rm = r.view(-1, embedding_size + 1, embedding_size + 1)
        cr = torch.matmul(rm, torch.unsqueeze(c, 2)).squeeze()
        dr = torch.matmul(rm, torch.unsqueeze(d, 2)).squeeze()
        
        x1 = cr[:, 0:-1]
        x2 = dr[:, 0:-1]
#         x3 = x1 + r

        rc = F.relu(c[:, -1])
        rd = F.relu(d[:, -1])
        
        euc = torch.norm(x2 - x1, dim=1)
        dst = F.relu(euc + rc - rd - self.margin)
        
        return dst + self.reg(c[:, 0:-1]) + self.reg(d[:, 0:-1])
    
    def nf4_loss(self, input):
        # R some C subClassOf D
        r = input[:, 0]
        c = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        
        rm = r.view(-1, embedding_size + 1, embedding_size + 1)
        cr = torch.matmul(rm, torch.unsqueeze(c, 2)).squeeze()
        dr = torch.matmul(rm, torch.unsqueeze(d, 2)).squeeze()
        
        x1 = cr[:, 0:-1]
        x2 = dr[:, 0:-1]
        
        rc = F.relu(c[:, -1])
        rd = F.relu(d[:, -1])
        sr = rc + rd
        
        # c - r should intersect with d
        dst = torch.norm(x2 - x1, dim=1)
        dst_loss = F.relu(dst - sr - self.margin)
        return dst_loss + self.reg(c[:, 0:-1]) + self.reg(d[:, 0:-1])
    
    def top_loss(self, inp):
        d = self.cls_embeddings(inp)
        rd = F.relu(d[:, -1])
        return torch.abs(rd - self.inf)
    
    def nf3_neg_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        
        #t ransformation of centres and radius in relation space
        rm = r.view(-1, embedding_size + 1, embedding_size + 1)
        cr = torch.matmul(rm, torch.unsqueeze(c, 2)).squeeze()
        dr = torch.matmul(rm, torch.unsqueeze(d, 2)).squeeze()
        
        # center
        x1 = cr[:, 0:-1]
        x2 = dr[:, 0:-1]
        
        # radius
        rc = F.relu(c[:, -1])
        rd = F.relu(d[:, -1])
        
        euc = torch.norm(x2 - x1, dim=1)
        dst = -(euc - rc - rd) + self.margin
        
        return dst + self.reg(c[:, 0:-1]) + self.reg(d[:, 0:-1])
    
    def inclusion_loss(self,input):
        r1 = input[:, 0]
        r2 = input[:, 1]
        r1 = self.rel_embeddings(r1)
        r2 = self.rel_embeddings(r2)
        #print("r2 type------>",type(r2))
        
        euc = torch.norm(r2 - r1, dim=1)
        
        normalize_a = torch.norm(r1, dim=1)        
        normalize_b = torch.norm(r2, dim=1)
        direction=torch.sum(normalize_a*normalize_b)
        dir_loss = torch.abs(1 - direction)
        dir_loss = dir_loss.view(-1, 1)
        dst = F.relu(euc - self.margin)
        
        return dst + self.reg(r1) + self.reg(r2) + dir_loss
    
    def chain_loss(self,input):
#         print('i', input, flush=True)
        r1 = input[:, 0]
        r2 = input[:, 1]
        r3 = input[:, 2]
        c = self.rel_embeddings(r1)
        d = self.rel_embeddings(r2)
        e = self.rel_embeddings(r3)
        
        c = c.view(-1, embedding_size + 1, embedding_size + 1)
        d = d.view(-1, embedding_size + 1, embedding_size + 1)
        e = e.view(-1, embedding_size + 1, embedding_size + 1)
        
        cd = torch.matmul(c, d)
        s = torch.abs(e - cd)
        
        element_sum = torch.mean(torch.mean(s, dim=1), dim=1)
        
        return element_sum
    
    def radius_loss(self, inp):
        d = self.cls_embeddings(inp)
        rd = d[:, -1]
        return torch.min(torch.zeros(rd.shape).cuda(),rd) 
    
    def forward(self, nf1, nf2, nf3, nf4,top, nf3_neg, nf_inclusion,  nf_chain, radius):
        loss1 = self.nf1_loss(nf1)
        loss2 = self.nf2_loss(nf2)
        loss3 = self.nf3_loss(nf3)
        loss4 = self.nf4_loss(nf4)
#         loss_dis = self.dis_loss(dis)
        loss_top = self.top_loss(top)
        loss_nf3_neg = self.nf3_neg_loss(nf3_neg)
#         loss5 = self.inclusion_loss(nf_inclusion)
        loss6 = self.chain_loss(nf_chain)
        loss7 = self.radius_loss(radius)
#         loss = loss1 + loss2 + loss3 + loss4 + loss_top + loss_nf3_neg + loss5 + loss6 - loss7
        loss = loss1 + loss2 + loss3 + loss4 + loss_top + loss_nf3_neg + loss6 -  loss7

        return torch.mean(loss.squeeze())

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

    embeds_list = model.cls_embeddings(torch.tensor(range(nb_classes)).cuda())
#     embeds_list = cls_df['embeddings'].values

#     classes = {v: k for k, v in enumerate(cls_df['classes'])}
    classes = cls_df['classes']
    embeds_list = embeds_list.detach().cpu().numpy()

    size = len(embeds_list[0])
    embeds = np.zeros((nb_classes, size), dtype=np.float32)
    for i, emb in enumerate(embeds_list):
        embeds[i, :] = emb
    embeds = embeds[:, :-1]
    
    top1 = 0
    top10 = 0
    top100 = 0
    mean_rank = 0
    rank_vals =[]
    for test_pts in data:
        c = test_pts[0]
        d = test_pts[1]
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
    
    n = len(data)
    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank /= n
    total_classes = len(embeds)
    return top1,top10,top100,mean_rank,rank_vals,total_classes  

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

def print_results(rks_vals,n):
    print("top1:",top1)
    print("top10:",top10)
    print("top100:",top100)
    print("Mean Rank:",mean_rank)
    rank_dicts = dict(Counter(rks_vals))
    print("AUC:",compute_rank_roc(rank_dicts,n))
    out_results(rks_vals) 


tag='GALEN'
AEL_dir = 'experiments/results/EmEL_dir/'
test_file = 'experiments/data/'+tag+'/'+tag+'_test.txt'
test_data = load_eval_data(test_file)

margin = 0
embedding_size = 50
batch_size =  256
reg_norm=1
learning_rate=3e-4
cls_embeds_file = AEL_dir+tag+'_{'+str(embedding_size)+'}_{'+str(margin)+'}_{1000}.pkl'


# In[13]:

print('start evaluation........')
top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size,batch_size,margin,reg_norm)


# In[14]:


#####50,0
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


