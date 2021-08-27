import os
import click as ck
import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
import pickle as pkl
import random

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'SubClassOf'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data

def load_data(filename, all_subcls):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [],'nf_inclusion':[],'nf_chain':[]}
    with open(filename) as f:
        for line in f:
            # Ignore SubObjectPropertyOf
            if line.startswith('SubObjectPropertyOf'):
                line = line.strip()[20:-1]
                if line.startswith('ObjectPropertyChain'):
                    line_chain = line.strip()[20:-1]
                    line1 = line.split(")")
                    line10 = line1[0].split()
                    r1 = line10[0]
                    r2 = line10[1]
                    r3 = line1[1]
                    if r1 not in relations:
                        relations[r1] = len(relations)
                    if r2 not in relations:
                        relations[r2] = len(relations)
                    if r3 not in relations:
                        relations[r3] = len(relations)
                    data['nf_chain'].append((relations[r1],relations[r2],relations[r3]))
                else:
#                     print("Inside sub obj prop")
                    it = line.split(' ')
                    r1 = it[0]
                    r2 = it[1]
                    if r1 not in relations:
                        relations[r1] = len(relations)
                    if r2 not in relations:
                        relations[r2] = len(relations)
                    data['nf_inclusion'].append((relations[r1], relations[r2]))
            # Ignore SubClassOf()
            line = line.strip()[11:-1]
            if not line:
                continue
            if line.startswith('ObjectIntersectionOf('):
                # C and D SubClassOf E
                it = line.split(' ')
                c = it[0][21:]
                d = it[1][:-1]
                e = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if e not in classes:
                    classes[e] = len(classes)
                form = 'nf2'
                if e == 'owl:Nothing':
                    form = 'disjoint'
                data[form].append((classes[c], classes[d], classes[e]))
                
            elif line.startswith('ObjectSomeValuesFrom('):
                # R some C SubClassOf D
                it = line.split(' ')
                r = it[0][21:]
                c = it[1][:-1]
                d = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf4'].append((relations[r], classes[c], classes[d]))
            elif line.find('ObjectSomeValuesFrom') != -1:
                # C SubClassOf R some D
                it = line.split(' ')
                c = it[0]
                r = it[1][21:]
                d = it[2][:-1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                r = 'SubClassOf'
                if r not in relations:
                    relations[r] = len(relations)
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c],relations[r],classes[d]))
                
    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
#changing by adding sub classes of train_data ids to prot_ids
    prot_ids = []
    class_keys = list(classes.keys())
    for val in all_subcls:
        if val not in class_keys:
            cid = len(classes)
            classes[val] = cid
            prot_ids.append(cid)
        else:
            prot_ids.append(classes[val])
#     for k, v in classes.items():
#         if k in all_subcls:
#             prot_ids.append(v)
        
    prot_ids = np.array(prot_ids)
    
    # Add corrupted triples nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    for c, r, d in data['nf3']:
        x = np.random.choice(prot_ids)
        while x == c:
            x = np.random.choice(prot_ids)
            
        y = np.random.choice(prot_ids)
        while y == d:
             y = np.random.choice(prot_ids)
        data['nf3_neg'].append((c, r,x))
        data['nf3_neg'].append((y, r, d))

    data['radius'] = []
    for val in classes:
        data['radius'].append(classes[val])
    
    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([classes['owl:Thing'],])
    data['nf3_neg'] = np.array(data['nf3_neg'])
    data['nf_inclusion'] = np.array(data['nf_inclusion'])
    data['nf_chain'] = np.array(data['nf_chain'])
    data['radius'] = np.array(data['radius'])
                            
    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]
    
    return data, classes, relations



def load_cls(train_data_file):
    train_subs=list()
    counter=0
    with open(train_data_file,'r') as f:
        for line in f:
            counter+=1
            it = line.strip().split()
            cls1 = it[0]
            cls2 = it[1]
            train_subs.append(cls1)
            train_subs.append(cls2)
    train_cls = list(set(train_subs))
    return train_cls,counter

class Generator(Dataset):

    def __init__(self, data, batch_size=128, steps=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = steps
        self.start = 0
        
    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.steps:
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf2_index = np.random.choice(
                self.data['nf2'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
#             dis_index = np.random.choice(
#                 self.data['disjoint'].shape[0], self.batch_size)
            top_index = np.random.choice(
                self.data['top'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            nf_inclusion_index = np.random.choice(self.data['nf_inclusion'].shape[0],self.batch_size)
            nf_chain_index = np.random.choice(self.data['nf_chain'].shape[0],self.batch_size)
            radius_index = np.random.choice(
                self.data['radius'].shape[0], self.batch_size)
            nf1 = torch.tensor(self.data['nf1'][nf1_index])
            nf2 = torch.tensor(self.data['nf2'][nf2_index])
            nf3 = torch.tensor(self.data['nf3'][nf3_index])
            nf4 = torch.tensor(self.data['nf4'][nf4_index])
#             dis = self.data['disjoint'][dis_index]
            top = torch.tensor(self.data['top'][top_index])
            nf3_neg = torch.tensor(self.data['nf3_neg'][nf3_neg_index])
            nf_inclusion = torch.tensor(self.data['nf_inclusion'][nf_inclusion_index])
            nf_chain = torch.tensor(self.data['nf_chain'][nf_chain_index])
            radius = torch.tensor(self.data['radius'][radius_index])
#             labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            
            return (nf1, nf2, nf3, nf4, top, nf3_neg,nf_inclusion,nf_chain,radius)
        else:
            self.reset()
            
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

def build_model(train_data,classes,relations,valid_data, margin, embedding_size, batch_size, reg_norm, learning_rate, epochs, out_file):
    proteins = {}#substitute for classes with subclass case
    for val in all_subcls:
        proteins[val] = classes[val]
    nb_classes = len(classes)
    nb_relations = len(relations)
    print("no. classes:",nb_classes)
    print("no. relations:",nb_relations)
    nb_train_data = 0
    for key, val in train_data.items():
        nb_train_data = max(len(val), nb_train_data)
    train_steps = int(math.ceil(nb_train_data / (1.0 * batch_size)))
    train_generator = Generator(train_data, batch_size, steps=train_steps)

    cls_dict = {v: k for k, v in classes.items()}
    rel_dict = {v: k for k, v in relations.items()}
    best_rank = 100000000
    n = len(valid_data)
    
    cls_list = []
    rel_list = []
    for i in range(nb_classes):
        cls_list.append(cls_dict[i])
    for i in range(nb_relations):
        rel_list.append(rel_dict[i])
    
    model = ELModel(nb_classes, nb_relations, embedding_size, batch_size, margin, reg_norm).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    prot_index = list(proteins.values())
    prot_dict = {v: k for k, v in enumerate(prot_index)}
    
    for epoch in range(epochs):
        model.train()
        print("EPOCH : ", epoch)
        train_loss = 0
        for batch in train_generator:
            if batch is None:
                break
            nf1, nf2, nf3, nf4, top, nf3_neg,  nf_inclusion,  nf_chain, radius = batch
            
            nf1 = nf1.cuda()
            nf2 = nf2.cuda()
            nf3 = nf3.cuda()
            nf4 = nf4.cuda()
            top = top.cuda()
            nf3_neg = nf3_neg.cuda()
            nf_inclusion = nf_inclusion.cuda()
            nf_chain = nf_chain.cuda()
            radius = radius.cuda()
            
            loss = model(nf1, nf2, nf3, nf4, top, nf3_neg, nf_inclusion,  nf_chain, radius)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= train_steps
        print('Train Loss :', train_loss.item())
        model.eval()
        prot_embeds = model.cls_embeddings(torch.tensor(prot_index).cuda())        
        rel = model.rel_embeddings(torch.tensor(0).cuda()) # taking only sub class as relation
        rel = rel.view(embedding_size + 1, embedding_size + 1)
        prot_emeds = torch.matmul(rel, torch.unsqueeze(prot_embeds, 2)).squeeze()

        prot_rs = prot_embeds[:, -1].detach().cpu().numpy()
        prot_embeds = prot_embeds[:, :-1].detach().cpu().numpy()
        mean_rank = 0
        
        for c, r, d in valid_data:
            c, r, d = prot_dict[c], r, prot_dict[d]
            ec = prot_embeds[c, :]
            rc = prot_rs[c]
            dst = np.linalg.norm(prot_embeds - ec.reshape(1, -1), axis=1)
            res = np.maximum(0, dst + rc - prot_rs + margin)
            res = res.flatten()
            index = rankdata(res, method='average')
            rank = index[d]
            mean_rank += rank
        mean_rank /= n 
        print(f'\n Validation {epoch + 1} {mean_rank}\n')
        if mean_rank < best_rank:
            best_rank = mean_rank
            print(f'\n Saving embeddings {epoch + 1} {mean_rank}\n')
            df = {'classes': classes, 'relations': relations, 'embeddings': model.state_dict(), 'cls':cls_list, 'rel':rel_list}
            
            with open(out_file, 'wb') as f:
                pkl.dump(df, f)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, action = 'store', help="provide the data older name eg.Go", default = 'GALEN')
    parser.add_argument("--bs", type=int, action = 'store', help="batch size", default = 256)
    parser.add_argument("--seed", type=int, action = 'store', help="seed", default = 256)
    parser.add_argument("--reg_norm", type=int, action = 'store', help="regularising norm", default=1)
    parser.add_argument("--lr", type=int, action = 'store', help="learning rate", default=3e-4)
    parser.add_argument("--epoch", type=int, action = 'store', help="epochs", default=1000)
    
    args = parser.parse_args()

    total_sub_cls=[]
    train_file = "../data/"+args.data+"/"+args.data+"_train.txt"
    val_file = "../data/"+args.data+"/"+args.data+"_valid.txt"
    test_file = "../data/"+args.data+"/"+args.data+"_test.txt"
    gdata_file="../data/"+args.data+"/"+args.data+"_norm_mod.owl"
    valid_data_file="../data/"+args.data+"/"+args.data+"_valid.txt"

    
    train_sub_cls,train_samples = load_cls(train_file)
    valid_sub_cls,valid_samples = load_cls(val_file)
    test_sub_cls,test_samples = load_cls(test_file)
    total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
    all_subcls = list(set(total_sub_cls))
    
    train_data_model, classes, relations = load_data(gdata_file, all_subcls)
    valid_data_model = load_valid_data(valid_data_file, classes, relations)
    
    margins = [-0.1,0,0.1]
    embed_dims = [50, 100]
    batch_size = args.bs
    reg_norm = args.reg_norm
    learning_rate = args.lr
    epochs = args.epoch
    set_seed(args)
    
    for d in embed_dims:
        embedding_size = d
        print("***************Embedding Dim:",embedding_size,'****************')
        for m in margins:
            margin = m
            print("**************Margin Loss:",margin,"***************")
            out_file = f'../results/EmEL_dir/'+args.data+'_{'+str(embedding_size)+'}_{'+str(margin)+'}_{'+str(epochs)+'}.pkl'
            loss_history_file= f'../results/EmEL_dir/'+args.data+'_lossHis_{'+str(embedding_size)+'}_{'+str(margin)+'}_{'+str(epochs)+'}.csv'
            build_model(train_data_model,classes,relations,valid_data_model, margin, embedding_size, batch_size, reg_norm, learning_rate, epochs, out_file)

