#/usr/bin/python

##Installations
#!pip install dgl-cu100==0.4.3
#!pip install h5py

##Dependencies
import argparse
import json
import numpy as np
import torch
import pandas as pd
import pickle
import h5py

import math
import time
import graph_utils
from graph_utils import build_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import conv as dgl_conv

# Load Pytorch as backend
dgl.load_backend('pytorch')

# See random for reproducibility.
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(0)

np.random.seed(0)

dgl.random.seed(0)

### Helper Functions ###
# NCE loss
def NCE_loss(pos_score, neg_score, neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score).reshape(-1, neg_sample_size)
    return -pos_score - torch.sum(neg_score, dim=1)

# DGL: edge sampler
def edge_sampler(g,batch_size, neg_sample_size,edges=None,shuffle=True,exclude_positive=True,return_false_neg=True):
#Exclude corrupted triplet already existing in the graph: exclude_positive=True
#Exclude generated negative edges in the graph: return_false_neg=True
    sampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=batch_size, 
                                               seed_edges=edges,
                                               neg_sample_size=neg_sample_size,
                                               negative_mode='tail',
                                               shuffle=shuffle,
                                               return_false_neg=return_false_neg)
    sampler = iter(sampler)
    return sampler #return a list of all samples

def get_src_nid(sub_g):
    src_nid, dst_nid = sub_g.all_edges(order='eid')
    # Get the node Ids in the parent graph.
    src_nid = sub_g.parent_nid[src_nid]
    dst_nid = sub_g.parent_nid[dst_nid]
    return src_nid, dst_nid

def score_func(sub_g, emb, sub_w):
    # Compute energy for positive edges
    # Read the node embeddings of the source nodes and destination nodes.
    src_nid, dst_nid = get_src_nid(sub_g)
    pos_heads = emb[src_nid]
    pos_tails = emb[dst_nid]
    return torch.sum(pos_heads * sub_w * pos_tails, dim=1) 

### Model Classes ###
# GraphSage model
class GraphSAGEModel(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type, mode='node'):
        super(GraphSAGEModel, self).__init__()
        self.mode = mode #'node', 'img', 'gl', 'both'
        #node embedding
        self.embedding = nn.Embedding(num_nodes,n_hidden)
        #node gating: img feature
        if self.mode == 'img' or self.mode == 'both':
            self.proj_img = nn.Linear(2048, n_hidden, bias=False)
            self.MLP_img = nn.Sequential(
                                  nn.Linear(2*n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, 1),
                                  nn.Sigmoid()
                                )
        #node gating: txt feature
        if self.mode == 'gl' or self.mode == 'both':
            self.proj_gl = nn.Linear(300, n_hidden, bias=False)
            self.MLP_gl = nn.Sequential(
                                  nn.Linear(2*n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, 1),
                                  nn.Sigmoid()
                                )
        #node gating: txt + img features
        if self.mode == 'both': 
            self.linear =  nn.Linear(2*n_hidden, n_hidden)

        #graphsage layers
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        #output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type,
                                         feat_drop=dropout, activation=None))

    def forward(self, g, img_feat=None, gl_feat=None):
        embed = self.embedding(g.ndata["node_id"]) #obtain minibatch node embedding
        gate_img, gate_gl = None, None 
        if self.mode == 'img': #img features
            img_feat = self.proj_img(img_feat)  #project img features to latent space
            gate_img = self.MLP_img(torch.cat([embed, img_feat],dim=-1)) #compute gating scalar through MLP
            h = gate_img * img_feat + (1 - gate_img) * embed #gated hidden state with img features
        elif self.mode == 'gl': #txt features
            gl_feat = self.proj_gl(gl_feat)  #project txt features to latent space
            gate_gl = self.MLP_gl(torch.cat([embed, gl_feat],dim=-1)) #compute gating scalar through MLP
            h = gate_gl * gl_feat + (1 - gate_gl) * embed #gated hidden state with txt features
        elif self.mode == 'both': #img + txt features
            img_feat = self.proj_img(img_feat)  
            gate_img = self.MLP_img(torch.cat([embed, img_feat],dim=-1))
            img_gated = gate_img * img_feat + (1 - gate_img) * embed
            gl_feat = self.proj_gl(gl_feat)  
            gate_gl = self.MLP_gl(torch.cat([embed, gl_feat],dim=-1))
            gl_gated = gate_gl * gl_feat + (1 - gate_gl) * embed
            #combine img + txt features
            combined_feat = torch.cat([img_gated, gl_gated], dim=-1) #2*hidden_dim
            h = self.linear(combined_feat) #project back to hidden_dim            
        else: #featureless
            h = embed
        #graphsage layers
        for layer in self.layers:
            h = layer(g, h)

        return h, gate_img, gate_gl   


#GraphSage w/ DistMult layer on top
class LinkPredict_Dist(nn.Module):
    def __init__(self, gconv_model,n_hidden, num_rels,batch_size, neg_sample_size, use_cuda=False, img_feat=None, gl_feat=None, mode='img'):
        super(LinkPredict_Dist, self).__init__()
        self.gconv_model = gconv_model
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size
        self.img_feat = img_feat
        self.gl_feat = gl_feat
        self.mode = mode #'img', 'gloss', 'both', 'node'
        self.proj_img , self.MLP_img, self.proj_gl, self.MLP_gl = None, None, None, None
        self.gate_scalar_img, self.gate_scalar_gl = None, None 
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, n_hidden)) #DistMult head
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        #Edge gating weights
        if self.mode == 'img' or self.mode == 'both':
            self.proj_img = nn.Linear(2048, n_hidden, bias=False)
            self.MLP_img = nn.Sequential(
                                  nn.Linear(2*n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, 1),
                                  nn.Sigmoid()
                                )
        if self.mode == 'gl' or self.mode == 'both':
            self.proj_gl = nn.Linear(300, n_hidden, bias=False)
            self.MLP_gl = nn.Sequential(
                                  nn.Linear(2*n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, 1),
                                  nn.Sigmoid()
                                )
        if self.mode == 'both': #combined both image and gloss
            self.linear =  nn.Linear(2*n_hidden, n_hidden)

    def forward(self, g): #obtain node hidden states with node gated features
        self.g = g
        return self.gconv_model(self.g, self.img_feat, self.gl_feat)

    def get_loss(self, emb, pos_g, neg_g, img_feat=None, gl_feat=None):
        # init gate scalar
        gate_img, gate_gl = None, None
        # get edge (relation) type
        pos_edge_type = self.g.edges[pos_g.parent_eid].data['type']
        sub_w_pos = self.w_relation[pos_edge_type]
        # obtain edge hidden states with edge gated features
        if self.mode == 'node': #no gating
            combined_sub_w_pos = sub_w_pos
        else:
            if img_feat is not None:
                combined_sub_w_pos_img, gate_img = self.combine_feat(pos_g, sub_w_pos, img_feat, feat_type='img')
            if gl_feat is not None:
                combined_sub_w_pos_gl, gate_gl = self.combine_feat(pos_g, sub_w_pos, gl_feat, feat_type='gl')
            #combine both img and gl
            if self.mode == 'both':
                combined_features = torch.cat([combined_sub_w_pos_img,combined_sub_w_pos_gl], dim=-1) #2*hidden_dim #correction! (prev dupliacte pos_img without using pos_gl)
                combined_sub_w_pos = self.linear(combined_features) #project back to hidden_dim
            elif self.mode == 'img':
                combined_sub_w_pos = combined_sub_w_pos_img
            else: #txt
                combined_sub_w_pos = combined_sub_w_pos_gl

        pos_score = score_func(pos_g, emb, combined_sub_w_pos)
        combined_sub_w_neg = torch.cat([combined_sub_w_pos.repeat_interleave(self.neg_sample_size)]).reshape(-1,self.n_hidden)
        neg_score = score_func(neg_g, emb, combined_sub_w_neg) 

        return torch.mean(NCE_loss(pos_score, neg_score, self.neg_sample_size)), gate_img, gate_gl

    
    def combine_feat(self, pos_g, sub_w_pos, feat, feat_type):
        '''
        Edge gating:
        1. Average node features (0.5*head + 0.5*tail)
        2. Concatenate averaged node features with edge features from DistMult head
        3. Obtain gating scalar through MLP
        4. Compute edge hidden states with edge gated features
        '''
        # get the parent node id
        src_nid_pos, dst_nid_pos = get_src_nid(pos_g)  
        if feat_type == 'img':
            head_pos = self.proj_img(feat[src_nid_pos])
            tail_pos = self.proj_img(feat[dst_nid_pos])
        else:
            head_pos = self.proj_gl(feat[src_nid_pos])
            tail_pos = self.proj_gl(feat[dst_nid_pos])            
        features = 0.5*head_pos + 0.5*tail_pos #average of head/tail features
        # concat with DistMult
        input_relations = torch.cat([features, sub_w_pos], dim=-1) 
        if feat_type == 'img':
            gate_scalar = self.MLP_img(input_relations)
        else:
            gate_scalar = self.MLP_gl(input_relations)
        combined_sub_w_pos = gate_scalar * features + (1-gate_scalar) * sub_w_pos

        return combined_sub_w_pos, gate_scalar   

###Train script
def model_train(train_g, g, valid_eids, test_eids, in_feats,n_hidden,n_layers,dropout,aggregator_type,num_rel,img_feat, gl_feat, mode='img',gate='graph',
                weight_decay=5e-4,n_epochs=100,batch_size=1000,neg_sample_size=100,lr=2e-3,save_file='./GraphSage_base.pt',use_cuda=False, 
                pretrain=False,pre_file=None):

    # create GraphSAGE model
    num_nodes = train_g.number_of_nodes()
    if gate == 'graph': #node gating only
      gconv_model = GraphSAGEModel(num_nodes,in_feats,n_hidden, n_hidden,n_layers,F.relu,dropout,aggregator_type, mode) 
      model = LinkPredict_Dist(gconv_model,n_hidden,num_rel,batch_size, neg_sample_size,use_cuda, img_feat, gl_feat,'node')  
    elif gate == 'dist': #edge gating only
      gconv_model = GraphSAGEModel(num_nodes,in_feats,n_hidden, n_hidden,n_layers,F.relu,dropout,aggregator_type, 'node') 
      model = LinkPredict_Dist(gconv_model,n_hidden,num_rel,batch_size, neg_sample_size,use_cuda, img_feat, gl_feat,mode) 
    else: #node + edge gating
      gconv_model = GraphSAGEModel(num_nodes,in_feats,n_hidden, n_hidden,n_layers,F.relu,dropout,aggregator_type, mode) 
      model = LinkPredict_Dist(gconv_model,n_hidden,num_rel,batch_size, neg_sample_size,use_cuda, img_feat, gl_feat,mode)    
      if pretrain:
        model.load_state_dict(torch.load(pre_file))
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    #send to GPU
    if use_cuda:
      model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #start without any weight_decay first, and then reg (gradually increase)

    # initialize graph
    old_mrr = 0

    for epoch in range(n_epochs):
        model.train()
        #mini-batch training
        edge_batches = edge_sampler(train_g, batch_size, neg_sample_size,edges=None,exclude_positive=True,return_false_neg=True)
        for batch in range(math.ceil(int(train_g.number_of_edges()/batch_size))+1):
            try:
              pos_g, neg_g = next(edge_batches)
            except StopIteration: #stopIteration flag by DGL
              break
            else:
              embed, gate_img_graph, gate_gl_graph = model(train_g) #GraphSage on structural node features only
              loss, gate_img, gate_gl = model.get_loss(embed, pos_g, neg_g,img_feat, gl_feat) #Pass in img_features with DistMult/Score_func            
              if batch % 1000 == 0:
                print('batch: ', batch)
                print('loss: ', loss)
                if gate_gl_graph is not None:
                    print('graph_txt_gates_max: ', gate_gl_graph.data.max(),'graph_txt_gates_min: ', gate_gl_graph.data.min())
                if gate_img_graph is not None:
                    print('graph_img_gates_max: ', gate_img_graph.data.max(),'graph_img_gates_min: ', gate_img_graph.data.min())
                if gate_gl is not None:
                    print('txt_gates_max: ', gate_gl.data.max(),'txt_gates_min: ', gate_gl.data.min())
                if gate_img is not None:
                    print('img_gates_max: ', gate_img.data.max(),'img_gates_min: ', gate_img.data.min())
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
 
        #eval    
        model.eval()
        gconv_model.eval()
        with torch.no_grad():
          embed, gate_img_graph, gate_gl_graph = model(train_g) #model(train_g) #GraphSage on structural node features only
          MC_results = LPEvaluate_relation(embed, g, valid_eids, batch_size, neg_sample_size, n_hidden, img_feat, gl_feat, model)   
          mrr_m, hits10_m, hits3_m, hits1_m = tuple(MC_results.mean().values[1:5]) #omit type and gates here
          mrr_std, hits10_std, hits3_std, hits1_std = tuple(MC_results.std().values[1:5]) 
        print("Epoch {:05d} | MRR_mean {:.4f} | Hits10_mean {:.4f} | Hits3_mean {:.4f} | Hits1_mean {:.4f}".format(epoch,mrr_m, hits10_m, hits3_m, hits1_m))
        print("Epoch {:05d} | MRR_std {:.4f} | Hits10_std {:.4f} | Hits3_std {:.4f} | Hits1_std {:.4f}".format(epoch,mrr_std, hits10_std, hits3_std, hits1_std))                                                        

        #  save model of best mrr
        if mrr_m > old_mrr:
            old_mrr = mrr_m
            torch.save(model.state_dict(), save_file)
            print('Model Saved')
            early_stop = 0
        else: 
            #early stopping after no improvements of 10 epochs
            early_stop += 1 
            if early_stop == 20:
                break
    print()

    return old_mrr

###Eval script
def LPEvaluate_relation(emb, g, valid_eids, batch_size, neg_sample_size, n_hidden,img_feat, gl_feat, model, graph_gate_img=None, graph_gate_gl=None):
    #df to store the aggregated results from MC runs
    df = pd.DataFrame(columns=['type','mrr','hits10','hits3','hits1'], dtype='float64')
    if model.mode != 'node': #edge gating
      if (img_feat is not None) or (graph_gate_img is not None): 
        df['gate_img'] = ""
      if (gl_feat is not None) or (graph_gate_gl is not None):
        df['gate_gl'] = ""

    #5 runs of eval to obtain mean and std: MRR, Hits10, Hits3, Hits1
    for i in range(5):
        #store per run results
        df_run = df.copy() #deep copy
        #mini-batch eval
        edge_batches = edge_sampler(g, batch_size, neg_sample_size,edges=valid_eids,shuffle=False) #no shuffling -> deterministic batches
        for batch in range(math.ceil(int(valid_eids.size()[0])/batch_size)+1): 
            try:
                pos_g, neg_g = next(edge_batches)
                #print(pos_g.parent_eid)
            except StopIteration: #stopIteration flag by DGL
                break
            else:
                # get edge features
                pos_edge_type = g.edges[pos_g.parent_eid].data['type'] #back to the parent graph
                sub_w_pos = model.w_relation[pos_edge_type]
                if model.mode != 'node': #if edge gating is on
                  if img_feat is not None:
                      combined_sub_w_pos_img, gate_img = model.combine_feat(pos_g, sub_w_pos, img_feat, feat_type='img')
                  if gl_feat is not None:
                      combined_sub_w_pos_gl, gate_gl = model.combine_feat(pos_g, sub_w_pos, gl_feat, feat_type='gl')
                  #combine both img and gl
                  if model.mode == 'both':
                      combined_features = torch.cat([combined_sub_w_pos_img,combined_sub_w_pos_gl], dim=-1) #2*hidden_dim #correction
                      combined_sub_w_pos = model.linear(combined_features) #project back to hidden_dim
                  elif model.mode == 'img':
                      combined_sub_w_pos = combined_sub_w_pos_img
                  else: #gloss
                      combined_sub_w_pos = combined_sub_w_pos_gl
                else:
                  combined_sub_w_pos = sub_w_pos #no gating!

                pos_score = score_func(pos_g, emb, combined_sub_w_pos)
                combined_sub_w_neg = torch.cat([combined_sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1,n_hidden)
                neg_score = score_func(neg_g, emb, combined_sub_w_neg) 

                #filter out existing edges in the graph
                filter_bias = neg_g.edata['false_neg'].reshape(-1, neg_sample_size)
                pos_score = F.logsigmoid(pos_score)
                neg_score = F.logsigmoid(neg_score).reshape(-1, neg_sample_size)
                neg_score -= filter_bias.float().cuda()
                pos_score = pos_score.unsqueeze(1)
                rankings = torch.sum(neg_score >= pos_score, dim=1) + 1

                #calculate mrr and hits
                rank = rankings.cpu().numpy()
                mrr = 1.0/rank
                hits10 = (rank <= 10).astype(int)
                hits3 = (rank <= 3).astype(int)
                hits1 = (rank <= 1).astype(int)

                ### output the raw rank (without averaging), then average per relation types
                if model.mode != 'node': #edge gating only
                  if graph_gate_img is None and graph_gate_gl is None: #ONLY edge gating, NO node gating
                      if img_feat is not None and gl_feat is not None: 
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_img': gate_img.detach().cpu().numpy().flatten(),
                                                  'gate_gl': gate_gl.detach().cpu().numpy().flatten()})
                      elif img_feat is not None:
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_img': gate_img.detach().cpu().numpy().flatten()})      
                      else:
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_gl': gate_gl.detach().cpu().numpy().flatten()}) 
                  else: #both edge gating + node gating
                      src_nid, dst_nid = get_src_nid(pos_g) #get src nid
                      if img_feat is not None and gl_feat is not None and graph_gate_img is not None and graph_gate_gl is not None: 
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_img_head': graph_gate_img[src_nid].detach().cpu().numpy().flatten(),
                                                  'gate_img_tail': graph_gate_img[dst_nid].detach().cpu().numpy().flatten(),
                                                  'gate_gl_head': graph_gate_gl[src_nid].detach().cpu().numpy().flatten(),
                                                  'gate_gl_tail': graph_gate_gl[dst_nid].detach().cpu().numpy().flatten(),
                                                  'gate_img': gate_img.detach().cpu().numpy().flatten(),
                                                  'gate_gl': gate_gl.detach().cpu().numpy().flatten()})   
                      elif img_feat is not None and graph_gate_img is not None:    
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_img_head': graph_gate_img[src_nid].detach().cpu().numpy().flatten(),
                                                  'gate_img_tail': graph_gate_img[dst_nid].detach().cpu().numpy().flatten(),
                                                  'gate_img': gate_img.detach().cpu().numpy().flatten()})   
                      else:
                          df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                                  'gate_gl_head': graph_gate_gl[src_nid].detach().cpu().numpy().flatten(),
                                                  'gate_gl_tail': graph_gate_gl[dst_nid].detach().cpu().numpy().flatten(),
                                                  'gate_gl': gate_gl.detach().cpu().numpy().flatten()})                             

                else: #node gating only
                  src_nid, dst_nid = get_src_nid(pos_g) #get src nid
                  if graph_gate_img is not None and graph_gate_gl is not None: 
                      df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                              'gate_img_head': graph_gate_img[src_nid].detach().cpu().numpy().flatten(),
                                              'gate_img_tail': graph_gate_img[dst_nid].detach().cpu().numpy().flatten(),
                                              'gate_gl_head': graph_gate_gl[src_nid].detach().cpu().numpy().flatten(),
                                              'gate_gl_tail': graph_gate_gl[dst_nid].detach().cpu().numpy().flatten()})
                  elif graph_gate_img is not None:
                      df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                              'gate_img_head': graph_gate_img[src_nid].detach().cpu().numpy().flatten(),
                                              'gate_img_tail': graph_gate_img[dst_nid].detach().cpu().numpy().flatten()})      
                  elif graph_gate_gl is not None:
                      df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1,
                                              'gate_gl_head': graph_gate_gl[src_nid].detach().cpu().numpy().flatten(),
                                              'gate_gl_tail': graph_gate_gl[dst_nid].detach().cpu().numpy().flatten()})
                  else:
                      df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),'mrr': mrr,'hits10': hits10,'hits3':hits3,'hits1':hits1})                              
                df_run = pd.concat((df_run, df_batch.groupby('type').mean().reset_index()))
        df = pd.concat((df,df_run.groupby('type').mean().reset_index()))
    return df

