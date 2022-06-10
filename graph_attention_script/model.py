# imports
import json
import numpy as np
import torch
import pandas as pd
from pandas.io.json import json_normalize
import os
import time
from graph_utils import build_graph, get_train_val_test_masks
import dgl
from dgl.contrib.data import load_data
import math
import torch.nn.functional as F
from torch import nn
import torch
import dgl
use_cuda = True

def build_graph_from_triplets(num_nodes, train_data, valid_data, test_data):
    # Create a DGL graph. The graph is uni-directional for our model comparisons
    # Allow edge_id to subset graph
    # source: https://github.com/dmlc/dgl/blob/1abe87f586e1f2983d4e521e69e82e3723037c68/examples/pytorch/rgcn/utils.py#L128
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    # train edges
    g.add_edges(train_data[:,0], train_data[:,2])
    print("# nodes: {}, #train edges: {}".format(num_nodes, len(train_data[:,0])))

    # valid edges
    g.add_edges(valid_data[:,0], valid_data[:,2])
    print("# nodes: {}, #valid edges: {}".format(num_nodes, len(valid_data[:,0])))

    # test edges
    g.add_edges(test_data[:,0], test_data[:,2])
    print("# nodes: {}, #test edges: {}".format(num_nodes, len(test_data[:,0])))
    # add edge types
    g.edata['type'] = torch.tensor(np.concatenate([train_data[:,1],valid_data[:,1],test_data[:,1]]))

    return g

# import benchmark dataset
from dgl.contrib.data import load_data
from dglke.dataloader.KGDataset import get_dataset

def build_g_from_data(dataset, data_location, num_nodes, num_rels, use_cuda = False):

    if dataset == 'VisualSem':

        num_nodes = num_nodes
        train_data = np.load(os.path.join(data_location, "train.npy"))
        valid_data = np.load(os.path.join(data_location, "valid.npy"))
        test_data = np.load(os.path.join(data_location, "test.npy"))
        num_rels = num_rels

    else:
        data = load_data(dataset)
        num_nodes = data.num_nodes
        train_data = data.train
        valid_data = data.valid
        test_data = data.test
        num_rels = data.num_rels

    # get unique number of nodes
    if dataset == 'FB15k-237' or dataset == 'wn18rr':
        g = build_graph_from_triplets(num_nodes, train_data, valid_data, test_data)
    else:
        # uniq_v, edges = np.unique((train_data[:, 0], train_data[:, 2]), return_inverse=True)
        g = build_graph_from_triplets(num_nodes, train_data, valid_data, test_data)

    train_cut = len(train_data[:,0])
    valid_cut = len(train_data[:,0])+len(valid_data[:,0])
    test_cut = len(train_data[:,0])+len(valid_data[:,0])+len(test_data[:,0])

    train_eids = torch.arange(train_cut) #.cuda()
    valid_eids = torch.arange(train_cut,valid_cut) #.cuda()
    test_eids = torch.arange(valid_cut,test_cut) #.cuda()

    if use_cuda:
        train_eids, valid_eids, test_eids = train_eids.cuda(), valid_eids.cuda(), test_eids.cuda()

    return g, train_eids, valid_eids, test_eids, num_rels

# loss and training mechanism

# NCE loss
def NCE_loss(pos_score, neg_score, neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score).reshape(-1, neg_sample_size)
    return -pos_score - torch.sum(neg_score, dim=1)

class LinkPrediction(nn.Module):
    def __init__(self, GAT_model):
        super(LinkPrediction, self).__init__()
        self.GAT_model = GAT_model

    def forward(self, g, features, pos_g, neg_g, neg_sample_size):
        emb = self.GAT_model(g, features)
        pos_score = score_func(pos_g, emb)
        neg_score = score_func(neg_g, emb)
        return emb, torch.mean(NCE_loss(pos_score, neg_score, neg_sample_size))


#Link Predict w/ Distmult layer on top
class LinkPredict_Dist(nn.Module):
    def __init__(self, GAT_model, h_dim, num_rels, batch_size, neg_sample_size,
                 use_cuda=False, img_feat=None, txt_feat=None):
        super(LinkPredict_Dist, self).__init__()
        self.GAT_model = GAT_model
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size
        self.h_dim = h_dim
        self.img_feat = img_feat
        self.txt_feat = txt_feat
        self.w_relation = nn.Parameter(torch.Tensor(int(num_rels), int(h_dim)))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

        if self.img_feat is not None:

            self.img_proj = nn.Linear(img_feat.shape[1], h_dim, bias = False)
            self.img_MLP = nn.Sequential(
                           nn.Linear(2*h_dim, h_dim),
                           nn.ReLU(),
                           nn.Linear(h_dim, h_dim),
                           nn.ReLU(),
                           nn.Linear(h_dim, 1),
                           nn.Sigmoid()
                           )

        if self.txt_feat is not None:

           self.txt_proj = nn.Linear(txt_feat.shape[1], h_dim, bias = False)
           self.txt_MLP = nn.Sequential(
                                  nn.Linear(2*h_dim, h_dim),
                                  nn.ReLU(),
                                  nn.Linear(h_dim, h_dim),
                                  nn.ReLU(),
                                  nn.Linear(h_dim, 1),
                                  nn.Sigmoid()
                                )

        if (self.txt_feat is not None) & (self.img_feat is not None):

            self.combined_MLP = nn.Sequential(
                                  nn.Linear(2*h_dim, h_dim)
                                )

    def forward(self, g, img_features, txt_features):

        self.g = g

        return self.GAT_model(self.g, img_features, txt_features), self.w_relation


    def get_loss(self, emb, pos_g, neg_g, img_feat=None, txt_feat=None):
        # get additional w relation
        pos_edge_type = self.g.edges[pos_g.parent_eid].data['type']
        sub_w_pos = self.w_relation[pos_edge_type]

        sub_w_neg = torch.cat([sub_w_pos.repeat_interleave(self.neg_sample_size)]).reshape(-1, self.h_dim)
        pos_score = score_func(pos_g, emb, sub_w_pos)
        neg_score = score_func(neg_g, emb, sub_w_neg)

        return torch.mean(NCE_loss(pos_score, neg_score, self.neg_sample_size))


def edge_sampler(g, batch_size, neg_sample_size, edges = None,
                 shuffle = True, exclude_positive = True, return_false_neg = True):
    sampler = dgl.contrib.sampling.EdgeSampler(g, batch_size = batch_size, #int(g.number_of_edges()/10
                                               seed_edges = edges,
                                               neg_sample_size = neg_sample_size,
                                               negative_mode = 'tail',
                                               shuffle = shuffle,
                                               return_false_neg = return_false_neg)
    sampler = iter(sampler)
    return sampler #next(sampler) #return a list of all samples instead of just one

def get_src_nid(sub_g):
    src_nid, dst_nid = sub_g.all_edges(order='eid')
    # Get the node Ids in the parent graph.
    src_nid = sub_g.parent_nid[src_nid]
    dst_nid = sub_g.parent_nid[dst_nid]
    return src_nid, dst_nid

def score_func(sub_g, emb, sub_w, img_heads=None, img_tails=None, MLPs=None):
    # energy for positive edges
    # Read the node embeddings of the source nodes and destination nodes.
    src_nid, dst_nid = get_src_nid(sub_g)
    pos_heads = emb[src_nid]
    pos_tails = emb[dst_nid]
    if MLPs is None: #original
    # cosine similarity
      return torch.sum(pos_heads * sub_w * pos_tails, dim=1) #ensure emb has same norm
    else: #idea 2
    # combine (projected) image features with head and tail node features
      input_heads = torch.cat([img_heads, pos_heads], dim=-1)
      input_tails = torch.cat([img_tails, pos_tails], dim=-1)
      gate_scalar_heads = MLPs(input_heads) #[0]
      gate_scalar_tails = MLPs(input_tails) #[1]
      combined_heads = (gate_scalar_heads) * img_heads + (1 - gate_scalar_heads) * pos_heads #only use emb_head
      combined_tails = (gate_scalar_tails) * img_tails + (1 - gate_scalar_tails) * pos_tails #only use image_tails (as they are different)
      return torch.sum(combined_heads * sub_w * combined_tails, dim=1), gate_scalar_heads, gate_scalar_tails


def LPEvaluate_relation(emb, w, g, valid_eids, batch_size, neg_sample_size, h_dim, img_feat = None, txt_feat = None,
                        img_proj = None, txt_proj = None, img_MLP = None, txt_MLP = None, combined_MLP = None, gates=None):
    #df to store the aggregated results from MC runs
    df = pd.DataFrame(columns=['type', 'mrr', 'hits10', 'hits3', 'hits1'], dtype='float64')
    if gates == 1: #idea 3
      df['gate'] = ""
    elif gates == 'both':
      df['gate_img_m'] = ""
      df['gate_img_sd'] = ""
      df['gate_txt_m'] = ""
      df['gate_txt_sd'] = ""
    else: #idea2
      df['gate_head'] = ""
      df['gate_tail'] = ""
    #5 runs of eval to obtain mean and std: MRR, Hits10, Hits3, Hits1
    for i in range(5):
        #store per run results
        df_run = df.copy() #deep copy
        #mini-batch eval
        edge_batches = edge_sampler(g, batch_size, neg_sample_size, edges=valid_eids, shuffle=False) #no shuffling -> deterministic batches
        for batch in range(math.ceil(int(valid_eids.size()[0])/batch_size)+1):
            try:
                pos_g, neg_g = next(edge_batches)
                #print(pos_g.parent_eid)
            except StopIteration: #stopIteration flag by DGL
                break
            else:
                # get additional w relation
                pos_edge_type = g.edges[pos_g.parent_eid].data['type']
                sub_w_pos = w[pos_edge_type]

                # get the parent node id
                src_nid_pos, dst_nid_pos = get_src_nid(pos_g)

                if (img_feat is None) & (txt_feat is None):
                  pos_score = score_func(pos_g, emb, sub_w_pos)
                  sub_w_neg = torch.cat([sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1,h_dim)
                  neg_score = score_func(neg_g, emb, sub_w_neg).reshape(-1, neg_sample_size)

                elif (img_feat is not None) & (txt_feat is not None):

                  # Img features
                  img_head_pos = img_proj(img_feat[src_nid_pos])
                  img_tail_pos = img_proj(img_feat[dst_nid_pos])

                  txt_head_pos = txt_proj(txt_feat[src_nid_pos])
                  txt_tail_pos = txt_proj(txt_feat[dst_nid_pos])

                  txt_features = 0.5*txt_head_pos + 0.5*txt_tail_pos
                  img_features = 0.5*img_head_pos + 0.5*img_tail_pos #[batch_size, 100], take the average of head/tail features

                  # concat with DistMult
                  input_relations_img = torch.cat([img_features, sub_w_pos], dim=-1) #[batch_size, 2*100]
                  gate_scalar_img = img_MLP(input_relations_img)
                  img_sub_w_pos = (gate_scalar_img) * img_features + (1 - gate_scalar_img) * sub_w_pos

                  input_relations_txt = torch.cat([txt_features, sub_w_pos], dim=-1)
                  gate_scalar_txt = txt_MLP(input_relations_txt)
                  txt_sub_w_pos = (gate_scalar_txt) * txt_features + (1 - gate_scalar_txt) * sub_w_pos

                  input_relations = torch.cat([img_sub_w_pos, txt_sub_w_pos], dim=-1)
                  # gate_scalar = combined_MLP(input_relations)
                  combined_sub_w_pos = combined_MLP(input_relations)
                  # use a single gating scalar to select how much of projected image features vs. graph node embeddings to use
                  # combined_sub_w_pos = (1 - gate_scalar) * img_sub_w_pos + (gate_scalar) * txt_sub_w_pos
                  pos_score = score_func(pos_g, emb, combined_sub_w_pos)
                  # print("pos score shape: {}".format(pos_score.shape))

                  combined_sub_w_neg = torch.cat([combined_sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1, h_dim)
                  neg_score = score_func(neg_g, emb, combined_sub_w_neg).reshape(-1, neg_sample_size)
                  # print("neg score shape: {}".format(neg_score.shape))

                elif (txt_feat is None) or (img_feat is None):

                    if img_feat is not None:

                        img_head_pos = img_proj(img_feat[src_nid_pos])
                        img_tail_pos = img_proj(img_feat[dst_nid_pos])

                        img_features = 0.5*img_head_pos + 0.5*img_tail_pos #[batch_size, 100], take the average of head/tail features

                        # concat with DistMult
                        input_relations_img = torch.cat([img_features, sub_w_pos], dim=-1) #[batch_size, 2*100]
                        gate_scalar_img = img_MLP(input_relations_img)
                        img_sub_w_pos = (gate_scalar_img) * img_features + (1 - gate_scalar_img) * sub_w_pos

                        # use a single gating scalar to select how much of projected image features vs. graph node embeddings to use
                        # combined_sub_w_pos = (1 - gate_scalar) * img_sub_w_pos + (gate_scalar) * txt_sub_w_pos
                        pos_score = score_func(pos_g, emb, img_sub_w_pos)
                        # print("pos score shape: {}".format(pos_score.shape))

                        img_sub_w_neg = torch.cat([img_sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1, h_dim)
                        neg_score = score_func(neg_g, emb, img_sub_w_neg).reshape(-1, neg_sample_size)

                    else:

                        txt_head_pos = txt_proj(txt_feat[src_nid_pos])
                        txt_tail_pos = txt_proj(txt_feat[dst_nid_pos])

                        txt_features = 0.5*txt_head_pos + 0.5*txt_tail_pos #[batch_size, 100], take the average of head/tail features

                        # concat with DistMult
                        input_relations_txt = torch.cat([txt_features, sub_w_pos], dim=-1) #[batch_size, 2*100]
                        gate_scalar_txt = txt_MLP(input_relations_txt)
                        txt_sub_w_pos = (gate_scalar_txt) * txt_features + (1 - gate_scalar_txt) * sub_w_pos

                        # use a single gating scalar to select how much of projected image features vs. graph node embeddings to use
                        # combined_sub_w_pos = (1 - gate_scalar) * img_sub_w_pos + (gate_scalar) * txt_sub_w_pos
                        pos_score = score_func(pos_g, emb, txt_sub_w_pos)
                        # print("pos score shape: {}".format(pos_score.shape))

                        txt_sub_w_neg = torch.cat([img_sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1, h_dim)
                        neg_score = score_func(neg_g, emb, txt_sub_w_neg).reshape(-1, neg_sample_size)

                #filter out existing edges in the graph
                filter_bias = neg_g.edata['false_neg'].reshape(-1, neg_sample_size)
                pos_score = F.logsigmoid(pos_score)
                neg_score = F.logsigmoid(neg_score)
                neg_score -= filter_bias.float().cuda()
                pos_score = pos_score.unsqueeze(1)
                rankings = torch.sum(neg_score >= pos_score, dim=1) + 1

                #calculate mrr and hits
                rank = rankings.cpu().numpy()
                mrr = 1.0/rank
                hits10 = (rank <= 10).astype(int)
                hits3 = (rank <= 3).astype(int)
                hits1 = (rank <= 1).astype(int)

                ### output the raw rank (without averaging) => average per relation types (flag - sub_w_pos for relation type)

                df_batch = pd.DataFrame({'type':pos_edge_type.cpu().numpy(),
                                         'mrr': mrr, 'hits10': hits10, 'hits3': hits3, 'hits1': hits1,
                                         # 'gate_img_m': gate_scalar_img.detach().cpu().mean(),
                                         # 'gate_img_sd': gate_scalar_img.detach().cpu().std(),
                                         # 'gate_txt_m': gate_scalar_txt.detach().cpu().mean(),
                                         # 'gate_txt_sd': gate_scalar_txt.detach().cpu().std()
                                         })

                df_run = pd.concat((df_run, df_batch.groupby('type').mean().reset_index()))
        df = pd.concat((df, df_run.groupby('type').mean().reset_index()))
    return df

def LPEvaluate(emb, w, g, valid_eids, batch_size, neg_sample_size, h_dim):
    #5 runs of eval to obtain mean and std: MRR, Hits10, Hits3, Hits1
    MC_results = np.zeros((5,4))
    for i in range(5):
        #mini-batch eval
        mrr_list, hits_list = [],[]
        edge_batches = edge_sampler(g, batch_size, neg_sample_size,edges=valid_eids,shuffle=False) #no shuffling -> deterministic batches
        for batch in range(math.ceil(int(valid_eids.size()[0])/batch_size)+1):
            try:
                pos_g, neg_g = next(edge_batches)
                #print(pos_g.parent_eid)
            except StopIteration: #stopIteration flag by DGL
                break
            else:
                # get additional w relation
                pos_edge_type = g.edges[pos_g.parent_eid].data['type']
                sub_w_pos = w[pos_edge_type]
                pos_score = score_func(pos_g, emb, sub_w_pos)

                sub_w_neg = torch.cat([sub_w_pos.repeat_interleave(neg_sample_size)]).reshape(-1,h_dim)
                neg_score = score_func(neg_g, emb, sub_w_neg).reshape(-1, neg_sample_size)
                filter_bias = neg_g.edata['false_neg'].reshape(-1, neg_sample_size)

                pos_score = F.logsigmoid(pos_score)
                neg_score = F.logsigmoid(neg_score)
                neg_score -= filter_bias.float().cuda()
                pos_score = pos_score.unsqueeze(1)
                rankings = torch.sum(neg_score >= pos_score, dim=1) + 1

                #calculate mrr and hits
                rank = rankings.cpu().numpy()

                mrr = np.mean(1.0/rank)
                hits10 = np.mean(rank <= 10)
                hits3 = np.mean(rank <= 3)
                hits1 = np.mean(rank <= 1)

                mrr_list.append(mrr)
                hits_list.append([hits10,hits3,hits1])

        MC_results[i,0] = np.mean(np.array(mrr_list))
        MC_results[i,1:] = np.mean(np.array(hits_list),axis=0)
    return MC_results

# define graph network layers
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):

        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, img_h, txt_h):

      emb = self.embedding(g.ndata['node_id'])
      return emb

class GATLayer(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, use_emb = True,
                img_feat = None, txt_feat = None, concat = False, MLP = True):
        super(GATLayer, self).__init__()
        # equation (1)
        self.num_nodes = num_nodes
        self.embedding = EmbeddingLayer(num_nodes, in_dim)
        self.MLP = MLP
        self.in_dim = in_dim
        self.concat = concat
        self.use_emb = use_emb


        if (img_feat is not None): # if there are image features available
          if concat: # if we concat image rep. with no rep. without gating
              self.img_proj = nn.Linear(img_feat.size(1) + in_dim, hidden_dim, bias = False) #[300+100,100] projection

          # MLP layer in GAT instead of DistMult head, obsolote code
          elif MLP: # gating, to learn one scalar weight
              self.img_proj = nn.Linear(img_feat.size(1), hidden_dim, bias = False)
              self.img_MLP = nn.Sequential(
                          nn.Linear(2*hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, 1),
                          nn.Sigmoid()
                        )
        elif (img_feat is None):

            self.img_proj = None
            self.img_MLP = None

        if (txt_feat is not None): # if there are image features available
          if concat: # if we concat image rep. with no rep. without gating
              self.txt_proj = nn.Linear(txt_feat.size(1) + in_dim, hidden_dim, bias = False) #[300+100,100] projection

          # MLP layer in GAT instead of DistMult head, obsolote code
          elif MLP: # gating, to learn one scalar weight
              self.txt_proj = nn.Linear(txt_feat.size(1), hidden_dim, bias = False)
              self.txt_MLP = nn.Sequential(
                          nn.Linear(2*hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, 1),
                          nn.Sigmoid()
                        )

        elif (txt_feat is None):

            self.txt_proj = None
            self.txt_MLP = None

        if (img_feat is not None) & (txt_feat is not None):

            self.combined_MLP = nn.Sequential(
                      nn.Linear(2*hidden_dim, hidden_dim)
                    )
        else:
            self.combined_MLP = None

        if self.use_emb:
          self.fc = nn.Linear(hidden_dim, hidden_dim, bias = False)
        else:
          self.fc = nn.Linear(2 * hidden_dim, hidden_dim, bias = False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1, bias = False)
        self.use_emb = use_emb

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim = 1)
        a = self.attn_fc(z2)

        return {'e' : F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z' : edges.src['z'], 'e' : edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def forward(self, g, img_h, txt_h):

        # equation (1)
        if self.use_emb:
            h = self.embedding(g, img_h, txt_h)
            embed = h.detach().clone()

        else:
            h = img_h

        if self.concat: # if we concat image rep. with no rep. without gating
            embed = self.embedding(g, img_h, txt_h)

            img_feat_concat = torch.cat([embed, img_h], dim = 1)
            img_h = self.img_proj(img_feat_concat)

            txt_feat_concat = torch.cat([embed, txt_h], dim = 1)
            txt_h = self.txt_proj(txt_feat_concat)

        elif self.MLP: # gating, to learn one scalar weight

            if self.img_proj is not None:
                img_features = self.img_proj(img_h)  #project visual/textual feat to h-dim
                gate_scalar_img = self.img_MLP(torch.cat([embed, img_features], dim = 1))
                img_h = (gate_scalar_img) * img_features + (1 - gate_scalar_img) * embed

            if self.txt_proj is not None:
                txt_features = self.txt_proj(txt_h)  #project visual/textual feat to h-dim
                gate_scalar_txt = self.txt_MLP(torch.cat([embed, txt_features], dim = 1))
                txt_h = (gate_scalar_txt) * txt_features + (1 - gate_scalar_txt) * embed

            if (self.combined_MLP is not None):
                gate_scalar = self.combined_MLP(torch.cat([img_h, txt_h], dim = 1))
                h = gate_scalar * img_h + (1 - gate_scalar) * txt_h

        z = self.fc(h.float())
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_heads,
                 img_feat, txt_feat, concat, MLP, merge = 'cat', use_emb = True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.use_emb = use_emb
        for i in range(num_heads):
            self.heads.append(GATLayer(num_nodes, in_dim, hidden_dim, use_emb = use_emb,
                                       img_feat = img_feat, txt_feat = txt_feat, concat = concat, MLP = MLP))

        self.merge = merge

    def forward(self, g, img_h, txt_h):
        head_outs = [attn_head(g, img_h, txt_h) for attn_head in self.heads]

        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            head_outs_ = torch.cat(head_outs, dim = 1)
            return head_outs_
        else:
            # merge using average
            head_outs_ = torch.mean(torch.stack(head_outs))
            return head_outs_


class GAT(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_heads,
                 img_feat, txt_feat, concat, MLP, use_emb = True):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(num_nodes, in_dim, hidden_dim, num_heads, img_feat, txt_feat, concat, MLP, use_emb = use_emb)
        # Be aware that the input dimension is hidden_dim*num_heads since
        #   multiple head outputs are concatenated together. Also, only
        #   one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(num_nodes, hidden_dim * num_heads, hidden_dim, 1, img_feat, txt_feat,
                                        concat = False, MLP = False, use_emb = False)

    def forward(self, g, img_h, txt_h):

        h = self.layer1(g, img_h, txt_h)
        h = F.elu(h)
        # pass prev layer output as image feat argument in layer 2
        h = self.layer2(g, h, txt_h)

        return h

def model_train(train_g, valid_g, test_g, g, valid_eids, test_eids, img_features, txt_features,
                in_feats, n_hidden, num_heads, concat, MLP, num_rels, weight_decay=5e-4, n_epochs=100,
                batch_size=1e4, neg_sample_size=100, lr=2e-3,save_file='./GAT_base.pt',
                use_cuda = True, evaluate_every_ = 100):

    # create GAT model
    num_nodes = train_g.number_of_nodes()

    GAT_model = GAT(num_nodes, in_dim = in_feats, img_feat = img_features,
                    txt_feat = txt_features, hidden_dim = n_hidden, num_heads = num_heads,
                    concat = concat, MLP = MLP, use_emb = True)

    # print(GAT_model)

    # create DistMult head
    model = LinkPredict_Dist(GAT_model, n_hidden, num_rels, batch_size, neg_sample_size, use_cuda,
                             img_feat = img_features, txt_feat = txt_features)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model params: {}".format(pytorch_total_params))

    #send to GPU
    if use_cuda:
      model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize graph
    old_mrr = 0

    for epoch in range(n_epochs):
        model.train()
        #mini-batch training
        edge_batches = edge_sampler(train_g, batch_size, neg_sample_size,edges=None,exclude_positive=True,return_false_neg=True)
        for batch in range(math.ceil(int(train_g.number_of_edges()/batch_size))+1):
        #for pos_g, neg_g in edge_sampler(train_g,batch_size, neg_sample_size,replacement=False):
            try:
              pos_g, neg_g = next(edge_batches)
            except StopIteration: #stopIteration flag by DGL
              break
            else:
              embed, _ = model(train_g, img_features = img_features, txt_features = txt_features)
              loss = model.get_loss(embed, pos_g, neg_g, img_features, txt_features)

              if batch % evaluate_every_ == 0:
                print('batch: ', batch)
                print('loss: ', loss)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

        #eval
        GAT_model.eval()
        with torch.no_grad():

          emb, w = model(train_g, img_features, txt_features)

          if use_cuda:
            emb.cuda()

          MC_results = LPEvaluate(emb, w, g, valid_eids, batch_size, neg_sample_size, n_hidden)
          mrr_m, hits10_m, hits3_m, hits1_m = np.mean(MC_results, axis=0)

        print("Epoch {:05d} | MRR_mean {:.4f} | Hits10_mean {:.4f} | Hits3_mean {:.4f} | Hits1_mean {:.4f}".format(epoch, mrr_m, hits10_m, hits3_m, hits1_m))

        #  save model of best mrr
        if mrr_m > old_mrr:
            old_mrr = mrr_m
            torch.save(model.state_dict(), save_file)
            print('Model Saved')
            early_stop = 0
        else:
            #early stopping after no improvements of 10 epochs
            early_stop += 1
            if early_stop == 10:
                break

    # Let's save the trained node embeddings.
    # model.load_state_dict(torch.load(save_file))
    print("Evaluating model on validation set.")
    model.eval()
    emb, w = model(g, img_features, txt_features)
    valid_results = LPEvaluate(emb, w, g, valid_eids, batch_size, neg_sample_size, n_hidden)
    mrr_m, hits10_m, hits3_m, hits1_m = np.mean(valid_results, axis=0)
    print("Validation set results: MRR {:.4f}| Hits@10 {:.4f} | Hits@3 {:.4f} | Hits@1 (accuracy) {:.4f} |".format(mrr_m, hits10_m, hits3_m, hits1_m))

    print("Evaluating model on test set.")
    test_results = LPEvaluate(emb, w, g, test_eids, batch_size, neg_sample_size, n_hidden)
    mrr_m, hits10_m, hits3_m, hits1_m = np.mean(test_results, axis=0)
    print("Test set results: MRR {:.4f}| Hits@10 {:.4f} | Hits@3 {:.4f} | Hits@1 (accuracy) {:.4f} |".format(mrr_m, hits10_m, hits3_m, hits1_m))

    return model
