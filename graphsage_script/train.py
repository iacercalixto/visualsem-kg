#/usr/bin/python

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
from model import model_train

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph

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

#Helper function for data consolidation
def double_map(df,edge_map_y,edge_ids):
	'''
	df: train/valid/test, only contains r_id_reset
	edge_map_y: Yash's mapping of r_id to r_id_reset
	edge_ids: Teresa's mapping of r_id to graph_id
	r_id_reset => r_id => graph_id 
	'''
	df = pd.merge(df,edge_map_y[['r_id','r_id_reset']], how='left', on='r_id_reset')
	df = pd.merge(df,edge_ids, how='left', on='r_id')
	return df

def data_split(edge_ids):
	#original data split
	train_data = pd.read_csv('80_10_10_split/train2id.txt',delimiter=' ') 
	valid_data =pd.read_csv('80_10_10_split/valid2id.txt',delimiter=' ') 
	test_data = pd.read_csv('80_10_10_split/test2id.txt',delimiter=' ')
	#original edge ID mapping
	edge_map_y = pd.read_csv('rel_map.csv',delimiter='|')
	#graph edge ID mapping
	train_data = double_map(train_data,edge_map_y,edge_ids)
	valid_data = double_map(valid_data,edge_map_y,edge_ids)
	test_data = double_map(test_data,edge_map_y,edge_ids)
	#convert edge ID to tensors
	train_eids = torch.tensor(train_data.graph_r_id.values).long()
	valid_eids = torch.tensor(valid_data.graph_r_id.values).long()
	test_eids = torch.tensor(test_data.graph_r_id.values).long()

	return train_eids, valid_eids, test_eids

def get_features():
	#read into reference dict (node, edge numbering)
	ref_dict = pickle.load(open("./ref_dict.pkl",'rb'))
	#create edge id df: to be mapped with data split
	edge_ids = pd.DataFrame(list(ref_dict['edge_id'].items()),
	                      columns=['r_id','graph_r_id'])
	#build node_id df
	df_node_id = pd.DataFrame(list(ref_dict['node_num'].items()),
	                      columns=['node_id','graph_id'])

	#read txt features
	df_gloss = h5py.File('text_features_multi.h5', 'r')
	df_nid = df_gloss.get('bnids')
	df_emb = df_gloss.get('feature_means')
	#df_vis in pandas
	df_gloss = pd.DataFrame(data=np.array(df_nid),index=np.arange(np.array(df_nid).shape[0]),columns=['node_id'])
	df_gloss['gloss_emb'] = np.array(df_emb).tolist()
	#join with node_id, and then sort by graph_id so that we assign the same text feature to the same node
	df_gloss = df_gloss.merge(df_node_id, on='node_id')
	df_gloss = df_gloss.sort_values(by=['graph_id'])

	#read img features
	df_vis = h5py.File('visualsem_features.h5', 'r')
	vis_nid = df_vis.get('bnids')
	vis_emb = df_vis.get('feature_means')
	#df_vis in pandas
	df_vis = pd.DataFrame(data=np.array(vis_nid),index=np.arange(np.array(vis_nid).shape[0]),columns=['node_id'])
	df_vis['vis_emb'] = np.array(vis_emb).tolist()
	#join with node_id, and then sort by graph_id so that we assign the same text feature to the same node
	df_vis = df_vis.merge(df_node_id, on='node_id')
	df_vis = df_vis.sort_values(by=['graph_id'])

	return torch.tensor(df_vis.vis_emb).float().cuda(), torch.tensor(df_gloss.gloss_emb).float().cuda(), edge_ids

if __name__=="__main__":
	p = argparse.ArgumentParser()
	p.add_argument('--in_feats', type=int, default=100)
	p.add_argument('--n_hidden', type=int, default=100)
	p.add_argument('--n_layers', type=int, default=2)
	p.add_argument('--dropout', type=float, default=0.5)
	p.add_argument('--aggregator_type', type=str, default='gcn')
	p.add_argument('--reg', type=float, default=0) #weight decay
	p.add_argument('--lr',type=float, default=1e-3) #learning rate
	p.add_argument('--n_epochs', type=int, default=100)
	p.add_argument('--batch_size', type=int, default=1000)
	p.add_argument('--neg_sample_size', type=int, default=1000)
	p.add_argument('--mode', type=str, default='both') #features: 'img', 'gl', 'both', 'node'
	p.add_argument('--gate', type=str, default='both') #gating mechanism: 'graph' (node gating), 'dist' (edge gating), 'both'
	args = p.parse_args()
	#get features
	img_features, gl_features, edge_ids = get_features()
	print("features done")
	#get train/valid/test edge split
	train_eids, valid_eids, test_eids = data_split(edge_ids)
	print("data split done")
	#build DGL graph
	join_df = pickle.load(open( "./join_df.pkl", "rb" )) 
	g = build_graph(join_df, num_nodes=101244,directed=True,edge_feat=True) #use edge type
	g.ndata['node_id'] = torch.arange(g.number_of_nodes())
	g.readonly()
	
	#build training graph
	train_g = g.edge_subgraph(train_eids, preserve_nodes=True)
	train_g.ndata['node_id'] = g.ndata['node_id']
	train_g.edata['type'] = g.edges[train_eids].data['type'] 
	train_g.to(torch.device('cuda:0'))
	g.to(torch.device('cuda:0'))
	print("graph ready, start training...")

	#train model
	save_file = 'mode_' + args.mode + '_gate_' + args.gate +'.pt'
	print(save_file)
	if args.mode == 'img' or args.mode == 'node':
		gl_features = None
	if args.mode == 'gl' or args.mode == 'node':
		img_features = None
	best_mrr = model_train(train_g, g, valid_eids, test_eids, args.in_feats,args.n_hidden,
		args.n_layers,args.dropout,args.aggregator_type,num_rel=13,
		img_feat=img_features, gl_feat=gl_features, mode=args.mode,gate=args.gate,
		weight_decay=args.reg, n_epochs=args.n_epochs, batch_size=args.batch_size,
		neg_sample_size=args.neg_sample_size,lr=args.lr, 
		save_file=save_file,
		use_cuda=True, pretrain=False, pre_file=None)

