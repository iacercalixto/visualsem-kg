# imports
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import math
import time
import graph_utils
import numpy as np
from graph_utils import build_graph, get_train_val_test_masks
import argparse
import os

# parse arguments
parser = argparse.ArgumentParser(description='Train graph attention network')

parser.add_argument('--num_graph_nodes', type=int, default=101244, metavar='N',
                    help='number of nodes in the graph dataset')
parser.add_argument('--num_rels', type=str, default=13, metavar='N',
                    help='number of relations in the graph dataset')
parser.add_argument('--dataset_name', type=str, default='VisualSem', metavar='N',
                    help='dataset name')
parser.add_argument('--data_location', type=str, default='/data/visualsem/', metavar='N',
                    help='data root directory')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')

parser.add_argument('--no_concat', default=False, action='store_false',
                    help='dont concatenate')
parser.add_argument('--concat', default=True, action='store_true',
                    help='concatenate')
parser.add_argument('--use_mlp', type=str, default='no', metavar='N',
                    help='use MLP')
parser.add_argument('--batch_size', type=int, default=2e4, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--neg_sample_size', type=int, default=100, metavar='N',
                    help='negative sample size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=int, default=0, metavar='N',
                    help='weight decay')
parser.add_argument('--img_feature_path', type=str, default=None, metavar='N',
                    help='path to image feature tensor')
parser.add_argument('--txt_feature_path', type=str, default=None, metavar='N',
                    help='path to text feature tensor')
parser.add_argument('--num_input_feats', type=int, default=100, metavar='N',
                    help='input dimension')
parser.add_argument('--num_hidden', type=int, default=100, metavar='N',
                    help='hidden dimension')
parser.add_argument('--num_attn_heads', type=int, default=2, metavar='N',
                    help='number of attention heads')
parser.add_argument('--save_location', type=str, default='/model.pt', metavar='N',
                    help='save location for dumped model')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--evaluate_interval', type=int, default=100, metavar='N',
                    help='evaluate every n batches')
args = parser.parse_args()

# set PyTorch backend for DGL, set up CUDA
dgl.load_backend('pytorch')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = False
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(0)
  use_cuda = True

# set random seed
np.random.seed(0)
dgl.random.seed(0)

#define training graph
num_nodes = args.num_graph_nodes
train_data = np.load(os.path.join(args.data_location, "train.npy"))
valid_data = np.load(os.path.join(args.data_location, "valid.npy"))
test_data = np.load(os.path.join(args.data_location, "test.npy"))
num_rels = args.num_rels
use_cuda = args.cuda
print("num_rels: {}".format(num_rels))
print("cuda: {}".format(use_cuda))

g, train_eids, valid_eids, test_eids, num_rels = model.build_g_from_data(args.dataset_name, args.data_location, args.num_graph_nodes, args.num_rels, use_cuda = use_cuda)
g.ndata['node_id'] = torch.arange(g.number_of_nodes())
g.readonly()

# define training graph
train_g = g.edge_subgraph(train_eids, preserve_nodes = True)
train_g.ndata['node_id'] = g.ndata['node_id']
train_g.edata['type'] = torch.tensor(train_data[:, 1])
if use_cuda:
    train_g.to(torch.device('cuda:0'))

# define validation graph
valid_g = g.edge_subgraph(valid_eids, preserve_nodes = True)
valid_g.ndata['node_id'] = g.ndata['node_id']
valid_g.edata['type'] = torch.tensor(valid_data[:, 1])
if use_cuda:
    valid_g.to(torch.device('cuda:0'))

# define test graph
test_g = g.edge_subgraph(test_eids, preserve_nodes = True)
test_g.ndata['node_id'] = g.ndata['node_id']
test_g.edata['type'] = torch.tensor(test_data[:, 1])
if use_cuda:
    test_g.to(torch.device('cuda:0'))

# model hyperparameters
in_feats =args.num_input_feats
n_hidden = args.num_hidden
num_heads = args.num_attn_heads

# load feature set
img_features = None
txt_features = None

if args.img_feature_path != None:
    img_features = torch.load(args.img_feature_path).float().cuda()

if args.txt_feature_path != None:
    txt_features = torch.load(args.txt_feature_path).float().cuda()

#define training graph
train_g = g.edge_subgraph(train_eids, preserve_nodes = True)
train_g.ndata['node_id'] = g.ndata['node_id']
train_g.edata['type'] = g.edges[train_eids].data['type']
train_g.to(torch.device('cuda:0'))

# send g and valid eid, test eid to device
g.to(torch.device('cuda:0'))

train_g.readonly()
# train graph attention network
if args.use_mlp == 'yes':
  use_mlp = True
else:
  use_mlp = False

model_MLP = modeling.model_train(train_g = train_g, valid_g = valid_g, test_g = test_g, g = g, valid_eids = valid_eids, test_eids = test_eids,
                                img_features = img_features, txt_features = txt_features, in_feats = in_feats, n_hidden = n_hidden, num_heads = num_heads,
                                concat = False, MLP = use_mlp, num_rels = args.num_rels, weight_decay = args.weight_decay,
                                n_epochs = args.epochs, batch_size = args.batch_size, neg_sample_size = args.neg_sample_size, lr = args.lr,
                                save_file = args.save_location, use_cuda = args.cuda, evaluate_every_ = args.evaluate_interval)
