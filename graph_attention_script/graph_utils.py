import torch
from collections import defaultdict
import time
#from utils import get_metrics_dict
import subprocess

import dgl
from dgl import DGLGraph

# Load Pytorch as backend
dgl.load_backend('pytorch')

def get_train_val_test_masks(num_nodes, train_size=0.8, val_size=0.1):
    torch.manual_seed(57)
    rand = torch.rand(num_nodes)
    train_mask = rand < train_size
    val_mask = (train_size <= rand) & (rand < (train_size + val_size))
    test_mask = torch.ones_like(train_mask) ^ train_mask ^ val_mask
    return train_mask, val_mask, test_mask

def build_graph(graph_df,num_nodes, directed=False,edge_feat=False): #added encode edge feature
    g = dgl.DGLGraph()
    #g.add_nodes(graph_df.shape[0])
    g.add_nodes(num_nodes)
    pairs_edges = set()
    for _, row in graph_df.iterrows():
        node_i = row["node_id"]
        to_nodes = row["to_nodes"]
        edge_type = row["edge_types"]
        pairs_edges.update(list(zip(len(to_nodes) * [node_i], to_nodes, edge_type)))

    src, dst, feat = list(zip(*(list(pairs_edges))))
    if not directed:
        # add reversed pairs of nodes
        pairs_edges.update(list(zip(dst, src)))
        # add edges to graph: 
        # between nodes a and b there are edges a->b and a<-b if they are connected at least once in data
        src, dst = list(zip(*(list(pairs_edges))))
    g.add_edges(src, dst)
    if edge_feat == True:
        print('adding edge features')
        g.edata['type'] = torch.tensor(feat)
    return g

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def predict(model, graph, threshold=0.5):
    return (torch.exp(model(graph)) > threshold).float()

def train_GraphSAGE(model, criterion, optimizer, scheduler, 
                    device, out_path, options, n_epochs,
                    graph, labels, train_mask, val_mask, return_model=False):
    full_metrics_dict = {"train": defaultdict(list), "val": defaultdict(list)}
    full_metrics_dict["val"]["best_f1_micro"] = 0
    
    time_start = time.time()
    
    for epoch in range(n_epochs):
        # TRAIN
        model.train()
        optimizer.zero_grad()
        # forward
        logits = model(graph)
        loss = criterion(logits[train_mask], labels[train_mask])
        # backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        
#         # EVAL
        with torch.no_grad():
            loss_val = criterion(logits[val_mask], labels[val_mask])
            y_pred = predict(model, graph)
            cur_metrics_dict = get_metrics_dict(labels[val_mask].cpu(), y_pred[val_mask].cpu())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4} | Validation f1_micro {cur_metrics_dict['f1_micro']}")
        
        # SAVE
        # to metrics dict
        full_metrics_dict["train"]["loss_hist"].append(loss.item())
        full_metrics_dict["val"]["loss_hist"].append(loss_val.item())
        full_metrics_dict["val"]["metrics_epochwise"].append(cur_metrics_dict)
        
        if cur_metrics_dict["f1_micro"] > full_metrics_dict["val"]["best_f1_micro"]:
            full_metrics_dict["val"]["best_f1_micro"] = cur_metrics_dict["f1_micro"]
#             torch.save({
#                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
#                 'full_metrics': full_metrics_dict,
#                 'options': options,
#                 }, f'{PATH_TO_MODELS}{out_path}/best.pth')

      
    time_diff = time.time() - time_start
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_diff // 60, time_diff % 60))
    print('Best val f1_micro: {:.4f} \n'.format(full_metrics_dict["val"]["best_f1_micro"]))
    
#     torch.save({
#                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
#                 'full_metrics': full_metrics_dict,
#                 'options': options,
#                 }, f'{PATH_TO_MODELS}{out_path}/net_epoch_{epoch}.pth')
#     torch.save({"args": options, "full_metrics":full_metrics_dict}, f'{PATH_TO_MODELS}{out_path}.pth')
    if return_model:
        return model, full_metrics_dict
    return full_metrics_dict