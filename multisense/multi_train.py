#/usr/bin/python
import csv
import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
from PIL import Image
import numpy
from collections import defaultdict
import h5py
import numpy as np
import pickle
import pandas as pd

#BERT
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_query_trans_dict(file, lang):
    '''
    Create the Query->Translation lookup dictionary from the file `gold_german_query_classes.csv`
    (src: https://github.com/spandanagella/multisense/blob/master/eval_verb_accuracy.py)
    '''
    query_trans_dict = dict()

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query = row['query'].strip()
            if lang == 'en':
                query_trans_dict[query] = row['verb'].strip()
            elif lang == 'de':
                query_trans_dict[query] = row['verb_translation'].strip()

    return query_trans_dict

class ImageQueryDataset(Dataset):
    '''
    Dataset class for image + query + nodes_hidden_state
    '''
    def __init__(self, in_file, ref_dict,verb_map, query_dict, tokenizer, max_len, node_feat=None, node_map=None):
        super(ImageQueryDataset, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.ref_dict = pickle.load(open(ref_dict,"rb")) #one-to-one mapping for image (file_path, file_name, verb)
        #map unique verb to integer labels
        self.verb_map = pickle.load(open(verb_map,"rb"))  #verb_map is an unique dictionary determined by training set ref_dict
        self.labels = [self.verb_map[verb] for verb in self.ref_dict['verbs']]
        #look-up table with query as key and source verb as value
        self.query_trans_dict = load_query_trans_dict(query_dict, 'en')
        self.tokenizer = tokenizer #BERT tokenizer
        self.max_len = max_len #Default 128
        #additional node hidden state
        if node_feat is not None:
            self.node_feat = torch.load(node_feat) 
            #mapping of retrieved top-1 nodes to graph_id
            self.node_df = pd.read_pickle(node_map) 
        else:
            self.node_feat = None
 
    def __getitem__(self, index):
        image_feat = self.file['global_features'][index]  #ResNet152 embedding
        name = self.ref_dict['names'][index] #image name
        query = name.split("__")[0].replace("_", " ") #image query
        source_verb = self.query_trans_dict[query] #image source verb
        query_prepend = source_verb + ' ' + query #prepend source verb to the query 
        encoding = self.tokenizer.encode_plus(
          query_prepend,
          add_special_tokens=True,
          max_length=self.max_len,
          truncation=True,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        target = self.labels[index] #the target verb in German, mapped to integer
        #additional node features
        if self.node_feat is not None:
          graph_id = self.node_df[self.node_df['query']==query]['graph_id'].to_numpy()[0]
          node_feat = self.node_feat[graph_id]
          return {
          'image_feat': image_feat,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long),
          'node_feat': node_feat
           }
        else:
          return {
          'image_feat': image_feat,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
          }

    def __len__(self):
        return self.file['global_features'].shape[0]

class Multimodel(nn.Module):
    '''
    Model class for Multimodal Verb Sense Disambiguation
    [Textual query => BERT text feature; Image features; Node hidden state] => predict translated verb
    '''
    def __init__(self, n_classes, n_hidden=None, node_flag=False, nonlinearity=False, drop_prob=0, projection=False, proj_dim=100):
        super(Multimodel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=drop_prob)
        self.n_hidden = n_hidden
        self.nonlinear = nonlinearity
        self.projection = projection 
        if node_flag:
            node_dim = 100 #dim of hidden states
        else:
            node_dim = 0
        ## two-layer
        if self.n_hidden is not None:
            if self.projection: #first project both img & text features to 100 dim, then concat w/ node hidden state (100-dim)
                self.proj_img = nn.Linear(2048, proj_dim)
                self.proj_txt = nn.Linear(self.bert.config.hidden_size , proj_dim)
                self.hidden = nn.Linear(2*proj_dim + node_dim, n_hidden) 
            else: #one giant projection layer on concatenated features
                self.hidden = nn.Linear(self.bert.config.hidden_size + 2048 + node_dim, n_hidden) 
            self.out = nn.Linear(n_hidden, n_classes) #output layer
        ## one-layer
        else:
            self.out = nn.Linear(self.bert.config.hidden_size + 2048 + node_dim, n_classes) 

    def forward(self, input_ids, attention_mask, image_features, node_features):
        outputs = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask)
        sequence_output = outputs[0]
        textual_features = sequence_output[:,1,:] #[batch_size, seq_length, hidden_dim], #index 1 correspond to the prepended English verb embedding
        if node_features is not None:
            if self.projection: 
                img_projected = self.proj_img(image_features)
                txt_projected = self.proj_txt(textual_features)
                features = torch.cat([txt_projected, img_projected, node_features],dim=-1)
            else:
                features = torch.cat([textual_features, image_features, node_features],dim=-1)
        else:
            if self.projection:
                img_projected = self.proj_img(image_features)
                txt_projected = self.proj_txt(textual_features)
                features = torch.cat([txt_projected, img_projected],dim=-1)
            else:
                features = torch.cat([textual_features, image_features],dim=-1)
        if self.n_hidden is not None:   
            h = self.hidden(features)
            h = self.drop(h)
            if self.nonlinear:
                h = F.relu(h)
        else:
            h = features
        return self.out(h)

def train_epoch(model,data_loader,loss_fn,optimizer,device,n_examples,node_flag=False):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        image_feat = d["image_feat"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        if node_flag: #add node hidden state
            node_feat = d["node_feat"].to(device)
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              image_features=image_feat,
              node_features=node_feat
            )
        else:
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,
              image_features=image_feat, node_features=None)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples, node_flag=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
          image_feat = d["image_feat"].to(device)
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)
          if node_flag: #add node features
              node_feat = d["node_feat"].to(device)
              outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=image_feat,
                node_features=node_feat
              )
          else:
              outputs = model(input_ids=input_ids,attention_mask=attention_mask,
              image_features=image_feat, node_features=None)
          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs, targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--decay', type=float, default=0)
    p.add_argument('--node', action='store_true', default=False)
    p.add_argument('--node_file', type=str, default="/content/drive/My Drive/graph/nyu_multimodal_kb/NER/graph_emb_node.t")
    p.add_argument('--num_layer', type=int, default=2)
    p.add_argument('--nonlinear', action='store_true', default=False)
    p.add_argument('--dropout', type=float, default=0)
    p.add_argument('--projection', action='store_true', default=False)
    p.add_argument('--model_file', type=str, default="./multisense_model_weights/plustest/baseline_lr0.0005_2.bin")
    p.add_argument('--subset', type=float, default=1) #determine whether subset training samples for low-resource regime
    p.add_argument('--proj_dim', type=int, default=100) #specify the projection dimension (raw feature -> projected feature)
    p.add_argument('--hidden_dim', type=int, default=128) #specify the projection dimension (concatenated projected features -> hidden layer)

    
    args = p.parse_args()
    ##BERT params
    PRE_TRAINED_MODEL_NAME = 'bert-large-cased'  
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    ##create data loaders
    if args.node:
        train_dataset = ImageQueryDataset(in_file='features_per_image_train_german.h5',ref_dict='train_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128,
                                 node_feat=args.node_file, node_map="query_nodes.pkl")  
        valid_dataset = ImageQueryDataset(in_file='features_per_image_val_german.h5',ref_dict='val_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128,
                                 node_feat=args.node_file, node_map="query_nodes.pkl")      
        test_dataset = ImageQueryDataset(in_file='features_per_image_test_german.h5',ref_dict='test_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128,
                                 node_feat=args.node_file, node_map="query_nodes.pkl")  
    else:
        train_dataset = ImageQueryDataset(in_file='features_per_image_train_german.h5',ref_dict='train_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128)
        valid_dataset = ImageQueryDataset(in_file='features_per_image_val_german.h5',ref_dict='val_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128)
        test_dataset = ImageQueryDataset(in_file='features_per_image_test_german.h5',ref_dict='test_german.pkl',
                                 verb_map = "verb_map.pkl", query_dict='gold_german_query_classes.csv',
                                 tokenizer=tokenizer, max_len=128)
    #create training data loaders
    if args.subset == 1:
      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers = 1, shuffle=True)
    else: #subset training set for low-resource scenario
      subset_idx = list(range(0, len(train_dataset), int(1/args.subset)))
      trainsub = torch.utils.data.Subset(train_dataset, subset_idx)
      train_loader = torch.utils.data.DataLoader(trainsub, batch_size=args.batch_size,num_workers = 1, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,num_workers = 1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,num_workers = 1)

    ##model
    num_verbs = len(train_dataset.verb_map)
    print(num_verbs)
    if args.num_layer == 2:
        model = Multimodel(num_verbs, args.hidden_dim, args.node, args.nonlinear, args.dropout, args.projection, args.proj_dim)
    else: 
        model = Multimodel(num_verbs, None, args.node, False, 0)
    for param in model.bert.parameters():
        param.requires_grad = False #freeze BERT
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(pytorch_total_params) 
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    loss_fn = nn.CrossEntropyLoss().to(device)

    ##training
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(args.epochs):
      if epoch % 1 == 0:
          print(f'Epoch {epoch + 1}/{args.epochs}')
          print('-' * 10)
          train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            int(len(train_dataset)*args.subset),
            args.node
          )
          print(f'Train loss {train_loss} accuracy {train_acc}')
          val_acc, val_loss = eval_model(
            model,
            valid_loader,
            loss_fn,
            device,
            len(valid_dataset),
            args.node
          )
          print(f'Val   loss {val_loss} accuracy {val_acc}')
          print()
          history['train_acc'].append(train_acc)
          history['train_loss'].append(train_loss)
          history['val_acc'].append(val_acc)
          history['val_loss'].append(val_loss)
          if val_acc > best_accuracy:
            if args.node:
              model_path = './multisense_model_weights/plustest/'+ args.node_file.split('/')[-1][:-2] +'.bin'
              torch.save(model.state_dict(), model_path)
            else:
              model_path = './multisense_model_weights/plustest/baseline_lr' + str(args.lr) + '_' + str(args.num_layer) + '.bin'
              torch.save(model.state_dict(), model_path)
            best_accuracy = val_acc
    #eval on test
    model.load_state_dict(torch.load(model_path))
    test_acc, test_loss = eval_model(
            model,
            test_loader,
            loss_fn,
            device,
            len(test_dataset),
            args.node
          )
    print(f"test acc: {test_acc}")
    print(f"test loss: {test_loss}")
    #save history
    f_file=open("./multisense_model_weights/model_results.txt", "a+")
    f_file.write(f"\n")
    if args.node:
      f_file.write("Setting: " + args.node_file.split('/')[-1][:-2])
    else:
      f_file.write(f"Setting: Baseline with lr {args.lr}" )
    f_file.write(f"\n")
    f_file.write(f"Result (Best Validation):  {best_accuracy}")
    f_file.write(f"\n")
    f_file.write(f"Result (Best Test):  {test_acc}")
    f_file.close()
