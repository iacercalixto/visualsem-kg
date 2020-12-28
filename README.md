# visualsem-kg
Representation learning for VisualSem knowledge graph.

### Knowledge Representation Learning
1. Requirements
```
- Python 3.6+
- PyTorch 1.1.0+
- dgl-cu100 0.4.3
```

2. Data
- Graph:
  - Dataframe storing graph nodes, its neighbors, and their relation types: ```join_df.pkl``` (download [here](https://drive.google.com/file/d/1GYErLbLWJ3x-xtsCY3Kgb5PCSiBvVWbY/view?usp=sharing))
  - Reference dict (node, edge numbering): ```ref_dict.pkl``` (download [here](https://drive.google.com/file/d/13afpqNpOdRMRL5WLfkFW4I0Xw5G4HmAt/view?usp=sharing))
  - Edge ID mapping for data split: ```rel_map.csv``` (download [here](https://drive.google.com/file/d/1xxaMs-tCFIx1nuG0wIPi_YOtQi7bGUDv/view?usp=sharing))
- Features:
  - Image features: ```visualsem_features.h5``` (download [here](https://drive.google.com/file/d/1J6qx4-ho24DxGueXONH9ap0s26oijJSy/view?usp=sharing))
  - Text features: ```text_features_multi.h5``` (download [here](https://drive.google.com/file/d/1rtvYaVR0RAG218o3wLWpJWeI2ao0s4Mn/view?usp=sharing))

3. GraphSage+DistMult: 
- Training:
  - To train our best model using image+text features and node+edge gating: ```python train.py```  
  - To train model without image or text features: ```python train.py --mode node```
  - To train model using only image features and node+edge gating: ```python train.py --mode img```
  - To train model using only text features and node+edge gating: ```python train.py --mode gl```
  - To modify the number of negative samples (default: 1000): ```python train.py --neg_sample_size 100 --batch_size 10000```
- Evaluation:
  - Download pretrain model weights [here](https://drive.google.com/drive/folders/1FAOEBU67rqhzfGr4GmE2w9vmnbwSr3le?usp=sharing) and unzip it. Place it at the same folder as ```eval.py```
  - To evaluate our best model using image+text features and node+edge gating: ```python eval.py```  
  - To evaluate model without image or text features: ```python eval.py --mode node```
  - To evaluate model using only image features and node+edge gating: ```python eval.py --mode img```
  - To evaluate model using only text features and node+edge gating: ```python eval.py --mode gl```
  - To modify the number of negative samples (default: 1000): ```python eval.py --neg_sample_size 100 --batch_size 10000```

### Downstram task 1: NER

### Downstream task 2: Multisense
1. Requirements
```
- Python 3.6+
- PyTorch 1.1.0+
- transformers-3.5.0
```

2. Data
- To download raw images in [MultiSense dataset](https://github.com/spandanagella/multisense) please follow the link [here](https://drive.google.com/open?id=1e0ebK7KWlBzlc0j2u3CpXWJ0zVupPxM9). Alternatively, run commands:  
  - ```pip install gdown```
  - ```gdown https://drive.google.com/uc?id=1e0ebK7KWlBzlc0j2u3CpXWJ0zVupPxM9 -O YOURLOCATION```
  - ```!tar xvf multiSenseImagesAll.tar.gz```
- To download other files used in training, please follow the link [here](TBA - Iacer)  
  - Reference file for verb, query phrase and its German translation: ```gold_german_query_classes.csv```
  - ResNet152 image features for train/valid/test sets: ```features_per_image_train_german.h5``` / ```features_per_image_val_german.h5```/ ```features_per_image_test_german.h5```
  - Look-up table for (image_path, image_name, image_verb) for train/valid/test sets: ```train_german.pkl```/ ```val_german.pkl```/ ```test_german.pkl```
  - Look-up table for verb to integer index (based on training set): ```verb_map.pkl```
  - Look-up table for top-1 retrieved node hidden state for each query: ```query_nodes.pkl```
  - Node hidden state files: (TBA - from Yash's gdrive)

3. Training/Evaluatoin
- To train our baseline: ```python multi_train.py --epochs 10 --num_layer 2 --projection --lr 5e-4 --dropout 0.1 --nonlinear```
- To train with node hidden states: ```python multi_train.py --epochs 10 --node --num_layer 2 --projection --lr 5e-4 --dropout 0.1 --nonlinear```
- To subset training data in the low-resource regime, add argument: ```--subset 0.1```
- To specify node hidden state files, add argument: ```--node_file "/content/drive/My Drive/graph/nyu_multimodal_kb/NER/graph_emb_img.t"```

4. Misc
- Script for creating ResNet152 image features: ```img_feature.py```
- Model checkpoints: (TBA - gdrive)
