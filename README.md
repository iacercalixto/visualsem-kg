# visualsem-kg
Representation learning for VisualSem knowledge graph.

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
