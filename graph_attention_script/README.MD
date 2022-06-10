## Train GAT example

```
python train.py --data_location '/content/drive/MyDrive/nyu_multimodal_kb/WN9_IMG/' --img_feature_path '/content/drive/MyDrive/nyu_multimodal_kb/WN9_IMG/visual_features.pt' --txt_feature_path '/content/drive/MyDrive/nyu_multimodal_kb/WN9_IMG/text_features.pt' --epochs 20 --evaluate_interval 1000 --num_rels 9 --num_graph_nodes 6555 --batch_size 100 --neg_sample_size 1000
```
