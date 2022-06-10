## Train a graph attention network

```
python train.py --data_location '/data/visualsem/' --img_feature_path 'path/to/image/features/img_features.pt' --txt_feature_path 'path/to/image/features/txt_features.pt' --epochs 20 --evaluate_interval 1000 --num_rels N1 --num_graph_nodes N2 --batch_size 100 --neg_sample_size 1000
```

### Arguments:

```data_location```: path to input data triplets; `train.npy`, `valid.npy`, `test.npy`

```img_feature_path```: see above

```txt_feature_path```: see above

```epochs```: number of training epochs

```evaluate_interval```: evaluate on validation set every N steps

```num_rels```: number of distinct relations in the input graph

```num_graph_nodes```: number of distinct nodes in the input graph

```batch_size```: training batch size

```neg_sample_size```: number of permuted triplets to generate against each true triplet
