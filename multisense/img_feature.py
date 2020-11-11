#/usr/bin/python

import sys
import torch
from torch import nn
from tqdm import tqdm
import os
import torchvision.models as models
import torchvision.transforms as transforms
import json
from PIL import Image
import numpy
from collections import defaultdict
import h5py
import argparse

import numpy as np
import pickle
import pandas as pd

###Modify to your own path
root_path = '/beegfs/nh1724/Graph/multisense/' 
###error logs
error_paths = []
class Identity(nn.Module):

###Helper class: replace the last layer of resnet152
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def load_multisense_images(split='train', language='german'):
    ###create train/val/test image paths & associated image name & translated verb
    image_paths, image_names, image_verbs = [],[],[]
    if language=='german':
        if split != 'test':
            file_path = os.path.join(root_path+'multiSenseImagesAll/german_train_val_splits/all/',split)
        else:
            file_path = os.path.join(root_path+'multiSenseImagesAll/german_test_splits/all/',split)
           
    else: #spanish
        if split != 'test':
            file_path = os.path.join(root_path+'multiSenseImagesAll/spanish_train_val_splits/all/',split)
        else:
            file_path = os.path.join(root_path+'multiSenseImagesAll/spanish_test_splits/all/',split)
    #list of all image folders: translated verbs
    folders = next(os.walk(file_path))[1]
    #loop through each folder to get image paths & associated image name
    for folder in folders:
        img_path = os.path.join(file_path, folder)
        image_names.extend([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
        image_paths.extend([os.path.join(img_path,f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
        image_verbs.extend([folder for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
    #save
    ref_dict = {'paths': image_paths,
                'names': image_names,
                'verbs': image_verbs}
    pickle.dump(ref_dict, open(root_path+split+'_'+language+'.pkl', 'wb'))
    return image_paths, image_names, image_verbs


def extract_image_features(batch_size=10, ngpus=1, debug=False, split_flag='train', lang='german'):
    image_paths,image_names, image_verbs = load_multisense_images(split=split_flag, language=lang)
    global error_paths

    if debug:
        # use very small subset of the data to make sure everything works.
        image_paths = image_paths[:200]

    def process_minibatches(image_paths, start_idx, end_idx):
        """ Closure used to process a set of minibatches with images for MSCOCO. """
        batch_ims = []
        offset = 0
        n_images = end_idx - start_idx
        for idx in range(start_idx, end_idx):
            img_temp = Image.open(image_paths[idx])
            #check if it's grayscale, and convert to rgb
            if img_temp.mode != 'RGB':
                img_temp = img_temp.convert('RGB')
            try:
                img_temp = preprocessing_numpy(img_temp)
            except:
                print("ERROR processing ", image_paths[idx], "...")
                offset += 1
                error_paths.append((idx,image_paths[idx]))
                continue
            # create batch dimension
            batch_ims.append( img_temp[None,:,:,:] )

        # concatenate list of tensors
        batch_ims = torch.cat(batch_ims)
        outputs = resnet152(batch_ims.to('cuda'))

        # sanity-check
        #print("n_images: ", n_images, "len(all_outputs): ", len(all_outputs), "offset: ", offset)
        assert( n_images - len(outputs) == offset ), "image number doesn't match extracted image features + error processed ones"
        return outputs.cpu().numpy()

    # create pre-trained CNN, make sure we're downloading pretrained models to right directory 
    os.environ['TORCH_HOME'] = "/beegfs/nh1724/torch_model_zoo/" #could modify to your own directory
    resnet152 = models.resnet152(pretrained=True).to('cuda')
    # remove last fully connected layer
    resnet152.fc = Identity()
    if ngpus>1:
        assert(torch.cuda.device_count() > 1), "CUDA device count: {}".format(torch.cuda.device_count())
        resnet152 = nn.DataParallel(resnet152)

    ## freeze network
    #for param in resnet152.parameters():
    #    param.requires_grad = False
    preprocessing_numpy = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # for each BabelNet ID
    print("total #images: ",    len(image_paths),
          "batch_size: ",       batch_size)
 
    with torch.no_grad():
        output_fname = root_path+'features_per_image_' + split_flag + '_' + lang + ".h5"

        with h5py.File(output_fname, 'w') as out_h5:
            shape_feats = (len(image_paths), 2048)
            shape_ids = (len(image_names), 1)

            # will store image features, one per image in VisualSem
            out_h5.create_dataset("global_features", shape_feats, dtype='float32', chunks=(1,2048), maxshape=(None, 2048), compression="gzip")

            # start adding features from index 0
            count_entries_added = 0
            for bn_idx in tqdm(range(0, len(image_paths), batch_size)):
                assert(count_entries_added == bn_idx)
                # handle last minibatch with less images than batch_size
                idx_end = bn_idx + batch_size 
                if idx_end > len(image_paths):
                    idx_end = len(image_paths) 

                node_features_numpy = process_minibatches(image_paths, bn_idx, idx_end)
                #print("from_idx: ", from_idx, ", n_images_in_node: ", n_images_in_node, ", n_features: ", n_features)

                out_h5['global_features'][ bn_idx: (bn_idx+node_features_numpy.shape[0]), : ] = node_features_numpy

                count_entries_added += (idx_end - bn_idx)

            # resize the dataset if needed
            out_h5['global_features'].resize( (count_entries_added, 2048) )
    print("Multisense image features saved in: %s"%output_fname)
    pickle.dump(error_paths, open('error_images.pkl', 'wb'))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--debug', action='store_true', default=False)
    p.add_argument('--ngpus', type=int, default=1)
    p.add_argument('--split', type=str, default='train')
    p.add_argument("--lang", type=str, default="german")
    args = p.parse_args()

    print("INFO: Running on %i GPU(s)...!"%args.ngpus)
    print("Creating h5 for split: ",args.split)
    extract_image_features(batch_size=args.batch_size, ngpus=args.ngpus, debug=args.debug, split_flag=args.split, lang=args.lang)
