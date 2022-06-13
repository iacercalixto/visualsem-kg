import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import torch
import time
import argparse
import configparser

root_dir = './../'

def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data', type=str, default='80-10-10/')
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--margin', type=float, default=5.0)
	parser.add_argument('--batchsize', type=int, default=1000)
	parser.add_argument('--epochs', type=int, default=1000)
	args = parser.parse_args()
	data_path = root_dir+'data/'+args.data
	epochs = args.epochs

	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path=data_path,
		nbatches=args.batchsize,
		threads=8,
		sampling_mode="normal",
		bern_flag=1,
		filter_flag=1,
		neg_ent=25,
		neg_rel=0)

	# dataloader for test
	test_dataloader = TestDataLoader(data_path, "link")

	# define the model
	transe = TransE(
		ent_tot=train_dataloader.get_ent_tot(),
		rel_tot=train_dataloader.get_rel_tot(),
		dim=200,
		p_norm=1,
		norm_flag=True)

	# define the loss function
	model = NegativeSampling(
		model=transe,
		loss=MarginLoss(margin=args.margin),
		batch_size=train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model=model, data_loader=train_dataloader, train_times=epochs, alpha=args.lr, # train_times=1000, alpha=1.0
					  use_gpu=torch.cuda.is_available())
	trainer.run()
	t_mark = time.strftime("%m%d-%H:%M", time.localtime())
	ckpt_path = root_dir+'checkpoint/transe-'+t_mark+'.ckpt'
	if not os.path.exists(root_dir+'checkpoint/'):
		os.mkdir(root_dir+'checkpoint/')
	transe.save_checkpoint(ckpt_path)
	print('ckpt saved at: '+ckpt_path)

	# test the model
	transe.load_checkpoint(ckpt_path)
	tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
	tester.run_link_prediction(type_constrain=False)

if __name__=='__main__':
	main()