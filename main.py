"""
Fun: CNN for MNIST classification
"""


import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from util import _create_batch
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel
from utils import str2bool
import wandb
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# import DataLoader


## input hyper-paras
parser = argparse.ArgumentParser(description = "nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate")
parser.add_argument("-decay", dest ="decay", type=float, default=0.5, help = "learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
parser.add_argument("-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob")
parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=str2bool, default=False, help="true of false")

parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
# parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out3", dest='channel_out3', type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

args = parser.parse_args()
	

def _load_data(DATA_PATH, batch_size):
	'''Data loader'''

	print("data_path: ", DATA_PATH)
	train_trans = transforms.Compose([transforms.RandomRotation(args.rotation),transforms.RandomHorizontalFlip(),\
								transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	
	train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True,train=True, transform=train_trans)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
	
	## for testing
	test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	
	return train_loader, test_loader


def _compute_accuracy(y_pred, y_batch):
	## --------------------------------------------
	## write the code of computing accuracy below
	## --------------------------------------------
	correct= (y_pred==y_batch).sum().item()
	total= y_batch.size(0)
	accy= correct/total
	return accy

def _save_checkpoint(ckp_path, model, epoch, optimizer, global_step):
    ## save checkpoint to ckp_path: 'checkpoint/step_100.pt'
    ckp_path = ckp_path + 'ckp_{}.pt'.format(epoch+1)
    checkpoint = {'epoch': epoch,
                  'global_step': global_step,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, ckp_path)

def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
	"""Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
	lr = learning_rate
	if (epoch > 5):
		lr = 0.001
	if (epoch >= 10):
		lr = 0.0001
	if (epoch > 20):
		lr = 0.00001
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	print("learning_rate: ", lr)
	
writer=SummaryWriter(log_dir='./logs')	

def main():

	use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.cuda.set_device(device=0) ## choose gpu number 0
	print("device: ", device)
	if use_cuda:
		torch.cuda.manual_seed(72)

	## initialize hyper-parameters
	num_epoches = args.num_epoches              # default: 40
	decay = args.decay                          # default: 0.5
	learning_rate = args.learning_rate          # default: 0.001

	## step 1: Load data
	DATA_PATH = "./data/"
	train_loader, test_loader=_load_data(DATA_PATH, args.batch_size)

	##-------------------------------------------------------
	## please write the code about model initialization below
	##-------------------------------------------------------
	model = CNNModel(args)
	## load model to gpu or cpu
	model.to(device)
	
	## --------------------------------------------------
	## Complete code about defining the LOSS FUNCTION
	## --------------------------------------------------
	optimizer = optim.Adam(model.parameters(),lr=learning_rate)  ## optimizer
	loss_fun =  nn.CrossEntropyLoss()  ## cross entropy loss
	
	##--------------------------------------------
	## load checkpoint below if you need
	##--------------------------------------------
	# if load_checkpoint:
		## write load checkpoint code below

	
	##  model training
	iteration=0
	if args.mode == 'train':
		model = model.train() ## model training
		for epoch in range(num_epoches): #10-50
			## learning rate
			adjust_learning_rate(learning_rate, optimizer, epoch, decay)
			
			for batch_id, (x_batch,y_labels) in enumerate(train_loader):
				iteration+=1
				x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

				## feed input data x into model
				output_y = model(x_batch)
				
				##---------------------------------------------------
				## write loss function below, refer to tutorial slides
				##----------------------------------------------------
				loss = loss_fun(output_y, y_labels)
				

				##----------------------------------------
				## write back propagation below
				##----------------------------------------
				optimizer.zero_grad()
				loss.backward()
				optimizer.step() # update params

				##------------------------------------------------------
				## get the predict result and then compute accuracy below
				##------------------------------------------------------
				y_pred = torch.argmax(output_y.data, 1)
				accy= _compute_accuracy(y_pred, y_labels)
				
				
				##----------------------------------------------------------
				## loss.item() or use tensorboard to monitor the loss blow
				## if use loss.item(), you may use log txt files to save loss
				##----------------------------------------------------------
				if iteration%10==0:
					print('iter: {} loss: {}, accy: {}'.format(iteration, loss.item(), accy))
					# writer.add_scalar('Training Loss', loss.item(), iteration)
					# writer.add_scalar('Training Accuracy', accy, iteration)
					wandb.log({'iter': iteration, 'loss': loss.item()})
					wandb.log({'iter': iteration, 'accy': accy})
			## -------------------------------------------------------------------
			## save checkpoint below (optional), every "epoch" save one checkpoint
			## -------------------------------------------------------------------
			# _save_checkpoint(args.ckp_path, model, epoch, optimizer, iteration)
			
				

	##------------------------------------
	##    model testing code below
	##------------------------------------
	model.eval()
	correct=0
	total=0
	with torch.no_grad():
		for batch_id, (x_batch,y_labels) in enumerate(test_loader):
			x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
			##---------------------------------------
			## write the predict result below
			##---------------------------------------
			output_y=model(x_batch)
			y_pred=torch.argmax(output_y.data, 1)

			##--------------------------------------------------
			## complete code for computing the accuracy below
			##---------------------------------------------------
			correct+= (y_pred==y_labels).sum().item()
			total+= y_labels.size(0)
	print("testing accy: ", accy)
		

if __name__ == '__main__':
	# time_start = time.time()
	# main()
	# time_end = time.time()
	# print("running time: ", (time_end - time_start)/60.0, "mins")
	with wandb.init(project='CNN', name='CNN_demo'):
		main()
	

