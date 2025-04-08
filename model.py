"""
define moduals of model
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------
		
		## define CNN layers below
		self.conv = nn.Sequential( 	
			nn.Conv2d(in_channels =1, out_channels=args.channel_out1, kernel_size=args.k_size, stride=args.stride),
			nn.BatchNorm2d(args.channel_out1),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.MaxPool2d(kernel_size=args.pooling_size, stride=args.stride),

			nn.Conv2d(in_channels =args.channel_out1, out_channels=args.channel_out2, kernel_size=args.k_size, stride=args.stride),
			nn.BatchNorm2d(args.channel_out2),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.MaxPool2d(kernel_size=args.pooling_size, stride=args.stride),

			nn.Conv2d(in_channels =args.channel_out2, out_channels=args.channel_out3, kernel_size=args.k_size, stride=args.stride),
			nn.BatchNorm2d(args.channel_out3),
			nn.ReLU(),
			nn.Dropout(args.dropout),
			nn.MaxPool2d(kernel_size=args.pooling_size, stride=args.stride) 
								)

		##-------------------------------------------------
		## write code to define fully connected layers below
		##-------------------------------------------------
		dummy_input = torch.zeros(args.batch_size, 1, 28, 28)
		with torch.no_grad():
			out=self.conv(dummy_input)
			flatten_size=out.view(1,-1).shape[1]
		
		in_size = 16384
		print(in_size)
		out_size = 10
		# self.fc = nn.Linear(in_size, out_size)

		self.fc = nn.Sequential(  nn.Linear(in_size, args.fc_hidden1),
								  nn.ReLU(),
								  nn. Linear(args.fc_hidden1, args.fc_hidden2),
								  nn.ReLU(),
								  nn. Linear(args.fc_hidden2, out_size),
								)
		

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------
		x_out = self.conv(x)

		## write flatten tensor code below (it is done)
		x = torch.flatten(x_out,1) # x_out is output of last layer

		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc(x)   # predict y
		
		
		return result
        
		
		
	
		