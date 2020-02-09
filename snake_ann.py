import torch
import torch.nn as nn



class SnakeANN(nn.Module):
	def __init__(self, input_size):
		super(SnakeANN, self).__init__()
		
		self.lin_layer = nn.Sequential(
			nn.Linear(input_size, 50),
			nn.ReLU(),
			nn.Linear(50, 3)
		)

		self.cuda()


	def forward(self, x):

		
		out = self.lin_layer(torch.flatten(x))

		return out


