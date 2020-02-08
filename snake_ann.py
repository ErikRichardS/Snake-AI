import torch
import torch.nn as nn



class SnakeANN(nn.Module):
	def __init__(self):
		super(DeepANN, self).__init__()
		
		self.lin_layer = nn.Sequential(
			nn.Linear(1000, 10),
			nn.ReLU(),
			nn.Linear(10, 3)
		)

		self.cuda()


	def forward(self, x):
		
		out = self.lin_layer(x)

		return out


