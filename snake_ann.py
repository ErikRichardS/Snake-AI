import torch
import torch.nn as nn



class SnakeANN(nn.Module):
	def __init__(self, input_size):
		super(SnakeANN, self).__init__()
		self.input_size = input_size

		#self.prev_input = 
		
		self.lin_layer = nn.Sequential(
			nn.Linear(input_size, 50),
			nn.ReLU(),
			nn.Linear(50, 3),
			nn.Softmax(1)
		)

		self.cuda()


	def forward(self, field):
		field = field.view(-1, self.input_size)
		#field_prev = field_prev.view(-1, self.input_size)

		out = self.lin_layer(x)

		return out


