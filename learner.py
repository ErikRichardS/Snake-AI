import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import random

from settings import *

input_fields = []
output_vectors = []
label_vectors = []

criterion = nn.MSELoss() # pos_weight = torch.tensor(WEIGHTS).cuda() )
learning_rate = 1e-3
optimizer = None

def initialize(ann):
	global optimizer
	optimizer = optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate)

def make_choice(ann, playfield, snake):
	output = torch.squeeze( ann.forward(playfield) )

	input_fields.append(playfield)
	output_vectors.append(output)
	label_vectors.append(torch.zeros([3]))

	#output = nn.functional.softmax( output, 0 )

	if len(input_fields) > 10:
		input_fields.pop(0)
		output_vectors.pop(0)
		label_vectors.pop(0)

	r = random.uniform(0, 1)


	if r < output[0].item():
		#print("Left")
		label_vectors[-1][0] = 1
		snake.turn_left()
	elif r < (output[0].item()+output[1].item()):
		#print("Right")
		label_vectors[-1][1] = 1
		snake.turn_right()
	else:
		label_vectors[-1][2] = 1



def train(ann, success):
	l = len(input_fields)
	s = width*height
	input_matrix = torch.zeros([l, s]).cuda()
	#output_matrix = torch.zeros([l, 3]).cuda()
	label_matrix = torch.zeros([l, 3]).cuda()

	for i in range(l):
		input_matrix[i] = input_fields[i].view(-1, s)
		#output_matrix[i] = output_vectors[i]
		label_matrix[i] = label_vectors[i]


	optimizer.zero_grad()
	output_matrix = torch.log( ann(input_matrix) )
	loss = criterion(output_matrix, label_matrix*success)
	loss.backward()
	optimizer.step()
