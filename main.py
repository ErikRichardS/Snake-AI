import torch
import torch.nn as nn

from snake_ann import SnakeANN
import learner as lr
from settings import *

import pygame

import time
import random



class Player:
	def __init__(self, playfield):
		self.x = []
		self.y = []

		self.direction = 1
		self.length = 4

		for i in range(self.length):
			self.x.append(1)
			self.y.append(10-i)

		for i in range(self.length):
			playfield[self.x[i], self.y[i]] = -1

	def update(self, playfield):
		# Remove last tail square
		tail_coord = (self.x[-1], self.y[-1])
		playfield[tail_coord] = 0

		# Loops through all body coordinates and 
		# move them forward one step
		for i in range(self.length-1, 0, -1):
			self.x[i] = self.x[i-1]
			self.y[i] = self.y[i-1]

		# Update head with new position
		if self.direction == 0: # Right
			self.x[0] += 1
		elif self.direction == 1: # Down
			self.y[0] += 1
		elif self.direction == 2: # Left
			self.x[0] -= 1 
		elif self.direction == 3: # Up
			self.y[0] -= 1


		# Check if snake has run into a wall
		if self.x[0] < 0 or self.x[0] >= width or self.y[0] < 0 or self.y[0] >= height:
			print("Wall!")
			return -1

		# Check if snake has run into itself
		if playfield[self.x[0], self.y[0]] == -1:
			print("Snake!")
			return -1

		if playfield[self.x[0], self.y[0]] == 1:
			self.length += 1
			self.x.append(tail_coord[0])
			self.y.append(tail_coord[1])
			playfield[tail_coord] = -1
			playfield[self.x[0], self.y[0]] = -1
			place_food()
			return 1
		else:
			playfield[self.x[0], self.y[0]] = -1

		return 0

	def turn_right(self):
		self.direction += 1
		if self.direction > 3:
			self.direction = 0

	def turn_left(self):
		self.direction -= 1
		if self.direction < 0:
			self.direction = 3

def render(screen):
	screen.fill((0, 0, 0))

	for x in range(width):
		for y in range(height):
			if playfield[x, y] == -1:
				pygame.draw.rect(screen, snake_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))
			elif playfield[x, y] == 1:
				pygame.draw.rect(screen, food_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))

def place_food():
	placed = False

	while not placed:
		random_x = random.randrange(width)
		random_y = random.randrange(height)

		if playfield[random_x, random_y] != -1:
			playfield[random_x, random_y] = 1
			placed = True

def game_loop(screen, playfield, brain):
	clock = pygame.time.Clock()
	snake_player = Player(playfield)

	place_food()

	done = False

	while not done:	
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True


		lr.make_choice(brain, playfield, snake_player)

		status = snake_player.update(playfield)

		if status == 1:
			lr.train(brain, 1)

		elif status == -1:
			lr.train(brain, -1)
			done = True

		# Update screen with new image
		render(screen)
		pygame.display.flip()

		#done = True


		#clock.tick(60)

		


snake_brain = SnakeANN(width*height)
lr.initialize(snake_brain)

pygame.init()
screen = pygame.display.set_mode((width*square_size, height*square_size))
#playfield = torch.zeros([width, height]).cuda()

while True:
	playfield = torch.zeros([width, height]).cuda()
	game_loop(screen, playfield, snake_brain)
	torch.save(snake_brain, "snake_brain.pt")
	print("Restart")