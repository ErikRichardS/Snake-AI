import torch
import torch.nn as nn

#from snake_ann import SnakeANN
#import learner as lr
from hamilton_brain import HamiltonBrain
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

	def set_direction(self, direction):
		self.direction = direction

	def get_body_coordinates(self):
		return (self.x, self.y)

def render(screen):
	screen.fill((0, 0, 0))

	for x in range(width):
		for y in range(height):
			if playfield[x, y] == -1:
				pygame.draw.rect(screen, snake_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))
			elif playfield[x, y] == 1:
				pygame.draw.rect(screen, food_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))
			elif playfield[x, y] == -2: 
				pygame.draw.rect(screen, neck_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))
			elif playfield[x, y] == -3: 
				pygame.draw.rect(screen, head_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))

def place_food():

	for i in range(100):
		random_x = random.randrange(width)
		random_y = random.randrange(height)

		if playfield[random_x, random_y] != -1:
			playfield[random_x, random_y] = 1
			return (random_x, random_y)

	for x in range(width):
		for y in range(height):
			if playfield[x, y] != -1:
				playfield[x, y] = 1
				return (x, y)

	return (-1, -1)


def game_loop(screen, playfield, brain):
	clock = pygame.time.Clock()
	snake_player = Player(playfield)

	food_coord = place_food()

	done = False
	pause = False

	final_score = 0

	while not done:	
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True
			if pygame.key.get_pressed()[pygame.K_SPACE]:
				pause = not pause

		
		#last_move = playfield.clone()
		if not pause:
			direction = brain.decide( playfield, snake_player.get_body_coordinates(), food_coord )
			snake_player.set_direction(direction)

			status = snake_player.update(playfield)

			if status == 1:
				final_score += 1
				food_coord = place_food()
				if food_coord[0] == -1:
					print("Game won!")
					done = True
			elif status == -1:
				pause = True
				#playfield = last_move.clone()
				snake = snake_player.get_body_coordinates()
				playfield[snake[0][0], snake[1][0]] = -3
				playfield[snake[0][1], snake[1][1]] = -2
				playfield[snake[0][-1], snake[1][-1]] = -3
				render(screen)
				pygame.display.flip()

				print("Snake died")
				while True: 
					n = 1

			
		
		# Update screen with new image
		render(screen)
		pygame.display.flip()

		#done = True


		#clock.tick(60)

	return final_score

		


#snake_brain = SnakeANN(width*height)
#lr.initialize(snake_brain)

pygame.init()
screen = pygame.display.set_mode((width*square_size, height*square_size))



#playfield = torch.zeros([width, height]).cuda()

for i in range(1):
	playfield = torch.zeros([width, height])
	snake_brain = HamiltonBrain(playfield)
	score = game_loop(screen, playfield, snake_brain)
	print(score)