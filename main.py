import torch
import torch.nn as nn

from snake_ann import SnakeANN

import pygame

import time
import random



width = 40
height = 30
square_size = 10

snake_color = (255, 255, 255)
food_color = (0, 255, 0)

playfield = torch.zeros([width, height]).cuda()


class Player:
	x = []
	y = []

	direction = 1
	length = 4

	def __init__(self):
		for i in range(self.length):
			self.x.append(1)
			self.y.append(10-i)

		for i in range(self.length):
			playfield[self.x[i], self.y[i]] = -1

	def update(self):
		# Remove last tail square
		playfield[self.x[-1], self.y[-1]] = 0

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
		
		# Add new position of head square to game field
		playfield[self.x[0], self.y[0]] = -1
		return 1

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
				pygame.draw.rect(screen, food_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))
			elif playfield[x, y] == 1:
				pygame.draw.rect(screen, food_color, pygame.Rect(x*square_size+1, y*square_size+1, square_size-2, square_size-2))



def game_loop(player):
	pygame.init()
	screen = pygame.display.set_mode((width*square_size, height*square_size))
	clock = pygame.time.Clock()
	snake_player = Player()

	done = False

	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True

		

		output = nn.functional.softmax( player.forward(playfield), 0 )

		r = random.uniform(0, 1)
		if output[0] < r:
			snake_player.turn_left()
		elif output[0]+output[1] < r:
			snake_player.turn_right()



		alive = snake_player.update()

		if alive == -1:
			done = True

		# Update screen with new image
		render(screen)
		pygame.display.flip()


		clock.tick(10)

		


snake_brain = SnakeANN(width*height)


game_loop(snake_brain)