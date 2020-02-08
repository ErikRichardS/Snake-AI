import torch 

import pygame

import time



width = 40
height = 30
square_size = 10

snake_color = (255, 255, 255)

playfield = torch.zeros([width, height])


class Player:
	x = []
	y = []

	direction = 0
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
				pygame.draw.rect(screen, snake_color, pygame.Rect(x*square_size, y*square_size, square_size, square_size))


def game_loop():
	pygame.init()
	screen = pygame.display.set_mode((width*10, height*10))
	clock = pygame.time.Clock()
	snake_player = Player()

	done = False

	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True

		render(screen)

		alive = snake_player.update()

		if alive == -1:
			done = True

		clock.tick(10)

		# Update screen with new image
		pygame.display.flip()





game_loop()