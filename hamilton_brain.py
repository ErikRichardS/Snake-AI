import torch



def hamilton_simple(playfield):
	path_field = torch.zeros(playfield.shape, dtype=torch.int32)

	n = 0
	forward = True
	for i in range(path_field.shape[0]):
		if forward:
			for j in range(1, path_field.shape[1]):
				path_field[i,j] = n
				n += 1
		else:
			for j in range(path_field.shape[1]-1, 0, -1):
				path_field[i,j] = n
				n += 1


		forward = not forward

	for i in range(path_field.shape[0]-1,-1,-1):
		path_field[i,0] = n
		n += 1

	return path_field


def hamilton_backtrack(playfield):
	nr_v = playfield.shape[0] * playfield.shape[1]
	path = [-1] * nr_v
	path[0] = 0

	def has_edge(v1, v2):
		if v1 == v2+1 or v2 == v1+1 or v1 == v2 + playfield.shape[0] or v2 == v1 + playfield.shape[0]:
			return True

		return False

	def hamilton_recursive(v):

		if v == nr_v:
			if has_edge(path[0], path[v-1]):
				return True
			else:
				return False


		return nr_v

	hamilton_recursive(1)


def sum_tuples(t1, t2):
	return (t1[0]+t2[0], t1[1]+t2[1])


def is_valid_position(playfield, coord):
	x = coord[0]
	y = coord[1]
	if x >= playfield.shape[0] or x < 0 or y >= playfield.shape[1] or y < 0 or playfield[ x, y ] == -1:
		return False

	return True

def find_food_coordinates(playfield):
	for i in range(playfield.shape[1]):
		for j in range(playfield.shape[1]):
			if playfield[i,j] == 1:
				return (i, j)


class HamiltonBrain:
	def __init__(self, playfield):
		self.path = hamilton_simple(playfield)
		hamilton_backtrack(playfield)

		self.path_size = self.path.shape[0] * self.path.shape[1]

		self.directions = [
							(1,0), # Right
		 					(0,1), # Down
		 					(-1,0), # Left
		 					(0,-1) # Up
		 					]


	def decide(self, playfield, body_coordinates, food_coord):
		head_coord = (body_coordinates[0][0], body_coordinates[1][0])
		tail_coord = (body_coordinates[0][-1], body_coordinates[1][-1])

		#print(food_coord)

		direction_score = [0 for i in range(4)]


		for i in range(4):
			new_coord = sum_tuples(head_coord, self.directions[i])
			if is_valid_position( playfield, new_coord ):
				score = 0
				if (self.path[head_coord] < self.path[new_coord] or self.path[new_coord] == 0) and not (self.path[tail_coord] < self.path[new_coord] and self.path[head_coord] < self.path[tail_coord]):
					
					distance_to_food = (self.path[food_coord] - self.path[new_coord]) % self.path_size
					distance_to_tail = (self.path[tail_coord] - self.path[new_coord]) % self.path_size			

					score = 2000 - distance_to_food.item()

					if distance_to_tail < 10:
						score -= 100 * (10 - distance_to_tail.item())
						#print(distance_to_tail.item())
				else:
					score = -5000

				direction_score[i] = score
			else:
				direction_score[i] = -100000

		
		
		#c = sum_tuples(head_coord, self.directions[direction_score.index(max(direction_score)) ])

		#print(direction_score)

		#print( str(i) + " " + str(self.path[c].item()) + " " + str(self.path[tail_coord].item()))

		return direction_score.index(max(direction_score))

