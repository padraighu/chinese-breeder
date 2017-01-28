import numpy as np 
import matplotlib.pyplot as plt
import random
import mnist
from sklearn import svm
from sklearn.externals import joblib
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from os import getcwd

def read_jpg(name):
	img = imread(name)
	x = img_as_float(img)
	x = rgb2gray(x)
	for i in range(len(x)):
		for j in range(len(x[i])):
			# turn strokes to black and background to white for readability
			x[i][j] = 1 - x[i][j] 
	return x

class Image:
	def __init__(self, height, width, first_gen=False):
		self.matrix = np.zeros((height, width))
		self.h = height
		self.w = width
		if first_gen:
			for i in range(self.matrix.shape[0]):
				for j in range(self.matrix.shape[1]):
					self.matrix[i][j] = 0 if random.random() < 0.8 else 1

	def load_matrix(self, mat):
		assert(self.matrix.shape == mat.shape)
		child = Image(self.matrix.shape[0], self.matrix.shape[1])
		for i in range(self.matrix.shape[0]):
			for j in range(self.matrix.shape[1]):
				child.matrix[i][j] = mat[i][j]
		return child

	def breed(self, mate):
		assert(self.matrix.shape == mate.matrix.shape)
		child = Image(self.matrix.shape[0], self.matrix.shape[1])
		for i in range(self.matrix.shape[0]):
			new_row = []
			begin = int(random.random() * self.matrix.shape[1])
			new_row = np.concatenate((self.matrix[i][:begin], mate.matrix[i][begin:]))
			child.matrix[i] = new_row
		return child

	def mutate(self):
		child = Image(self.matrix.shape[0], self.matrix.shape[1])
		for i in range(self.matrix.shape[0]):
			for j in range(self.matrix.shape[1]):
				child.matrix[i][j] = 0 if random.random() < 0.8 else 1 
		return child

	def display_one(self):
		plt.axis('off')
		plt.imshow(self.matrix.reshape((self.h, self.w)), cmap=plt.cm.gray_r)
		plt.show()

	def add_display(self, nrow, ncol, i):
		plt.subplot(nrow, ncol, i)
		plt.axis('off')
		plt.imshow(self.matrix.reshape((self.h, self.w)), cmap=plt.cm.gray_r)

	def __str__(self):
		return str(self.matrix)

class Generation():
	def __init__(self, size, image_height, image_width, target, mutate_rate):
		self.pool = {}
		self.size = size
		self.mutate_rate = mutate_rate
		self.target = target
		for i in range(size):
			self.pool[Image(image_height, image_width, True)] = float('inf')

	def fitness(self, img):
		distance = 0.0
		one_target = self.target
		for i in range(img.matrix.shape[0]):
			for j in range(img.matrix.shape[1]):
				distance += (img.matrix[i][j] - one_target[i][j]) ** 2
		return distance

	def select(self):
		sorted_images = sorted(self.pool, key=self.pool.get)
		dead_images = sorted_images[int(0.7 * len(sorted_images)):]
		for img in dead_images:
			self.pool.pop(img)

	def next_gen(self):
		self.select()
		new_pool = {}
		for i in range(self.size):
			key_lst = list(self.pool.keys())
			parent_a = random.choice(key_lst)
			parent_b = random.choice(key_lst)
			child = parent_a.breed(parent_b)
			child = child.mutate() if random.random() < self.mutate_rate else child
			new_pool[child] = self.fitness(child)
		self.pool = new_pool

	def get_lowest_distance(self):
		return min(self.pool.values())

	def get_mean_distance(self):
		distances = self.pool.values()
		return sum(distances) / float(len(distances))

	def display_best(self, height, width, iteration, save_jpg=False, path=''):
		i = 1
		plt.suptitle('Iteration {}'.format(iteration))
		for img in sorted(self.pool, key=self.pool.get)[:height * width]:
			img.add_display(height, width, i)
			i += 1
		if not save_jpg:
			plt.show()
		else:
			plt.savefig(path)

	def __str__(self):
		return str(self.pool)

def main():
	four = mnist.load_mnist(digits=[4], return_labels=False, path=getcwd())[4]
	zi = read_jpg('letter_small.jpg')
	iteration = 1000
	mutate_rate = 0.2
	height = 50
	width = 50
	pool_size = 1000
	x = []
	y1 = []
	y2 = []
	save_jpg = True
	gen = Generation(pool_size, height, width, zi, mutate_rate)
	for i in range(iteration):
		x.append(i)
		mean_loss = gen.get_mean_distance()
		min_loss = gen.get_lowest_distance()
		y1.append(min_loss)
		y2.append(mean_loss)
		print('Iteration: {} Minimum loss: {} Mean loss: {}'.format(i, min_loss, mean_loss))
		if i % 10 == 0:
			gen.display_best(5, 10, i, save_jpg, 'zi\\'+str(i)+'a.jpg')
			plt.clf()
			plt.xlabel('Iterations')
			plt.ylabel('Loss')
			p1, = plt.plot(x, y1, label='Minimum Loss')
			p2, = plt.plot(x, y2, label='Mean Loss')
			plt.legend(handles=[p1, p2])
			if not save_jpg:
				plt.show()
			else:
				plt.savefig('zi\\'+str(i)+'b.jpg')
		gen.next_gen()
	gen.display_best(5, 10)

test_generation = False
test_image = False
test_plot = False

if test_plot:
	plt.legend(['hi', 'hello'])
	plt.plot([1, 2, 3], [7, 8, 9], [1, 2, 3], [4, 5, 6])
	plt.show()

	line_up, = plt.plot([1,2,3], label='Line 2')
	line_down, = plt.plot([3,2,1], label='Line 1')
	plt.legend(handles=[line_up, line_down])
	plt.show()
if test_image:
	four = mnist.load_mnist(digits=[4], return_labels=False, path=getcwd())[0]
	four = Image(28, 28).load_matrix(four.reshape(28, 28))
	four.display_one()

	i = Image(28, 28, True)
	j = Image(28, 28, True)
	four.breed(i).display_one()
	four.mutate().display_one()
if test_generation:
	gen = Generation(10, 28, 28, 4, 0.2)
	gen.display_best(2, 5)
	gen.select()
	print(gen)

if __name__ == '__main__':
	main()
	#pass