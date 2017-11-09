import sys
import random

class SOM(object):

	def __init__(self, txtfile):
		
		self.inputs = self.import_from_file(txtfile)
		self.outputs = None
		self.numCities = None
		self.lrate = None
		self.minX = None
		self.maxX = None
		self.minY = None
		self.maxY = None

		self.createOutputLayer()
	

	def import_from_file(self, txtfile):

		coords = []

		with open(txtfile) as f:
			line_list = f.readlines()
			line_list = line_list[5:]

			for line in line_list:
				line = line.strip()
				line_split = line.split(" ")
				if(len(line_split) == 1):
					break
				coords.append([float(x) for x in line_split[1:]])

		return coords

	def createOutputLayer(self):

		xList = [x[0] for x in self.inputs]
		yList = [y[1] for y in self.inputs]

		self.minX = min(xList)
		self.maxX = max(xList)
		self.minY = min(yList)
		self.maxY = max(yList)

		self.numCities = len(self.inputs)

		self.outputs = [[random.randint(self.minX,self.maxX), random.randint(self.minY, self.maxY)] for x in range(0,self.numCities)]


	def train_network(self, epochs):
		for epoch in range(epochs):
			for i, inpt in enumerate(self.inputs):
				winning_node = None
				winning_distance = None
				for j, outpt in enumerate(self.outputs):
					if(j == 0):
						winning_node = 0
						winning_distance = (inpt[0] - outpt[0])**2 + (inpt[1] - outpt[1])**2

					distance = (inpt[0] - outpt[0])**2 + (inpt[1] - outpt[1])**2
					if(distance < winning_distance):
						winning_distance = distance
						winning_node = j

				


SOM("1.txt")


