import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class SOM(object):

	def __init__(self, txtfile, lrate, tauLearn, tauTop, toprate):
		
		self.inputs = self.import_from_file(txtfile)
		self.outputs = None
		self.numCities = None
		self.lrate = lrate
		self.tauLearn = tauLearn
		self.toprate = toprate
		self.tauTop = tauTop
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
			print(epoch)
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

				for j, outpt in enumerate(self.outputs):
					top_dist = math.fabs(winning_node-j)
					if(top_dist > len(self.inputs)/2):
						top_dist = len(self.inputs) - top_dist
					
					top_val = math.exp(-((top_dist)**2)/self.toprate**2)

					outpt[0] = outpt[0] + self.lrate * top_val * (self.inputs[j][0] - outpt[0])
					outpt[1] = outpt[1] + self.lrate * top_val * (self.inputs[j][1] - outpt[1])

			self.toprate = self.toprate * math.exp(-epoch/self.tauTop)
			self.lrate = self.lrate * math.exp(-epoch/self.tauLearn)
		print(self.inputs)
		print(self.outputs)

	def plotResults(self):
		plt.plot(np.array(self.inputs[0]), np.array(self.inputs[1]))
		

		a = input()


				


som = SOM(txtfile = "1.txt", lrate = 0.1, tauLearn = 100, tauTop = 100 , toprate = 0.1 )
som.train_network(250)
som.plotResults()




