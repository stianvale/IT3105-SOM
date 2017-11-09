import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

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
		print(self.outputs)
		self.outputsPlots = [self.outputs[:]]
		print(self.outputsPlots)
	

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
		#self.plotResults()
		for epoch in range(epochs):
			print(epoch)
			if((epoch) % 100 == 0):
				self.plotMap(epoch)

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

					outpt[0] = outpt[0] + self.lrate * top_val * (inpt[0] - outpt[0])
					outpt[1] = outpt[1] + self.lrate * top_val * (inpt[1] - outpt[1])

			self.toprate = self.toprate * math.exp(-epoch/self.tauTop)
			self.lrate = self.lrate * math.exp(-epoch/self.tauLearn)

		print(self.inputs)
		print(self.outputs)

	def plotResults(self):
		p1 = plt.plot([x[0] for x in self.outputs] + [self.outputs[0][0]], [x[1] for x in self.outputs] + [self.outputs[0][1]], '--bo')
		p2 = plt.plot([x[0] for x in self.inputs], [x[1] for x in self.inputs], 'rx')
		plt.show()


	def plotMap(self,epoch):
		plt.clf()
		p1 = plt.plot([x[0] for x in self.outputs] + [self.outputs[0][0]], [x[1] for x in self.outputs] + [self.outputs[0][1]], '--bo')
		p2 = plt.plot([x[0] for x in self.inputs], [x[1] for x in self.inputs], 'rx')
		#plt.title('Epoch # ', epoch)
		#plt.title('Epoch #{:06d}'.format(epoch))
		plt.title('Epoch #%d' % epoch)
		plt.draw()
		plt.pause(0.1)



				

som = SOM(txtfile = "1.txt", lrate = 0.7, tauLearn = 100000, tauTop = 10000 , toprate = 40)
som.train_network(1000)
#som.plotProgression()




