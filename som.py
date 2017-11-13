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
		#self.createOutputRing()
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

		return np.array(coords)

	def createOutputLayer(self):

		xList = [x[0] for x in self.inputs]
		yList = [y[1] for y in self.inputs]

		self.minX = min(xList)
		self.maxX = max(xList)
		self.minY = min(yList)
		self.maxY = max(yList)

		self.numCities = len(self.inputs)

		self.outputs = np.array([[random.uniform(self.minX,self.maxX), random.uniform(self.minY, self.maxY)] for x in range(0,300)])


	def createOutputRing(self):
		self.outputs = []

		self.outSize = 200
		edgeSize = int(self.outSize/4)

		for i in range(edgeSize):
			self.outputs.append([self.minX + (self.maxX-self.minX)*i/(self.outSize/4), self.minY])
		for i in range(edgeSize):
			self.outputs.append([self.maxX, self.minY + (self.maxY-self.minY)*i/(self.outSize/4)])
		for i in range(edgeSize):
			self.outputs.append([self.maxX - (self.maxX-self.minX)*i/(self.outSize/4), self.maxY])
		for i in range(edgeSize):
			self.outputs.append([self.minX, self.maxY - (self.maxY-self.minY)*i/(self.outSize/4)])


	def train_network(self, epochs):
		#self.plotResults()
		for epoch in range(epochs):
			#print(epoch)
			if((epoch) % 100 == 0):
				self.plotMap()

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
					if(top_dist > len(self.outputs)/2):
						top_dist = len(self.outputs) - top_dist
					
					top_val = math.exp(-((top_dist)**2)/self.toprate**2)

					self.outputs[j] = outpt + self.lrate * top_val * (inpt - outpt)

					#outpt[0] = outpt[0] + self.lrate * top_val * (inpt[0] - outpt[0])
					#outpt[1] = outpt[1] + self.lrate * top_val * (inpt[1] - outpt[1])

			self.toprate = self.toprate * math.exp(-epoch/self.tauTop)
			self.lrate = self.lrate * math.exp(-epoch/self.tauLearn)

		print(self.inputs)
		print(self.outputs)

	def plotResults(self):
		p1 = plt.plot([x[0] for x in self.outputs] + [self.outputs[0][0]], [x[1] for x in self.outputs] + [self.outputs[0][1]], '--bo')
		p2 = plt.plot([x[0] for x in self.inputs], [x[1] for x in self.inputs], 'rx')
		plt.show()


	def plotMap(self):
		plt.clf()
		p1 = plt.plot([x[0] for x in self.outputs] + [self.outputs[0][0]], [x[1] for x in self.outputs] + [self.outputs[0][1]], '--bo')
		p2 = plt.plot([x[0] for x in self.inputs], [x[1] for x in self.inputs], 'rx')
		plt.draw()
		plt.pause(0.1)
		print(self.return_ring_length())

	def return_ring_length(self):
		print(self.outputs)
		length = 0
		for j, outpt in enumerate(self.outputs):
			segment_length = 0
			if(j == 0):
				segment_length = np.linalg.norm(np.array(self.outputs[-1]) - np.array(outpt))
			else:
				segment_length = np.linalg.norm(np.array(self.outputs[j-1]) - np.array(outpt))

			length += segment_length

		return length


				

som = SOM(txtfile = "2.txt", lrate = 0.8, tauLearn = 100000, tauTop = 100000 , toprate = 10)
som.train_network(1500)
a = input()
#som.plotProgression()




