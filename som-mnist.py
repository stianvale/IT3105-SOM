import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

class SOM(object):

	def __init__(self, txtfile, lrate, tauLearn, tauTop, toprate):
		

		self.targets = []
		self.inputs = np.array(self.import_from_file(txtfile))
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
		self.outputsPlots = [self.outputs[:]]
	

	def import_from_file(self, txtfile):

		coords = []

		with open(txtfile) as f:
			line_list = f.readlines()

			for line in line_list:
				line = line.strip()
				line_split = line.split(",")

				coords.append([int(x) for x in line_split[:-1]])
				self.targets.append(int(line_split[-1]))

		return coords[0:500]

	def createOutputLayer(self):

		self.outputs = np.array([[random.uniform(0,1) for x in self.inputs[0]] for x in range(0,100)])


	def getTopDist(self, i, j):

		topDist = math.fabs((i % 10) - (j % 10)) + math.fabs((math.floor(i/10)) - math.floor(j/10))

		return topDist

	def train_network(self, epochs):
		#self.plotResults()
		for epoch in range(epochs):
			print(epoch)
			if((epoch+1) % 100 == 0):
				#self.plotMap(epoch)
				self.test_network()


			for i, inpt in enumerate(self.inputs):
				winning_node = None
				winning_distance = None
				for j, outpt in enumerate(self.outputs):
					if(j == 0):
						winning_node = 0
						winning_distance = np.linalg.norm(inpt - outpt)

					distance = np.linalg.norm(inpt - outpt)
					if(distance < winning_distance):
						winning_distance = distance
						winning_node = j

				for j, outpt in enumerate(self.outputs):
					top_dist = self.getTopDist(i,j)
					
					top_val = math.exp(-((top_dist)**2)/self.toprate**2)

					self.outputs[j] = outpt + self.lrate * top_val * (inpt - outpt)

					#outpt[0] = outpt[0] + self.lrate * top_val * (inpt[0] - outpt[0])
					#outpt[1] = outpt[1] + self.lrate * top_val * (inpt[1] - outpt[1])

			self.toprate = self.toprate * math.exp(-epoch/self.tauTop)
			self.lrate = self.lrate * math.exp(-epoch/self.tauLearn)

		#print(self.inputs)
		#print(self.outputs)

	def classify(self):
		self.classes = [0 for x in self.outputs]
		self.winners = [[] for x in self.outputs]
		for i, inpt in enumerate(self.inputs):
			winning_node = None
			winning_distance = None
			for j, outpt in enumerate(self.outputs):
				if(j == 0):
					winning_node = 0
					winning_distance = np.linalg.norm(inpt - outpt)

				distance = np.linalg.norm(inpt - outpt)
				if(distance < winning_distance):
					winning_distance = distance
					winning_node = j

			self.winners[winning_node].append(self.targets[i])

		for i in range(len(self.classes)):
			print(self.winners[i])
			bincount = np.bincount(self.winners[i])
			print(bincount)
			if (len(bincount) == 0):
				self.classes[i] = 10
			else:
				self.classes[i] = np.argmax(bincount)


	def test_network(self):
		self.classify()
		correct = 0
		for i, inpt in enumerate(self.inputs):
			winning_node = None
			winning_distance = None
			for j, outpt in enumerate(self.outputs):
				if(j == 0):
					winning_node = 0
					winning_distance = np.linalg.norm(inpt - outpt)

				distance = np.linalg.norm(inpt - outpt)
				if(distance < winning_distance):
					winning_distance = distance
					winning_node = j

			if(self.classes[winning_node] == self.targets[i]):
				correct += 1

		print("Correct: " + str(correct*100/len(self.inputs)) + "%")

		testlist = []

		for i in range(0,100):
			if (i % 10 == 0):
				testlist.append([])
			testlist[-1].append(self.classes[i])

		print(np.array(testlist))


		return correct




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
		print(self.return_ring_length())

	def return_ring_length(self):
		length = 0
		for j, outpt in enumerate(self.outputs):
			segment_length = 0
			if(j == 0):
				segment_length = np.linalg.norm(np.array(self.outputs[-1]) - np.array(outpt))
			else:
				segment_length = np.linalg.norm(np.array(self.outputs[j-1]) - np.array(outpt))

			length += segment_length

		return length





som = SOM(txtfile = "mnist.txt", lrate = 0.5, tauLearn = 100000, tauTop = 10000 , toprate = 10)
som.train_network(500)

a = input()
#som.plotProgression()




