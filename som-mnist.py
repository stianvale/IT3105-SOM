import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import cProfile

class SOM(object):

	def __init__(self, txtfile, gridSize, lrate, tauLearn, tauTop, toprate):
		

		self.targets = []
		self.rand = None
		self.testset = None
		self.import_from_file(txtfile)
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
		self.gridSize = gridSize

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

		self.rand = random.randint(0,len(coords)-601)

		self.inputs = np.array(coords[self.rand:self.rand + 500])
		self.testset = np.array(coords[self.rand + 500:self.rand + 600])

	def createOutputLayer(self):

		self.outputs = np.array([[random.uniform(0,1) for x in self.inputs[0]] for x in range(0,self.gridSize**2)])


	def getTopDist(self, i, j):

		topDist = math.fabs((i % self.gridSize) - (j % self.gridSize)) + math.fabs((math.floor(i/self.gridSize)) - math.floor(j/self.gridSize))

		return topDist

	def train_network(self, epochs):
		#self.plotResults()
		for epoch in range(epochs):
			print(epoch)
			if((epoch+1) % 100 == 0):
				#self.plotMap(epoch)
				self.test_network()
				self.plotGrid()


			for i, inpt in enumerate(self.inputs):
				# winning_node = None
				# winning_distance = None
				# for j, outpt in enumerate(self.outputs):
				# 	if(j == 0):
				# 		winning_node = 0
				# 		winning_distance = np.linalg.norm(inpt - outpt)

				# 	distance = np.linalg.norm(inpt - outpt)
				# 	if(distance < winning_distance):
				# 		winning_distance = distance
				# 		winning_node = j

				distances = np.linalg.norm(inpt - self.outputs, axis=1)

				winning_node = np.argmin(distances)

				top_dists = np.array([self.getTopDist(winning_node, x) for x in range(len(self.outputs))])
				top_vals = np.exp([-((top_dists)**2)/self.toprate**2])

				self.outputs = self.outputs + self.lrate * np.transpose(top_vals) * (inpt - self.outputs) 

				#self.outputs = np.array([self.outputs[j] + self.lrate * top_vals[j] * vector_diffs[j] for j in range(len(self.outputs))])
				# for j, outpt in enumerate(self.outputs):
				# 	top_dist = self.getTopDist(winning_node,j)
					
				# 	top_val = math.exp(-((top_dist)**2)/self.toprate**2)

				# 	self.outputs[j] = outpt + self.lrate * top_val * (inpt - outpt)

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

			self.winners[winning_node].append(self.targets[self.rand + i])

		for i in range(len(self.classes)):
			bincount = np.bincount(self.winners[i])
			if (len(bincount) == 0):
				self.classes[i] = -1
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

			if(self.classes[winning_node] == self.targets[self.rand + i]):
				correct += 1

		print("Training correct: " + str(correct*100/len(self.inputs)) + "%")

		correct = 0
		for i, inpt in enumerate(self.testset):
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

			if(self.classes[winning_node] == self.targets[self.rand + 500 + i]):
				correct += 1

		print("Testing correct: " + str(correct*100/len(self.testset)) + "%")

		'''testlist = []

		for i in range(0,100):
			if (i % 10 == 0):
				testlist.append([])
			testlist[-1].append(self.classes[i])

		print(np.array(testlist))'''


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

	def plotGrid(self):
		plt.clf()
		G=nx.grid_2d_graph(self.gridSize,self.gridSize)
		pos = dict( (n, (n[1], self.gridSize-1-n[0])) for n in G.nodes() )

		val_map = {
			0: 'khaki', 1: 'orange', 2: 'cornflowerblue', 3: 'red', 4: 'tomato',
			5: 'purple', 6: 'gold', 7: 'lime', 8: 'brown', 9: 'cyan', -1: 'grey'
		}

		
		colors = [val_map[x] for x in self.classes]

		values = dict((n,v) for n, v in list(zip(G.nodes(),self.classes)))

		nx.draw_networkx(G, pos=pos, node_color = colors, labels = values)

		plt.axis('off')
		plt.ion()
		plt.show()
		a = input()


def main():
	som = SOM(txtfile = "mnist.txt", gridSize = 12, lrate = 0.5, tauLearn = 20000, tauTop = 500 , toprate = 10)
	som.train_network(200)

	a = input()


main()



