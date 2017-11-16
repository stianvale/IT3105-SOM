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
		self.opt = None


		if(txtfile[0] == "1"):
			self.opt = 7542
		elif(txtfile[0] == "2"):
			self.opt = 6110
		elif(txtfile[0] == "3"):
			self.opt = 629
		elif(txtfile[0] == "4"):
			self.opt = 22068
		elif(txtfile[0] == "5"):
			self.opt = 14379
		elif(txtfile[0] == "6"):
			self.opt = 108159
		elif(txtfile[0] == "7"):
			self.opt = 59030
		elif(txtfile[0] == "8"):
			self.opt = 1211

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
				if line == "EOF":
					return coords
				line_split = line.split(" ")
				line_split = [x.strip() for x in line_split]
				line_split = [x for x in line_split if x != ""]
				if(len(line_split) == 1):
					break
				print(line_split)
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

		self.outputs = [[random.uniform(self.minX,self.maxX), random.uniform(self.minY, self.maxY)] for x in range(0,300)]


	def train_network(self, epochs):
		#self.plotResults()
		for epoch in range(epochs):
			print(epoch)
			print("Learning rate: " + str(self.lrate))
			print("Neighbourhood size: " + str(self.toprate))
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
					if(top_dist > len(self.outputs)/2):
						top_dist = len(self.outputs) - top_dist
					
					top_val = math.exp(-((top_dist)**2)/self.toprate**2)

					outpt[0] = outpt[0] + self.lrate * top_val * (inpt[0] - outpt[0])
					outpt[1] = outpt[1] + self.lrate * top_val * (inpt[1] - outpt[1])

			self.toprate = self.toprate * math.exp(-epoch/self.tauTop)
			self.lrate = self.lrate * math.exp(-epoch/self.tauLearn)

		print("Final length: " + str(self.return_ring_length()))

		if(self.opt != None):
			print("Optimal path has length: " + str(self.opt))
			if(self.return_ring_length() <= 1.1 * self.opt):
				print("Congratulations! Final length within 10 per cent of optimal.")
			else:
				print("Not good enough length. Try again!")

		#print(self.inputs)
		#print(self.outputs)

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


				

som = SOM(txtfile = "1.txt", lrate = 0.5, tauLearn = 100000, tauTop = 10000 , toprate = 20)
som.train_network(400)
a = input()

#som.plotProgression()




