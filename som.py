

class SOM(object):

	def __init__(self, txtfile, lrate):
		
		self.inputs = self.import_from_file(txtfile)
		self.outputs = None
		self.lrate = lrate

		print(self.inputs)
	

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

	def createOutputLayer():





SOM("1.txt")


