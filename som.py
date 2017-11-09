
class SOM(object):
	
	def import_from_file(txtfile):

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


