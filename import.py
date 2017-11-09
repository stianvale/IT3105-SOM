def import_from_file(txtfile):

	coords = []

	with open(txtfile) as f:
		line_list = f.readlines()
		line_list = line_list[5:]

		for line in line_list:
			line_split = line.split(" ")
			coords.append(line_split[1:])

	return coords


print(import_from_file("1.txt"))
