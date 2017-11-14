
import networkx as nx
import matplotlib.pyplot as plt
import random

list = 	[4, 7, 9, 9, 4, 7, 2, 5, 3, 2, 4, 9, 8, 2, 10, 3, 10, 3, 3, 3, 10, 10, 10, 0, 10, 3, 6, 10, 10, 2, 5, 1, 1, 2, 8, 3, 1, 5, 10, 2, 8, 3, 10, 5, 6, 3, 1, 2, 10, 10, 8, 3, 2, 2, 6, 3, 9, 10, 10, 5, 2, 2, 2, 2, 10, 10, 10, 8, 10, 5, 5, 2, 2, 8, 10, 10, 5, 8, 2, 7, 10, 2, 6, 4, 7, 8, 0, 8, 9, 7, 6, 0, 0, 0, 7, 9, 5, 5, 1, 2]


N = 10
G=nx.grid_2d_graph(N,N)
pos = dict( (n, (n[1], N-1-n[0])) for n in G.nodes() )


print(G.nodes())
print(pos)

val_map = {
	0: 'red',
	1: 'orange',
	2: 'blue',
	3: 'crimson',
	4: 'pink',
	5: 'purple',
	6: 'green',
	7: 'black',
	8: 'brown',
	9: 'cyan',
	10: 'grey'
}


values = [val_map[x] for x in list]


nx.draw_networkx(G, pos=pos, node_color = values, with_labels = False)

plt.axis('off')
plt.show()