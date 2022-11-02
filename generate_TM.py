import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

filename = './data/abilene_edgelist_noweight.txt'  # 需要使用权重设置为0的边进行创建
G = nx.read_edgelist(filename, create_using=nx.Graph(), nodetype=int, data=(("weight", float),))
new_filename = filename[:-13] + '_TM.txt'
with open(new_filename, 'w') as f:
# 为边设置权重
    for i in range(1, 12):
        for j in range(1, 12):
            if i != j:
                i_degree = len(G.degree._nodes[i])
                j_degree = len(G.degree._nodes[j])
                demand = i_degree * j_degree / nx.shortest_path_length(G, i, j)


# pos = {1: [5.0, 5.0],
#        2: [1.0, 1.0],
#        3: [12.0, 1.5],
#        4: [4.0, -2.5],
#        5: [17.0, -4.0],
#        6: [18.0, 1.0],
#        7: [25.0, -1.50],
#        8: [24.0, 2.0],
#        9: [21.0, 5.0],
#        10: [30.0, 1.0],
#        11: [35.0, 3.0]}
#
# edge_weights = nx.get_edge_attributes(G, 'weight')  # 边的权重
# nx.draw(G, pos, with_labels=True)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
# plt.show()
