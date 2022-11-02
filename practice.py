import networkx as nx

filename = './data/abilene_edgelist_noweight.txt'  # 使用权重设置为0的边进行创建
G = nx.read_edgelist(filename, create_using=nx.Graph(), nodetype=int, data=(("weight", float),))
print(1)