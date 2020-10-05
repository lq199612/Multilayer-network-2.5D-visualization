# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:03:50 2020

@author: lq
"""

import json
import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymnet import *
import pymnet
import networkx as nx

source_data_path = './source_data/London_Multiplex_Transport/Dataset'
net = pymnet.MultilayerNetwork(aspects=1, fullyInterconnected=False)
nodes_df = pd.read_table(
    source_data_path + 'london_transport_nodes.txt', sep='[ |\t]', engine='python')
edges_df = pd.read_table(
    source_data_path + 'london_transport_multiplex.txt', sep='[ |\t]', engine='python')
edges_df.columns = ['layerID', 'nodeID', 'nodeID_', 'weight']
layes_df = pd.read_table(
    source_data_path + 'london_transport_layers.txt', sep='[ |\t]', engine='python')
layers = {}
for index, row in layes_df.iterrows():
    layers[row['layerID']] = row['layerLabel']

for index, row in edges_df.iterrows():
    net[row['nodeID'], layers[row['layerID']]
        ][row['nodeID_'], layers[row['layerID']]] = 1


# %% 节点连接矩阵
def matrix(nodes_df, edges_df):
    nodeId = nodes_df['nodeID']
    node_num = len(nodeId)
    mx = np.zeros((node_num, node_num), list)
    for index, row in edges_df.iterrows():
        if not mx[row['nodeID']][row['nodeID_']]:
            mx[row['nodeID']][row['nodeID_']] = [row['layerID']]
        else:
            mx[row['nodeID']][row['nodeID_']].append(row['layerID'])
        if not mx[row['nodeID_']][row['nodeID']]:
            mx[row['nodeID_']][row['nodeID']] = [row['layerID']]
        else:
            mx[row['nodeID_']][row['nodeID']].append(row['layerID'])
    return mx


mx = matrix(nodes_df, edges_df)

# %%
idx = 0
layers_info = {}
all_nodes = {}
all_edges = list(net.edges)
need_edges = []
for n in net.iter_nodes():
    # print(n)
    if n not in all_nodes:
        all_nodes[n] = idx
        idx += 1

# %% 求所有边
edges = set()
for e in all_edges:
    if ((e[0], e[1]) not in edges) and ((e[1], e[0]) not in edges):
        edges.add((e[0], e[1]))
g = nx.Graph()
g.add_nodes_from(list(all_nodes.values()))
g.add_edges_from(edges)
pos = nx.spring_layout(g)

# %%
to3d_nodes = []
to3d_edges = []


def normalize_pos(pos, boxSize=[1, 1]):
    x = []
    y = []
    for k in pos:
        x.append(pos[k][0])
        y.append(pos[k][1])
    maxX = max(x)
    minX = min(x)
    maxY = max(y)
    minY = min(y)
    for k in pos:
        pos[k][0] = (pos[k][0] - minX) / (maxX-minX) * boxSize[0]
        pos[k][1] = (pos[k][1] - minY) / (maxY-minY) * boxSize[1]


for k in pos:
    pos[k] = pos[k].tolist()

normalize_pos(pos)
idx = 0
for k in layers:
    layer = layers[k]
    layer_nodes = []
    layer_edges = []
    ns = []
    ids = []
    for n in net.iter_nodes(layer):
        ns.append(n)
        npos = pos[all_nodes[n]]
        ids.append(int(n))
        layer_nodes.append(
            {'id': str(k)+'-'+str(n), 'pos': [npos[0], npos[1], idx]})
    layer_edges_ = []
    for e in net.edges:
        if e[-2] == layer:
            layer_edges_.append(e)
            source = all_nodes[e[0]]
            target = all_nodes[e[1]]
            layer_edges.append(
                {'source': str(k)+'-'+str(source), 'target': str(k)+'-'+str(target)})

    idx += 1
    layers_info[layer] = {'nodes': layer_nodes, 'edges': layer_edges}
    to3d_nodes.append({'list': layer_nodes, 'name': layers[k], 'ids': ids})
    to3d_edges.extend(layer_edges)
# %% 相邻层同id节点的边
for idx, p in enumerate(to3d_nodes):
    if idx+1 < len(to3d_nodes):
        cur = set(p['ids'])
        nxt = set(to3d_nodes[idx+1]['ids'])
        com = cur & nxt
        for n in com:
            nn = all_nodes[n]
            to3d_edges.append(
                {'source': str(idx+1)+'-'+str(nn), 'target': str(idx+2)+'-'+str(nn)})


# %% 为边增加weight信息
to3d_edges_ = copy.deepcopy(to3d_edges)
for e in to3d_edges_:
    source = e['source']
    target = e['target']
    layer, sourceId = source.split('-')
    layer_, targetId = target.split('-')
    if layer == layer_:
        e['weight'] = edges_df[(edges_df['layerID'] == int(layer)) & (edges_df['nodeID'] == int(
            sourceId)) & (edges_df['nodeID_'] == int(targetId))]['weight'].values[0]
    else:
        e['weight'] = 0


# %%


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        return json_data


def write_dict(d, file_path):
    jsObj = json.dumps(d, cls=NpEncoder)
    fileObject = open(file_path, 'w')
    fileObject.write(jsObj)


# %%
write_dict(to3d_nodes, './data/london_station_nodes.json')
write_dict(to3d_edges_, './data/london_station_edges.json')
write_dict(mx, './data/matrix.json')
