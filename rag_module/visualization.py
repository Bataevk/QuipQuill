import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random


# Проверка дирректории
import os
if os.path.exists("./.everlasting_summer_minirag"):
    print("Directory exists")
    if os.path.exists("./.everlasting_summer_minirag/graph_chunk_entity_relation.graphml"):
        print("File exists")
    else:
        print("File does not exist")
        exit(1)
else:
    print("Directory does not exist")
    exit(1)

# Load the GraphML file
G = nx.read_graphml("./.everlasting_summer_minirag/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)


# Add colors and title to nodes
for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

# Add title to edges
for edge in net.edges:
    if "description" in edge:
        edge["title"] = edge["description"]

# Save and display the network
net.show("knowledge_graph.html")