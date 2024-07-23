import keras
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances

# Assuming these utility functions and layers are defined elsewhere in your project
from ..layers import KDMLayer, RBFKernelLayer, CosineKernelLayer
from ..utils import pure2dm, dm2discrete, cartesian_product

class KDMGraphModel(keras.Model):
    def __init__(self,
                 encoded_size,
                 dim_y,
                 encoder,
                 n_comp,
                 sigma=0.1,
                 nodes=[],
                 edges=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.graph = nx.DiGraph()
        
        # Adding nodes and edges to the graph
        self.graph.add_nodes_from([node['name'] for node in nodes])
        self.graph.add_edges_from(edges)
        
        # Creating the layer dictionary
        self.layers_dict = {}
        for node in nodes:
            if 'type' in node and node['type'] == 'input':
                self.layers_dict[node['name']] = keras.layers.Input(shape=(1,), dtype='float32')
            else:
                kernel = node.get('kernel', RBFKernelLayer(sigma=sigma, dim=encoded_size, trainable=True))
                dim_x = node.get('dim_x', encoded_size)
                dim_y = node.get('dim_y', dim_y)
                n_comp = node.get('n_comp', n_comp)
                self.layers_dict[node['name']] = KDMLayer(kernel=kernel, dim_x=dim_x, dim_y=dim_y, n_comp=n_comp)

    def call(self, inputs):
        node_outputs = {node: None for node in self.graph.nodes}
        
        # Setting the inputs to the graph
        for node, value in inputs.items():
            node_outputs[node] = value
        
        # Processing nodes in topological order
        for node in nx.topological_sort(self.graph):
            if node_outputs[node] is not None:
                continue
            # Get the inputs to this node
            input_nodes = list(self.graph.predecessors(node))
            if len(input_nodes) > 1:
                # Merge the inputs if there are multiple

                input_data = [node_outputs[in_node] for in_node in input_nodes]
                merged_prbs = cartesian_product(input_data)
                rho_x = merged_prbs
                rho_x = pure2dm(rho_x)
            else:
                rho_x = node_outputs[input_nodes[0]]
                rho_x = pure2dm(rho_x)
            
            # Compute the output for this node
            rho_y = self.layers_dict[node](rho_x)
            node_outputs[node] = dm2discrete(rho_y)
        
        # Return the output of the final node
        final_node = list(self.graph.nodes)[-1]

        return node_outputs[final_node]

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1, node_name=None):
        samples_x = samples_x.astype(np.float32)
        samples_y = samples_y.astype(np.float32)
        encoded_x = keras.layers.Identity()(samples_x)
        if init_sigma:
            np_encoded_x = keras.backend.eval(encoded_x)
            distances = pairwise_distances(np_encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.layers_dict[node_name].kernel.sigma.assign(sigma)
        self.layers_dict[node_name].c_x.assign(encoded_x)
        self.layers_dict[node_name].c_y.assign(samples_y)
        self.layers_dict[node_name].c_w.assign(
            keras.ops.ones((self.layers_dict[node_name].n_comp,)) / self.layers_dict[node_name].n_comp)
