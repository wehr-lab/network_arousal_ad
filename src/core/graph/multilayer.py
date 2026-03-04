import numpy as np
import networkx as nx
import graphUtils

class MultilayerGraph:
    """
    A class to represent a multilayer graph.
    """

    def __init__(self, layers=None):
        """
        Initialize the multilayer graph with given layers.
        
        Parameters:
        layers (list of numpy.ndarray): List of adjacency matrices for each layer.
        """
        self.layers = layers if layers is not None else []

    def add_layer(self, layer):
        """
        Add a new layer to the multilayer graph.
        
        Parameters:
        layer (numpy.ndarray): The adjacency matrix of the new layer.
        """
        self.layers.append(layer)