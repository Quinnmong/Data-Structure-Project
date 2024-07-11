"""
NAME: Mong Diem Quynh
YID: 2022148019

Data Structures
mst.py
Instructor: Seong Jae Hwang

Minimum Spanning Tree utils for project.

Modified from the code provided in
"Data Structures & Algorithms in Python" by
Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser

"""

import numpy as np
from queue import PriorityQueue

class TreePartition:
    """ TreePartition class
    
    This is the tree-based partition class which was described in class
    consisting of 
    
    classes:
        1. Position
    functions:
        1. make_group(e)
        2. find(p)
        3. union(p, q)
    """
    
    class Position:
        """ Position class
        
        Position is like a node in a tree. A position of holds 
        1. _element: An arbitrary element. In HW4, this is a Vertex object.
        2. _tree_size: The number of nodes this Position object belongs to.
        3. _parent: The parent node of this Position object.
        """
        def __init__(self, e):
            self._element = e
            self._tree_size = 1    # initially, the "tree" only has this Position (node)
            self._parent = self    # initially, the parent is itself
        
    def __init__(self):
        """ Initializes the TreePartition object
        
        _num_trees is the number of trees this object has. Initially 0, and increments
        as make_group(e) gets called. If the Kruskal's algorithm is used, then each tree
        is a minimum spanning tree.
        """
        self._num_trees = 0        # number of trees (clusters or partitions) to keep track
        
    def make_group(self, e):
        """ Assign a group to element e.
        
        In a tree-based partition, this simply puts e into a single-node tree.
        The _num_trees (number of clusters) increments by 1 every time this
        function gets called.
        """
        self._num_trees += 1
        return self.Position(e)
    
    def find(self, p):
        """ Finds and returns the group "leader" of p
        
        In a tree-based partition, this finds and returns the root of the tree which p belongs to.
        """
        if p._parent != p:
            p._parent = self.find(p._parent)
        return p._parent
    
    def union(self, p, q):
        """ If p and q belong to different groups, merge their groups.
        
        In a tree-based partition, this function
        1. Finds the roots of p and q
        2. If the roots are different, merge the trees by
          a. First finding the larger tree by checking the number of nodes
          b. Then setting the smaller tree's root's parent to be the larger tree's root
          c. Update the size of the larger tree
          d. Also decrement _num_trees of TreePartition object since two trees merged into one tree
        """
        a = self.find(p)
        b = self.find(q)
        if a is not b:
            self._num_trees -= 1
            if a._tree_size > b._tree_size:
                b._parent = a
                a._tree_size += b._tree_size
            else:
                a._parent = b
                b._tree_size += a._tree_size
                

class Graph:
    """ Graph class
    
    Holds vertex and edge sets and provides minimal functionalities.
    
    classes:
        1. Vertex
        2. Edge
    functions:
        1. vertex_count()
        2. vertices()
        3. edges()
        4. insert_vertex(color, x, y)
        5. insert_edge(u, v, weight)
    """
    class Vertex:
        """ Vertex class
        
        Each pixel of an image constructs a vertex:
        1. _color: value of the 2D array (single or RGB values)
        2. _i, _j: i and j coordinates of the 2D array entry which this Vertex object holds
        """
        def __init__(self, color, i, j):
            self._color = color
            self._i, self._j = i, j
    
    class Edge:
        """ Edge class
        
        Stores two endpoint vertices and the edge weight.
        Assume an undirected edge throughout HW4.
        1. _u, _v: two endpoint vertices, representing an edge e=(u,v)
        2. _weight: edge weight for e=(u,v)
        """
        def __init__(self, u, v, weight):
            self._u, self._v = u, v
            self._weight = weight
    
    def __init__(self):
        """ Initializes the Graph object
        
        _edge_map is a map where
        1. the first keys are Vertex objects, and
        2. the second keys are Vertex objects which is the endpoints of first key

        ex: _edge_map[u][v] takes Vertex objects u and v, 
            and it returns an Edge object representing an edge (u,v)
            
        Note: Every Vertex 'u' added to the graph is added to _edge_map[u].
        """
        self._edge_map = {}
    
    def vertex_count(self):
        """ Count and return the number of vertices in this Graph
        """
        return len(self._edge_map)
    
    def vertices(self):
        """ Returns all the vertices of this Graph
        
        Since every Vertex 'u' added to this Graph is inserted into _edge_map[u],
        simply returning the keys of _edge_map is equivalent to returning all the vertices.
        """
        return self._edge_map.keys()
    
    def edges(self):
        """ Returns all the edges of this Graph
        
        
        For each _edge_map[u], each value is an endpoint Vertex object 'v',
        so simply collecting all values of _edge_map[u][v] is equivalent to
        collecting all possible edges.
        """
        result = set()
        for secondary_map in self._edge_map.values():
            result.update(secondary_map.values())
        return result
            
    def insert_vertex(self, color, i, j):
        """ Inserts a Vertex object into this Graph, and return this Vertex object
        
        Each Vertex object is constructed from a 2D array element.
        Thus, if it's created from I[i,j], it stores the following information:
        1. color: The value at I[i,j]. Note that row i is y-coordinate, and column j is 
                x-coordinate. Thus, [i,j] = [y,x].
        2. i, j: The coordinates of the pixel at [i,j].
        
        For a new Vertex v, the _edge_map[v] is initialized.
        """
        v = self.Vertex(color, i, j)
        self._edge_map[v] = {}
        return v
    
    def insert_edge(self, u, v, weight=None):
        """ Inserts an Edge object into this Graph
        
        The edge (u,v) with weight = w(u,v) is inserted into _edge_map[u,v].
        """
        self._edge_map[u][v] = self.Edge(u, v, weight)
        
        
def compute_weight(u, v):
    """ Computes the weight between colors u and v
    
    This is simply an L2-distance between vectors u and v.
    """
    return np.linalg.norm(u - v)

def mst_kruskal(G):
    """ Compute a minimum spanning tree of a graph using Kruskal's algorithm.
    
    Return a list of edges that comprise the MST.
    
    The elements of the graph's edges are assumed to be weights.
    
    ** This is all the comment for this function in the textbook.
    """
    tree = []                    # list of edges in spanning tree
    pq = PriorityQueue()         # entries are edges in G, with weight as key
    forest = TreePartition()     # keeps track of forest clusters
    vertex_to_position = {}      # map each vertex to its TreePartition entry Position

    for v in G.vertices():
        vertex_to_position[v] = forest.make_group(v)

    counter = 0                  # counter to break ties in PQ
    for e in G.edges():
        pq.put((e._weight, counter, e))
        counter += 1

    while forest._num_trees != 1 and not pq.empty():
        weight, _, edge = pq.get()
        u, v = edge._u, edge._v
        a = forest.find(vertex_to_position[u])
        b = forest.find(vertex_to_position[v])
        if a != b:
            tree.append(edge)
            forest.union(a,b)
            
    return tree

# -------------------------- DO NOT CHANGE ABOVE CODES ------------------- #

# -------------------------- IMPLEMENT BELOW YOURSELF -------------------- #
    
def image_to_graph(I):
    """ Contructs a Graph from an image 2D array
    
    input:
        I: 2D image matrix
    output:
        G: Graph object with vertices and edges. It holds
            1. Vertices where each Vertex is from I with color = I[i,j] and
            2. Edges which connects two Vertex objects u and v which are from two adjacent
                pixels in I. The edge weight is computed using compute_weight(u,v).
            
    Complete remaining part of this function.
    """
    # Use these initializations
    G = Graph()
    vertices_2d = np.empty( (I.shape[0],I.shape[1]), dtype=G.Vertex)
    height = I.shape[0]
    width = I.shape[1]

    # -------------------------- IMPLEMENT BELOW YOURSELF -------------------- #
    # Insert vertices and construct vertices_2d such that vertices_2d[i,j] holds
    # the Vertex object which is based on I[i,j]. This is useful to construct
    # since each Edge needs to connect the vertices in a special way as instructed.
    

    # Insert edges where connects every Vertex object to its adjacent Vertex 
    # objects vertically and horizontally.
    # ex: [i,j] connects to [i-1,j], [i+1,j], [i,j-1], and [i,j+1].
    # note: Don't add parallel edges (u,v) and (v,u). This can easily happen if you 
    #       naively add and repeat the edge constructions.
    # note: Don't forget to assign the weights.
    
    
    # -------------------------- IMPLEMENT ABOVE YOURSELF -------------------- #
    # Insert vertices
    for i in range(height):
        for j in range(width):
            vertex = G.insert_vertex(color=I[i,j], i=i, j=j)
            vertices_2d[i, j] = vertex

    # Insert edges to graph
    for i in range(height):
        for j in range(width):
            vertex = vertices_2d[i, j]

            # Connect with edge below 
            if i < height - 1:
                neighbor = vertices_2d[i+1, j]
                weight = compute_weight(neighbor._color, vertex._color)
                G.insert_edge(vertex, neighbor, weight)

            # Connect with edge right
            if j < width - 1:
                neighbor = vertices_2d[i, j+1]
                weight = compute_weight(neighbor._color, vertex._color)
                G.insert_edge(vertex, neighbor, weight)

            # Connect with edge top
            if i > 0:
                neighbor = vertices_2d[i-1, j]
                weight = compute_weight(neighbor._color, vertex._color)
                G.insert_edge(vertex, neighbor, weight)
            
            # Connect with edge left
            if j > 0:
                uneighbor= vertices_2d[i, j-1]
                weight = compute_weight(neighbor._color, vertex._color)
                G.insert_edge(vertex, neighbor, weight)
    
    return G

def segment_kruskal(I, num_seg):
    """ Segment the 2D array I by constructing a spanning forest using Kruskal's algorithm
    
    Thus function is a modification of the provided mst_kruskal function.
    Instead of making a single MST, you want to make a spanning forest: multiple MSTs that span the whole G.
    This can be achieved by **slightly** modifying mst_kruskal function, and this is your goal.
    
    Specifically, here are some differences:
    1. The graph G is based on image_to_graph(I)
    2. The edges are stored in spanning_forest. Unlike 'tree' which holds a single MST in mst_kruskal, 
        spanning_forest holds edges which construct multiple MSTs. Each MST thus is a 'cluster' for 
        our segmentation task. The number of MSTs is based on the input 'num_seg': number of segmentation.
    3. The segmentation matrix: Just like how we have been performing compression in HW3, 
        we "segment" the image by creating a new image where each pixel holds the color of its cluster.
        ** This means that the 'cluster color' is the color of the root Vertex of the pixel's Vertex!
        ** Meaning, segmentation[i,j] is the color of the I[i,j]'s Vertex object's root's color.
    
    Goal: In this function, modify the specified part of this function achieve 
        the Kruskal's algorithm segmentation.
        
    input:
        I: 2D image matrix
    output:
        segmentation: 2D array where each entry holds the color of the cluster leader (root Vertex).
        spanning_forest: This is identical to 'tree' in 'mst_kruskal' function, except this holds
            the edges for multiple MSTs which form our spanning forest.
            
    Hint: You do not need a lot of changes in this function. Closely refer to 'mst_kruskal'.
    """
    G = image_to_graph(I)

    spanning_forest = []
    pq = PriorityQueue()
    forest = TreePartition()
    vertex_to_position = {}
    segmentation = np.empty( I.shape )

    for v in G.vertices():
        vertex_to_position[v] = forest.make_group(v)

    counter = 0
    for e in G.edges():
        pq.put((e._weight, counter, e))
        counter += 1

    # -------------------------- IMPLEMENT BELOW YOURSELF -------------------- #
    # Perform the Kruskal's algorithm. This is VERY similar to the part in mst_kruskal function.
    # How would you modify the Kruskal's algorithm so you end up with a spanning forest (multiple trees)?
    
            
    # Construct the segmentation matrix. If all the vertices are correctly inserted
    # into G, then iterating through the vertices and identifying their roots (and their colors)
    # would be useful.
    
    
    # -------------------------- IMPLEMENT ABOVE YOURSELF -------------------- #
    while forest._num_trees != num_seg and not pq.empty():
        weight, _, edge = pq.get()
        u, v = edge._u, edge._v
        u_root = forest.find(vertex_to_position[u])
        v_root = forest.find(vertex_to_position[v])

        if u_root != v_root:
            spanning_forest.append(edge)
            forest.union(u_root, v_root)
            # spanning_forest.append(edge)

    # Construct the segmentation matrix
    for v in G.vertices():
        i = v._i
        j=v._j
        root = forest.find(vertex_to_position[v])
        r_root = root._element._color
        segmentation[i][j] = r_root
            
    return segmentation, spanning_forest
