#-------------------------------------------------------------------------------
# Name:        refpGraph
# Purpose:     Working with matrices and graphs
#
# Author:      roberto ferrer
#
# Created:     31/05/2012
# Copyright:   (c) roberto ferrer 2012
# Licence:     GNU General Public License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

def main():
    pass

if __name__ == '__main__':
    main()


import numpy as np
import networkx as nx
from operator import mul, mod


def all_simple_paths_graph(G, source, target, cutoff=None):
	#Author: Sergio Nery Simoes
	#From https://networkx.lanl.gov/trac/attachment/ticket/713/simple_paths6.py
    if cutoff is None:
        cutoff = len(G)-1
    if cutoff < 1:
        yield []
    else:
        visited = [source]
        stack = [iter(G[source])]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child == target:
                    yield visited + [target]
                elif child not in visited:
                    visited.append(child)
                    stack.append(iter(G[child]))
            else: #len(visited) == cutoff:
                if child == target or target in children:
                    yield visited + [target]
                stack.pop()
                visited.pop()


def all_simple_paths_multigraph(G, source, target, cutoff=None):
    #Author: Sergio Nery Simoes
	#From https://networkx.lanl.gov/trac/attachment/ticket/713/simple_paths6.py
    if cutoff is None:
        cutoff = len(G)-1
    if cutoff < 1:
        yield []
    else:
        visited = [source]
        stack = [(v for u,v in G.edges(source))]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child == target:
                    yield visited + [target]
                elif child not in visited:
                    visited.append(child)
                    stack.append((v for u,v in G.edges(child)))
            else: #len(visited) == cutoff:
                count = ([child]+list(children)).count(target)
                for i in range(count):
                    yield visited + [target]
                stack.pop()
                visited.pop()


def all_simple_paths(G, source, target, cutoff=None):
    #Author: Sergio Nery Simoes
	#From https://networkx.lanl.gov/trac/attachment/ticket/713/simple_paths6.py
    """Generate all simple paths in the graph G from source to target.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    target : node
       Ending node for path.

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> for path in nx.all_simple_paths(G,source=0,target=4):
    ...    print(path)
    [0, 1, 2, 3, 4]

    Notes
    -----
    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in `O(V+E)` time but the
    number of simple paths in a graph can be very large, e.g. `O(n!)` in
    the complete graph of order n.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.

    See Also
    --------
    shortest_path
    """
    if G.is_multigraph():
        return all_simple_paths_multigraph(G, source, target, cutoff=cutoff)
    else:
        return all_simple_paths_graph(G, source, target, cutoff=cutoff)



def path_edges(path):
    """Create the list of edges corresponding with a path (i.e. list of nodes).
    The edges are inverted to reflect the fact that in a multiplier matrix
    the effect of account i on account j is represented by the coefficient a_ji.
    """
    if len(path) < 2:
        print "Length of path must be a least 2."
        return
    shortPath = path[1:]
    edges = zip(path, shortPath)
    return edges


def direct_influence(G, edgeList):
    """Calculate the direct influence of a list of edges corresponding with a
    path.
    """
    # Get weights of graph G
    w = nx.get_edge_attributes(G,'weight')
    # Create the list of weights corresponding with edgeList (elementary path)
    desiredWeightList = [w[edge] for edge in edgeList]
    # Calculate direct influence of the elementary path
    directInfluence = reduce(mul, desiredWeightList)
    # Pair the path with its influence
    t = edgeList
    u = t, (directInfluence)
    return u



def minor(arr, i, j):
    # ith row, jth column removed from arr (i.e. minor in matrix form)
    # From http://stackoverflow.com/questions/3858213/numpy-routine-for-computing-matrix-minors
    # User: unutbu
    return arr[np.array(range(i)+range(i+1,arr.shape[0]))[:,np.newaxis],
               np.array(range(j)+range(j+1,arr.shape[1]))]


def cofactor(arr, i, j):
    # ijth cofactor (in scalar form) of arr
    if mod(i+j,2) == 0:
        return np.linalg.det(minor(arr,i,j))
    else:
        return np.linalg.det(minor(arr,i,j)) * -1


def newEdgeList(edgeList, source, target):
    ''' Produces a new edgeList out of an old one. This function is used in
    reduceMatrix function to take into account that edges in the elementary
    paths must be redefined when deleting row and column in deltaijMatrix.
    '''
    rowLi = []
    colLi = []
    for edge in edgeList:
        if edge[0] > source:
            rowLi.append(edge[0] -1)
        if edge[0] < source:
            rowLi.append(edge[0])
        if edge[1] > target:
            colLi.append(edge[1] -1)
        if edge[1] < target:
            colLi.append(edge[1])
    return [(x,y) for x,y in zip(rowLi,colLi)]


##def reduce_matrix(matrix, edgeList, source, target):
##    ''' Reduces a matrix by eliminating the complete row and column at the
##    intersection of each element a_ij with ij defined as each edge in edgeList.
##    Uses source and target to determine existing nodes for matrix.
##    Remember this function is used on deltaijMatrix which has already been
##    reduced and so source row and target column have been deleted.
##    '''
##    # Get order of matrix
##    N = matrix.shape[1]
##    # Create list with all nodes present in the matrix
##    allNodes = range(N)
##    allNodesRow.remove(source)
##    allNodesCol = range(N)
##    allNodesCol.remove(target)
##
##    # Create nodes that constitute elementary paths and that are to be
##    # substracted from the list of total nodes.
##    rowNodes = [edge[0] for edge in edgeList]
##    colNodes = [edge[1] for edge in edgeList]
##
##    # Substracting elementary path nodes from all nodes
##    #coR = list(allNodes)
##    coR = list(allNodesRow)
##    [coR.remove(x) for x in rowNodes]
##
##    #coC = list(allNodes)
##    coC = list(allNodesCol)
##    [coC.remove(x) for x in colNodes]
##
##    # Create indices of elements to keep
##    indices = np.ix_(coR,coC)
##
##    return matrix[indices]



def reduce_matrix(matrix, edgeList, source, target):
    ''' Reduces a matrix by eliminating the complete row and column at the
    intersection of each element a_ij with ij defined as each edge in edgeList.
    Uses source and target to determine existing nodes for matrix.
    Remember this function is used on deltaijMatrix which has already been
    reduced and so source row and target column have been deleted.
    '''
    # Create new edgelist that considers deleted row/column that is used
    # as input.
    edgeList = newEdgeList(edgeList, source, target)

    # Get order of matrix
    N = matrix.shape[1]
    # Create list with all nodes present in the matrix
    allNodes = range(N)

    # Create nodes that constitute elementary paths and that are to be
    # substracted from the list of total nodes.
    rowNodes = [edge[0] for edge in edgeList]
    colNodes = [edge[1] for edge in edgeList]

    # Substracting elementary path nodes from all nodes
    coR = list(allNodes)
    [coR.remove(x) for x in rowNodes]

    coC = list(allNodes)
    [coC.remove(x) for x in colNodes]

    # Create indices of elements to keep
    indices = np.ix_(coR,coC)

    return matrix[indices]