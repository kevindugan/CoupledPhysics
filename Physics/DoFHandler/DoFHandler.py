from ..Geometry.Geometry2D import Geometry2D
from ..Quadrature.Quadrature import Quadrature2D
from numpy import array, cumsum, empty
from matplotlib import pyplot as plt
from scipy.sparse import csr_array

class DoFHandler():

    def __init__(self, geom: Geometry2D, quadrature: Quadrature2D):
        self._geom = geom
        self._quad = quadrature

        # Build DoF Map
        # print(self._geom.globalNpoints, self._geom.globalNedges, self._geom.globalNelements)
        self._globalDoFsize = self._geom.globalNpoints + \
            self._geom.globalNedges*(self._quad.order-1) + \
            self._geom.globalNelements*(self._quad.order-1)**2
        self.dof_map, self.dof_graph = self.__build_dof_map()

    @property
    def globalDoFsize(self):
        return self._globalDoFsize
    
    @property
    def dofMap(self):
        return self.dof_map
    
    @property
    def dofConnectivity(self):
        return self.dof_connectivity
    
    @property
    def sparsity(self):
        '''Return (rowPtr, colIndices)'''
        return self._rowPointer, self._columnIndices

    def __build_dof_map(self):
        # Will need to know the number of elements for each proc to number higher-order dofs
        # per element
        nElementsPerProc = self._geom.mpiComm.allgather(len(self._geom.localConnectivity))
        myElementOffset = 0
        for it in range(self._geom.mpiRank):
            myElementOffset += nElementsPerProc[it]

        # Build DoF Connectivity
        # Each cell should have a mapping to several DoFs, which correspond to basis
        # functions
        pointsPerElement = 4 + (self._quad.order-1)*4 + (self._quad.order-1)**2
        nGlobalPoints = self._geom.globalNpoints
        nGlobalEdges = self._geom.globalNedges
        self.dof_connectivity = [[0]*pointsPerElement for _ in self._geom.localConnectivity]
        for it,(elem,edge) in enumerate(zip(self._geom.localConnectivity, self._geom.localEdgeConnectivity)):
            # Vertex Dofs
            self.dof_connectivity[it][0:4] = list(elem)
            # Edge Dofs
            edge_end = self._quad.order*4
            self.dof_connectivity[it][4:edge_end] = [nGlobalPoints + it*nGlobalEdges + e for it in range(self._quad.order-1) for e in edge]
            # Bubble Dofs
            start = self._geom.globalNpoints + self._geom.globalNedges*(self._quad.order-1) + (it+myElementOffset)*(pointsPerElement-4*self._quad.order)
            end = start + (pointsPerElement-4*self._quad.order)
            self.dof_connectivity[it][edge_end:] = list(range(start,end))

        # print(f"Rank {self._geom.mpiRank} {self.dof_connectivity} {self._geom.localEdgeConnectivity}")
        # DoF graph
        localGraph = {it:set() for it in range(self._globalDoFsize)}
        for elem in self.dof_connectivity:
            for it in elem:
                for jt in elem:
                    localGraph[it].add(jt)

        # DoF Mapping [Identity]
        dof_map = [it for it in localGraph.keys()]

        return dof_map, localGraph
    
    def renumberDoFs(self):
        # Collect graphs
        allGraphs = self._geom.mpiComm.allgather(self.dof_graph)
        globalGraph = {it:set() for it in range(self._globalDoFsize)}
        for graph in allGraphs:
            for key,val in graph.items():
                for v in val:
                    globalGraph[key].add(v)

        # Determine Starting index
        rank = [len(val) for val in globalGraph.values()]
        start = rank.index(min(rank))

        # Breadth First Search
        self.dof_map = [None for _ in globalGraph.keys()]
        visited = set([start])
        queue = [start]
        count = 0
        while len(queue) > 0:
            it = queue.pop(0)
            self.dof_map[it] = count
            count += 1
            for v in globalGraph[it]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        order = sorted([it for it,val in enumerate(self.dof_map) if val is not None], key=lambda x: self.dof_map[x])

        # Renumber graph connections
        self.dof_graph = {it:set() for it in range(self._globalDoFsize)}
        for elem in self.dof_connectivity:
            for it in elem:
                for jt in elem:
                    self.dof_graph[self.dof_map[it]].add(self.dof_map[jt])

        # Clear sparsity if it has been built
        if hasattr(self, "_columnIndices"):
            del self._columnIndices
        if hasattr(self, "_rowPointer"):
            del self._rowPointer

    def buildSparsity(self):
        self._columnIndices = [it for val in self.dof_graph.values() for it in val]
        self._rowPointer = [0] + list(cumsum([len(val) for val in self.dof_graph.values()]))

    def plotSparsity(self, fileroot="dof_sparsity", show=False, colorByRank=True):
        assert hasattr(self, "_columnIndices") and hasattr(self, "_rowPointer"), "Sparsity not Built. Must Call DoFHandler.buildSparsity() first"

        colorWheel = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"] if colorByRank else ["red"]
        symbWheel = ["o", "x", "+", "^", "2"] if colorByRank else ["o"]

        if self._geom.mpiRank == 0:
            globalA = csr_array(
                    (array([1]*len(self._columnIndices), dtype="int8"),
                    self._columnIndices,
                    self._rowPointer), shape=(self._globalDoFsize, self._globalDoFsize))

            plt.figure()
            plt.spy(globalA, marker=symbWheel[self._geom.mpiRank%len(symbWheel)], markersize=3, color=colorWheel[self._geom.mpiRank%len(colorWheel)])

            for proc in range(1, self._geom.mpiSize):
                col = self._geom.mpiComm.recv(source=proc, tag=12)
                row = self._geom.mpiComm.recv(source=proc, tag=13)
                globalA = csr_array(([1]*len(col), col, row), shape=(self._globalDoFsize,self._globalDoFsize))
                plt.spy(globalA, marker=symbWheel[proc%len(symbWheel)], markersize=3, color=colorWheel[proc%len(colorWheel)])
            if show:
                plt.show()
            else:
                plt.savefig(f"{fileroot}.png", bbox_inches="tight", dpi=600)
        else:
            self._geom.mpiComm.send(self._columnIndices, dest=0, tag=12)
            self._geom.mpiComm.send(self._rowPointer, dest=0, tag=13)