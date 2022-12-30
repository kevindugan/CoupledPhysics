import meshio
from numpy import linspace, sum, array
from numpy.linalg import inv
from mpi4py import MPI

class Geometry2D():
    def __init__(self):
        self._comm = MPI.COMM_WORLD
        self._mpiRank = self._comm.Get_rank()
        self._mpiSize = self._comm.Get_size()

        # Parallel Decomposition
        nBlockX, nBlockY = MPI.Compute_dims(self._mpiSize, 2)
        self.cartComm = self._comm.Create_cart([nBlockX, nBlockY], [False, False], True)

    def __iter__(self):
        return GeometryIterator(self)

    @property
    def mpiComm(self):
        return self.cartComm

    @property
    def mpiRank(self):
        return self._mpiRank

    @property
    def mpiSize(self):
        return self._mpiSize
    
    @property
    def globalNpoints(self):
        return len(self._globalPoints)
    
    @property
    def globalNedges(self):
        return self._nGlobalEdges

    @property
    def globalNelements(self):
        return self._nGlobalElements
    
    @property
    def localConnectivity(self):
        return self._localConn
    
    @property
    def localEdgeConnectivity(self):
        return self._local_edge_conn
    
    @property
    def localBoundaryNodes(self):
        return self._boundaryNodes

    def readInternal(self, xExtent=(0,1), nX=4, yExtent=(0,1), nY=5):
        xCoor = linspace(xExtent[0], xExtent[1], nX+1)
        yCoor = linspace(yExtent[0], yExtent[1], nY+1)
        self._globalPoints = array([[x,y,0] for y in yCoor for x in xCoor], dtype="float64")
        cells = array([[jt*(nX+1)+it, jt*(nX+1)+it+1, (jt+1)*(nX+1)+it+1, (jt+1)*(nX+1)+it] for jt in range(nY) for it in range(nX)])
        self._nGlobalElements = len(cells)

        nBlockX, nBlockY = self.cartComm.dims
        xDomain = linspace(xExtent[0], xExtent[1], nBlockX+1)
        yDomain = linspace(yExtent[0], yExtent[1], nBlockY+1)
        myExtentX = [ xDomain[self.cartComm.coords[0]], xDomain[self.cartComm.coords[0]+1] ]
        myExtentY = [ yDomain[self.cartComm.coords[1]], yDomain[self.cartComm.coords[1]+1] ]
    
        myCell = lambda x: x[0] >= myExtentX[0] and x[0] < myExtentX[1] and x[1] >= myExtentY[0] and x[1] < myExtentY[1]
        self._localConn = [MeshCell2D(v,self._globalPoints) for v in cells if myCell(sum( self._globalPoints[v], axis=0 )/4.0)]

        edges = set()
        for elem in cells:
            for it,v in enumerate(elem):
                e = tuple( sorted([v,elem[(it+1)%len(elem)]]) )
                edges.add(e)
        self._nGlobalEdges = len(edges)

        # Construct Edge Connectivity
        edge_indices = {e:it for it,e in enumerate(edges)}
        self._local_edge_conn = []
        for elem in self._localConn:
            self._local_edge_conn.append([])
            for it,v in enumerate(elem):
                e = tuple( sorted([v,elem[(it+1)%len(elem)]]) )
                self._local_edge_conn[-1].append(edge_indices[e])

    def readGMsh(self, filename, boundaryNames = []):
        assert filename[-4:] == ".msh", f"Expected GMsh *.msh found *{filename[-4:]}"
        localMesh = None
        if self.mpiRank == 0:
            localMesh = meshio.read(filename, file_format="gmsh")
        mesh = self.mpiComm.bcast(localMesh, root=0)
        self._globalPoints = mesh.points

        # Decompose the domain over processors
        # Let MPI compute number of procs per dimension, then partition Lx Ly into that
        # many sub regions
        nBlockX, nBlockY = self.cartComm.dims
        xExtent = [ min([x[0] for x in mesh.points]), max([x[0] for x in mesh.points]) ]
        yExtent = [ min([x[1] for x in mesh.points]), max([x[1] for x in mesh.points]) ]
        xDomain = linspace(xExtent[0], xExtent[1], nBlockX+1)
        yDomain = linspace(yExtent[0], yExtent[1], nBlockY+1)

        # Get the Extent of the domain for this proc.
        # Use a cartesian topology to get the X,Y bounds for this proc
        myExtentX = [ xDomain[self.cartComm.coords[0]], xDomain[self.cartComm.coords[0]+1] ]
        myExtentY = [ yDomain[self.cartComm.coords[1]], yDomain[self.cartComm.coords[1]+1] ]

        # Partition the connectivity
        # Compute the centroid for a cell and determine whether it is bound by the current
        # processor's extent. Assume there is only one region in the mesh
        cells = [cell for cell in mesh.cells if cell.type == "quad"][0] # Reading in only quad elements, expecting only 1 region
        self._nGlobalElements = len(cells)
        myCell = lambda x: x[0] >= myExtentX[0] and x[0] < myExtentX[1] and x[1] >= myExtentY[0] and x[1] < myExtentY[1]
        self._localConn = [MeshCell2D(v,self._globalPoints) for v in cells.data if myCell(sum( mesh.points[v], axis=0 )/4.0)]

        # Calculate unique edge in mesh
        # Iterate over all edges in an element, collect sorted pairs in a set
        edges = set()
        for elem in cells.data:
            for it,v in enumerate(elem):
                e = tuple( sorted([v,elem[(it+1)%len(elem)]]) )
                edges.add(e)
        self._nGlobalEdges = len(edges)

        # Construct Edge Connectivity
        edge_indices = {e:it for it,e in enumerate(edges)}
        self._local_edge_conn = []
        for elem in self._localConn:
            self._local_edge_conn.append([])
            for it,v in enumerate(elem):
                e = tuple( sorted([v,elem[(it+1)%len(elem)]]) )
                self._local_edge_conn[-1].append(edge_indices[e])
        
        # Boundary sets
        # Boundaries show up as line-type cells. The order for the rectangle with hole
        # starts at the hole and then goes -y, -x, +x, +y. Labeling physical lines in
        # Gmsh disrupts the surface cell numbering for some reason.
        # boundaryNames = ["inner", "yneg", "xneg", "xpos", "ypos"] # imposed based on hole geometry
        boundary = [cell for cell in mesh.cells if cell.type == "line"]
        assert len(boundary) == len(boundaryNames), "Boundary names don't match"

        # Calculate the intersection between nodes contained in this proc's connectivity
        # and the nodes on a given boundary.
        nodeSet = set([vertex for elem in self._localConn for vertex in elem])
        self._boundaryNodes = {key: list(set([vertex for elem in lines.data for vertex in elem]).intersection(nodeSet)) for key,lines in zip(boundaryNames, boundary)}

    def writeVTKsolution(self, fileroot="solution", in_cell_data = {}, in_point_data = {}):
        self.__writePVTU(fileroot)

        # Add Domain decomposition data to output
        local_point_data = {key:value for key,value in in_point_data.items()}
        local_cell_data = {key:value for key,value in in_cell_data.items()}
        local_cell_data["Rank"] = [ array([self._mpiRank for _ in self._localConn], dtype="int64") ]

        # Create a new local mesh object
        newMesh = meshio.Mesh(
            self._globalPoints,
            [("quad", [v.connectivity for v in self._localConn])],
            cell_data=local_cell_data,
            point_data=local_point_data
        )
        newMesh.write(f"solution_{self._mpiRank:0>5d}.vtu")

    def __writePVTU(self, fileroot):
        # Only write out 1 pvtu file for all procs
        if self._mpiRank != 0:
            return

        # Output Header and parallel cell data
        data = [
            "<?xml version=\"1.0\"?>\n",
            "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n",
            "  <PUnstructuredGrid GhostLevel=\"0\">\n",
            "    <PPoints>\n",
            "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n",
            "    </PPoints>\n",
            "    <PCellData>\n",
            "      <PDataArray type=\"Int64\" Name=\"Rank\" format=\"binary\"/>\n",
            "    </PCellData>\n"
        ]

        # Output parallel pieces
        data += [f"    <Piece Source=\"{fileroot}_{it:0>5d}.vtu\"/>\n" for it in range(self._mpiSize)]

        # Output Footer
        data += [
            "  </PUnstructuredGrid>\n",
            "</VTKFile>\n"
        ]

        with open(fileroot+".pvtu", "w") as f:
            f.writelines(data)

class GeometryIterator():
    def __init__(self, geom):
        self._geom = geom
        self._index = 0

    def __next__(self):
        if self._index < len(self._geom.localConn):
            self._index += 1
            return self._geom.localConn[self._index-1]
        raise StopIteration

class MeshCell2D():
    def __init__(self, connectivity, points):
        self.vertices = [points[c] for c in connectivity]
        self.connectivity = [c for c in connectivity]

    def __iter__(self):
        return MeshCellIterator(self)

    def __repr__(self):
        values = ", ".join([str(s) for s in self.connectivity])
        return f"[{values}]"
    
    def __len__(self):
        return len(self.connectivity)

    def __getitem__(self, i):
        return self.connectivity[i]

    def getJacobian(self, xi, eta):
        db_dxi  = 0.25 * array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        db_deta = 0.25 * array([ -(1-xi), -(1+xi),  (1+xi),  (1-xi)])
        return array([[ db_dxi.dot(array([v[0] for v in self.vertices])),  db_dxi.dot(array([v[1] for v in self.vertices]))],
                      [db_deta.dot(array([v[0] for v in self.vertices])), db_deta.dot(array([v[1] for v in self.vertices]))]])

    def getInvJacobian(self, xi, eta):
        return inv(self.getJacobian(xi,eta))

    def getPhysicalLocation(self, xi, eta):
        mapping = 0.25 * array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        return mapping.dot(array([v[0] for v in self.vertices])), mapping.dot(array([v[1] for v in self.vertices]))

class MeshCellIterator():
    def __init__(self, meshCell):
        self._cell = meshCell
        self._index = 0

    def __next__(self):
        if self._index < len(self._cell.connectivity):
            self._index += 1
            return self._cell.connectivity[self._index-1]
        raise StopIteration