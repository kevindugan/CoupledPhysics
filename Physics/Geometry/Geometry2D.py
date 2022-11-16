import meshio
from numpy import linspace, sum, array
from mpi4py import MPI

class Geometry2D():
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.mpiRank = self.comm.Get_rank()
        self.mpiSize = self.comm.Get_size()

        # Parallel Decomposition
        nBlockX, nBlockY = MPI.Compute_dims(self.mpiSize, 2)
        self.cartComm = self.comm.Create_cart([nBlockX, nBlockY], [False, False], True)

    def readGMsh(self, filename, boundaryNames = []):
        assert filename[-4:] == ".msh", f"Expected GMsh *.msh found *{filename[-4:]}"
        mesh = meshio.read(filename)
        self.globalPoints = mesh.points

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
        myCell = lambda x: x[0] >= myExtentX[0] and x[0] < myExtentX[1] and x[1] >= myExtentY[0] and x[1] < myExtentY[1]
        self.localConn = [v for v in cells.data if myCell(sum( mesh.points[v], axis=0 )/4.0)]
        
        # Boundary sets
        # Boundaries show up as line-type cells. The order for the rectangle with hole
        # starts at the hole and then goes -y, -x, +x, +y. Labeling physical lines in
        # Gmsh disrupts the surface cell numbering for some reason.
        # boundaryNames = ["inner", "yneg", "xneg", "xpos", "ypos"] # imposed based on hole geometry
        boundary = [cell for cell in mesh.cells if cell.type == "line"]
        assert len(boundary) == len(boundaryNames), "Boundary names don't match"

        # Calculate the intersection between nodes contained in this proc's connectivity
        # and the nodes on a given boundary.
        nodeSet = set([vertex for elem in self.localConn for vertex in elem])
        self.boundaryNodes = {key: list(set([vertex for elem in lines.data for vertex in elem]).intersection(nodeSet)) for key,lines in zip(boundaryNames, boundary)}
        # printBoundary = {key: [v+1 for v in value] for key,value in self.boundaryNodes.items()}
        # print(f"Rank {self.mpiRank}: {printBoundary}")

    def writeVTKsolution(self, fileroot="solution", in_cell_data = {}, in_point_data = {}):
        self.__writePVTU(fileroot)

        # Add Domain decomposition data to output
        local_point_data = {key:value for key,value in in_point_data.items()}
        local_cell_data = {key:value for key,value in in_cell_data.items()}
        local_cell_data["Rank"] = [ array([self.mpiRank for _ in self.localConn], dtype="int64") ]

        # Create a new local mesh object
        newMesh = meshio.Mesh(
            self.globalPoints,
            [("quad", self.localConn)],
            cell_data=local_cell_data,
            point_data=local_point_data
        )
        newMesh.write(f"solution_{self.mpiRank:0>5d}.vtu")

    def __writePVTU(self, fileroot):
        # Only write out 1 pvtu file for all procs
        if self.mpiRank != 0:
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
        data += [f"    <Piece Source=\"{fileroot}_{it:0>5d}.vtu\"/>\n" for it in range(self.mpiSize)]

        # Output Footer
        data += [
            "  </PUnstructuredGrid>\n",
            "</VTKFile>\n"
        ]

        with open(fileroot+".pvtu", "w") as f:
            f.writelines(data)