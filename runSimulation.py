import meshio
from numpy import linspace, sum, array
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

def run():
    mesh = meshio.read("geometry2D_fine.msh")

    # Decompose the domain over processors
    # Let MPI compute number of procs per dimension, then partition Lx Ly into that
    # many sub regions
    nBlockX, nBlockY = MPI.Compute_dims(mpiSize, 2)
    xExtent = [ min([x[0] for x in mesh.points]), max([x[0] for x in mesh.points]) ]
    yExtent = [ min([x[1] for x in mesh.points]), max([x[1] for x in mesh.points]) ]
    xDomain = linspace(xExtent[0], xExtent[1], nBlockX+1)
    yDomain = linspace(yExtent[0], yExtent[1], nBlockY+1)

    # Get the Extent of the domain for this proc.
    # Use a cartesian topology to get the X,Y bounds for this proc
    cartComm = comm.Create_cart([nBlockX, nBlockY], [False, False], True)
    myExtentX = [ xDomain[cartComm.coords[0]], xDomain[cartComm.coords[0]+1] ]
    myExtentY = [ yDomain[cartComm.coords[1]], yDomain[cartComm.coords[1]+1] ]

    # Partition the connectivity
    # Compute the centroid for a cell and determine whether it is bound by the current
    # processor's extent. Assume there is only one region in the mesh
    cells = [cell for cell in mesh.cells if cell.type == "quad"][0] # Reading in only quad elements, expecting only 1 region
    myCell = lambda x: x[0] >= myExtentX[0] and x[0] < myExtentX[1] and x[1] >= myExtentY[0] and x[1] < myExtentY[1]
    myConnectivity = [v for v in cells.data if myCell(sum( mesh.points[v], axis=0 )/4.0)]

    # Write the mesh files with the rank solution
    writePVTU("solution")
    newMesh = meshio.Mesh(mesh.points, [("quad", myConnectivity)],
        cell_data={
            "Rank": [ array([mpiRank for _ in myConnectivity], dtype="int64") ]
        }
    )
    newMesh.write(f"solution_{mpiRank:0>5d}.vtu")

def writePVTU(fileroot):
    # Only write out 1 pvtu file for all procs
    if mpiRank != 0:
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
    data += [f"    <Piece Source=\"{fileroot}_{it:0>5d}.vtu\"/>\n" for it in range(mpiSize)]

    # Output Footer
    data += [
        "  </PUnstructuredGrid>\n",
        "</VTKFile>\n"
    ]

    with open(fileroot+".pvtu", "w") as f:
        f.writelines(data)

if __name__ == "__main__":
    run()