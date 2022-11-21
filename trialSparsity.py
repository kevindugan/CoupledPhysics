from numpy import linspace, array, zeros, cumsum
from scipy.sparse import csr_array
from matplotlib import pyplot as plt
import meshio

def run():
    # sparsity_1D()
    sparsity_2D()

def sparsity_1D():
    Nelem = 100
    qOrder = 1
    geom = linspace(1, 5, Nelem+1)

    pointsPerElement = qOrder+1
    connectivity = [[0]*pointsPerElement for _ in range(Nelem)]
    for i,_ in enumerate(connectivity):
        connectivity[i][0:2] = [i, i+1]
        start = i*(pointsPerElement-2) + Nelem+1
        end = start + (pointsPerElement-2)
        connectivity[i][2:] = list(range(start, end))

    connectivity = array(connectivity)

    globalSize = (Nelem+1) + Nelem*(qOrder-1)
    globalA = zeros((globalSize, globalSize))
    for elem in connectivity:
        for it in elem:
            for jt in elem:
                globalA[it,jt] = 1

    plt.figure()
    plt.spy(globalA, marker="o", markersize=5)
    plt.show()

def sparsity_2D():
    qOrder = 1
    # mesh = meshio.read("geometry2D_minimal.msh")
    # cells = [v for v in mesh.cells if v.type == "quad"][0].data
    # points = mesh.points

    Nx, Ny = 3,4
    xCoor = linspace(1, 5, Nx+1)
    yCoor = linspace(1, 4, Ny+1)
    points = array([[x,y] for y in yCoor for x in xCoor])
    cells = array([[jt*(Nx+1)+it, jt*(Nx+1)+it+1, (jt+1)*(Nx+1)+it+1, (jt+1)*(Nx+1)+it] for jt in range(Ny) for it in range(Nx)])

    pointsPerElement = qOrder+3
    connectivity = [[0]*pointsPerElement for _ in cells]
    for it,elem in enumerate(cells):
        connectivity[it][0:4] = elem
        start = len(points) + it*(pointsPerElement-4)
        end = start + (pointsPerElement-4)
        connectivity[it][4:] = list(range(start, end))

    connectivity = array(connectivity)

    # Global Matrix
    globalSize = len(points) + len(cells)*(qOrder-1)

    # Connectivity for renumbering
    graph = {it:set() for it in range(globalSize)}
    for elem in connectivity:
        for it in elem:
            for jt in elem:
                graph[it].add(jt)

    # DoF Renumbering
    dof_map = renumber(graph)
    renum_graph = {it:set() for it in range(globalSize)}
    for elem in connectivity:
        for it in elem:
            for jt in elem:
                renum_graph[dof_map[it]].add(dof_map[jt])

    # Setup Sparsity pattern
    col = [it for val in renum_graph.values() for it in val]
    rowPtr = [0] + list(cumsum([len(val) for val in renum_graph.values()]))
    globalA = csr_array((array([1]*len(col), dtype="int8"), col, rowPtr), shape=(globalSize,globalSize))

    plt.figure()
    plt.spy(globalA, marker="o", markersize=5)
    plt.show()
    
def renumber(graph):
    # return [it for it in graph.keys()] # Identity Mapping

    rank = [len(val) for val in graph.values()]
    start = rank.index(min(rank))

    newNumbering = [None for _ in graph.keys()]
    visited = set([start])
    queue = [start]
    count = 0
    while len(queue) > 0:
        it = queue.pop(0)
        newNumbering[it] = count
        count += 1
        for v in graph[it]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    return newNumbering

if __name__ == "__main__":
    run()