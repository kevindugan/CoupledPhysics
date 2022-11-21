from ..DoFHandler import DoFHandler
from ...Geometry.Geometry2D import Geometry2D
from ...Quadrature.Quadrature import Quadrature2D
import pytest
from scipy.sparse import csr_array
from numpy import array

def test_dof_constructor():
    geom = Geometry2D()
    geom.readInternal(nX=2,nY=2)
    quad = Quadrature2D(order=2)

    dofs = DoFHandler(geom=geom, quadrature=quad)
    # Hierarchical Quadrature: N global dofs should be nPoints + nElements*(q-1)
    # One DoF for each vertex, then One for every element and order above linear
    assert dofs.globalDoFsize == 9 + 4*(2-1)
    # Expecting the dof map to be identity [ range(nDofs) ]
    assert dofs.dofMap == [it for it in range(13)]

def test_dof_renumbering():
    geom = Geometry2D()
    geom.readInternal(nX=2,nY=2)
    quad = Quadrature2D(order=2)

    dofs = DoFHandler(geom=geom, quadrature=quad)
    dofs.renumberDoFs()

    # DoF map should be breadth first of nodes from start
    #
    #   6 ----- 7 ----- 8
    #   |  10   |   12  |
    #   3 ----- 4 ----- 5
    #   |   9   |   11  |
    #   0 ----- 1 ----- 2
    #
    # Node Ordering: [0, 1, 3, 4, 9, 2, 5, 11, 6, 7, 10, 8, 12]
    # Map should be inverse of this ordering
    order = [0, 1, 3, 4, 9, 2, 5, 11, 6, 7, 10, 8, 12]
    expected = sorted([it for it in range(13)], key=lambda x: order[x])
    assert dofs.dofMap == expected

@pytest.mark.mpi(min_size=4, max_size=4)
def test_sparsity():
    geom = Geometry2D()
    geom.readInternal(nX=2,nY=2)
    quad = Quadrature2D(order=2)

    dofs = DoFHandler(geom=geom, quadrature=quad)
    dofs.buildSparsity()
    row, col = dofs.sparsity

    if geom.mpiRank == 0:
        expectedRow = [0, 5, 10, 10, 15, 20, 20, 20, 20, 20, 25, 25, 25, 25]
        expectedCol = [0, 1, 3, 4, 9, 0, 1, 3, 4, 9, 0, 1, 3, 4, 9, 0, 1, 3, 4, 9, 0, 1, 3, 4, 9]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 1:
        expectedRow = [0, 0, 0, 0, 5, 10, 10, 15, 20, 20, 20, 25, 25, 25]
        expectedCol = [3, 4, 6, 7, 10, 3, 4, 6, 7, 10, 3, 4, 6, 7, 10, 3, 4, 6, 7, 10, 3, 4, 6, 7, 10]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 2:
        expectedRow = [0, 0, 5, 10, 10, 15, 20, 20, 20, 20, 20, 20, 25, 25]
        expectedCol = [1, 2, 4, 5, 11, 1, 2, 4, 5, 11, 1, 2, 4, 5, 11, 1, 2, 4, 5, 11, 1, 2, 4, 5, 11]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 3:
        expectedRow = [0, 0, 0, 0, 0, 5, 10, 10, 15, 20, 20, 20, 20, 25]
        expectedCol = [4, 5, 7, 8, 12, 4, 5, 7, 8, 12, 4, 5, 7, 8, 12, 4, 5, 7, 8, 12, 4, 5, 7, 8, 12]
        assert expectedRow == row
        assert expectedCol == col

@pytest.mark.mpi(max_size=4)
def test_sparse_mult():
    geom = Geometry2D()
    geom.readInternal(nX=2,nY=2)
    quad = Quadrature2D(order=2)

    dofs = DoFHandler(geom=geom, quadrature=quad)
    dofs.buildSparsity()
    row, col = dofs.sparsity

    # Construct local sparse matrix, and global x vector
    localA = csr_array((array([1]*len(col), dtype="int32"), col, row), shape=(dofs.globalDoFsize, dofs.globalDoFsize))
    globalX = array([1]*dofs.globalDoFsize, dtype="int32")

    # Matrix-vector product produces local result
    localB = localA.dot(globalX)

    # Collect global result
    from mpi4py import MPI
    globalB = geom.mpiComm.allreduce(localB, op=MPI.SUM)

    assert (globalB == array([5, 10, 5, 10, 20, 10, 5, 10, 5, 5, 5, 5, 5])).all()
