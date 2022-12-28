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
    # Hierarchical Quadrature:
    #       N global dofs = nPoints + nEdges*(q-1) + nElements*(q-1)**2
    # One DoF for each vertex, then one for every edge and order above linear, then bubble functions growing at (q-1)**2
    assert dofs.globalDoFsize == 9 + 12*(2-1) + 4*(2-1)**2
    # Expecting the dof map to be identity [ range(nDofs) ]
    assert dofs.dofMap == [it for it in range(25)]

def test_dof_renumbering():
    geom = Geometry2D()
    geom.readInternal(nX=2,nY=2)
    quad = Quadrature2D(order=2)

    dofs = DoFHandler(geom=geom, quadrature=quad)
    dofs.renumberDoFs()

    # DoF map should be breadth first of nodes from start
    #
    #   6 ---15--- 7 ---20--- 8
    #   |          |          |
    #  17    22    19   24    12
    #   |          |          |
    #   3 ---11--- 4 ---16--- 5
    #   |          |          |
    #  13    21    14   23    18
    #   |          |          |
    #   0 ----9--- 1 ---10--- 2
    #
    # Node Ordering: [0 1 3 4 9 11 13 14 21 2 5 10 16 18 23 6 7 15 17 19 22 8 12 20 24]           [0, 1, 3, 4, 9, 2, 5, 11, 6, 7, 10, 8, 12]
    # Map should be inverse of this ordering
    order = [0, 1, 3, 4, 9, 11, 13, 14, 21, 2, 5, 10, 16, 18, 23, 6, 7, 15, 17, 19, 22, 8, 12, 20, 24]
    expected = sorted([it for it in range(25)], key=lambda x: order[x])
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
        expectedRow = [0, 9, 18, 18, 27, 36, 36, 36, 36, 36, 45, 45, 54, 54, 63, 72, 72, 72, 72, 72, 72, 72, 81, 81, 81, 81]
        expectedCol = [it for _ in range(9) for it in [0, 1, 3, 4, 9, 11, 13, 14, 21]]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 1:
        expectedRow = [0, 0, 0, 0, 9, 18, 18, 27, 36, 36, 36, 36, 45, 45, 45, 45, 54, 54, 63, 63, 72, 72, 72, 81, 81, 81]
        expectedCol = [it for _ in range(9) for it in [3, 4, 6, 7, 11, 15, 17, 19, 22]]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 2:
        expectedRow = [0, 0, 9, 18, 18, 27, 36, 36, 36, 36, 36, 45, 45, 45, 45, 54, 54, 63, 63, 72, 72, 72, 72, 72, 81, 81]
        expectedCol = [it for _ in range(9) for it in [1, 2, 4, 5, 10, 14, 16, 18, 23]]
        assert expectedRow == row
        assert expectedCol == col
    elif geom.mpiRank == 3:
        expectedRow = [0, 0, 0, 0, 0, 9, 18, 18, 27, 36, 36, 36, 36, 45, 45, 45, 45, 54, 54, 54, 63, 72, 72, 72, 72, 81]
        expectedCol = [it for _ in range(9) for it in [4, 5, 7, 8, 12, 16, 19, 20, 24]]
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

    matVect = array([9, 18, 9, 18, 36, 18, 9, 18, 9, 9, 9, 18, 9, 9, 18, 9, 18, 9, 9, 18, 9, 9, 9, 9, 9])
    assert (globalB == matVect).all()

    # Check Renumbering
    dofs.renumberDoFs()
    dofs.buildSparsity()
    row, col = dofs.sparsity

    localA = csr_array((array([1]*len(col), dtype="int32"), col, row), shape=(dofs.globalDoFsize, dofs.globalDoFsize))
    globalX = array([1]*dofs.globalDoFsize, dtype="int32")
    localB = localA.dot(globalX)
    globalB = geom.mpiComm.allreduce(localB, op=MPI.SUM)

    # Permute matvec array based on renumbered node ordering
    order = sorted([it for it in range(len(dofs.dofMap))], key=lambda x: dofs.dofMap[x])
    matVec_renum = array([matVect[it] for it in order])
    assert (globalB == matVec_renum).all()