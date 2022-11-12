from ..Geometry2D import Geometry2D
import pytest, os, time

@pytest.mark.mpi(max_size=5)
def test_GMshConstructor():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    geom = Geometry2D()
    if size == 1:
        assert geom.cartComm.dims == [1,1]
    if size == 2:
        assert geom.cartComm.dims == [2,1]
    if size == 3:
        assert geom.cartComm.dims == [3,1]
    if size == 4:
        assert geom.cartComm.dims == [2,2]
    if size == 5:
        assert geom.cartComm.dims == [5,1]
    
@pytest.mark.mpi(max_size=4)
def test_readGmshGeom(tmpdir):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Write Input Meshfile
    with open(os.path.join(tmpdir, "test.msh"), "w") as f:
        f.writelines(getTestGMshFile())

    geom = Geometry2D()
    geom.readGMsh(os.path.join(tmpdir, "test.msh"))
    if size == 1:
        assert len(geom.localConn) == 18.0
    elif size == 2:
        if rank == 0:
            assert len(geom.localConn) == 8.0
        elif rank == 1:
            assert len(geom.localConn) == 10.0
    elif size == 3:
        if rank == 0:
            assert len(geom.localConn) == 6.0
        elif rank == 1:
            assert len(geom.localConn) == 4.0
        elif rank == 2:
            assert len(geom.localConn) == 8.0
    elif size == 4:
        if rank == 0:
            assert len(geom.localConn) == 4.0
        elif rank == 1:
            assert len(geom.localConn) == 4.0
        elif rank == 2:
            assert len(geom.localConn) == 5.0
        elif rank == 3:
            assert len(geom.localConn) == 5.0


#====================================================================================
# Helper functions
#====================================================================================
def getTestGMshFile():
    return [
        "$MeshFormat\n",
        "2.2 0 8\n",
        "$EndMeshFormat\n",
        "$Nodes\n",
        "28\n",
        "1 300 300 0\n",
        "2 0 0 0\n",
        "3 400 0 0\n",
        "4 0 600 0\n",
        "5 400 600 0\n",
        "6 270.7106781186548 370.7106781186548 0\n",
        "7 200 400 0\n",
        "8 129.2893218813452 370.7106781186548 0\n",
        "9 100 300 0\n",
        "10 129.2893218813452 229.2893218813452 0\n",
        "11 200 200 0\n",
        "12 270.7106781186548 229.2893218813452 0\n",
        "13 200 0 0\n",
        "14 0 450 0\n",
        "15 0 300 0\n",
        "16 0 150 0\n",
        "17 400 150 0\n",
        "18 400 300 0\n",
        "19 400 450 0\n",
        "20 200 600 0\n",
        "21 284.1201619439036 152.6017742227093 0\n",
        "22 284.122687323472 447.3971797308233 0\n",
        "23 116.1333879374688 447.4089387035017 0\n",
        "24 116.112263106978 152.5823111052062 0\n",
        "25 312.0412020593913 375.5759683637214 0\n",
        "26 88.16057859347984 375.5798118136721 0\n",
        "27 88.04757763488278 224.3733816567158 0\n",
        "28 312.0412020570409 224.424031635305 0\n",
        "$EndNodes\n",
        "$Elements\n",
        "43\n",
        "1 15 2 0 5 1\n",
        "2 15 2 0 6 2\n",
        "3 15 2 0 7 3\n",
        "4 15 2 0 8 4\n",
        "5 15 2 0 9 5\n",
        "6 1 2 0 5 1 6\n",
        "7 1 2 0 5 6 7\n",
        "8 1 2 0 5 7 8\n",
        "9 1 2 0 5 8 9\n",
        "10 1 2 0 5 9 10\n",
        "11 1 2 0 5 10 11\n",
        "12 1 2 0 5 11 12\n",
        "13 1 2 0 5 12 1\n",
        "14 1 2 0 6 2 13\n",
        "15 1 2 0 6 13 3\n",
        "16 1 2 0 7 4 14\n",
        "17 1 2 0 7 14 15\n",
        "18 1 2 0 7 15 16\n",
        "19 1 2 0 7 16 2\n",
        "20 1 2 0 8 3 17\n",
        "21 1 2 0 8 17 18\n",
        "22 1 2 0 8 18 19\n",
        "23 1 2 0 8 19 5\n",
        "24 1 2 0 9 5 20\n",
        "25 1 2 0 9 20 4\n",
        "26 3 2 0 1 13 21 11 24\n",
        "27 3 2 0 1 20 23 7 22\n",
        "28 3 2 0 1 18 28 21 17\n",
        "29 3 2 0 1 15 26 23 14\n",
        "30 3 2 0 1 18 19 22 25\n",
        "31 3 2 0 1 15 16 24 27\n",
        "32 3 2 0 1 15 27 10 9\n",
        "33 3 2 0 1 18 25 6 1\n",
        "34 3 2 0 1 18 1 12 28\n",
        "35 3 2 0 1 15 9 8 26\n",
        "36 3 2 0 1 28 12 11 21\n",
        "37 3 2 0 1 27 24 11 10\n",
        "38 3 2 0 1 25 22 7 6\n",
        "39 3 2 0 1 26 8 7 23\n",
        "40 3 2 0 1 3 17 21 13\n",
        "41 3 2 0 1 24 16 2 13\n",
        "42 3 2 0 1 22 19 5 20\n",
        "43 3 2 0 1 4 14 23 20\n",
        "$EndElements\n"
    ]