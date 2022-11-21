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
    geom.readGMsh(os.path.join(tmpdir, "test.msh"),
        boundaryNames=["inner", "yneg", "xneg", "xpos", "ypos"])
    if size == 1:
        assert len(geom.localConnectivity) == 18.0
    elif size == 2:
        if rank == 0:
            assert len(geom.localConnectivity) == 8.0
        elif rank == 1:
            assert len(geom.localConnectivity) == 10.0
    elif size == 3:
        if rank == 0:
            assert len(geom.localConnectivity) == 6.0
        elif rank == 1:
            assert len(geom.localConnectivity) == 4.0
        elif rank == 2:
            assert len(geom.localConnectivity) == 8.0
    elif size == 4:
        if rank == 0:
            assert len(geom.localConnectivity) == 4.0
        elif rank == 1:
            assert len(geom.localConnectivity) == 4.0
        elif rank == 2:
            assert len(geom.localConnectivity) == 5.0
        elif rank == 3:
            assert len(geom.localConnectivity) == 5.0


#====================================================================================
# Helper functions
#====================================================================================
def getTestGMshFile():
    return [
        "$MeshFormat\n",
        "4.1 0 8\n",
        "$EndMeshFormat\n",
        "$Entities\n",
        "5 5 1 0\n",
        "5 300 300 0 0 \n",
        "6 0 0 0 0 \n",
        "7 400 0 0 0 \n",
        "8 0 600 0 0 \n",
        "9 400 600 0 0 \n",
        "5 99.99999990000001 199.9999999 -1e-07 300.0000001 400.0000001 1e-07 0 2 5 -5 \n",
        "6 -9.999999406318238e-08 -1e-07 -1e-07 400.0000001 1e-07 1e-07 0 2 6 -7 \n",
        "7 -1e-07 -1.000000224848918e-07 -1e-07 1e-07 600.0000001 1e-07 0 2 8 -6 \n",
        "8 399.9999999 -1.000000224848918e-07 -1e-07 400.0000001 600.0000001 1e-07 0 2 7 -9 \n",
        "9 -9.999999406318238e-08 599.9999999 -1e-07 400.0000001 600.0000001 1e-07 0 2 9 -8 \n",
        "1 -9.999999406318238e-08 -1.000000224848918e-07 -1e-07 400.0000001 600.0000001 1e-07 0 5 6 8 9 7 5 \n",
        "$EndEntities\n",
        "$Nodes\n",
        "11 28 1 28\n",
        "0 5 0 1\n",
        "1\n",
        "300 300 0\n",
        "0 6 0 1\n",
        "2\n",
        "0 0 0\n",
        "0 7 0 1\n",
        "3\n",
        "400 0 0\n",
        "0 8 0 1\n",
        "4\n",
        "0 600 0\n",
        "0 9 0 1\n",
        "5\n",
        "400 600 0\n",
        "1 5 0 7\n",
        "6\n",
        "7\n",
        "8\n",
        "9\n",
        "10\n",
        "11\n",
        "12\n",
        "270.7106781186548 370.7106781186548 0\n",
        "200 400 0\n",
        "129.2893218813452 370.7106781186548 0\n",
        "100 300 0\n",
        "129.2893218813452 229.2893218813452 0\n",
        "200 200 0\n",
        "270.7106781186548 229.2893218813452 0\n",
        "1 6 0 1\n",
        "13\n",
        "200 0 0\n",
        "1 7 0 3\n",
        "14\n",
        "15\n",
        "16\n",
        "0 450 0\n",
        "0 300 0\n",
        "0 150 0\n",
        "1 8 0 3\n",
        "17\n",
        "18\n",
        "19\n",
        "400 150 0\n",
        "400 300 0\n",
        "400 450 0\n",
        "1 9 0 1\n",
        "20\n",
        "200 600 0\n",
        "2 1 0 8\n",
        "21\n",
        "22\n",
        "23\n",
        "24\n",
        "25\n",
        "26\n",
        "27\n",
        "28\n",
        "284.1201619439036 152.6017742227093 0\n",
        "284.122687323472 447.3971797308233 0\n",
        "116.1333879374688 447.4089387035017 0\n",
        "116.112263106978 152.5823111052062 0\n",
        "312.0412020593913 375.5759683637214 0\n",
        "88.16057859347984 375.5798118136721 0\n",
        "88.04757763488278 224.3733816567158 0\n",
        "312.0412020570409 224.424031635305 0\n",
        "$EndNodes\n",
        "$Elements\n",
        "11 43 1 43\n",
        "0 5 15 1\n",
        "1 1 \n",
        "0 6 15 1\n",
        "2 2 \n",
        "0 7 15 1\n",
        "3 3 \n",
        "0 8 15 1\n",
        "4 4 \n",
        "0 9 15 1\n",
        "5 5 \n",
        "1 5 1 8\n",
        "6 1 6 \n",
        "7 6 7 \n",
        "8 7 8 \n",
        "9 8 9 \n",
        "10 9 10 \n",
        "11 10 11 \n",
        "12 11 12 \n",
        "13 12 1 \n",
        "1 6 1 2\n",
        "14 2 13 \n",
        "15 13 3 \n",
        "1 7 1 4\n",
        "16 4 14 \n",
        "17 14 15 \n",
        "18 15 16 \n",
        "19 16 2 \n",
        "1 8 1 4\n",
        "20 3 17 \n",
        "21 17 18 \n",
        "22 18 19 \n",
        "23 19 5 \n",
        "1 9 1 2\n",
        "24 5 20 \n",
        "25 20 4 \n",
        "2 1 3 18\n",
        "26 13 21 11 24 \n",
        "27 20 23 7 22 \n",
        "28 18 28 21 17 \n",
        "29 15 26 23 14 \n",
        "30 18 19 22 25 \n",
        "31 15 16 24 27 \n",
        "32 15 27 10 9 \n",
        "33 18 25 6 1 \n",
        "34 18 1 12 28 \n",
        "35 15 9 8 26 \n",
        "36 28 12 11 21 \n",
        "37 27 24 11 10 \n",
        "38 25 22 7 6 \n",
        "39 26 8 7 23 \n",
        "40 3 17 21 13 \n",
        "41 24 16 2 13 \n",
        "42 22 19 5 20 \n",
        "43 4 14 23 20 \n",
        "$EndElements\n"
    ]