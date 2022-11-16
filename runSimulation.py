from Physics.Geometry.Geometry2D import Geometry2D

def run():
    geom = Geometry2D()
    geom.readGMsh(filename="geometry2D_coarse.msh",
        boundaryNames=["inner", "yneg", "xneg", "xpos", "ypos"])
    geom.writeVTKsolution(fileroot="solution")

    for elem in geom:
        if geom.mpiRank == 0:
            print(elem.getJacobian(-1,-1))

if __name__ == "__main__":
    run()