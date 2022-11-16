from Physics.Geometry.Geometry2D import Geometry2D

def run():
    geom = Geometry2D()
    geom.readGMsh(filename="Physics/Geometry/rectangleWithHole.msh",
        boundaryNames=["inner", "yneg", "xneg", "xpos", "ypos"])
    geom.writeVTKsolution(fileroot="solution")

if __name__ == "__main__":
    run()