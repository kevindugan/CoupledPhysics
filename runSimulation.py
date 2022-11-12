from Physics.Geometry.Geometry2D import Geometry2D

def run():
    geom = Geometry2D()
    geom.readGMsh("geometry2D_fine.msh")
    geom.writeVTKsolution(fileroot="solution")

if __name__ == "__main__":
    run()