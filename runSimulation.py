from Physics.Geometry.Geometry2D import Geometry2D
from Physics.Quadrature.Quadrature import Quadrature2D
from Physics.DoFHandler.DoFHandler import DoFHandler

def run():
    geom = Geometry2D()

    # Read in GMsh
    # geom.readGMsh(filename="geometry2D_minimal.msh",
    #     boundaryNames=["inner", "yneg", "xneg", "xpos", "ypos"])
    geom.readInternal(xExtent=(1,4), nX=2, yExtent=(1,5), nY=2)
    geom.writeVTKsolution(fileroot="solution")

    # Sparsity Pattern
    quad = Quadrature2D(order=2)
    dofs = DoFHandler(geom, quad)
    dofs.buildSparsity()
    dofs.plotSparsity()
    # Renumbering
    dofs.renumberDoFs()
    dofs.buildSparsity()
    dofs.plotSparsity(fileroot="dof_sparsity_renum")

if __name__ == "__main__":
    run()