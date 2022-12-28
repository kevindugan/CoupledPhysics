from Physics.Geometry.Geometry2D import Geometry2D
from Physics.Quadrature.Quadrature import Quadrature2D
from Physics.DoFHandler.DoFHandler import DoFHandler

from numpy import array, zeros, outer
from scipy.sparse import csr_array

def run():
    geom = Geometry2D()

    # Read in GMsh
    # geom.readGMsh(filename="geometry2D_fine.msh",
    #     boundaryNames=["inner", "yneg", "xneg", "xpos", "ypos"])
    geom.readInternal(xExtent=(1,4), nX=4, yExtent=(1,5), nY=2)
    geom.writeVTKsolution(fileroot="solution")

    # Setup DoFs
    quad = Quadrature2D(order=2)
    dofs = DoFHandler(geom, quad)
    # dofs.renumberDoFs()
    dofs.buildSparsity()
    dofs.plotSparsity()

    return

    # Allocate Space for Matrix & RHS
    row, col = dofs.sparsity
    localA = csr_array(
                    (array([0]*len(col), dtype="float64"), col, row),
                    shape=(dofs.globalDoFsize,dofs.globalDoFsize)
                )
    localRHS = array([0]*dofs.globalDoFsize, dtype="float64")

    # Construct Matrix & RHS
    for it,elem in enumerate(dofs.dofConnectivity):
        local_mat = zeros((quad.order+1, quad.order+1))
        local_rhs = zeros(quad.order+1)

        for q_xi, q_eta, q_w in quad:
            # Element to Reference Transformation
            J = geom.localConnectivity[it].getJacobian(q_xi, q_eta)
            J_inv = geom.localConnectivity[it].getInvJacobian(q_xi, q_eta)

            # Local Matrix Contribution [Stiffness]
            print(quad.get_local_grad_shape_vector((q_xi,q_eta)).shape)
            stiffness = outer(quad.get_local_grad_shape_vector((q_xi,q_eta)), quad.get_local_grad_shape_vector((q_xi,q_eta)))
            local_mat += J_inv * stiffness * q_w

            # Local RHS Contribution
            x_q = J_inv.dot([q_xi, q_eta])
            Q_q = calculate_Q(x_q)
            local_rhs += Q_q * J * quad.get_local_shape_vector((q_xi,q_eta)) * q_w

        # Update global system
        for it,rhs in enumerate(local_rhs):
            for jt,val in enumerate(local_mat[it]):
                localA[dofs.dofMap[it],dofs.dofMap[jt]] += val
            localRHS[dofs.dofMap[it]] += rhs

    # Dirichelt Boundary Conditions
    # TODO
            
    if geom.mpiRank == 0:
        print(localA)

def calculate_Q(point3d):
    return 11.2
            
if __name__ == "__main__":
    run()