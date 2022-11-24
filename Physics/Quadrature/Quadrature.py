from numpy import sqrt, array, linspace, zeros, einsum, meshgrid, ravel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

class QuadratureBase():
    def __init__(self, order=1):
        self._order = order
        if self._order == 1:
            self.q = [  [-1.0/sqrt(3.0), 1.0],
                        [ 1.0/sqrt(3.0), 1.0]
                     ]
        elif self._order == 2:
            self.q = [  [-sqrt(3.0/5.0), 5.0/9.0],
                        [ 0.0,              8.0/9.0],
                        [ sqrt(3.0/5.0), 5.0/9.0]
                     ]
        elif self._order == 3:
            self.q = [  [-sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0)), (18.0 - sqrt(30.0))/36.0],
                        [-sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0)), (18.0 + sqrt(30.0))/36.0],
                        [ sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0)), (18.0 + sqrt(30.0))/36.0],
                        [ sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0)), (18.0 - sqrt(30.0))/36.0]
                     ]
        else:
            raise RuntimeError("Unsupported Quadrature Order: {0:2g}".format(self._order))

    def __iter__(self):
        return QuadratureIterator(self)

    @property
    def order(self):
        return self._order

    # Hierarchical shape functions from Lagrange polynomials
    def get_local_shape_vector(self, quadrature_point):
        if self._order == 1:
            return 0.5 * array([1.0 + quadrature_point,
                                1.0 - quadrature_point
                               ])
        elif self._order == 2:
            return 0.5 * array([1.0 + quadrature_point,
                                1.0 - quadrature_point,
                                2.0 * (1.0 - quadrature_point**2)
                               ])
        elif self._order == 3:
            return 0.5 * array([1.0 + quadrature_point,
                                1.0 - quadrature_point,
                                2.0 * (1.0 - quadrature_point**2),
                                1.125 * (1.0 - 3.0*quadrature_point - quadrature_point**2 + 3.0*quadrature_point**3)
                               ])
        else:
            raise RuntimeError("Unsupported Quadrature Order: {0:2g}".format(self._order))

    # Gradients of shape functions
    def get_local_grad_shape_vector(self, quadrature_point):
        if self._order == 1:
            return 0.5 * array([ 1,
                                -1
                               ])
        elif self._order == 2:
            return 0.5 * array([ 1.0,
                                -1.0,
                                -4.0 * quadrature_point
                               ])
        elif self._order == 3:
            return 0.5 * array([ 1.0,
                                -1.0,
                                -4.0 * quadrature_point,
                                -1.125 * (3.0 + 2.0*quadrature_point - 9.0*quadrature_point**2)
                               ])
        else:
            raise RuntimeError("Unsupported Quadrature Order: {0:2g}".format(self._order))

    def plot_shape_functions(self, show_fig=False, save_fig=False):
        ref_geom = linspace(-1, 1, 100)
        shape = zeros((100, self._order+1))

        for i, x in enumerate(ref_geom):
            q = self.get_local_shape_vector(x)
            shape[i,:] = q

        plt.figure(figsize=(16,9))
        plt.title("Basis Functions", fontdict={'fontsize':18,'fontweight':4})
        for i in range(self._order+1):
            plt.plot(ref_geom, shape[:,i])

        plt.plot(1.5*ref_geom, 0.0*ref_geom, 'k', alpha=0.3)
        plt.xlabel("Reference Domain", fontdict={'fontsize':16,'fontweight':4})
        plt.tick_params(axis='both', labelsize=12)
        plt.xlim([-1.1, 1.1])

        if save_fig:
            plt.savefig("quadrature.png", bbox_inches='tight', pad_inches=0.1, orientation='landscape')

        if show_fig:
            plt.show()


class Quadrature1D(QuadratureBase):
    def __init__(self, order=1):
        super().__init__(order)
        self.dim = 1

    @property
    def nBasisFunctions(self):
        return len(self.get_local_shape_vector(0))

    def get_local_shape_vector(self, quadrature_point):
        return super().get_local_shape_vector(quadrature_point)

    def get_local_grad_shape_vector(self, quadrature_point):
        return super().get_local_grad_shape_vector(quadrature_point)

    def plot_shape_functions(self, show_fig=False, save_fig=False):
        return super().plot_shape_functions(show_fig, save_fig)

class Quadrature2D(QuadratureBase):
    def __init__(self, order=1):
        super().__init__(order)
        self.dim = 2

    @property
    def nBasisFunctions(self):
        return len(self.get_local_shape_vector((0,0)))

    def get_local_shape_vector(self, quadrature_point):
        x = super().get_local_shape_vector(quadrature_point[0])
        y = super().get_local_shape_vector(quadrature_point[1])
        return ravel(einsum("i,j->ij", x.ravel(), y.ravel()))

    def get_local_grad_shape_vector(self, quadrature_point):
        x = super().get_local_grad_shape_vector(quadrature_point[0])
        y = super().get_local_grad_shape_vector(quadrature_point[1])
        return ravel(einsum("i,j->ij", x.ravel(), y.ravel()))

    def plot_shape_functions(self, show_fig=False, save_fig=False):
        x = y = linspace(-1, 1, 100)
        ref_geom = meshgrid(x, y)

        nFunctions = len(self.get_local_shape_vector((0,0)))
        for it in range(nFunctions):
            z = array([self.get_local_shape_vector((x,y))[it] for x,y in zip(ravel(ref_geom[0]), ravel(ref_geom[1]))])
            z = z.reshape(ref_geom[0].shape)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(ref_geom[0], ref_geom[1], z)

            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([0, 1])
            ax.grid(False)

            if show_fig:
                plt.show()

# Iterator class for ease-of-use of Quadrature formula
class QuadratureIterator():
    def __init__(self, quadrature):
        self._quadrature = quadrature
        self._index = 0
        # Generate tensor product of Gaussian Quadrature points for multi-dimensional
        if self._quadrature.dim == 1:
            self._local_coords = [x for x in range(len(self._quadrature.q))]
        elif self._quadrature.dim == 2:
            self._local_coords = [(x,y) for x in range(len(self._quadrature.q)) for y in range(len(self._quadrature.q))]
        elif self._quadrature.dim == 3:
            self._local_coords = [(x,y,z) for x in range(len(self._quadrature.q)) for y in range(len(self._quadrature.q)) for z in range(len(self._quadrature.q))]
    def __next__(self):
        if self._index < len(self._local_coords):
            self._index += 1
            # Decompose local coordinates
            if self._quadrature.dim == 1:
                xCoord = self._local_coords[self._index-1]
                xi, xwi = self._quadrature.q[xCoord]
                return xi, xwi
            elif self._quadrature.dim == 2:
                xCoord = self._local_coords[self._index-1][0]
                yCoord = self._local_coords[self._index-1][1]
                xi, xwi = self._quadrature.q[xCoord]
                yi, ywi = self._quadrature.q[yCoord]
                return xi, yi, xwi*ywi
            elif self._quadrature.dim == 3:
                xCoord = self._local_coords[self._index-1][0]
                yCoord = self._local_coords[self._index-1][1]
                zCoord = self._local_coords[self._index-1][2]
                xi, xwi = self._quadrature.q[xCoord]
                yi, ywi = self._quadrature.q[yCoord]
                zi, zwi = self._quadrature.q[zCoord]
                return xi, yi, zi, xwi*ywi*zwi
        raise StopIteration