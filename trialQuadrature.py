from Physics.Quadrature.Quadrature import Quadrature2D

def run():
    quad = Quadrature2D(order = 3)
    quad.plot_shape_functions(show_fig=True)

if __name__ == "__main__":
    run()