import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualise(func, x_range=(-5, 5), y_range=(-5, 5), resolution=100, point=None):
    """
    Draws a 3D surface plot of `func(x)` over the grid defined by x_range and y_range,
    and optionally plots a point on the surface.

    Parameters:
    -----------
    func : callable
        A function that takes a 2D numpy array [x, y] and returns a scalar.
    x_range : tuple of two floats, optional
        (min, max) range for x-axis.
    y_range : tuple of two floats, optional
        (min, max) range for y-axis.
    resolution : int, optional
        Number of points along each axis.
    point : array-like of shape (2,), optional
        If provided, should be [x1, x2], and this point will be plotted on the surface.

    Returns:
    --------
    None. Displays a 3D plot.
    """
    # Prepare grid
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda x, y: func(np.array([x, y])))(X, Y)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # If point provided, plot it
    if point is not None:
        px, py = point
        pz = func(np.array([px, py]))
        ax.scatter([px], [py], [pz], color='r', s=50)
        ax.text(px, py, pz, f'({px:.2f}, {py:.2f}, {pz:.2f})')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.show()