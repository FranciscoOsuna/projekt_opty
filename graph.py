import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.patches as patches


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


def animate(func, history, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    history = np.array(history)
    fitness = np.array([func(p) for p in history])

    # Grid for contour plot
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 5, height_ratios=[2, 1, 0.3])
    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[1, :4])
    ax_status = fig.add_subplot(gs[0:2, 4])
    ax_status.axis('off')
    rect = patches.FancyBboxPatch((0, 0), 1, 1,
                                  boxstyle="round,pad=0.02",
                                  linewidth=1,
                                  edgecolor='gray',
                                  facecolor='lightgray',
                                  transform=ax_status.transAxes,
                                  zorder=0)
    ax_status.add_patch(rect)
    status_text = ax_status.text(0.05, 0.95, '', va='top', ha='left', fontsize=10, wrap=True, transform=ax_status.transAxes)

    # Contour plot and path
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1)
    point, = ax1.plot([], [], 'ro-', label="Path")
    title = ax1.set_title("Optimization Path - Generation 0")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # Fitness chart
    ax2.set_xlim(0, len(history) - 1)
    ax2.set_ylim(np.min(fitness) * 0.9, np.max(fitness) * 1.1)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    fitness_line, = ax2.plot([], [], 'b-', label="Fitness")
    ax2.legend()

    # State
    current_frame = [0]
    playing = [False]
    log_scale = [False]

    def update_plot(frame):
        path = history[:frame + 1]
        point.set_data(path[:, 0], path[:, 1])
        fitness_line.set_data(range(frame + 1), fitness[:frame + 1])
        title.set_text(f"Optimization Path - Generation {frame}")
        ax2.set_yscale("log" if log_scale[0] else "linear")
        current_point = history[frame]
        current_fitness = fitness[frame]
        status_text.set_text(
            f"Generation: {frame}\n"
            f"Position: [{current_point[0]:.4f}, {current_point[1]:.4f}]\n"
            f"Fitness: {current_fitness:.6f}"
        )
        fig.canvas.draw_idle()

    def init():
        point.set_data([], [])
        fitness_line.set_data([], [])
        return point, fitness_line, title

    def update(frame):
        if playing[0]:
            if current_frame[0] < len(history) - 1:
                current_frame[0] += 1
                update_plot(current_frame[0])
        return point, fitness_line, title

    ani = animation.FuncAnimation(
        fig, update,
        init_func=init,
        frames=len(history),
        interval=300,
        blit=False,
        repeat=False,
        save_count=len(history),
        cache_frame_data=False  # Suppress caching warning
    )
    ani.event_source.stop()  # Start paused

    # Buttons
    ax_play = plt.axes([0.32, 0.01, 0.15, 0.05])
    ax_back = plt.axes([0.2, 0.01, 0.1, 0.05])
    ax_forward = plt.axes([0.49, 0.01, 0.1, 0.05])
    ax_log = plt.axes([0.65, 0.01, 0.15, 0.05])
    ax_reset = plt.axes([0.05, 0.01, 0.1, 0.05])

    b_play = Button(ax_play, 'Play')
    b_back = Button(ax_back, 'Back')
    b_forward = Button(ax_forward, 'Forward')
    b_log = Button(ax_log, 'Log scale: Off')
    b_reset = Button(ax_reset, 'Reset')

    def on_play(event):
        playing[0] = not playing[0]
        if playing[0]:
            ani.event_source.start()
            b_play.label.set_text('Pause')
        else:
            ani.event_source.stop()
            b_play.label.set_text('Play')

    def on_back(event):
        if current_frame[0] > 0:
            current_frame[0] -= 1
            update_plot(current_frame[0])
            if playing[0]:
                ani.event_source.stop()
                playing[0] = False
                b_play.label.set_text('Play')

    def on_forward(event):
        if current_frame[0] < len(history) - 1:
            current_frame[0] += 1
            update_plot(current_frame[0])
            if playing[0]:
                ani.event_source.stop()
                playing[0] = False
                b_play.label.set_text('Play')

    def on_log(event):
        if np.any(fitness <= 0):
            b_log.label.set_text('Log scale: N/A')
            return
        log_scale[0] = not log_scale[0]
        b_log.label.set_text('Log scale: On' if log_scale[0] else 'Log scale: Off')
        update_plot(current_frame[0])

    def on_reset(event):
        ani.event_source.stop()
        playing[0] = False
        current_frame[0] = 0
        update_plot(0)
        b_play.label.set_text('Play')

    b_play.on_clicked(on_play)
    b_back.on_clicked(on_back)
    b_forward.on_clicked(on_forward)
    b_log.on_clicked(on_log)
    b_reset.on_clicked(on_reset)

    fig.subplots_adjust(bottom=0.12, right=0.94)  # Space for buttons and side box
    plt.show()