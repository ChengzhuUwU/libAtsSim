import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font',family='Times New Roman')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
          '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']

global color_idx
color_idx = 0
def get_color():
    global color_idx
    color_idx += 1
    return colors[(color_idx - 1)%20]

def proc_redual_list(list_residual, name):
    list_residual = np.array(list_residual)
    # print(f"{name:<8s}[40] = {list_residual[40] : .4f} ||| {name:<8s}[80] = {list_residual[80] : .4f}")
    print(f'{name:10} : After {len(list_residual) - 1 : 5} Iterations, Final Residual = {list_residual[-1]}')
    plt.plot(list_residual, label=name, color=get_color())

def draw_all(fig, name, xlable, ylable, figure_name = ""):

    def call_back(event):
        axtemp=event.inaxes
        x_min, x_max = axtemp.get_xlim()
        fanwei = (x_max - x_min) / 10
        if event.button == 'up':
            axtemp.set(xlim=(x_min + fanwei, x_max - fanwei))
        elif event.button == 'down':
            axtemp.set(xlim=(x_min - fanwei, x_max + fanwei))
        fig.canvas.draw_idle() 

    fig.canvas.mpl_connect('scroll_event', call_back)
    fig.canvas.mpl_connect('button_press_event', call_back)


    plt.title(name)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.legend()

    if figure_name != "":
        plt.savefig(f"documents/{figure_name}.png", format="png", dpi=300, bbox_inches="tight")

    plt.grid(True)
    plt.show()

    

    

# energy_optimal = np.min(energy_weighted_delta_with_main_device)
# energy_max = np.max(energy_weighted_delta_with_main_device)
# energy_init = energy_gs[0]
fig = plt.figure(figsize=(10, 6))

def proc_energy_list(list_energy, name):
    global energy_optimal
    global energy_init
    list_energy = np.array(list_energy)
    # list_energy[list_energy > energy_max] = energy_max
    # list_energy = (list_energy - energy_optimal) / (energy_init - energy_optimal)
    # print(f"{name:<8s}[40] = {list_energy[40] : .4f} ||| {name:<8s}[80] = {list_energy[80] : .4f}")
    plt.plot(list_energy, label=name, color=get_color())


global_init = 216.659
global_optimal = 99.9921
def normalize(data, global_min, global_max):
    return [(x - global_optimal) / (global_init - global_optimal) for x in data]



list_sync = [
442.325, 442.036, 441.87, 441.731, 441.603, 441.482, 441.366, 441.254, 441.146, 441.04, 440.937, 440.836, 440.737, 440.64, 440.545, 440.452, 440.36, 440.269, 440.18, 440.092, 440.005, 439.919, 439.835, 439.751, 439.668, 439.587, 439.506, 439.426, 439.347, 439.268, 439.191, 439.114, 439.038, 438.962, 438.887, 438.813, 438.74, 438.667, 438.594, 438.523, 438.451, 438.381, 438.311, 438.241, 438.172, 438.103, 438.035, 437.968, 437.9, 437.834, 437.767, 437.702, 437.636, 437.571, 437.507, 437.443, 437.379, 437.316, 437.253, 437.191, 437.128, 437.067, 437.005, 436.944, 436.884, 436.824, 436.764, 436.704, 436.645, 436.586, 436.528, 436.47, 436.412, 436.354, 436.297, 436.24, 436.184, 436.127, 436.072, 436.016, 435.961, 435.906, 435.851, 435.797, 435.742, 435.689, 435.635, 435.582, 435.529, 435.476, 435.424, 435.372, 435.32, 435.268, 435.217, 435.166, 435.115, 435.064, 435.014, 434.964, 434.914,

]
list_async = [
442.325, 442.14, 441.925, 441.781, 441.65, 441.527, 441.41, 441.297, 441.187, 441.08, 440.976, 440.875, 440.775, 440.678, 440.582, 440.488, 440.396, 440.305, 440.215, 440.127, 440.04, 439.953, 439.869, 439.785, 439.702, 439.62, 439.539, 439.458, 439.379, 439.3, 439.222, 439.145, 439.069, 438.993, 438.918, 438.844, 438.77, 438.697, 438.624, 438.552, 438.481, 438.41, 438.339, 438.27, 438.2, 438.132, 438.063, 437.996, 437.928, 437.861, 437.795, 437.729, 437.663, 437.598, 437.534, 437.469, 437.406, 437.342, 437.279, 437.217, 437.154, 437.092, 437.031, 436.97, 436.909, 436.849, 436.789, 436.729, 436.67, 436.611, 436.552, 436.494, 436.436, 436.378, 436.321, 436.264, 436.207, 436.151, 436.095, 436.039, 435.984, 435.929, 435.874, 435.819, 435.765, 435.711, 435.658, 435.604, 435.551, 435.498, 435.446, 435.394, 435.342, 435.29, 435.238, 435.187, 435.136, 435.086, 435.035, 434.985, 434.935, 434.935,

]
proc_energy_list(list_async, "Async")
proc_energy_list(list_sync, "Sync")





draw_all(fig, "Iterations", "Iteration", "Relative Energy", "example2_iter_100_convergence")
