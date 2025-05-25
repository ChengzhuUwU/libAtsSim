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
442.325, 442.036, 441.87, 441.731, 441.603, 441.482, 441.366, 441.254, 441.146, 441.04, 440.937, 440.836, 440.737, 440.641, 440.545, 440.452, 440.36, 440.269, 440.18, 440.092, 440.005, 439.92, 439.835, 439.751, 439.669, 439.587, 439.506, 439.426, 439.347, 439.269, 439.191, 439.114, 439.038, 438.963, 438.888, 438.814, 438.74, 438.667, 438.595, 438.523, 438.452, 438.381, 438.311, 438.241, 438.172, 438.104, 438.035, 437.968, 437.901, 437.834, 437.768, 437.702, 437.637, 437.572, 437.507, 437.443, 437.379, 437.316, 437.253, 437.191, 437.129, 437.067, 437.006, 436.945, 436.884, 436.824, 436.764, 436.705, 436.645, 436.587, 436.528, 436.47, 436.412, 436.355, 436.297, 436.241, 436.184, 436.128, 436.072, 436.016, 435.961, 435.906, 435.851, 435.797, 435.743, 435.689, 435.635, 435.582, 435.529, 435.477, 435.424, 435.372, 435.32, 435.269, 435.217, 435.166, 435.115, 435.065, 435.015, 434.965, 434.915,

]
list_async = [
442.325, 442.141, 441.926, 441.782, 441.651, 441.528, 441.41, 441.297, 441.187, 441.081, 440.977, 440.875, 440.776, 440.678, 440.583, 440.489, 440.396, 440.305, 440.215, 440.127, 440.04, 439.954, 439.869, 439.785, 439.702, 439.62, 439.539, 439.459, 439.379, 439.301, 439.223, 439.146, 439.069, 438.993, 438.918, 438.844, 438.77, 438.697, 438.624, 438.552, 438.481, 438.41, 438.34, 438.27, 438.201, 438.132, 438.064, 437.996, 437.929, 437.862, 437.795, 437.729, 437.664, 437.599, 437.534, 437.47, 437.406, 437.343, 437.28, 437.217, 437.155, 437.093, 437.031, 436.97, 436.909, 436.849, 436.789, 436.729, 436.67, 436.611, 436.553, 436.494, 436.436, 436.379, 436.321, 436.264, 436.208, 436.151, 436.095, 436.04, 435.984, 435.929, 435.874, 435.82, 435.765, 435.712, 435.658, 435.604, 435.551, 435.499, 435.446, 435.394, 435.342, 435.29, 435.239, 435.188, 435.137, 435.086, 435.036, 434.986, 434.936, 434.935,

]
proc_energy_list(list_async, "Async")
proc_energy_list(list_sync, "Sync")





draw_all(fig, "Iterations", "Iteration", "Relative Energy", "example2_iter_100_convergence")
