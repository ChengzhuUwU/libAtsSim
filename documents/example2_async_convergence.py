import sys
import matplotlib.pyplot as plt
import numpy as np

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
6.21544, 6.10051, 6.0613, 6.03175, 6.00706, 5.98552, 5.96635, 5.94909, 5.93342, 5.91912, 5.90601, 5.89392, 5.88275, 5.87238, 5.86274, 5.85373, 5.84531, 5.83742, 5.82999, 5.823, 5.8164, 5.81016, 5.80425, 5.79865, 5.79332, 5.78824, 5.78341, 5.77879, 5.77438, 5.77016, 5.76611, 5.76222, 5.75849, 5.75491, 5.75145, 5.74812, 5.74491, 5.74181, 5.73881, 5.73591, 5.7331, 5.73038, 5.72774, 5.72518, 5.72269, 5.72027, 5.71792, 5.71563, 5.7134, 5.71123, 5.70911, 5.70704, 5.70502, 5.70305, 5.70113, 5.69924, 5.6974, 5.6956, 5.69384, 5.69211, 5.69042, 5.68877, 5.68714, 5.68555, 5.68399, 5.68245, 5.68095, 5.67947, 5.67802, 5.67659, 5.67519, 5.67381, 5.67246, 5.67113, 5.66982, 5.66853, 5.66726, 5.66602, 5.66479, 5.66358, 5.66239, 5.66121, 5.66006, 5.65892, 5.6578, 5.65669, 5.6556, 5.65453, 5.65347, 5.65242, 5.65139, 5.65037, 5.64937, 5.64838, 5.64741, 5.64644, 5.64549, 5.64455, 5.64363, 5.64271, 5.64181,

]
list_async = [
6.21899, 6.14208, 6.07743, 6.04463, 6.01826, 5.99564, 5.97568, 5.95779, 5.94161, 5.92688, 5.91339, 5.90098, 5.88953, 5.87891, 5.86905, 5.85985, 5.85125, 5.84319, 5.83563, 5.8285, 5.82179, 5.81544, 5.80943, 5.80372, 5.79831, 5.79316, 5.78825, 5.78356, 5.77908, 5.7748, 5.77069, 5.76675, 5.76297, 5.75934, 5.75583, 5.75246, 5.74921, 5.74607, 5.74303, 5.7401, 5.73725, 5.7345, 5.73183, 5.72924, 5.72672, 5.72428, 5.7219, 5.71958, 5.71733, 5.71513, 5.71299, 5.7109, 5.70887, 5.70688, 5.70493, 5.70303, 5.70117, 5.69935, 5.69757, 5.69583, 5.69412, 5.69245, 5.69081, 5.6892, 5.68763, 5.68608, 5.68456, 5.68307, 5.68161, 5.68017, 5.67875, 5.67736, 5.676, 5.67466, 5.67334, 5.67204, 5.67076, 5.6695, 5.66826, 5.66704, 5.66584, 5.66466, 5.66349, 5.66235, 5.66121, 5.6601, 5.659, 5.65792, 5.65685, 5.6558, 5.65476, 5.65373, 5.65272, 5.65173, 5.65074, 5.64977, 5.64881, 5.64787, 5.64694, 5.64601, 5.6451, 5.64509,

]
proc_energy_list(list_async, "Async")
proc_energy_list(list_sync, "Sync")





draw_all(fig, "Iterations", "Iteration", "Relative Energy", "iter_100_convergence")
