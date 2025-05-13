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
9.37928, 9.2527, 9.20737, 9.1725, 9.1429, 9.11672, 9.09315, 9.07171, 9.05208, 9.03403, 9.01736, 9.00192, 8.98757, 8.97421, 8.96174, 8.95007, 8.93912, 8.92883, 8.91914, 8.91001, 8.90137, 8.8932, 8.88544, 8.87808, 8.87107, 8.8644, 8.85803, 8.85194, 8.84612, 8.84054, 8.83519, 8.83005, 8.82511, 8.82035, 8.81577, 8.81135, 8.80708, 8.80295, 8.79895, 8.79509, 8.79134, 8.78771, 8.78418, 8.78075, 8.77742, 8.77418, 8.77102, 8.76794, 8.76495, 8.76202, 8.75917, 8.75638, 8.75365, 8.75099, 8.74838, 8.74584, 8.74334, 8.74089, 8.7385, 8.73615, 8.73385, 8.73159, 8.72937, 8.7272, 8.72506, 8.72296, 8.7209, 8.71887, 8.71688, 8.71491, 8.71299, 8.71109, 8.70922, 8.70738, 8.70557, 8.70378, 8.70203, 8.7003, 8.69859, 8.69691, 8.69525, 8.69362, 8.69201, 8.69042, 8.68885, 8.6873, 8.68577, 8.68427, 8.68278, 8.68131, 8.67986, 8.67843, 8.67701, 8.67562, 8.67424, 8.67287, 8.67153, 8.6702, 8.66888, 8.66758, 8.6663,

]
list_async = [
9.37928, 9.29539, 9.22184, 9.18349, 9.15209, 9.12476, 9.10034, 9.07822, 9.05803, 9.0395, 9.02242, 9.00661, 8.99195, 8.9783, 8.96557, 8.95367, 8.94251, 8.93204, 8.92218, 8.91289, 8.90411, 8.8958, 8.88793, 8.88046, 8.87335, 8.86658, 8.86012, 8.85396, 8.84806, 8.84241, 8.83699, 8.83179, 8.82679, 8.82198, 8.81734, 8.81287, 8.80856, 8.80439, 8.80035, 8.79645, 8.79266, 8.78899, 8.78543, 8.78197, 8.77861, 8.77534, 8.77216, 8.76905, 8.76603, 8.76308, 8.7602, 8.75739, 8.75465, 8.75196, 8.74934, 8.74677, 8.74426, 8.74179, 8.73938, 8.73702, 8.7347, 8.73243, 8.73019, 8.728, 8.72585, 8.72374, 8.72166, 8.71962, 8.71762, 8.71565, 8.71371, 8.7118, 8.70992, 8.70807, 8.70625, 8.70445, 8.70269, 8.70095, 8.69923, 8.69754, 8.69587, 8.69423, 8.69261, 8.69101, 8.68944, 8.68788, 8.68635, 8.68483, 8.68334, 8.68186, 8.6804, 8.67897, 8.67755, 8.67614, 8.67476, 8.67339, 8.67203, 8.6707, 8.66938, 8.66807, 8.66678, 8.66677, 

]
proc_energy_list(list_async, "Async")
proc_energy_list(list_sync, "Sync")





draw_all(fig, "Iterations", "Iteration", "Relative Energy", "example2_iter_100_convergence")
