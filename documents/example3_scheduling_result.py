import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.font_manager import FontProperties

plt.rc('font',family='Times New Roman')

# plt.rcParams['xtick.labelsize'] = 16  # 设置字体大小
# plt.rcParams['xtick.labelweight'] = 'bold'  # 设置字体加粗

# Define colors for each job
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
          '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
          '#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#33fff1',
          '#ffae33', '#a833ff', '#33ffa8', '#ff3333',
          '#a8ff33', '#e833ff', '#33ffe8', '#ff8333', '#3383ff',
          '#ff33f1', '#a6ff33', '#ff33a8', '#33ff83', '#f1ff33']
# Function to visualize the scheduling results
def visualize_schedule(schedule):
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Processor')
    ax.set_yticks(range(2))
    ax.set_yticklabels([f'{["CPU", "GPU"][i]}' for i in range(2)])
    ax.grid(True)

    schedule_data = schedule[0: 2]

    legend_patches = [mpatches.Patch(color=colors[i % len(colors)], label=str(i)) for i in range(max(job for jobs in schedule_data for job, _, _, _, _ in jobs) + 1)]
    plt.legend(handles=legend_patches, loc='upper right', title='Job')
    for processor_id, processor_schedule in enumerate(schedule_data):
        for job, tid, buffer_idx, start_time, end_time in processor_schedule:
            duration = end_time - start_time
        
            bar = ax.broken_barh([(start_time, end_time - start_time + 0.015)], (processor_id - 0.4, 0.8), facecolors=colors[job % len(colors)])
            # ax.text(start_time + (end_time - start_time) / 2, processor_id, str(f'{tid}\n({job})\n({buffer_idx if buffer_idx != 4294967295 else "/"})'), ha='center', va='center', color='white' if duration > 1e-8 else 'red')
            ax.text(start_time + (end_time - start_time) / 2, processor_id, str(f'{job}\n({buffer_idx if buffer_idx != 4294967295 else "/"})'), ha='center', va='center', color='white' if duration > 1e-8 else 'red')
    
    
    if len(schedule) > 2:
        connection_data = schedule[2]
        for left_proc, left_tid, right_proc, right_tid, send_time, recv_time in connection_data:
            start_x = send_time
            end_x = recv_time
            start_y = left_proc + 0.42 if left_proc == 0 else left_proc - 0.42
            end_y = right_proc + 0.42 if right_proc == 0 else right_proc - 0.42
            
            ax.annotate(
                '',  # No text
                xy=(end_x, end_y),  # Arrow head position
                xytext=(start_x, start_y),  # Arrow tail position
                arrowprops=dict(
                    arrowstyle='->', 
                    color='red', 
                    lw=2
                )
            )
        connection_data = schedule[3]
        for left_proc, left_tid, right_proc, right_tid, send_time, recv_time in connection_data:
            start_x = send_time
            end_x = recv_time
            start_y = left_proc + 0.42 if left_proc == 0 else left_proc - 0.42
            end_y = right_proc + 0.42 if right_proc == 0 else right_proc - 0.42
            ax.annotate(
                '',  # No text
                xy=(end_x, end_y),  # Arrow head position
                xytext=(start_x, start_y),  # Arrow tail position
                arrowprops=dict(
                    arrowstyle='->', 
                    color='blue', 
                    lw=2
                )
            )
    plt.tight_layout()
    
# article_colors = [
#     '#e7e6e6', '#e56b6f', '#006400', '#540d6e', '#DDDE76',  # 4
#     '#DB6387', '#BFAD86', '#F5BE33', '#014f86', '#ff7900',  # 9
#     '#004e98', '#0582ca', # 11
#     '#b388eb', '#3bceac', 
#     '#87bba2', '#b5e2fa',  # Stretch # 15
#     '#fce5d5', '#e1afa9',  # 17
#     '#fe938c', '#bdb2ff', '#ffcad4', '#0a9396', 
#     '#fec89a', '#d88c9a', # Collison
#     '#5e548e', '#3a5a40',  # 25
# ]
article_colors = [
    # '#e7e6e6', '#e56b6f', '#006400', '#540d6e', '#DDDE76',  # 4
    # '#DB6387', '#BFAD86', '#F5BE33', '#014f86', '#ff7900',  # 9
    # '#004e98', '#0582ca', # 11
    # '#b388eb', '#3bceac', 
    # '#87bba2', '#b5e2fa',  # Stretch # 15
    '#e7e6e6', '#e8e5e6', '#fff2ca', '#fff2ca', '#ffe0e1',  # 4
    '#ffe0e1', '#f3e1ff', '#f3e1ff', '#e8e5e6', '#e8e5e6',  # 9
    '#defffe', '#defffe', # 11
    '#f0ffd7', '#e7e5e7', 
    '#dff0d6', '#ddebf8',  # Stretch # 15
    '#fce5d5', '#e1afa9',  # 17
    '#fe938c', '#bdb2ff', '#ffcad4', '#0a9396', 
    '#fec89a', '#d88c9a', # Collison
    '#5e548e', '#3a5a40',  # 25
]
vbd_task_name_map = {
    0:"Misc.",
    14:"Block 0",
    15:"Block 1",
    16:"Block 2",
    17:"Block 3",
    18:"Block 4",
    19:"Block 5",
    20:"Block 6",
    21:"Block 7",
    22:"Block 8",
    23:"Block 9",
    24:"Block 10",
    25:"Block 11",
    26:"Block 12",
    27:"Block 13",
}

def hex_to_rgb(hex_color):
    # Remove '#' symbol if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    fr = r / 256.0
    fg = g / 256.0
    fb = b / 256.0
    print(f"[{fr:.4f}, {fg:.4f}, {fb:.4f}], ")
    return np.array([fr, fg, fb])

def map_orig_idx_to_arcle(func_id):
    print()

def visualize_schedule3(schedule, name_map, with_text = False, figure_name = ""):

    device_count = 2
    block_to_space_scale = 0.65

    schedule_data = schedule[0: device_count]
    # device_labels = ["CPU", "GPU"]
    device_labels = ["", ""]
    device_spacing = 0.4 / device_count

    end_time = max(end_time for processor_schedule in schedule_data for _, _, _, _, end_time in processor_schedule) + 0.1
    
    bold_font = FontProperties(weight='bold', size=14)
    # fig, ax = plt.subplots(figsize=(18, 3.5))
    fig, ax = plt.subplots(figsize=(18, 3.5))
    # ax.set_xlabel('Time (ms)')
    ax.set_xlim(0, max(end_time for processor_schedule in schedule_data for _, _, _, _, end_time in processor_schedule) + 0.1)
    # ax.set_title("Heterogeneous Task Scheduling Visualization")
    ax.tick_params(axis='x')  
    for label in ax.get_xticklabels():
        label.set_fontproperties(bold_font)

    ax.spines['top'].set_visible(False)   
    ax.spines['right'].set_visible(False) 
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)  

    # ax.set_ylabel('Processor')
    # ax.set_yticks(range(2))
    # ax.set_yticklabels([f'{["CPU", "GPU"][i]}' for i in range(2)])
    yticks = [i * device_spacing for i in range(len(device_labels))]  # 根据缩放后的值计算刻度位置
    ax.set_yticks(yticks)  # 设置 y 轴刻度
    ax.set_yticklabels(device_labels)  # 设置 y 轴标签
    ax.grid(True, linestyle='--', alpha=0.9)

    

    # Create a legend for job numbers
    id_set = set()
    for processor_id, processor_schedule in enumerate(schedule_data):
        for job, tid, buffer_idx, start_time, end_time in processor_schedule:
            id_set.add(job)
    # legend_patches = [mpatches.Patch(color=colors[i % len(colors)], label=str(i)) for i in range(max(job for jobs in schedule_data for job, _, _, _, _ in jobs) + 1)]
    legend_patches = [
        mpatches.Patch(color=article_colors[i % len(article_colors)], label=name_map[i] if (i >= 14) or i == 0 else f"{i}")
            for i in id_set
    ]
    bold_font = FontProperties(weight='bold', size=16)
    plt.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),  # 放置在图形下方
        ncol=math.ceil(len(id_set)/2),
        frameon=False,
        columnspacing = 0.5,
        prop={
            'size':16,
            'weight':'bold',
        }
    )
    
    for processor_id, processor_schedule in enumerate(schedule_data):
        y_base = processor_id * device_spacing 
        for job, tid, buffer_idx, start_time, end_time in processor_schedule:
            duration = end_time - start_time
            color = article_colors[job % len(article_colors)]
            # smoother = -0.01 if processor_id == 0 else 0
            smoother = 0.0 if processor_id == 0 else 0
            ax.broken_barh(
                # [(start_time, end_time - start_time + 0.04)], 
                [(start_time, end_time - start_time + smoother)], 
                (y_base, device_spacing * block_to_space_scale),  # 高度缩放到设备区间
                facecolors=color,
                # edgecolor='black'
            )
            # bar = ax.broken_barh([(start_time, end_time - start_time + 0.04)], (processor_id / 3.6, 0.2), facecolors=color)
            if with_text:
                ax.text(
                    start_time + (end_time - start_time) / 2, 
                    y_base + (device_spacing * block_to_space_scale) / 2, 
                    str(f'{f"Block {job - 14}" if job >= 14 else "/"}\n({f"Buffer {buffer_idx}" if buffer_idx != 4294967295 else "/"})'), 
                    # str(f'{tid}\n({job})\n({buffer_idx if buffer_idx != 4294967295 else "/"})'), 
                    ha='center', va='center', color='black' if duration > 1e-8 else 'red')
    
    if len(schedule) > 2:
        connection_data = schedule[2]
        for left_proc, left_tid, right_proc, right_tid, send_time, recv_time in connection_data:
            # array_offset = 0.02
            array_offset = 0.0
            start_x = send_time - array_offset
            end_x = recv_time + array_offset

            # y_base = processor_id * device_spacing 
            delta = device_spacing * block_to_space_scale
            start_y = delta - 0.00 if left_proc == 0  else left_proc * device_spacing  + 0.00
            end_y   = delta - 0.00 if right_proc == 0 else right_proc * device_spacing + 0.00
            ax.annotate(
                '',  # No text
                xy=(end_x, end_y),  # Arrow head position
                xytext=(start_x, start_y),  # Arrow tail position
                arrowprops=dict(
                    arrowstyle='->', 
                    color='crimson' if left_proc == 0 else 'green', 
                    connectionstyle='arc3',
                    # shrink=0.05
                    lw=1.5
                )
            )

    if len(schedule) > 3:
        connection_data = schedule[3]
        for left_proc, left_tid, right_proc, right_tid, send_time, recv_time in connection_data:
            start_x = send_time - 0.01
            # end_x = send_time + 0.03
            end_x = recv_time + 0.01
            delta = device_spacing * 0.65
            start_y = delta if left_proc == 0 else  left_proc * device_spacing  
            end_y   = delta if right_proc == 0 else right_proc * device_spacing 

            ax.annotate(
                '',  # No text
                xy=(end_x, end_y),  # Arrow head position
                xytext=(start_x, start_y),  # Arrow tail position
                arrowprops=dict(
                    arrowstyle='->', 
                    color='blue', 
                    lw=2
                )
            )

    plt.tight_layout()
    
    if figure_name != "":
        # plt.savefig(f"documents/{figure_name}.svg", format="svg", dpi=300, bbox_inches="tight") // Save as .svg
        plt.savefig(f"documents/{figure_name}.png", format="png", dpi=300, bbox_inches="tight")


#
# Paste the scheduling result from 
#

# visualize_schedule3([ # Iter 2
#     [(19, 6, 1, 0.555, 1.555), (15, 12, 0, 1.557, 2.557), (20, 17, 1, 2.559, 3.559)],
#     [(0, 23, 4294967295, 0.000, 0.200), (1, 0, 4294967295, 0.210, 0.410), (14, 1, 0, 0.420, 0.620), (15, 2, 0, 0.630, 0.830), (16, 3, 0, 0.840, 1.040), (17, 4, 0, 1.050, 1.250), (18, 5, 2, 1.260, 1.460), (20, 7, 2, 1.470, 1.670), (21, 8, 2, 1.680, 1.880), (22, 9, 1, 1.890, 2.090), (23, 10, 1, 2.100, 2.300), (14, 11, 3, 2.310, 2.510), (16, 13, 3, 2.520, 2.720), (17, 14, 3, 2.730, 2.930), (18, 15, 0, 2.940, 3.140), (19, 16, 0, 3.150, 3.350), (21, 18, 0, 3.360, 3.560), (22, 19, 0, 3.570, 3.770), (23, 20, 1, 3.780, 3.980), (0, 21, 1, 3.990, 4.190), (0, 22, 4294967295, 4.200, 4.400), (0, 24, 4294967295, 4.410, 4.610)],
#     [(0, 6, 1, 9, 1.555, 1.890), (1, 4, 0, 12, 1.250, 1.557), (0, 12, 1, 15, 2.557, 2.940), (1, 10, 0, 17, 2.300, 2.559), (0, 17, 1, 20, 3.559, 3.780), ],
#     [(1, 0, 1, 1, 0.410, 0.420), (1, 4, 1, 5, 1.250, 1.260), (1, 0, 0, 6, 0.410, 0.555), (1, 8, 1, 9, 1.880, 1.890), (1, 10, 1, 11, 2.300, 2.310), (1, 14, 1, 15, 2.930, 2.940), (1, 19, 1, 20, 3.770, 3.780), ],
    
# ], vbd_task_name_map, with_text=True, figure_name="")


visualize_schedule3([ # Iter 12
    [(0, 123, 4294967295, 0.000, 0.000), (18, 5, 1, 0.153, 0.323), (21, 8, 0, 0.325, 0.433), (23, 10, 2, 0.435, 0.463), (15, 12, 4, 0.465, 0.633), (18, 15, 3, 0.635, 0.814), (21, 18, 5, 0.816, 0.999), (15, 22, 0, 1.001, 1.162), (19, 26, 4, 1.164, 1.298), (20, 27, 3, 1.300, 1.463), (22, 29, 7, 1.465, 1.527), (16, 33, 8, 1.529, 1.729), (19, 36, 0, 1.731, 1.897), (21, 38, 4, 1.899, 2.029), (23, 40, 3, 2.031, 2.064), (17, 44, 7, 2.066, 2.269), (19, 46, 8, 2.271, 2.452), (22, 49, 0, 2.454, 2.520), (23, 50, 4, 2.522, 2.549), (16, 53, 11, 2.551, 2.687), (18, 55, 3, 2.689, 2.881), (21, 58, 7, 2.883, 2.999), (23, 60, 8, 3.001, 3.032), (16, 63, 4, 3.034, 3.208), (19, 66, 11, 3.210, 3.409), (22, 69, 3, 3.411, 3.508), (23, 70, 7, 3.510, 3.525), (16, 73, 15, 3.527, 3.689), (19, 76, 4, 3.691, 3.852), (21, 78, 14, 3.854, 3.983), (23, 80, 11, 3.985, 4.002), (16, 83, 16, 4.004, 4.222), (19, 86, 15, 4.224, 4.380), (21, 88, 4, 4.382, 4.532), (23, 90, 11, 4.534, 4.563), (16, 93, 20, 4.565, 4.749), (19, 96, 16, 4.751, 4.894), (22, 99, 15, 4.896, 4.977), (23, 100, 15, 4.979, 5.007), (16, 103, 4, 5.009, 5.151), (19, 106, 11, 5.153, 5.314), (21, 108, 20, 5.316, 5.462), (23, 110, 16, 5.464, 5.486), (16, 113, 15, 5.488, 5.653), (20, 117, 4, 5.655, 5.792), (22, 119, 22, 5.794, 5.877), (23, 120, 11, 5.879, 5.923), (0, 121, 16, 6.033, 6.061), (0, 122, 4294967295, 6.063, 6.099)],
    [(1, 0, 4294967295, 0.000, 0.008), (14, 1, 0, 0.018, 0.106), (15, 2, 0, 0.116, 0.167), (16, 3, 2, 0.177, 0.255), (17, 4, 3, 0.265, 0.339), (19, 6, 3, 0.349, 0.393), (20, 7, 3, 0.403, 0.457), (22, 9, 5, 0.467, 0.512), (14, 11, 5, 0.522, 0.580), (16, 13, 1, 0.590, 0.672), (17, 14, 0, 0.682, 0.774), (19, 16, 2, 0.784, 0.862), (20, 17, 4, 0.872, 0.937), (22, 19, 4, 0.947, 0.995), (23, 20, 6, 1.005, 1.031), (14, 21, 3, 1.041, 1.111), (16, 23, 7, 1.121, 1.202), (17, 24, 7, 1.212, 1.287), (18, 25, 5, 1.297, 1.387), (21, 28, 0, 1.397, 1.480), (23, 30, 0, 1.490, 1.535), (14, 31, 4, 1.545, 1.616), (15, 32, 4, 1.626, 1.706), (17, 34, 3, 1.716, 1.797), (18, 35, 7, 1.807, 1.887), (20, 37, 9, 1.897, 1.956), (22, 39, 8, 1.966, 2.025), (14, 41, 8, 2.035, 2.087), (15, 42, 10, 2.097, 2.169), (16, 43, 0, 2.179, 2.259), (18, 45, 4, 2.269, 2.358), (20, 47, 3, 2.368, 2.447), (21, 48, 3, 2.457, 2.538), (14, 51, 7, 2.548, 2.617), (15, 52, 7, 2.627, 2.706), (17, 54, 8, 2.716, 2.792), (19, 56, 4, 2.802, 2.858), (20, 57, 12, 2.868, 2.944), (22, 59, 11, 2.954, 3.007), (14, 61, 13, 3.017, 3.077), (15, 62, 13, 3.087, 3.149), (17, 64, 3, 3.159, 3.239), (18, 65, 7, 3.249, 3.340), (20, 67, 8, 3.350, 3.428), (21, 68, 4, 3.438, 3.509), (14, 71, 14, 3.519, 3.591), (15, 72, 14, 3.601, 3.680), (17, 74, 11, 3.690, 3.767), (18, 75, 7, 3.777, 3.866), (20, 77, 7, 3.876, 3.954), (22, 79, 15, 3.964, 4.023), (14, 81, 17, 4.033, 4.103), (15, 82, 4, 4.113, 4.184), (17, 84, 18, 4.194, 4.268), (18, 85, 11, 4.278, 4.368), (20, 87, 19, 4.378, 4.455), (22, 89, 16, 4.465, 4.523), (14, 91, 16, 4.533, 4.595), (15, 92, 15, 4.605, 4.669), (17, 94, 15, 4.679, 4.743), (18, 95, 4, 4.753, 4.835), (20, 97, 11, 4.845, 4.899), (21, 98, 11, 4.909, 4.989), (14, 101, 20, 4.999, 5.069), (15, 102, 20, 5.079, 5.159), (17, 104, 16, 5.169, 5.245), (18, 105, 15, 5.255, 5.330), (20, 107, 21, 5.340, 5.395), (22, 109, 4, 5.405, 5.438), (14, 111, 4, 5.448, 5.503), (15, 112, 22, 5.513, 5.578), (17, 114, 11, 5.588, 5.670), (18, 115, 23, 5.680, 5.747), (19, 116, 16, 5.757, 5.809), (21, 118, 16, 5.819, 5.888), (0, 124, 4294967295, 6.099, 6.102)],
    [(1, 2, 0, 8, 0.167, 0.325), (1, 3, 0, 10, 0.255, 0.435), (0, 5, 1, 13, 0.323, 0.590), (0, 8, 1, 14, 0.433, 0.682), (1, 7, 0, 15, 0.457, 0.635), (0, 10, 1, 16, 0.463, 0.784), (0, 12, 1, 17, 0.633, 0.872), (1, 11, 0, 18, 0.580, 0.816), (0, 15, 1, 21, 0.814, 1.041), (1, 14, 0, 22, 0.774, 1.001), (0, 18, 1, 25, 0.999, 1.297), (1, 19, 0, 26, 0.995, 1.164), (1, 21, 0, 27, 1.111, 1.300), (0, 22, 1, 28, 1.162, 1.397), (1, 24, 0, 29, 1.287, 1.465), (0, 26, 1, 31, 1.298, 1.545), (0, 27, 1, 34, 1.463, 1.716), (0, 29, 1, 35, 1.527, 1.807), (1, 30, 0, 36, 1.535, 1.731), (1, 32, 0, 38, 1.706, 1.899), (0, 33, 1, 39, 1.729, 1.966), (1, 34, 0, 40, 1.797, 2.031), (0, 36, 1, 43, 1.897, 2.179), (1, 35, 0, 44, 1.887, 2.066), (0, 38, 1, 45, 2.029, 2.269), (1, 41, 0, 46, 2.087, 2.271), (0, 40, 1, 47, 2.064, 2.368), (1, 43, 0, 49, 2.259, 2.454), (1, 45, 0, 50, 2.358, 2.522), (0, 44, 1, 51, 2.269, 2.548), (0, 46, 1, 54, 2.452, 2.716), (1, 48, 0, 55, 2.538, 2.689), (0, 49, 1, 56, 2.520, 2.802), (0, 50, 1, 56, 2.549, 2.802), (1, 52, 0, 58, 2.706, 2.883), (0, 53, 1, 59, 2.687, 2.954), (1, 54, 0, 60, 2.792, 3.001), (1, 56, 0, 63, 2.858, 3.034), (0, 55, 1, 64, 2.881, 3.159), (0, 58, 1, 65, 2.999, 3.249), (1, 59, 0, 66, 3.007, 3.210), (0, 60, 1, 67, 3.032, 3.350), (0, 63, 1, 68, 3.208, 3.438), (1, 64, 0, 69, 3.239, 3.411), (1, 65, 0, 70, 3.340, 3.510), (0, 66, 1, 74, 3.409, 3.690), (0, 69, 1, 75, 3.508, 3.777), (0, 70, 1, 75, 3.525, 3.777), (1, 68, 0, 76, 3.509, 3.691), (1, 72, 0, 78, 3.680, 3.854), (0, 73, 1, 79, 3.689, 3.964), (1, 74, 0, 80, 3.767, 3.985), (0, 76, 1, 82, 3.852, 4.113), (0, 78, 1, 85, 3.983, 4.278), (0, 80, 1, 85, 4.002, 4.278), (1, 79, 0, 86, 4.023, 4.224), (1, 82, 0, 88, 4.184, 4.382), (0, 83, 1, 89, 4.222, 4.465), (1, 85, 0, 90, 4.368, 4.534), (0, 86, 1, 92, 4.380, 4.605), (0, 88, 1, 95, 4.532, 4.753), (1, 91, 0, 96, 4.595, 4.751), (0, 90, 1, 97, 4.563, 4.845), (1, 94, 0, 99, 4.743, 4.896), (0, 93, 1, 101, 4.749, 4.999), (1, 95, 0, 103, 4.835, 5.009), (0, 96, 1, 104, 4.894, 5.169), (0, 100, 1, 105, 5.007, 5.255), (1, 98, 0, 106, 4.989, 5.153), (1, 102, 0, 108, 5.159, 5.316), (0, 103, 1, 109, 5.151, 5.405), (1, 104, 0, 110, 5.245, 5.464), (1, 105, 0, 113, 5.330, 5.488), (0, 106, 1, 114, 5.314, 5.588), (0, 108, 1, 116, 5.462, 5.757), (0, 110, 1, 116, 5.486, 5.757), (1, 111, 0, 117, 5.503, 5.655), (1, 112, 0, 119, 5.578, 5.794), (1, 114, 0, 120, 5.670, 5.879), (1, 118, 0, 121, 5.888, 6.033), ],
    [(1, 0, 1, 1, 0.008, 0.018), (1, 2, 1, 3, 0.167, 0.177), (1, 3, 1, 4, 0.255, 0.265), (1, 0, 0, 5, 0.008, 0.153), (1, 7, 1, 9, 0.457, 0.467), (0, 10, 0, 12, 0.463, 0.465), (1, 11, 1, 13, 0.580, 0.590), (1, 13, 1, 14, 0.672, 0.682), (1, 14, 1, 16, 0.774, 0.784), (1, 16, 1, 17, 0.862, 0.872), (1, 19, 1, 20, 0.995, 1.005), (1, 20, 1, 21, 1.031, 1.041), (1, 21, 1, 23, 1.111, 1.121), (1, 24, 1, 25, 1.287, 1.297), (1, 25, 1, 28, 1.387, 1.397), (1, 30, 1, 31, 1.535, 1.545), (0, 29, 0, 33, 1.527, 1.529), (1, 32, 1, 34, 1.706, 1.716), (1, 34, 1, 35, 1.797, 1.807), (1, 35, 1, 37, 1.887, 1.897), (1, 37, 1, 39, 1.956, 1.966), (1, 41, 1, 42, 2.087, 2.097), (1, 42, 1, 43, 2.169, 2.179), (1, 43, 1, 45, 2.259, 2.269), (1, 45, 1, 47, 2.358, 2.368), (1, 48, 1, 51, 2.538, 2.548), (0, 50, 0, 53, 2.549, 2.551), (1, 52, 1, 54, 2.706, 2.716), (1, 54, 1, 56, 2.792, 2.802), (1, 56, 1, 57, 2.858, 2.868), (1, 57, 1, 59, 2.944, 2.954), (1, 59, 1, 61, 3.007, 3.017), (1, 62, 1, 64, 3.149, 3.159), (1, 64, 1, 65, 3.239, 3.249), (1, 65, 1, 67, 3.340, 3.350), (1, 67, 1, 68, 3.428, 3.438), (1, 68, 1, 71, 3.509, 3.519), (0, 70, 0, 73, 3.525, 3.527), (1, 72, 1, 74, 3.680, 3.690), (1, 74, 1, 75, 3.767, 3.777), (1, 77, 1, 79, 3.954, 3.964), (1, 79, 1, 81, 4.023, 4.033), (1, 81, 1, 82, 4.103, 4.113), (0, 80, 0, 83, 4.002, 4.004), (1, 82, 1, 84, 4.184, 4.194), (1, 84, 1, 85, 4.268, 4.278), (1, 85, 1, 87, 4.368, 4.378), (1, 87, 1, 89, 4.455, 4.465), (1, 91, 1, 92, 4.595, 4.605), (0, 90, 0, 93, 4.563, 4.565), (1, 94, 1, 95, 4.743, 4.753), (1, 95, 1, 97, 4.835, 4.845), (1, 98, 1, 101, 4.989, 4.999), (1, 102, 1, 104, 5.159, 5.169), (1, 104, 1, 105, 5.245, 5.255), (1, 105, 1, 107, 5.330, 5.340), (1, 107, 1, 109, 5.395, 5.405), (1, 111, 1, 112, 5.503, 5.513), (1, 112, 1, 114, 5.578, 5.588), (1, 114, 1, 115, 5.670, 5.680), (1, 115, 1, 116, 5.747, 5.757), (0, 113, 0, 117, 5.653, 5.655), (0, 117, 0, 119, 5.792, 5.794), (0, 119, 0, 120, 5.877, 5.879), (0, 120, 0, 121, 5.923, 6.033), ],

], vbd_task_name_map, with_text=False, figure_name="example3_iter_12_schedule")




plt.show()
plt.ion()

