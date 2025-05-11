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

visualize_schedule3([ # Iter 1
    [(19, 6, 1, 0.555, 1.555), (15, 12, 0, 1.557, 2.557), (20, 17, 1, 2.559, 3.559), (16, 23, 0, 3.561, 4.561), (22, 29, 1, 4.563, 5.563), (18, 35, 0, 5.565, 6.565), (14, 41, 1, 6.567, 7.567), (19, 46, 0, 7.569, 8.569), (15, 52, 1, 8.571, 9.571), (21, 58, 0, 9.573, 10.573), (17, 64, 1, 10.575, 11.575), (22, 69, 0, 11.577, 12.577), (18, 75, 1, 12.579, 13.579)],
    [(0, 83, 4294967295, 0.000, 0.200), (1, 0, 4294967295, 0.210, 0.410), (14, 1, 0, 0.420, 0.620), (15, 2, 0, 0.630, 0.830), (16, 3, 0, 0.840, 1.040), (17, 4, 0, 1.050, 1.250), (18, 5, 2, 1.260, 1.460), (20, 7, 2, 1.470, 1.670), (21, 8, 2, 1.680, 1.880), (22, 9, 1, 1.890, 2.090), (23, 10, 1, 2.100, 2.300), (14, 11, 3, 2.310, 2.510), (16, 13, 3, 2.520, 2.720), (17, 14, 3, 2.730, 2.930), (18, 15, 0, 2.940, 3.140), (19, 16, 0, 3.150, 3.350), (21, 18, 4, 3.360, 3.560), (22, 19, 4, 3.570, 3.770), (23, 20, 1, 3.780, 3.980), (14, 21, 1, 3.990, 4.190), (15, 22, 1, 4.200, 4.400), (17, 24, 5, 4.410, 4.610), (18, 25, 5, 4.620, 4.820), (19, 26, 0, 4.830, 5.030), (20, 27, 0, 5.040, 5.240), (21, 28, 6, 5.250, 5.450), (23, 30, 6, 5.460, 5.660), (14, 31, 6, 5.670, 5.870), (15, 32, 1, 5.880, 6.080), (16, 33, 1, 6.090, 6.290), (17, 34, 7, 6.300, 6.500), (19, 36, 7, 6.510, 6.710), (20, 37, 7, 6.720, 6.920), (21, 38, 0, 6.930, 7.130), (22, 39, 0, 7.140, 7.340), (23, 40, 8, 7.350, 7.550), (15, 42, 8, 7.560, 7.760), (16, 43, 8, 7.770, 7.970), (17, 44, 1, 7.980, 8.180), (18, 45, 1, 8.190, 8.390), (20, 47, 9, 8.400, 8.600), (21, 48, 9, 8.610, 8.810), (22, 49, 0, 8.820, 9.020), (23, 50, 0, 9.030, 9.230), (14, 51, 10, 9.240, 9.440), (16, 53, 10, 9.450, 9.650), (17, 54, 10, 9.660, 9.860), (18, 55, 1, 9.870, 10.070), (19, 56, 1, 10.080, 10.280), (20, 57, 11, 10.290, 10.490), (22, 59, 11, 10.500, 10.700), (23, 60, 11, 10.710, 10.910), (14, 61, 0, 10.920, 11.120), (15, 62, 0, 11.130, 11.330), (16, 63, 12, 11.340, 11.540), (18, 65, 12, 11.550, 11.750), (19, 66, 12, 11.760, 11.960), (20, 67, 1, 11.970, 12.170), (21, 68, 1, 12.180, 12.380), (23, 70, 13, 12.390, 12.590), (14, 71, 13, 12.600, 12.800), (15, 72, 0, 12.810, 13.010), (16, 73, 0, 13.020, 13.220), (17, 74, 0, 13.230, 13.430), (19, 76, 0, 13.440, 13.640), (20, 77, 0, 13.650, 13.850), (21, 78, 1, 13.860, 14.060), (22, 79, 1, 14.070, 14.270), (23, 80, 1, 14.280, 14.480), (0, 81, 1, 14.490, 14.690), (0, 82, 4294967295, 14.700, 14.900), (0, 84, 4294967295, 14.910, 15.110)],
    [(0, 6, 1, 9, 1.555, 1.890), (1, 4, 0, 12, 1.250, 1.557), (0, 12, 1, 15, 2.557, 2.940), (1, 10, 0, 17, 2.300, 2.559), (0, 17, 1, 20, 3.559, 3.780), (1, 16, 0, 23, 3.350, 3.561), (0, 23, 1, 26, 4.561, 4.830), (1, 22, 0, 29, 4.400, 4.563), (0, 29, 1, 32, 5.563, 5.880), (1, 27, 0, 35, 5.240, 5.565), (0, 35, 1, 38, 6.565, 6.930), (1, 33, 0, 41, 6.290, 6.567), (0, 41, 1, 44, 7.567, 7.980), (1, 39, 0, 46, 7.340, 7.569), (0, 46, 1, 49, 8.569, 8.820), (1, 45, 0, 52, 8.390, 8.571), (0, 52, 1, 55, 9.571, 9.870), (1, 50, 0, 58, 9.230, 9.573), (0, 58, 1, 61, 10.573, 10.920), (1, 56, 0, 64, 10.280, 10.575), (0, 64, 1, 67, 11.575, 11.970), (1, 62, 0, 69, 11.330, 11.577), (0, 69, 1, 72, 12.577, 12.810), (1, 68, 0, 75, 12.380, 12.579), (0, 75, 1, 78, 13.579, 13.860), ],
    [(1, 0, 1, 1, 0.410, 0.420), (1, 4, 1, 5, 1.250, 1.260), (1, 0, 0, 6, 0.410, 0.555), (1, 8, 1, 9, 1.880, 1.890), (1, 10, 1, 11, 2.300, 2.310), (1, 14, 1, 15, 2.930, 2.940), (1, 16, 1, 18, 3.350, 3.360), (1, 19, 1, 20, 3.770, 3.780), (1, 22, 1, 24, 4.400, 4.410), (1, 25, 1, 26, 4.820, 4.830), (1, 27, 1, 28, 5.240, 5.250), (1, 31, 1, 32, 5.870, 5.880), (1, 33, 1, 34, 6.290, 6.300), (1, 37, 1, 38, 6.920, 6.930), (1, 39, 1, 40, 7.340, 7.350), (1, 43, 1, 44, 7.970, 7.980), (1, 45, 1, 47, 8.390, 8.400), (1, 48, 1, 49, 8.810, 8.820), (1, 50, 1, 51, 9.230, 9.240), (1, 54, 1, 55, 9.860, 9.870), (1, 56, 1, 57, 10.280, 10.290), (1, 60, 1, 61, 10.910, 10.920), (1, 62, 1, 63, 11.330, 11.340), (1, 66, 1, 67, 11.960, 11.970), (1, 68, 1, 70, 12.380, 12.390), (1, 71, 1, 72, 12.800, 12.810), (1, 77, 1, 78, 13.850, 13.860), ],
    
], vbd_task_name_map, with_text=True, figure_name="")




plt.show()
plt.ion()

