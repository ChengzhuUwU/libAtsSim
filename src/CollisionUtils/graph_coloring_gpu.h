#pragma once

#include "cpu_parallel.h"
#include "gpu_function.h"
#include "xpbd_data.h"
#include "xpbd_constraints.h"
#include "shared/vivace_kernel.h"
#include "graph_coloring_cpu.h"

class RandomGraphColoringGPU {
private:
    VivaceColoringData *vivace_data;
    XpbdSelfCollision *self_collision_data;
    RandomGraphColoringCPU *vivace_cpu;

private:
    gpuFunction fn_scan_uncolored_set;
    gpuFunction fn_copy_scaned_indices_from;
    gpuFunction fn_reduce_degree_vv;
    gpuFunction fn_reduce_degree_vf;
    gpuFunction fn_reduce_degree_second_pass_AND_set_max_color_from_global_degree;
    gpuFunction fn_init_palette;
    gpuFunction fn_tentative_coloring;
    gpuFunction fn_copy_colred;
    gpuFunction fn_conflict_resolution_vv;
    gpuFunction fn_conflict_resolution_vf;
    gpuFunction fn_update_palatte_from_current_tentative_coloring_result_vv;
    gpuFunction fn_feed_the_hungry;
    gpuFunction fn_put_rest_vertices_into_additional_color;
    gpuFunction fn_put_rest_vertices_into_random_color;
    gpuFunction fn_scan_num_verts_in_color;
    gpuFunction fn_fill_in_cluster_indices_vv;

public:
    void init_graph_coloring_system(VivaceColoringData &input_data, XpbdSelfCollision &input_self_collision, RandomGraphColoringCPU &input_vivace_cpu);

public:
    void graph_coloring_vivace();

    void reduce_degree_and_set_max_color_from_max_degree();
    void scan_uncolored_set(const uint curr_loop);
    void init_palette();
    void tentative_coloring(const uint curr_loop);
    void conflict_resolution_per_vert_vv(const uint curr_loop);
    void conflict_resolution_per_vert_vf(const uint curr_loop);
    void conflict_resolution_per_element_vv(const uint curr_loop);
    void conflict_resolution_per_element_vf(const uint curr_loop);
    void feed_the_hungry(const uint curr_loop);
    void put_rest_vertices_into_additional_color();
    void put_rest_vertices_into_random_color();
    void make_cluster_indirect_cmd_buffer();

    uint fn_get_current_uncolored_count(const uint curr_loop) {
        return vivace_data->uncolored_verts_count[curr_loop];
    };
    void fn_launch_function_in_loop(const uint curr_loop, gpuFunction &func) {
        func.launch_async(vivace_data->uncolored_verts_indirect_cmd_buffer, curr_loop);
    }
    void print_node_neighbor(const uint pair_idx);
};