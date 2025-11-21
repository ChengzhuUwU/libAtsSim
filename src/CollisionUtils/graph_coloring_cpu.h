#pragma once

#include "cpu_parallel.h"
#include "xpbd_data.h"
#include "xpbd_constraints.h"
#include "shared/vivace_kernel.h"

class RandomGraphColoringCPU {
private:
    VivaceColoringData *vivace_data;
    XpbdSelfCollision *self_collision_data;

public:
    void init_graph_coloring_system(VivaceColoringData &input_data, XpbdSelfCollision &input_self_collision);

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
    void print_node_neighbor(const uint pair_idx);
};