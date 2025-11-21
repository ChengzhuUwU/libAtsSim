#define METAL_CODE
#define SIM_USE_SIMD true

#include <metal_stdlib>
using namespace metal;
#include "../../SharedDefine/float_n.h"
#include "../../SharedDefine/float_n_n.h"
#include "../../SharedDefine/gpu_algorism.h"
#include "shared/vivace_kernel.h"


//
// Vivave Parallel Graph-Coloring
//

// Update Function
kernel void scan_uncolored_set_GPU(
	Pointer(uint) collision_count,
	Pointer(uint) uncolored_verts,
	Pointer(uint) uncolored_verts_copy,
	Pointer(uint) uncolored_verts_count, 
	Pointer(uchar) colored, 
	Constant(uint) curr_loop,
    const uint i [[thread_position_in_grid]])
{
    // const uint prev_uncolored = 
    //     curr_loop == 0 ? collision_count[0] : 
    //     VivaceGraphCloring::fn_get_current_num_uncolored(uncolored_verts_count, curr_loop - 1);
    // if (i >= prev_uncolored) return;

    VivaceGraphCloring::scan_uncolored_set_GPU(i, 
        collision_count, 
        uncolored_verts, uncolored_verts_copy, 
        uncolored_verts_count, colored, 
        curr_loop);
}

kernel void copy_scaned_indices_from(
    Pointer(uint) collision_count,
	Pointer(uint) uncolored_verts_copy,
	Pointer(uint) uncolored_verts,
	Pointer(uint) uncolored_verts_count,
	Pointer(Int4) uncolored_verts_indirect_cmd_buffer,

	Constant(uint) curr_loop,
    const uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::copy_scaned_indices_from(i, 
        collision_count, 
        uncolored_verts_copy, uncolored_verts, uncolored_verts_count,
        uncolored_verts_indirect_cmd_buffer, 
        curr_loop);
}

// Init Vivace
kernel void reduce_degree_vv(
	Pointer(Int2) collision_pair,
	Pointer(uint) vert_adj_collsion_pair_num,
	Pointer(uint) uncolored_verts,
	Pointer(uint) num_verts_in_cluster,
	Pointer(uint) clusterd_constraint_self_collision,

	Pointer(uchar) colored,
	Pointer(uchar) c_v,

    Pointer(uint) self_collision_count,
	Pointer(Int2) block_min_max_degree,

    uint element_id [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids)
{
    const uint num_collision = self_collision_count[0];
    const bool is_valid = element_id < num_collision;

    Int2 min_max_degree = is_valid ? 
        VivaceGraphCloring::reduce_degree_and_set_zero_degree_nodes_template(
            element_id, 
            collision_pair, vert_adj_collsion_pair_num, uncolored_verts, num_verts_in_cluster, clusterd_constraint_self_collision,
            colored, c_v) : 
        makeInt2(1000, 0);

    reduce_min_max(min_max_degree, tid, 256);

    if (tid == 0)
    {
        block_min_max_degree[bid] = min_max_degree;
    }
}
kernel void reduce_degree_vf(
	Pointer(Int4) collision_pair,
	Pointer(uint) vert_adj_collsion_pair_num,
	Pointer(uint) uncolored_verts,
	Pointer(uint) num_verts_in_cluster,
	Pointer(uint) clusterd_constraint_self_collision,

	Pointer(uchar) colored,
	Pointer(uchar) c_v,

    Pointer(uint) self_collision_count,
	Pointer(Int2) block_min_max_degree,

    uint element_id [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids)
{
    const uint num_collision = self_collision_count[0];
    const bool is_valid = element_id < num_collision;

    Int2 min_max_degree = is_valid ? 
        VivaceGraphCloring::reduce_degree_and_set_zero_degree_nodes_template(
            element_id, 
            collision_pair, vert_adj_collsion_pair_num, uncolored_verts, num_verts_in_cluster, clusterd_constraint_self_collision,
            colored, c_v) : 
        makeInt2(1000, 0);

    reduce_min_max(min_max_degree, tid, 256);

    if (tid == 0)
    {
        block_min_max_degree[bid] = min_max_degree;
    }
}

kernel void reduce_degree_second_pass_AND_set_max_color_from_global_degree(
    Pointer(uint) num_verts_in_cluster,
	Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
	Pointer(uint) num_colors_self_collision_vv,
	Pointer(uint) collision_count,

    Pointer(Int2) block_min_max_degree,

    uint index [[thread_position_in_grid]],
    threadgroup_ids)
{
    const uint num_verts_total = collision_count[0];
    const uint needed_num_threads = get_dispatch_num(num_verts_total, 256);
    
    Int2 min_max_degree = tid < needed_num_threads ? block_min_max_degree[tid] : makeInt2(1000, 0);

    if (SECOND_REDUCE_DIM < needed_num_threads)
    {
        for (uint bid = 1; bid < get_dispatch_num(needed_num_threads, SECOND_REDUCE_DIM); bid++)
        {
            const uint mapped_index = bid * SECOND_REDUCE_DIM + tid;
            Int2 curr_min_max_degree = mapped_index < needed_num_threads ? block_min_max_degree[mapped_index] : makeInt2(1000, 0);
            min_max_degree = VivaceGraphCloring::reduce_degree_binary_function(min_max_degree, curr_min_max_degree);
        }
    }
    
    reduce_min_max(min_max_degree, tid, SECOND_REDUCE_DIM);

    if (index == 0)
    {
        // Set Max Color From Max Degreee, And Init Clusters Count
        VivaceGraphCloring::set_max_color_from_max_degree(
            num_verts_in_cluster, uncolored_verts_indirect_cmd_buffer,
            num_colors_self_collision_vv, 
            collision_count, min_max_degree);
    }
}

kernel void init_palette(
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
	Pointer(uint) collision_count,
	Pointer(uint) num_colors_self_collision_vv,
	Pointer(uint64) P_v,
	Pointer(uint64) P_v_prev,
	Pointer(uchar) next_color,
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::init_palette(i, 
        uncolored_verts_indirect_cmd_buffer, uncolored_verts_count, uncolored_verts, collision_count, num_colors_self_collision_vv,
        P_v, P_v_prev, next_color);
}


kernel void tentative_coloring(
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint) num_colors_self_collision_vv,
	Pointer(uint64) P_v,
    Pointer(uint64) P_v_prev,
	Pointer(uchar) next_color,
	Pointer(uchar) c_v,
	Pointer(uchar) pre_computed_random_number_256,
    Pointer(uint) collision_count,
	Constant(uint) curr_loop, 
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::tentative_coloring(i, 
                uncolored_verts_count, uncolored_verts, num_colors_self_collision_vv,
                P_v, P_v_prev, next_color, c_v, 
                pre_computed_random_number_256, collision_count,
                curr_loop);
}

kernel void copy_colred(
    Pointer(uchar) colored_in_curr_pass,
    Pointer(uchar) colored,
    Pointer(uint) collision_count,
    uint i [[thread_position_in_grid]]
)
{
    if (i < collision_count[0])
    {
        // colored_in_curr_pass[i] = colored[i];
        colored_in_curr_pass[i] = 0;
    }
}

kernel void conflict_resolution_per_element_vv(
    Pointer(Int2) collision_pair,
	Pointer(uint) vert_adj_collsion_pair_num,
	Pointer(uint) vert_adj_collsion_pair_prefix,
	Pointer(uint) vert_adj_collsion_pair_list,

    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,

	Pointer(uint64) P_v,
	Pointer(uchar) c_v,
	Pointer(uchar) colored,
	Pointer(uchar) colored_in_curr_pass,

	Pointer(uint) clusterd_constraint_self_collision,
	Pointer(uint) num_verts_in_cluster,

    Constant(uint) curr_loop, 
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::conflict_resolution_PerConstraint_template(i, 
                collision_pair, vert_adj_collsion_pair_num, vert_adj_collsion_pair_prefix, vert_adj_collsion_pair_list,
                uncolored_verts_count, uncolored_verts,
                P_v, c_v, colored, colored_in_curr_pass,
                clusterd_constraint_self_collision, num_verts_in_cluster,
                curr_loop);
}
kernel void update_palatte_from_current_tentative_coloring_result_vv(
    Pointer(Int2) collision_pair,
	Pointer(uint) vert_adj_collsion_pair_num,
	Pointer(uint) vert_adj_collsion_pair_prefix,
	Pointer(uint) vert_adj_collsion_pair_list,

    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,

	Pointer(uint64) P_v,
	Pointer(uchar) c_v,
	Pointer(uchar) colored,
	Pointer(uchar) colored_in_curr_pass,

	Pointer(uint) clusterd_constraint_self_collision,
	Pointer(uint) num_verts_in_cluster,

    Constant(uint) curr_loop, 
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::update_palatte_from_current_tentative_coloring_result_template(i, 
                collision_pair, vert_adj_collsion_pair_num, vert_adj_collsion_pair_prefix, vert_adj_collsion_pair_list,
                uncolored_verts_count, uncolored_verts,
                P_v, c_v, colored, colored_in_curr_pass,
                clusterd_constraint_self_collision, num_verts_in_cluster,
                curr_loop);
}

kernel void conflict_resolution_per_element_vf(
    Pointer(Int4) collision_pair,
	Pointer(uint) vert_adj_collsion_pair_num,
	Pointer(uint) vert_adj_collsion_pair_prefix,
	Pointer(uint) vert_adj_collsion_pair_list,

    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,

	Pointer(uint64) P_v,
	Pointer(uchar) c_v,
	Pointer(uchar) colored,
	Pointer(uchar) colored_in_curr_pass,

	Pointer(uint) clusterd_constraint_self_collision,
	Pointer(uint) num_verts_in_cluster,

    Constant(uint) curr_loop, 
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::conflict_resolution_PerConstraint_template(i, 
                collision_pair, vert_adj_collsion_pair_num, vert_adj_collsion_pair_prefix, vert_adj_collsion_pair_list,
                uncolored_verts_count, uncolored_verts,
                P_v, c_v, colored, colored_in_curr_pass,
                clusterd_constraint_self_collision, num_verts_in_cluster,
                curr_loop);
}


// fn_update_uncolored_count

kernel void feed_the_hungry(
    Pointer(uint) uncolored_verts,
	Pointer(uint64) P_v,
	Pointer(uint) collision_count,
	Pointer(uchar) next_color,
	Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) num_colors_self_collision_vv,
	Constant(uint) curr_loop,
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::feed_the_hungry(i, 
                uncolored_verts,
                P_v,
                collision_count,
                next_color,
                uncolored_verts_indirect_cmd_buffer, 
                uncolored_verts_count,
                num_colors_self_collision_vv,
                curr_loop);
}

kernel void put_rest_vertices_into_additional_color(
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint64) P_v,
	Pointer(uint) collision_count,
	Pointer(uchar) pre_computed_random_number_256,
	Pointer(uint) num_colors_self_collision_vv,
	Pointer(uint) num_verts_in_cluster,
	Pointer(uint) clusterd_constraint_self_collision_vv,
	Constant(uint) curr_loop,
	Constant(uint) num_verts_total,
    uint i [[thread_position_in_grid]])
{
    // VivaceGraphCloring::put_rest_vertices_into_additional_color(i, 
    //         uncolored_verts_count, uncolored_verts,
    //         P_v, collision_count,
    //         pre_computed_random_number_256,
    //         num_colors_self_collision_vv,
    //         num_verts_in_cluster, 
    //         clusterd_constraint_self_collision_vv, 
    //         curr_loop, num_verts_total);
}

kernel void put_rest_vertices_into_random_color(
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint64) P_v,
	Pointer(uint) collision_count,
	Pointer(uchar) pre_computed_random_number_256,
	Pointer(uint) num_colors_self_collision_vv,
	Pointer(uint) num_verts_in_cluster,
	Pointer(uint) clusterd_constraint_self_collision_vv,
    Constant(uint) curr_loop,
    uint i [[thread_position_in_grid]])
{
    VivaceGraphCloring::put_rest_vertices_into_random_color(i, 
            uncolored_verts_count, uncolored_verts,
            P_v, collision_count,
            pre_computed_random_number_256,
            num_colors_self_collision_vv,
            num_verts_in_cluster, 
            clusterd_constraint_self_collision_vv, 
            curr_loop);
}

kernel void scan_num_verts_in_color(
    Pointer(uint) num_verts_in_cluster,
    Pointer(uint) cluster_prefix,
	Pointer(Int4) clusterd_constraint_self_collision_indirect_cmd_buffer,
    const uint i [[thread_position_in_grid]]
)
{   
    if (i < 60)
    {
        const uint color = i;
        const uint num_verts = num_verts_in_cluster[color];
        clusterd_constraint_self_collision_indirect_cmd_buffer[i] = make_indirect_command_buffer(num_verts);
        uint prefix = warp_prefix_sum_exclusive(num_verts);

        ThreadGroup uint cache_prefix = 0;
        if (i == 31)
        {
            cache_prefix = prefix + num_verts;
        }

        THREAD_GROUP_SYNC;

        if (i > 31)
        {
            prefix += cache_prefix;
        }

        cluster_prefix[i] = prefix;
    }
}

kernel void fill_in_cluster_indices_vv(
    Pointer(uint) verts_prefix_in_cluster,
	Pointer(uchar) c_v,
	Pointer(uint) cluster_prefix,
	Pointer(uint) clusterd_constraint_self_collision,
	Pointer(uint) self_collision_count,
    Pointer(ProximityVV) narrow_phase_list_pair_vv ,
	Pointer(ProximityVV) narrow_phase_list_pair_vv_merged,
    const uint i [[thread_position_in_grid]])
{
    const uint num_collision = self_collision_count[0];
    if (i < num_collision)
    {
        VivaceGraphCloring::fill_in_cluster_indices(i, 
            verts_prefix_in_cluster,
            c_v, cluster_prefix, clusterd_constraint_self_collision,
            narrow_phase_list_pair_vv, narrow_phase_list_pair_vv_merged);
    }
}
