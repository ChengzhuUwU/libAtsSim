#define METAL_CODE
#define SIM_USE_SIMD true

#include <metal_stdlib>
using namespace metal;
#include "../../SharedDefine/float_n.h"
#include "../../SharedDefine/float_n_n.h"
#include "../../SharedDefine/gpu_algorism.h"
// #include "shared/dynamics_utils.h"
// #include "shared/dynamics_kernel.h"
#include "../Solver/xpbd_constraints.h"

//
// Broad Phase
//

kernel void reset_collision_system(
	Pointer(uint) collision_count,
	Pointer(Int4) collision_indirect_cmd_buffer,
	Pointer(uint) num_verts_in_cluster,
	Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
	Pointer(uint) uncolored_verts_count,
	Pointer(uint) hash_table_count,
	Pointer(uint) hash_table_prefix,
    Pointer(uint) hash_table_belongs,
	Pointer(uint) vert_VV_num_broad_phase,
	Pointer(uint) vert_VV_num_narrow_phase,
	Constant(bool) reset_coloring_system,
	Constant(bool) reset_hash_table,
	Constant(bool) reset_broad_narrow_count,
    uint vid [[thread_position_in_grid]])
{
    SpatialHashing::reset_collision_system(vid, 
        collision_count, collision_indirect_cmd_buffer, 
        num_verts_in_cluster, uncolored_verts_indirect_cmd_buffer, uncolored_verts_count,
        hash_table_count, hash_table_prefix, hash_table_belongs,
        vert_VV_num_broad_phase, vert_VV_num_narrow_phase,
        reset_coloring_system, reset_hash_table, reset_broad_narrow_count);
}
kernel void reset_broad_narrow_count(
	Pointer(uint) vert_VV_num_broad_phase,
	Pointer(uint) vert_VV_num_narrow_phase,
    uint vid [[thread_position_in_grid]])
{
    SpatialHashing::reset_broad_narrow_count(vid, 
        vert_VV_num_broad_phase, vert_VV_num_narrow_phase);
}

kernel void update_obstacle_position_in_substep(
	Pointer(Float3) sa_start_position, 
	Pointer(Float3) sa_next_position,
	Pointer(Float3) sa_substep_position, 
	Pointer(Float3) sa_substep_velocity, 
	Constant(float) alpha,
	Constant(float) substep_dt,
    uint vid [[thread_position_in_grid]])
{
    Constrains::Core::update_obstacle_position_in_substep(vid, 
        sa_start_position, sa_next_position, sa_substep_position, sa_substep_velocity,
        alpha, substep_dt);
}
kernel void update_obstacle_normal_in_substep(
	Pointer(Int3) sa_faces, 
	Pointer(Float3) sa_substep_position, 
	Pointer(Float3) sa_face_normal,
    uint fid [[thread_position_in_grid]])
{
    Constrains::Core::update_obstacle_normal_in_substep(fid, 
        sa_faces, sa_substep_position, sa_face_normal);
}

kernel void prepare_position_for_collision_detection(
	Pointer(Float3) sa_position_cloth,
	Pointer(Float3) sa_position_tet,
	Pointer(uint) sa_surface_verts,
	Pointer(Float3) sa_position_for_detection_bg,
	Pointer(Float3) sa_position_for_detection_ed,
	Constant(uint) num_verts_cloth,
    uint vid [[thread_position_in_grid]])
{
    Constrains::prepare_position_for_collision_detection(vid, 
        sa_position_cloth, sa_position_tet, sa_surface_verts, 
        sa_position_for_detection_bg, sa_position_for_detection_ed, 
        num_verts_cloth);
}
kernel void update_tet_surface_position_for_collision_detection(
	Pointer(Float3) sa_iter_start_position,
	Pointer(uint) sa_surface_verts,
	Pointer(Float3) sa_surface_position,
    uint surface_id [[thread_position_in_grid]])
{
    const uint vid = sa_surface_verts[surface_id];
    sa_surface_position[surface_id] = sa_iter_start_position[vid];
}

kernel void compute_global_aabb(
    Pointer(Float3) sa_soft_body_position_bg,
    Pointer(Float3) sa_soft_body_position_ed,
    Pointer(Float3) sa_obstacle_position,
    Pointer(AABB) sa_block_aabb,
    Constant(uint) num_verts_collision_total,
    Constant(uint) num_verts_obstacle,
    Constant(float) thickness,
    uint vid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids
)
{
    AABB aabb;
    if (vid < num_verts_collision_total)
    {
        aabb = AABB(sa_soft_body_position_bg[vid]) + AABB(sa_soft_body_position_ed[vid]);
    }
    else if (vid < num_verts_collision_total + num_verts_obstacle)
    {
        aabb = AABB(sa_obstacle_position[vid - num_verts_collision_total]);
    }
    reduce_aabb(aabb, tid, 256);
    if(tid == 0)
    {
        sa_block_aabb[bid] = aabb + 3.0f * thickness;
    }
}
// kernel void compute_global_aabb(
//     Pointer(Float3) sa_cloth_position,
//     Pointer(Float3) sa_tet_position,
//     Pointer(Float3) sa_obstacle_position,
//     Pointer(AABB) sa_block_aabb,
//     Constant(uint) num_verts_cloth,
//     Constant(uint) num_verts_tet,
//     Constant(uint) num_verts_obstacle,
//     Constant(float) thickness,
//     uint vid [[thread_position_in_grid]],
//     uint bid [[threadgroup_position_in_grid]],
//     threadgroup_ids
// )
// {
//     AABB aabb;
//     if (vid < num_verts_cloth)
//     {
//         aabb = AABB(sa_cloth_position[vid]);
//     }
//     else if (vid < num_verts_cloth + num_verts_tet)
//     {
//         aabb = AABB(sa_tet_position[vid - num_verts_cloth]);
//     }
//     else if (vid < num_verts_cloth + num_verts_tet + num_verts_obstacle)
//     {
//         aabb = AABB(sa_obstacle_position[vid - num_verts_cloth - num_verts_tet]);
//     }
//     reduce_aabb(aabb, tid, 256);
//     if(tid == 0)
//     {
//         sa_block_aabb[bid] = aabb + 3.0f * thickness;
//     }
// }
kernel void compute_global_aabb_second_pass(
    Pointer(AABB) sa_block_aabb,
    Constant(uint) num_verts_in_scene,
    uint index [[thread_position_in_grid]],
    threadgroup_ids
)
{
    const uint desire_count = (num_verts_in_scene + 256 - 1) / 256;
    const uint curr_block_dim = SECOND_REDUCE_DIM;
    const uint loop_count = get_dispatch_num(desire_count, curr_block_dim);

    AABB aabb;

    for (uint blockIdx = 0; blockIdx < loop_count; blockIdx++)
    {
        const uint actual_idx = blockIdx * SECOND_REDUCE_DIM + index;
        bool is_valid = actual_idx < desire_count;
        const AABB read_aabb = is_valid ? sa_block_aabb[actual_idx] : AABB();
        aabb += read_aabb;
    }
    
    reduce_aabb(aabb, tid, SECOND_REDUCE_DIM);
    if(index == 0)
    {
        sa_block_aabb[0] = aabb;
    }
}

kernel void fill_in_hash_table(
	Pointer(Float3) sa_iter_position,
	Pointer(uint) hash_table_count,
    Pointer(uint) hash_table_belongs,
    Pointer(uint) hash_table_cell_accessed_count,
	Pointer(uint) hash_table_vert_offset,
    Pointer(AABB) sa_block_aabb,
	Constant(uint) table_size,
    uint vid [[thread_position_in_grid]])
{
    SpatialHashing::fill_in_hash_table(vid, 
        sa_iter_position, 
        hash_table_count, hash_table_belongs, hash_table_cell_accessed_count, hash_table_vert_offset, sa_block_aabb,
        table_size);
}
kernel void set_hash_table_flag(
	Pointer(uchar) hash_table_flag, 
	Pointer(uint) hash_table_cell_accessed_count,
    uint cell_id [[thread_position_in_grid]])
{
	SpatialHashing::set_hash_table_flag(cell_id, 
        hash_table_flag, hash_table_cell_accessed_count);
}

inline uint fn_atomic_add_numVerts_in_cell_in_block(Pointer(uint) collision_count, const uint num_verts_in_cell) { return atomic_add(collision_count[5], num_verts_in_cell); }

kernel void scan_hash_table(
	Pointer(uint) collision_count,
	Pointer(uint) hash_table_count,
	Pointer(uint) hash_table_prefix,
	Constant(uint) table_size,
    uint cell_id [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids)
{
    const bool is_valid = cell_id < table_size;
    uint num_verts_in_cell = is_valid ? hash_table_count[cell_id] : 0;

    const uint blockDim = 256;
    ThreadGroup uint cache_scan[blockDim]; // 1 MB
    cache_scan[tid] = num_verts_in_cell;
    THREAD_GROUP_SYNC;

    uint temp;
    SCAN(tid, blockDim, cache_scan, temp);	
    THREAD_GROUP_SYNC;
    uint prefix = cache_scan[tid]; // exclusive

    ThreadGroup uint block_prefix_in_global = 0;
    if (tid == blockDim - 1)
    {
        const uint num_verts_in_block = prefix + num_verts_in_cell;
        block_prefix_in_global = fn_atomic_add_numVerts_in_cell_in_block(collision_count, num_verts_in_block);
    }
    THREAD_GROUP_SYNC;
    
    if (is_valid) hash_table_prefix[cell_id] = block_prefix_in_global + prefix;
}

kernel void insert_vert_into_hash_table(
	Pointer(Float3) sa_iter_position,
	Pointer(uint) hash_table_vert_offset,
	Pointer(uint) hash_table_prefix,
	Pointer(uint) hash_table_belongs,
	Pointer(uint) hash_table,
	Pointer(AABB) sa_block_aabb,
	Constant(uint) table_size,
    uint vid [[thread_position_in_grid]])
{
    SpatialHashing::insert_vert_into_hash_table(vid, 
            sa_iter_position, hash_table_vert_offset,
            hash_table_prefix, hash_table_belongs, hash_table, sa_block_aabb, table_size);
    
}

kernel void spatial_hashing_query_vv(
	Pointer(Float3) sa_iter_position,
	Pointer(Float3) sa_next_position,
	Pointer(uint) hash_table_count,
	Pointer(uint) hash_table_prefix,
    Pointer(uint) hash_table_belongs,
    Pointer(uchar) hash_table_flag,
	Pointer(uint) hash_table,
	Pointer(uint) vert_VV_num_broad_phase,
	Pointer(uint) broad_phase_list,
    Pointer(AABB) sa_block_aabb,
	Constant(uint) table_size,
    Constant(uint) max_vv_per_vert_broad_phase,
	Constant(float) query_range,
    uint vid [[thread_position_in_grid]])
{
    SpatialHashing::spatial_hashing_query_vv(vid,
            sa_iter_position, sa_next_position, 
            hash_table_count, hash_table_prefix, hash_table_belongs, hash_table_flag, hash_table,
            vert_VV_num_broad_phase,  broad_phase_list, sa_block_aabb, table_size, 
            max_vv_per_vert_broad_phase, query_range);
}

//
// Narrow Phase
//
kernel void narrow_phase_vv_self_collision_from_collision_pair(
	Pointer(Float3) sa_detection_position_bg,
	Pointer(Float3) sa_detection_position_ed,
	Pointer(Float3) sa_detection_rest_position, 

	Pointer(uint) vert_VV_num_broad_phase,
	Pointer(uint) broad_phase_list,

	Pointer(Int2) narrow_phase_list_indices_vv,
	Pointer(ProximityVV) narrow_phase_list_pair_vv,
	Pointer(uint) collision_count,
	Pointer(Int4) self_collision_indirect_cmd_buffer,
	Pointer(uint) vert_VV_num_narrow_phase,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uchar) collision_pair_offset_in_vert,
	Pointer(uint) vert_adj_pairs,

	Constant(float) thickness_1,
	Constant(float) thickness_2,
	Constant(float) stiffness_collision,

    uint vid [[thread_position_in_grid]])
{
    NarrowPhase::narrow_phase_vv_self_collision_from_collision_pair(vid,
        sa_detection_position_bg,
        sa_detection_position_ed,
        sa_detection_rest_position, 

        vert_VV_num_broad_phase,
        broad_phase_list,

        narrow_phase_list_indices_vv,
        narrow_phase_list_pair_vv,
        collision_count,
        self_collision_indirect_cmd_buffer,
        vert_VV_num_narrow_phase,
        vert_VV_prefix_narrow_phase,
        collision_pair_offset_in_vert,
        vert_adj_pairs,

	    thickness_1,
	    thickness_2,
	    stiffness_collision
    );
}

kernel void narrow_phase_vf_obstacle_collision_from_collision_pair(
	Pointer(Float3) sa_detection_position_bg,
	Pointer(Float3) sa_detection_position_ed,
	Pointer(Float3) sa_obstacle_position,
	
	Pointer(float) sa_detection_vert_area,
	Pointer(Float3) sa_obs_vert_normal,
	Pointer(Float3) sa_obs_face_normal,
	Pointer(Int3) sa_obstacle_faces,

	Pointer(uint) vert_VV_num_broad_phase,
	Pointer(uint) broad_phase_list,

	Pointer(Int4) narrow_phase_list_indices_vf,
	Pointer(ProximityVF) narrow_phase_list_pair_vf,
	Pointer(uint) collision_count,
	Pointer(Int4) indirect_command_buffer,
	
	Pointer(uint) vert_VV_num_narrow_phase,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uchar) collision_pair_offset_in_vert,
	Pointer(uint) vert_adj_verts_vv,

	Constant(uint ) max_vf_broad_phase_num,
	Constant(uint ) max_vf_narrow_phase_num,
	Constant(float) thickness_1,
	Constant(float) thickness_2,
	Constant(float) stiffness_collision,
    uint vid [[thread_position_in_grid]])
{
    NarrowPhase::narrow_phase_vf_obstacle_collision_from_collision_pair(vid,
        sa_detection_position_bg,
        sa_detection_position_ed,
        sa_obstacle_position,
    
        sa_detection_vert_area,
        sa_obs_vert_normal,
        sa_obs_face_normal,
        sa_obstacle_faces,

        vert_VV_num_broad_phase,
        broad_phase_list,

        narrow_phase_list_indices_vf,
        narrow_phase_list_pair_vf,
        collision_count,
        indirect_command_buffer,
    
        vert_VV_num_narrow_phase,
        vert_VV_prefix_narrow_phase,
        collision_pair_offset_in_vert,
        vert_adj_verts_vv,

        max_vf_broad_phase_num,
        max_vf_narrow_phase_num,
        thickness_1,
        thickness_2,
        stiffness_collision
        );
}


//
// Scan and fill-in
//
kernel void narrow_phase_scan_collision_pair(
    Pointer(uint) vert_VV_num_narrow_phase,
    Pointer(uint) vert_VV_prefix_narrow_phase,
    Pointer(uint) collision_count,
    Pointer(Int4) self_collision_indirect_cmd_buffer,
    Constant(uint) num_verts_total,
    uint vid [[thread_position_in_grid]],
    threadgroup_ids
)
{
    if (vid == 0)
    {
        const uint num_vf_total = (collision_count[0]);
        self_collision_indirect_cmd_buffer[0] = make_indirect_command_buffer(num_vf_total, 256);
    }

    const bool is_valid = vid < num_verts_total;
    uint num_vf = is_valid ? 
        NarrowPhase::narrow_phase_scan_get_num(vid, vert_VV_num_narrow_phase) : 0;

    // Get Prefix Sum
    const uint blockDim = 256;
    ThreadGroup uint cache_scan[blockDim]; // 1 MB
    cache_scan[tid] = num_vf;
    THREAD_GROUP_SYNC;

    uint temp;
    SCAN(tid, blockDim, cache_scan, temp);	
    THREAD_GROUP_SYNC;
    uint prefix = cache_scan[tid]; // exclusive

    ThreadGroup uint block_prefix_in_global = 0;
    if (tid == blockDim - 1)
    {
        const uint num_vf_in_block = prefix + num_vf;
        block_prefix_in_global = NarrowPhase::fn_atomic_add_num_collision_in_block(collision_count, num_vf_in_block);
    }
    THREAD_GROUP_SYNC;
    
    if (is_valid) vert_VV_prefix_narrow_phase[vid] = block_prefix_in_global + prefix;

    // Or Just Use...
    // if (is_valid) vert_VV_prefix_narrow_phase[vid] = NarrowPhase::fn_atomic_add_num_collision_in_block(collision_count, num_vf);
}
kernel void self_collision_fill_in_vf(
    Pointer(uchar) collision_pair_offset_in_vert, 
	Pointer(Int4) narrow_phase_list_vf,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uint) vert_adj_element,
	Pointer(Int4) self_collision_indirect_cmd_buffer,
    uint pair_idx [[thread_position_in_grid]]
)
{
    const uint num_vf_total = self_collision_indirect_cmd_buffer[0][3];
	if (pair_idx >= num_vf_total) return;

    NarrowPhase::self_collision_fill_in(pair_idx, 
        collision_pair_offset_in_vert, 
        narrow_phase_list_vf, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_element);
}
kernel void self_collision_fill_in_vv(
    Pointer(uchar) collision_pair_offset_in_vert, 
	Pointer(Int2) narrow_phase_list_vv,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uint) vert_adj_element,
	Pointer(uint) collision_count,
    uint pair_idx [[thread_position_in_grid]]
)
{
    const uint num_vf_total = collision_count[0];
	if (pair_idx >= num_vf_total) return;

    NarrowPhase::self_collision_fill_in(pair_idx, 
        collision_pair_offset_in_vert, 
        narrow_phase_list_vv, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_element);
}
kernel void obstacle_collision_fill_in_vf(
    Pointer(uchar) collision_pair_offset_in_vert, 
	Pointer(Int4) narrow_phase_list_vf,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uint) vert_adj_element,
	Pointer(Int4) obstacle_collision_indirect_cmd_buffer,
    uint pair_idx [[thread_position_in_grid]]
)
{
    const uint num_vf_total = obstacle_collision_indirect_cmd_buffer[0][3];
	if (pair_idx >= num_vf_total) return;

    NarrowPhase::obstacle_collision_fill_in(pair_idx, 
        collision_pair_offset_in_vert, 
        narrow_phase_list_vf, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_element);
}

kernel void obstacle_collision_fill_in_vv(
    Pointer(uchar) collision_pair_offset_in_vert, 
	Pointer(Int2) narrow_phase_list_vv,
	Pointer(uint) vert_VV_prefix_narrow_phase,
	Pointer(uint) vert_adj_element,
	Pointer(uint) collision_count,
    uint pair_idx [[thread_position_in_grid]]
)
{
    const uint num_vf_total = collision_count[0];
	if (pair_idx >= num_vf_total) return;

    NarrowPhase::obstacle_collision_fill_in(pair_idx, 
        collision_pair_offset_in_vert, 
        narrow_phase_list_vv, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_element);
}



kernel void reset_collision_constraint(
    Pointer(float) lambda_self_collision,
    Pointer(float) lambda_self_collision_friction,
    Pointer(uint)  self_collision_count,
    uint pair_idx [[thread_position_in_grid]]
)
{
    if (pair_idx < self_collision_count[0])
    {
        lambda_self_collision[pair_idx] = 0.0f;
        lambda_self_collision_friction[pair_idx] = 0.0f;
    }
}


kernel void constraint_self_collision_vv_with_tet(

    Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet, 
	Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet, 

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(ProximityVV) self_collision_pair_vv,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,
    Constant(uint) num_verts_cloth,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    // const uint cluster_prefix = curr_color_prefix[cluster_idx];
    // const uint cluster_size = curr_color_num_elements[cluster_idx];
    // if (i >= cluster_size) return;
    // const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    // const uint vid = cluster[i];
    // const uint vid = cluster_prefix + i;

    // Constrains::solve_self_collision_vv_per_collision_pair_template_with_tet(vid, 
    //     substep_start_position_cloth, 
    //     substep_start_position_tet, 
    //     iter_position_cloth, 
    //     iter_position_tet, 
    //     output_position_cloth, 
    //     output_position_tet, 
        
    //     sa_surface_verts,
    //     sa_vert_mass_inv_cloth, 
    //     sa_vert_mass_inv_tet, 
    //     sa_vert_mutex_cloth,
    //     sa_vert_mutex_tet,

    //     self_collision_pair_vv,
    //     lambda_self_collision,
    //     lambda_self_collision_friction,

    //     substep_dt, 
    //     use_atomic,
    //     thickness,
    //     stiffness_collision,
    //     stiffness_friction,
    //     num_verts_cloth);
}
kernel void constraint_self_collision_vv_cloth(

    Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet, 
	Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet, 

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(ProximityVV) self_collision_pair_vv,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,
    Constant(uint) num_verts_cloth,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    const uint cluster_prefix = curr_color_prefix[cluster_idx];
    const uint cluster_size = curr_color_num_elements[cluster_idx];
    if (i >= cluster_size) return;
    // const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    // const uint vid = cluster[i];
    const uint vid = cluster_prefix + i;

    Constrains::solve_self_collision_vv_per_collision_pair_template_cloth(vid, 
        substep_start_position_cloth, 
        substep_start_position_tet, 
        iter_position_cloth, 
        iter_position_tet, 
        output_position_cloth, 
        output_position_tet, 
        
        sa_surface_verts,
        sa_vert_mass_inv_cloth, 
        sa_vert_mass_inv_tet, 
        sa_vert_mutex_cloth,
        sa_vert_mutex_tet,

        self_collision_pair_vv,
        lambda_self_collision,
        lambda_self_collision_friction,

        substep_dt, 
        use_atomic,
        thickness,
        stiffness_collision,
        stiffness_friction,
        num_verts_cloth);
}
kernel void constraint_self_collision_vv_tet(

    Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet, 
	Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet, 

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(ProximityVV) self_collision_pair_vv,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,
    Constant(uint) num_verts_cloth,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    const uint cluster_prefix = curr_color_prefix[cluster_idx];
    const uint cluster_size = curr_color_num_elements[cluster_idx];
    if (i >= cluster_size) return;
    // const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    // const uint vid = cluster[i];
    const uint vid = cluster_prefix + i;

    Constrains::solve_self_collision_vv_per_collision_pair_template_tet(vid, 
        substep_start_position_cloth, 
        substep_start_position_tet, 
        iter_position_cloth, 
        iter_position_tet, 
        output_position_cloth, 
        output_position_tet, 
        
        sa_surface_verts,
        sa_vert_mass_inv_cloth, 
        sa_vert_mass_inv_tet, 
        sa_vert_mutex_cloth,
        sa_vert_mutex_tet,

        self_collision_pair_vv,
        lambda_self_collision,
        lambda_self_collision_friction,

        substep_dt, 
        use_atomic,
        thickness,
        stiffness_collision,
        stiffness_friction,
        num_verts_cloth);
}
kernel void constraint_self_collision_vv_cross(

    Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet, 
	Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet, 

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(ProximityVV) self_collision_pair_vv,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,
    Constant(uint) num_verts_cloth,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    const uint cluster_prefix = curr_color_prefix[cluster_idx];
    const uint cluster_size = curr_color_num_elements[cluster_idx];
    if (i >= cluster_size) return;
    // const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    // const uint vid = cluster[i];
    const uint vid = cluster_prefix + i;

    Constrains::solve_self_collision_vv_per_collision_pair_template_cross(vid, 
        substep_start_position_cloth, 
        substep_start_position_tet, 
        iter_position_cloth, 
        iter_position_tet, 
        output_position_cloth, 
        output_position_tet, 
        
        sa_surface_verts,
        sa_vert_mass_inv_cloth, 
        sa_vert_mass_inv_tet, 
        sa_vert_mutex_cloth,
        sa_vert_mutex_tet,

        self_collision_pair_vv,
        lambda_self_collision,
        lambda_self_collision_friction,

        substep_dt, 
        use_atomic,
        thickness,
        stiffness_collision,
        stiffness_friction,
        num_verts_cloth);
}
kernel void constraint_self_collision_vv(

    Pointer(Float3) substep_start_position, 
	Pointer(Float3) input_position, 
	Pointer(Float3) output_position, 

    Pointer(float) sa_vert_mass_inv, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex,

	Pointer(ProximityVV) self_collision_pair_vv,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    const uint cluster_prefix = curr_color_prefix[cluster_idx];
    const uint cluster_size = curr_color_num_elements[cluster_idx];
    if (i >= cluster_size) return;
    const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    const uint vid = cluster[i];

    Constrains::solve_self_collision_vv_per_collision_pair_template(vid, 
            substep_start_position, input_position, output_position,
            sa_vert_mass_inv, sa_vert_mutex, 
            self_collision_pair_vv, 
            lambda_self_collision, lambda_self_collision_friction,
            substep_dt, use_atomic, thickness, 
            stiffness_collision, stiffness_friction);
}
kernel void constraint_self_collision_vf(

    Pointer(Float3) substep_start_position, 
	Pointer(Float3) input_position, 
	Pointer(Float3) output_position, 

    Pointer(float) sa_vert_mass_inv, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex,

	Pointer(ProximityVF) self_collision_pair_vf,
	Pointer(float) lambda_self_collision,
	Pointer(float) lambda_self_collision_friction,

    Constant(float) substep_dt, 
    Constant(bool ) use_atomic,
    Constant(float) thickness,
    Constant(float) stiffness_collision,
    Constant(float) stiffness_friction,

    Pointer(uint) clusterd_constraint_self_collision_vv,
    Pointer(uint) curr_color_prefix,
    Pointer(uint) curr_color_num_elements,
    Constant(uint) cluster_idx,

    uint i [[thread_position_in_grid]])
{
    const uint cluster_prefix = curr_color_prefix[cluster_idx];
    const uint cluster_size = curr_color_num_elements[cluster_idx];
    if (i >= cluster_size) return;
    const Pointer(uint) cluster = &clusterd_constraint_self_collision_vv[cluster_prefix];
    const uint vid = cluster[i];

    Constrains::solve_self_collision_vf_per_collision_pair_template(vid, 
            substep_start_position, input_position, output_position,
            sa_vert_mass_inv, sa_vert_mutex, 
            self_collision_pair_vf, 
            lambda_self_collision, lambda_self_collision_friction,
            substep_dt, use_atomic, thickness, 
            stiffness_collision, stiffness_friction);
}



kernel void constraint_obstacle_collision_vv_with_tet(

    Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) sa_obstacle_start_position,
    Pointer(Float3) sa_obstacle_velocity,

	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet,

	Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet,

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVV) narrow_phase_list_vv,

	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

	Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,
	Constant(uint ) num_verts_cloth,

    uint vid [[thread_position_in_grid]])
{
    // Constrains::solve_obstacle_collision_vv_template_with_tet(vid, 
    //     iter_position_cloth, 
    //     iter_position_tet, 
    //     sa_obstacle_start_position,
    //     sa_obstacle_velocity,

    //     output_position_cloth, 
    //     output_position_tet,

    //     substep_start_position_cloth, 
    //     substep_start_position_tet,

    //     sa_surface_verts, 
    //     sa_vert_mass_inv_cloth, 
    //     sa_vert_mass_inv_tet, 
    //     sa_vert_mutex_cloth,
    //     sa_vert_mutex_tet,

    //     vert_VV_num_narrow_phase, 
    //     vert_VV_prefix_narrow_phase, 
    //     vert_adj_elements,
    //     narrow_phase_list_vv,

    //     lambda_obstacle_collision,
    //     lambda_obstacle_collision_friction,

    //     max_vv_per_vert_narrow_obstacle_collision,
    //     thickness, 
    //     substep_dt,
    //     stiffness_collision,
    //     stiffness_friction,
    //     num_verts_cloth);
}
kernel void constraint_obstacle_collision_vv_cloth(

    Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) sa_obstacle_start_position,
    Pointer(Float3) sa_obstacle_velocity,

	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet,

	Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet,

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVV) narrow_phase_list_vv,

	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

	Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,
	Constant(uint ) num_verts_cloth,

    uint vid [[thread_position_in_grid]])
{
    // Constrains::solve_obstacle_collision_vv_template_cloth(vid, 
    //     iter_position_cloth, 
    //     iter_position_tet, 
    //     sa_obstacle_start_position,
    //     sa_obstacle_velocity,

    //     output_position_cloth, 
    //     output_position_tet,

    //     substep_start_position_cloth, 
    //     substep_start_position_tet,

    //     sa_surface_verts, 
    //     sa_vert_mass_inv_cloth, 
    //     sa_vert_mass_inv_tet, 
    //     sa_vert_mutex_cloth,
    //     sa_vert_mutex_tet,

    //     vert_VV_num_narrow_phase, 
    //     vert_VV_prefix_narrow_phase, 
    //     vert_adj_elements,
    //     narrow_phase_list_vv,

    //     lambda_obstacle_collision,
    //     lambda_obstacle_collision_friction,

    //     max_vv_per_vert_narrow_obstacle_collision,
    //     thickness, 
    //     substep_dt,
    //     stiffness_collision,
    //     stiffness_friction,
    //     num_verts_cloth);
}
kernel void constraint_obstacle_collision_vv_tet(

    Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) sa_obstacle_start_position,
    Pointer(Float3) sa_obstacle_velocity,

	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet,

	Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet,

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVV) narrow_phase_list_vv,

	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

	Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,
	Constant(uint ) num_verts_cloth,

    uint surface_id [[thread_position_in_grid]])
{
    // Constrains::solve_obstacle_collision_vv_template_tet(surface_id, 
    //     iter_position_cloth, 
    //     iter_position_tet, 
    //     sa_obstacle_start_position,
    //     sa_obstacle_velocity,

    //     output_position_cloth, 
    //     output_position_tet,

    //     substep_start_position_cloth, 
    //     substep_start_position_tet,

    //     sa_surface_verts, 
    //     sa_vert_mass_inv_cloth, 
    //     sa_vert_mass_inv_tet, 
    //     sa_vert_mutex_cloth,
    //     sa_vert_mutex_tet,

    //     vert_VV_num_narrow_phase, 
    //     vert_VV_prefix_narrow_phase, 
    //     vert_adj_elements,
    //     narrow_phase_list_vv,

    //     lambda_obstacle_collision,
    //     lambda_obstacle_collision_friction,

    //     max_vv_per_vert_narrow_obstacle_collision,
    //     thickness, 
    //     substep_dt,
    //     stiffness_collision,
    //     stiffness_friction,
    //     num_verts_cloth);
}


kernel void constraint_obstacle_collision_vf_cloth(

    Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) sa_obstacle_start_position,
    Pointer(Float3) sa_obstacle_velocity,

	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet,

	Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet,

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVF) narrow_phase_list_vf,

	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

	Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,
	Constant(uint ) num_verts_cloth,

    uint vid [[thread_position_in_grid]])
{
    Constrains::solve_obstacle_collision_vf_template_cloth(vid, 
        iter_position_cloth, 
        iter_position_tet, 
        sa_obstacle_start_position,
        sa_obstacle_velocity,

        output_position_cloth, 
        output_position_tet,

        substep_start_position_cloth, 
        substep_start_position_tet,

        sa_surface_verts, 
        sa_vert_mass_inv_cloth, 
        sa_vert_mass_inv_tet, 
        sa_vert_mutex_cloth,
        sa_vert_mutex_tet,

        vert_VV_num_narrow_phase, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_elements,
        narrow_phase_list_vf,

        lambda_obstacle_collision,
        lambda_obstacle_collision_friction,

        max_vv_per_vert_narrow_obstacle_collision,
        thickness, 
        substep_dt,
        stiffness_collision,
        stiffness_friction,
        num_verts_cloth);
}
kernel void constraint_obstacle_collision_vf_tet(

    Pointer(Float3) iter_position_cloth, 
	Pointer(Float3) iter_position_tet, 
	Pointer(Float3) sa_obstacle_start_position,
    Pointer(Float3) sa_obstacle_velocity,

	Pointer(Float3) output_position_cloth, 
	Pointer(Float3) output_position_tet,

	Pointer(Float3) substep_start_position_cloth, 
	Pointer(Float3) substep_start_position_tet,

	Pointer(uint) sa_surface_verts, 
	Pointer(float) sa_vert_mass_inv_cloth, 
	Pointer(float) sa_vert_mass_inv_tet, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex_cloth,
	Pointer(ATOMIC_FLAG) sa_vert_mutex_tet,

	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVF) narrow_phase_list_vf,

	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

	Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,
	Constant(uint ) num_verts_cloth,

    uint surface_id [[thread_position_in_grid]])
{
    Constrains::solve_obstacle_collision_vf_template_tet(surface_id, 
        iter_position_cloth, 
        iter_position_tet, 
        sa_obstacle_start_position,
        sa_obstacle_velocity,

        output_position_cloth, 
        output_position_tet,

        substep_start_position_cloth, 
        substep_start_position_tet,

        sa_surface_verts, 
        sa_vert_mass_inv_cloth, 
        sa_vert_mass_inv_tet, 
        sa_vert_mutex_cloth,
        sa_vert_mutex_tet,

        vert_VV_num_narrow_phase, 
        vert_VV_prefix_narrow_phase, 
        vert_adj_elements,
        narrow_phase_list_vf,

        lambda_obstacle_collision,
        lambda_obstacle_collision_friction,

        max_vv_per_vert_narrow_obstacle_collision,
        thickness, 
        substep_dt,
        stiffness_collision,
        stiffness_friction,
        num_verts_cloth);
}

kernel void constraint_obstacle_collision_vv(

    Pointer(Float3) substep_start_position, 
	Pointer(Float3) input_position, 
	Pointer(Float3) output_position, 
	Pointer(Float3) sa_obstacle_start_position, 
	Pointer(Float3) sa_obstacle_vert_normal,
	Pointer(float) sa_vert_mass_inv, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex, 
    
	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVV) narrow_phase_list_vv,
	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

    Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,

    uint vid [[thread_position_in_grid]])
{
    // Constrains::solve_obstacle_collision_vv_template(vid, 
    //     substep_start_position, 
    //     input_position, 
    //     output_position, 
    //     sa_obstacle_start_position, 
    //     sa_obstacle_vert_normal,
    //     sa_vert_mass_inv, 
    //     sa_vert_mutex, 
    //     vert_VV_num_narrow_phase, 
    //     vert_VV_prefix_narrow_phase, 
    //     vert_adj_elements,
    //     narrow_phase_list_vv,
    //     lambda_obstacle_collision,
    //     lambda_obstacle_collision_friction,

    //     max_vv_per_vert_narrow_obstacle_collision,
    //     thickness, 
    //     substep_dt,
    //     stiffness_collision,
    //     stiffness_friction);
}
kernel void constraint_obstacle_collision_vf(

    Pointer(Float3) substep_start_position, 
	Pointer(Float3) input_position, 
	Pointer(Float3) output_position, 
	Pointer(Float3) sa_obstacle_start_position, 
	Pointer(Float3) sa_obstacle_vert_normal,
	Pointer(float) sa_vert_mass_inv, 
	Pointer(ATOMIC_FLAG) sa_vert_mutex, 
    
	Pointer(uint) vert_VV_num_narrow_phase, 
	Pointer(uint) vert_VV_prefix_narrow_phase, 
	Pointer(uint) vert_adj_elements,
	Pointer(ProximityVF) narrow_phase_list_vf,
	Pointer(float) lambda_obstacle_collision,
	Pointer(float) lambda_obstacle_collision_friction,

    Constant(uint ) max_vv_per_vert_narrow_obstacle_collision,
	Constant(float) thickness, 
	Constant(float) substep_dt,
	Constant(float) stiffness_collision,
	Constant(float) stiffness_friction,

    uint vid [[thread_position_in_grid]])
{
    // Constrains::solve_obstacle_collision_vf_template(vid, 
    //     substep_start_position, 
    //     input_position, 
    //     output_position, 
    //     sa_obstacle_start_position, 
    //     sa_obstacle_vert_normal,
    //     sa_vert_mass_inv, 
    //     sa_vert_mutex, 
    //     vert_VV_num_narrow_phase, 
    //     vert_VV_prefix_narrow_phase, 
    //     vert_adj_elements,
    //     narrow_phase_list_vf,
    //     lambda_obstacle_collision,
    //     lambda_obstacle_collision_friction,

    //     max_vv_per_vert_narrow_obstacle_collision,
    //     thickness, 
    //     substep_dt,
    //     stiffness_collision,
    //     stiffness_friction);
}

