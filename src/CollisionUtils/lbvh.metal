#define METAL_CODE
#define SIM_USE_SIMD true

#include <metal_stdlib>
using namespace metal;
#include "../../SharedDefine/float_n.h"
#include "../../SharedDefine/float_n_n.h"
#include "../../SharedDefine/gpu_algorism.h"
#include "shared/lbvh_kernel.h"

kernel void empty_task(
){
    
}

kernel void compute_vert_aabb_and_center(
    Constant(LbvhArgs) bvh,
    Pointer(Float3) start_position,
    GPU_PREFIX
    uint vid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids
){
    bool is_valid = vid < bvh.num_leaves;
    AABB aabb = is_valid ? LBVH::Construct::kernel_compute_vert_aabb_and_center(vid, bvh, start_position) : AABB();
    reduce_aabb(aabb, tid, 256);
    if(tid == 0){
        LBVH::Construct::save_aabb(bid, bvh.sa_block_aabb, aabb);
    }
}

kernel void compute_face_aabb_and_center(
    Constant(LbvhArgs) bvh,
    Pointer(Int3) input_face,
    Pointer(Float3) start_position,
    GPU_PREFIX
    uint fid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids
){
    bool is_valid = fid < bvh.num_leaves;
    AABB aabb = is_valid ? LBVH::Construct::kernel_compute_face_aabb_and_center(fid, bvh, input_face, start_position) : AABB();
    reduce_aabb(aabb, tid, 256);
    if(tid == 0){
        LBVH::Construct::save_aabb(bid, bvh.sa_block_aabb, aabb);
    }
}

kernel void compute_edge_aabb_and_center(
    Constant(LbvhArgs) bvh,
    Pointer(Int2) input_edge,
    Pointer(Float3) start_position,
    GPU_PREFIX
    uint eid [[thread_position_in_grid]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup_ids
){
    bool is_valid = eid < bvh.num_leaves;
    AABB aabb = is_valid ? LBVH::Construct::kernel_compute_edge_aabb_and_center(eid, bvh, input_edge, start_position) : AABB();
    reduce_aabb(aabb, tid, 256);
    if(tid == 0){
        LBVH::Construct::save_aabb(bid, bvh.sa_block_aabb, aabb);
    }
}

kernel void reduce_global_aabb(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint index [[thread_position_in_grid]],
    threadgroup_ids
){
    bool is_valid = index < (bvh.num_leaves + 256 - 1) / 256;
    AABB aabb = is_valid ? bvh.sa_block_aabb[index] : AABB();
    reduce_aabb(aabb, tid, SECOND_REDUCE_DIM);
    if(index == 0){
        LBVH::Construct::compute_global_aabb_additional_operation(bvh, aabb);
    }
}

kernel void compute_morton(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint lid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_compute_morton(lid, bvh);
}

kernel void init_tree(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_init_tree(nid, bvh);
}

//
// Sort
//

kernel void apply_sorted_morton(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint lid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_apply_sorted_morton(lid, bvh);
}

kernel void construct_tree(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_construct_tree(nid, bvh.num_inner_nodes, bvh);
}

kernel void check_healthy(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]]
){
    // bool is_construct_healthy = LBVH::Construct::kernel_check_healthy(nid, bvh);
    LBVH::Construct::kernel_check_healthy(nid, bvh);
}

kernel void compute_escape_index(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_compute_escape_index(nid, bvh);
}

kernel void compute_left_index(
    Constant(LbvhArgs) bvh,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]]
){
    LBVH::Construct::kernel_compute_left_index(nid, bvh);
}



//
// Refit
//
kernel void update_vert_aabb(
    Constant(LbvhArgs) bvh, 
    Pointer(Float3) start_position,
    Constant(float) thickness,
    GPU_PREFIX
    uint lid [[thread_position_in_grid]]
){
    LBVH::Refit::kernel_update_vert_aabb(lid, bvh, start_position, thickness);
}

kernel void update_face_aabb(
    Constant(LbvhArgs) bvh, 
    Pointer(Int3) input_face, 
    Pointer(Float3) start_position,
    Constant(float) thickness,
    GPU_PREFIX
    uint lid [[thread_position_in_grid]]
){
    LBVH::Refit::kernel_update_face_aabb(lid, bvh, input_face, start_position, thickness);
}


kernel void update_edge_aabb(
    Constant(LbvhArgs) bvh, 
    Pointer(Int2) input_edge, 
    Pointer(Float3) start_position,
    Constant(float) thickness,
    GPU_PREFIX
    uint lid [[thread_position_in_grid]]
){
    LBVH::Refit::kernel_update_edge_aabb(lid, bvh, input_edge, start_position, thickness);
}

// bvh.sa_apply_flag.set_zero();

kernel void apply_leaves_aabb(
    Constant(LbvhArgs) bvh, 
    GPU_PREFIX
    uint lid [[thread_position_in_grid]])
{
    LBVH::Refit::kernel_apply_leaves_aabb(lid, bvh);
}

kernel void reset_apply_flag(
    Pointer(uint) sa_apply_flag,
    GPU_PREFIX
    uint nid [[thread_position_in_grid]])
{
    sa_apply_flag[nid] = 0;
}

kernel void query_from_vert_atomic(
    Constant(LbvhArgs) bvh, 
    Constant(bool) is_self_collision,
    Pointer(Float3) start_position, Pointer(uint) broad_phase_list, Pointer(Int4) indirect_command_buffer,
    Constant(float) query_range, 
    Constant(uint) max_broad_phase_count, 
    GPU_PREFIX
    uint vid [[thread_position_in_grid]]
){
    LBVH::Query::kernel_query_from_vert_atomic(vid, bvh, is_self_collision, start_position, broad_phase_list, indirect_command_buffer, query_range, max_broad_phase_count);
}
kernel void make_broadphase_indirect_command_buffer(
    Pointer(Int4) indirect_command_buffer,
    uint vid [[thread_position_in_grid]])
{
    if (vid == 0)
    {
        const uint num_collision = indirect_command_buffer[0][3];
        indirect_command_buffer[0] = make_indirect_command_buffer(num_collision);
    }
}