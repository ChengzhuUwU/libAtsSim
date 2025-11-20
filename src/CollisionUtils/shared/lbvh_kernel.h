#pragma once

#include "sim_data.h"
#include "collision_data.h"
#include "energy_utils.h"
#include "lbvh_args.h"
#include "lbvh_utils.h"
#include <aabb.h>

namespace LBVH {

namespace Construct {

inline AABB kernel_compute_vert_aabb_and_center(const uint vid, Constant(LbvhArgs) bvh, Pointer(Float3) start_position) {
    Float3 vert_pos = start_position[vid];
    bvh.sa_leaf_center[vid] = vert_pos;

    AABB aabb = AABB(vert_pos) + SimContactEnergy::_eps;
    return aabb;
}

inline AABB kernel_compute_face_aabb_and_center(const uint fid, Constant(LbvhArgs) bvh, Pointer(Int3) input_face, Pointer(Float3) start_position) {
    Int3 face = input_face[fid];
    Float3 face_pos[3] = {start_position[face[0]], start_position[face[1]], start_position[face[2]]};
    bvh.sa_leaf_center[fid] = average_array<Float3, 3>(face_pos);

    AABB aabb = AABB(face_pos[0], face_pos[1], face_pos[2]) + SimContactEnergy::_eps;
    // bvh.sa_leaf_aabb[fid] = aabb;
    return aabb;
}

inline AABB kernel_compute_edge_aabb_and_center(const uint eid, Constant(LbvhArgs) bvh, Pointer(Int2) input_edge, Pointer(Float3) start_position) {
    Int2 edge = input_edge[eid];
    Float3 edge_pos[2] = {start_position[edge[0]], start_position[edge[1]]};
    bvh.sa_leaf_center[eid] = average_array<Float3, 2>(edge_pos);

    AABB aabb = AABB(edge_pos[0], edge_pos[1]) + SimContactEnergy::_eps;
    // bvh.sa_leaf_aabb[eid] = aabb;
    return aabb;
}

inline void save_aabb(const uint blockIdx, Pointer(AABB) sa_block_aabb, ConstRef(AABB) aabb) {
    sa_block_aabb[blockIdx] = aabb;
}

inline void compute_global_aabb_additional_operation(Constant(LbvhArgs) bvh, ConstRef(AABB) global_aabb) {
    Float3 global_min = global_aabb.min_pos;
    Float3 global_width_inv = global_aabb.range_inv();

    bvh.sa_block_aabb[0].min_pos = global_min;
    bvh.sa_block_aabb[0].max_pos = global_width_inv;
}

inline void kernel_compute_morton(const uint lid, Constant(LbvhArgs) bvh) {
    AABB global_aabb = bvh.sa_block_aabb[0];
    Float3 global_min = global_aabb.min_pos;
    Float3 global_width_inv = global_aabb.max_pos;

    Float3 orig_pos = bvh.sa_leaf_center[lid];
    Float3 norm_position = (orig_pos - global_min) * global_width_inv;

    Morton32 mc32 = Morton32(norm_position);
    Morton mc64;
    mc64.data = ((uint64(mc32.data) << 32ul) | lid);
    bvh.sa_morton[lid] = mc64;

    bvh.sa_sorted_get_original[lid] = lid;
}

inline void kernel_init_tree(const uint nid, Constant(LbvhArgs) bvh) {
    if (nid == 0) {
        bvh.sa_is_healthy[0] = true;
        bvh.sa_parrent[0] = -1u;
    }
    bvh.sa_object_idx[nid] = -1u;// = ~0
}

//
// Then Sort
//

inline void kernel_apply_sorted_morton(const uint lid, Constant(LbvhArgs) bvh) {
    uint orig_index = bvh.sa_sorted_get_original[lid];
    bvh.sa_morton_sorted[lid] = bvh.sa_morton[orig_index];
    bvh.sa_children[bvh.num_inner_nodes + lid] = make<Int2>(orig_index);
}

inline void kernel_construct_tree(const uint nid, const uint num_innder_nodes, Constant(LbvhArgs) bvh) {
    uint num_leaves = num_innder_nodes + 1;

    Int2 range;
    if (nid == 0)
        range = make<Int2>(0u, num_innder_nodes);
    else
        range = determineRange(bvh.sa_morton_sorted, nid, num_leaves);
    uint i = range.x;
    uint j = range.y;

    uint split = findSplit(bvh.sa_morton_sorted, range);

    //
    //	Output child pointers
    //	if min(i, j) = split	    then left = Leave[split].self	     else   left = Interior[split].self
    //	if max(i, j) = split + 1    then left = Leave[split + 1].self    else   left = Interior[split + 1].self
    //	Interior[i].child = (left, right)
    //

    uint child_left = min_scalar(i, j) == split ? num_innder_nodes + split : split;
    uint child_right = max_scalar(i, j) == split + 1 ? num_innder_nodes + split + 1 : split + 1;

    if (child_right >= num_innder_nodes) swap_scalar(child_left, child_right);
    // if (child_right >= num_innder_nodes) printf("out of range!!!\n");
    bvh.sa_parrent[child_left] = nid;
    bvh.sa_parrent[child_right] = nid;
    bvh.sa_children[nid] = make<Int2>(child_left, child_right);
}

inline bool kernel_check_healthy(const uint nid, Constant(LbvhArgs) bvh) {
    bool is_construct_healthy = true;

    auto child = bvh.sa_children[nid];
    uint parrent_of_left = bvh.sa_parrent[child[0]];
    uint parrent_of_right = bvh.sa_parrent[child[1]];

    if (parrent_of_left != nid || parrent_of_right != nid) {
        is_construct_healthy = false;
        bvh.sa_is_healthy[0] = false;
    }

    return is_construct_healthy;
}

}// namespace Construct

namespace Refit {

inline void kernel_update_vert_aabb(const uint lid, Constant(LbvhArgs) bvh, Pointer(Float3) start_position, const float thickness) {
    const uint vid = bvh.sa_sorted_get_original[lid];
    Float3 vert_pos = start_position[vid];
    AABB aabb = AABB(vert_pos) + thickness;
    bvh.sa_node_aabb[bvh.num_inner_nodes + lid] = aabb;
}

inline void kernel_update_face_aabb(const uint lid, Constant(LbvhArgs) bvh, Pointer(Int3) input_face, Pointer(Float3) start_position, const float thickness) {
    const uint fid = bvh.sa_sorted_get_original[lid];
    Int3 face = input_face[fid];
    Float3 face_pos[3] = {start_position[face[0]], start_position[face[1]], start_position[face[2]]};
    AABB aabb = AABB(face_pos[0], face_pos[1], face_pos[2]) + thickness;
    bvh.sa_node_aabb[bvh.num_inner_nodes + lid] = aabb;
}

inline void kernel_update_edge_aabb(const uint lid, Constant(LbvhArgs) bvh, Pointer(Int2) input_edge, Pointer(Float3) start_position, const float thickness) {
    const uint eid = bvh.sa_sorted_get_original[lid];
    Int2 edge = input_edge[eid];
    Float3 edge_pos[2] = {start_position[edge[0]], start_position[edge[1]]};
    AABB aabb = AABB(edge_pos[0], edge_pos[1]) + thickness;
    bvh.sa_node_aabb[bvh.num_inner_nodes + lid] = aabb;
}

inline void kernel_apply_leaves_aabb(const uint lid, Constant(LbvhArgs) bvh) {

    if (!bvh.sa_is_healthy[0]) return;

    uint current = lid + bvh.num_inner_nodes;
    uint parrent = bvh.sa_parrent[current];

    uint loop = 0;
    // From Leaves to Root
    while (parrent != -1u) {// bvh.sa_parrent[0] = -1u;

        if (loop++ > 10000) {
            bvh.sa_is_healthy[0] = false;
            break;
        }
        // THREAD_FENCE;

        uint orig_flag = 0;

        // orig_flag = atomic_cas(bvh.sa_apply_flag[parrent], 0u, 1u); // Or AtomicAdd
        orig_flag = atomic_cas(bvh.sa_apply_flag[parrent], 0u, current);// Or AtomicAdd

        if (orig_flag == 0u) {
            return;
        }
        // else if (orig_flag == 1u){
        else if (orig_flag != -1u) {

            // orig_flag = atomic_cas(bvh.sa_apply_flag[parrent], 1u, 2u);
            uint brother = atomic_cas(bvh.sa_apply_flag[parrent], orig_flag, -1u);

            // if(!atomic_load<bool>(bvh.sa_node_mutex[child_of_parrent.x])
            // || !atomic_load<bool>(bvh.sa_node_mutex[child_of_parrent.y]))
            // {
            //     atomic_store(bvh.sa_apply_flag[parrent], 1u);
            //     continue;
            // }

            AABB aabb_left = bvh.sa_node_aabb[brother];
            AABB aabb_right = bvh.sa_node_aabb[current];
            bvh.sa_node_aabb[parrent] = aabb_left + aabb_right;

            // auto child_of_parrent = bvh.sa_children[parrent];
            // AABB aabb_left =  bvh.sa_node_aabb[child_of_parrent.x];
            // AABB aabb_right = bvh.sa_node_aabb[child_of_parrent.y];
            // bvh.sa_node_aabb[parrent] = aabb_left + aabb_right;

            // atomic_aabb(sa_node_aabb_atomic, parrent, tmp);
            // volatile AABB aabb_left = sa_node_aabb[child_of_parrent.x];
            // volatile AABB aabb_right = sa_node_aabb[child_of_parrent.y];
            // volatile AABB tmp = AABB(aabb_left) + AABB(aabb_right);
            // sa_node_aabb[parrent].min_pos = tmp.min_pos;
            // sa_node_aabb[parrent].max_pos = tmp.max_pos;
            // atomic_store(bvh.sa_node_mutex[parrent], true);

            current = parrent;
            parrent = bvh.sa_parrent[current];

        } else {
            bvh.sa_is_healthy[0] = false;
            break;
        }

        // THREAD_GROUP_SYNC;
    }
}

}// namespace Refit

namespace Query {

inline void kernel_query_from_vert_atomic(const uint vid, Constant(LbvhArgs) bvh, const bool is_self_collision, Pointer(Float3) start_position, Pointer(uint) broad_phase_list, Pointer(Int4) indirect_command_buffer, const float query_thickness, const uint max_broad_phase_count) {
    if (!bvh.sa_is_healthy[0]) return;

    const auto vertPos = start_position[vid];
    const AABB query_box = AABB(vertPos) + query_thickness;
    traversal_tree_and_find_overlap_atomic<AABB, true>(bvh, is_self_collision, broad_phase_list, indirect_command_buffer, query_box, vid, max_broad_phase_count);
}

}// namespace Query

}// namespace LBVH