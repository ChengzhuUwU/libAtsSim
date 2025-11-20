#pragma once

#include "aabb.h"
#include "address_space.h"
#include "collision_data.h"
#include "make_arguments.h"
#include "morton.h"

struct LbvhArgs {

    Pointer(Float3)
        sa_leaf_center;
    Pointer(AABB)
        sa_block_aabb;
    Pointer(Morton)
        sa_morton;
    Pointer(Morton)
        sa_morton_sorted;
    Pointer(uint)
        sa_sorted_get_original;

    Pointer(uint)
        sa_parrent;
    Pointer(Int2)
        sa_children;
    Pointer(uint)
        sa_object_idx;
    Pointer(AABB)
        sa_node_aabb;
    Pointer(bool)
        sa_is_healthy;
    Pointer(ATOMIC_UINT)
        sa_apply_flag;
    Pointer(FlagType)
        sa_node_mutex;

    Pointer(uint)
        sa_broad_phase_list_vf;
    Pointer(uint)
        sa_broad_phase_list_ee;

    uint num_leaves;
    uint num_nodes;
    uint num_inner_nodes;
    LBVHTreeType tree_type;
    LBVHUpdateType update_type;

#ifndef METAL_CODE
    template<PtrType ptr_type>
    void set(LbvhData &data, CollisionList &list) {

        tree_type = data.tree_type;
        update_type = LBVHUpdateTypeCloth;

        num_leaves = data.num_leaves;
        num_inner_nodes = data.num_inner_nodes;
        num_nodes = data.num_nodes;

        sa_leaf_center = get_ptr(data.sa_leaf_center, ptr_type);
        sa_block_aabb = get_ptr(data.sa_block_aabb, ptr_type);
        sa_morton = get_ptr(data.sa_morton, ptr_type);
        sa_morton_sorted = get_ptr(data.sa_morton_sorted, ptr_type);
        sa_sorted_get_original = get_ptr(data.sa_sorted_get_original, ptr_type);
        sa_parrent = get_ptr(data.sa_parrent, ptr_type);
        sa_children = get_ptr(data.sa_children, ptr_type);
        sa_object_idx = get_ptr(data.sa_object_idx, ptr_type);
        sa_node_aabb = get_ptr(data.sa_node_aabb, ptr_type);
        sa_apply_flag = get_ptr(data.sa_apply_flag, ptr_type);
        sa_node_mutex = get_ptr(data.sa_node_mutex, ptr_type);

        sa_is_healthy = get_ptr(data.sa_is_healthy, ptr_type);

        sa_broad_phase_list_vf = get_ptr(list.broad.list_vf, ptr_type);
        sa_broad_phase_list_ee = get_ptr(list.broad.list_ee, ptr_type);
    }
#endif
};