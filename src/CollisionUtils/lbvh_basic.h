#pragma once

#include "launcher.h"
#include "shared/lbvh_args.h"
#include "shared_array.h"
#include "float_n.h"
#include "float_n_n.h"
#include "scene_params.h"
#include "utils.h"
#include "collision_data.h"

// template<LBVHTreeType tree_type, LBVHUpdateType update_type>
class LBVHInterface {
protected:
    LbvhArgs lbvh;

public:
    virtual void init_cloth_lbvh(LbvhData &bvh, CollisionList &list) = 0;
    virtual void init_obstacle_lbvh(LbvhData &bvh, CollisionList &list) = 0;
    // virtual void register_implementation(Launcher::Scheduler& scheduler) = 0;

private:
    virtual void check_healthy() = 0;

public:
    virtual void compute_morton() = 0;
    virtual void compute_vert_aabb_and_center(const SharedArray<Float3> &start_position) = 0;
    virtual void compute_face_aabb_and_center(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position) = 0;
    virtual void compute_edge_aabb_and_center(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position) = 0;
    virtual void construct_tree() = 0;

    virtual void update_vert_aabb(const SharedArray<Float3> &start_position, const float thickness) = 0;
    virtual void update_vert_aabb(const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) = 0;
    virtual void update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const float thickness) = 0;
    virtual void update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) = 0;
    virtual void update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const float thickness) = 0;
    virtual void update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) = 0;
    virtual void apply_leaves_aabb() = 0;
    virtual void apply_leaves_aabb_affine_body() = 0;

    // query
    virtual void query_from_vert_atomic(const SharedArray<Float3> &start_position, SharedArray<uint> &broad_phase_list, SharedArray<Int4> &indirect_command_buffer, const uint offset, const float query_thickness) = 0;

public:
    virtual void init_tree() = 0;
    virtual void sort_by_morton() = 0;
    virtual void apply_sorted_morton() = 0;
    virtual void construct_tree_Karras2012() = 0;

public:
    virtual bool is_tree_healthy() = 0;
};
