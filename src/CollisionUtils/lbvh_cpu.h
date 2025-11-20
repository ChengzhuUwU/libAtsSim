#pragma once

#include "shared/lbvh_args.h"
#include "lbvh_basic.h"
#include "shared_array.h"
#include "float_n.h"
#include "float_n_n.h"
#include "scene_params.h"
#include "utils.h"
#include "collision_data.h"

// #define TEMPLATE_LBVH_CPU template<LBVHTreeType tree_type, LBVHUpdateType update_type>
// #define TEMPLATE_LBVH_CPU_NAME LbvhCpu<tree_type, update_type>
#define TEMPLATE_LBVH_CPU
#define TEMPLATE_LBVH_CPU_NAME LbvhCpu

TEMPLATE_LBVH_CPU
class LbvhCpu : public LBVHInterface {
    // private:
    //     bool is_healthy = false;
    //     LbvhArgs lbvh;
    //     uint num_leaves;
    //     uint num_innder_nodes;
    //     uint num_nodes;

public:
    void init_cloth_lbvh(LbvhData &bvh, CollisionList &list) override;
    void init_obstacle_lbvh(LbvhData &bvh, CollisionList &list) override;

public:
    void check_healthy() override;

public:
    void compute_morton() override;

    void compute_vert_aabb_and_center(const SharedArray<Float3> &start_position) override;
    void compute_face_aabb_and_center(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position) override;
    void compute_edge_aabb_and_center(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position) override;
    void construct_tree() override;

    void update_vert_aabb(const SharedArray<Float3> &start_position, const float thickness) override;
    void update_vert_aabb(const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) override;
    void update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const float thickness) override;
    void update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) override;
    void update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const float thickness) override;
    void update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) override;
    void apply_leaves_aabb() override;
    void apply_leaves_aabb_affine_body() override;

    // query
    void query_from_vert_atomic(const SharedArray<Float3> &start_position, SharedArray<uint> &broad_phase_list, SharedArray<Int4> &indirect_command_buffer, const uint offset, const float query_thickness) override;

public:
    void init_tree() override;
    void sort_by_morton() override;
    void apply_sorted_morton() override;
    void construct_tree_Karras2012() override;

public:
    bool is_tree_healthy() override;
};
