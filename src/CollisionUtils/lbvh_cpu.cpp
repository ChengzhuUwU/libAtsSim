#include "lbvh_cpu.h"
#include "struct_to_string.h"
#include "shared/lbvh_utils.h"
#include "shared/lbvh_kernel.h"

#define for_leaves for_loop(index, lbvh.num_leaves)
#define for_inner_nodes for_loop(index, lbvh.num_innder_nodes)
#define for_nodes for_loop(index, lbvh.num_nodes)

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::init_cloth_lbvh(LbvhData &bvh) {

    lbvh.set<PtrTypeCpu>(bvh);
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::init_obstacle_lbvh(LbvhData &bvh) {

    lbvh.set<PtrTypeCpu>(bvh);
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::compute_vert_aabb_and_center(const SharedArray<Float3> &start_position) {

    AABB global_aabb = parallel_for_and_reduce_sum<AABB>(0, start_position.size(), [&](uint vid) {
        return LBVH::Construct::kernel_compute_vert_aabb_and_center(vid, lbvh, start_position.ptr());
    });
    LBVH::Construct::compute_global_aabb_additional_operation(lbvh, global_aabb);
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::compute_face_aabb_and_center(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position) {

    AABB global_aabb = parallel_for_and_reduce_sum<AABB>(0, input_face.size(), [&](uint fid) {
        return LBVH::Construct::kernel_compute_face_aabb_and_center(fid, lbvh, input_face.ptr(), start_position.ptr());
    });
    LBVH::Construct::compute_global_aabb_additional_operation(lbvh, global_aabb);
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::compute_edge_aabb_and_center(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position) {

    AABB global_aabb = parallel_for_and_reduce_sum<AABB>(0, input_edge.size(), [&](uint eid) {
        return LBVH::Construct::kernel_compute_edge_aabb_and_center(eid, lbvh, input_edge.ptr(), start_position.ptr());
    });
    LBVH::Construct::compute_global_aabb_additional_operation(lbvh, global_aabb);
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::compute_morton() {

    parallel_for(0, lbvh.num_leaves, [&](uint lid) {
        LBVH::Construct::kernel_compute_morton(lid, lbvh);
    });
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::construct_tree() {

    compute_morton();

    uint num_leaves = lbvh.num_leaves;
    uint num_innder_nodes = lbvh.num_inner_nodes;
    uint num_nodes = lbvh.num_nodes;

    init_tree();

    sort_by_morton();

    apply_sorted_morton();

    construct_tree_Karras2012();

    check_healthy();
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::init_tree() {
    parallel_for(0, lbvh.num_nodes, [&](uint nid) {
        LBVH::Construct::kernel_init_tree(nid, lbvh);
    });
}
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::sort_by_morton() {
    parallel_sort(lbvh.sa_sorted_get_original, lbvh.sa_sorted_get_original + lbvh.num_leaves, [&](uint idx1, uint idx2) -> bool {
        return lbvh.sa_morton[idx1] < lbvh.sa_morton[idx2];
    });

    // for (uint lid = 0; lid < lbvh.num_leaves - 1; lid++)
    // {
    //     const uint orig_vid = lbvh.sa_sorted_get_original[lid];
    //     const uint next_vid = lbvh.sa_sorted_get_original[lid + 1];
    //     if (lbvh.sa_morton[orig_vid].data == lbvh.sa_morton[next_vid].data)
    //     {
    //         fast_format_err("LBVH With The Same Morton : {} & {}", lbvh.sa_morton[orig_vid].data, lbvh.sa_morton[next_vid].data);
    //     }
    // }
}
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::apply_sorted_morton() {
    parallel_for(0, lbvh.num_leaves, [&](uint lid) {
        LBVH::Construct::kernel_apply_sorted_morton(lid, lbvh);
    });
}
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::construct_tree_Karras2012() {
    parallel_for(0, lbvh.num_inner_nodes, [&](uint nid) {
        LBVH::Construct::kernel_construct_tree(nid, lbvh.num_inner_nodes, lbvh);
    });
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::check_healthy() {

    bool is_construct_healthy = true;

    parallel_for_in_block(0, lbvh.num_inner_nodes, 256, [&](uint start, uint end) {
        // bool is_block_healthy = true;
        for (uint i = start; i < end; i++) {
            bool thread_result = LBVH::Construct::kernel_check_healthy(i, lbvh);
            if (thread_result == false) {
                is_construct_healthy = false;

                Int2 child = lbvh.sa_children[i];
                uint parrent_of_left = lbvh.sa_parrent[child[0]];
                uint parrent_of_right = lbvh.sa_parrent[child[1]];
                // printf("Build Tree Filad : Node's %d Child is : [%d & %d] , Their Parrents is [%d & %d]",
                //     i, child.x, child.y, parrent_of_left, parrent_of_right);
                // break;
            }
        }
    });

    if (!is_construct_healthy) {
        fast_format_err("Build Tree Failed : {} ", lbvh.update_type == LBVHUpdateTypeObstacle ? "Obstacle Tree" : "Cloth Tree");
        fast_print_err(lbvh.tree_type == LBVHTreeTypeVert ? "vert" : lbvh.tree_type == LBVHTreeTypeFace ? "face" :
                                                                                                          "edge");
        fast_print_err(lbvh.update_type == LBVHUpdateTypeCloth ? "cloth" : "obs");
    } else {
        // lbvh.sa_is_healthy[0] = true;
        // fast_print("Build Tree success!");
    }
}

TEMPLATE_LBVH_CPU
bool TEMPLATE_LBVH_CPU_NAME::is_tree_healthy() {
    if (!lbvh.sa_is_healthy[0]) {
        std::cerr << "LBVH Tree is not healthy!!!" << std::endl;
    }
    return lbvh.sa_is_healthy[0];
}

// DCD VV
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_vert_aabb(const SharedArray<Float3> &start_position, const float thickness) {

    parallel_for(0, start_position.size(), [&](uint vid) {
        LBVH::Refit::kernel_update_vert_aabb(vid, lbvh, start_position.ptr(), thickness);
    });
}

// DCD VF
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const float thickness) {

    parallel_for(0, input_face.size(), [&](uint fid) {
        LBVH::Refit::kernel_update_face_aabb(fid, lbvh, input_face.ptr(), start_position.ptr(), thickness);
    });
}

// DCD EE
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const float thickness) {

    parallel_for(0, input_edge.size(), [&](uint eid) {
        LBVH::Refit::kernel_update_edge_aabb(eid, lbvh, input_edge.ptr(), start_position.ptr(), thickness);
    });
}

// CCD VV
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_vert_aabb(const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
    parallel_for(0, start_position.size(), [&](uint lid) {
        const uint vid = lbvh.sa_sorted_get_original[lid];
        Float3 start_pos = start_position[vid];
        Float3 end_pos = next_position[vid];
        AABB aabb = AABB(start_pos, end_pos);
        lbvh.sa_node_aabb[lbvh.num_inner_nodes + lid] = aabb;
    });
}
// CCD VF
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
    parallel_for(0, input_face.size(), [&](uint lid) {
        const uint fid = lbvh.sa_sorted_get_original[lid];
        Int3 face = input_face[fid];
        AABB aabb = AABB(start_position[face[0]], start_position[face[1]], start_position[face[2]],
                         next_position[face[0]], next_position[face[1]], next_position[face[2]]) +
                    SimContactEnergy::_ccd_eps;
        lbvh.sa_node_aabb[lbvh.num_inner_nodes + lid] = aabb;
    });
}
// CCD EE
TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
    if (lbvh.tree_type == LBVHTreeTypeFace) {
    }
    parallel_for(0, input_edge.size(), [&](uint lid) {
        const uint eid = lbvh.sa_sorted_get_original[lid];
        Int2 edge = input_edge[eid];
        AABB aabb = AABB(start_position[edge[0]], start_position[edge[1]], next_position[edge[0]], next_position[edge[1]]) + SimContactEnergy::_ccd_eps;
        lbvh.sa_node_aabb[lbvh.num_inner_nodes + lid] = aabb;
    });
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::apply_leaves_aabb() {

    std::memset(lbvh.sa_apply_flag, 0, lbvh.num_nodes * sizeof(ATOMIC_UINT));
    // lbvh.sa_apply_flag.set_zero();

    parallel_for(0, lbvh.num_leaves, [&](uint lid) {
        LBVH::Refit::kernel_apply_leaves_aabb(lid, lbvh);
    });

    if (
        is_nan_vec(lbvh.sa_node_aabb[0].min_pos) || is_nan_vec(lbvh.sa_node_aabb[0].max_pos) ||
        is_inf_vec(lbvh.sa_node_aabb[0].min_pos) || is_inf_vec(lbvh.sa_node_aabb[0].max_pos)) {
        fast_print_err("LBVH AABB is NaN");
        exit(0);
    }

    // fast_print("global_aabb", SimString::AABB_to_string(lbvh.sa_node_aabb[0]));
    // fast_print("global_aabb", SimString::AABB_to_string(lbvh.sa_leaf_aabb.get_sum()));
    // AABB global1;
    // for (uint vid = 0; vid < lbvh.num_verts_total; vid++) {
    //     global1 += lbvh.sa_start_position[vid];
    // }
    // AABB global2 = lbvh.sa_node_aabb[0];
    // fast_print("actual", SimString::AABB_to_string(global1));
    // fast_print("actual", SimString::AABB_to_string(global2));
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::apply_leaves_aabb_affine_body() {

    /// To be implement

    // parallel_for({0, lbvh.num_nodes}, [&](uint nid){

    //     const uint obs_idx = lbvh.sa_node_object_id[nid];
    //     const uint fid = lbvh.sa_sorted_get_original[nid];
    //     const AABB model_space_aabb = lbvh.sa_node_aabb_model_position[nid];

    //     if(obs_idx != 0xFF){
    //         const Float4x4 model_matrix = lbvh.sa_model_matrix[obs_idx];
    //         AABB world_space_aabb;
    //         world_space_aabb.min_pos = makeFloat3(model_matrix * makeFloat4(model_space_aabb.min_pos));
    //         world_space_aabb.max_pos = makeFloat3(model_matrix * makeFloat4(model_space_aabb.max_pos));
    //         lbvh.sa_node_aabb[nid] = world_space_aabb;
    //     }
    //     else{
    //         // if(nid == 0){
    //         Int2 children = lbvh.sa_children[nid];
    //         uchar id_x = lbvh.sa_node_object_id[children.x];
    //         uchar id_y = lbvh.sa_node_object_id[children.y];
    //         if((id_x == id_y) && (id_x != 0xFF) && (id_y != 0xFF)){

    //         }
    //         // printf("can not find objidx : %d", nid);
    //         // }
    //     }

    // });
}

TEMPLATE_LBVH_CPU
void TEMPLATE_LBVH_CPU_NAME::query_from_vert_atomic(const SharedArray<Float3> &start_position, SharedArray<uint> &broad_phase_list, SharedArray<Int4> &indirect_command_buffer, const uint offset, const float query_thickness) {

    const uint num_vert_total = start_position.size();
    const uint max_broad_phase_count = broad_phase_list.size() / 2;
    const bool is_self_collision = lbvh.update_type == LBVHUpdateTypeCloth;

    // Stackless Traversal
    parallel_for(0, num_vert_total, [&](uint vid) {
        LBVH::Query::kernel_query_from_vert_atomic(vid,
                                                   lbvh, is_self_collision, start_position.ptr(), broad_phase_list.ptr(), indirect_command_buffer.ptr() + offset, query_thickness, max_broad_phase_count);
    });

    const uint num_collision = indirect_command_buffer[offset][3];
    indirect_command_buffer[offset] = make_indirect_command_buffer(num_collision);
}
