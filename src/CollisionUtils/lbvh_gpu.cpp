#include "lbvh_gpu.h"
#include "shared/lbvh_kernel.h"
#include "command_list.h"
#include "gpu_algorism.h"
#include "gpu_function.h"
#include "struct_to_string.h"

#define for_leaves for_loop(index, lbvh->num_leaves)
#define for_inner_nodes for_loop(index, lbvh->num_innder_nodes)
#define for_nodes for_loop(index, lbvh->num_nodes)

#if __APPLE__
// MTL::Library* library_lbvh = nullptr;
#endif

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::load_functions() {
#if __APPLE__
    NS::Error *err;

    std::string full_path = std::string(SELF_RESOURCES_PATH) + std::string("/metal_libs/") + std::string("lbvh.metallib");

    static MTL::Library *library_lbvh = nullptr;
    static uint library_entity_count = 0;
    bool is_first_load = library_entity_count == 0;
    library_entity_count++;

    if (is_first_load) {
        library_lbvh = get_device()->newLibrary(std_string_to_ns_string(full_path), &err);
    }
    check_err(library_lbvh, err);

    // functions

    fn_empty_task.load_with_multiple_entity(library_lbvh, "empty_task", is_first_load);

    fn_compute_vert_aabb_and_center.load_with_multiple_entity(library_lbvh, "compute_vert_aabb_and_center", is_first_load);
    fn_compute_face_aabb_and_center.load_with_multiple_entity(library_lbvh, "compute_face_aabb_and_center", is_first_load);
    fn_compute_edge_aabb_and_center.load_with_multiple_entity(library_lbvh, "compute_edge_aabb_and_center", is_first_load);
    fn_reduce_global_aabb.load_with_multiple_entity(library_lbvh, "reduce_global_aabb", is_first_load);
    fn_compute_morton.load_with_multiple_entity(library_lbvh, "compute_morton", is_first_load);
    fn_init_tree.load_with_multiple_entity(library_lbvh, "init_tree", is_first_load);

    fn_apply_sorted_morton.load_with_multiple_entity(library_lbvh, "apply_sorted_morton", is_first_load);
    fn_construct_tree.load_with_multiple_entity(library_lbvh, "construct_tree", is_first_load);
    fn_check_healthy.load_with_multiple_entity(library_lbvh, "check_healthy", is_first_load);
    fn_compute_escape_index.load_with_multiple_entity(library_lbvh, "compute_escape_index", is_first_load);
    fn_compute_left_index.load_with_multiple_entity(library_lbvh, "compute_left_index", is_first_load);

    fn_update_vert_aabb.load_with_multiple_entity(library_lbvh, "update_vert_aabb", is_first_load);
    fn_update_face_aabb.load_with_multiple_entity(library_lbvh, "update_face_aabb", is_first_load);
    fn_update_edge_aabb.load_with_multiple_entity(library_lbvh, "update_edge_aabb", is_first_load);
    fn_reset_apply_flag.load_with_multiple_entity(library_lbvh, "reset_apply_flag", is_first_load);
    fn_apply_leaves_aabb.load_with_multiple_entity(library_lbvh, "apply_leaves_aabb", is_first_load);

    fn_query_from_vert_atomic.load_with_multiple_entity(library_lbvh, "query_from_vert_atomic", is_first_load);
    fn_make_broadphase_indirect_command_buffer.load_with_multiple_entity(library_lbvh, "make_broadphase_indirect_command_buffer", is_first_load);

    // library_lbvh -> release();

#endif
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::init_cloth_lbvh(LbvhData &bvh, CollisionList &list) {

    load_functions();

    lbvh_arr.resize(1);
    lbvh_arr[0].set<PtrTypeGpu>(bvh, list);// GPU T*
    lbvh_cpu.set<PtrTypeCpu>(bvh, list);   // CPU T*
    lbvh.set<PtrTypeMtl>(bvh, list);       // MTL::Buffer()

    set_sivibal();

    // dispatch_leaves = get_dispatch_num(lbvh.num_leaves, 256);
    // excution_leaves_256 = get_excution_threads_256(lbvh.num_leaves);
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::init_obstacle_lbvh(LbvhData &bvh, CollisionList &list) {

    load_functions();

    lbvh_arr.resize(1);
    lbvh_arr[0].set<PtrTypeGpu>(bvh, list);// GPU T*
    lbvh_cpu.set<PtrTypeCpu>(bvh, list);   // CPU T*
    lbvh.set<PtrTypeMtl>(bvh, list);       // MTL::Buffer()

    set_sivibal();
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::set_sivibal() {

    const bool is_obs_tree = lbvh.update_type == LBVHUpdateTypeObstacle;

#if __APPLE__
    /// Construct
    {
        fn_compute_face_aabb_and_center.add_argument_buffer(lbvh_arr);
        fn_compute_face_aabb_and_center.set_buffer_visibal(lbvh.sa_leaf_center, AccessTypeWrite);
        fn_compute_face_aabb_and_center.set_buffer_visibal(lbvh.sa_block_aabb, AccessTypeWrite);

        fn_compute_vert_aabb_and_center.add_argument_buffer(lbvh_arr);
        fn_compute_vert_aabb_and_center.set_buffer_visibal(lbvh.sa_leaf_center, AccessTypeWrite);
        fn_compute_vert_aabb_and_center.set_buffer_visibal(lbvh.sa_block_aabb, AccessTypeWrite);

        fn_compute_edge_aabb_and_center.add_argument_buffer(lbvh_arr);
        fn_compute_edge_aabb_and_center.set_buffer_visibal(lbvh.sa_leaf_center, AccessTypeWrite);
        fn_compute_edge_aabb_and_center.set_buffer_visibal(lbvh.sa_block_aabb, AccessTypeWrite);

        fn_reduce_global_aabb.add_argument_buffer(lbvh_arr);
        fn_reduce_global_aabb.set_buffer_visibal(lbvh.sa_block_aabb, AccessTypeReadWrite);

        fn_compute_morton.add_argument_buffer(lbvh_arr);
        fn_compute_morton.set_buffer_visibal(lbvh.sa_block_aabb, AccessTypeRead);
        fn_compute_morton.set_buffer_visibal(lbvh.sa_leaf_center, AccessTypeRead);
        fn_compute_morton.set_buffer_visibal(lbvh.sa_morton, AccessTypeWrite);
        fn_compute_morton.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeWrite);

        fn_init_tree.add_argument_buffer(lbvh_arr);
        fn_init_tree.set_buffer_visibal(lbvh.sa_is_healthy, AccessTypeWrite);
        fn_init_tree.set_buffer_visibal(lbvh.sa_parrent, AccessTypeWrite);
        fn_init_tree.set_buffer_visibal(lbvh.sa_object_idx, AccessTypeWrite);

        fn_apply_sorted_morton.add_argument_buffer(lbvh_arr);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeRead);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_morton, AccessTypeRead);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_morton_sorted, AccessTypeWrite);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeWrite);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_children, AccessTypeWrite);
        fn_apply_sorted_morton.set_buffer_visibal(lbvh.sa_object_idx, AccessTypeWrite);

        fn_construct_tree.add_argument_buffer(lbvh_arr);
        fn_construct_tree.set_buffer_visibal(lbvh.sa_morton_sorted, AccessTypeRead);
        fn_construct_tree.set_buffer_visibal(lbvh.sa_parrent, AccessTypeWrite);
        fn_construct_tree.set_buffer_visibal(lbvh.sa_children, AccessTypeWrite);

        fn_check_healthy.add_argument_buffer(lbvh_arr);
        fn_check_healthy.set_buffer_visibal(lbvh.sa_children, AccessTypeRead);
        fn_check_healthy.set_buffer_visibal(lbvh.sa_parrent, AccessTypeRead);
        fn_check_healthy.set_buffer_visibal(lbvh.sa_is_healthy, AccessTypeWrite);
    }

    // Refit
    {
        fn_update_vert_aabb.add_argument_buffer(lbvh_arr);
        fn_update_vert_aabb.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeWrite);
        fn_update_vert_aabb.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeRead);

        fn_update_face_aabb.add_argument_buffer(lbvh_arr);
        fn_update_face_aabb.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeWrite);
        fn_update_face_aabb.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeRead);

        fn_update_edge_aabb.add_argument_buffer(lbvh_arr);
        fn_update_edge_aabb.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeWrite);
        fn_update_edge_aabb.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeRead);

        fn_apply_leaves_aabb.add_argument_buffer(lbvh_arr);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_is_healthy, AccessTypeReadWrite);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_sorted_get_original, AccessTypeRead);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_parrent, AccessTypeRead);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_apply_flag, AccessTypeReadWrite);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_children, AccessTypeRead);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeReadWrite);
        fn_apply_leaves_aabb.set_buffer_visibal(lbvh.sa_node_mutex, AccessTypeReadWrite);
    }

    // Query
    {
        fn_query_from_vert_atomic.add_argument_buffer(lbvh_arr);
        fn_query_from_vert_atomic.set_buffer_visibal(lbvh.sa_is_healthy, AccessTypeRead);
        fn_query_from_vert_atomic.set_buffer_visibal(lbvh.sa_node_aabb, AccessTypeRead);
        fn_query_from_vert_atomic.set_buffer_visibal(lbvh.sa_object_idx, AccessTypeRead);
        fn_query_from_vert_atomic.set_buffer_visibal(lbvh.sa_broad_phase_list_vf, AccessTypeReadWrite);
    }

#endif
}

// 构造bvh本身不需要ccd，只需要大致的位置信息
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::compute_vert_aabb_and_center(const SharedArray<Float3> &start_position) {

    get_command_list().add_task(fn_compute_vert_aabb_and_center);
    fn_compute_vert_aabb_and_center.bind_ptr(start_position);
    fn_compute_vert_aabb_and_center.launch_async(get_excution_threads_256(lbvh.num_leaves), 256);

    add_compute_task(fn_reduce_global_aabb, 0).launch_async(SECOND_REDUCE_DIM, SECOND_REDUCE_DIM);
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::compute_face_aabb_and_center(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position) {

    get_command_list().add_task(fn_compute_face_aabb_and_center);
    fn_compute_face_aabb_and_center.bind_ptr(input_face);
    fn_compute_face_aabb_and_center.bind_ptr(start_position);
    fn_compute_face_aabb_and_center.launch_async(get_excution_threads_256(lbvh.num_leaves), 256);

    add_compute_task(fn_reduce_global_aabb, 0).launch_async(SECOND_REDUCE_DIM, SECOND_REDUCE_DIM);
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::compute_edge_aabb_and_center(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position) {

    get_command_list().add_task(fn_compute_edge_aabb_and_center);
    fn_compute_edge_aabb_and_center.bind_ptr(input_edge);
    fn_compute_edge_aabb_and_center.bind_ptr(start_position);
    fn_compute_edge_aabb_and_center.launch_async(get_excution_threads_256(lbvh.num_leaves), 256);

    // get_command_list().send_and_wait();
    // AABB block_aabb;
    // for (uint bid = 0; bid < (lbvh_cpu.num_leaves + 256 - 1) / 256; bid++) {
    //     block_aabb += lbvh_cpu.sa_block_aabb[bid];
    // }
    // block_aabb.max_pos = block_aabb.range_inv();

    add_compute_task(fn_reduce_global_aabb, 0).launch_async(SECOND_REDUCE_DIM, SECOND_REDUCE_DIM);

    // get_command_list().send_and_wait();
    // AABB global_aabb;
    // for (uint eid = 0; eid < lbvh_cpu.num_leaves; eid++) {
    //     global_aabb += lbvh_cpu.sa_leaf_aabb[eid];
    // }
    // global_aabb.max_pos = global_aabb.range_inv();
    // AABB compute_aabb = lbvh_cpu.sa_block_aabb[0];

    // fast_print("true", SimString::AABB_to_string(global_aabb));
    // fast_print("compute", SimString::AABB_to_string(compute_aabb));
    // fast_print("block", SimString::AABB_to_string(block_aabb));
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::compute_morton() {

    add_compute_task(fn_compute_morton, 0).launch_async(lbvh.num_leaves);

    // uint* sa_sorted_get_original = SHARED_ARRAY::get_cpuPtr_from_gpuPtr(lbvh.sa_sorted_get_original);
    // Morton* sa_morton = SHARED_ARRAY::get_cpuPtr_from_gpuPtr(lbvh.sa_morton);
    // AABB* sa_leaf_aabb = SHARED_ARRAY::get_cpuPtr_from_gpuPtr(lbvh.sa_leaf_aabb);
    // AABB* sa_aabb_block = SHARED_ARRAY::get_cpuPtr_from_gpuPtr(lbvh.sa_aabb_block);
    // AABB global1 = parallel_for_and_reduce_sum<AABB>(0, lbvh.num_leaves, [&](uint lid){return sa_leaf_aabb[lid];});
    // AABB global2 = sa_aabb_block[0];
    // fast_print("true", SimString::AABB_to_string(global1));
    // fast_print("compute", SimString::AABB_to_string(global2));
    // for (uint i = 0; i < dispatch_leaves; i++) {
    //     fast_print(i, SimString::AABB_to_string(sa_aabb_block[i]));
    // }
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::construct_tree() {

    compute_morton();

    init_tree();

    sort_by_morton();

    apply_sorted_morton();

    construct_tree_Karras2012();

    // // CheckSort
    // for (uint index = 1; index < lbvh.num_leaves; index++){
    //     if(index < 100 || index > lbvh.num_leaves - 100){
    //         auto morton_self = sa_morton[index];
    //         auto morton_prev = sa_morton[index - 1];
    //         if(morton_self <= morton_prev)
    //             fast_print("sort error", morton_self.data, morton_prev.data);
    //         std::cout << (sa_morton_sorted[index].data) << ", ";
    //     }
    // }

    check_healthy();
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::init_tree() {
    add_compute_task(fn_init_tree, 0).launch_async(lbvh.num_nodes);
}
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::sort_by_morton() {
    get_command_list().send_and_wait();// wait !!!

    parallel_sort(lbvh_cpu.sa_sorted_get_original, lbvh_cpu.sa_sorted_get_original + lbvh.num_leaves, [&](uint idx1, uint idx2) -> bool {
        return lbvh_cpu.sa_morton[idx1] < lbvh_cpu.sa_morton[idx2];
    });
}
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::apply_sorted_morton() {
    add_compute_task(fn_apply_sorted_morton, 0).launch_async(lbvh.num_leaves);
}
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::construct_tree_Karras2012() {
    add_compute_task(fn_construct_tree, 0).launch_async(lbvh.num_inner_nodes);
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::check_healthy() {

    bool is_construct_healthy = false;

    add_compute_task(fn_check_healthy, 0).launch_async(lbvh.num_inner_nodes);

    get_command_list().send_and_wait();

    is_construct_healthy = lbvh_cpu.sa_is_healthy[0];

    if (!is_construct_healthy) {
        fast_format_err("Build Tree Failed : {}", lbvh.update_type == LBVHUpdateTypeObstacle ? "Obstacle Tree" : "Cloth Tree");
    } else {
        // fast_print("Build Tree success!");
    }
}

TEMPLATE_LBVH_GPU
bool TEMPLATE_LBVH_GPU_NAME::is_tree_healthy() {
    // if(!lbvh_cpu.sa_is_healthy[0]){
    //     std::cerr << "LBVH Tree is not healthy!!!" << std::endl;
    // }
    return lbvh_cpu.sa_is_healthy[0];
}

// DCD VV
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_vert_aabb(const SharedArray<Float3> &start_position, const float thickness) {
    get_command_list().add_task(fn_update_vert_aabb);
    fn_update_vert_aabb.bind_ptr(start_position);
    fn_update_vert_aabb.bind_constant(thickness);
    fn_update_vert_aabb.launch_async(lbvh.num_leaves);
}
// DCD VF
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const float thickness) {
    get_command_list().add_task(fn_update_face_aabb);
    fn_update_face_aabb.bind_ptr(input_face);
    fn_update_face_aabb.bind_ptr(start_position);
    fn_update_face_aabb.bind_constant(thickness);
    fn_update_face_aabb.launch_async(lbvh.num_leaves);
}
// DCD EE
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const float thickness) {
    get_command_list().add_task(fn_update_edge_aabb);
    fn_update_edge_aabb.bind_ptr(input_edge);
    fn_update_edge_aabb.bind_ptr(start_position);
    fn_update_edge_aabb.bind_constant(thickness);
    fn_update_edge_aabb.launch_async(lbvh.num_leaves);
}

// CCD VV
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_vert_aabb(const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
}
// CCD VF
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_face_aabb(const SharedArray<Int3> &input_face, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
}
// CCD EE
TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::update_edge_aabb(const SharedArray<Int2> &input_edge, const SharedArray<Float3> &start_position, const SharedArray<Float3> &next_position) {
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::apply_leaves_aabb() {

    // std::memset(lbvh_cpu.sa_apply_flag, 0, lbvh.num_inner_nodes * sizeof(ATOMIC_UINT));
    // std::memset(lbvh_cpu.sa_node_mutex, 0, lbvh.num_nodes * sizeof(FlagType));

    // get_command_list().add_task(fn_reset_apply_flag);
    // fn_reset_apply_flag.bind_ptr(lbvh.sa_apply_flag);
    // fn_reset_apply_flag.launch_async(lbvh.num_innder_nodes);

    // get_command_list().add_reset_task_async(lbvh.sa_apply_flag, lbvh.num_nodes);
    // get_command_list().make_fence(get_fence());

    // add_compute_task(fn_apply_leaves_aabb, 0).launch_async(lbvh.num_leaves);
    // get_command_list().wait_fence(get_fence());

    get_command_list().send_and_wait();
    std::memset(lbvh_cpu.sa_apply_flag, 0, lbvh_cpu.num_nodes * sizeof(ATOMIC_UINT));
    parallel_for(0, lbvh_cpu.num_leaves, [&](uint lid) {
        LBVH::Refit::kernel_apply_leaves_aabb(lid, lbvh_cpu);
    });

    // get_command_list().send_and_wait();
    // AABB global1;
    // for (uint vid = 0; vid < lbvh.num_verts_total; vid++) {
    //     global1 += lbvh_cpu.sa_start_position[vid];
    // }
    // AABB global2 = lbvh_cpu.sa_node_aabb[0];
    // fast_print("actual", SimString::AABB_to_string(global1));
    // fast_print("actual", SimString::AABB_to_string(global2));
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::apply_leaves_aabb_affine_body() {
}

TEMPLATE_LBVH_GPU
void TEMPLATE_LBVH_GPU_NAME::query_from_vert_atomic(const SharedArray<Float3> &start_position, SharedArray<uint> &broad_phase_list, SharedArray<Int4> &indirect_command_buffer, const uint offset, const float query_thickness) {

    const uint num_verts_total = start_position.size();
    const uint max_broad_phase_count = broad_phase_list.size() / 2;
    const bool is_self_collision = lbvh.update_type == LBVHUpdateTypeCloth;

    get_command_list().add_task(fn_query_from_vert_atomic);
    fn_query_from_vert_atomic.bind_constant(is_self_collision);
    // fn_query_from_vert_atomic.bind_constant(num_verts_total);
    fn_query_from_vert_atomic.bind_ptr(start_position);
    fn_query_from_vert_atomic.bind_ptr(broad_phase_list);
    fn_query_from_vert_atomic.bind_ptr(indirect_command_buffer, offset);
    fn_query_from_vert_atomic.bind_constant(query_thickness);
    fn_query_from_vert_atomic.bind_constant(max_broad_phase_count);
    fn_query_from_vert_atomic.launch_async(num_verts_total);

    get_command_list().add_task(fn_make_broadphase_indirect_command_buffer);
    fn_make_broadphase_indirect_command_buffer.bind_ptr(indirect_command_buffer, offset);
    fn_make_broadphase_indirect_command_buffer.launch_async(1);
}
