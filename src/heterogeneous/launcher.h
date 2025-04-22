#pragma once

// #include "reduce_task.h"

#include "float_n.h"
#include "scene_params.h"
#include <bits_utils.h>
#include <functional>
#include <iostream>
#include <string>
#include <vector>


namespace Launcher{

enum FunctionID{

    id_additional_root,
    id_additional_terminal,

    id_task_heft_2002,

    /// XPBD
    id_xpbd_constraint_stretch_mass_spring,
    id_xpbd_constraint_stretch_fem,
    id_xpbd_constraint_bending,
    id_xpbd_constraint_stress,
    id_xpbd_constraint_inner_stress,
    id_xpbd_constraint_outer_stress,

    id_xpbd_constraint_stretch_mass_spring_half,
    id_xpbd_constraint_stretch_fem_half,
    id_xpbd_constraint_bending_half,
    id_xpbd_constraint_stress_half,
    id_xpbd_constraint_inner_stress_half,
    id_xpbd_constraint_outer_stress_half,
    id_xpbd_constraint_self_collision_vv_addition_half,
    id_xpbd_constraint_self_collision_vv_half,
    id_xpbd_constraint_self_collision_vv_half_cloth,
    id_xpbd_constraint_self_collision_vv_half_tet,
    id_xpbd_constraint_self_collision_vv_half_cross,
    id_xpbd_constraint_self_collision_vf_half,

    id_xpbd_constraint_self_collision_vv_addition,
    id_xpbd_constraint_self_collision_vv,
    id_xpbd_constraint_self_collision_vv_cloth,
    id_xpbd_constraint_self_collision_vv_tet,
    id_xpbd_constraint_self_collision_vv_cross,
    id_xpbd_constraint_self_collision_vf,
    id_xpbd_constraint_obstacle_collision_vv,
    id_xpbd_constraint_obstacle_collision_vv_cloth,
    id_xpbd_constraint_obstacle_collision_vv_tet,
    id_xpbd_constraint_obstacle_collision_vf,

    id_xpbd_constraint_stretch_prev,
    id_xpbd_constraint_stretch_post,
    id_xpbd_constraint_bending_prev,
    id_xpbd_constraint_bending_post,
    id_xpbd_constraint_stress_prev,
    id_xpbd_constraint_stress_post,
    id_xpbd_constraint_self_collision_prev,
    id_xpbd_constraint_self_collision_post,

    id_xpbd_constraint_ground_collision,
    id_xpbd_constraint_ground_collision_cloth,
    id_xpbd_constraint_ground_collision_tet,
    id_xpbd_constraint_ground_collision_stretch,
    id_xpbd_constraint_ground_collision_bending,
    id_xpbd_constraint_ground_collision_self_collision,
    id_xpbd_constraint_last_node,

    // VBD Evaluate Force & Kernel Part
    id_vbd_evaluate_inertia,
    id_vbd_evaluate_stretch_mass_spring,
    id_vbd_evaluate_stretch_fem,
    id_vbd_evaluate_bending,
    id_vbd_evaluate_ground_collision,
    id_vbd_evaluate_obstacle_collision,
    id_vbd_evaluate_self_collision,
    id_vbd_step,
    id_vbd_all_in_one,

    // SOD 
    id_sod_evaluate_energy_gauss_seidel,
    id_sod_evaluate_energy_hybrid,
    id_sod_stretch_stencil_gauss_seidel,
    id_sod_stretch_stencil_hybrid,
    id_sod_init,
    id_sod_step,
    id_sod_all_in_one,

    // Core
    id_xpbd_predict_position,
    id_xpbd_reset_constrains,
    id_xpbd_reset_collision_constrains,
    id_xpbd_store_previous_position,
    id_xpbd_update_velocity,
    id_xpbd_copy_current_position_to_2_constraints,
    id_xpbd_copy_current_position_to_3_constraints,
    id_xpbd_assemble_result_from_2_constraints,
    id_xpbd_assemble_result_from_3_constraints,
    
    id_xpbd_copy_current_position_to_2_devices,
    id_xpbd_copy_current_position_to_constraints,
    id_xpbd_assemble_result_from_constraints,

    id_xpbd_assemble_result_from_stretch_bending,
    id_xpbd_assemble_result_from_stretch_collision,
    id_xpbd_assemble_result_from_bending_collision,
    id_xpbd_assemble_result_from_stress_collision,

    id_xpbd_copy_to_stretch,
    id_xpbd_copy_to_bending,
    id_xpbd_copy_to_self_collision_cloth,
    id_xpbd_copy_to_self_collision_tet,
    id_xpbd_copy_to_sdf_collision_cloth,
    id_xpbd_copy_to_sdf_collision_tet,
    id_xpbd_copy_to_stress,
    id_xpbd_get_from_stretch,
    id_xpbd_get_from_bending,
    id_xpbd_get_from_self_collision_cloth,
    id_xpbd_get_from_self_collision_tet,
    id_xpbd_get_from_sdf_collision_cloth,
    id_xpbd_get_from_sdf_collision_tet,
    id_xpbd_get_from_stress,

    id_xpbd_read_and_solve_conflict,
    id_xpbd_copy_to_cpu_gpu,
    id_xpbd_get_from_cpu_gpu,
    id_xpbd_apply_constraint_delta_from_2_delta,
    id_xpbd_sync_position_to_another_device,
    id_xpbd_update_cloth_position_from_2_devices,
    id_xpbd_update_tet_position_from_2_devices,


    // Collision Detection Part
    id_prepare_collision_detection_position,
    id_update_global_aabb,
    id_self_collision_detection,
    id_self_collision_spatial_hashing_update,
    id_self_collision_reset_collision_system_and_spatial_hashing,
    id_self_collision_reset_collision_system_and_lbvh,
    id_self_collision_fill_in_hash_table,
    id_self_collision_scan_hash_table,
    id_self_collision_insert_vert_into_hash_table,
    id_self_collision_spatial_hashing_query,
    id_self_collision_narrow_phase_vv,
    id_self_collision_narrow_phase_vf,
    id_self_collision_scan_and_fill_in_vv,
    id_self_collision_scan_and_fill_in_vf,

    id_obstacle_collision_detection,
    id_obstacle_collision_spatial_hashing_update,
    id_obstacle_collision_reset_collision_system_and_spatial_hashing,
    id_obstacle_collision_reset_collision_system_and_lbvh,
    id_obstacle_collision_fill_in_hash_table,
    id_obstacle_collision_scan_hash_table,
    id_obstacle_collision_insert_vert_into_hash_table,
    id_obstacle_collision_spatial_hashing_query,
    id_obstacle_collision_narrow_phase_vv,
    id_obstacle_collision_narrow_phase_vf,
    id_obstacle_collision_scan_and_fill_in_vv,
    id_obstacle_collision_scan_and_fill_in_vf,

    // XPBD-LBVH
    id_self_collision_compute_vert_aabb_and_center,
    id_self_collision_compute_face_aabb_and_center,
    id_self_collision_compute_morton,
    id_self_collision_init_tree,
    id_self_collision_sort_by_morton,
    id_self_collision_apply_sorted_morton,
    id_self_collision_construct_tree_karras2012,
    id_self_collision_check_healthy,
    id_self_collision_compute_escape_index,
    id_self_collision_compute_left_index,
    id_self_collision_update_vert_aabb_dcd,
    id_self_collision_update_face_aabb_dcd,
    id_self_collision_apply_leaves_aabb,
    id_self_collision_reset_lbvh,
    id_self_collision_query_from_cloth_vert,

    id_obstacle_collision_compute_vert_aabb_and_center,
    id_obstacle_collision_compute_face_aabb_and_center,
    id_obstacle_collision_compute_morton,
    id_obstacle_collision_init_tree,
    id_obstacle_collision_sort_by_morton,
    id_obstacle_collision_apply_sorted_morton,
    id_obstacle_collision_construct_tree_karras2012,
    id_obstacle_collision_check_healthy,
    id_obstacle_collision_compute_escape_index,
    id_obstacle_collision_compute_left_index,
    id_obstacle_collision_update_vert_aabb_dcd,
    id_obstacle_collision_update_face_aabb_dcd,
    id_obstacle_collision_apply_leaves_aabb,
    id_obstacle_collision_reset_lbvh,
    id_obstacle_collision_query_from_cloth_vert,

    // GraphColoring
    id_graph_coloring_vivace_vv,
    id_graph_coloring_vivace_vv_cloth,
    id_graph_coloring_vivace_vv_tet,
    id_graph_coloring_vivace_vv_cross,
    id_graph_coloring_vivace_vv_wait_for_num_color,
    id_graph_coloring_vivace_vf,
    id_reduce_degree_and_set_max_color_from_max_degree_vv,
    id_reduce_degree_and_set_max_color_from_max_degree_vf,
    id_scan_uncolored_set,
    id_init_palette,
    id_random_coloring_loops_vv,
    id_random_coloring_loops_vf,
    id_tentative_coloring,
    id_conflict_resolution_vv,
    id_conflict_resolution_vf,
    id_feed_hungry_and_make_indirect_buffer,
    id_put_rest_vertices_into_additional_color,
    id_put_rest_vertices_into_random_color,
    id_make_cluster_indirect_cmd_buffer,

    id_xpbd_non,

    TaskID_count
};

// extern std::unordered_map<FunctionID, std::string> taskNames;
const std::unordered_map<FunctionID, std::string> taskNames = {

    {id_xpbd_constraint_stretch_mass_spring, "id_xpbd_constraint_stretch_mass_spring"},
    {id_xpbd_constraint_stretch_fem, "id_xpbd_constraint_stretch_fem"},
    {id_xpbd_constraint_bending, "id_xpbd_constraint_bending"},
    {id_xpbd_constraint_stress, "id_xpbd_constraint_stress"},
    {id_xpbd_constraint_inner_stress, "id_xpbd_constraint_inner_stress"},
    {id_xpbd_constraint_outer_stress, "id_xpbd_constraint_outer_stress"},
    
    {id_xpbd_constraint_self_collision_vv_addition, "id_xpbd_constraint_self_collision_vv_addition"},
    {id_xpbd_constraint_self_collision_vv, "id_xpbd_constraint_self_collision_vv"},
    {id_xpbd_constraint_self_collision_vv_cloth, "id_xpbd_constraint_self_collision_vv_cloth"},
    {id_xpbd_constraint_self_collision_vv_tet, "id_xpbd_constraint_self_collision_vv_tet"},
    {id_xpbd_constraint_self_collision_vv_cross, "id_xpbd_constraint_self_collision_vv_cross"},
    {id_xpbd_constraint_self_collision_vf, "id_xpbd_constraint_self_collision_vf"},
    {id_xpbd_constraint_obstacle_collision_vv, "id_xpbd_constraint_obstacle_collision_vv"},
    {id_xpbd_constraint_obstacle_collision_vv_cloth, "id_xpbd_constraint_obstacle_collision_vv_cloth"},
    {id_xpbd_constraint_obstacle_collision_vv_tet, "id_xpbd_constraint_obstacle_collision_vv_tet"},
    {id_xpbd_constraint_obstacle_collision_vf, "id_xpbd_constraint_obstacle_collision_vf"},

    {id_xpbd_constraint_stretch_mass_spring_half, "id_xpbd_constraint_stretch_mass_spring_half" },
    {id_xpbd_constraint_stretch_fem_half, "id_xpbd_constraint_stretch_fem_half" },
    {id_xpbd_constraint_bending_half, "id_xpbd_constraint_bending_half" },
    {id_xpbd_constraint_stress_half, "id_xpbd_constraint_stress_half" },
    {id_xpbd_constraint_inner_stress_half, "id_xpbd_constraint_inner_stress_half" },
    {id_xpbd_constraint_outer_stress_half, "id_xpbd_constraint_outer_stress_half" },
    {id_xpbd_constraint_self_collision_vv_addition_half, "id_xpbd_constraint_self_collision_vv_addition_half" },
    {id_xpbd_constraint_self_collision_vv_half, "id_xpbd_constraint_self_collision_vv_half" },
    {id_xpbd_constraint_self_collision_vv_half_cloth, "id_xpbd_constraint_self_collision_vv_half_cloth" },
    {id_xpbd_constraint_self_collision_vv_half_tet, "id_xpbd_constraint_self_collision_vv_half_tet" },
    {id_xpbd_constraint_self_collision_vv_half_cross, "id_xpbd_constraint_self_collision_vv_half_cross" },
    {id_xpbd_constraint_self_collision_vf_half, "id_xpbd_constraint_self_collision_vf_half" },

    {id_xpbd_constraint_stretch_prev, "id_xpbd_constraint_stretch_prev"},
    {id_xpbd_constraint_stretch_post, "id_xpbd_constraint_stretch_post"},
    {id_xpbd_constraint_bending_prev, "id_xpbd_constraint_bending_prev"},
    {id_xpbd_constraint_bending_post, "id_xpbd_constraint_bending_post"},
    {id_xpbd_constraint_stress_prev, "id_xpbd_constraint_stress_prev"},
    {id_xpbd_constraint_stress_post, "id_xpbd_constraint_stress_post"},
    {id_xpbd_constraint_self_collision_prev, "id_xpbd_constraint_self_collision_prev"},
    {id_xpbd_constraint_self_collision_post, "id_xpbd_constraint_self_collision_post"},
    {id_xpbd_constraint_last_node, "id_xpbd_constraint_last_node"},


    {id_vbd_evaluate_inertia, "id_vbd_evaluate_inertia"},
    {id_vbd_evaluate_stretch_mass_spring, "id_vbd_evaluate_stretch_mass_spring"},
    {id_vbd_evaluate_stretch_fem, "id_vbd_evaluate_stretch_fem"},
    {id_vbd_evaluate_bending, "id_vbd_evaluate_bending"},
    {id_vbd_evaluate_ground_collision, "id_vbd_evaluate_ground_collision"},
    {id_vbd_evaluate_obstacle_collision, "id_vbd_evaluate_obstacle_collision"},
    {id_vbd_evaluate_self_collision, "id_vbd_evaluate_self_collision"},
    {id_vbd_step, "id_vbd_step"},
    {id_vbd_all_in_one, "id_vbd_all_in_one"},


    {id_sod_evaluate_energy_gauss_seidel, "id_sod_evaluate_energy_gauss_seidel"},
    {id_sod_evaluate_energy_hybrid, "id_sod_evaluate_energy_hybrid"},
    {id_sod_stretch_stencil_gauss_seidel, "id_sod_stretch_stencil_gauss_seidel"},
    {id_sod_stretch_stencil_hybrid, "id_sod_stretch_stencil_hybrid"},
    {id_sod_init, "id_sod_init"},
    {id_sod_step, "id_sod_step"},
    {id_sod_all_in_one, "id_sod_all_in_one"},
    

    {id_xpbd_predict_position, "id_xpbd_predict_position"},
    {id_xpbd_reset_constrains, "id_xpbd_reset_constrains"},
    {id_xpbd_reset_collision_constrains, "id_xpbd_reset_collision_constrains"},
    {id_xpbd_store_previous_position, "id_xpbd_store_previous_position"},
    {id_xpbd_update_velocity, "id_xpbd_update_velocity"},

    {id_xpbd_copy_current_position_to_2_constraints, "id_xpbd_copy_current_position_to_2_constraints"},
    {id_xpbd_copy_current_position_to_3_constraints, "id_xpbd_copy_current_position_to_3_constraints"},
    {id_xpbd_assemble_result_from_2_constraints, "id_xpbd_assemble_result_from_2_constraints"},
    {id_xpbd_assemble_result_from_3_constraints, "id_xpbd_assemble_result_from_3_constraints"},

    {id_xpbd_copy_current_position_to_2_devices, "id_xpbd_copy_current_position_to_2_devices"},
    {id_xpbd_copy_current_position_to_constraints, "id_xpbd_copy_current_position_to_constraints"},
    {id_xpbd_assemble_result_from_constraints, "id_xpbd_assemble_result_from_constraints"},

    {id_xpbd_assemble_result_from_stretch_bending, "id_xpbd_assemble_result_from_stretch_bending"},
    {id_xpbd_assemble_result_from_stretch_collision, "id_xpbd_assemble_result_from_stretch_collision"},
    {id_xpbd_assemble_result_from_bending_collision, "id_xpbd_assemble_result_from_bending_collision"},
    {id_xpbd_assemble_result_from_stress_collision, "id_xpbd_assemble_result_from_stress_collision"},

    {id_xpbd_copy_to_stretch, "id_xpbd_copy_to_stretch"},
    {id_xpbd_copy_to_bending, "id_xpbd_copy_to_bending"},
    {id_xpbd_copy_to_self_collision_cloth, "id_xpbd_copy_to_self_collision_cloth"},
    {id_xpbd_copy_to_self_collision_tet, "id_xpbd_copy_to_self_collision_tet"},
    {id_xpbd_copy_to_sdf_collision_cloth, "id_xpbd_copy_to_sdf_collision_cloth"},
    {id_xpbd_copy_to_sdf_collision_tet, "id_xpbd_copy_to_sdf_collision_tet"},
    {id_xpbd_copy_to_stress, "id_xpbd_copy_to_stress"},

    {id_xpbd_get_from_stretch, "id_xpbd_get_from_stretch"},
    {id_xpbd_get_from_bending, "id_xpbd_get_from_bending"},
    {id_xpbd_get_from_self_collision_cloth, "id_xpbd_get_from_self_collision_cloth"},
    {id_xpbd_get_from_self_collision_tet, "id_xpbd_get_from_self_collision_tet"},
    {id_xpbd_get_from_sdf_collision_cloth, "id_xpbd_get_from_sdf_collision_cloth"},
    {id_xpbd_get_from_sdf_collision_tet, "id_xpbd_get_from_sdf_collision_tet"},
    {id_xpbd_get_from_stress, "id_xpbd_get_from_stress"},
    {id_xpbd_get_from_cpu_gpu, "id_xpbd_get_from_cpu_gpu"},
    {id_xpbd_copy_to_cpu_gpu, "id_xpbd_copy_to_cpu_gpu"},

    {id_xpbd_sync_position_to_another_device, "id_xpbd_sync_position_to_another_device"},
    {id_xpbd_read_and_solve_conflict, "id_xpbd_read_and_solve_conflict"},
    {id_xpbd_update_cloth_position_from_2_devices, "id_xpbd_update_cloth_position_from_2_devices"},
    {id_xpbd_update_tet_position_from_2_devices, "id_xpbd_update_tet_position_from_2_devices"},
    {id_xpbd_apply_constraint_delta_from_2_delta, "id_xpbd_apply_constraint_delta_from_2_delta"},

    {id_xpbd_constraint_ground_collision, "id_xpbd_constraint_ground_collision"},
    {id_xpbd_constraint_ground_collision_cloth, "id_xpbd_constraint_ground_collision_cloth"},
    {id_xpbd_constraint_ground_collision_tet, "id_xpbd_constraint_ground_collision_tet"},
    {id_xpbd_constraint_ground_collision_stretch, "id_xpbd_constraint_ground_collision_stretch"},
    {id_xpbd_constraint_ground_collision_bending, "id_xpbd_constraint_ground_collision_bending"},
    {id_xpbd_constraint_ground_collision_self_collision, "id_xpbd_constraint_ground_collision_self_collision"},

    {id_prepare_collision_detection_position, "id_prepare_collision_detection_position"},
    {id_update_global_aabb, "id_update_global_aabb"},
    {id_self_collision_detection, "id_self_collision_detection"},
    {id_self_collision_spatial_hashing_update, "id_self_collision_spatial_hashing_update"},
    {id_graph_coloring_vivace_vv, "id_graph_coloring_vivace_vv"},
    {id_graph_coloring_vivace_vv_cloth, "id_graph_coloring_vivace_vv_cloth"},
    {id_graph_coloring_vivace_vv_tet, "id_graph_coloring_vivace_vv_tet"},
    {id_graph_coloring_vivace_vv_cross, "id_graph_coloring_vivace_vv_cross"},
    {id_graph_coloring_vivace_vv_wait_for_num_color, "id_graph_coloring_vivace_vv_wait_for_num_color"},

    {id_graph_coloring_vivace_vf, "id_graph_coloring_vivace_vf"},
    {id_self_collision_reset_collision_system_and_spatial_hashing, "id_self_collision_reset_collision_system_and_spatial_hashing"},
    {id_self_collision_reset_collision_system_and_lbvh, "id_self_collision_reset_collision_system_and_lbvh"},
    {id_self_collision_fill_in_hash_table, "id_self_collision_fill_in_hash_table"},
    {id_self_collision_scan_hash_table, "id_self_collision_scan_hash_table"},
    {id_self_collision_insert_vert_into_hash_table, "id_self_collision_insert_vert_into_hash_table"},
    {id_self_collision_spatial_hashing_query, "id_self_collision_spatial_hashing_query"},
    {id_self_collision_narrow_phase_vv, "id_self_collision_narrow_phase_vv"},
    {id_self_collision_narrow_phase_vf, "id_self_collision_narrow_phase_vf"},
    {id_self_collision_scan_and_fill_in_vv, "id_self_collision_scan_and_fill_in_vv"},
    {id_self_collision_scan_and_fill_in_vf, "id_self_collision_scan_and_fill_in_vf"},

    {id_obstacle_collision_detection, "id_obstacle_collision_detection"},
    {id_obstacle_collision_spatial_hashing_update, "id_obstacle_collision_spatial_hashing_update"},
    {id_obstacle_collision_reset_collision_system_and_spatial_hashing, "id_obstacle_collision_reset_collision_system_and_spatial_hashing"},
    {id_obstacle_collision_reset_collision_system_and_lbvh, "id_obstacle_collision_reset_collision_system_and_lbvh"},
    {id_obstacle_collision_fill_in_hash_table, "id_obstacle_collision_fill_in_hash_table"},
    {id_obstacle_collision_scan_hash_table, "id_obstacle_collision_scan_hash_table"},
    {id_obstacle_collision_insert_vert_into_hash_table, "id_obstacle_collision_insert_vert_into_hash_table"},
    {id_obstacle_collision_spatial_hashing_query, "id_obstacle_collision_spatial_hashing_query"},
    {id_obstacle_collision_narrow_phase_vv, "id_obstacle_collision_narrow_phase_vv"},
    {id_obstacle_collision_narrow_phase_vf, "id_obstacle_collision_narrow_phase_vf"},
    {id_obstacle_collision_scan_and_fill_in_vv, "id_obstacle_collision_scan_and_fill_in_vv"},
    {id_obstacle_collision_scan_and_fill_in_vf, "id_obstacle_collision_scan_and_fill_in_vf"},


    {id_reduce_degree_and_set_max_color_from_max_degree_vv, "id_reduce_degree_and_set_max_color_from_max_degree_vv"},
    {id_reduce_degree_and_set_max_color_from_max_degree_vf, "id_reduce_degree_and_set_max_color_from_max_degree_vf"},
    {id_scan_uncolored_set, "id_scan_uncolored_set"},
    {id_init_palette, "id_init_palette"},
    {id_random_coloring_loops_vv, "id_random_coloring_loops_vv"},
    {id_random_coloring_loops_vf, "id_random_coloring_loops_vf"},
    {id_tentative_coloring, "id_tentative_coloring"},
    {id_conflict_resolution_vv, "id_conflict_resolution_vv"},
    {id_conflict_resolution_vf, "id_conflict_resolution_vf"},
    {id_feed_hungry_and_make_indirect_buffer, "id_feed_hungry_and_make_indirect_buffer"},
    {id_put_rest_vertices_into_additional_color, "id_put_rest_vertices_into_additional_color"},
    {id_put_rest_vertices_into_random_color, "id_put_rest_vertices_into_random_color"},
    {id_make_cluster_indirect_cmd_buffer, "id_make_cluster_indirect_cmd_buffer"},

    {id_self_collision_compute_morton, "id_self_collision_compute_morton"},
    {id_self_collision_compute_vert_aabb_and_center, "id_self_collision_compute_vert_aabb_and_center"},
    {id_self_collision_compute_face_aabb_and_center, "id_self_collision_compute_face_aabb_and_center"},
    {id_self_collision_init_tree, "id_self_collision_init_tree"},
    {id_self_collision_sort_by_morton, "id_self_collision_sort_by_morton"},
    {id_self_collision_apply_sorted_morton, "id_self_collision_apply_sorted_morton"},
    {id_self_collision_construct_tree_karras2012, "id_self_collision_construct_tree"},
    {id_self_collision_check_healthy, "id_self_collision_check_healthy"},
    {id_self_collision_compute_escape_index, "id_self_collision_compute_escape_index"},
    {id_self_collision_compute_left_index, "id_self_collision_compute_left_index"},
    {id_self_collision_update_vert_aabb_dcd, "id_self_collision_update_vert_aabb_dcd"},
    {id_self_collision_update_face_aabb_dcd, "id_self_collision_update_face_aabb_dcd"},
    {id_self_collision_apply_leaves_aabb, "id_self_collision_apply_leaves_aabb"},
    {id_self_collision_reset_lbvh, "id_self_collision_reset_lbvh"},
    {id_self_collision_query_from_cloth_vert, "id_self_collision_query_from_cloth_vert"},
    {id_obstacle_collision_compute_morton, "id_obstacle_collision_compute_morton"},
    {id_obstacle_collision_compute_vert_aabb_and_center, "id_obstacle_collision_compute_vert_aabb_and_center"},
    {id_obstacle_collision_compute_face_aabb_and_center, "id_obstacle_collision_compute_face_aabb_and_center"},
    {id_obstacle_collision_init_tree, "id_obstacle_collision_init_tree"},
    {id_obstacle_collision_sort_by_morton, "id_obstacle_collision_sort_by_morton"},
    {id_obstacle_collision_apply_sorted_morton, "id_obstacle_collision_apply_sorted_morton"},
    {id_obstacle_collision_construct_tree_karras2012, "id_obstacle_collision_construct_tree"},
    {id_obstacle_collision_check_healthy, "id_obstacle_collision_check_healthy"},
    {id_obstacle_collision_compute_escape_index, "id_obstacle_collision_compute_escape_index"},
    {id_obstacle_collision_compute_left_index, "id_obstacle_collision_compute_left_index"},
    {id_obstacle_collision_update_vert_aabb_dcd, "id_obstacle_collision_update_vert_aabb_dcd"},
    {id_obstacle_collision_update_face_aabb_dcd, "id_obstacle_collision_update_face_aabb_dcd"},
    {id_obstacle_collision_apply_leaves_aabb, "id_obstacle_collision_apply_leaves_aabb"},
    {id_obstacle_collision_reset_lbvh, "id_obstacle_collision_reset_lbvh"},
    {id_obstacle_collision_query_from_cloth_vert, "id_obstacle_collision_query_from_cloth_vert"},
    

    {id_xpbd_non, "id_xpbd_non"},
    
    {id_additional_root, "id_additional_root"},
    {id_additional_terminal, "id_additional_terminal"},
    {id_task_heft_2002, "id_task_heft_2002"},
};

#define CONSTRAINT_IDX_STRETCH 0u
#define CONSTRAINT_IDX_BENDING 1u
#define CONSTRAINT_IDX_STRESS 2u
#define CONSTRAINT_IDX_COLLISION 3u
#define NUM_CONSTRAINS 4


struct LaunchParam{
    FunctionID function_id;
    
    uint startIdx = 0;
    uint endIdx = 1;
    uint blockDim = 256;
    uint queue_idx = 0;
    uint fence_idx = 0;
    uint iter_idx = 0;
    uint cluster_idx = 0;
    bool launch_indirect = false;
    bool test_time = false;
    bool use_hetero = false;

    uint left_buffer_idx = -1u;
    // uint input_buffer_idx = -1u;
    uint output_buffer_idx = -1u;
    uint buffer_idx = 0;

    std::vector<uint> input_buffer_idxs;
    
    // bool make_fence = false;
    // bool wait_fence = false;
    // bool memory_only_pass = false;
    // uint deault_thread_num = 256;

    LaunchParam(){}
    LaunchParam(
        const FunctionID& input_task_idx, const uint& input_start_idx, const uint& input_end_idx, 
        const uint& input_blockDim, const uint& input_queue_id,  const uint& input_fence_id,
        const bool& input_launch_indirect, const bool& input_test_time, const bool& input_use_hetero
        // , const bool& input_make_fence, const bool& input_wait_fence
        ) : 
            function_id(input_task_idx), 
            startIdx(input_start_idx), 
            endIdx(input_end_idx), 
            blockDim(input_blockDim),
            queue_idx(input_queue_id),
            fence_idx(input_fence_id),
            launch_indirect(input_launch_indirect),
            test_time(input_test_time),
            use_hetero(input_use_hetero)
            // make_fence(input_wait_fence),
            // wait_fence(input_wait_fence)
            {}

    LaunchParam(
        const FunctionID& input_task_idx, 
        const uint& input_blockDim,
        const uint& input_iter_idx,
        const uint& input_cluster_idx
        ) : 
            function_id(input_task_idx), 
            blockDim(input_blockDim),
            iter_idx(input_iter_idx),
            cluster_idx(input_cluster_idx)
            {}

    LaunchParam(
        const FunctionID& input_task_idx, 
        const uint& input_blockDim,
        const uint& input_iter_idx,
        const uint& input_cluster_idx,
        const uint& task_buffer_idx,
        const uint& task_left_buffer_idx,
        const std::vector<uint>& task_input_buffer_idx,
        const uint& task_output_buffer_idx
        ) : 
            function_id(input_task_idx), 
            blockDim(input_blockDim),
            iter_idx(input_iter_idx),
            cluster_idx(input_cluster_idx),
            buffer_idx(task_buffer_idx),
            left_buffer_idx(task_left_buffer_idx),
            input_buffer_idxs(task_input_buffer_idx),
            output_buffer_idx(task_output_buffer_idx)
            {}

    
    uint get_thread_num() const{
        return endIdx - startIdx;
    }
    void print() const {
        auto it = taskNames.find(function_id);
        if (it != taskNames.end()) {
            std::cout << "Launch Task : " << function_id << " , " << it->second << " for ( " << endIdx - startIdx << " ) threads , queue/fence id = " 
                << queue_idx << " / "<< fence_idx << " \n";
        } else {
            std::cerr << "Function Does Not Exist\n";
        }
        
    }
};

using DeviceType = uint;
const DeviceType DeviceTypeCpu = 0;
const DeviceType DeviceTypeGpu = 1;
const DeviceType DeviceType0 = 0;
const DeviceType DeviceType1 = 1;
const DeviceType DeviceType2 = 2;

// enum DeviceType{
//     DeviceTypeGpu = 0,
//     DeviceTypeCpu = 1,
//     DeviceType0 = 0,
//     DeviceType1 = 1,
//     DeviceType2 = 2,
// };

struct Implementation{
    DeviceType device;
    // uint implementation_id; // seems unnecessary... maybe used for saperate different implementation for each task instead of using function interface
    std::function<void(const Launcher::LaunchParam&)> implementation;

    Implementation():
        device(DeviceTypeCpu), implementation([](const Launcher::LaunchParam&)->void{}){}

    Implementation(const DeviceType& device_type, const std::function<void(const Launcher::LaunchParam&)>& Implementation):
        device(device_type), implementation(Implementation) {}

    Implementation(const Implementation& other):
        device(other.device), implementation(other.implementation) {}

    void launch_task(const LaunchParam& param) const {
        // if (get_scene_params().print_task_info) param.print();
        // if (param.function_id == id_additional_root || param.function_id == id_additional_terminal ) return;
        implementation(param);
    }

    bool operator==(const Implementation& right) const {
        return device == right.device;
    }
};

struct State{
    bool _has_done = false;
    void reset_state() { _has_done = false; }
    void mark_done() { _has_done = true; }
    bool has_done() const { return _has_done; }
    // State(const State& input_state) : ()
};

struct Task{
    FunctionID func_id;
    uint num_threads = 0;
    uint start_idx = 0;
    uint block_dim = 256;
    uint iter_idx = -1u;
    uint constraint_idx = -1u;
    uint cluster_idx = 0;
    bool can_be_seperate = false;
    bool is_num_threads_constant = true;
    bool is_computation_time_constant = false;
    bool is_memory_only_pass = false;
    bool use_indirect_dispatch = false;
    bool is_empty_node = false;
    std::vector< Implementation > list_implementation;

    uint buffer_left = -1u;  uint task_left = -1u;
    uint buffer_in = -1u;    uint task_in = -1u;
    uint buffer_out = -1u;   uint task_out = -1u;
    uint buffer_idx = -1u;

    std::vector<uint> buffer_lefts;
    std::vector<uint> buffer_ins;
    std::vector<uint> buffer_outs;
    std::vector<uint> buffer_idxs;
    std::vector<uint> task_lefts;
    std::vector<uint> task_ins;
    std::vector<uint> task_outs;
    std::vector<uint> task_idxs;

    std::vector< uint > resources; 
    
    std::vector< uint > predecessors; // back : current's rely
    std::vector< uint > successors;   // front : what task rely on currents

    std::vector< float > list_weight;
    // std::vector< uint >  list_offset; // offset of current node to its predecessors

    Task() :
        func_id(id_xpbd_non), num_threads(0), start_idx(0), block_dim(256), can_be_seperate(true), is_num_threads_constant(true), list_implementation({}),
        use_indirect_dispatch(false), indirect_buffer_cpu(nullptr), indirect_buffer_gpu(nullptr) {}
    
    Task(const FunctionID& functionID, const uint& numThreads, const bool& canBeSeparate, const bool& isNumThreadsConstant, const std::vector< Implementation >& listInterface):
        func_id(functionID), num_threads(numThreads), can_be_seperate(canBeSeparate), is_num_threads_constant(isNumThreadsConstant), list_implementation(listInterface),
        start_idx(0), block_dim(256), use_indirect_dispatch(false), indirect_buffer_cpu(nullptr), indirect_buffer_gpu(nullptr) {}

    Task(const FunctionID& functionID, const uint& clusterIdx, const std::vector< Implementation >& listInterface):
        func_id(functionID), cluster_idx(clusterIdx), list_implementation(listInterface),
        block_dim(256){}

    Task(const FunctionID& functionID, const uint& iterIdx, const uint& constraintIdx, const uint& clusterIdx, const std::vector< Implementation >& listInterface):
        func_id(functionID), iter_idx(iterIdx), constraint_idx(constraintIdx), cluster_idx(clusterIdx), list_implementation(listInterface),
        block_dim(256){}

    Task(const FunctionID& functionID, const uint& iterIdx, const uint& constraintIdx, const uint& clusterIdx, const std::vector<uint>& resources, const std::vector< Implementation >& listInterface):
        func_id(functionID), iter_idx(iterIdx), constraint_idx(constraintIdx), cluster_idx(clusterIdx), resources(resources), list_implementation(listInterface),
        block_dim(256){}

    Task(const Task& input_task) :
        func_id(input_task.func_id), 
        num_threads(input_task.num_threads), 
        start_idx(input_task.start_idx), 
        block_dim(input_task.block_dim),
        iter_idx(input_task.iter_idx),
        constraint_idx(input_task.constraint_idx),
        cluster_idx(input_task.cluster_idx),
        can_be_seperate(input_task.can_be_seperate), 
        is_num_threads_constant(input_task.is_num_threads_constant), 
        is_memory_only_pass(input_task.is_memory_only_pass), 
        use_indirect_dispatch(input_task.use_indirect_dispatch), 
        is_empty_node(input_task.is_empty_node), 
        list_implementation(input_task.list_implementation),

        buffer_left(input_task.buffer_left), 
        buffer_in(input_task.buffer_in), 
        buffer_out(input_task.buffer_out), 
        task_left(input_task.task_left), 
        task_in(input_task.task_in), 
        task_out(input_task.task_out), 
        buffer_idx(input_task.buffer_idx), 
        
        buffer_ins(input_task.buffer_ins), 
        buffer_outs(input_task.buffer_outs), 
        buffer_lefts(input_task.buffer_lefts), 
        buffer_idxs(input_task.buffer_idxs), 
        task_lefts(input_task.task_lefts), 
        task_ins(input_task.task_ins), 
        task_outs(input_task.task_outs), 
        resources(input_task.resources), 

        predecessors(input_task.predecessors), 
        successors(input_task.successors), 
        
        list_weight(input_task.list_weight),
        indirect_buffer_cpu(input_task.indirect_buffer_cpu), 
        indirect_buffer_gpu(input_task.indirect_buffer_gpu)
        // , list_offset(input_task.list_offset) 
        {}

    void print_with_cluster(const uint tid) const {
        if (iter_idx != -1u) 
        {
            fast_format("    Task {} in iteraion {:2}'s cluster {} => {}", tid, iter_idx, cluster_idx, taskNames.at(func_id));
        }
        else 
        {
            fast_format("    Task {} with cluster {} => {}", tid, cluster_idx, taskNames.at(func_id));
        }
    }

    void set_indirect_buffer(Int4* indirect_command_buffer_cpu, Int4* indirect_command_buffer_gpu){
        use_indirect_dispatch = true;
        indirect_buffer_cpu = indirect_command_buffer_cpu;
        indirect_buffer_gpu = indirect_command_buffer_gpu;
    }
    uint get_dispatch_num_by_indirect_buffer(){
        return indirect_buffer_cpu[0][3];
    }
    void set_block_dim(const uint input_block_dim){
        block_dim = input_block_dim;
    }
    void adapt_block_dim(){
        if(block_dim != 256) {
            if(num_threads < 512) { block_dim = 32; } /// 16 * 32 = 
            else if(num_threads < 1024) { block_dim = 64; }
            else if(num_threads < 2048) { block_dim = 128; }
        }
        // else { block_dim = 256; }
    }
    void set_task_empty() { is_empty_node = true; }

    // Connect
    void add_front(const uint& front_idx)   { successors.push_back(front_idx); }
    void add_back(const uint& back_idx)     { predecessors.push_back(back_idx); }

    // Implementation
    const Implementation& get_implementation(const DeviceType& type, bool& find_implement) const{
        for(auto& imp : list_implementation){
            if(imp.device == type){
                find_implement = true;
                return imp;
            }
        }
        find_implement = false;
        return list_implementation[0];
    }
    bool has_implementation(const DeviceType& type) const {
        for (auto& imp : list_implementation){
            if (imp.device == type){
                return true;
            }
        }
        return false;
    }

private:
    // State state;
    Int4* indirect_buffer_cpu;
    Int4* indirect_buffer_gpu;
};

struct MergedTask {        
    std::vector< uint > tasks; uint constraint_idx;

    std::vector< uint > predecessors; // back : current's rely
    std::vector< uint > successors;   // front : what task rely on currents
    std::vector< float > costs;
    std::vector< float > weights;

    MergedTask() {}
    MergedTask(const std::vector<uint>& input_tasks, const uint& input_constraint_idx) : tasks(input_tasks), constraint_idx(input_constraint_idx) {}
};

#define SWITCH_TASK(task_id, FUNC)  \
    case Launcher::task_id: {FUNC; break;}
        

// Structure to represent a scheduled task
struct ScheduleEvent {
    uint task_id;
    uint proc;  // 0 for CPU, 1 for GPU
    float start;
    float end;
    ScheduleEvent()
        : task_id(0), proc(0), start(0.f), end(0.f) {}
    ScheduleEvent(uint t, uint p, float s, float e) 
        : task_id(t), proc(p), start(s), end(e) {}
    ScheduleEvent(const ScheduleEvent& input)
        : task_id(input.task_id), proc(input.proc), start(input.start), end(input.end) {}

    bool operator<(const ScheduleEvent& right) const {
        if(end != right.end)
            return end < right.end;
        else
            return start < right.start;
    }
    bool operator>(const ScheduleEvent& right) const {
        if(end != right.end)
            return end > right.end;
        else
            return start > right.start;
    }
    void print() const{
        std::cout << "TaskSchedule : node = " << task_id << " , proc = " << proc << " , from " << start << " to " << end << std::endl; 
    }
    std::string get_str() const {
        return std::format(" [ tid = {} , proc = {} , from {} to {}] ", task_id, proc, start, end);
    }
};

struct LaunchEvent{
    uint start_idx;
    uint end_idx;
    uint wait;
    uint signal;
    LaunchEvent() 
        : start_idx(0), end_idx(0), wait(-1u), signal(-1u) {}
    LaunchEvent(const uint& stt, const uint& ed, const uint& wt, const uint& sgn) 
        : start_idx(stt), end_idx(ed), wait(wt), signal(sgn) {}
    void print() const {
        std::string str_wait   = (wait == -1u)   ? "null" : std::to_string(wait);
        std::string str_signal = (signal == -1u) ? "null" : std::to_string(signal);
        std::cout << "    From " << start_idx << " to " << end_idx << " , Wait / Signal " << str_wait << " / " << str_signal << std::endl;
    }
}; 

#define BROTHER_CONNECTION_RALATIONSHIP 3

class Scheduler{

private:
    enum OpMode {
        OpModeEFT,
        OpModeEDP_ABS,
        OpModeEDP_REL,
        OpModeENERGY
    };

    enum RankMetric {
        RankMetricWORST,
        RankMetricBEST,
        RankMetricEDP,

        RankMetricMEAN,
        RankMetricLookAhead, // 4-order
        RankMetricOCT, // 2-order
    };

    using ListTask = std::vector<uint>;
    using ListCost = std::vector<float>;
    using ListWeight = std::vector<float>;
    using ListMask = std::vector<bool>;
    using ListSchedule = std::vector<ScheduleEvent>;

private:
    

private:
    // std::vector< Connect > list_connect;
    std::vector< Task > list_task;
    std::vector<uint> list_order;
    std::vector<uint> sorted_nodes;
    uint sync_count = 1;

private:
    
    std::vector< MergedTask > list_task_merged; 
    std::vector< uint > task_to_merged_task_map; 

    std::vector< ScheduleEvent > task_schedules_merged;
    std::vector< std::vector<ScheduleEvent> > proc_schedules_merged;

    std::vector<Task> assemble_implementations;
    std::vector<uint> constraint_task_orders;

public:
    Scheduler() {}
    Scheduler(const Scheduler& input_scheduler){
        list_task = input_scheduler.list_task;
        // list_connect = input_scheduler.list_connect;
    }

    
    void set_as_sync_as_possible(const std::vector<Task>& assemble_impl); 
    void set_sync_count(const uint& input_sync_count) ;
    void set_constraint_task_orders(const std::vector<uint>& constraint_task_orders);
    void set_constant_computation_time_task();
    void get_dag_from(const Scheduler& input_scheduler);
    uint add_task(const Task& task);
    Task& get_task_by_tid(const uint& task_id);
    uint find_task_by_func_id(const Launcher::FunctionID id);

    void set_connect(const uint& task_idx1, const uint& task_idx2, const float& weight = 1.0);
    float delete_connect(const uint& task_idx1, const uint& task_idx2);

public:
    using OptScalar = double;

    bool topological_sort();
    void set_loop_count(const uint loop_count);
    void print_sort_by_typology();
    void print_sort_by_ranku();
    void profile(const std::function<void()>& clear_func, const std::function<LaunchParam(const Task&)> task_to_param, const std::vector< std::vector<float> >& profiled_comp_matrix = {});
    void profile(std::function<void(
        const std::vector<uint>& list_order, const std::vector<Task>& list_task, 
        std::vector<std::vector<double>>& cost_list_cpu, std::vector<std::vector<double>>& cost_list_gpu, std::vector<double> & cost_total)>); // Given the task-order and task-list, need to get the cost in several iteration
    void profile_from(
        const std::vector< std::pair<Launcher::FunctionID, uint> >& map_tasks, 
        const std::vector< std::vector<double> >& map_costs,
        std::vector<double> & cost_total);

    void standardizing_dag(const std::vector< std::function<void(const Launcher::LaunchParam&)> >& list_fn_empty_func = {});
    void scheduler_dag();
    void post_process();
    void make_wait_events();

    float fn_get_communication_cost(const uint proc, const uint pred_node, const uint input_node);
    float fn_get_inner_communication_cost(const uint proc);

public:

    
    enum LaunchMode{
        LaunchModeCpu, // Run tasks sequencely based on Topology Sorting on the CPU (Considering all tasks are supported by CPU)
        LaunchModeGpu, // Run tasks sequencely based on Topology Sorting on the GPU (If some tasks are not supported by GPU, they will wait and run on CPU)
        LaunchModePartialCPU, // Run SCHEDULED tasks that allocated to the CPU
        LaunchModePartialGPU, // Run SCHEDULED tasks that allocated to the CPU
        LaunchModeHetero,     // Run SCHEDULED tasks with Hybrid CPU/GPU, with asynchronous communication
        LaunchModeProgressiveHetero, // Run SCHEDULED tasks in 'LaunchModeHetero' may not work when there are too many tasks (e.g. 40 command-buffers on the GPU).
                                     // This may be limited to hardware or Metal API. This problem may not happen on other platforms.
                                     // For now, we use 'LaunchModeProgressiveHetero' to run the tasks in a progressive way that 
                                     //     we do not send too many command buffers to the GPU at once. 
        LaunchModeFakeHetero, // Run SCHEDULED tasks with synchronous waiting (the final result should be the same as LaunchModeHetero)
        LaunchModeSequeceHetero, // Run SCHEDULED tasks on the CPU, not eqaul to 'LaunchModeCpu'
    };
    void launch(LaunchMode mode, const std::function<LaunchParam(const Task&)> task_to_param, const bool fully_not_wait, const std::vector<std::function<void()>>& assemble_impl = {});
    void launch_mixed(uint stop_cpu_idx, uint stop_gpu_idx, std::function<void()> additioal_func);
    
    void print_tasks();
    void print_costs(bool use_sort = true);
    void print_schedule();
    void print_dag();
    void print_scheduling_with_waiting_events();
    void print_proc_schedules();
    void print_schedule_to_graph_xpbd();
    void print_speedups_to_each_device();
    
    float get_scheduled_end_time();
    float get_theoretical_time();
    std::vector<float> get_proc_usage();
    std::vector<float> get_scheduled_speedups();
    void reset_scheduler_system();
    void set_communication_matrix(const float cpu_to_gpu, const float gpu_to_cpu) 
    { 
        uint cpu_idx = 0;
        uint gpu_idx = 1;
        communication_cost_matrix_uma[cpu_idx][gpu_idx] = cpu_to_gpu; 
        communication_cost_matrix_uma[gpu_idx][cpu_idx] = gpu_to_cpu; 
    }

private:
    std::vector< Task > orig_list_task; /// (list_task);
    std::vector< ListCost > orig_computation_matrix; /// (computation_matrix);
    std::vector< double > list_convergence;
    std::vector<uint> list_additional_task_orig_tid;
    // std::unordered_map<uint, uint>list_additional_task_orig_tid_map;

private:
    void compute_ranku(uint num_procs = 2);
    
    ScheduleEvent _compute_eft(uint node, uint proc);
    ScheduleEvent _compute_eft_extend(uint node, uint proc);
    ScheduleEvent _compute_eft_merged(uint node, uint proc);

    /// For Post Process
    // bool procFree(const uint& proc, const ListSchedule& jobs, const float& time, uint& idx);
    // uint taskidxTotaskpos(const ListSchedule& jobs, const uint& taskidx);
    
private:
    std::vector< ScheduleEvent > task_schedules;
    std::vector< double > ranku;
    std::vector< std::vector<double> > oct; // Optimisic Cost Table
    std::vector< std::vector<ScheduleEvent> > proc_schedules;

    std::vector< std::vector<uint> > list_cmd_idx;
    std::vector< std::vector<LaunchEvent> > launch_events;

    // std::vector< MTL::CommandBuffer* > list_cmd_buffer;

public:
    std::vector< ListCost > computation_matrix;
    std::vector< double > summary_of_costs_each_device;

public:
    // Communication speed between processors, diagonal should be the constant communication cost inner device
    std::vector<std::vector<float>> communication_speed_matrix = { 
        {0, 1, 1}, 
        {1, 0, 1}, 
        {1, 1, 0}
    };     
    // Communication speed between processors  
    std::vector<std::vector<float>> communication_cost_matrix_uma = {
        {0.002, 0.220},  /// gpu wait cpu
        {0.145, 0.01}    /// cpu wait gpu
    };
    // First call cost   
    std::vector<float> communication_startup = {0, 0, 0}; 
    // Should be additional root & additional terminal                    
    uint root_node, terminal_node;

    /// Map Of Cost : CPU to GPU , GPU to CPU , Cost , Theory Cost (Sort By Real Time)
    //  0.220 ,  0.145 =>  2.811 (Theory =  2.889)
    //  0.220 ,  0.145 =>  2.813 (Theory =  2.889)
    //  0.240 ,  0.110 =>  2.848 (Theory =  2.838)
    //  0.235 ,  0.115 =>  2.887 (Theory =  2.838)
    //  0.235 ,  0.115 =>  2.890 (Theory =  2.838)
    //  0.245 ,  0.110 =>  2.896 (Theory =  2.843)
    //  0.245 ,  0.105 =>  2.943 (Theory =  2.838)
    //  0.240 ,  0.115 =>  2.947 (Theory =  2.843)
    //  0.240 ,  0.110 =>  2.951 (Theory =  2.838)
    /// Map Of Cost : CPU to GPU , GPU to CPU , Cost , Theory Cost (Sort By Delta)
    // 0.240 ,  0.120 =>  3.698 (Theory =  3.437)
    //  0.220 ,  0.145 =>  2.813 (Theory =  2.889)
    //  0.240 ,  0.110 =>  2.848 (Theory =  2.838)
    //  0.235 ,  0.115 =>  2.887 (Theory =  2.838)
    //  0.235 ,  0.115 =>  2.890 (Theory =  2.838)
    //  0.245 ,  0.110 =>  2.896 (Theory =  2.843)
    //  0.245 ,  0.105 =>  2.943 (Theory =  2.838)
    //  0.240 ,  0.115 =>  2.947 (Theory =  2.843)
    //  0.240 ,  0.110 =>  2.951 (Theory =  2.838)

private:

    
public:
    void test_case_2002();

};


}


// using TaskPtr = SharedTask*;

// namespace Launcher{

// const uint num_launcher = 2;

// // template<typename T, typename... Args>
// // void profile(ReduceTask<T, Args...> reduce_task){
// //     if(max_threads == 0){
// //         std::cerr << "thread is zero"; return;
// //     }
// //     get_command_list();
// // }

// inline TaskPtr launch_all_parallel_gpu(TaskPtr start_task, uint& command_list_index){

//     TaskPtr ptr = start_task;
//     do {
//         ptr -> set_command_list_index(command_list_index);
//         command_list_index = 1 - command_list_index;
//         ptr -> launch_gpu_partial_async();
//         if(ptr -> has_brother()){
//             ptr = ptr -> get_brother();
//         }
//         else {
//             ptr = ptr -> get_next();
//             break;
//         }
//     } while (true);
//     return ptr;
// }

// inline void start_launch_loop(SharedTask* start_task){
//     TaskPtr ptr = start_task;

//     // MTL::Fence* adsf;
//     // 先调度gpu的部分
//     uint command_list_index = 0;
//     TaskPtr current_end = launch_all_parallel_gpu(ptr, command_list_index);

//     while(!(ptr -> is_last_task())){
        
//         if(ptr -> has_brother()){
//             ptr = ptr -> parallel_ptr;
//         }
//         else{
//             ptr = ptr -> next_ptr;
//         }
//     }

// }



// }

