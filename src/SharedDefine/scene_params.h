#pragma once

#ifndef  METAL_CODE
#include "shared_array.h"
#endif

#include "float_n.h"

struct SceneParams {
// public:
    bool simulate_cloth = false;
    bool simulate_tet = false;
    bool simulate_obstacle = false;
    
    bool use_explicit = false;
    bool use_gpu = false;
    bool use_jacobi = false;
    bool use_substep = true;
    bool use_mas = true;
    bool use_test_solver = true;
    bool use_hete_solve = false;
    bool use_homo_solve = false;
    bool profiling_communication_matrix = false;
    bool print_pcg_convergence = false;
    bool use_bending = true;
    bool use_simple_pcg = false;
    bool draw_line = true;
    bool draw_obstacle = true;
    bool draw_cloth = true;
    bool draw_deformable_body = true;
    bool is_interactive_frame = false;
    bool use_self_collision = true;
    bool use_obstacle_collision = true;
    bool use_collision_vf = false;
    bool use_collision_ee = false;
    bool print_collision_count = false;
    bool use_untangling = true;
    bool use_cloth_order = false;
    bool use_habok_bvh = false;
    bool use_ccd_constraint = false;
    bool use_floor = true;
    bool print_task_info = false;
    bool fix_scene = false;
    bool disable_enable_fixed_cloth = true;
    bool disable_enable_fixed_tet = true;
    bool print_cost_detail = false;
    bool print_system_energy = false;
    bool use_async_pred = false;
    bool use_multi_buffer = false;
    bool assemble_with_projection = true;
    bool use_quadratic_bending_model = false;
    bool use_small_timestep = false;
    bool use_saperation_constraint = false;
    bool use_obstacle_animation = true;
    bool output_all_frame = false;
    bool use_big_damping = false;
    bool use_stretch_animation = false;
    bool use_fake_hetero = false;
    bool print_xpbd_convergence = false;
    bool use_xpbd_solver = true;
    bool use_vbd_solver = false;
    bool use_sod_solver = false;
    bool sod_use_jacobi = false;

    uint current_frame = 0;
    
    uint num_iteration = 96;   
    uint num_substep = 8;
    uint collision_detection_frequece = 1;
    uint current_substep = 0;
    uint constraint_iter_count = 8;
    uint current_it = 0;

    uint animation_start_frame = 9999;

    
    uint max_vv_per_vert_broad_self_collision = 32;
    uint max_vf_per_vert_broad_self_collision = 32;
    uint max_vv_per_vert_narrow_self_collision = 16;
    uint max_vf_per_vert_narrow_self_collision = 16;
    uint max_vv_per_vert_broad_obstacle_collision = 32;
    uint max_vf_per_vert_broad_obstacle_collision = 32;
    uint max_vv_per_vert_narrow_obstacle_collision = 16;
    uint max_vf_per_vert_narrow_obstacle_collision = 16;

    uint laplasion_damping_cloth_count = 1;
    uint laplasion_damping_tet_count = 1;
    float laplasion_damping_cloth_weight = 0.5;
    float laplasion_damping_tet_weight = 0.5;

    uint use_big_damping_frame = 999999;
    uint dag_mode = -1u; // -1u:non, 0:full cloth, 1:full tet, 2:hybrid

    float implicit_dt = 1.f/30.f;
    // float implicit_dt = 1.f/1000.f;
    float explicit_dt = 1E-4;
    float dt = implicit_dt;
    float dt_inv = 1.0f / dt;
    float dt_2_inv = dt_inv * dt_inv;
    float stiffness_stretch_BaraffWitkin = 200.f;
    float stiffness_stretch_spring = 1e4;
    float youngs_modulus_cloth = 1e7;
    float youngs_modulus_tet = 1e7;
    float poisson_ratio_cloth = 0.2f;
    float poisson_ratio_tet = 0.2f;
    float stiffness_bending_ui = 0.5f;
    float stiffness_bending_basic = 1E-4f;
    float stiffness_bending = stiffness_bending_ui * stiffness_bending_basic;
    float stiffness_quadratic_bending = 0.05f;
    float stiffness_DAB_bending = stiffness_quadratic_bending;
    float stiffness_ef_ui = 5.f;
    float stiffness_ef_basic = 0.1f;
    float balloon_scale_rate = 1.0f; 
    float stiffness_pressure = 1e5; 
    float damping_cloth = 2.0f; 
    float damping_tet = 0.0f; 

    // float stiffness_collision = pow_scalar(10.f, stiffness_collision_ui);
    float stiffness_ef = stiffness_ef_ui * stiffness_ef_basic;
    float stiffness_ee = 1.f;
    float xpbd_stiffness_collision = 1e7;
    float stiffness_tet_volumn = 1.f;
    float stiffness_tet_shear = 1.f;
    float default_mass = 1.0f;
    float thick_ness = 0.01f;
    float stretch_bending_assemble_weight = 1.f;
    float thickness_vv = 0.01f;
    float thickness_vv_obstacle = 0.01f;
    float thickness_vv_cloth = 0.01f;
    float thickness_vv_tet = 0.01f;
    float thickness_vv_cross = 0.01f;
    float thickness_vf = 0.01f;
    float thickness_ee = 0.01f;
    float friction_cloth = 0.25;
    float friction_tet = 0.1;
    float friction_cross = 0.1;
    float friction_obstacle_cloth = 0.25;
    float friction_obstacle_tet = 0.1;
    float collision_query_range_vv = thickness_vv * Sqrt_2;
    float density_cloth = 0.01;
    float density_tet = 10.0;
    float chebyshev_omega = 1.0f;
    

    Float3 gravity{0, -9.8f, 0};
    Float3 floor{0, -0.5f, 0};

    // constant
    bool use_half_triangle = false;

    bool prev_assemble = false;
    bool use_direct_atomic_assemble = false;
    bool use_fixed_verts = false;
    bool use_attach = false;

    SceneParams(){}
    
    uint get_curr_iteration_with_substep() { return num_substep * current_substep + current_it; }
    float get_substep_dt() { return implicit_dt / float(num_substep); }
    float get_stiffness_quadratic_bending() { return stiffness_quadratic_bending * stiffness_bending_ui; }
    float get_stiffness_DAB_bending() { return stiffness_DAB_bending * stiffness_bending_ui; }
    
};

#ifndef  METAL_CODE

void init_scene_params();
SceneParams& get_scene_params();
SharedArray<SceneParams>& get_scene_params_array();

#endif

// extern SharedArray<SceneParams> scene_params;
// extern std::shared_ptr<SceneParams> scene_params;