#pragma once

///
/// Define The XPBD Structure
///

#include "address_space.h"
#include "atomic.h"

#include "sim_data.h"
#include "collision_data.h"
#include "make_arguments.h"
#include "obstacle_data.h"

#ifndef METAL_CODE
#include "shared_array.h"
#include "clock.h"
#endif

namespace Constrains {

enum StretchType {
    StretchTypeMassSpring,
    StretchTypeStVK,
    StretchTypeBaraffWitkin,
};
enum BendingType {
    BendingTypeNone,
    BendingTypeQuadratic,
    BendingTypeDAB,
    BendingTypeMassSpring,
};

enum ConstraintSolverType {
    ConstraintSolverTypeGaussSeidel,
    ConstraintSolverTypeColoredGaussSeidel,
    ConstraintSolverTypeJacobi,
};

enum ColoringMethod {
    ColoringMethodSequecedConstraint,
    ColoringMethodSequecedVertex,
    ColoringMethodRandomVertex,
    ColoringMethodComplementaryColoring,
};

// CONSTEXPR Constrains::ColoringMethod coloring_method = Constrains::ColoringMethodSequecedConstraint;

}// namespace Constrains

struct XpbdSelfCollision {
    // Broad Phase
    Array(uint)
        hash_table;
    Array(uint)
        hash_table_count;
    Array(uint)
        hash_table_prefix;// CSR
    Array(uint)
        hash_table_belongs;
    Array(uchar)
        hash_table_flag;
    Array(uint)
        hash_table_vert_offset;
    Array(uint)
        broad_phase_list;

    // Narrow Phase
    Array(uint)
        collision_count;// [0 : collision pair count][1 : num_not_collide][2 : min_degree][3 : max_degree]
    Array(float)
        max_vert_rest_distance;

    Array(Int2)
        narrow_phase_list_indices_vv;
    Array(ProximityVV)
        narrow_phase_list_pair_vv;
    Array(ProximityVV)
        narrow_phase_list_pair_vv_merged;
    Array(Int4)
        narrow_phase_list_indices_vf;
    Array(ProximityVF)
        narrow_phase_list_pair_vf;
    Array(uint)
        vert_adj_elements;
    Array(uchar)
        collision_pair_offset_in_vert;
    Array(Int4)
        self_collision_indirect_cmd_buffer;

    Array(uint)
        vert_VV_num_broad_phase;
    Array(uint)
        vert_VV_num_narrow_phase;
    Array(uint)
        vert_VV_prefix_narrow_phase;

    uint table_size;
};

struct XpbdObstacleCollision {
    // Broad Phase
    Array(uint)
        hash_table;
    Array(uint)
        hash_table_count;
    Array(uint)
        hash_table_prefix;// CSR
    Array(uint)
        hash_table_belongs;
    Array(uchar)
        hash_table_flag;
    Array(uint)
        hash_table_vert_offset;// CSR
    Array(uint)
        broad_phase_list;// ELL

    // Narrow Phase
    Array(uint)
        collision_count;// [0 : collision pair count][1 : num_not_collide][2 : min_degree][3 : max_degree]
    Array(Int2)
        narrow_phase_list_indices_vv;
    Array(ProximityVV)
        narrow_phase_list_pair_vv;
    Array(Int4)
        narrow_phase_list_indices_vf;
    Array(ProximityVF)
        narrow_phase_list_pair_vf;
    Array(uint)
        vert_adj_elements;// CSR

    Array(uint)
        vert_VV_num_broad_phase;
    Array(uint)
        vert_VV_num_narrow_phase;
    Array(uint)
        vert_VV_prefix_narrow_phase;
    Array(uchar)
        collision_pair_offset_in_vert;
    Array(Int4)
        obstacle_collision_indirect_cmd_buffer;

    uint table_size;
};

struct XpbdData {

    Array(float)
        sa_system_energy;

    Array(Float3) sa_x_frame;
    Array(Float3) sa_v_frame;

    Array(Float3) sa_x_tilde;
    Array(Float3) sa_x;
    Array(Float3) sa_v;
    Array(Float3) sa_x_iter_start;
    Array(Float3) sa_x_step_start;

    Array(float)
        debug_buffer;
    Array(AABB)
        sa_block_aabb;

    Array(uint)
        sa_surface_verts;
    Array(Float3)
        sa_surface_faces;

    Array(uint)
        clusterd_constraint_tet_stress;
    Array(uint)
        prefix_tet_stress;

    Array(Float3)
        sa_iter_position_tet_cpu;
    Array(Float3)
        sa_iter_position_tet_gpu;
    Array(Float3)
        sa_begin_position_tet_cpu;
    Array(Float3)
        sa_begin_position_tet_gpu;
    Array(Float3)
        sa_async_iter_positions_tet[32];
    Array(Float3)
        sa_async_begin_positions_tet[32];

    Array(Int3)
        sa_detection_faces;
    Array(Float3)
        sa_detection_position_bg;
    Array(Float3)
        sa_detection_position_ed;

    Array(float)
        lambda_ground_collision_tet;

    Array(float)
        lambda_tet_stress_hydrostatic_term;
    Array(float)
        lambda_tet_stress_deviatoric_term;

    Array(float)
        lambda_self_collision_tet;
    Array(float)
        lambda_self_collision_friction_tet;
    Array(float)
        lambda_sdf_collision_tet;
    Array(float)
        lambda_sdf_collision_tet_friction;

    // Sorted By Graph Coloring
    Array(Int4)
        sa_merged_tets;
    Array(Float3x3)
        sa_merged_Dm_inv;
    Array(float)
        sa_merged_tet_volumn;

    Array(Int4)
        sa_merged_inner_tets;
    Array(Int4)
        sa_merged_outer_tets;
    Array(Float3x3)
        sa_merged_inner_Dm_inv;
    Array(Float3x3)
        sa_merged_outer_Dm_inv;
    Array(float)
        sa_merged_outer_tet_volumn;
    Array(float)
        sa_merged_inner_tet_volumn;

    XpbdSelfCollision tet_collision;
    XpbdObstacleCollision obs_collision_tet;

    LbvhFaceEdgeData lbvh_data_obstacle;
    LbvhFaceEdgeData lbvh_data_tet;

    uint num_clusters_tet_stress;

    uint num_combined_clusters_self_collision = 2;
    uint num_combined_clusters_stress = 2;

    uint num_verts_collision_total = 0;
    uint num_faces_collision_total = 0;

    Constrains::StretchType stretch_type = Constrains::StretchTypeBaraffWitkin;                                  // Stretch Model
    Constrains::BendingType bending_type = Constrains::BendingTypeQuadratic;                                     // Bending Model
    Constrains::ConstraintSolverType constraint_solver_type = Constrains::ConstraintSolverTypeColoredGaussSeidel;//
    bool use_chebyshev_accelaration = false && constraint_solver_type == Constrains::ConstraintSolverTypeJacobi;
    bool compute_material_energy_only = false;

    uint get_num_tets_clusters_neohookean_fem(const uint cluster_id) { return clusterd_constraint_tet_stress[cluster_id + 1] - clusterd_constraint_tet_stress[cluster_id]; }

#ifndef METAL_CODE
    void resize(TetData *tetrahedral, ObstacleData *obstacle) {
        const uint num_verts_tet = tetrahedral->num_verts_total;
        const uint num_surface_verts_tet = tetrahedral->num_surface_verts_total;
        const uint num_surface_faces_tet = tetrahedral->num_surface_faces_total;
        const uint num_tets_total = tetrahedral->num_tets_total;

        num_verts_collision_total = num_surface_verts_tet;
        num_faces_collision_total = num_surface_faces_tet;

        // Resize
        debug_buffer.resize(1024);

        lambda_tet_stress_hydrostatic_term.resize(num_tets_total);
        lambda_tet_stress_deviatoric_term.resize(num_tets_total);

        lambda_ground_collision_tet.resize(num_verts_tet);

        sa_iter_position_tet_cpu.resize(num_verts_tet);
        sa_iter_position_tet_gpu.resize(num_verts_tet);
        sa_begin_position_tet_cpu.resize(num_verts_tet);
        sa_begin_position_tet_gpu.resize(num_verts_tet);

        for (auto &buffer : sa_async_iter_positions_tet) { buffer.resize(num_verts_tet); }
        for (auto &buffer : sa_async_begin_positions_tet) { buffer.resize(num_verts_tet); }

        sa_detection_position_bg.resize(num_verts_collision_total);
        sa_detection_position_ed.resize(num_verts_collision_total);
        sa_detection_faces.resize(num_surface_faces_tet);
        sa_surface_verts.resize(num_verts_collision_total);
        sa_surface_faces.resize(num_faces_collision_total);

        sa_block_aabb.resize(get_dispatch_num(num_verts_tet, 256));

        const uint B_selfcollision = 48;
        const uint N_selfcollision = 32;
        const uint B_obscollision = 32;
        const uint N_obscollision = 16;
        const uint vert_num_each_collision_pair_self_collision = 2;// VV
        const uint vert_num_each_collision_pair_obs_collision = 1; // V in VF
        {
            lambda_self_collision_tet.resize(num_surface_verts_tet * N_selfcollision);
            lambda_self_collision_friction_tet.resize(num_surface_verts_tet * N_selfcollision);

            lambda_sdf_collision_tet.resize(num_surface_verts_tet * (N_obscollision + 1));
            lambda_sdf_collision_tet_friction.resize(num_surface_verts_tet * (N_obscollision + 1));
        }
        {
            lbvh_data_tet.vert_tree.tree_type = LBVHTreeTypeVert;
            lbvh_data_tet.face_tree.tree_type = LBVHTreeTypeFace;
            lbvh_data_tet.edge_tree.tree_type = LBVHTreeTypeEdge;
            lbvh_data_tet.vert_tree.update_type = LBVHUpdateTypeCloth;
            lbvh_data_tet.face_tree.update_type = LBVHUpdateTypeCloth;
            lbvh_data_tet.edge_tree.update_type = LBVHUpdateTypeCloth;
            lbvh_data_tet.vert_tree.allocate(num_surface_verts_tet);
            lbvh_data_tet.edge_tree.allocate(1);
            lbvh_data_tet.face_tree.allocate(1);

            lbvh_data_obstacle.vert_tree.tree_type = LBVHTreeTypeVert;
            lbvh_data_obstacle.face_tree.tree_type = LBVHTreeTypeFace;
            lbvh_data_obstacle.edge_tree.tree_type = LBVHTreeTypeEdge;
            lbvh_data_obstacle.vert_tree.update_type = LBVHUpdateTypeObstacle;
            lbvh_data_obstacle.face_tree.update_type = LBVHUpdateTypeObstacle;
            lbvh_data_obstacle.edge_tree.update_type = LBVHUpdateTypeObstacle;
            lbvh_data_obstacle.vert_tree.allocate(1);
            lbvh_data_obstacle.edge_tree.allocate(1);
            lbvh_data_obstacle.face_tree.allocate(num_surface_faces_tet);
        }
        {
            const uint table_size = 1;
            tet_collision.table_size = table_size;
            tet_collision.hash_table.resize(table_size);
            tet_collision.hash_table_count.resize(table_size);
            tet_collision.hash_table_prefix.resize(table_size + 1);
            tet_collision.hash_table_belongs.resize(table_size);
            tet_collision.hash_table_flag.resize(table_size);
            tet_collision.hash_table_prefix[0] = 0;
            tet_collision.hash_table_vert_offset.resize(1);// num_surface_verts_tet

            tet_collision.vert_VV_num_broad_phase.resize(num_surface_verts_tet);
            tet_collision.broad_phase_list.resize(num_surface_verts_tet * B_selfcollision * 2);

            tet_collision.collision_count.resize(256);
            tet_collision.vert_VV_num_narrow_phase.resize(num_surface_verts_tet);
            tet_collision.vert_VV_prefix_narrow_phase.resize(num_surface_verts_tet + 1);
            tet_collision.narrow_phase_list_indices_vv.resize(num_surface_verts_tet * N_selfcollision);
            tet_collision.narrow_phase_list_pair_vv.resize(num_surface_verts_tet * N_selfcollision);
            tet_collision.narrow_phase_list_pair_vv_merged.resize(num_surface_verts_tet * N_selfcollision);
            tet_collision.narrow_phase_list_indices_vf.resize(1);
            tet_collision.narrow_phase_list_pair_vf.resize(1);

            tet_collision.max_vert_rest_distance.resize(num_surface_verts_tet);
            tet_collision.vert_adj_elements.resize(num_surface_verts_tet * (N_selfcollision + 1));
            tet_collision.collision_pair_offset_in_vert.resize(num_surface_verts_tet * N_selfcollision * vert_num_each_collision_pair_self_collision);// 1 for vv , 4 for vf
            tet_collision.self_collision_indirect_cmd_buffer.resize(4);
        }

        {
            const uint table_size = 1;
            obs_collision_tet.table_size = table_size;
            obs_collision_tet.hash_table.resize(table_size);
            obs_collision_tet.hash_table_count.resize(table_size);
            obs_collision_tet.hash_table_prefix.resize(table_size + 1);
            obs_collision_tet.hash_table_prefix[0] = 0;
            obs_collision_tet.hash_table_belongs.resize(table_size);
            obs_collision_tet.hash_table_flag.resize(table_size);
            obs_collision_tet.hash_table_vert_offset.resize(1);// num_verts_obstacle

            obs_collision_tet.broad_phase_list.resize(num_surface_verts_tet * B_obscollision * 2);

            obs_collision_tet.collision_count.resize(256);
            obs_collision_tet.narrow_phase_list_indices_vv.resize(1);
            obs_collision_tet.narrow_phase_list_pair_vv.resize(1);
            obs_collision_tet.narrow_phase_list_indices_vf.resize(num_surface_verts_tet * N_obscollision);
            obs_collision_tet.narrow_phase_list_pair_vf.resize(num_surface_verts_tet * N_obscollision);
            obs_collision_tet.vert_adj_elements.resize(num_surface_verts_tet * (N_obscollision + 1));
            obs_collision_tet.collision_pair_offset_in_vert.resize(num_surface_verts_tet * N_obscollision * vert_num_each_collision_pair_obs_collision);
            obs_collision_tet.obstacle_collision_indirect_cmd_buffer.resize(4);

            obs_collision_tet.vert_VV_num_broad_phase.resize(num_surface_verts_tet);
            obs_collision_tet.vert_VV_num_narrow_phase.resize(num_surface_verts_tet);
            obs_collision_tet.vert_VV_prefix_narrow_phase.resize(num_surface_verts_tet + 1);
        }
    }

#endif
};