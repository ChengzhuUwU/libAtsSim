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

    Array(FlagType) sa_vert_mutex;

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
    void resize(TetData *tetrahedral, ObstacleData *obstacle);
#endif
};