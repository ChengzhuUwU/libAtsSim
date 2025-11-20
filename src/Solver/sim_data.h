#pragma once

///
/// Define The ClothData Structure
/// ClothData Will be Allocated In Initializer (cloth_initializer.cpp)
///

#include "address_space.h"
#include "atomic.h"

#include "make_arguments.h"

#ifndef METAL_CODE
#include "shared_array.h"
#include "mesh_reader.h"

#endif

struct TetData {
    /// faces means faces on the surface
    /// edges means edges on the surface
    Array(uchar)
        sa_is_fixed;
    Array(uchar)
        sa_is_fixed_copy;
    Array(uchar)
        sa_tet_is_boundary;
    Array(uchar)
        sa_tet_id;
    Array(uchar)
        sa_vert_is_boundary;// vert is boundary
    Array(uint)
        sa_surface_verts;
    Array(uint)
        sa_surface_verts_reverse_index;
    Array(Int4)
        sa_outer_tets;
    Array(Int4)
        sa_inner_tets;
    Array(uint)
        sa_outer_tets_indices;
    Array(uint)
        sa_inner_tets_indices;
    Array(Int2)
        sa_surface_edges;
    Array(Int3)
        sa_surface_faces;
    Array(Int4)
        sa_tets;
    Array(Float4)
        sa_vert_color;
    Array(Float4)
        sa_default_color;

    Array(uint)
        sa_vert_adj_faces;
    // Array(uint) sa_vert_adj_edges;
    // Array(uint) sa_vert_adj_verts;
    Array(uint)
        sa_vert_adj_verts_csr;
    // Array(Int2) sa_edge_adj_faces;
    Array(uint)
        sa_vert_adj_tets_csr;
    Array(uint)
        sa_vert_adj_inner_tets_csr;
    Array(uint)
        sa_vert_adj_outer_tets_csr;
    Array(uint)
        sa_vert_adj_tets_num;
    // Array(uint) sa_tet_adj_faces;
    // Array(uint) sa_face_adj_tets;

    Array(float)
        sa_vert_mass;
    Array(float)
        sa_vert_mass_inv;
    Array(float)
        sa_face_area;
    Array(float)
        sa_edge_area;
    Array(float)
        sa_vert_area;
    Array(float)
        sa_vert_volumn;
    Array(float)
        sa_tet_volumn;
    Array(Float3)
        sa_face_normal;
    Array(Float3)
        sa_vert_normal;

    Array(Float3)
        sa_model_position;
    Array(Float2)
        sa_meterial_position;// uv
    Array(Float3)
        sa_start_position;
    Array(Float3)
        sa_surface_position;
    Array(Float3)
        sa_surface_rest_position;
    Array(Float3)
        sa_iter_position;// x_{k}
    Array(Float3)
        sa_iter_start_position;// copy of x_{k} copy
    Array(Float3)
        sa_prev_1_iter_position;// x_{k - 1}
    Array(Float3)
        sa_prev_2_iter_position;// x_{k - 2}
    Array(Float3)
        sa_iter_position_copy_for_jacobi;
    Array(Float3)
        sa_next_position;
    Array(Float3)
        sa_rest_position;
    Array(Float3)
        sa_vert_force;
    Array(Float3)
        sa_start_velocity;
    Array(Float3)
        sa_vert_velocity_jacobi;
    Array(Float3)
        sa_vert_velocity;

    Array(Float4x4)
        sa_model_matrix;

    /// inner force infomation
    Array(Float3x3)
        sa_Dm;
    Array(Float3x3)
        sa_Dm_inv;

    /// mutex
    Array(FlagType)
        sa_vert_mutex;
    Array(FlagType)
        sa_face_mutex;
    Array(FlagType)
        sa_tet_mutex;

    Array(Float3)
        m_translation;
    Array(Float3)
        m_rotation;
    Array(Float3)
        m_scale;

    // Constant Data
    uint num_meshes;
    uint num_verts_total;
    uint num_tets_total;
    uint num_inner_tets_total;
    uint num_outer_tets_total;
    uint num_surface_verts_total;
    uint num_surface_edges_total;
    uint num_surface_faces_total;

    Array(uint)
        num_verts;
    Array(uint)
        num_inner_tets;
    Array(uint)
        num_outer_tets;
    Array(uint)
        num_surface_verts;
    Array(uint)
        num_surface_faces;
    Array(uint)
        num_surface_edges;
    Array(uint)
        num_tets;

    Array(uint)
        prefix_verts;
    Array(uint)
        prefix_tets;
    Array(uint)
        prefix_inner_tets;
    Array(uint)
        prefix_outer_tets;
    Array(uint)
        prefix_surface_verts;
    Array(uint)
        prefix_surface_edges;
    Array(uint)
        prefix_surface_faces;

    uint max_vert_adj_inner_tets;
    uint max_vert_adj_outer_tets;
    uint max_vert_adj_surface_faces;
    uint max_vert_adj_surface_edges;
    uint max_vert_adj_verts;
    uint max_vert_adj_tets;

    TetData() {
        num_meshes = 0;
        num_verts_total = 0;
        num_tets_total = 0;
        num_inner_tets_total = 0;
        num_outer_tets_total = 0;
        num_surface_verts_total = 0;
        num_surface_edges_total = 0;
        num_surface_faces_total = 0;

        max_vert_adj_inner_tets = 0;
        max_vert_adj_outer_tets = 0;
        max_vert_adj_surface_faces = 0;
        max_vert_adj_surface_edges = 0;
        max_vert_adj_verts = 0;
        max_vert_adj_tets = 0;
    }
};