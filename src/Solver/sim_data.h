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

#if !defined(METAL_CODE)
    std::vector<std::vector<uint>> vert_adj_faces;
    std::vector<std::vector<uint>> vert_adj_tets;
    std::vector<std::vector<uint>> vert_adj_verts;
#endif

    Array(uint)
        sa_vert_adj_faces;
    Array(uint)
        sa_vert_adj_verts_csr;
    Array(uint)
        sa_vert_adj_tets_csr;

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
        sa_rest_position;
    Array(Float3)
        sa_rest_velocity;
    Array(Float3)
        sa_surface_position;
    Array(Float3)
        sa_surface_rest_position;
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
    uint num_surface_verts_total;
    uint num_surface_edges_total;
    uint num_surface_faces_total;

    Array(uint)
        num_verts;
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
        prefix_surface_verts;
    Array(uint)
        prefix_surface_edges;
    Array(uint)
        prefix_surface_faces;

    TetData() {
        num_meshes = 0;
        num_verts_total = 0;
        num_tets_total = 0;
        num_surface_verts_total = 0;
        num_surface_edges_total = 0;
        num_surface_faces_total = 0;
    }
};