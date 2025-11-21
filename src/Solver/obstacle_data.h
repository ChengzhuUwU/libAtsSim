#pragma once

///
/// Obstacle Data & Argument
/// ObstacleData Init At ClothInitialier
/// ObstacleArgs Form As Params In CPU & GPU Functions
///

#include "address_space.h"
#include "atomic.h"
#include "make_arguments.h"

#ifndef METAL_CODE
#include "shared_array.h"
#include "mesh_reader.h"
#endif

struct ObstacleData {

    // basic infomation
    Array(uchar) sa_obstacle_id;
    Array(Float4x4) sa_prev_model_matrix;
    Array(Float4x4) sa_model_matrix;
    Array(Float3) sa_vert_velocity;
    Array(Float3) sa_start_position;
    Array(Float3) sa_substep_position;
    Array(Float3) sa_next_position;
    Array(Float3) sa_model_position;
    Array(Float3) sa_rest_position;
    Array(Float3) sa_rest_velocity;
    Array(Float3) sa_rest_model_position;
    Array(Float2) sa_uv_position;
    Array(Int3) sa_faces;
    Array(Int2) sa_edges;
    Array(Int3) sa_face_adj_faces;
    Array(uint) sa_vert_adj_faces_csr;

    Array(Float3) sa_face_normal;
    Array(Float3) sa_vert_normal;
    Array(Float3) sa_edge_normal;
    Array(Float3) sa_face_normal_model_space;
    Array(Float3) sa_vert_normal_model_space;
    Array(Float3) sa_edge_normal_model_space;
    Array(float) sa_face_area;
    Array(float) sa_edge_area;
    Array(float) sa_vert_area;

    Array(Float3) m_translation;
    Array(Float3) m_rotation;
    Array(Float3) m_scale;

#ifndef METAL_CODE
    inline void save_solver_data(uint frame) {
        // save_to_binary(sa_is_boundary, "is_boundary_" + std::to_string(frame));
        // save_to_binary(sa_edges, "edges_" + std::to_string(frame));
    }
#endif

    uint num_obstacles;
    uint num_verts_total;
    uint num_edges_total;
    uint num_faces_total;

    ObstacleData() {
        num_obstacles = 0;
        num_verts_total = 0;
        num_edges_total = 0;
        num_faces_total = 0;
    }
};

struct ObstacleArgs {

    // basic infomation
    Pointer(uchar)
        sa_obstacle_id;
    Pointer(Float4x4)
        sa_model_matrix;
    Pointer(Float3)
        sa_start_position;
    Pointer(Float3)
        sa_rest_position;
    Pointer(Float3)
        sa_model_position;
    Pointer(Float2)
        sa_uv_position;
    Pointer(Int3)
        sa_faces;
    Pointer(Int2)
        sa_edges;
    Pointer(Int3)
        sa_face_adj_faces;

    Pointer(Float3)
        sa_face_normal;
    Pointer(Float3)
        sa_vert_normal;
    Pointer(Float3)
        sa_edge_normal;
    Pointer(Float3)
        sa_face_normal_model_space;
    Pointer(Float3)
        sa_vert_normal_model_space;
    Pointer(Float3)
        sa_edge_normal_model_space;
    Pointer(float)
        sa_face_area;
    Pointer(float)
        sa_edge_area;
    Pointer(float)
        sa_vert_area;

    Pointer(Float3)
        m_translation;
    Pointer(Float3)
        m_rotation;
    Pointer(Float3)
        m_scale;

    uint num_obstacles;
    uint num_verts_total;
    uint num_edges_total;
    uint num_faces_total;

#ifndef METAL_CODE
    template<PtrType ptr_type>
    void set(ObstacleData &obstacle) {
        sa_obstacle_id = get_ptr(obstacle.sa_obstacle_id, ptr_type);
        sa_model_matrix = get_ptr(obstacle.sa_model_matrix, ptr_type);
        sa_start_position = get_ptr(obstacle.sa_start_position, ptr_type);
        sa_rest_position = get_ptr(obstacle.sa_rest_position, ptr_type);
        sa_model_position = get_ptr(obstacle.sa_model_position, ptr_type);
        sa_uv_position = get_ptr(obstacle.sa_uv_position, ptr_type);
        sa_faces = get_ptr(obstacle.sa_faces, ptr_type);
        sa_edges = get_ptr(obstacle.sa_edges, ptr_type);
        sa_face_adj_faces = get_ptr(obstacle.sa_face_adj_faces, ptr_type);

        sa_face_normal = get_ptr(obstacle.sa_face_normal, ptr_type);
        sa_vert_normal = get_ptr(obstacle.sa_vert_normal, ptr_type);
        sa_edge_normal = get_ptr(obstacle.sa_edge_normal, ptr_type);
        sa_face_normal_model_space = get_ptr(obstacle.sa_face_normal_model_space, ptr_type);
        sa_vert_normal_model_space = get_ptr(obstacle.sa_vert_normal_model_space, ptr_type);
        sa_edge_normal_model_space = get_ptr(obstacle.sa_edge_normal_model_space, ptr_type);
        sa_face_area = get_ptr(obstacle.sa_face_area, ptr_type);
        sa_edge_area = get_ptr(obstacle.sa_edge_area, ptr_type);
        sa_vert_area = get_ptr(obstacle.sa_vert_area, ptr_type);

        m_translation = get_ptr(obstacle.m_translation, ptr_type);
        m_rotation = get_ptr(obstacle.m_rotation, ptr_type);
        m_scale = get_ptr(obstacle.m_scale, ptr_type);

        num_obstacles = obstacle.num_obstacles;
        num_verts_total = obstacle.num_verts_total;
        num_edges_total = obstacle.num_edges_total;
        num_faces_total = obstacle.num_faces_total;
    }
#endif
};
