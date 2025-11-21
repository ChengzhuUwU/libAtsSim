#pragma once

#include "address_space.h"
#include "atomic.h"

#include "make_arguments.h"
#include "xpbd_data.h"

#ifndef METAL_CODE
#include "shared_array.h"
#include <unordered_set>
#endif

enum VivaceGraphColoringElementType {
    VivaceGraphColoringElementTypePerVertVV,
    VivaceGraphColoringElementTypePerVertVF,
    VivaceGraphColoringElementTypePerPairVV,
    VivaceGraphColoringElementTypePerPairVF,
};

struct VivaceColoringData {
    // Vivace Graph-Coloring
    Array(Int2) block_min_max_degree;
    Array(uint) uncolored_verts;
    Array(Int4) uncolored_verts_indirect_cmd_buffer;
    Array(uint) uncolored_verts_count;

    Array(uchar) colored;// Uncolored
    Array(uchar) colored_in_curr_pass;
    Array(uchar) next_color;
    Array(uint64) P_v;// palette , use mask
    Array(uint64) P_v_prev;
    Array(uchar) c_v;
    Array(uchar) pre_computed_random_number_256;

    // Graph-Coloring Result
    Array(uint) clusterd_constraint_self_collision;// Adjacent Collision Vert
    Array(Int4) clusterd_constraint_self_collision_indirect_cmd_buffer;
    Array(uint) num_verts_in_cluster;
    Array(uint) cluster_prefix;
    Array(uint) verts_prefix_in_cluster;
    Array(uint) num_colors_self_collision;
    // Array(uint) collision_count;

    // Collision Data
    // SharedArray<Int4>* self_collision_indirect_cmd_buffer;
    // SharedArray<Int2>* collision_pair;
    // SharedArray<uint>* vert_adj_collsion_pair_num;
    // SharedArray<uint>* vert_adj_collsion_pair_prefix;
    // SharedArray<uint>* vert_adj_collsion_pair_list;

    VivaceGraphColoringElementType element_type = VivaceGraphColoringElementTypePerPairVV;

#ifndef METAL_CODE
    // NumVerts In Topology, Or Regard Each Collision Pair As Vertex
    void resize(const uint num_verts_total) {
        colored.resize(num_verts_total);// Uncolored
        colored_in_curr_pass.resize(num_verts_total);
        P_v.resize(num_verts_total);// palette , use mask
        P_v_prev.resize(num_verts_total);
        next_color.resize(num_verts_total);
        c_v.resize(num_verts_total);
        pre_computed_random_number_256.resize(num_verts_total);
        // pre_computed_random_number_256.resize(num_verts_total * 20);

        block_min_max_degree.resize(get_dispatch_num(num_verts_total, 256));
        uncolored_verts.resize(num_verts_total);
        uncolored_verts_indirect_cmd_buffer.resize(128);
        uncolored_verts_count.resize(128);

        num_colors_self_collision.resize(1);
        clusterd_constraint_self_collision.resize(num_verts_total);
        clusterd_constraint_self_collision_indirect_cmd_buffer.resize(128);
        num_verts_in_cluster.resize(128);
        cluster_prefix.resize(128);
        verts_prefix_in_cluster.resize(num_verts_total);
        // collision_count.resize(128);
    }
#endif
};

struct VivaceColoringArgs {
    uint allocation_size;

    // Vivace Graph-Coloring
    Pointer(Int2) block_min_max_degree;
    Pointer(uint) uncolored_verts;
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer;
    Pointer(uint) uncolored_verts_count;

    Pointer(uchar) colored;
    Pointer(uchar) colored_in_curr_pass;
    Pointer(uchar) next_color;
    Pointer(uint64) P_v;
    Pointer(uint64) P_v_prev;
    Pointer(uchar) c_v;
    Pointer(uchar) pre_computed_random_number_256;

    // Graph-Coloring Result
    Pointer(uint) clusterd_constraint_self_collision;// Adjacent Collision Vert
    Pointer(Int4) clusterd_constraint_self_collision_indirect_cmd_buffer;
    Pointer(uint) num_verts_in_cluster;
    Pointer(uint) cluster_prefix;
    Pointer(uint) verts_prefix_in_cluster;
    Pointer(uint) num_colors_self_collision_vv;
    // Pointer(uint) collision_count;

    // Collision Data
    Pointer(Int4) self_collision_indirect_cmd_buffer;
    Pointer(Int2) collision_pair_vv;
    Pointer(Int4) collision_pair_vf;
    Pointer(uint) vert_adj_collsion_pair_num;
    Pointer(uint) vert_adj_collsion_pair_prefix;
    Pointer(uint) vert_adj_collsion_pair_list;

    VivaceGraphColoringElementType element_type = VivaceGraphColoringElementTypePerPairVV;

#ifndef METAL_CODE
    template<PtrType ptr_type>
    void set(VivaceColoringData &data, XpbdSelfCollision &self_collision) {
        allocation_size = data.c_v.size();

        block_min_max_degree = get_ptr(data.block_min_max_degree, ptr_type);
        uncolored_verts = get_ptr(data.uncolored_verts, ptr_type);
        uncolored_verts_indirect_cmd_buffer = get_ptr(data.uncolored_verts_indirect_cmd_buffer, ptr_type);
        uncolored_verts_count = get_ptr(data.uncolored_verts_count, ptr_type);

        colored = get_ptr(data.colored, ptr_type);
        colored_in_curr_pass = get_ptr(data.colored_in_curr_pass, ptr_type);
        next_color = get_ptr(data.next_color, ptr_type);
        P_v = get_ptr(data.P_v, ptr_type);
        P_v_prev = get_ptr(data.P_v_prev, ptr_type);
        c_v = get_ptr(data.c_v, ptr_type);
        pre_computed_random_number_256 = get_ptr(data.pre_computed_random_number_256, ptr_type);

        clusterd_constraint_self_collision = get_ptr(data.clusterd_constraint_self_collision, ptr_type);
        clusterd_constraint_self_collision_indirect_cmd_buffer = get_ptr(data.clusterd_constraint_self_collision_indirect_cmd_buffer, ptr_type);
        num_verts_in_cluster = get_ptr(data.num_verts_in_cluster, ptr_type);
        cluster_prefix = get_ptr(data.cluster_prefix, ptr_type);
        verts_prefix_in_cluster = get_ptr(data.verts_prefix_in_cluster, ptr_type);
        num_colors_self_collision_vv = get_ptr(data.num_colors_self_collision, ptr_type);
        // collision_count = get_ptr(data.collision_count, ptr_type);

        self_collision_indirect_cmd_buffer = get_ptr(self_collision.self_collision_indirect_cmd_buffer, ptr_type);
        collision_pair_vv = get_ptr(self_collision.narrow_phase_list_indices_vv, ptr_type);
        collision_pair_vf = get_ptr(self_collision.narrow_phase_list_indices_vf, ptr_type);
        vert_adj_collsion_pair_num = get_ptr(self_collision.vert_VV_num_narrow_phase, ptr_type);
        vert_adj_collsion_pair_prefix = get_ptr(self_collision.vert_VV_prefix_narrow_phase, ptr_type);
        vert_adj_collsion_pair_list = get_ptr(self_collision.vert_adj_elements, ptr_type);
    }
#endif
};

namespace VivaceGraphCloring {

ConstExpr uint max_graph_coloring_iterations = 20;
ConstExpr uint max_graph_coloring_colors = 60;

inline uint fn_get_num_collision(Pointer(uint) collision_count) { return collision_count[0]; }
inline uint fn_get_num_not_collide(Pointer(uint) collision_count) { return collision_count[1]; }
inline uint fn_get_min_degree(Pointer(uint) collision_count) { return collision_count[2]; }
inline uint fn_get_max_degree(Pointer(uint) collision_count) { return collision_count[3]; }

inline void fn_set_num_collision(Pointer(uint) collision_count, const uint num_collision) { collision_count[0] = num_collision; }
inline uint fn_atomic_add_collision_count(Pointer(uint) collision_count) { return atomic_add(collision_count[0], 1); }
inline void fn_set_num_not_collide(Pointer(uint) collision_count, const uint num_not_collide) { collision_count[1] = num_not_collide; }
inline void fn_set_min_degree(Pointer(uint) collision_count, const uint min_degree) { collision_count[2] = min_degree; }
inline void fn_set_max_degree(Pointer(uint) collision_count, const uint max_degree) { collision_count[3] = max_degree; }
inline uint fn_atomic_add_numVerts_in_cell_in_block(Pointer(uint) collision_count, const uint num_verts_in_cell) { return atomic_add(collision_count[5], num_verts_in_cell); }
// collision_count[6] => For VF Pair Scan

inline uint fn_get_vid_from_uncolored_verts(const uint i, Pointer(uint) uncolored_verts) { return uncolored_verts[i]; };

inline uint fn_get_current_num_uncolored(Pointer(uint) uncolored_verts_count, const uint curr_loop) { return uncolored_verts_count[curr_loop]; }
inline void fn_set_current_num_uncolored(Pointer(uint) uncolored_verts_count, const uint curr_loop, const uint num_uncolored) { uncolored_verts_count[curr_loop] = num_uncolored; }
inline uint fn_atomic_push_into_current_uncolored_set(Pointer(uint) uncolored_verts_count, const uint curr_loop) { return atomic_add(uncolored_verts_count[curr_loop], 1); }

inline uint scan_uncolored_set_CPU_1(
    const uint i,
    Pointer(uint) uncolored_verts,
    Pointer(uchar) colored) {
    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);
    const bool uncolored = !colored[vid];
    return uncolored ? 1 : 0;
}
inline void scan_uncolored_set_CPU_2(
    const uint i, ConstRef(uint) scan_result, ConstRef(uint) self_result,
    Pointer(uint) uncolored_verts,
    Pointer(uint) uncolored_verts_copy,
    Pointer(uint) collision_count,
    Pointer(uint) uncolored_verts_count,
    const uint curr_loop) {
    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const bool uncolored = self_result == 1;
    if (uncolored) {
        uncolored_verts_copy[scan_result - 1] = vid;
        // vivace_data->uncolored_verts[scan_result - 1] = vid;
    }

    const uint prev_uncolored = curr_loop == 0 ? collision_count[0] : fn_get_current_num_uncolored(uncolored_verts_count, curr_loop - 1);
    if (i == prev_uncolored - 1)// Last Vert
    {
        const uint new_num_uncolored = scan_result;
        VivaceGraphCloring::fn_set_current_num_uncolored(uncolored_verts_count, curr_loop, new_num_uncolored);
    }
}
inline void scan_uncolored_set_GPU(
    const uint i,
    Pointer(uint) collision_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint) uncolored_verts_copy,
    Pointer(uint) uncolored_verts_count,
    Pointer(uchar) colored,
    const uint curr_loop) {
    const uint prev_uncolored = curr_loop == 0 ? collision_count[0] : fn_get_current_num_uncolored(uncolored_verts_count, curr_loop - 1);
    if (i >= prev_uncolored) return;
    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const bool uncolored = !colored[vid];
    if (uncolored) {
        uint idx = atomic_add(uncolored_verts_count[curr_loop], 1);
        uncolored_verts_copy[idx] = vid;
    }
}
inline void copy_scaned_indices_from(
    const uint i,
    Pointer(uint) collision_count,
    Pointer(uint) uncolored_verts_copy,
    Pointer(uint) uncolored_verts,
    Pointer(uint) uncolored_verts_count,
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored) return;
    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts_copy);

    uncolored_verts[i] = vid;

    if (i == 0) {
        const uint curr_uncolored = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
        uncolored_verts_indirect_cmd_buffer[curr_loop] = make_indirect_command_buffer(curr_uncolored);
    }
}
inline void make_uncolored_verts_indirect_command_buffer(
    Pointer(uint) uncolored_verts_count,
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    const uint curr_loop) {
    const uint curr_uncolored = VivaceGraphCloring::fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    uncolored_verts_indirect_cmd_buffer[curr_loop] = make_indirect_command_buffer(curr_uncolored);
}

inline Int2 reduce_degree_binary_function(ConstRef(Int2) left, ConstRef(Int2) right) {
    return makeInt2(min_scalar(left[0], right[0]), max_scalar(left[1], right[1]));
}

static inline void save_reduced_degree(const Int2 min_max_degree, Pointer(uint) collision_count) {
    fn_set_min_degree(collision_count, min_max_degree.x);
    fn_set_max_degree(collision_count, min_max_degree.y);
}
inline void set_max_color_from_max_degree(
    Pointer(uint) num_verts_in_cluster,
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    Pointer(uint) num_colors_self_collision_vv,
    Pointer(uint) collision_count,
    const Int2 min_max_degree) {
    const uint min_degree = min_max_degree.x;
    const uint max_degree = min_scalar(min_max_degree.y, max_graph_coloring_colors - 1);

    fn_set_min_degree(collision_count, min_degree);
    fn_set_max_degree(collision_count, max_degree);// should be 31 ???

    // const uint init_color = min_scalar(max_degree + 1, 12u);
    const uint init_color = max_degree / 2 + 1;
    // const uint init_color = max_degree + 1;
    num_colors_self_collision_vv[0] = init_color;

    // const uint max_color = min_scalar(
    // 	min_max_degree.y / max_scalar(min_max_degree.x, 1u),
    // 	31u) + 1;
    // num_colors_self_collision_vv[0] = max_color; // Max Colors = Max(Degree) + 1
}

inline void init_palette(
    const uint i,
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint) collision_count,
    Pointer(uint) num_colors_self_collision_vv,
    Pointer(uint64) P_v,
    Pointer(uint64) P_v_prev,
    Pointer(uchar) next_color) {
    const uint curr_uncolored_count = fn_get_current_num_uncolored(uncolored_verts_count, 0);
    if (i >= curr_uncolored_count) { return; }
    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const uint max_degree = fn_get_max_degree(collision_count);

    // const uint init_color = num_colors_self_collision_vv[0];
    const uint64 initial_palatte = make_lane_mask_64(max_degree + 1);
    P_v[vid] = initial_palatte;
    P_v_prev[vid] = 0;
    // . The maximal amount of colors allowed is ∆v + 1, but in our experiments we never reached this maximal threshold
    // next_color[vid] = init_color;
}

#ifndef METAL_CODE
inline void set_random_value_256(const uint i, Pointer(uchar) pre_computed_random_number_256) {
    // std::srand(i);
    // uchar random_value = std::rand() % 256;

    std::mt19937 generator(i);
    std::uniform_int_distribution<int> distribution(0, 255);
    uchar random_value = distribution(generator);
    pre_computed_random_number_256[i] = random_value;
}
#endif
inline uint get_random_value_256(const uint vid, const uint curr_loop, const uint num_verts_total, Pointer(uchar) pre_computed_random_number_256) {
    // return pre_computed_random_number_256[(curr_loop % max_graph_coloring_iterations) * num_verts_total + vid];
    return pre_computed_random_number_256[(7 * curr_loop + vid) % num_verts_total];
}

inline void tentative_coloring(
    const uint i,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint) num_colors_self_collision_vv,
    Pointer(uint64) P_v,
    Pointer(uint64) P_v_prev,
    Pointer(uchar) next_color,
    Pointer(uchar) c_v,
    Pointer(uchar) pre_computed_random_number_256,
    Pointer(uint) collision_count,
    const uint curr_loop) {
    const uint curr_uncolored_count = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored_count) { return; }
    const uint vid = fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    uint curr_color;

    // Get random color from palatte
    // const uint valid_color_count = next_color[vid];
    const uint valid_color_count = num_colors_self_collision_vv[0];
    auto Pv = P_v[vid] & make_lane_mask_64(valid_color_count);

    // Palatte Filering
    auto prev_Pv = P_v_prev[vid];
    if ((Pv & ~prev_Pv) == 0) { prev_Pv = 0; }
    Pv &= ~prev_Pv;

    // if (curr_loop != 0)
    // {
    // 	uint prev_color = c_v[vid];
    // 	Pv &= ~(1ul << prev_color);
    // }

    const uint num_P_v = popc_uint64(Pv);

    const uint num_verts_total = collision_count[0];
    uint random_idx = get_random_value_256(vid, curr_loop, num_verts_total, pre_computed_random_number_256) % num_P_v;

    {
        auto mask = Pv;
        for (uint j = 0; j < random_idx; j++)// Drop Bits In Right Than random_idx
            ffs_and_pop64(mask);
        curr_color = ffs_uint64(mask) - 1;
    }

    prev_Pv |= (1ul << curr_color);
    P_v_prev[vid] = prev_Pv;

    c_v[vid] = curr_color;
}

inline void feed_the_hungry(
    const uint i,
    Pointer(uint) uncolored_verts,
    Pointer(uint64) P_v,
    Pointer(uint) collision_count,
    Pointer(uchar) next_color,
    Pointer(Int4) uncolored_verts_indirect_cmd_buffer,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) num_colors_self_collision_vv,
    const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored) { return; }
    const uint vid = fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    // uint curr_next_color = next_color[vid];
    uint curr_next_color = num_colors_self_collision_vv[0];
    const auto Pv = P_v[vid] & make_lane_mask_64(curr_next_color);
    if (Pv == 0)// popc_uint(Pv) == 0
    {
        uint max_degree = VivaceGraphCloring::fn_get_max_degree(collision_count);
        curr_next_color = min_scalar(curr_next_color, max_degree);

        // . The maximal amount of colors allowed is ∆v + 1, but in our experiments we never reached this maximal threshold

        // P_v[vid] = (1 << curr_next_color);
        next_color[vid] = curr_next_color + 1;
        if (curr_next_color + 1 > num_colors_self_collision_vv[0]) {
            num_colors_self_collision_vv[0] = curr_next_color + 1;
        }
    }
}

inline void put_rest_vertices_into_additional_color(
    const uint i,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint64) P_v,
    Pointer(uint) collision_count,
    Pointer(uchar) pre_computed_random_number_256,
    Pointer(uint) num_colors_self_collision_vv,
    Pointer(uint) num_verts_in_cluster,
    Pointer(uint) clusterd_constraint_self_collision_vv,
    const uint curr_loop,
    const uint num_verts_total) {
    const uint curr_uncolored = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored) { return; }
    const uint vid = fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const uint max_degree = fn_get_max_degree(collision_count);
    const uint curr_color = max_degree;
    uint idx = atomic_add(num_verts_in_cluster[curr_color], 1);
    clusterd_constraint_self_collision_vv[curr_color * num_verts_total + idx] = vid;
}

inline void put_rest_vertices_into_random_color(
    const uint i,
    Pointer(uint) uncolored_verts_count,
    Pointer(uint) uncolored_verts,
    Pointer(uint64) P_v,
    Pointer(uint) collision_count,
    Pointer(uchar) pre_computed_random_number_256,
    Pointer(uint) num_colors_self_collision_vv,
    Pointer(uint) num_verts_in_cluster,
    Pointer(uint) clusterd_constraint_self_collision_vv,
    const uint curr_loop) {
    const uint curr_uncolored = fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored) { return; }
    const uint vid = fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const auto Pv = P_v[vid];
    const uint num_P_v = popc_uint64(Pv);

    const uint num_verts_total = collision_count[0];
    uint random_idx = get_random_value_256(vid, 19, num_verts_total, pre_computed_random_number_256) % num_P_v;

    uint curr_color;
    {
        uint mask = Pv;
        for (uint j = 0; j < random_idx; j++)
            ffs_and_pop(mask);
        curr_color = ffs_uint(mask) - 1;
    }
    uint offset = atomic_add(num_verts_in_cluster[curr_color], 1);
    clusterd_constraint_self_collision_vv[vid] = offset;
    // clusterd_constraint_self_collision_vv[curr_color * num_verts_total + idx] = vid;
}

inline void make_cluster_indirect_cmd_buffer(
    const uint i,
    Pointer(uint) num_verts_in_cluster,
    Pointer(Int4) clusterd_constraint_self_collision_vv_indirect_cmd_buffer) {
    if (i < max_graph_coloring_colors) {
        const uint color = i;
        const uint num_verts = num_verts_in_cluster[color];
        clusterd_constraint_self_collision_vv_indirect_cmd_buffer[i] = make_indirect_command_buffer(num_verts);
    }
}
// template <typename T>
inline void fill_in_cluster_indices(
    const uint element_id,
    Pointer(uint) verts_prefix_in_cluster,
    Pointer(uchar) c_v,
    Pointer(uint) cluster_prefix,
    Pointer(uint) clusterd_constraint_self_collision,

    Pointer(ProximityVV) narrow_phase_list_pair_vv,
    Pointer(ProximityVV) narrow_phase_list_pair_vv_merged) {
    const uint offset = verts_prefix_in_cluster[element_id];
    const uint curr_color = c_v[element_id];
    const uint prefix = cluster_prefix[curr_color];
    const uint index = prefix + offset;
    clusterd_constraint_self_collision[index] = element_id;

    const auto pair = narrow_phase_list_pair_vv[element_id];
    narrow_phase_list_pair_vv_merged[index] = pair;
}

//
// Need To Connect With Collision System
//

namespace PairMeta {

template<typename T>
struct CollisionPairType;

template<>
struct CollisionPairType<Int2> {
    typedef Int2 IndicesType;
};
template<>
struct CollisionPairType<Int4> {
    typedef Int4 IndicesType;
};
template<>
struct CollisionPairType<ProximityVV> {
    typedef Int2 IndicesType;
};
template<>
struct CollisionPairType<ProximityVF> {
    typedef Int4 IndicesType;
};

template<typename T>
using get_indices_type = typename CollisionPairType<T>::IndicesType;

template<typename T, typename TT>
inline TT get_indices_func(ConstRef(T) value);

template<>
inline Int2 get_indices_func<ProximityVV>(ConstRef(ProximityVV) value) { return value.get_indices(); }
template<>
inline Int4 get_indices_func<ProximityVF>(ConstRef(ProximityVF) value) { return value.get_indices(); }
template<>
inline Int2 get_indices_func<Int2>(ConstRef(Int2) value) { return value; }
template<>
inline Int4 get_indices_func<Int4>(ConstRef(Int4) value) { return value; }

}// namespace PairMeta

template<typename T, typename TT = PairMeta::get_indices_type<T>, uint N = Meta::get_vec_length<TT>()>
inline uint reduce_degree_and_set_zero_degree_nodes_template(
    const uint element_id,
    Pointer(T) collision_pair,
    Pointer(uint) vert_adj_collsion_pair_num,
    Pointer(uint) uncolored_verts,
    Pointer(uint) num_verts_in_cluster,
    Pointer(uint) verts_prefix_in_cluster,
    Pointer(uchar) colored,
    Pointer(uchar) c_v) {
    const T vv_pair = collision_pair[element_id];
    const TT indices = PairMeta::get_indices_func<T, TT>(vv_pair);
    uncolored_verts[element_id] = element_id;// Init Index

    // Access All Verts' Adjacent Num In Pair - 1 (Current Pair)
    uint curr_degree = 0;
    for (uint j = 0; j < N; j++) { curr_degree += (vert_adj_collsion_pair_num[indices[j]] - 1); }

    if (curr_degree == 0) {
        colored[element_id] = true;

        const uint default_color = 0;
        c_v[element_id] = default_color;
        uint offset = atomic_add(num_verts_in_cluster[default_color], 1);
        verts_prefix_in_cluster[element_id] = offset;
    } else {
        colored[element_id] = false;
    }

    return curr_degree;
}

inline void remove_color_from_adj_palatte(GLOBAL uint64 &adj_palette, const uint curr_selected_color) {
    adj_palette &= ~(1ul << curr_selected_color);
    // GLOBAL uint64* adj_palette_ptr = &adj_palette;
    // GLOBAL uint* adj_palette_left = ((GLOBAL uint*)adj_palette_ptr);
    // GLOBAL uint* adj_palette_right = ((GLOBAL uint*)adj_palette_ptr) + 1;
    // if (curr_selected_color < 32)
    //     (*adj_palette_right &= ~(1 << curr_selected_color));
    //     // atomic_and(adj_palette_right, ~(1 << curr_selected_color));
    // else
    //     (*adj_palette_left &= ~(1 << (curr_selected_color - 32)));
    //     // atomic_and(adj_palette_left, ~(1 << (curr_selected_color - 32)));
}

template<typename T, typename TT = PairMeta::get_indices_type<T>, uint N = Meta::get_vec_length<TT>()>
inline void conflict_resolution_PerConstraint_template(const uint i,
                                                       Pointer(T) collision_pair,
                                                       Pointer(uint) vert_adj_collsion_pair_num,
                                                       Pointer(uint) vert_adj_collsion_pair_prefix,
                                                       Pointer(uint) vert_adj_collsion_pair_list,

                                                       Pointer(uint) uncolored_verts_count,
                                                       Pointer(uint) uncolored_verts,

                                                       Pointer(uint64) P_v,
                                                       Pointer(uchar) c_v,
                                                       Pointer(uchar) colored,
                                                       Pointer(uchar) colored_in_curr_pass,

                                                       Pointer(uint) verts_prefix_in_cluster,
                                                       Pointer(uint) num_verts_in_cluster,
                                                       const uint curr_loop) {
    const uint curr_uncolored_count = VivaceGraphCloring::fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored_count) { return; }
    const uint element_id = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const T pair_vv = collision_pair[element_id];
    const TT indices = PairMeta::get_indices_func<T, TT>(pair_vv);

    const uchar curr_selected_color = c_v[element_id];

    bool cv_not_in_S = true;

    for (uint j = 0; j < N; j++) {
        const uint vert = indices[j];
        const uint num_adj = vert_adj_collsion_pair_num[vert];
        const uint start_idx = vert_adj_collsion_pair_prefix[vert];

        for (uint jj = 0; jj < num_adj; jj++) {
            const uint adj_pair_idx = vert_adj_collsion_pair_list[start_idx + jj];
            if (adj_pair_idx != element_id) {
                const uint adj_selected_color = c_v[adj_pair_idx];
                const bool adj_is_colored = colored[adj_pair_idx];
                if (curr_selected_color == adj_selected_color && ((adj_is_colored) || (!adj_is_colored && element_id < adj_pair_idx))) {
                    // Each vertex checks that none of its neighbors has selected the same tentative color
                    //     Using Hungarian heuristic [Luby 1985]:
                    //           In case of conflict, if the node has the higher index among its neighbors then the coloring is considered legitimate
                    cv_not_in_S = false;
                }
            }
        }
    }

    if (cv_not_in_S) {
        colored_in_curr_pass[element_id] = true;
        uint prefix = atomic_add(num_verts_in_cluster[curr_selected_color], 1);
        verts_prefix_in_cluster[element_id] = prefix;
    }

    // Remove Color Based On Current Pass Coloring In The Next Pass

    // if (cv_not_in_S ) // || has_higher_index_than_neighbors)
    // {
    // 	// The coloring of v is accepted and c(v) is removed from the palette of the neighbo
    // 	colored[element_id] = true;

    // 	uint prefix = atomic_add(num_verts_in_cluster[curr_selected_color], 1);
    // 	clusterd_constraint_self_collision[element_id] = prefix;

    // 	for (uint j = 0; j < N; j++)
    // 	{
    // 		const uint vert = indices[j];
    // 		const uint num_adj = vert_adj_collsion_pair_num[vert];
    // 		const uint start_idx = vert_adj_collsion_pair_prefix[vert];

    // 		for (uint jj = 0; jj < num_adj; jj++)
    // 		{
    // 			const uint adj_pair_idx = vert_adj_collsion_pair_list[start_idx + jj];
    // 			if (adj_pair_idx != element_id)
    // 			{
    // 				// c(v) is removed from the palette of the neighbor
    // 				remove_color_from_adj_palatte(P_v[adj_pair_idx], curr_selected_color);
    // 				// GLOBAL uint64* adj_palette = P_v[adj_pair_idx];
    // 				// atomic_and(adj_palette, ~(1 << curr_selected_color));
    // 			}
    // 		}
    // 	}
    // }
}

template<typename T, typename TT = PairMeta::get_indices_type<T>, uint N = Meta::get_vec_length<TT>()>
inline void update_palatte_from_current_tentative_coloring_result_template(const uint i,
                                                                           Pointer(T) collision_pair,
                                                                           Pointer(uint) vert_adj_collsion_pair_num,
                                                                           Pointer(uint) vert_adj_collsion_pair_prefix,
                                                                           Pointer(uint) vert_adj_collsion_pair_list,

                                                                           Pointer(uint) uncolored_verts_count,
                                                                           Pointer(uint) uncolored_verts,

                                                                           Pointer(uint64) P_v,
                                                                           Pointer(uchar) c_v,
                                                                           Pointer(uchar) colored,
                                                                           Pointer(uchar) colored_in_curr_pass,

                                                                           Pointer(uint) clusterd_constraint_self_collision,
                                                                           Pointer(uint) num_verts_in_cluster,
                                                                           const uint curr_loop) {
    const uint curr_uncolored_count = VivaceGraphCloring::fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored_count) { return; }
    const uint element_id = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const bool curr_is_colored_in_current_pass = colored_in_curr_pass[element_id];
    if (curr_is_colored_in_current_pass) {
        colored[element_id] = true;
    }

    const T pair_vv = collision_pair[element_id];
    const TT indices = PairMeta::get_indices_func<T, TT>(pair_vv);
    auto Pv = P_v[element_id];

    for (uint j = 0; j < N; j++) {
        const uint vert = indices[j];
        const uint num_adj = vert_adj_collsion_pair_num[vert];
        const uint start_idx = vert_adj_collsion_pair_prefix[vert];

        for (uint jj = 0; jj < num_adj; jj++) {
            const uint adj_pair_idx = vert_adj_collsion_pair_list[start_idx + jj];

            if (adj_pair_idx != element_id) {
                const bool adj_is_colored_in_current_pass = colored_in_curr_pass[adj_pair_idx];
                if (adj_is_colored_in_current_pass) {
                    // c(v) is removed from the palette of the neighbor
                    const uint adj_selected_color = c_v[adj_pair_idx];
                    Pv &= ~(1ul << adj_selected_color);
                    // remove_color_from_adj_palatte(P_v[adj_pair_idx], adj_selected_color);
                }
            }
        }
    }

    P_v[element_id] = Pv;
};

template<typename T, typename TT = PairMeta::get_indices_type<T>, uint N = Meta::get_vec_length<TT>()>
inline void conflict_resolution_PerVert_template(const uint i,
                                                 Pointer(T) collision_pair,
                                                 Pointer(uint) vert_adj_collsion_pair_num,
                                                 Pointer(uint) vert_adj_collsion_pair_prefix,
                                                 Pointer(uint) vert_adj_collsion_pair_list,

                                                 Pointer(uint) uncolored_verts_count,
                                                 Pointer(uint) uncolored_verts,
                                                 Pointer(uint64) P_v,
                                                 Pointer(uchar) c_v,
                                                 Pointer(uchar) colored,
                                                 Pointer(uchar) colored_in_curr_pass,
                                                 Pointer(uint) clusterd_constraint_self_collision,
                                                 Pointer(uint) num_verts_in_cluster,
                                                 const uint curr_loop) {
    const uint curr_uncolored_count = VivaceGraphCloring::fn_get_current_num_uncolored(uncolored_verts_count, curr_loop);
    if (i >= curr_uncolored_count) { return; }

    const uint vid = VivaceGraphCloring::fn_get_vid_from_uncolored_verts(i, uncolored_verts);

    const uchar curr_selected_color = c_v[vid];

    bool cv_not_in_S = true;

    const uint num_adj = vert_adj_collsion_pair_num[vid];
    const uint start_idx = vert_adj_collsion_pair_prefix[vid];

    for (uint jj = 0; jj < num_adj; jj++) {
        const uint adj_pair_idx = vert_adj_collsion_pair_list[start_idx + jj];
        const T adj_pair = collision_pair[adj_pair_idx];
        const TT indices = PairMeta::get_indices_func<T, TT>(adj_pair);

        for (uint j = 0; j < N; j++) {
            const uint adj_vid = indices[j];
            const uint adj_selected_color = c_v[adj_vid];
            const bool adj_is_colored = colored_in_curr_pass[adj_vid];
            if (curr_selected_color == adj_selected_color && ((adj_is_colored) || (!adj_is_colored && vid < adj_vid))) {
                // Each vertex checks that none of its neighbors has selected the same tentative color
                //     Using Hungarian heuristic [Luby 1985]:
                //           In case of conflict, if the node has the higher index among its neighbors then the coloring is considered legitimate
                cv_not_in_S = false;
            }
        }
    }

    if (cv_not_in_S)// || has_higher_index_than_neighbors)
    {
        // The coloring of v is accepted and c(v) is removed from the palette of the neighbo
        colored[vid] = true;

        uint prefix = atomic_add(num_verts_in_cluster[curr_selected_color], 1);
        clusterd_constraint_self_collision[vid] = prefix;

        for (uint jj = 0; jj < num_adj; jj++) {
            const uint adj_pair_idx = vert_adj_collsion_pair_list[start_idx + jj];
            const T adj_pair = collision_pair[adj_pair_idx];
            const TT indices = PairMeta::get_indices_func<T, TT>(adj_pair);

            for (uint j = 0; j < N; j++) {
                const uint adj_vid = indices[j];
                if (adj_vid != vid) {
                    // c(v) is removed from the palette of the neighbo
                    remove_color_from_adj_palatte(P_v[adj_pair_idx], curr_selected_color);
                    // GLOBAL uint64* adj_palette = P_v[adj_pair_idx];
                    // atomic_and(adj_palette, ~(1 << curr_selected_color));
                }
            }
        }
    }
}

}// namespace VivaceGraphCloring