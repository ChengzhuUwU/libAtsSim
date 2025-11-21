#pragma once

#include "aabb.h"
#include "address_space.h"
#include "atomic.h"
#include "morton.h"

#ifndef METAL_CODE
#include "shared_array.h"
#endif

ConstExpr uint mask_is_negative = 1u << 31;

struct ProximityVV {
    inline uint get_vert1() const { return indeces[0]; }
    inline uint get_vert2() const { return indeces[1]; }
    inline Float2 get_weight() const { return makeFloat2(-1, 1); }
    inline float get_weight(uint idx) const { return idx == 0 ? -1 : 1; }

    // normal : Normal of v2, Which Is The Same Direction To Push v1
    inline Float3 get_normal() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const {
        Float3 t = get_normal();
        return outer_product(t, t);
    }
    inline float get_stiff() const { return vec2[3]; }
    inline Int2 get_indices() const { return makeInt2(indeces[0], indeces[1]); }

    // Self Collision Is Always Positive (You Dont Know The Correct Direction ~ , Just Fix It In E-F Pair Or CCD-Based Response)
    template<uint NV>
    inline void get_indices(Thread uint tet[NV]) const {
        static_assert(NV == 1 || NV == 2, "Wrong NumVerts In Collision Pair, Should Be 1 or 2");
        for (uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative;
    }

    inline void set_negative() {
        for (uint i = 0; i < 2; i++) indeces[i] |= mask_is_negative;
    }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int2 get_indices_negative() const {
        Int2 tmp;
        for (uint i = 0; i < 2; i++) { tmp[i] = indeces[i] & ~mask_is_negative; }
        return tmp;
    }
    inline uint get_vert1_negative() const { return indeces[0] & ~mask_is_negative; }
    inline uint get_vert2_negative() const { return indeces[1] & ~mask_is_negative; }

    ProximityVV() {}
    ProximityVV(ConstRef(ProximityVV) right) : indeces(right.indeces), vec2(right.vec2) {}

#ifdef METAL_CODE
    ProximityVV(GLOBAL const ProximityVV &right) : indeces(right.indeces), vec2(right.vec2) {}
#endif

    // 'normal' Is The Normal of vid2, Because We Want That 'project(v0 - v1, normal)' Is Their Distance
    // (a(x) - b(x), In Paper Small Steps)
    // Or 'normal' Is The Direction To Push vid1 Away
    ProximityVV(ConstRef(uint) vid1, ConstRef(uint) vid2, ConstRef(float) stiffness, ConstRef(float) area, ConstRef(Float3) normal) {
        indeces = makeInt4(vid1, vid2, 0, 0);
        vec2 = makeFloat4(normal[0], normal[1], normal[2], stiffness * area);
    }

protected:
    Int4 indeces;
    Float4 vec2;// normal(3), stiff(1) = stiffness * area
};

struct ProximityVF {
    inline uint get_vert() const { return indeces[0]; }
    inline Int3 get_face() const { return makeInt3(indeces[0], indeces[1], indeces[2]); }
    inline Float4 get_weight() const { return vec1; }
    inline float get_weight(uint idx) const { return get_weight()[idx]; }
    inline float get_vert_weight(uint idx) const { return 1.0f; }
    inline Float3 get_face_weight() const { return makeFloat3(vec1[1], vec1[2], vec1[3]); }

    template<uint NV>
    inline void get_weights(Thread float weight[NV]) const {
        static_assert(NV == 1 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 1 or 4");
        for (uint i = 0; i < NV; i++) weight[i] = vec1[i];
    }
    inline Float3 get_t() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3 get_normal() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const {
        Float3 t = get_t();
        return outer_product(t, t);
    }
    inline float get_area() const { return vec2[3]; }
    inline float get_stiff() const { return vec2[3]; }
    inline Int4 get_indices() const { return indeces; }

    // Self Collision Is Always Positive (You Dont Know The Correct Direction ~ , Just Fix It In E-F Pair Or CCD-Based Response)
    template<uint NV>
    inline void get_indices(Thread uint tet[NV]) const {
        static_assert(NV == 1 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 1 or 4");
        for (uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative;
    }

    inline void set_negative() {
        for (uint i = 0; i < 4; i++) indeces[i] |= mask_is_negative;
    }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int4 get_indices_negative() const {
        Int4 tmp;
        for (uint i = 0; i < 4; i++) { tmp[i] = indeces[i] & ~mask_is_negative; }
        return tmp;
    }
    inline uint get_vert_negative() const { return indeces[0] & ~mask_is_negative; }

    ProximityVF() {}
    ProximityVF(ConstRef(uint) vid, ConstRef(Int3) f_vid, ConstRef(Float4) weight, ConstRef(float) area, ConstRef(Float3) t) {
        indeces = makeInt4(vid, f_vid[0], f_vid[1], f_vid[2]);
        vec1 = weight;
        vec2 = makeFloat4(t[0], t[1], t[2], area);
    }
    ProximityVF(ConstRef(uint) vid, ConstRef(Int3) f_vid, ConstRef(Float4) weight, ConstRef(float) stiffness, ConstRef(float) area, ConstRef(Float3) normal) {
        indeces = makeInt4(vid, f_vid[0], f_vid[1], f_vid[2]);
        vec1 = weight;
        vec2 = makeFloat4(normal[0], normal[1], normal[2], stiffness * area);
    }

private:
    Int4 indeces;
    Float4 vec1;// weight = (1, -bary[0], -bary[1], -bary[2])
    Float4 vec2;// t(3), area(1)
};

struct ProximityEE {
    inline Int2 get_edge1() const { return make<Int2>(indeces[0], indeces[1]); }
    inline Int2 get_edge2() const { return make<Int2>(indeces[2], indeces[3]); }
    inline Int4 get_indices() const { return indeces; }

    template<uint NV>
    inline void get_indices(Thread uint tet[NV]) const {
        static_assert(NV == 2 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 2 or 4");
        for (uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative;
    }

    inline Float4 get_weight() const { return vec1; }
    inline float get_weight(uint idx) const { return vec1[idx]; }

    template<uint NV>
    inline void get_weights(Thread float weight[NV]) const {
        static_assert(NV == 2 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 2 or 4");
        for (uint i = 0; i < NV; i++) weight[i] = vec1[i];
    }

    inline float get_area() const { return vec2[3]; }
    inline Float3 get_t() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const {
        Float3 t = get_t();
        return outer_product(t, t);
    }
    inline Float2 get_a() const { return make<Float2>(-vec1[0], -vec1[1]); }
    inline Float2 get_b() const { return make<Float2>(vec1[2], vec1[3]); }

    inline void set_negative() {
        for (uint i = 0; i < 4; i++) indeces[i] |= mask_is_negative;
    }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int4 get_indices_negative() const {
        Int4 tmp;
        for (uint i = 0; i < 4; i++) { tmp[i] = indeces[i] & ~mask_is_negative; }
        return tmp;
    }
    // inline Int2 get_edge1_negative()   const { return makeInt2(indeces[0] & ~mask_is_negative, indeces[1] & ~mask_is_negative); }

    ProximityEE() {}
    ProximityEE(ConstRef(Int2) edge1, ConstRef(Int2) edge2, ConstRef(Float4) weight, ConstRef(float) area, ConstRef(Float3) t) {
        indeces = make<Int4>(edge1[0], edge1[1], edge2[0], edge2[1]);
        vec1 = weight;
        vec2 = make<Float4>(t[0], t[1], t[2], area);
    }

private:
    Int4 indeces;
    Float4 vec1;// weight = (-a[0], -a[1], b[0], b[1])
    Float4 vec2;// t(3), area(1)
};

struct ProximityEF {
    inline Int2 get_edge() const { return make<Int2>(indeces[0], indeces[1]); }
    inline Int3 get_face() const { return makeInt3(indeces[2], indeces[3], indeces[4]); }

    template<uint NV>
    inline void get_indices(Thread uint tet[NV]) const {
        static_assert(NV == 2 || NV == 5, "Wrong NumVerts In Collision Pair, Should Be 2 or 5");
        for (uint i = 0; i < NV; i++) tet[i] = indeces[i];
    }
    template<uint NV>
    inline void get_weights(Thread float weight[NV]) const {
        static_assert(NV == 2 || NV == 5, "Wrong NumVerts In Collision Pair, Should Be 2 or 5");
        weight[0] = vec1.x;
        weight[1] = 1.f - vec1.x;
        if (NV == 5) {
            weight[2] = vec1.y;
            weight[3] = vec1.z;
            weight[4] = 1.f - vec1.y - vec1.z;
        }
    }
    inline float get_weight(uint idx) const {
        switch (idx) {
            case 0:
                return vec1.x;
            case 1:
                return 1.f - vec1.x;
            case 2:
                return vec1.y;
            case 3:
                return vec1.z;
            case 4:
                return 1.f - vec1.y - vec1.z;
            default:
                return 0.f;
        }
    }
    inline float get_area() const { return vec2[3]; }
    inline Float3 get_G() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const {
        Float3 t = get_G();
        return outer_product(t, t);
    }

    ProximityEF() {}
    ProximityEF(const uint tet[5], ConstRef(Float3) weight, ConstRef(float) area, ConstRef(Float3) G) {
        indeces[0] = tet[0];
        indeces[1] = tet[1];
        indeces[2] = tet[2];
        indeces[3] = tet[3];
        indeces[4] = tet[4];
        vec1 = weight;
        vec2 = make<Float4>(G[0], G[1], G[2], area);
    }

private:
    uint indeces[5];
    Float3 vec1;// (x, y, z) => weight = (x, 1-x, y, z, 1-y-z)
    Float4 vec2;// t(3), area(1)
};

enum LBVHTreeType {
    LBVHTreeTypeVert,
    LBVHTreeTypeFace,
    LBVHTreeTypeEdge
};
enum LBVHUpdateType {
    LBVHUpdateTypeCloth,
    LBVHUpdateTypeObstacle
};

struct LbvhData {

    Array(Float3)
        sa_leaf_center;
    // Array(AABB) sa_leaf_aabb;
    Array(AABB)
        sa_block_aabb;
    Array(Morton)
        sa_morton;
    Array(Morton)
        sa_morton_sorted;
    Array(uint)
        sa_sorted_get_original;

    Array(uint)
        sa_parrent;
    Array(Int2)
        sa_children;
    Array(uint)
        sa_object_idx;
    Array(AABB)
        sa_node_aabb;
    Array(bool)
        sa_is_healthy;
    Array(ATOMIC_UINT)
        sa_apply_flag;
    Array(FlagType)
        sa_node_mutex;

    Array(AABB)
        sa_node_aabb_model_position;
    Array(uchar)
        sa_node_object_id;
    Array(uchar)
        sa_sub_tree_refit_order;

    uint num_leaves;
    uint num_nodes;
    uint num_inner_nodes;
    LBVHTreeType tree_type;
    LBVHUpdateType update_type;

    // bool is_tree_healthy() { return sa_is_healthy[0]; }

#ifndef METAL_CODE
    void allocate(uint input_num) {
        num_leaves = input_num;
        num_inner_nodes = num_leaves - 1;
        num_nodes = num_leaves + num_inner_nodes;

        sa_leaf_center.resize(num_leaves);
        // sa_leaf_aabb.resize(num_leaves);
        sa_block_aabb.resize(get_dispatch_num(num_leaves, 256));
        sa_morton.resize(num_leaves);
        sa_morton_sorted.resize(num_leaves);
        sa_sorted_get_original.resize(num_leaves);

        sa_parrent.resize(num_nodes);
        sa_children.resize(num_nodes);
        sa_object_idx.resize(num_nodes);
        sa_node_aabb.resize(num_nodes);
        sa_apply_flag.resize(num_nodes);
        sa_node_mutex.resize(num_nodes);

        if (update_type == LBVHUpdateTypeObstacle) {
            sa_node_aabb_model_position.resize(num_nodes);
            sa_node_object_id.resize(num_nodes);
            sa_sub_tree_refit_order.resize(255);
        }

        sa_is_healthy.resize(1);
    }
#endif
};

struct LbvhFaceEdgeData {
    LbvhData vert_tree;
    LbvhData face_tree;
    LbvhData edge_tree;
};

struct CollisionList {
    struct CollisionListBroadPhase {
        Array(uint)
            list_vf;
        Array(uint)
            list_ee;
    };

    struct CollisionListNarrowPhase {

        Array(Int4)
            sa_indirect_command_buffer;/// [vert, VF, EE, EF]
        Array(uint)
            total_collision_num;/// [VF, EE, EF]

        Array(uint)
            active_vert_prefix;
        Array(Int4)
            active_vert_prefix_block;
        Array(uint)
            active_vertices;

        /// List
        Array(ProximityVF)
            list_vf;
        Array(ProximityEE)
            list_ee;
        Array(ProximityEF)
            list_ef;

        /// CSR
        Array(uint)
            vert_VF_num;
        Array(uint)
            vert_EE_num;
        Array(uint)
            vert_EF_num;

        Array(uchar)
            sa_vf_pair_offset_in_vert;
        Array(uchar)
            sa_ee_pair_offset_in_vert;
        Array(uchar)
            sa_ef_pair_offset_in_vert;
        Array(uint)
            sa_vert_VF_prefix;
        Array(uint)
            sa_vert_EE_prefix;
        Array(uint)
            sa_vert_EF_prefix;
        Array(uint)
            sa_vert_VF_indices;
        Array(uint)
            sa_vert_EE_indices;
        Array(uint)
            sa_vert_EF_indices;

        /// Untangling
        Array(bool)
            face_intersected;
        Array(ATOMIC_FLAG)
            face_accessed;
        Array(ATOMIC_UINT)
            face_curve_idx;
        Array(ATOMIC_UINT)
            curve_count;
        Array(uint)
            curve_idx_map;
        Array(Float3)
            curve_G;
        // Array(ATOMIC_UINT) curve;

        Array(uint)
            prefix_sum_ccd_verts;
        uint total_vert_to_be_optimized;
    };

    CollisionListBroadPhase broad;
    CollisionListNarrowPhase narrow;

#ifndef METAL_CODE
    template<bool is_self_collision = true>
    void allocate(ConstRef(uint) num_verts, ConstRef(uint) num_faces, ConstRef(uint) num_edges) {

        const uint max_collision_bp_vf = 64;// broadphase VF
        const uint max_collision_bp_ee = 96;// broadphase EE

        const uint max_collision_np_vert_vf = 48;// narrowphase VF
        const uint max_collision_np_vert_ee = 96;// narrowphase EE
        const uint max_collision_np_vert_ef = 36;// narrowphase EE

        broad.list_vf.resize(num_verts * (max_collision_bp_vf + 2));
        broad.list_ee.resize(num_edges * (max_collision_bp_ee + 2));

        narrow.list_vf.resize(num_verts * max_collision_np_vert_vf);
        narrow.list_ee.resize(num_verts * max_collision_np_vert_ee);
        narrow.list_ef.resize(num_verts * max_collision_np_vert_ef);

        narrow.total_collision_num.resize(3);
        narrow.sa_indirect_command_buffer.resize(4);

        narrow.vert_VF_num.resize(num_verts);
        narrow.vert_EE_num.resize(num_verts);
        narrow.vert_EF_num.resize(num_verts);

        /// CSR
        narrow.sa_vert_VF_prefix.resize(num_verts + 1);
        narrow.sa_vert_EE_prefix.resize(num_verts + 1);
        narrow.sa_vert_EF_prefix.resize(num_verts + 1);
        narrow.sa_vert_VF_indices.resize(num_verts * max_collision_np_vert_vf);
        narrow.sa_vert_EE_indices.resize(num_verts * max_collision_np_vert_ee);
        narrow.sa_vert_EF_indices.resize(num_verts * max_collision_np_vert_ef);
        narrow.sa_vf_pair_offset_in_vert.resize(num_verts * max_collision_np_vert_vf * 4);
        narrow.sa_ee_pair_offset_in_vert.resize(num_verts * max_collision_np_vert_ee * 4);
        narrow.sa_ef_pair_offset_in_vert.resize(num_verts * max_collision_np_vert_ef * 5);

        narrow.active_vert_prefix.resize(num_verts * 4);
        narrow.active_vert_prefix_block.resize(1024);
        narrow.active_vertices.resize(num_verts);
        narrow.face_intersected.resize(num_faces);
        narrow.prefix_sum_ccd_verts.resize(num_verts);
    }

    template<bool is_self_collision = true>
    void clear(ConstRef(uint) num_verts, ConstRef(uint) num_faces, ConstRef(uint) num_edges) {

        const bool not_allocated = broad.list_vf.is_empty();
        if (not_allocated) {
            fast_print_err("Collision List Is Not Allocated!!");
            return;
        }

        std::memset(broad.list_vf.begin(), 0, (num_verts * sizeof(uint)));
        std::memset(broad.list_ee.begin(), 0, (num_edges * sizeof(uint)));
        std::memset(narrow.vert_VF_num.begin(), 0, (num_verts * sizeof(uint)));
        std::memset(narrow.vert_EE_num.begin(), 0, (num_verts * sizeof(uint)));
        std::memset(narrow.vert_EF_num.begin(), 0, (num_verts * sizeof(uint)));

        std::memset(narrow.active_vert_prefix.begin(), 0, (num_verts * sizeof(Int4)));
        std::memset(narrow.face_intersected.begin(), 0, (num_faces * sizeof(bool)));
        std::memset(narrow.prefix_sum_ccd_verts.begin(), 0, (num_verts * sizeof(uint)));

        narrow.total_collision_num[0] = 0;
        narrow.total_collision_num[1] = 0;
        narrow.total_collision_num[2] = 0;

        narrow.total_vert_to_be_optimized = 0;
        narrow.sa_indirect_command_buffer[0] = makeInt4(1, 1, 1, 0);
        narrow.sa_indirect_command_buffer[1] = makeInt4(1, 1, 1, 0);
        narrow.sa_indirect_command_buffer[2] = makeInt4(1, 1, 1, 0);
        narrow.sa_indirect_command_buffer[3] = makeInt4(1, 1, 1, 0);
    }
#endif
};
