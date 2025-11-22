#pragma once

#include "aabb.h"
#include "bits_utils.h"
#include "collision_data.h"
#include "float_n.h"
#include "lbvh_args.h"
#include "morton.h"
#include "scalar.h"

namespace LBVH {

namespace Construct {

static inline int find_common_prefix(ConstRef(Morton) left, ConstRef(Morton) right) {
    return clz_ulong(left.data ^ right.data);
}
#ifdef METAL_CODE
static inline int find_common_prefix(ConstRef(Morton) left, const GLOBAL Morton &right) {
    return clz_ulong(left.data ^ right.data);
}
#endif

template<typename BufferType>
static inline int cp_i_j(ConstRef(BufferType) sa_morton_sorted, ConstRef(Morton) mi, ConstRef(int) j, ConstRef(int) num_leaves) {
    //
    // we define that sp(i, j) = -1 when j not in [0, n-1]
    //
    bool isValid = (j >= 0 && j < num_leaves);
    return isValid ? find_common_prefix(mi, sa_morton_sorted[j]) : -1;
}

template<typename BufferType>
static inline int cp_i_j(ConstRef(BufferType) sa_morton_sorted, ConstRef(int) i, ConstRef(int) j, ConstRef(uint) num_leaves) {
    return cp_i_j(sa_morton_sorted, sa_morton_sorted[i], j, num_leaves);
}

template<typename BufferType>
inline Int2 determineRange(ConstRef(BufferType) sa_morton_sorted, ConstRef(uint) index, ConstRef(uint) num_leaves) {

    //
    //	 // Determine direction of the range (+1 or -1)
    //			d = sign(  commonPrefix(i, i + 1) - commonPrefix(i, i - 1)  )
    //
    //	 // Compute upper bound for the length of the range
    //			commonPrefixMin = commonPrefix(i, i - d) = min((commonPrefix(i, i + 1), (commonPrefix(i, i - 1))
    //			lmax = 2
    //			while ( commonPrefix(i, i + lmax * d) > commonPrefixMin ) do
    //					lmax = lmax * 2
    //
    //	 // Find the other end using binary search
    //			l = 0
    //			for t in {lmax / 2, lmax / 4, lmax / 8 ...... 1} do
    //					if commonPrefix(i, i + (l + t) * d) > commonPrefixMin then
    //						l = l + t
    //			j = i + l * d (upper/lower bound)
    //

    uint i = index;
    auto mi = sa_morton_sorted[i];
    int cp_left = find_common_prefix(mi, sa_morton_sorted[i - 1]);
    int cp_right = find_common_prefix(mi, sa_morton_sorted[i + 1]);

    int d = cp_left < cp_right ? 1 : -1;
    int cp_min = min_scalar(cp_left, cp_right);
    uint lmax = 2;

    while (cp_i_j(sa_morton_sorted, mi, i + lmax * d, num_leaves) > cp_min) {
        lmax <<= 1;
    }
    uint l = 0;
    for (uint t = lmax >> 1; t >= 1; t >>= 1) {
        if (cp_i_j(sa_morton_sorted, mi, i + (l + t) * d, num_leaves) > cp_min) {
            l += t;
        }
    }

    uint j = i + l * d;
    // Int2 range = j > i ? make<Int2>(i, j) : make<Int2>(j, i);
    Int2 range = make<Int2>(i, j);
    return range;
};

template<typename BufferType>
inline uint findSplit(ConstRef(BufferType) sa_morton_sorted, ThreadRef(Int2) range) {

    //
    //	 // Find the split position using binary search
    //			commonPrefixNode = commonPrefix(i, j)
    //			s = 0
    //			for t in {l / 2, l / 4 ...... 1} do
    //				if commonPrefix(i, i + (s + t) * d) > commonPrefixNode then
    //					s = s + t
    //			split = i + s * d + min(d, 0)

    int d = range.x < range.y ? 1 : -1;
    if (d < 0) range = make<Int2>(range.y, range.x);

    int i = range.x;
    int j = range.y;
    // int l = abs_scalar(j - i);

    auto mi = sa_morton_sorted[i];
    auto mj = sa_morton_sorted[j];
    uint cp_node = find_common_prefix(mi, mj);

    uint split;
    if (mi.data == mj.data) {
        split = ((i + j) >> 1);
    } else {
        uint t = j - i;

        split = i;
        do {
            t = (t + 1) >> 1;
            int newSplit = split + t;
            if (newSplit < j) {
                auto ms = sa_morton_sorted[newSplit];
                uint cp_split = find_common_prefix(mi, ms);
                if (cp_split > cp_node) {
                    split = newSplit;
                }
            }
        } while (t > 1);
    }
    return split;
}

}// namespace Construct

ConstExpr uint mask_is_leaf = 1u << 31;

namespace Query {

///
/// TODO : CPU Optimization : Add to local vector first
///
static inline void add_to_collision_list_ell(Pointer(uint) broad_phase_list, ConstRef(uint) index, ConstRef(uint) target, ThreadRef(uint) numCollision, ConstRef(uint) numElement, const uint max_collision) {
    broad_phase_list[max_collision * index + 1 + numCollision] = target;// First Position For Storing Collision Num
    numCollision++;
}
static inline void add_to_collision_per_element_csr(Pointer(uint) broad_phase_list, ConstRef(uint) target, ThreadRef(uint) numCollision) {
    broad_phase_list[numCollision] = target;
    numCollision++;
}
static inline bool add_to_collision_list_atomic(Pointer(uint) broad_phase_list, ConstRef(uint) left, ConstRef(uint) right, GLOBAL ATOMIC_UINT &collision_count, ConstRef(uint) max_broad_phase_count) {
    uint idx = atomic_add(collision_count, 1);
    if (idx < max_broad_phase_count) {
        broad_phase_list[idx * 2 + 0] = left;
        broad_phase_list[idx * 2 + 1] = right;
        return true;
    } else {
        atomic_sub(collision_count, 1);
        return false;
    }
}

template<typename Primitive, bool is_vf>
inline void traversal_tree_and_find_overlap_atomic(Constant(LbvhArgs) bvh,
                                                   ConstRef(bool) is_self_collision,
                                                   Pointer(uint) broad_phase_list, Pointer(Int4) indirect_command_buffer,
                                                   ConstRef(Primitive) pos, ConstRef(uint) index, ConstRef(uint) max_broad_phase_count) {

    GLOBAL ATOMIC_UINT &collision_count = *((GLOBAL ATOMIC_UINT *)indirect_command_buffer + 3);
    const int STACK_SIZE = 32;
    uint stack[STACK_SIZE];
    int stack_ptr = 0;

    stack[stack_ptr] = 0u;
    stack_ptr += 1;

    uint loop = 0;
    while (stack_ptr > 0) {
        if (loop++ > 10000) { break; }

        stack_ptr -= 1;
        uint node = stack[stack_ptr];
        Int2 child = bvh.sa_children[node];

        for (uint ii = 0; ii < 2; ii++) {
            const uint current = child[ii];
            AABB aabb = bvh.sa_node_aabb[current];

            if (aabb.is_overlap(pos)) {
                uint adj_vid = bvh.sa_object_idx[current];
                if (adj_vid != -1u) {
                    uint collision_obj = adj_vid;

                    if (is_vf) {
                        bool succ = add_to_collision_list_atomic(broad_phase_list, index, collision_obj, collision_count, max_broad_phase_count);
                        if (!succ) break;
                    } else {
                        /// Drop redundent
                        if ((collision_obj > index) || !is_self_collision) {
                            bool succ = add_to_collision_list_atomic(broad_phase_list, index, collision_obj, collision_count, max_broad_phase_count);
                            if (!succ) break;
                        }
                    }
                } else {
                    if (stack_ptr < STACK_SIZE) {
                        stack[stack_ptr] = current;
                        stack_ptr += 1;
                    } else {
                        // overflowed
                        break;
                    }
                }
            }
        }
    };
}

}// namespace Query

}// namespace LBVH
