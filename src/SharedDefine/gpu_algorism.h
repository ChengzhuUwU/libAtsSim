#pragma once

#include <address_space.h>

#define MAX_WARP_NUM 32
#define SECOND_REDUCE_DIM 512
#define SECOND_SCAN_DIM 512

#ifdef METAL_CODE
    #define THREAD_GROUP_SYNC threadgroup_barrier(mem_flags::mem_threadgroup)
    #define THREAD_FENCE __atomic_thread_fence(memory_order_relaxed);
#else
    #define THREAD_GROUP_SYNC
    #define THREAD_FENCE atomic_thread_fence(std::memory_order::relaxed);
#endif



#ifdef METAL_CODE
    // https://github.com/KhronosGroup/SPIRV-Cross/blob/68d401117c85219ee6b2aba9a0cded314c55798f/reference/opt/shaders-msl/comp/shader_ballot.msl22.comp#L46
    // FIXME: This won't include higher bits if Apple ever supports 64 lanes in an SIMD-group.
    uint cast_vote_to_mask(CREF(simd_vote) mask) { return static_cast<simd_vote::vote_t>(mask); }
    template<typename T> inline T warp_reduce_sum(CREF(T) value)  { return simd_sum(value); }
    template<typename T> inline T warp_prefix_sum_inclusive(CREF(T) value)  { return simd_prefix_inclusive_sum(value); }
    template<typename T> inline T warp_prefix_sum_exclusive(CREF(T) value)  { return simd_prefix_exclusive_sum(value); }
    template<typename T> inline T warp_shuffle(CREF(T) value, CREF(uint) lane_id)      { return simd_shuffle(value, lane_id); }
    template<typename T> inline T warp_shuffle_down(CREF(T) value, CREF(uint) delta)   { return simd_shuffle_down(value, delta); }
    inline uint warp_vote()                     { return cast_vote_to_mask(simd_vote()); }
    inline uint warp_ballot(CREF(bool) value)   { return cast_vote_to_mask(simd_ballot(value)); }
#else
    template<typename T> inline T warp_reduce_sum(CREF(T) value){ return (value); }
    template<typename T> inline T warp_prefix_sum_inclusive(CREF(T) value)  { return (value); }
    template<typename T> inline T warp_prefix_sum_exclusive(CREF(T) value)  { return (value); }
    template<typename T> inline T warp_shuffle(CREF(T) value, CREF(uint) lane_id)     { return (value); }
    template<typename T> inline T warp_shuffle_down(CREF(T) value, CREF(uint) delta)  { return (value); }
    inline uint warp_vote()                     { return 0; }
    inline uint warp_ballot(CREF(bool) value)   { return (value); } // Vote true or false Into A 32-bits Mask
#endif





//
// Reduce With Warp-Level Intrinsic
//

// #define Reduce(val, tid, warp_num)              \
//     val = simd_sum(val);                        \
//     if(tid >= warp_num) return;                 \
//     if(tid < warp_num){                         \
//         val = simd_shuffle_down(val, 31 * tid); \
//         val = simd_sum(val);                    \
//     }

#define threadgroup_ids                             \
    uint tid [[thread_position_in_threadgroup]],    \
    uint lid [[thread_index_in_simdgroup]],         \
    uint wid [[simdgroup_index_in_threadgroup]]     

#define reduce_local(val, tid, lid, wid, reduce_op)      \
    val = reduce_op(val);                                \
    threadgroup float cache_reduce[MAX_WARP_NUM];        \
    if(wid == 0) cache_reduce[tid] = 0;                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);     \
    if(lid == 0){                                        \
        cache_reduce[wid] = val;                         \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    }                                                    \
    if(wid == 0){                                        \
        val = simd_sum(cache_reduce[tid]);               \
    }             

#define reduce_local_n(val, tid, lid, wid, reduce_op, n)    \
    val = reduce_op(val);                                   \
    threadgroup float cache_reduce_##n[MAX_WARP_NUM];       \
    if(wid == 0 ) cache_reduce_##n[tid] = 0;                \
    threadgroup_barrier(mem_flags::mem_threadgroup);        \
    if(lid == 0){                                           \
        cache_reduce_##n[wid] = val;                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);    \
    }                                                       \
    if(wid == 0){                                           \
        val = simd_sum(cache_reduce_##n[tid]);              \
    }                                                    

#define reduce_add(val) reduce_local(val, tid, lid, wid, simd_sum)
#define reduce_add_n(val, n) reduce_local_n(val, tid, lid, wid, simd_sum, n)

// Usage Template : 
//      float dot_pq = data.sa_block_info[index];
//      reduce_add(dot_pq);
//      if(tid == 0){ atomic_add(sa_convergence[0], dot_pq); }


//
// Reduce With Cache
//
#define reduce_with_cache(aabb, tid, dim, type, set_op, reduce_op, get_op)      \
            threadgroup type cache_aabb[dim];                           \
            set_op(cache_aabb[tid], aabb)                               \
            threadgroup_barrier(mem_flags::mem_threadgroup);            \
			for (uint s = dim >> 1; s > 0; s >>= 1){	                \
				if(tid < s) {                                           \
                    reduce_op(cache_aabb[tid], cache_aabb[tid + s]);    \
                }                                                       \
				threadgroup_barrier(mem_flags::mem_threadgroup);	    \
			}                                                           \
            if(wid == 0) {                                              \
                get_op(aabb, cache_aabb[0]);                            \
            }

#define set_op_aabb(a, b)    a[0] = b.min_pos; a[1] = b.max_pos;
#define get_op_aabb(a, b)    a.min_pos = b[0]; a.max_pos = b[1];
#define reduce_op_aabb(a, b) a[0] = simd::min(a[0], b[0]); a[1] = simd::max(a[1], b[1]);

#define reduce_aabb(aabb, tid, dim) reduce_with_cache(aabb, tid, dim, Float2x3, set_op_aabb, reduce_op_aabb, get_op_aabb)         


// Usage Template : 
//      AABB aabb = is_valid ? bvh.sa_block_aabb[index] : AABB();
//      reduce_aabb(aabb, tid, SECOND_REDUCE_DIM);
//      if(tid == 0){ aabb.max_pos = aabb.get_inv_dim(); sa_block_aabb[0] = aabb) }


#define set_op_uint4(a, b)    a = b;
#define get_op_uint4(a, b)    a = b;
#define reduce_op_uint4(a, b) a = (a + b);

#define reduce_uint4(vec, tid, dim) reduce_with_cache(vec, tid, dim, Int4, set_op_uint4, reduce_op_uint4, get_op_uint4)       

#define set_op_min_max(a, b)    a = b;
#define get_op_min_max(a, b)    a = b;
#define reduce_op_min_max(a, b) a[0] = simd::min(a[0], b[0]); a[1] = simd::max(a[1], b[1]);

#define reduce_min_max(aabb, tid, dim) reduce_with_cache(aabb, tid, dim, Int2, set_op_min_max, reduce_op_min_max, get_op_min_max)   


// #define reduce_aabb(aabb, tid, dim)                                     \
//             threadgroup Float2x3 cache_aabb[dim];                       \
//             cache_aabb[tid][0] = aabb.min_pos;                          \
//             cache_aabb[tid][1] = aabb.max_pos;                          \
//             threadgroup_barrier(mem_flags::mem_threadgroup);            \
// 			for (uint s = dim >> 1; s > 0; s >>= 1){	                \
// 				if(tid < s) {\
//                     cache_aabb[tid][0] = simd::min(cache_aabb[tid][0], cache_aabb[tid + s][0]); \
//                     cache_aabb[tid][1] = simd::max(cache_aabb[tid][1], cache_aabb[tid + s][1]); \
//                 }                                                   \
// 				threadgroup_barrier(mem_flags::mem_threadgroup);	\
// 			}                                                       \
//             if(tid == 0) {                              \
//                 aabb.min_pos = cache_aabb[0][0];        \
//                 aabb.max_pos = cache_aabb[0][1];        \
//             }


// #define GROUP_SIZE 256u
// #define KEYS_PER_THREAD 4u
// groupshared int lds_keys[GROUP_SIZE];

#ifdef METAL_CODE
#define scan_exclusive(cache_value, val, prefix, tid, lid, wid)         \
    cache_value[lid] = 0;                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                \
    auto wave_scanned = simd_prefix_exclusive_sum(val);              \
    if (lid == 32 - 1) cache_value[wid] = wave_scanned + val;       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    if (wid == 0) cache_value[lid] = simd_prefix_exclusive_sum(cache_value[lid]);       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    prefix = wave_scanned + cache_value[wid];      

#define scan_inclusive(cache_value, val, prefix, tid, lid, wid)         \
    cache_value[lid] = 0;                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                \
    auto wave_scanned = simd_prefix_inclusive_sum(val);              \
    if (lid == 32 - 1) cache_value[wid] = wave_scanned;             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    if (wid == 0) cache_value[lid] = simd_prefix_exclusive_sum(cache_value[lid]);       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    prefix = wave_scanned + cache_value[wid];  

#else
#define scan_exclusive(cache_value, val, prefix, tid, lid, wid)
#define scan_inclusive(cache_value, val, prefix, tid, lid, wid)
#endif



#define REDUCE(TID,DIM,CACHE)							   \
			for (uint S = DIM >> 1; S > 0; S >>= 1){	   \
				if(TID < S) CACHE[TID] += CACHE[TID + S];  \
				GroupMemoryBarrierWithGroupSync();		   \
			}

// Exclusive Scan
#define SCAN(lidx, GROUP_SIZE, lds_keys, temp)	                                                    \
    uint stride = 0;                                                                                \
    for (stride = 1; stride < GROUP_SIZE; stride <<= 1) {                                           \
        if (lidx < GROUP_SIZE / (2 * stride)) {                                                     \
            lds_keys[2 * (lidx + 1) * stride - 1] =  lds_keys[2 * (lidx + 1) * stride - 1] + lds_keys[(2 * lidx + 1) * stride - 1]; \
        }                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);        \
    }                                                           \
    if (lidx == 0) lds_keys[GROUP_SIZE - 1] = 0;                \
    threadgroup_barrier(mem_flags::mem_threadgroup);            \
    for (stride = GROUP_SIZE >> 1; stride > 0; stride >>= 1) {  \
        if (lidx < GROUP_SIZE / (2 * stride)) {                 \
            temp                                  = lds_keys[(2 * lidx + 1) * stride - 1];          \
            lds_keys[(2 * lidx + 1) * stride - 1] = lds_keys[2 * (lidx + 1) * stride - 1];          \
            lds_keys[2 * (lidx + 1) * stride - 1] = lds_keys[2 * (lidx + 1) * stride - 1] + temp;   \
        }                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);        \
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup);


// Usage Template : 
//      THREADGROUP Int4 cache_scan[blockDim]; // 4 MB
//      cache_scan[tid] = value;
//      THREAD_GROUP_SYNC;
//
//      Int4 temp;
//      SCAN(tid, blockDim, cache_scan, temp);	
//      THREAD_GROUP_SYNC;
//      Int4 local_prefix_exclusive = cache_scan[tid];
//      Int4 local_prefix_inclusive = cache_scan[tid] + value;



// template<typename T, typename ReduceFunc>
// inline void reduce_block(THREAD T& val, threadgroup T* cache_reduce, uint tid, uint lid, uint wid, ReduceFunc reduce_op)  
// {
//     val = reduce_op(val);                                
//     if(wid == 0 ) cache_reduce[tid] = 0;
//     threadgroup_barrier(mem_flags::mem_threadgroup);     
//     if(lid == 0){                                        
//         cache_reduce[wid] = val;                         
//         threadgroup_barrier(mem_flags::mem_threadgroup); 
//     }                                                    
//     if(wid == 0){                                        
//         val = simd_sum(cache_reduce[tid]);               
//     }                                                   
// }

// template<typename T>
// inline void reduce_block_add(T val, threadgroup T* cache_reduce,, uint tid, uint lid, uint wid) 
// {
//     reduce_block(val, cache_reduce, tid, lid, wid, simd_sum);
// }
