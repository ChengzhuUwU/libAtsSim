#pragma once

#include "address_space.h"

using uchar = unsigned char;

#if defined (__APPLE__)
using uint64 = unsigned long; // 'long long' is not supported in Metal [GCC]
using int64  = long;
#else
using uint64 = unsigned long long;
using int64  = long long;
#endif

inline uint get_dispatch_num(uint num_threads, uint block_dim){
    return (num_threads + block_dim - 1) / block_dim;
}

inline uint get_excution_threads(uint num_threads, uint block_dim){
    uint mask_block_dim = ~(block_dim - 1);
    return ((num_threads - 1) & mask_block_dim) + block_dim;
}

inline uint get_excution_threads_256(uint num_threads){
    uint mask_block_dim_256 = 0xFFFFFF00;
    return ((num_threads - 1) & mask_block_dim_256) + 256;
}

// Count Leading Zero : From The Highest Bits (From Left To Right)
inline int clz_uint(uint x) {
#if !defined(METAL_CODE) &&   (defined(__GNUC__) || defined(__clang__))
    return __builtin_clz(x);
#elif defined(METAL_CODE)
    return clz(x);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - index;
#else
    uint count = 0;
    while (x) {
        count++;
        x >>= 1;
    }
    return 32 - count;
#endif
}

// Count Leading Zero : From The Highest Bits (From Left To Right)
inline int clz_ulong(uint64 x) {
#if !defined(METAL_CODE) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_clzll(x);
#elif defined(_MSC_VER)
    uint64 index;
    _BitScanReverse(&index, x);
    return 63 - index;
#else
    uint count = 0;
    while (x) {
        count++;
        x >>= 1;
    }
    return 64 - count;
#endif
}



#if defined (METAL_CODE)
#include <metal_integer>
inline constexpr uint reverse_uint(CONST(uint) v) { return reverse_bits(v); }
inline constexpr uint64 reverse_uint64(CREF(uint64) v) { return reverse_bits(v); }
inline constexpr int ffs_uint(CONST(uint) v) { 
    auto reversed_x = reverse_uint(v);
    return clz_uint(reversed_x) + 1;
}
inline constexpr int ffs_uint64(CREF(uint64) v) { 
    auto reversed_x = reverse_uint64(v);
    return clz_ulong(reversed_x) + 1;
}
// inline constexpr int ffs_uint(uint v) { 
//     // https://blog.csdn.net/suz_cheney/article/details/112456368
//     int n = 0;
// 	if (!v) return 0;
// 	if (!(v & 0x0000FFFF)) { v >>= 16; n += 16; }
// 	if (!(v & 0x000000FF)) { v >>=  8; n += 8;  }
// 	if (!(v & 0x0000000F)) { v >>=  4; n += 4;  }
// 	if (!(v & 0x00000003)) { v >>=  2; n += 2;  }
// 	if (!(v & 0x00000001)) { v >>=  1; n += 1;  }
// 	return n + 1;
// }
// inline constexpr int fls_uint(uint v) { 
//     // https://blog.csdn.net/suz_cheney/article/details/112456368
//     int n = 32;
// 	if (!v) return 0;
// 	if (!(v & 0xFFFF0000)) { v <<= 16; n -= 16; }
// 	if (!(v & 0xFF000000)) { v <<=  8; n -= 8;  }
// 	if (!(v & 0xF0000000)) { v <<=  4; n -= 4;  }
// 	if (!(v & 0xC0000000)) { v <<=  2; n -= 2;  }
// 	if (!(v & 0x80000000)) { v <<=  1; n -= 1;  }
// 	return n;
// }
inline constexpr int popc_uint(CONST(uint) value) { return popcount(value); }
inline constexpr int popc_uint64(CREF(uint64) value) { return popcount(value); }

#else

#include <bit>
// Reverse The Bits (Exchange Each (i)_bit and (31-i)_bit
inline constexpr int reverse_uint(CONST(uint) v) { return __builtin_bitreverse32(v); }

// Find The First BitSet (From Right To Left)
inline constexpr int ffs_uint(CONST(uint) value) { return __builtin_ffs(value); } // Different From std::countr_one !!!
inline constexpr int ffs_uint64(CREF(uint64) value) { return static_cast<uint64>(__builtin_ffsl(value)); } // Different From std::countr_one !!!

// Find The Last BitSet (From Left To Right)
// inline constexpr uint fls_uint(CONST(uint) value) { return __builtin_fls(value); }

// Population Count
inline constexpr int popc_uint(CONST(uint) value) { return std::popcount(value); }
inline constexpr int popc_uint64(CREF(uint64) value) { return std::popcount(value); }
#endif


// 00000111111 (length(11...11) == laneId)
inline constexpr uint LanemaskLt(CONST(uint) laneId) { return  (1u << laneId) - 1; }
inline constexpr uint make_lane_mask(CONST(uint) laneId) { return  (1u << laneId) - 1; }
inline constexpr uint64 make_lane_mask_64(CONST(uint) laneId) { return  (1ul << laneId) - 1; }

inline int ffs_and_pop(TREF(uint) mask){
    int pos = ffs_uint(mask) - 1;
    mask &= ~(1 << pos);
    return pos + 1;
}
inline int ffs_and_pop64(TREF(uint64) mask){
    int pos = ffs_uint64(mask) - 1;
    mask &= ~(1ul << pos);
    return pos + 1;
}


// Count Trailing Zero : From The Lowest Bits (From Right To Left)


// inline uint make_mask(uint value){
//     return 1 << value;
// }

// template<typename... Args>
// inline uint make_mask(uint value, Args... args){
//     make_mask(value);
//     make_mask(args...);
// }

// inline uint  
