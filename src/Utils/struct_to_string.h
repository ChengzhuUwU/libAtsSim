#pragma once
#include "large_matrix.h"
#include "morton.h"
#ifndef METAL_CODE

#include "aabb.h"
#include <string>
#include <sstream>

namespace SimString {

template<typename T>
inline std::string bit_to_radix_string(const T& object, const uint print_bits = 64){
    uint bit_width = min_scalar(sizeof(T) * 8, print_bits);
    std::ostringstream oss;
    for (int i = bit_width - 1; i >= 0; i--) {
        oss << ((object & (T(1) << i)) ? "1" : "0");
        if (i % 8 == 0) oss << " "; 
    }
    return oss.str();
}


inline std::string AABB_to_string(CREF(AABB) aabb) {
    // 使用 std::ostringstream 来构建字符串
    std::ostringstream oss;
    oss << " [(" << aabb.min_pos.x << ", " << aabb.min_pos.y << ", " << aabb.min_pos.z << ")";
    oss << "(" << aabb.max_pos.x << ", " << aabb.max_pos.y << ", " << aabb.max_pos.z << ")] ";
    return oss.str();
}
inline std::string Morton_to_string(CREF(Morton) morton) {
    std::ostringstream oss;
    oss << " (" << morton.data << ")";
    return oss.str();
}

template<typename VecType, uint N = Meta::get_vec_length<VecType>()>
inline std::string vec_to_string(CREF(VecType) vec){
    std::ostringstream oss;
    oss << " (" << vec[0];
    for(uint i = 1; i < N; i++){
        oss << ", " << vec[i];
    }
    oss << ") ";
    return oss.str();
}
template<uint N>
inline std::string large_vec_to_string(const LargeVector<N>& vec){
    std::ostringstream oss;
    oss << " (" << vec(0)[0]; 
    for (uint i = 0; i < N/3; i++) { for(uint j = 0; j < 3; j++) { 
        if(i != 0 || j != 0) 
        oss << ", " << vec[i][j]; } }
    oss << ") "; 
    return oss.str();
}

template<typename MatType, uint M = Meta::get_mat_row_length<MatType>(), uint N = Meta::get_mat_column_length<MatType>()>
inline std::string mat_to_string(const MatType& mat){
    static_assert(M <= 4 && N <= 4, "Matrix is too Large !!");
    std::ostringstream oss;
    oss << " ["; 
    for (uint i = 0; i < N; i++) {
        oss << "\n";
        for(uint j = 0; j < M; j++){
            oss << get(mat, j, i) << " ";
        }
    }  
    oss << "\n]\n"; 
    return oss.str();
}

template<uint M, uint N>
inline std::string large_mat_to_string(const LargeMatrix<M, N>& mat){
    std::ostringstream oss;
    oss << "["; 
    for (uint i = 0; i < N/3; i++) {
        for(uint n = 0; n < 3; n++){
            oss << "\n[";
            for(uint j = 0; j < M/3; j++){
                for(uint m = 0; m < 3; m++){
                    auto tmp = mat(j, i);
                    // oss << std::fixed << std::setprecision(6) << get(tmp, m, n);
                    oss << get(tmp, m, n);
                    if(j != M/3 - 1 || m != 2) oss << ", ";
                }
            }
            oss << "]";
            if(i != N/3 - 1 || n != 2) oss << ",";
        }
    }  
    oss << "\n]\n"; 
    return oss.str();
}

template<typename T1, typename T2>
inline std::string pair_to_string(std::pair<T1, T2> value){
    std::ostringstream oss;
    oss << " (" << value.first << ", " << value.second << ") ";
    return oss.str();
}

template<typename VecType> inline std::string Vec2_to_string(CREF(VecType) vec) {return vec_to_string<VecType, 2>(vec); }
template<typename VecType> inline std::string Vec3_to_string(CREF(VecType) vec) {return vec_to_string<VecType, 3>(vec); }
template<typename VecType> inline std::string Vec4_to_string(CREF(VecType) vec) {return vec_to_string<VecType, 4>(vec); }

// inline std::string Proximity_to_string(CREF(ProximityVF) pair) {
//     std::ostringstream oss;
//     oss << " [ indices = " << Vec4_to_string(pair.get_indices()) << " , area = "
//         << pair.get_area() << " , t = " << Vec3_to_string(pair.get_t()) << " , weight = " << Vec4_to_string(pair.get_weight()) << "]";
//     return oss.str();
// }
// inline std::string Proximity_to_string(CREF(ProximityEE) pair) {
//     std::ostringstream oss;
//     oss << " [ indices = " << Vec4_to_string(pair.get_indices()) << " , area = "
//         << pair.get_area() << " , t = " << Vec3_to_string(pair.get_t()) << " , weight = " << Vec4_to_string(pair.get_weight()) << "]";
//     return oss.str();
// }







};



#endif