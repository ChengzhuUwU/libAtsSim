#pragma once

// 向量计算 - Vector

#include "address_space.h"
#include "constant_value.h"
#include "scalar.h"
#include "bits_utils.h"

#if SIM_USE_SIMD

#include <simd/simd.h>
// using SimdVector<T, N> = typedef __attribute__((__ext_vector_type__(N))) T
using Float2 = simd::float2;
using Float3 = simd::float3;
using Float4 = simd::float4;
using Int2 = simd::uint2;
using Int3 = simd::uint3;
using Int4 = simd::uint4;
using Uchar2 = simd::uchar2;
using Uchar4 = simd::uchar4;
using Float2x2 = simd::float2x2;
using Float3x3 = simd::float3x3;
using Float4x4 = simd::float4x4;
using Float2x3 = simd::float2x3;
using Float3x2 = simd::float3x2;
using Float4x3 = simd::float4x3;
using ElementOffset = simd::uchar4;
// using Bool3 = simd::bool3;

// using VECTOR3 = Float3;
using REAL = float;

#elif SIM_USE_EIGEN
// 不会有人用eigen吧
// #include <eigen/Eigen>
// using Float2 = Eigen::Vector<float, 2>;
// using Float3 = Eigen::Vector<float, 3>;
// using Float4 = Eigen::Vector<float, 4>;
// using Int2 = Eigen::Vector<uint, 2>;
// using Int3 = Eigen::Vector<uint, 3>;
// using Int4 = Eigen::Vector<uint, 4>;
// using Uchar4 = Eigen::Vector<uchar, 4>;
// using Float2x2 = Eigen::Matrix<float, 2, 2>;
// using Float3x3 = Eigen::Matrix<float, 3, 3>;
// using Float4x4 = Eigen::Matrix<float, 4, 4>;
// using Float2x3 = Eigen::Matrix<float, 3, 2>; // 注意到eigen的行列是反过来的
// using Float3x2 = Eigen::Matrix<float, 2, 3>; 
// using Float4x3 = Eigen::Matrix<float, 3, 4>;
// using ElementOffset = Uchar4;
// using Bool3 = Eigen::Vector<bool, 3>;

#elif SIM_USE_GLM

#include <glm/glm.hpp>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/matrix.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi

using Float2 = glm::vec<2, float>;
using Float3 = glm::vec<3, float>;
using Float4 = glm::vec<4, float>;
using Int2 = glm::vec<2, uint>;
using Int3 = glm::vec<3, uint>;
using Int4 = glm::vec<4, uint>;
using Uchar2 = glm::vec<2, uint>;
using Uchar4 = glm::vec<4, uint>;
using Float2x2 = glm::mat<2, 2, float>;
using Float3x3 = glm::mat<3, 3, float>;
using Float4x4 = glm::mat<4, 4, float>;
using Float2x3 = glm::mat<2, 3, float>;
using Float3x2 = glm::mat<3, 2, float>;
using Float4x3 = glm::mat<4, 3, float>;
using ElementOffset = Uchar4;
 // 注意到eigen的行列是反过来的
#endif

// 理论上应该这样？
// template<typename VecType, typename T, typename... Args>
// inline VecType make(Args&&... args) {
//     return VecType{ T(args)... }; // or : return VecType{ std::forward<Args>(args)... };
// }

// make by single scalar
template<typename T, typename ScalarType>
inline constexpr T make(CREF(ScalarType) scalar) {
#if SIM_USE_SIMD || SIM_USE_GLM
    return T(scalar);
#elif SIM_USE_EIGEN
    return T::Constant(value);
#endif
}

// make by several scalar / vec
template<typename T, typename... Args>
inline constexpr T make(Args... args) {
    return T{ args... };
}

namespace Meta{

//
// Vector's Length
//
template <typename T> struct VectorLength;
template <> struct VectorLength<Int2> { enum { column = 2 }; };
template <> struct VectorLength<Int3> { enum { column = 3 }; };
template <> struct VectorLength<Int4> { enum { column = 4 }; };
template <> struct VectorLength<Float2> { enum { column = 2 }; };
template <> struct VectorLength<Float3> { enum { column = 3 }; };
template <> struct VectorLength<Float4> { enum { column = 4 }; };

template <typename T> constexpr inline uint get_vec_length() { return VectorLength<T>::column; }

//
// Vector's Element Type
//
template <typename T> struct VectorScalarType;
template <> struct VectorScalarType<Float2> { typedef float Type; };
template <> struct VectorScalarType<Float3> { typedef float Type; };
template <> struct VectorScalarType<Float4> { typedef float Type; };
template <> struct VectorScalarType<Int2> { typedef uint Type; };
template <> struct VectorScalarType<Int3> { typedef uint Type; };
template <> struct VectorScalarType<Int4> { typedef uint Type; };

template <typename T> using get_vec_scalar_type = typename VectorScalarType<T>::Type;

//
// Matrix's Length
//
template <typename T>  struct MatrixLength;
template <> struct MatrixLength<Float2x2> { enum { row = 2 }; enum { column = 2}; };
template <> struct MatrixLength<Float2x3> { enum { row = 2 }; enum { column = 3}; };
template <> struct MatrixLength<Float3x2> { enum { row = 3 }; enum { column = 2}; };
template <> struct MatrixLength<Float3x3> { enum { row = 3 }; enum { column = 3}; };
template <> struct MatrixLength<Float4x3> { enum { row = 4 }; enum { column = 3}; };
template <> struct MatrixLength<Float4x4> { enum { row = 4 }; enum { column = 4}; };

// Length of Row, Instead Of Number of Row
template <typename T>  constexpr inline uint get_mat_row_length()    { return MatrixLength<T>::row; }
// Length of Column, Instead Of Number of Column
template <typename T>  constexpr inline uint get_mat_column_length() { return MatrixLength<T>::column; }
// Number of Row, Instead Of Length of Row
template <typename T>  constexpr inline uint get_mat_row_num()    { return MatrixLength<T>::column; }
// Number of Column, Instead Of Length of Column
template <typename T>  constexpr inline uint get_mat_column_num() { return MatrixLength<T>::row; }

//
// Matrix's Column Vector Type
//
template <typename T> struct MatrixType;

template <> struct MatrixType<Float2x2> { typedef Float2 ColumnType; typedef Float2 RowType; };
template <> struct MatrixType<Float2x3> { typedef Float3 ColumnType; typedef Float2 RowType; };
template <> struct MatrixType<Float3x2> { typedef Float2 ColumnType; typedef Float3 RowType; };
template <> struct MatrixType<Float3x3> { typedef Float3 ColumnType; typedef Float3 RowType; };
template <> struct MatrixType<Float4x3> { typedef Float3 ColumnType; typedef Float4 RowType; };

template <typename T> using get_mat_column_type = typename MatrixType<T>::ColumnType;
template <typename T> using get_mat_row_type    = typename MatrixType<T>::RowType;

//
// Matrix's Element Type
//
template <typename T> struct MatrixScalarType;

template <> struct MatrixScalarType<Float2x2> { typedef float Type; };
template <> struct MatrixScalarType<Float2x3> { typedef float Type; };
template <> struct MatrixScalarType<Float3x2> { typedef float Type; };
template <> struct MatrixScalarType<Float3x3> { typedef float Type; };
template <> struct MatrixScalarType<Float4x3> { typedef float Type; };
template <> struct MatrixScalarType<Float4x4> { typedef float Type; };

template <typename T> using get_mat_scalar_type = typename MatrixScalarType<T>::Type;



//
// Matrix's Element Type
//
template <uint NumColumn, uint numRow, typename Scalar> struct TemplateToType;

template <> struct TemplateToType<1, 2, float> { typedef Float2   Type; };
template <> struct TemplateToType<1, 3, float> { typedef Float3   Type; };
template <> struct TemplateToType<1, 4, float> { typedef Float4   Type; };
template <> struct TemplateToType<1, 2, uint>  { typedef Int2     Type; };
template <> struct TemplateToType<1, 3, uint>  { typedef Int3     Type; };
template <> struct TemplateToType<1, 4, uint>  { typedef Int4     Type; };
template <> struct TemplateToType<1, 2, bool>  { typedef Uchar2   Type; };
template <> struct TemplateToType<1, 4, bool>  { typedef Uchar4   Type; };

template <> struct TemplateToType<2, 2, float> { typedef Float2x2 Type; };
template <> struct TemplateToType<3, 3, float> { typedef Float3x3 Type; };
template <> struct TemplateToType<4, 4, float> { typedef Float4x4 Type; };
template <> struct TemplateToType<2, 3, float> { typedef Float2x3 Type; };
template <> struct TemplateToType<3, 2, float> { typedef Float3x2 Type; };
template <> struct TemplateToType<4, 3, float> { typedef Float4x3 Type; };

template <uint NumColumn, uint numRow, typename Scalar>  using get_type_by_template = typename TemplateToType<NumColumn, numRow, Scalar>::Type;


}


inline constexpr Float2 makeFloat2(const float x, const float y) { return make<Float2>(x, y); }
inline constexpr Float3 makeFloat3(const float x, const float y, const float z) { return make<Float3>(x, y, z); }
inline constexpr Float4 makeFloat4(const float x, const float y, const float z, const float w) { return make<Float4>(x, y, z, w); }
inline constexpr Float2 makeFloat2(const float x) { return make<Float2>(x); }
inline constexpr Float3 makeFloat3(const float x) { return make<Float3>(x); }
inline constexpr Float4 makeFloat4(const float x) { return make<Float4>(x); }
inline constexpr Int2 makeInt2(const uint x, const uint y) { return make<Int2>(x, y); }
inline constexpr Int3 makeInt3(const uint x, const uint y, const uint z) { return make<Int3>(x, y, z); }
inline constexpr Int4 makeInt4(const uint x, const uint y, const uint z, const uint w) { return make<Int4>(x, y, z, w); }
inline constexpr Int2 makeInt2(const uint x) { return make<Int2>(x); }
inline constexpr Int3 makeInt3(const uint x) { return make<Int3>(x); }
inline constexpr Int4 makeInt4(const uint x) { return make<Int4>(x); }
inline constexpr Uchar2 makeUchar2(const uchar x, const uchar y) { return make<Uchar2>(x, y); }
inline constexpr Uchar4 makeUchar4(const uchar x, const uchar y, const uchar z, const uchar w) { return make<Uchar4>(x, y, z, w); }
 
inline constexpr Float4 makeFloat4(CREF(Float3) vec) { return make<Float4>(vec.x, vec.y ,vec.z, 1.f); }
inline constexpr Float4 makeFloat4(CREF(Float3) vec, const float w) { return make<Float4>(vec.x, vec.y ,vec.z, w); }
inline constexpr Float3 makeFloat3(CREF(Float4) vec) { return make<Float3>(vec.x, vec.y ,vec.z); }

#define Float3_zero make<Float3>(0)
#define Float3_one  make<Float3>(1)
#define FLOAT3_MAX  make<Float3>(Float_max)
#define FLOAT3_EPSILON make<Float3>(EPSILON)

#define Color_Rre    make<Float4>(0.9f, 0.1f, 0.1f, 1.f)
#define Color_Green  make<Float4>(0.1f, 0.9f, 0.1f, 1.f)
#define Color_Blue   make<Float4>(0.1f, 0.1f, 0.9f, 1.f)
#define Color_Yellow make<Float4>(0.9f, 0.9f, 0.1f, 1.f)
#define Color_Orange make<Float4>(0.9f, 0.5f, 0.1f, 1.f)
#define Color_Purple make<Float4>(0.5f, 0.1f, 0.9f, 1.f)
#define Color_Cyan   make<Float4>(0.1f, 0.9f, 0.9f, 1.f)
#define Zero2        make<Float2>(0.f, 0.f)
#define Zero3        make<Float3>(0.f, 0.f, 0.f)
#define Zero4        make<Float4>(0.f, 0.f, 0.f, 0.f)

CONSTEXPR Float4 ColorContainer[] = {
    {0.1211, 0.4648, 0.7031, 1.0}, 
    {0.9961, 0.4961, 0.0547, 1.0}, 
    {0.1719, 0.6250, 0.1719, 1.0}, 
    // {0.8359, 0.1523, 0.1562, 1.0}, 
    {0.5781, 0.4023, 0.7383, 1.0}, 
    {0.5469, 0.3359, 0.2930, 1.0}, 
    {0.8867, 0.4648, 0.7578, 1.0}, 
    {0.4961, 0.4961, 0.4961, 1.0}, 
    {0.7344, 0.7383, 0.1328, 1.0}, 
    {0.0898, 0.7422, 0.8086, 1.0}, 
    {0.6797, 0.7773, 0.9062, 1.0}, 
    {0.9961, 0.7305, 0.4688, 1.0}, 
    {0.5938, 0.8711, 0.5391, 1.0}, 
    {0.9961, 0.5938, 0.5859, 1.0}, 
    {0.7695, 0.6875, 0.8320, 1.0}, 
    {0.7656, 0.6094, 0.5781, 1.0}, 
    {0.9648, 0.7109, 0.8203, 1.0}, 
    {0.7773, 0.7773, 0.7773, 1.0}, 
    {0.8555, 0.8555, 0.5508, 1.0}, 
    {0.6172, 0.8516, 0.8945, 1.0}};
inline Float4 get_default_color_by_id(uint idx){
    return ColorContainer[idx % sizeof(ColorContainer)];
}


#ifdef METAL_CODE
template<typename T>
inline DEVICE atomic_float* to_atomic_float_ptr(DEVICE T* ptr){
    return reinterpret_cast<DEVICE atomic_float*>(ptr);
}
#endif


#if SIM_USE_SIMD

template<typename Vec> inline Vec normalize_vec(CREF(Vec) vec) {return simd::normalize(vec);}
template<typename Vec> inline auto length_vec(CREF(Vec) vec) { return simd::fast::length(vec); }
template<typename Vec> inline Vec reverse_vec(CREF(Vec) vec) { return 1.f / vec;}
template<typename Vec> inline Vec abs_vec(CREF(Vec) vec) { return simd::abs(vec); }

template<typename Vec> inline Vec cross_vec(CREF(Vec) vec1, CREF(Vec) vec2)   { return simd::cross(vec1, vec2);   }
template<typename Vec> inline auto dot_vec(CREF(Vec) vec1, CREF(Vec) vec2)    { return simd::dot(vec1, vec2);     }
template<typename Vec> inline Vec max_vec(CREF(Vec) vec1, CREF(Vec) vec2) { return simd::max(vec1, vec2); }
template<typename Vec> inline Vec min_vec(CREF(Vec) vec1, CREF(Vec) vec2) { return simd::min(vec1, vec2); }

#ifdef METAL_CODE
template<typename Vec> inline Vec max_vec_copy(const Vec vec1, const Vec vec2) {return simd::max(vec1, vec2);}
template<typename Vec> inline Vec min_vec_copy(const Vec vec1, const Vec vec2) {return simd::min(vec1, vec2);}
#endif

#elif SIM_USE_GLM

template<typename Vec> inline Vec normalize_vec(CREF(Vec) vec) {return glm::normalize(vec);}
template<typename Vec> inline auto length_vec(CREF(Vec) vec) { return glm::length(vec); }
template<typename Vec> inline Vec reverse_vec(CREF(Vec) vec) { return 1.f / vec; }
template<typename Vec> inline Vec abs_vec(CREF(Vec) vec) { return glm::abs(vec); }

template<typename Vec> inline Vec cross_vec(CREF(Vec) vec1, CREF(Vec) vec2)   { return glm::cross(vec1, vec2);   }
template<typename Vec> inline auto dot_vec(CREF(Vec) vec1, CREF(Vec) vec2)    { return glm::dot(vec1, vec2);     }
template<typename Vec> inline Vec max_vec(CREF(Vec) vec1, CREF(Vec) vec2) {return glm::max(vec1, vec2);}
template<typename Vec> inline Vec min_vec(CREF(Vec) vec1, CREF(Vec) vec2) {return glm::min(vec1, vec2);}

#endif

template<typename Vec> inline auto safe_length_vec(CREF(Vec) vec) { return length_vec(vec) + Epsilon; }

template<typename Vec, uint N = Meta::get_vec_length<Vec>()> 
inline float sum_vec(CREF(Vec) vec) { 
    float value = vec[0];
    for(uint i = 1; i < N; i++){ value += vec[i];} 
    return value;
}


template<typename Vec, uint N> inline auto min_component_vec(CREF(Vec) vec) { 
    auto min_value = vec[0];
    for(uint i = 1; i < N; i++){ min_value = min_scalar(min_value, vec[i]);} 
    return min_value;
}
template<typename Vec, uint N> inline auto max_component_vec(CREF(Vec) vec) { 
    auto max_value = vec[0];
    for(uint i = 1; i < N; i++){ max_value = max_scalar(max_value, vec[i]); } 
    return max_value;
}

template<typename Vec, uint N>
inline float infinity_norm_vec(CREF(Vec) vec){
    return max_component_vec<Vec, N>(abs_vec(vec));
}

template<typename Vec> 
inline auto length_squared_vec(CREF(Vec) vec) { 
    return dot_vec(vec, vec);
}

template<typename Vec>
inline Vec clamp_vec(CREF(Vec) vec, CREF(Vec) lower, CREF(Vec) upper) {
    return max_vec(min_vec(vec, upper), lower);
}
template<typename Vec> 
inline constexpr Vec lerp_vec(CREF(Vec) left, CREF(Vec) right, CREF(float) lerp_value) { 
    return left + lerp_value * (right - left);
}

template<typename Vec, uint N = Meta::get_vec_length<Vec>()> 
inline bool is_inf_vec(CREF(Vec) vec) { 
    bool is_inf = false; 
    for (uint i = 0; i < N; i++) {
        if(is_inf_scalar(vec[i])) {
            is_inf = true;
        }
    } 
    return is_inf; 
}
template<typename Vec, uint N = Meta::get_vec_length<Vec>()> 
inline bool is_nan_vec(CREF(Vec) vec) { 
    bool is_nan = false; 
    for (uint i = 0; i < N; i++) {
        if(is_nan_scalar(vec[i])) {
            is_nan = true;
        }
    } 
    return is_nan; 
}
inline bool is_inf_float3(CREF(Float3) vec) { return is_inf_vec<Float3>(vec); }
inline bool is_inf_float4(CREF(Float4) vec) { return is_inf_vec<Float4>(vec); }
inline bool is_nan_float3(CREF(Float3) vec) { return is_nan_vec<Float3>(vec); }
inline bool is_nan_float4(CREF(Float4) vec) { return is_nan_vec<Float4>(vec); }

template<typename Vec> 
inline Vec project_vec(CREF(Vec) vec1, CREF(Vec) vec2) {
    // #if SIM_USE_SIMD && !defined(METAL_CODE)
    //     return simd::project(vec1, vec2);
    // // #elif SIM_USE_EIGEN
    // #else
        auto length_squred_vec2 = dot_vec(vec2, vec2); // u^2
        if (length_squred_vec2 != 0) return (dot_vec(vec1, vec2) / length_squred_vec2) * vec2; // dot(u, v)/dot(v, v)*v (u proj to v)
        else return make<Vec>(0);
    // #elif SIM_USE_GLM
    // #endif
}


// bool
#if SIM_USE_SIMD
template<typename Vec> inline auto is_greater_than(CREF(Vec) vec1, CREF(Vec) vec2) { return vec1 > vec2; }
template<typename Vec> inline auto is_greater_equal_than(CREF(Vec) vec1, CREF(Vec) vec2) { return vec1 >= vec2; }
template<typename Vec> inline auto is_smaller_than(CREF(Vec) vec1, CREF(Vec) vec2) { return vec1 < vec2; }
template<typename Vec> inline auto is_smaller_equal_than(CREF(Vec) vec1, CREF(Vec) vec2) { return vec1 <= vec2; }
#elif SIM_USE_GLM
template<typename Vec> inline auto is_greater_than(CREF(Vec) vec1, CREF(Vec) vec2) { return glm::greaterThan(vec1, vec2); }
template<typename Vec> inline auto is_greater_equal_than(CREF(Vec) vec1, CREF(Vec) vec2) { return glm::greaterThanEqual(vec1, vec2); }
template<typename Vec> inline auto is_smaller_than(CREF(Vec) vec1, CREF(Vec) vec2) { return glm::lessThan(vec1, vec2); }
template<typename Vec> inline auto is_smaller_equal_than(CREF(Vec) vec1, CREF(Vec) vec2) { return glm::lessThanEqual(vec1, vec2); }
#endif

#if SIM_USE_SIMD
template<typename Vec> inline bool all_vec(CREF(Vec) vec) {return simd::all(vec);}
template<typename Vec> inline bool any_vec(CREF(Vec) vec) {return simd::any(vec);}
#elif SIM_USE_GLM
template<typename Vec> inline bool all_vec(CREF(Vec) vec) {return glm::all(vec);}
template<typename Vec> inline bool any_vec(CREF(Vec) vec) {return glm::any(vec);}
#endif

template<typename Vec> inline Vec and_vec(CREF(Vec) vec1, CREF(Vec) vec2) {return vec1 && vec2;}
template<typename Vec> inline Vec or_vec(CREF(Vec) vec1, CREF(Vec) vec2) {return vec1 || vec2;}


inline float compute_edge_adj_face_angle(
    const THREAD Float3& vec0, const THREAD Float3& vec1, const THREAD Float3& vec2, const THREAD Float3& vec3, 
    THREAD Float3& normal1, THREAD Float3& normal2)
{
    normal1 = normalize_vec(cross_vec(vec0, vec1));
	normal2 = normalize_vec(cross_vec(vec2, vec0));
    float angle_sign = sign_scalar(dot_vec(vec3, normal1)); // 二面角是否大于180度，大于，则由cos算出来的角度取反
    float dot_value = dot_vec(normal1, normal2);
    dot_value = clamp_scalar(dot_value, -0.99999f, 0.99999f);
    float angle = Pi + angle_sign * acos(dot_value);
    return angle;
}

inline float compute_edge_adj_face_angle(Float3 vert_pos[4]){

    Float3 vec0 = vert_pos[1] - vert_pos[0];
    Float3 vec1 = vert_pos[2] - vert_pos[0];
    Float3 vec2 = vert_pos[3] - vert_pos[0];
    Float3 vec3 = (vert_pos[2] + vert_pos[3]) * 0.5f - vert_pos[0];
    Float3 normal1, normal2;
    return compute_edge_adj_face_angle(vec0, vec1, vec2, vec3, normal1, normal2);

}

inline Float3 to_spherical_coordinates(Float3 vec){
    
    float len = length_vec(vec);
    float theta = acos(vec[2] / len);
    float phi = atan(vec[1] / vec[0]);
    return Float3{len, theta, phi}; // r, theta, phi
}

inline Float3 inv_spherical_coordinates(Float3 vec){
    float x = vec[0] * sin(vec[1]) * cos(vec[2]);
    float y = vec[0] * sin(vec[1]) * sin(vec[2]);
    float z = vec[0] * cos(vec[1]);
    return Float3{x, y, z};
}

inline float compute_face_area(CREF(Float3) pos0, CREF(Float3) pos1, CREF(Float3) pos2){
    Float3 vec0 = pos1 - pos0;
    Float3 vec1 = pos2 - pos0;
    float area = length_vec(cross_vec(vec0, vec1)) * 0.5f;
    return area;
}
inline Float3 compute_face_normal(CREF(Float3) p1, CREF(Float3) p2, CREF(Float3) p3){
	Float3 s = p2 - p1;
	Float3 t = p3 - p1;
	Float3 n = normalize_vec(cross_vec(s, t));
	return n;
}

template<typename Vec>
inline float average_vec(Vec vec){
    return sum_vec(vec) / float(Meta::get_vec_length<Vec>());
}

template<typename T, size_t N>
inline T average_array(T arr[N]){
    T value = arr[0];
    for (size_t i = 1; i < N; i++) {
        value += arr[i];
    }
    return value / float(N);
}
// namespace sim{

// template<typename ScalarType, typename VecType, uint N>
// struct vec{

// };

// }