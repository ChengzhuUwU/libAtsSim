#pragma once

// 矩阵计算 - Matrix

#include "address_space.h"
#include "bits_utils.h"
#include "float_n.h"
// #include <math.h>

// Identity和Eigen库的Identity重叠了

#define Identity2x2 make<Float2x2>(make<Float2>(1.f, 0.f),  \
                                   make<Float2>(0.f, 1.f))

#define Identity3x3 make<Float3x3>(make<Float3>(1.f, 0.f, 0.f),  \
                                   make<Float3>(0.f, 1.f, 0.f),  \
                                   make<Float3>(0.f ,0.f, 1.f))

#define Identity4x4 make<Float4x4>(make<Float4>(1.f, 0.f, 0.f, 0.0f),  \
                                   make<Float4>(0.f, 1.f, 0.f, 0.0f),  \
                                   make<Float4>(0.f ,0.f, 1.f, 0.0f),  \
                                   make<Float4>(0.f ,0.f, 0.f, 1.0f))

#define Zero3x3 make<Float3x3>(make<Float3>(0.f, 0.f, 0.f),  \
                               make<Float3>(0.f, 0.f, 0.f),  \
                               make<Float3>(0.f ,0.f, 0.f))

#define Zero4x4 make<Float4x4>(make<Float4>(0.f, 0.f, 0.f, 0.f),  \
                               make<Float4>(0.f, 0.f, 0.f, 0.f),  \
                               make<Float4>(0.f, 0.f, 0.f, 0.f),  \
                               make<Float4>(0.f ,0.f, 0.f, 0.f))


#if SIM_USE_SIMD

// Get/Set Column
template<typename MatType, typename VecType> inline void set       (TREF(MatType) mat, CREF(uint) columnIdx, CREF(VecType) vec) { mat.columns[columnIdx] = vec;}
template<typename MatType, typename VecType> inline void set_column(TREF(MatType) mat, CREF(uint) columnIdx, CREF(VecType) vec) { mat.columns[columnIdx] = vec;}
template<typename MatType> inline        auto  get       (CREF(MatType) mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
template<typename MatType> inline        auto  get_column(CREF(MatType) mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
template<typename MatType> inline THREAD auto& get       (TREF(MatType) mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
template<typename MatType> inline THREAD auto& get_column(TREF(MatType) mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}

// Get/Set Scalar
template<typename MatType, typename ScalarType> inline void set       (TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat.columns[columnIdx][rowIdx] = value;}
template<typename MatType, typename ScalarType> inline void set_scalar(TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat.columns[columnIdx][rowIdx] = value;}
template<typename MatType>                      inline auto get       (CREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { return mat.columns[columnIdx][rowIdx]; }
template<typename MatType>                      inline auto get_scalar(CREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { return mat.columns[columnIdx][rowIdx]; }
template<typename MatType> 
inline THREAD auto& get(TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { 
    using ScalarType = Meta::get_mat_scalar_type<MatType>;
    THREAD ScalarType* ptr = (THREAD ScalarType*)(&mat.columns[columnIdx]);
    return ptr[rowIdx];
}
template<typename MatType> 
inline THREAD auto& get_scalar(TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { 
    using ScalarType = Meta::get_mat_scalar_type<MatType>;
    THREAD ScalarType* ptr = (THREAD ScalarType*)(&mat.columns[columnIdx]);
    return ptr[rowIdx];
}

    #ifdef METAL_CODE
    template<typename MatType>                   inline DEVICE auto& get       (DEVICE       MatType& mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
    template<typename MatType>                   inline DEVICE auto& get_scalar(DEVICE       MatType& mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
    template<typename MatType>                   inline        auto  get       (DEVICE const MatType& mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
    template<typename MatType>                   inline        auto  get_scalar(DEVICE const MatType& mat, CREF(uint) columnIdx)  { return mat.columns[columnIdx];}
    template<typename MatType, typename VecType> inline void set       (DEVICE MatType& mat, CREF(uint) columnIdx, CREF(VecType) vec) { mat.columns[columnIdx] = vec;}
    template<typename MatType, typename VecType> inline void set_scalar(DEVICE MatType& mat, CREF(uint) columnIdx, CREF(VecType) vec) { mat.columns[columnIdx] = vec;}
    
    // Get/Set Scalar
    template<typename MatType> 
    inline DEVICE auto& get(DEVICE MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { 
        using ScalarType = Meta::get_mat_scalar_type<MatType>;
        DEVICE ScalarType* ptr = (DEVICE ScalarType*)(&mat.columns[columnIdx]);
        return ptr[rowIdx];
    }
    template<typename MatType> 
    inline DEVICE auto& get_scalar(DEVICE MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { 
        using ScalarType = Meta::get_mat_scalar_type<MatType>;
        DEVICE ScalarType* ptr = (DEVICE ScalarType*)(&mat.columns[columnIdx]);
        return ptr[rowIdx];
    }
    template<typename MatType>                      inline auto get       (DEVICE const MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { return mat.columns[columnIdx][rowIdx]; }
    template<typename MatType>                      inline auto get_scalar(DEVICE const MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx) { return mat.columns[columnIdx][rowIdx]; }
    template<typename MatType, typename ScalarType> inline void set       (DEVICE MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat.columns[columnIdx][rowIdx] = value;}
    template<typename MatType, typename ScalarType> inline void set_scalar(DEVICE MatType& mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat.columns[columnIdx][rowIdx] = value;}
    #endif

#elif SIM_USE_GLM

// Get/Set Column
template<typename MatType>                   inline auto& get       (TREF(MatType) mat, CREF(uint) columnIdx)  { return mat[columnIdx];}
template<typename MatType>                   inline auto& get_column(TREF(MatType) mat, CREF(uint) columnIdx)  { return mat[columnIdx];}
template<typename MatType>                   inline auto  get       (CREF(MatType) mat, CREF(uint) columnIdx)  { return mat[columnIdx];}
template<typename MatType>                   inline auto  get_column(CREF(MatType) mat, CREF(uint) columnIdx)  { return mat[columnIdx];}
template<typename MatType, typename VecType> inline void  set       (TREF(MatType) mat, CREF(uint) columnIdx, CREF(VecType) vec){  mat[columnIdx] = vec;}
template<typename MatType, typename VecType> inline void  set_column(TREF(MatType) mat, CREF(uint) columnIdx, CREF(VecType) vec){  mat[columnIdx] = vec;}

// Get/Set Scalar
template<typename MatType>                      inline auto& get       (TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) {  return mat[columnIdx][rowIdx]; }
template<typename MatType>                      inline auto& get_scalar(TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) {  return mat[columnIdx][rowIdx]; }
template<typename MatType>                      inline auto  get       (CREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) {  return mat[columnIdx][rowIdx]; }
template<typename MatType>                      inline auto  get_scalar(CREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx) {  return mat[columnIdx][rowIdx]; }
template<typename MatType, typename ScalarType> inline void  set       (TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat[columnIdx][rowIdx] = value;}
template<typename MatType, typename ScalarType> inline void  set_scalar(TREF(MatType) mat, CREF(uint) columnIdx, CREF(uint) rowIdx, CREF(ScalarType) value) { mat[columnIdx][rowIdx] = value;}

#endif

template<typename MatType, typename RowType = Meta::get_mat_row_type<MatType>> 
inline RowType get_row(const THREAD MatType& mat, CREF(uint) rowIdx)  { 
    constexpr uint length_row = Meta::get_mat_row_length<MatType>();
    RowType vec;
    for (uint i = 0; i < length_row; i++) { 
        vec[i] = get_scalar(mat, i, rowIdx); 
    }
    return vec;
}
template<typename MatType, typename RowType = Meta::get_mat_row_type<MatType>> 
inline void set_row(THREAD MatType& mat, CREF(uint) rowIdx, CREF(RowType) vec)  { 
    constexpr uint length_row = Meta::get_mat_row_length<MatType>();
    for (uint i = 0; i < length_row; i++) { 
        get_scalar(mat, i, rowIdx) = vec[i];
    }
}
template<typename MatType, typename ColumnType = Meta::get_mat_column_type<MatType>> 
inline ColumnType get_diag(const THREAD MatType& mat)  { 
    constexpr uint length_column = Meta::get_mat_column_length<MatType>();
    static_assert(Meta::get_mat_column_length<MatType>() == Meta::get_mat_row_length<MatType>(), "Matrix Is Not Square Matrix!");
    ColumnType vec;
    for (uint i = 0; i < length_column; i++) { 
        vec[i] = get_scalar(mat, i, i); 
    }
    return vec;
}
template<typename MatType, typename ColumnType = Meta::get_mat_column_type<MatType>> 
inline void set_diag(THREAD MatType& mat, CREF(ColumnType) vec)  { 
    constexpr uint length_column = Meta::get_mat_column_length<MatType>();
    static_assert(Meta::get_mat_column_length<MatType>() == Meta::get_mat_row_length<MatType>(), "Matrix Is Not Square Matrix!");
    static_assert(Meta::get_mat_column_length<MatType>() == Meta::get_vec_length<ColumnType>(), "Length of Identity Vector Is NOT Equal To Matrix!");
    for (uint i = 0; i < length_column; i++) { 
        get_scalar(mat, i, i) = vec[i];
    }
}


inline constexpr Float3x3 makeFloat3x3() { return Zero3x3; }
inline constexpr Float3x3 makeFloat3x3(CREF(Float3) diag) { return make<Float3x3>(makeFloat3(diag.x, 0, 0), makeFloat3(0, diag.y, 0), makeFloat3(0, 0, diag.z)); }
inline constexpr Float3x3 makeFloat3x3(CREF(Float3) x, CREF(Float3) y, CREF(Float3) z) { return make<Float3x3>(x, y, z); }
inline constexpr Float3x3 makeFloat3x3(float m00, float m01, float m02, 
                                       float m10, float m11, float m12, 
                                       float m20, float m21, float m22) { return makeFloat3x3(makeFloat3(m00, m01, m02), makeFloat3(m10, m11, m12), makeFloat3(m20, m21, m22)); }

inline constexpr Float2x2 makeFloat2x2() { return make<Float2x2>(Zero2, Zero2); }
inline constexpr Float2x2 makeFloat2x2(CREF(Float2) column1, CREF(Float2) column2) { return make<Float2x2>(column1, column2); }
inline constexpr Float2x3 makeFloat2x3() { return make<Float2x3>(Zero3, Zero3); }
inline constexpr Float2x3 makeFloat2x3(CREF(Float3) column1, CREF(Float3) column2) { return make<Float2x3>(column1, column2); }
inline constexpr Float4x3 makeFloat4x3() { return make<Float4x3>(Zero3, Zero3, Zero3, Zero3); }
inline constexpr Float4x3 makeFloat4x3(CREF(Float3) c1, CREF(Float3) c2, CREF(Float3) c3, CREF(Float3) c4)  { return make<Float4x3>(c1, c2, c3, c4); }


struct Matrix3x3f{
    float mat[3][3];
    Matrix3x3f() {
        mat[0][0] = 0.f; mat[1][0] = 0.f; mat[2][0] = 0.f;
        mat[0][1] = 0.f; mat[1][1] = 0.f; mat[2][1] = 0.f;
        mat[0][2] = 0.f; mat[1][2] = 0.f; mat[2][2] = 0.f;
    }
    Matrix3x3f(const float diag) {
        mat[0][0] = diag; mat[1][0] = 0.f ; mat[2][0] = 0.f;
        mat[0][1] = 0.f ; mat[1][1] = diag; mat[2][1] = 0.f;
        mat[0][2] = 0.f ; mat[1][2] = 0.f ; mat[2][2] = diag;
    }
    Matrix3x3f(CREF(Float3x3) input) {
        mat[0][0] = get(input, 0, 0); mat[1][0] = get(input, 1, 0); mat[2][0] = get(input, 2, 0);
        mat[0][1] = get(input, 0, 1); mat[1][1] = get(input, 1, 1); mat[2][1] = get(input, 2, 1);
        mat[0][2] = get(input, 0, 2); mat[1][2] = get(input, 1, 2); mat[2][2] = get(input, 2, 2);
    }

    Float3x3 get_mat() { return makeFloat3x3(
        makeFloat3(mat[0][0], mat[0][1], mat[0][2]),
        makeFloat3(mat[1][0], mat[1][1], mat[1][2]),
        makeFloat3(mat[2][0], mat[2][1], mat[2][2])
    ); }
    Float3 get_diag() { return makeFloat3(mat[0][0], mat[1][1], mat[2][2]);  }

    Float3x3 operator()(){ return get_mat();  }
    Float3 operator[](const uint columnIdx) { return makeFloat3(mat[columnIdx][0], mat[columnIdx][1], mat[columnIdx][2]); }
    Float3 operator()(const uint columnIdx) { return makeFloat3(mat[columnIdx][0], mat[columnIdx][1], mat[columnIdx][2]); }
    THREAD       float& operator()(uint columnIdx, uint rowIdx)        { return mat[columnIdx][rowIdx]; }
    THREAD const float& operator()(uint columnIdx, uint rowIdx) const  { return mat[columnIdx][rowIdx]; }
    
};

template<typename Mat>
inline auto transpose_mat(CREF(Mat) mat) {
    #if SIM_USE_SIMD
        return simd::transpose(mat);
    #elif SIM_USE_GLM
        return glm::transpose(mat);
    #endif
}

template<typename Mat, uint M, uint N> inline float sum_mat(CREF(Mat) mat) { 
    float value = 0.f;
    for(uint i = 0; i < M; i++){ 
        for(uint j = 0; j < N; j++){ 
            value += get(mat, i, j);
        } 
    } 
    return value;
}

inline Float3x3 kronecker_product(CREF(Float3) left, CREF(Float3) right){
	return make<Float3x3>(left[0] * right, 
                          left[1] * right, 
                          left[2] * right); 
}

inline void kronecker_product(THREAD Float3 output[3], CREF(Float3) left, CREF(Float3) right){
	output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
}

inline Float4x3 kronecker_product(CREF(Float4) left, CREF(Float3) right){
	return make<Float4x3>(left[0] * right, 
                          left[1] * right, 
                          left[2] * right,
                          left[3] * right);
}

inline void kronecker_product(THREAD Float3 output[4], const THREAD Float4& left, const THREAD Float3& right){
	output[0] = left[0] * right;
    output[1] = left[1] * right;
    output[2] = left[2] * right;
    output[3] = left[3] * right;
}

inline Float2x2 outer_product(CREF(Float2) left, CREF(Float2) right){
	return make<Float2x2>(left * right[0], 
                          left * right[1]); 
}
inline Float3x3 outer_product(CREF(Float3) left, CREF(Float3) right){
    // 对于每一列，拿左向量乘以右向量的对应元素
	return make<Float3x3>(left * right[0], 
                          left * right[1], 
                          left * right[2]); 
}
inline Float4x4 outer_product(CREF(Float4) left, CREF(Float4) right){
	return make<Float4x4>(left * right[0], 
                          left * right[1], 
                          left * right[2],
                          left * right[3]); 
}

inline void outer_product(TREF(Float3x3) result, CREF(Float3) left, CREF(Float3) right){
	set(result, 0, left * right[0]);
	set(result, 1, left * right[1]);
	set(result, 2, left * right[2]);
}


template <typename MatType>
inline float frobenius_squared_norm_mat(CREF(MatType) mat){
    constexpr uint num_col = Meta::get_mat_column_num<MatType>();
    float result = 0.f;
    for (uint i = 0; i < num_col; i++) {
        result += length_squared_vec(get(mat, i));
    }
    return result;
}

template <typename MatType>
inline float frobenius_norm_mat(CREF(MatType) mat){
    return sqrt_scalar(frobenius_squared_norm_mat(mat));
}


// 代数余子式
static inline float det_1x1(float x00){return x00;}
static inline float det_2x2(float x00, float x01, float x10, float x11){return x00 * x11 - x01 * x10;}

#define COFACTOR_DET_2x2(mat,idx,idy) \
            det_1x1( \
                get(mat, idx, idy))

#define COFACTOR_DET_3x3(mat,idx1,idx2,idy1,idy2) \
            det_2x2( \
                get(mat, idx1, idy1), \
                get(mat, idx1, idy2), \
                get(mat, idx2, idy1), \
                get(mat, idx2, idy2))


// inline Float2x2 inverse_mat(float a, float b, float c, float d){
// 	float det = 1.0f / (a * d - b * c);
// 	return make<Float2x2>(make<Float2>(d, -c), make<Float2>(-b, a)) * det;
// }

inline Float2x2 inverse_mat(CREF(Float2x2) mat){
#if SIM_USE_SIMD
#if defined(METAL_CODE)
	float det = simd::determinant(mat);
	Float2x2 output;
    set(output, 0, 0, +(COFACTOR_DET_2x2(mat, 1, 1)) / det);
    set(output, 0, 1, -(COFACTOR_DET_2x2(mat, 1, 0)) / det);
    set(output, 1, 0, +(COFACTOR_DET_2x2(mat, 0, 1)) / det);
    set(output, 1, 1, -(COFACTOR_DET_2x2(mat, 0, 0)) / det);
	return output; 
#else
    return simd::inverse(mat);
#endif
#elif SIM_USE_GLM
    return glm::inverse(mat);
#endif
}

inline Float3x3 inverse_mat(CREF(Float3x3) mat){

#if SIM_USE_SIMD
#if defined(METAL_CODE)
	float det = simd::determinant(mat);
	Float3x3 output;
    set(output, 0, 0, +(COFACTOR_DET_3x3(mat, 1, 2, 1, 2)) / det);
    set(output, 0, 1, -(COFACTOR_DET_3x3(mat, 1, 2, 0, 2)) / det);
    set(output, 0, 2, +(COFACTOR_DET_3x3(mat, 1, 2, 0, 1)) / det);
    set(output, 1, 0, -(COFACTOR_DET_3x3(mat, 0, 2, 1, 2)) / det);
    set(output, 1, 1, +(COFACTOR_DET_3x3(mat, 0, 2, 0, 2)) / det);
    set(output, 1, 2, -(COFACTOR_DET_3x3(mat, 0, 2, 0, 1)) / det);
    set(output, 2, 0, +(COFACTOR_DET_3x3(mat, 0, 1, 1, 2)) / det);
    set(output, 2, 1, -(COFACTOR_DET_3x3(mat, 0, 1, 0, 2)) / det);
    set(output, 2, 2, +(COFACTOR_DET_3x3(mat, 0, 1, 0, 1)) / det);
	return output; 
#else
    return simd::inverse(mat);
#endif
#elif SIM_USE_GLM
    return glm::inverse(mat);
#endif

}
inline Float3x3 inverse_mat(CREF(Float3x3) mat, CREF(float) det){

	// float det = simd::determinant(mat);
	Float3x3 output;
    set(output, 0, 0, +(COFACTOR_DET_3x3(mat, 1, 2, 1, 2)) / det);
    set(output, 0, 1, -(COFACTOR_DET_3x3(mat, 1, 2, 0, 2)) / det);
    set(output, 0, 2, +(COFACTOR_DET_3x3(mat, 1, 2, 0, 1)) / det);
    set(output, 1, 1, +(COFACTOR_DET_3x3(mat, 0, 2, 0, 2)) / det);
    set(output, 1, 2, -(COFACTOR_DET_3x3(mat, 0, 2, 0, 1)) / det);
    set(output, 2, 2, +(COFACTOR_DET_3x3(mat, 0, 1, 0, 1)) / det);

    set(output, 1, 0, -(COFACTOR_DET_3x3(mat, 0, 2, 1, 2)) / det);
    set(output, 2, 0, +(COFACTOR_DET_3x3(mat, 0, 1, 1, 2)) / det);
    set(output, 2, 1, -(COFACTOR_DET_3x3(mat, 0, 1, 0, 2)) / det);
    
    // Symmetric Matrix : Transpose
    // set(output, 1, 0, get(output, 0, 1));
    // set(output, 2, 0, get(output, 0, 2));
    // set(output, 2, 1, get(output, 1, 2));

	return output; 
}
inline Float4x4 inverse_mat(CREF(Float4x4) mat){

	#if SIM_USE_SIMD
#if defined(METAL_CODE)
	return mat; 
#else
    return simd::inverse(mat);
#endif
#elif SIM_USE_GLM
    return glm::inverse(mat);
#endif

}

template<typename T>
static inline float determinant_mat(CREF(T) mat){
#if SIM_USE_SIMD
    return simd::determinant(mat);
#elif SIM_USE_GLM
    return glm::determinant(mat);
#endif
}

template<typename T>
static inline float trace_mat(CREF(T) mat){
    constexpr uint N = Meta::get_mat_column_length<T>();
    constexpr uint M = Meta::get_mat_row_length<T>();
    static_assert(N == M, "Matrix Is NOT Square Matrix When Getting Its Trance!");
    return sum_vec(get_diag(mat));
}


template<typename T, uint M = Meta::get_mat_column_num<T>(), uint N = Meta::get_mat_column_length<T>()> 
inline bool is_inf_mat(CREF(T) mat) { 
    bool is_inf = false; 
    for (uint i = 0; i < M; i++) {
        for(uint j = 0; j < N; j++){
            if(is_inf_scalar(get(mat, i, j))) {
                is_inf = true;
            }
        }
    } 
    return is_inf; 
}

template<typename T, uint M = Meta::get_mat_column_num<T>(), uint N = Meta::get_mat_column_length<T>()> 
inline bool is_nan_mat(CREF(T) mat) { 
    bool is_nan = false; 
    for (uint i = 0; i < M; i++) {
        for(uint j = 0; j < N; j++){
            if(is_nan_scalar(get(mat, i, j))) {
                is_nan = true;
            }
        }
    } 
    return is_nan; 
}
inline bool is_inf_mat3(CREF(Float3x3) mat) { return is_inf_mat<Float3x3>(mat); }
inline bool is_inf_mat4(CREF(Float4x4) mat) { return is_inf_mat<Float4x4>(mat); }
inline bool is_nan_mat3(CREF(Float3x3) mat) { return is_nan_mat<Float3x3>(mat); }
inline bool is_nan_mat4(CREF(Float4x4) mat) { return is_nan_mat<Float4x4>(mat); }


#ifndef METAL_CODE
#include <utils.h>
#endif
// 矩阵是否对称正定
inline bool is_positive_define_mat(CREF(Float3x3) A){
	//检查A是否对称
	// if (   !is_equal_scalar(get(A, 0, 1), get(A, 1, 0))
	// 	|| !is_equal_scalar(get(A, 0, 2), get(A, 2, 0))
	// 	|| !is_equal_scalar(get(A, 1, 2), get(A, 2, 1))){    
    //     #ifndef METAL_CODE
    //     fast_print("not systmatic");
    //     #endif
	// 	return false;
	// }
	
	//利用Sylvester判据，计算主子式的行列式
	const float d1 = get(A, 0, 0); //第一个主子式
	const float d2 = get(A, 0, 0) * get(A, 1, 1) - get(A, 0, 1) * get(A, 1, 0); //第二个主子式
	const float d3 = determinant_mat(A); //第三个主子式，即A的行列式
	
	//如果所有主子式都大于零，则A是正定的
	if (d1 > -Epsilon && d2 > -Epsilon && d3 > -Epsilon) { 
		return true;
	}
	else {
        // #ifndef METAL_CODE
        // if(d1 <= -Epsilon) fast_print("d1", d1);
        // if(d2 <= -Epsilon) fast_print("d2", d2);
        // if(d3 <= -Epsilon) fast_print("d3", d3);
        // #endif
		return false;
	}
}


inline float compute_tet_volumn(CREF(Float3) pos0, CREF(Float3) pos1, CREF(Float3) pos2, CREF(Float3) pos3){
    Float3 v1 = pos1 - pos0;
    Float3 v2 = pos2 - pos0;
    Float3 v3 = pos3 - pos0;
    // float volumn = determinant_mat(makeFloat3x3(v1, v2, v3)) / 6.0f;
    // float volumn = abs_scalar(dot_vec(v1, cross_vec(v2, v3))) / 6.0f;
    float volumn = abs_scalar(dot_vec(v1, cross_vec(v2, v3))) * static_cast<float>(0.16666666);
    return volumn;
}



static inline Float4x4 scale(const THREAD Float3& v) {

    // | sx  0   0   0 |
    // | 0   sy  0   0 |
    // | 0   0   sz  0 |
    // | 0   0   0   1 |

    Float4x4 result = Zero4x4;
    set(result, 0, 0, v[0]);
    set(result, 1, 1, v[1]);
    set(result, 2, 2, v[2]);
    set(result, 3, 3, 1.f);
    return result;
}
static inline Float4x4 translate(const THREAD Float3& v) {
    
    // | 1 0 0 x |
    // | 0 1 0 y |
    // | 0 0 1 z |
    // | 0 0 0 1 |
    
    // Float4x4 result = Identity4x4;
    // set(result, 0, 0, 1.0f);
    // set(result, 1, 1, 1.0f);
    // set(result, 2, 2, 1.0f);
    // set(result, 3, 3, 1.0f);
    // set(result, 3, 0, v[0]);
    // set(result, 3, 1, v[1]);
    // set(result, 3, 2, v[2]);
    // return result;

    return make<Float4x4>(
        makeFloat4(1.0f, 0.0f, 0.0f, 0.0f),
        makeFloat4(0.0f, 1.0f, 0.0f, 0.0f),
        makeFloat4(0.0f, 0.0f, 1.0f, 0.0f),
        makeFloat4(v[0], v[1], v[2], 1.0f)
    );
}
static inline Float4x4 rorateX(float angleX) {
    float cosX = cos_scalar(angleX);
    float sinX = sin_scalar(angleX);

    // |  1     0      0     0 |
    // |  0   cos(θ) -sin(θ) 0 |
    // |  0   sin(θ)  cos(θ) 0 |
    // |  0     0      0     1 |

    return transpose_mat(make<Float4x4>(
        makeFloat4(1.0f, 0.0f, 0.0f,  0.0f),
        makeFloat4(0.0f, cosX, sinX, 0.0f),
        makeFloat4(0.0f, -sinX, cosX,  0.0f),
        makeFloat4(0.0f, 0.0f, 0.0f,  1.0f)
    ));
}
static inline Float4x4 rorateY(float angleY) {
    float cosY = cos_scalar(angleY);
    float sinY = sin_scalar(angleY);

    // |  cos(θ)  0  sin(θ)  0 |
    // |    0     1    0     0 |
    // | -sin(θ)  0  cos(θ)  0 |
    // |    0     0    0     1 |
    return transpose_mat(make<Float4x4>(
        makeFloat4(cosY,  0.0f, -sinY, 0.0f),
        makeFloat4(0.0f,  1.0f, 0.0f, 0.0f),
        makeFloat4(sinY, 0.0f, cosY, 0.0f),
        makeFloat4(0.0f,  0.0f, 0.0f, 1.0f)
    ));
}
static inline Float4x4 rorateZ(float angleZ) {
    float cosZ = cos_scalar(angleZ);
    float sinZ = sin_scalar(angleZ);

    // | cos(θ) -sin(θ)  0  0 |
    // | sin(θ)  cos(θ)  0  0 |
    // |   0       0     1  0 |
    // |   0       0     0  1 |

    return transpose_mat(make<Float4x4>(
        makeFloat4(cosZ, -sinZ, 0.0f, 0.0f),
        makeFloat4(sinZ, cosZ,  0.0f, 0.0f),
        makeFloat4(0.0f, 0.0f,  1.0f, 0.0f),
        makeFloat4(0.0f, 0.0f,  0.0f, 1.0f)
    ));
}
static inline Float4x4 rotate(const THREAD Float3& axis) {
    return rorateX(axis[0]) * rorateY(axis[1]) * rorateZ(axis[2]);
}

inline Float4x4 make_model_matrix(const THREAD Float3& t, const THREAD Float3& r, const THREAD Float3& s){
    return translate(t) * rotate(r) * scale(s)  ;
    // return scale(s) * (rotate(r) * translate(t));
    // return translate(t) ;
}

inline Float3 affine_position(CREF(Float4x4) model_matrix, CREF(Float3) model_position){
    Float4 mult_position = model_matrix * makeFloat4(model_position[0], model_position[1], model_position[2], 1.0f);
    return makeFloat3(mult_position[0], mult_position[1], mult_position[2]);
}

inline Float3 compute_face_normal(CREF(Float3x3) triangle){
	return compute_face_normal(get(triangle, 0), get(triangle, 1), get(triangle, 2));
}



struct AnimationPerFrameData
{
    Float3 translation;
    Float3 rotation;
    Float3 scale;
    // uint frame = 0;
    AnimationPerFrameData() : translation(makeFloat3(0)), rotation(makeFloat3(0)), scale(makeFloat3(1)) { }
    AnimationPerFrameData(CREF(Float3) t, CREF(Float3) r, CREF(Float3) s) 
        : translation(t), rotation(r), scale(s) { }
    inline Float4x4 get_model_matrix() const
    {
        return make_model_matrix(translation, rotation, scale);
    }
};