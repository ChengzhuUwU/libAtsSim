#pragma once

#include "float_n_n.h"

// 列向量
template<uint N>
struct LargeVector{
    Float3 vec[N/3];

    LargeVector(){ for (uint i = 0; i < N/3; i++) { (*this)(i) = Zero3;} }
    LargeVector(float value){ for (uint i = 0; i < N/3; i++) { (*this)(i) = make<Float3>(value); } }
    LargeVector(const THREAD Float3& input_vec){ for (uint i = 0; i < N/3; i++) { (*this)(i) = input_vec; } }
    LargeVector(const THREAD Float3 input_vec[N/3]){ for (uint i = 0; i < N/3; i++) { (*this)(i) = input_vec[i]; } }
    LargeVector(const THREAD LargeVector<N>& input_vec){ for (uint i = 0; i < N/3; i++) { (*this)(i) = input_vec(i); } }
 #ifndef METAL_CODE
    LargeVector(std::initializer_list<float> list){ uint i = 0; for(auto it = list.begin(); it != list.end(); it++, i++) { vec[i/3][i%3] = *it; } }
#endif

    THREAD Float3& operator[](uint idx) { return vec[idx]; }
    THREAD Float3& operator()(uint idx) { return vec[idx]; }
    THREAD const Float3& operator[](uint idx) const { return vec[idx]; }
    THREAD const Float3& operator()(uint idx) const { return vec[idx]; }

    THREAD Float3* ptr(){ return vec; }
    THREAD const Float3* ptr() const { return vec; }
    Float3 to_Float3() { static_assert(N == 3, "Array is not Float3 Type!"); return vec[0]; }

    void operator=(const THREAD LargeVector<N>& input_vec) { for (uint i = 0; i < N/3; i++) { vec[i] = input_vec[i]; }} 
    LargeVector<N> operator*(float value) const{
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = vec[i] * value; }
        return output;
    }
    friend LargeVector<N> operator*(float value, const THREAD LargeVector<N>& right_vec) {
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = right_vec[i] * value; }
        return output;
    }
    LargeVector<N> operator+(CREF(Float3) value) const {
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = vec[i] + value; }
        return output;
    }
    LargeVector<N> operator+(const THREAD LargeVector<N>& right_vec) const {
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = vec[i] + right_vec[i]; }
        return output;
    }
    friend LargeVector<N> operator+(CREF(Float3) value, const THREAD LargeVector<N>& right_vec) {
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = right_vec[i] + value; }
        return output;
    }
    static LargeVector<N> Zero() {
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output[i] = Zero3; }
        return output;
    }
};


// M个VecN
template<uint M, uint N>
struct LargeMatrix{
    Float3x3 mat[M/3 * N/3];

    LargeMatrix(){ setHelper(Zero3x3); }
    LargeMatrix(float value){ setDiagonal(value); }
    LargeMatrix(const THREAD Float3x3& input_mat){ setHelper(input_mat); }
    LargeMatrix(const THREAD LargeMatrix<M, N>& input_mat){ setHelper(input_mat); }

          THREAD Float3x3& operator()(uint idx1, uint idx2)        { return mat[idx1 * (N/3) + idx2]; }
    const THREAD Float3x3& operator()(uint idx1, uint idx2) const  { return mat[idx1 * (N/3) + idx2]; }

          THREAD float& get_scalar(uint idx1, uint idx2)        { return get(mat[(idx1/3) * (N/3) + (idx2/3)], (idx1%3), (idx2%3)); }
    const THREAD float& get_scalar(uint idx1, uint idx2) const  { return get(mat[(idx1/3) * (N/3) + (idx2/3)], (idx1%3), (idx2%3)); }
    
    void setZero(){ setHelper(Float3x3(0.f)); }
    void setIdentity() { setDiag(Float3x3(1.f)); }
    void setDiagonal(float value) { setDiag(Float3x3(value)); }
    void set_column(uint columnIdx, const THREAD LargeVector<N>& input_vec){ for (uint i = 0; i < N/3; i++) { set((*this)(columnIdx/3, i), columnIdx%3, input_vec(i)); } }
    void set_column(uint columnIdx, const THREAD Float3& input_vec){ 
        static_assert(N == 3, "Num of Matrix Rows is not 3");
          set((*this)(columnIdx/3, 0), columnIdx%3, input_vec); 
    }
    LargeVector<N> get_column(uint columnIdx){ 
        LargeVector<N> output;
        for (uint i = 0; i < N/3; i++) { output(i) = get((*this)(columnIdx/3, i), columnIdx%3); }
        return output; 
    }
    static LargeMatrix<M, N> Zero() { return LargeMatrix<M, N>(); }


    // oparationg overload
    LargeMatrix<M, N> operator+(const THREAD LargeMatrix<M, N>& right_mat) const{
        LargeMatrix<M, N> output;
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                output(i, j) = (*this)(i, j) + right_mat(i, j);
        return output;
    }
    LargeMatrix<M, N> operator-(const THREAD LargeMatrix<M, N>& right_mat) const{
        LargeMatrix<M, N> output;
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                output(i, j) = (*this)(i, j) - right_mat(i, j);
        return output;
    }
    LargeMatrix<M, N> operator*(float value) const{
        LargeMatrix<M, N> output;
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                output(i, j) = (*this)(i, j) * value;
        return output;
    }
    friend LargeMatrix<M, N> operator*(float value, const THREAD LargeMatrix<M, N>& right_mat) {
        LargeMatrix<M, N> output;
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                output(i, j) = right_mat(i, j) * value;
        return output;
    }
    template<uint L> 
    LargeMatrix<L, N> operator*(const THREAD LargeMatrix<L, M>& right_mat) const{
        LargeMatrix<L, N> result;
        for (uint i = 0; i < L/3; i++) {
            for(uint j = 0; j < N/3; j++){
                result(i, j) = Zero3x3;
                for(uint k = 0; k < M/3; k++){
                    result(i, j) += (*this)(k, j) * right_mat(i, k);
                }
            }
        }
        return result;
    }
    LargeVector<N> operator*(const THREAD Float3& vec) const{
        static_assert(M == 3, "Can not multiply!");
        LargeVector<N> result;
        for(uint i = 0; i < N/3; i++){
            result(i) = (*this)(0, i) * vec;
        }
        return result;
    }
    friend LargeVector<N> operator*(const THREAD Float3& vec, const THREAD LargeMatrix<M, N>& right_mat) {
        static_assert(M == 3, "Can not multiply!");
        LargeVector<N> result;
        for(uint i = 0; i < N/3; i++){
            result(i) = right_mat(0, i) * vec;
        }
        return result;
    }
    // template<uint L>
    LargeVector<N> operator*(const THREAD LargeVector<M>& vec) const{
        // static_assert(M == L, "Can not multiply!");
        LargeVector<N> result;
        for (uint i = 0; i < N/3; i++) {
            result(i) = Zero3;
            for(uint j = 0; j < M/3; j++){
                result(i) += (*this)(j, i) * vec(j);
            }
        }
        return result;
    }

    // some methods
    LargeMatrix<N, M> transpose() const {
        LargeMatrix<N, M> output;
        for (uint i = 0; i < N/3; i++) 
            for(uint j = 0; j < M/3; j++)
                output(i, j) = transpose_mat((*this)(j, i));
        return output;
    }
    LargeMatrix<N, M> T() const { return transpose(); }

    // = left*(right)T
    static LargeMatrix<M, N> outerProduct(const THREAD LargeVector<N>& left_vec, const THREAD LargeVector<M>& right_vec){
        // 对于每一列，拿左向量乘以右向量的对应元素
        LargeMatrix<M, N> output;
        for(uint i = 0; i < M; i++){
            LargeVector<N> current_culumn = left_vec * right_vec(i/3)[i%3];
            output.set_column(i, current_culumn);
        }  
        return output;
    } 


private:
    void setHelper(CREF(Float3x3) input){
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                (*this)(i, j) = input;
    }
    void setHelper(const THREAD LargeMatrix<M, N>& input_mat){
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                (*this)(i, j) = input_mat(i, j);
    }
    void setDiag(CREF(Float3x3) input){
        static_assert(M == N, "Matrix is not Square Matrix");
        for (uint i = 0; i < M/3; i++) 
            for(uint j = 0; j < N/3; j++)
                if(i == j){
                    (*this)(i, j) = input;
                }
    }
};

using VECTOR9 = LargeVector<9>;
using VECTOR12 = LargeVector<12>;
using MATRIX9 = LargeMatrix<9, 9>;
using MATRIX12 = LargeMatrix<12, 12>;
using MATRIX12x3 = LargeMatrix<12, 3>;

// inline void outer_product_diag_12(CREF(VECTOR12) vec1, CREF(VECTOR12) vec2, THREAD Float3x3 diag[4]){
//     diag[0] = outer_product(vec1[0], vec2[0]);
//     diag[1] = outer_product(vec1[1], vec2[1]);
//     diag[2] = outer_product(vec1[2], vec2[2]);
//     diag[3] = outer_product(vec1[3], vec2[3]);
// }
// inline void outer_product_12(CREF(VECTOR12) vec1, CREF(VECTOR12) vec2, THREAD Float3x3 diag[10]){
    
    
// }

// 行向量
// template<uint N>
// struct LargeVectorT : LargeVector<N>{
//     LargeVectorT(const THREAD LargeVector<N>& input_vec){ for (uint i = 0; i < N/3; i++) override { (*this)(i) = input_vec[i]; } }
//     LargeVectorT(const THREAD LargeVectorT<N>& input_vec){ for (uint i = 0; i < N/3; i++) { (*this)(i) = input_vec[i]; } }

// };