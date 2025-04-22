#pragma once


#include <cstddef>
#include <iostream>
#include <format>
#include <vector>
#include "bits_utils.h"
#include "float_n.h"

inline void fast_print(){
    std::cout << std::endl;
}

template<typename T>
inline void fast_print(const T& object){
    std::cout << object << std::endl;
}
template<typename T>
inline void fast_print_err(const T& object){
    std::cerr << object << std::endl;
}
template<typename T1, typename T2>
inline void fast_print_err(const T1& obj1, const T2& obj2){
    std::cerr << obj1 << " : " << obj2 << std::endl;
}

template<typename T>
inline void fast_print_single(const T& object){
    std::cout << object << " " << std::flush;
}

template<typename... Args>
inline void fast_format(const std::string& str, Args... args){
    std::cout << std::vformat(str, std::make_format_args(args...)) << std::endl;
}
template<typename... Args>
inline void fast_format_err(const std::string& str, Args... args){
    std::cerr << std::vformat(str, std::make_format_args(args...)) << std::endl;
}
template<typename... Args>
inline void fast_format_single(const std::string& str, Args... args){
    std::cerr << std::vformat(str, std::make_format_args(args...)) << " " << std::flush;
}


// template<typename T>
// void fast_print(const std::vector<T>& vector_object){

//     if(vector_object.empty()) return;
    
//     std::cout << "vector data :";
//     size_t vector_length = vector_object.size();

//     for (size_t i = 0; i < min_scalar(vector_length, 256); i++) {
//         if(i % 32 == 0) std::cout << "\n";
//         std::cout << vector_object[i] << " ";
//     }

//     std::cout << "\n...............";

//     if(vector_length >= 512){
//         size_t start_pos = ((vector_length - 256) / 256) * 256; 
//         for (size_t i = start_pos; i < vector_length; i++) {
//             if(i % 32 == 0) std::cout << "\n";
//             std::cout << vector_object[i] << " ";
//         }
//     }
//     std::cout << std::endl;
// }

template<typename T, uint N>
inline void fast_print(const std::array<T, N>& vector_object){

    if(vector_object.empty()) return;
    
    std::cout << "vector data :";
    size_t vector_length = vector_object.size();

    for (size_t i = 0; i < vector_length; i++) {
        if(i % 32 == 0) std::cout << "\n";
        std::cout << vector_object[i] << " ";
    }

    std::cout << std::endl;
}

template<typename T, uint N>
inline void fast_print(const std::array<std::array<T, N>, N>& vector_object){

    if(vector_object.empty()) return;
    
    std::cout << "vector data :";

    for (size_t i = 0; i < N; i++) {
        std::cout << "\n";
        for (size_t j = 0; j < N; j++) {
            std::cout << vector_object[i][j] << " ";
        }
    }
    std::cout << std::endl;
}


template<typename T1, typename T2>
inline void fast_print(const T1& object1, const T2& object2){
    std::cout << object1 << " : " << object2 << std::endl;
}

template<typename T1, typename T2, typename T3>
inline void fast_print(const T1& object1, const T2& object2, const T3& object3){
    std::cout << object1 << " : " << object2  << " , " << object3 << std::endl;
}

template<typename T1, typename T2, typename T3, typename T4>
inline void fast_print(const T1& object1, const T2& object2, const T3& object3, const T4& object4){
    std::cout << object1 << " : " << object2  << " , " << object3 << " : " << object4 << std::endl;
}

template<size_t N, typename T1>
inline void fast_print(const T1& vec) {
    std::cout << " [ ";
    for (size_t j = 0; j < N - 1; j++) {
        std::cout << vec[j] << " , ";
    }
    std::cout << vec[N - 1] << " ] \n";
}

template<size_t N, typename T>
inline void fast_print(const std::vector<T>& vector_object){
    std::cout << "------- Vector Data-------";
    size_t vector_length = vector_object.size();

    for (size_t i = 0; i < min_scalar(vector_length, 64); i++) {
        if(i % 6 == 0) std::cout << "\n";

        T vec = vector_object[i];
        std::cout << "[";
        for (size_t j = 0; j < N - 1; j++) {
            std::cout << vec[j] << ",";
        }
        std::cout << vec[N - 1] << "] ";
    }

    if(vector_length >= 512){
        std::cout << "\n...............";
        size_t start_pos = ((vector_length - 64) / 64) * 64; 
        for (size_t i = start_pos; i < vector_length - 1; i++) {
            if(i % 6 == 0) std::cout << "\n";

            T vec = vector_object[i];
            std::cout << "[";
            for (size_t j = 0; j < N - 1; j++) {
                std::cout << vec[j] << ",";
            }
            std::cout << vec[N - 1] << "] ";
        }
    }
    std::cout << std::endl;
}

template<typename T, bool new_line = false>
inline void fast_print_iterator(const T& object, std::string name = "", uint max_count = 0){
    std::cout << "\n///////////////////////////////////\n";
    std::cout << name << " : \n";
    uint count = 0;
    for (const auto& value : object){
        std::cout << value << ", ";
        if (new_line && count % 20 == 19) { std::cout << "\n"; }
        count++;
        if (max_count != 0 && count == max_count - 1){ break; }
    }
    std::cout << "\n";
}


template<typename T>
inline void fast_print_radix(const T& object){
    uint bit_width = sizeof(T) * 8;
    // for (uint i = __builtin_fls(object) - 1; i >= 0; i--) {
    for (int i = bit_width; i >= 0; i--) {
        std::cout << ((object & (1u << i)) ? "1" : "0");
    }
    std::cout << std::endl;
}


template<typename T>
inline void fast_check_null(T object){
    if(object == nullptr){
        std::cerr << "object is null !!! \n";
    } 
}

template<typename T>
inline void fast_check_null(T object, std::string object_name){
    if(object == nullptr){
        std::cerr << "object : " << object_name << " is null !!! \n";
    } 
}

#ifdef __APPLE__
#include "Foundation/NSError.hpp"
#include "Foundation/NSString.hpp"
inline void print_err(NS::Error* err){
    if(err)
        __builtin_printf( "%s", err->localizedDescription()->utf8String() );
}

template<typename T>
inline void check_err(T data, NS::Error* err){
    if(!data){
        fast_print(err->localizedDescription()->utf8String());
        __builtin_printf( "%s", err->localizedDescription()->utf8String() );
        assert(false);
    }
}
#endif

template<typename T, typename ReduceFunc>
inline T fast_reduce(const std::vector<T>& vector_object, ReduceFunc func){
    T reduce_value = vector_object[0];
    for (const auto& value : vector_object) {
        func(reduce_value, value);
    }
    return reduce_value;
}

template<typename T>
inline T fast_max(const std::vector<T>& vector_object){
    return fast_reduce(vector_object, 
        [&](T& reduce_value, const T& input_value) -> void{
            reduce_value = max_scalar(reduce_value, input_value);
        });
}
template<typename T>
inline T fast_min(const std::vector<T>& vector_object){
    return fast_reduce(vector_object, 
        [&](T& reduce_value, const T& input_value) -> void{
            reduce_value = min_scalar(reduce_value, input_value);
        });
}
template<typename T, uint N>
inline T fast_max(const std::vector<T>& vector_object){
    return fast_reduce(vector_object, 
        [&](T& reduce_value, const T& input_value) -> void{
            reduce_value = max_vec(reduce_value, input_value);
        });
}

#define fast_for(max_val) for(uint i = 0; i < max_val; i++)
#define fast_for2(max_i, max_j)             \
    for(uint i = 0; i < max_i; i++)         \
        for(uint j = 0; j < max_j; j++)

#define for_loop(name,range) for(size_t name = 0; name < range; name++)

template<typename T>
inline void fast_set(std::vector<T>& vec,const T&& value){
    std::fill(vec.begin(), vec.end(), value);
}


// #define parallel_for \
//         #pragma omp for

// #define parallel_reduce \
//         #pragma omp parallel for reduction(+:sum)

