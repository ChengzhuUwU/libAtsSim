#pragma once

#include <cstring>
#include <iostream>
#include <vector>
#include "address_space.h"
#include "cpu_parallel.h"
#include <tbb/tbb.h>
#include "metal_system.h"
#include "scalar.h"
#include "utils.h"

#include "float_n.h"
#include "float_n_n.h"

#define SAFETY_CHECK false

namespace SHARED_ARRAY {
    extern uint memory_cost_total;
    extern uint memory_cost_solver;
    extern uint memory_cost_collision;
#if __APPLE__
    extern std::unordered_map<void*, void*> cpuPtr_to_gpuPtr_map;
    extern std::unordered_map<void*, void*> gpuPtr_to_cpuPtr_map;
#endif

    // template<typename T1, typename T2>
    // static inline void push_to_vector(T1& ptr, T2& buffer){
    //     #if __APPLE__
    //         gpuPtr_array_map[ptr] = buffer;
    //     #endif
    // }

};
using namespace SHARED_ARRAY;

template<typename T> class SharedArray
{

private:

#if __APPLE__
    MTL::Buffer* gpu_buffer = nullptr;
#elif _WIN32
    T* gpu_buffer = nullptr;
#endif

    T* cpu_ptr = nullptr;
    uint length = 0;
    uint byte_length = 0;
    uint width = 0;
    
public:   



#if __APPLE__
    inline void resize(uint input_length, 
                       unsigned long resource_mode = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked)
    {
        if(length != 0) {
            release();
            std::cerr << "array is not empty!!!\n";
        }
        if(input_length == 0){
            std::cerr << "Length of Array is Zero!!!\n" ;
            input_length = 1;
        }

        length = input_length;
        byte_length = length * sizeof(T);

        if(byte_length > 1024 * 1024 * 100) { // 100 MB
            // std::cerr << "array is too large!!!\n";
            if(byte_length > 1024 * 1024 * 1024 * 3u){ // 3GB
                std::cerr << "Out of buffer range !!! : " << byte_length / (1024 * 1024 * 1024) << " gb \n";
                return;
            }
        }
        // get_device()->newBuffer()
        gpu_buffer = get_device() -> newBuffer(byte_length, resource_mode);  
        cpu_ptr = static_cast<T*>(gpu_buffer -> contents()); // cpu_ptr = reinterpret_cast<T*>(gpu_buffer -> contents());
        memory_cost_total += gpu_buffer -> length();
        
        void* gpu_addr = (void*)gpu_buffer->gpuAddress();
        cpuPtr_to_gpuPtr_map.insert(std::make_pair(cpu_ptr, gpu_addr));
        gpuPtr_to_cpuPtr_map.insert(std::make_pair(gpu_addr, cpu_ptr));
        // cpuPtr_to_gpuPtr_map[cpu_ptr] = (void*)gpu_buffer->gpuAddress();
        // gpuPtr_to_cpuPtr_map[(void*)gpu_buffer->gpuAddress()] = cpu_ptr;

    }
#elif _WIN32
    // #define IS_BOOL(T) std::is_same<T, std::atomic<bool>>::value || std::is_same<T, bool>::value
    // #define NOT_BOOL(T) !std::is_same<T, std::atomic<bool>>::value && !std::is_same<T, bool>::value
    // template<typename U = T, typename std::enable_if<NOT_BOOL(U), int>::type = 0>

    inline void resize(uint input_length, 
                       unsigned long resource_mode = 0)
    {
        if(length != 0) 
            std::cerr << "array is not empty!!!\n";

        length = input_length;
        byte_length = length * sizeof(T);

        if(byte_length > 1024 * 1024 * 100) { // 100 MB
            // std::cerr << "array is too large!!!\n";
            if(byte_length > 1024 * 1024 * 1024 * 3u){ // 3GB
                std::cerr << "Out of buffer range !!! : " << byte_length / (1024 * 1024 * 1024) << " gb \n";
                return;
            }
        }
        
        gpu_buffer = new T[length];
        cpu_ptr = gpu_buffer;
        memory_cost_total += byte_length;
    }

    // // bool & std::atomic<bool>
    // template<typename U = T, typename std::enable_if<IS_BOOL(U), int>::type = 0>
    // inline void resize(uint input_length, 
    //                    unsigned long resource_mode = 0)
    // {
    //     length = input_length;
    //     byte_length = length * sizeof(T);

    //     if(byte_length > 1024 * 1024 * 10) std::cerr << "array is too large!!!\n";
        
    //     cpu_ptr = new T(length);
    //     memory_cost_total += byte_length;
    // }

#endif


    inline void release(){
        length = 0;
        byte_length = 0;
        cpu_ptr = nullptr;

        #if __APPLE__
        gpu_buffer -> release();
        #elif _WIN32
        // gpu_buffer.shrink_to_fit();
        delete[] gpu_buffer;
        #endif
        gpu_buffer = nullptr;

    }
    inline bool is_empty() const { return size() == 0; }
    
    // 可能buffer已经分配过空间了，这时候可以做一下判断再resize
    inline void check_size(uint input_length){ if(length != input_length){ this -> resize(input_length); }}
    inline void check_size(const std::vector<T>& input_data){ this -> check_size(static_cast<uint>(input_data.size())); }

    // 一维数组upload
    inline void upload(const std::vector<T>& input_data) { 
        this -> check_size(input_data);
        std::memcpy(cpu_ptr, input_data.data(), byte_length);  
    }
    inline void download(std::vector<T>& input_data) const { 
        if(length != input_data.size()) input_data.resize(length);
        std::memcpy(input_data.data(), cpu_ptr, byte_length);  
    }
    inline void upload(const T* input_data, uint input_length) { 
        this -> check_size(input_length);
        std::memcpy(cpu_ptr, input_data, byte_length);  
    }

    // 二维数组upload
    inline uint upload_2d_ell(const std::vector<std::vector<uint>>& input_map){

        uint num_outer = static_cast<uint>(input_map.size());
        uint max_count = 0;

        std::vector<uint> num_inner_vec(num_outer);

        for (uint i = 0; i < num_outer; i++) {
            const auto& inner_list = input_map[i];
            uint num_inner= static_cast<uint>(inner_list.size());
            num_inner_vec[i] = num_inner;
            max_count = max_scalar(num_inner, max_count);
        }

        this -> check_size((max_count + 1) * num_outer);
        std::memcpy(this -> cpu_ptr, num_inner_vec.data(), num_outer * sizeof(uint)); // 将第一列设置为相邻元素个数

        for (uint i = 0; i < num_outer; i++) {
            const auto& inner_list = input_map[i];
            // if(inner_list.size() > 32)  std::cerr << "single element is too large : " << inner_list.size() <<"\n";
            for (uint j = 0; j < inner_list.size(); j++) {
                this -> cpu_ptr[(j + 1) * num_outer + i] = inner_list[j];
            }
        }
        width = max_count;
        return max_count;
    }
    inline uint upload_2d_csr(const std::vector<std::vector<uint>>& input_map){
        uint num_outer = static_cast<uint>(input_map.size());
        uint current_prefix = num_outer + 1;
        
        std::vector<uint> prefix_list(num_outer + 1);

        uint max_count = 0;
        for (uint i = 0; i < num_outer; i++) {
            const auto& inner_list = input_map[i];
            uint num_inner = static_cast<uint>(inner_list.size()); max_count = max_scalar(max_count, num_inner);
            prefix_list[i] = current_prefix;
            current_prefix += num_inner;
        }
        uint num_data = current_prefix;
        prefix_list[num_outer] = current_prefix;
        
        this -> check_size(num_data); // 前n+1个位置存储prefix
        std::memcpy(this -> cpu_ptr, prefix_list.data(), (num_outer + 1) * sizeof(uint)); // 将第一列设置为相邻元素个数

        for (uint i = 0; i < num_outer; i++) {
            const auto& inner_list = input_map[i];
            uint current_prefix = prefix_list[i];
            uint current_end = prefix_list[i + 1];
            // if(inner_list.size() > 32)  std::cerr << "single element is too large : " << inner_list.size() <<"\n";
            for (uint j = current_prefix; j < current_end; j++) {
                this -> cpu_ptr[j] = inner_list[j - current_prefix];
            }
        }
        return max_count;
        // return num_data - (num_outer + 1);
    }
    inline void download_2d_ell(std::vector<std::vector<uint>>& input_map) const {
        
        uint num_outer = length / (width + 1);
        if(input_map.empty()){ input_map.resize(num_outer); }
        else if(input_map.size() != num_outer){ std::cerr << "Size Does Not Correspond!!" << std::endl; return; }
        for (uint i = 0; i < num_outer; i++) {
            auto& inner_list = input_map[i];
            uint num_adj = cpu_ptr[i];
            inner_list.resize(num_adj);
            for (uint j = 0; j < num_adj; j++) {  inner_list[j] = this -> cpu_ptr[(j + 1) * num_outer + i];  }
        }
    }
    inline void download_2d_csr(std::vector<std::vector<uint>>& input_map, const uint num_outer) const {
        
        if (input_map.empty()) { input_map.resize(num_outer); }
        else if (input_map.size() != num_outer){ std::cerr << "Size Does Not Correspond!!" << std::endl; return; }

        for (uint i = 0; i < num_outer; i++) {
            auto& inner_list = input_map[i];
            const uint curr_prefix = this->cpu_ptr[i];
            const uint next_prefix = this->cpu_ptr[i + 1];
            uint num_adj = next_prefix - curr_prefix;
            inner_list.resize(num_adj);
            for (uint j = 0; j < num_adj; j++) {  inner_list[j] = this -> cpu_ptr[curr_prefix + j];  }
        }
    }
    inline void download_2d_ell(std::vector<tbb::concurrent_vector<uint>>& input_map) const {
        
        uint num_outer = length / (width + 1);
        if(input_map.empty()){ input_map.resize(num_outer); }
        else if(input_map.size() != num_outer){ std::cerr << "Size Does Not Correspond!!" << std::endl; return; }
        for (uint i = 0; i < num_outer; i++) {
            auto& inner_list = input_map[i];
            uint num_adj = cpu_ptr[i];
            inner_list.resize(num_adj);
            for (uint j = 0; j < num_adj; j++) {  inner_list[j] = this -> cpu_ptr[(j + 1) * num_outer + i];  }
        }
    }

    #if __APPLE__
    inline void point_to(MTL::Buffer* buffer) { 
        byte_length = buffer -> length();
        length = byte_length / sizeof(T);
        
        gpu_buffer = buffer;  
        cpu_ptr = static_cast<T*>(gpu_buffer -> contents());
    }
    #endif
    
    inline void point_to(const SharedArray<T>& array) { 
        byte_length = array.get_byte_size();
        length = array.size();
        gpu_buffer = array.buffer();

        cpu_ptr = array.data();
    }

public:
    inline void set_zero(uint begin_idx = 0, uint end_idx = 0) { if(end_idx == 0) end_idx = length; std::memset(cpu_ptr + begin_idx, 0, (end_idx - begin_idx) * sizeof(T)); }
    inline void set_max() { std::memset(cpu_ptr, 0xFF, byte_length); }
    inline void set_data(const T& value){ std::fill(cpu_ptr, cpu_ptr + length, value); }

    #if __APPLE__
    inline void set_zero_async(MTL::CommandBuffer* command_buffer) {
        auto blit_encoder = command_buffer -> blitCommandEncoder();
        fast_check_null(blit_encoder, "blit_encoder");
        blit_encoder->fillBuffer(gpu_buffer, NS::Range(0, byte_length), 0);
        blit_encoder -> endEncoding();
        command_buffer->commit();
    }
    #endif

public:
    inline const uint size() const{ return length; }  
    inline const uint get_width() const { return width;}
    inline const uint get_byte_size() const  { return byte_length; }
    inline constexpr uint type_size() const  { return sizeof(T);}

    // inline void check_index_valid(uint index)             { if(index >= this -> length) printf("index out range!!"); }
    inline void check_index_valid(uint index) const { if(index >= this -> length) std::cerr << "index out range!!"; }

    #if __APPLE__
    inline MTL::Buffer* buffer()       { return gpu_buffer; }
    inline MTL::Buffer* buffer() const { return gpu_buffer; }
    #elif _WIN32
    inline T* buffer()       { return gpu_buffer; }
    inline T* buffer() const { return gpu_buffer; }
    #endif

    inline T* ptr()              { return cpu_ptr; }
    inline T* ptr()  const { return cpu_ptr; }
    
    inline T* data()             { return cpu_ptr; }
    inline T* data() const { return cpu_ptr; } 

    #if __APPLE__
    // Only Useful For MTLStorageModeManaged...
    inline void sync_to_gpu(uint start_idx = 0, uint width = 0) {
        if(width == 0) width = length;
        if(! (gpu_buffer->storageMode() & MTL::StorageModeManaged)) { fast_print_err(" Function \"didModifyRange\" May Be Only Useful For MTLStorageModeManaged... "); }
        gpu_buffer -> didModifyRange(NS::Range(start_idx * sizeof(T), width * sizeof(T))); 
    }
    #endif

    // inline T& get_first()              { return cpu_ptr[0]; }
    // inline const T& get_first() const  { return cpu_ptr[0]; }

    inline T& operator[](uint index) { 
#if SAFETY_CHECK
        check_index_valid(index); 
#endif
        return cpu_ptr[index]; 
    }

    inline const T& operator[] (uint index) const { 
#if SAFETY_CHECK
        check_index_valid(index); 
#endif
        return cpu_ptr[index]; 
    }

    inline T* begin() { return cpu_ptr; }
    inline T* end() { return cpu_ptr + size(); }
    template<typename Func> inline void for_each(Func func, bool use_parallel = true){
        if(use_parallel){ parallel_for(0, this->length, [&](uint i) { func(this -> cpu_ptr[i]); }); }
        else            { for (uint i = 0; i < length; i++)         { func(this -> cpu_ptr[i]); } }
    }
    template<typename Func> inline void for_each(Func func, bool use_parallel = true) const{
        if(use_parallel){ parallel_for(0, this->length, [&](uint i) { func(this -> cpu_ptr[i]); }); }
        else            { for (uint i = 0; i < length; i++)         { func(this -> cpu_ptr[i]); } }
    }

    // // atomic
    // inline std::atomic<T>* atomic_ptr()              {  return cpu_ptr_atomic; }
    // inline const std::atomic<T>* atomic_ptr() const  {  return cpu_ptr_atomic; }
    // inline T atomic_add(const uint& addr, const T& value){ return AtomicAdd(cpu_ptr_atomic, addr, value); }
    // inline T atomic_sub(const uint& addr, const T& value){ return AtomicSub(cpu_ptr_atomic, addr, value); }
    // inline T atomit_load(const uint& addr) { return cpu_ptr_atomic[addr].load(); }
    // inline void atomit_store(const uint& addr, const T& value) { cpu_ptr_atomic[addr].store(value); }


public:


inline void print() {
    std::cout << "------- Array Data-------";
    for (uint i = 0; i < min_scalar(length, 256); i++) {
        if(i % 32 == 0) std::cout << "\n";
        std::cout << (this -> cpu_ptr[i]) << " ";
    }
    std::cout << "\n";
}

// using ToStringFunc = std::function<std::string(T)>;
template<typename ToStringFunc>
inline void print(ToStringFunc to_string_func) const{
    std::cout << "------- Array Data-------";
    for (uint i = 0; i < min_scalar(length, 256); i++) {
        if(i % 32 == 0) std::cout << "\n";
        std::cout << to_string_func(this -> cpu_ptr[i]);
    }
    std::cout << "\n";
}

template<uint N, typename PrintType = float>
inline void print_vec(uint print_length = 256) const{
    std::cout << "------- Vector Array Data-------";
    for (uint i = 0; i < min_scalar(length, print_length); i++) {
        if(i % 32 == 0) std::cout << "\n";
        T current_data = this -> cpu_ptr[i];
        std::cout << " [";
        for (uint j = 0; j < N - 1; j++) {
            std::cout << (PrintType)current_data[j] << ",";
        }
        std::cout << (PrintType)current_data[N - 1] << "] ";
    }
    std::cout << "\n";
}

template<uint N>
inline void print_mat(uint print_length = 32) {
    std::cout << "------- Matrix Array Data-------";
    for (uint i = 0; i < print_length; i++) {
        if(i % 4 == 0) std::cout << "\n";
        T current_data = this -> cpu_ptr[i];
        std::cout << " { ";
        for (uint j = 0; j < N; j++) {
            std::cout << "[";
            for (uint k = 0; k < N - 1; k++)
                std::cout << get(current_data, j, k) << ",";
            std::cout << get(current_data, j, N - 1) << "] ";
        }
        std::cout << "} ";
    }
    std::cout << "\n";
}

template<uint N>
inline void print_mat(uint start_pos, uint end_pos) {
    std::cout << "------- Matrix Array Data-------";
    for (uint i = start_pos; i < end_pos; i++) {
        if(i % 4 == 0) std::cout << "\n";
        T current_data = this -> cpu_ptr[i];
        std::cout << " { ";
        for (uint j = 0; j < N; j++) {
            std::cout << "[";
            for (uint k = 0; k < N - 1; k++)
                std::cout << get(current_data, j, k) << ",";
            std::cout << get(current_data, j, N - 1) << "] ";
        }
        std::cout << "} ";
    }
    std::cout << "\n";
}

template<uint M, uint N>
inline void check_nan(){
    bool has_nan = false;
    for (uint i = 0; i < this -> length; i++) {
        T value = this -> cpu_ptr[i];
        for(uint j = 0; j < M; j++){
            for(uint k = 0; k < N; k++){
                if( std::isnan(value.columns[j][k]) ){
                    has_nan = true;
                    printf("Array has NAN %d\n", i);
                    break;
                }
            }
        }
    }
}

template<uint N>
inline void check_nan(){
    bool has_nan = false;
    for (uint i = 0; i < this -> length; i++) {
        auto data = this -> cpu_ptr[i];
        for(uint j = 0; j < N; j++){
            if(std::isnan(data[j])){
                has_nan = true;
                printf("Array has NAN %d\n", i);
                break;
            }
        }
    }
}

// inline void print_2d(uint print_length = 256) const{
//     std::cout << "------- Array2D Data-------";
//     for (uint i = 0; i < min_scalar(length, 256); i++) {
//         if(i % 32 == 0) std::cout << "\n";
//         std::cout << to_string_func(this -> cpu_ptr[i]);
//     }
//     std::cout << "\n";
// }

// this = array
inline void operator=(const SharedArray<T>& right_array) {
    std::memcpy(cpu_ptr, right_array.data(), byte_length);
}

// this += array
inline SharedArray<T>& operator+=(const SharedArray<T>& right_array) {
    parallel_for(0, this->length, [&](uint i) {
        this->cpu_ptr[i] += right_array[i]; });
    return *this;
}

// this += value
inline SharedArray<T>& operator+=(T&& input_value) {
    parallel_for(0, this->length, [&](uint i) {
        this->cpu_ptr[i] += input_value; });
    return *this;
}

// this .*= array
inline SharedArray<T>& operator*=(const SharedArray<T>& right_array) {
    parallel_for(0, this->length, [&](uint i) {
        this->cpu_ptr[i] *= right_array[i]; });
    return *this;
}

// this .*= value
inline SharedArray<T>& operator*=(T&& input_value) {
    parallel_for(0, this->length, [&](uint i) {
        this->cpu_ptr[i] *= input_value; });
    return *this;
}

// this ./= array
inline SharedArray<T>& operator/=(const SharedArray<T>& right_array) {
    parallel_for(0, this->length, [&](uint i) {
        this->cpu_ptr[i] /= right_array[i]; });
}

// this .*= value
inline SharedArray<T>& operator/=(T&& input_value) {
    parallel_for(0, this -> length, [&](uint i) {
        this->cpu_ptr[i] /= input_value; });
    return *this;
}

inline friend float dot(const SharedArray<T>& left_array, const SharedArray<T>& right_array) {
    return parallel_for_and_reduce_sum<float>(0, left_array.size(), [&](uint i) {
        return dot_vec(left_array[i], right_array[i]);
        });
}

// output = input1 + input2
friend void add(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] + input_array2[i]; });
}

// sort
template <class _Comp>
void sort(_Comp comp){
    parallel_sort(begin(), end(), comp);
}

// output = input1 + input2 * scale
inline friend void add(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2, float scale){
    parallel_for(0, output_array.size(),[&](uint i) { 
        output_array[i] = input_array1[i] + input_array2[i] * scale;});
}

// output = input1 + value
inline friend void add(SharedArray<T>& output_array, const SharedArray<T>& input_array1, float value){
    parallel_for(0, output_array.size(), [&](uint i) { 
        output_array[i] = input_array1[i] + value; });
}

// output = input1 - input2
inline friend void minus(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] - input_array2[i]; });
}

// output = input1 - input2 * scale
inline friend void minus(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2, float scale){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] - input_array2[i] * scale; });
}

// output = input1 .* input2
template<typename T1, typename T2>
inline friend void mul(SharedArray<T>& output_array, const SharedArray<T1>& input_array1, const SharedArray<T2>& input_array2){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] * input_array2[i]; });
}

// output = input1 /* scale
inline friend void mul(SharedArray<T>& output_array, const SharedArray<T>& input_array1, float value){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] * value; });
}

// output = input1 ./ input2
inline friend void devide(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] / input_array2[i]; });
}

// output = input1 ./ scale
inline friend void devide(SharedArray<T>& output_array, const SharedArray<T>& input_array1, float value){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = input_array1[i] / value; });
}

inline friend void get_inverse(SharedArray<T>& output_array, const SharedArray<T>& input_array1){
    parallel_for(0, output_array.size(), [&](uint i) {
        output_array[i] = inverse_mat(input_array1[i]); });
}

inline T get_min(uint begin_pos = 0, uint end_pos = 0) const {
    if(end_pos == 0) end_pos = this -> length;
    T value = cpu_ptr[begin_pos];
    for (uint i = begin_pos + 1; i < end_pos; i++) {
        value = min_vec(value, cpu_ptr[i]);
    }
    return value;
    /*return parallel_reduce(
        cpu_ptr, 0, size(),
        [&](T& x, const T& y) -> void{ x = min_vec(x, y); },
        [&](const T& x, const T& y) -> T{ return min_vec(x, y); }
    );*/
}

inline T get_max(uint begin_pos = 0, uint end_pos = 0) const {
    if(end_pos == 0) end_pos = this -> length; 
    T value = cpu_ptr[begin_pos];
    for (uint i = begin_pos + 1; i < end_pos; i++) {
        value = max_vec(value, cpu_ptr[i]);
    }
    return value;
    /*return parallel_reduce(
        cpu_ptr, 0, size(),
        [](T& x, const T& y) -> void{ x = max_vec(x, y); },
        [](const T& x, const T& y) -> T{ return max_vec(x, y); }
    );*/
}

inline T get_sum(uint begin_pos = 0, uint end_pos = 0) const {
    if(end_pos == 0) end_pos = this -> length;
    return parallel_for_and_reduce_sum<T>(begin_pos, end_pos, [&](uint i) { return cpu_ptr[i]; });
    /*return parallel_reduce(
        cpu_ptr, begin_pos, end_pos,
        [](T& x, const T& y) -> void{ x += y; },
        [](const T& x, const T& y) -> T{ return x + y; }
    );*/
}


};


namespace SHARED_ARRAY
{

inline void print_memory_cost_total(){
    if(memory_cost_total < 1024 * 1024){
        std::cout << "memory cost : " << memory_cost_total / 1024 << " kb \n";
    }
    else if(memory_cost_total < 1024 * 1024 * 1024){
        std::cout << "memory cost : " << memory_cost_total / (1024 * 1024) << " mb \n";
    }
    else{
        std::cout << "memory cost : " << memory_cost_total / (1024 * 1024 * 1024) << " gb \n";
    }
}

#if __APPLE__
    template<typename T>
    inline T* get_gpuPtr_from_cpuPtr(T* ptr){
        void* tmp = static_cast<void*>(ptr);
        if(cpuPtr_to_gpuPtr_map.find(tmp) != cpuPtr_to_gpuPtr_map.end()){
            return cpuPtr_to_gpuPtr_map[tmp];
        }
        else {
            fast_print("buffer is empty");
            return nullptr;
        }
    }
    template<typename T>
    inline T* get_cpuPtr_from_gpuPtr(T* ptr){
        void* tmp = static_cast<void*>(ptr);
        if(gpuPtr_to_cpuPtr_map.find(tmp) != gpuPtr_to_cpuPtr_map.end()){
            return static_cast<T*>(gpuPtr_to_cpuPtr_map[tmp]);
        }
        else {
            fast_print("buffer is empty");
            return nullptr;
        }
    }
#endif

}
