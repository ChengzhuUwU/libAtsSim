#pragma once

#undef max
#undef min
// #include <iostream>
// #include <omp.h>
#include <tbb/tbb.h>
#include "scalar.h"
#include <bits_utils.h>

// namespace SimParallel{

// }


// ------------------- openmp ------------------- //
extern int CPU_THREAD_NUM;

// inline int get_cpu_thread_num() { return CPU_THREAD_NUM; }
// inline int get_cpu_process_num() { return omp_get_num_procs(); }
// inline int get_current_thread_id() { return omp_get_thread_num(); }

// #define STR(x) #x

// #define omp_barrier _Pragma(STR(omp barrier))
// #define omp_single _Pragma(STR(omp single))
// #define omp_parallel _Pragma(STR(omp parallel num_threads(CPU_THREAD_NUM)))
// #define omp_for _Pragma(STR(omp for))
// #define omp_parallel_for _Pragma(STR(omp parallel for num_threads(CPU_THREAD_NUM)))

// #define omp_parallel_for_reduction(reduction_op, value) _Pragma(STR(omp parallel for num_threads(CPU_THREAD_NUM) reduction(reduction_op:value)))
// #define omp_parallel_for_reduction_sum(sum) omp_parallel_for_reduction(+, sum)

// ------------------- tbb ------------------- //

#define tbb_begin(range_start, range_end, index) tbb::parallel_for(tbb::blocked_range<uint>(range_start, range_end), \
    [&](tbb::blocked_range<uint> r) { \
    for (uint index = r.begin(); index < r.end(); index++) { 
        
#define tbb_end }});

#define tbb_fast_begin(range, index) tbb_begin(0, range, index)

struct ParallelRange{
    uint start_idx = 0;
    uint end_idx = 1;
    ParallelRange() : start_idx(0), end_idx(1) {}
    ParallelRange(uint end_index) : start_idx(0), end_idx(end_index){}
    ParallelRange(uint start_index, uint end_index) : start_idx(start_index), end_idx(end_index){}
    ParallelRange(std::initializer_list<uint> list) : start_idx(*list.begin()), end_idx(*(list.begin() + 1)){}
    uint get_width(){
        return end_idx - start_idx;
    }
    void set_range(uint end_index){
        start_idx = 0;
        end_idx = end_index;
    }
    void set_range(uint start_index, uint end_index){
        start_idx = start_index;
        end_idx = end_index;
    }
    
    uint start_blockIdx(uint blockDim = 256){
        return start_idx / blockDim;
    }
    uint end_blockIdx(uint blockDim = 256){
        return (end_idx + blockDim - 1) / blockDim;
    }
    void print_range(){
        printf("range = [%d , %d]", start_idx, end_idx);
    }
};

// template<typename FuncName, typename... Args>
// void parallel_for(uint start_thread, uint end_index, FuncName func, uint blockDim, Args... args){
//     parallel_for_with_fixed_range(start_thread, end_index, 256, [&](uint startIdx, uint endIdx){
//         for (uint i = start_thread; i < end_index; i++) {
//             func(i, args...);
//         }
//     }, blockDim, args...);
// }

template<typename FuncName>
void parallel_for(uint start_pos, uint end_pos, FuncName func, const uint blockDim = 256){

    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;

    tbb::parallel_for(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), 
        [&](tbb::blocked_range<uint> r) { 
            uint blockIdx = r.begin();
            uint startIdx = std::max(blockDim * blockIdx, start_pos);
            uint endIdx = std::min(blockDim * (blockIdx + 1), end_pos);
            for (uint index = startIdx; index < endIdx; index++) {
                func(index); 
            }
        }, tbb::simple_partitioner{});
}

// template<typename FuncName>
// void parallel_for(uint start_thread, uint end_index, FuncName func, Args... args){
//     // static_assert(!std::is_same<FuncName, int>::value || !std::is_same<FuncName, unsigned int>::value, "Function Should Not Be Value");
//     parallel_for(start_thread, end_index, blockDim, func, args...);
// }

template<typename FuncName>
void parallel_for(ParallelRange range, FuncName func){
    parallel_for(range.start_idx, range.end_idx, func);
}

template<typename FuncName>
void single_thread_for(ParallelRange range, FuncName func, const uint blockDim = 32){
    for(uint index = range.start_idx; index < range.end_idx; index++){ func(index);  }
}
template<typename FuncName>
void single_thread_for(uint start_idx, uint end_idx, FuncName func, const uint blockDim = 32){
    for(uint index = start_idx; index < end_idx; index++){ func(index); }
}

// 传入的是一个range的函数，也就是需要带个for循环
template<typename FuncName>
void parallel_for_in_block(uint start_pos, uint end_pos, uint blockDim, FuncName func){
    
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;
    
    tbb::parallel_for(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), 
        [&](tbb::blocked_range<uint> r) { 
            uint blockIdx = r.begin();
            uint startIdx = std::max(blockDim * blockIdx, start_pos);
            uint endIdx = std::min(blockDim * (blockIdx + 1), end_pos);
            func(startIdx, endIdx);
        }, 
        tbb::simple_partitioner{});
}

template<typename FuncName>
void parallel_for_each_core(uint start_core_idx, uint end_core_idx, FuncName func){
    
    tbb::parallel_for(tbb::blocked_range<uint>(start_core_idx, end_core_idx, 1), 
        [&](tbb::blocked_range<uint> r) { 
            uint blockIdx = r.begin();
            func(blockIdx);
        }, 
        tbb::simple_partitioner{});
}

template<typename T, typename ParallelFunc, typename ReduceFuncBinary>
inline T parallel_for_and_reduce(uint start_pos, uint end_pos, ParallelFunc func_parallel, ReduceFuncBinary func_binary, const T zero){
    const uint blockDim = 256;
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;
    return tbb::parallel_reduce(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), zero, 
        [&]( tbb::blocked_range<uint> r, T result ) {

            uint blockIdx = r.begin();
            uint startIdx = std::max(blockDim * blockIdx, start_pos);
            uint endIdx = std::min(blockDim * (blockIdx + 1), end_pos);

            for (uint index = startIdx; index < endIdx; index++) {
                T parallel_result = func_parallel(index);
                result = func_binary(result, parallel_result);
                // func_binary(result, parallel_result);
            }
            return result;  
        }, 
        func_binary, 
        tbb::simple_partitioner{} );

    // func_binary = [](T& result, const T& parallel_result) -> void { result = func_unary(parallel_result, parallel_result); }

}

template<typename T, typename ParallelFunc>
inline T parallel_for_and_reduce_sum(uint start_pos, uint end_pos, ParallelFunc func_parallel){
    return parallel_for_and_reduce<T>(start_pos, end_pos, 
        func_parallel, 
        // [](T& result, const T& parallel_result) -> void { result += parallel_result; }, // func_unary
        [](const T& x, const T& y) -> T{return x + y;}, // func_binary
        T()
        ); 
}
template<typename T, typename ParallelFunc>
inline T parallel_for_and_reduce_max(uint start_pos, uint end_pos, ParallelFunc func_parallel){
    return parallel_for_and_reduce<T>(start_pos, end_pos, 
        func_parallel, 
        // [](T& result, const T& parallel_result) -> void { result = std::max(parallel_result, parallel_result); }, // func_unary
        [](const T& x, const T& y) -> T{ return std::max(x, y); }, // func_binary
        std::numeric_limits<T>::lowest()
        ); 
}
template<typename T, typename ParallelFunc>
inline T parallel_for_and_reduce_min(uint start_pos, uint end_pos, ParallelFunc func_parallel){
    return parallel_for_and_reduce<T>(start_pos, end_pos, 
        func_parallel, 
        // [](T& result, const T& parallel_result) -> void { result = std::min(parallel_result, parallel_result); }, // func_unary
        [](const T& x, const T& y) -> T{ return std::min(x, y); }, // func_binary
        std::numeric_limits<T>::max()
        ); 
}

// inclusive : 包含第一个元素
template<typename T, typename ParallelFunc, typename OutputFunc>
inline void parallel_for_and_scan(uint start_pos, uint end_pos, ParallelFunc func_parallel, OutputFunc func_output, const T& zero){

    const uint blockDim = 256;
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;

    tbb::parallel_scan(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), zero, 
        [&]( tbb::blocked_range<uint> r, T block_prefix, auto is_final_scan) -> T{

            uint start_blockIdx = r.begin();
            uint end_blockIdx = r.end() - 1;

            uint startIdx = std::max(blockDim * start_blockIdx, start_pos);
            uint endIdx   = std::min(blockDim * (end_blockIdx + 1), end_pos);

            for (uint index = startIdx; index < endIdx; index++) {
                T parallel_result = func_parallel(index);
                block_prefix += parallel_result;
                if(is_final_scan) {
                    func_output(index, block_prefix, parallel_result);
                }
            }
            return block_prefix;
        }, 
        [](const T& x, const T& y) -> T{return x + y;},
        tbb::simple_partitioner{} );

    // tbb::parallel_scan(tbb::blocked_range<uint>(start_pos, end_pos), zero, 
    // [&]( tbb::blocked_range<uint> r, T block_prefix, auto is_final_scan) -> T{
    //     for (auto i = r.begin(); i != r.end(); ++i) {
    //         T parallel_result = func_parallel(i);
    //         block_prefix += parallel_result;
    //         if(is_final_scan) {
    //             func_output(i, block_prefix, parallel_result);
    //         }   
    //     }
    //     return block_prefix;
    // }, 
    // [](const T& x, const T& y) -> T{return x + y;} );

}

template <typename T>
inline void parallel_copy(T* dst, const T* src, const uint array_size)
{
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = src[index];
    });
}
template <typename T>
inline void parallel_set(T* dst, const uint array_size, const T& value)
{
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = value;
    });
}

template<typename T>
static inline bool default_compate(const T& left, const T& right){
    return left < right;
}

template <typename Ptr, typename _Comp>
inline void parallel_sort(Ptr begin, Ptr end, _Comp comp = default_compate){
    tbb::parallel_sort(begin, end, comp);
}


// [](float& x, const float& y) -> void{ x += y; },
// [](const float& x, const float& y) -> float{ return x + y; }
