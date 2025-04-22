#include "shared_array.h"
#include "utils.h"



namespace SHARED_ARRAY {
    uint memory_cost_total;
    uint memory_cost_solver;
    uint memory_cost_collision;
#if __APPLE__
    std::unordered_map<void*, void*> cpuPtr_to_gpuPtr_map;
    std::unordered_map<void*, void*> gpuPtr_to_cpuPtr_map;
#endif


};

// 定义一个全局的shared_device
// namespace SHARED_ARRAY {
    
//     size_t memory_cost_total;
//     size_t memory_cost_solver;
//     size_t memory_cost_collision;
// }

// 范型类的函数只能实现在.h里面
// void operator=(const SharedArray<T>& right_array);
// SharedArray<T>& operator+=(const SharedArray<T>& right_array);
// SharedArray<T>& operator+=(T input_value); 
// SharedArray<T>& operator*=(const SharedArray<T>& right_array);
// SharedArray<T>& operator*=(T input_value); 
// SharedArray<T>& operator/=(const SharedArray<T>& right_array);
// SharedArray<T>& operator/=(T input_value);

// friend float dot(const SharedArray<T>& left_array,const  SharedArray<T>& right_array);
// friend void add(SharedArray<T>& output_array,   const SharedArray<T>& input_array1, const SharedArray<T>& input_array2);
// friend void add(SharedArray<T>& output_array,   const SharedArray<T>& input_array1, const SharedArray<T>& input_array2, float scale);
// friend void add(SharedArray<T>& output_array,   const SharedArray<T>& input_array,  float value);
// friend void minus(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2);
// friend void minus(SharedArray<T>& output_array, const SharedArray<T>& input_array1, const SharedArray<T>& input_array2, float scale);
// friend void minus(SharedArray<T>& output_array, const SharedArray<T>& input_array,  float value);
// friend void mul(SharedArray<T>& output_array,   const SharedArray<T>& input_array1, const SharedArray<T>& input_array2);
// friend void mul(SharedArray<T>& output_array,   const SharedArray<T>& input_array,  float value);
// friend void devide(SharedArray<T>& output_array,const SharedArray<T>& input_array1, const SharedArray<T>& input_array2);
// friend void devide(SharedArray<T>& output_array,const SharedArray<T>& input_array,  float value);

// T min();
// T max();
// T sum();
// void print();



// 重载了构造函数则需要重写构造函数，否则在solver类构造时会找不到符号
// 这样的缺点是窗口创建的时候就要申请一个空间（而不是模型加载时或者点击开始仿真时）

// template<typename TypeInt> SharedArray(){ resize(1); }
// template<typename TypeInt> SharedArray(size_t input_length){  resize(input_length); }
// template<typename TypeInt> SharedArray(const std::vector<T>& input_data){
//     length = input_data.size();
//     byte_length = length * sizeof(T);
//     MTL::Device* device = (MTL::Device*)nanogui::metal_device();
//     gpu_buffer = device -> newBuffer(input_data.data(), length * sizeof(T), MTL::ResourceStorageModeShared);
//     cpu_ptr = static_cast<T>(gpu_buffer -> contents());
// }


// inline T min(){
//     T min_value;
//     for (size_t i = 0; i < this -> length; i++) {
//         min_value = simd::min(min_value, this -> cpu_ptr[i]);
//     }
//     return min_value;
// }

// inline T max(){
//     T max_value;
//     for (size_t i = 0; i < this -> length; i++) {
//         max_value = simd::max(max_value, this -> cpu_ptr[i]);
//     }
//     return max_value;
// }

// inline T sum(){
//     T sum_value;
//     omp_parallel_for_reduction_sum(sum_value)
//     for (size_t i = 0; i < this -> length; i++) {
//         sum_value += this -> cpu_ptr[i];
//     }
//     return sum_value;
// }

// inline T sum(size_t begin_pos, size_t end_pos){
//     T sum_value = 0;
//     #pragma omp for reduction(+ : sum_value)
//     for (size_t i = begin_pos; i < end_pos; i++) {
//         sum_value += this -> cpu_ptr[i];
//     }
//     return sum_value;
// }

