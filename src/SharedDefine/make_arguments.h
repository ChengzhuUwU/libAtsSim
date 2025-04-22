#pragma once

#include "address_space.h"

#ifndef METAL_CODE
#include "shared_array.h"
#endif

// M1 gpu采用64位地址
#define SIZE_PTR 8

enum PtrType{
    PtrTypeGpu, // Pointer That GPU Can Access (Should Store In Buffer)
    PtrTypeCpu, // Pointer That CPU Can Access (Just   Store As An Object)
    PtrTypeMtl  // Pointer As MTL::Buffer* (For Binding, Getting GPU/CPU Pointer And So On)
};

enum AccessType{
    AccessTypeRead,
    AccessTypeWrite,
    AccessTypeReadWrite
};

struct ReadWriteVector{
    #ifndef METAL_CODE
    std::vector<void*> buffer_vec;  
    std::vector<AccessType> type_vec;
    ReadWriteVector(){

    }
    ReadWriteVector(std::vector<void*> bv, std::vector<AccessType> tv) {
        buffer_vec = bv;
        type_vec = tv;
    }
    #endif
};

inline Int4 make_indirect_command_buffer(const uint num_threads, const uint blockDim = 256){
    return makeInt4((num_threads + blockDim - 1) / blockDim, 1, 1, num_threads);
}


#define GET_PTR_WITH_TYPE(source, type, name) name = get_ptr(source.name, type);
// #define GET_PTR_FROM_CLOTH_WITH_TYPE(name, type) GET_PTR_WITH_TYPE(cloth, name, type)

#define APPLY_GET_PTR_MACR_1(source, type, name) GET_PTR_WITH_TYPE(source, type, name)
#define APPLY_GET_PTR_MACR_2(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_1(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_3(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_2(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_4(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_3(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_5(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_4(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_6(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_5(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_7(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_6(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_8(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_7(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_9(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_8(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_10(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_9(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_11(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_10(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_12(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_11(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_13(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_12(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_14(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_13(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_15(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_14(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_16(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_15(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_17(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_16(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_18(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_17(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_19(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_18(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACR_20(source, type, name, ...) GET_PTR_WITH_TYPE(source, type, name) APPLY_GET_PTR_MACR_19(source, type, __VA_ARGS__)
#define APPLY_GET_PTR_MACRO_IMPL(source, type, num, ...) APPLY_GET_PTR_MACR_##num(source, type, __VA_ARGS__)

///
/// Usage :
///
/// #define GET_PTRS_IN_ARGS_FUNC(...) std::vector<void*> get() { return std::vector<void*>({ __VA_ARGS__ }); }
/// #define SET_PTRS_IN_ARGS_FROM_CLOTH_FUNC(num, ...) void set(TREF(ClothData) cloth, PtrType ptr_type){ APPLY_GET_PTR_MACRO_IMPL(cloth, ptr_type, num, __VA_ARGS__) }
/// Then : 
/// GET_PTRS_IN_ARGS_FUNC(num_verts, num_faces, num_edges, num_bending_edges)
/// SET_PTRS_IN_ARGS_FROM_CLOTH_FUNC(4, num_verts, num_faces, num_edges, num_bending_edges) 
///                                  4 is the number of args
///

           
template<typename T> PTR(T) get_ptr(ARRAYREF(T) input_array, PtrType ptr_type) { 
#ifdef METAL_CODE
    return input_array;
#elif __APPLE__
    if      (ptr_type == PtrTypeGpu) return (T*)(input_array.buffer() -> gpuAddress()); 
    else if (ptr_type == PtrTypeMtl) return (T*)(input_array.buffer());
    else if (ptr_type == PtrTypeCpu) return input_array.ptr(); 
    else                             return input_array.ptr();  
#elif _WIN32
    return input_array.ptr(); 
#endif
}

inline uint get_ptr(uint value, PtrType ptr_type) { return value; }



template<typename T> T get_cst(CREF(T) input_value) { 
    return input_value;
}

#ifdef METAL_CODE
// template<typename T> inline void set_ptr(ARRAYREF(T) input_array, PTR(T)& gpu, PTR(T)& cpu){}
// template<typename T> inline void set_cst(CREF(T) input_value, TREF(T) gpu, TREF(T) cpu){}
#else
// template<typename T> inline void set_ptr(ARRAYREF(T) input_array, PTR(T)& gpu, PTR(T)& cpu){ 
//     gpu = (T*)input_array.buffer() -> gpuAddress(); 
//     cpu = input_array.ptr();
// }
// template<typename T> inline void set_cst(CREF(T) input_value, TREF(T) gpu, TREF(T) cpu){ 
//     gpu = input_value;
//     cpu = input_value;
// }
#endif