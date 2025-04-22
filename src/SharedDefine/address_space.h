#pragma once

#ifdef  METAL_CODE
    #define THREAD thread
    #define TREF(T) thread T&
    #define DEVICE device 
    #define PTR(T) device T*
    #define CONSTANT(T) constant T&
    #define CREF(T) thread const T&
    #define CONST(T) const T
    #define THREADGROUP threadgroup
    #define CONSTEXPR constant
    #define CONSTIF if 
    #define ARRAY(T) device T*
    #define ARRAYREF(T) device T*
#else
    #define THREAD 
    #define TREF(T) T&
    #define DEVICE 
    #define PTR(T) T*
    #define CONSTANT(T) const T&
    #define CONST(T) const T
    #define CREF(T) const T&
    #define THREADGROUP
    #define CONSTEXPR constexpr
    #define CONSTIF if constexpr
    #define ARRAY(T) SharedArray<T>
    #define ARRAYREF(T) SharedArray<T>&
#endif

#ifdef  METAL_CODE
    #define ATOMIC_FLOAT atomic_float
    #define ATOMIC_UINT atomic_uint
    #define ATOMIC_INT atomic_int
    #define ATOMIC_FLAG atomic_bool
#else
    // #define ATOMIC_FLOAT std::atomic<float>
    #define ATOMIC_FLOAT float
    #define ATOMIC_UINT std::atomic<uint>
    #define ATOMIC_INT std::atomic<int>
    #define ATOMIC_FLAG std::atomic<bool>
#endif

using uint = unsigned int;

#ifdef METAL_CODE
    
#else
    
#endif

#if defined(__APPLE__)
#define SIM_USE_SIMD true
// #define SIM_USE_EIGEN false
#else
// #define SIM_USE_SIMD false
// #define SIM_USE_EIGEN true
#define SIM_USE_GLM true
#endif

// #define GPU_PREFIX CONSTANT(uint) prefix,
#define GPU_PREFIX 

template <typename T>
THREAD T tothread(DEVICE T& data) { return data; }