#pragma once

#ifdef  METAL_CODE
    #define Thread thread
    #define ThreadRef(T) thread T&
    #define GLOBAL device 
    #define Pointer(T) device T*
    #define Constant(T) constant T&
    #define ConstRef(T) thread const T&
    #define Const(T) const T
    #define ThreadGroup threadgroup
    #define ConstExpr constant
    #define CONSTIF if 
    #define Array(T) device T*
    #define ArrayRef(T) device T*
#else
    #define Thread 
    #define ThreadRef(T) T&
    #define GLOBAL
    #define Pointer(T) T*
    #define Constant(T) const T&
    #define Const(T) const T
    #define ConstRef(T) const T&
    #define ThreadGroup
    #define ConstExpr constexpr
    #define CONSTIF if constexpr
    #define Array(T) SharedArray<T>
    #define ArrayRef(T) SharedArray<T>&
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

// #define GPU_PREFIX Constant(uint) prefix,
#define GPU_PREFIX 

template <typename T>
Thread T tothread(GLOBAL T& data) { return data; }