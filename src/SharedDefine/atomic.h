#pragma once

#include "float_n.h"
#include "address_space.h"
#include "aabb.h"

// inline void atomic_add(simd::float3* addr, simd::float3&& value){
//     auto tmp = reinterpret_cast<volatile std::atomic<float*>>(*addr.x);
//     // std::atomic_fetch_add(&add.x, value.x);
// }

// template<typename Type> inline Type AtomicOr(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->fetch_or(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicXor(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->fetch_xor(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicAnd(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->fetch_and(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicAdd(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->fetch_add(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicSub(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->fetch_sub(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicExch(Type * Address, Type Val) { return reinterpret_cast<std::atomic<Type>*>(Address)->exchange(Val,std::memory_order_relaxed); }
// template<typename Type> inline Type AtomicCAS(Type* Address, Type Exp, Type Val) { reinterpret_cast<std::atomic<Type>*>(Address)->compare_exchange_strong(Exp, Val, std::memory_order_relaxed); return Exp; }

// template <typename T>
// inline void AtomicAdd(std::atomic<T>& ptr, const T& operand) {
//     T current = ptr.load();
//     T new_value;
//     do {
//         new_value = current + operand;
//     } while (!ptr.compare_exchange_weak(current, new_value));
// }

// template <typename T>
// inline T AtomicAdd(std::atomic<T>* ptr, const unsigned& addr, const T& operand) {
//     return ptr[addr].fetch_add(operand, std::memory_order_relaxed);
// }

// template <typename T>
// inline T AtomicSub(std::atomic<T>* ptr, const unsigned& addr, const T& operand) {
//     return ptr[addr].fetch_sub(operand);
// }

// inline void atomic_add_float3(std::atomic<float>* array, const unsigned& addr, const simd::float3& vec){
//     std::atomic_fetch_add_explicit(array + addr * 4 + 0, vec[0], std::memory_order_relaxed);
//     std::atomic_fetch_add_explicit(array + addr * 4 + 1, vec[1], std::memory_order_relaxed);
//     std::atomic_fetch_add_explicit(array + addr * 4 + 2, vec[2], std::memory_order_relaxed);
// }

#ifdef METAL_CODE
#define LOOPS_BEFORE_YIELD 16
#else
static constexpr int LOOPS_BEFORE_YIELD = 16;
#endif

static inline void yield() {
//#ifndef METAL_CODE
//#endif
}

#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#include <emmintrin.h> 
#endif

static inline void machine_pause(uint delay) {
    #if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
        while (delay-- > 0) { _mm_pause(); }
    #elif defined(__APPLE__)
        while (delay-- > 0) { __asm__ __volatile__("yield" ::: "memory"); }
    #else /* Generic */
        (void)delay; // suppress without including _template_helpers.h
        yield();
    #endif
}

inline void spinning(){
    int count = 1;
    machine_pause(count);
    count *= 2;
    // if (count <= LOOPS_BEFORE_YIELD) {
    //     machine_pause(count);
    //     count *= 2;
    // } else {
    //     // Pause is so long that we might as well yield CPU to scheduler.
    //     yield();
    // }
}

// enum lock_status{
//     prepare,
//     acquired
// };


#ifdef METAL_CODE
using FlagType = atomic_bool;
#else
#include <atomic>
using FlagType = std::atomic_bool;
#endif

static inline bool try_lock_flag(GLOBAL FlagType& flag)  { 
#ifdef METAL_CODE
    return atomic_exchange_explicit(&flag, true, memory_order_relaxed);
#else
    return flag.exchange(true);
    // return std::atomic_exchange_explicit(&flag, true, std::memory_order_relaxed);
#endif
}

static inline void free_lock_flag(GLOBAL FlagType& flag) { 
#ifdef METAL_CODE
    atomic_store_explicit(&flag, false, memory_order_relaxed);
#else
    flag.store(false); 
    // std::atomic_store_explicit(&flag, false, std::memory_order_relaxed);
#endif
}

// template<typename ValueType, typename WaitFunc, typename OperationFunc, typename... Args>
// inline ValueType atomic_basic_function(GLOBAL ValueType& data, GLOBAL FlagType& flag, WaitFunc wait_func, 
//     OperationFunc operation_func, Args... args) 
// {
//     ValueType orig_value;
//     while (try_lock_flag(flag)) wait_func();
//     orig_value = data;
//     ValueType new_value = operation_func(orig_value);
//     data = new_value;
//     free_lock_flag(flag);
//     return new_value;
// }


//
// Mutex-Form (CPU Only!!! GPU Will Be Locked)
//

template<typename WaitFunc, typename OperationFunc>
inline static void atomic_template(GLOBAL FlagType& flag, WaitFunc wait_func,
    OperationFunc operation_func) 
{
    while (try_lock_flag(flag)) wait_func();
    operation_func();
    free_lock_flag(flag);
}

template<typename WaitFunc, typename OperationFunc, typename ValueType>
inline static auto atomic_template_fetch(GLOBAL FlagType& flag, WaitFunc wait_func,
    OperationFunc operation_func, GLOBAL ValueType& a, const Thread ValueType& b) 
{
    while (try_lock_flag(flag)) wait_func();
    auto result = operation_func(a, b);
    free_lock_flag(flag);
    return result;
}

template<typename WaitFunc, typename OperationFunc, typename ValueType>
inline static auto atomic_template_fetch(GLOBAL FlagType& flag, WaitFunc wait_func,
    OperationFunc operation_func, GLOBAL ValueType& a, const Thread ValueType& b, const Thread ValueType& c) 
{
    while (try_lock_flag(flag)) wait_func();
    auto result = operation_func(a, b, c);
    free_lock_flag(flag);
    return result;
}

template<typename T> static inline T func_add(GLOBAL T& a, const Thread T& b) { T old_value = a; a += b; return old_value; }
template<typename T> static inline T func_sub(GLOBAL T& a, const Thread T& b) { T old_value = a; a -= b; return old_value; }
template<typename T> static inline T func_min(GLOBAL T& a, const Thread T& b) { T old_value = a; a = min_vec(a, b); return old_value; }
template<typename T> static inline T func_cas(GLOBAL T& a, const Thread T& comp, const Thread T& exch) { T old_value = a; if(old_value == comp) a = exch; return old_value; }

// non return
template<typename ValueType>
inline ValueType atomic_add(GLOBAL ValueType& data, GLOBAL FlagType& flag, ConstRef(ValueType) add_value) {
    return atomic_template_fetch(flag, spinning, func_add<ValueType>, data, add_value);
}
template<typename ValueType>
inline ValueType atomic_sub(GLOBAL ValueType& data, GLOBAL FlagType& flag, ConstRef(ValueType) sub_value) {
    return atomic_template_fetch(flag, spinning, func_sub<ValueType>, data, sub_value);
}
template<typename ValueType>
inline ValueType atomic_min(GLOBAL ValueType& data, GLOBAL FlagType& flag, ConstRef(ValueType) min_value) {
    return atomic_template_fetch(flag, spinning, func_min<ValueType>, data, min_value);
}
template<typename ValueType>
inline ValueType atomic_cas(GLOBAL ValueType& data, GLOBAL FlagType& flag, ConstRef(ValueType) compare_value, ConstRef(ValueType) exchange_value) {
    return atomic_template_fetch(flag, spinning, func_cas<ValueType>, data, compare_value, exchange_value);
}

///
/// Built-in Atomic Methods
///

// AtomicFloat Add/Sub (GPU Only)

ConstExpr float float_to_int_scale = 1 << 16;
ConstExpr float float_to_in_max = (1 << 30) / float_to_int_scale;

#ifdef METAL_CODE

GLOBAL      ATOMIC_FLOAT* float_to_atomic_ptr(GLOBAL        float& data)   { return (GLOBAL      ATOMIC_FLOAT*)(&data); }
ThreadGroup ATOMIC_FLOAT* float_to_atomic_ptr(ThreadGroup   float& data)   { return (ThreadGroup ATOMIC_FLOAT*)(&data); }
GLOBAL      ATOMIC_INT* int_to_atomic_ptr(GLOBAL      int& data) { return (GLOBAL      ATOMIC_INT*)(&data); }
ThreadGroup ATOMIC_INT* int_to_atomic_ptr(ThreadGroup int& data) { return (ThreadGroup ATOMIC_INT*)(&data); }

inline void atomic_add(GLOBAL float& data, ConstRef(float) add_value) { GLOBAL ATOMIC_FLOAT* tmp = float_to_atomic_ptr(data); atomic_fetch_add_explicit(tmp, add_value, memory_order_relaxed); }
inline void atomic_sub(GLOBAL float& data, ConstRef(float) add_value) { GLOBAL ATOMIC_FLOAT* tmp = float_to_atomic_ptr(data); atomic_fetch_sub_explicit(tmp, add_value, memory_order_relaxed); }

inline void atomic_add(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(float) add_value) { atomic_fetch_add_explicit(&ptr[addr], add_value, memory_order_relaxed); }
inline void atomic_sub(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(float) add_value) { atomic_fetch_sub_explicit(&ptr[addr], add_value, memory_order_relaxed); }

// inline void atomic_add(ThreadGroup int& data, ConstRef(float) add_value, const float scale = float_to_int_scale) { ThreadGroup ATOMIC_INT* tmp = int_to_atomic_ptr(data); atomic_fetch_add_explicit(tmp, scale * add_value, memory_order_relaxed); }
// inline void atomic_sub(ThreadGroup int& data, ConstRef(float) add_value, const float scale = float_to_int_scale) { ThreadGroup ATOMIC_INT* tmp = int_to_atomic_ptr(data); atomic_fetch_sub_explicit(tmp, scale * add_value, memory_order_relaxed); }
// inline void atomic_add(ThreadGroup atomic_int* ptr, ConstRef(uint) addr, ConstRef(float) add_value, const float scale = float_to_int_scale) { atomic_fetch_add_explicit(&ptr[addr], add_value * scale, memory_order_relaxed); }
// inline void atomic_sub(ThreadGroup atomic_int* ptr, ConstRef(uint) addr, ConstRef(float) add_value, const float scale = float_to_int_scale) { atomic_fetch_sub_explicit(&ptr[addr], add_value * scale, memory_order_relaxed); }

// Float3
inline void atomic_add(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3) add_value) {
    atomic_fetch_add_explicit(&ptr[addr * 4 + 0], add_value[0], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 4 + 1], add_value[1], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 4 + 2], add_value[2], memory_order_relaxed);
}
inline void atomic_sub(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3) sub_value) {
    atomic_fetch_sub_explicit(&ptr[addr * 4 + 0], sub_value[0], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 4 + 1], sub_value[1], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 4 + 2], sub_value[2], memory_order_relaxed);
}
// inline void atomic_max(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3) max_value) {
//     atomic_fetch_max_explicit(&ptr[addr * 4 + 0], max_value[0], memory_order_relaxed);
//     atomic_fetch_max_explicit(&ptr[addr * 4 + 1], max_value[1], memory_order_relaxed);
//     atomic_fetch_max_explicit(&ptr[addr * 4 + 2], max_value[2], memory_order_relaxed);
// }
// inline void atomic_min(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3) min_value) {
//     atomic_fetch_min_explicit(&ptr[addr * 4 + 0], min_value[0], memory_order_relaxed);
//     atomic_fetch_min_explicit(&ptr[addr * 4 + 1], min_value[1], memory_order_relaxed);
//     atomic_fetch_min_explicit(&ptr[addr * 4 + 2], min_value[2], memory_order_relaxed);
// }
// inline void atomic_aabb(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(AABB) aabb) {
//     atomic_fetch_min_explicit(&ptr[addr * 8 + 0], aabb.min_pos[0], memory_order_relaxed);
//     atomic_fetch_min_explicit(&ptr[addr * 8 + 1], aabb.min_pos[1], memory_order_relaxed);
//     atomic_fetch_min_explicit(&ptr[addr * 8 + 2], aabb.min_pos[2], memory_order_relaxed);
//     atomic_fetch_max_explicit(&ptr[addr * 8 + 4], aabb.max_pos[0], memory_order_relaxed);
//     atomic_fetch_max_explicit(&ptr[addr * 8 + 5], aabb.max_pos[1], memory_order_relaxed);
//     atomic_fetch_max_explicit(&ptr[addr * 8 + 6], aabb.max_pos[2], memory_order_relaxed);
// }

// Float3x3
inline void atomic_add(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3x3) add_value) {
    atomic_fetch_add_explicit(&ptr[addr * 12 + 0], add_value.columns[0][0], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 + 1], add_value.columns[0][1], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 + 2], add_value.columns[0][2], memory_order_relaxed);

    atomic_fetch_add_explicit(&ptr[addr * 12 + 4], add_value.columns[1][0], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 + 5], add_value.columns[1][1], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 + 6], add_value.columns[1][2], memory_order_relaxed);

    atomic_fetch_add_explicit(&ptr[addr * 12 + 8], add_value.columns[2][0], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 + 9], add_value.columns[2][1], memory_order_relaxed);
    atomic_fetch_add_explicit(&ptr[addr * 12 +10], add_value.columns[2][2], memory_order_relaxed);
}
inline void atomic_sub(GLOBAL atomic_float* ptr, ConstRef(uint) addr, ConstRef(Float3x3) sub_value) {
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 0], sub_value.columns[0][0], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 1], sub_value.columns[0][1], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 2], sub_value.columns[0][2], memory_order_relaxed);

    atomic_fetch_sub_explicit(&ptr[addr * 12 + 4], sub_value.columns[1][0], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 5], sub_value.columns[1][1], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 6], sub_value.columns[1][2], memory_order_relaxed);
    
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 8], sub_value.columns[2][0], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 + 9], sub_value.columns[2][1], memory_order_relaxed);
    atomic_fetch_sub_explicit(&ptr[addr * 12 +10], sub_value.columns[2][2], memory_order_relaxed);
}

#endif

// AtomicTemplate Load/Store/CAS

inline float cast_int_to_float(int value, const float scale = float_to_int_scale) { return float(value) / scale; }
inline int cast_float_to_int(float value, const float scale = float_to_int_scale) { return value * scale; }


#ifdef METAL_CODE
template<typename ValueType, typename AtomicType>
inline ValueType atomic_cas(GLOBAL AtomicType& data, ValueType compare_value, ConstRef(ValueType) exchange_value){
    atomic_compare_exchange_weak_explicit(&data, &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed);
    return compare_value; }

template<typename ValueType, typename AtomicType>
inline ValueType atomic_load(GLOBAL AtomicType& data){ return atomic_load_explicit(&data, memory_order_relaxed); }

template<typename ValueType, typename AtomicType>
inline void atomic_store(GLOBAL AtomicType& data, ConstRef(ValueType) store_value){ atomic_store_explicit(&data, store_value, memory_order_relaxed); }
#else
template<typename ValueType, typename AtomicType>
inline ValueType atomic_cas(GLOBAL AtomicType& data, ValueType compare_value, ConstRef(ValueType) exchange_value) {
    std::atomic_compare_exchange_strong(&data, &compare_value, exchange_value);
    return compare_value; }

template<typename ValueType, typename AtomicType>
inline ValueType atomic_load(GLOBAL AtomicType& data){ return std::atomic_load(&data); }

template<typename ValueType, typename AtomicType>
inline void atomic_store(GLOBAL AtomicType& data, ConstRef(ValueType) store_value){ std::atomic_store(&data, store_value); }
#endif

// AtomicUint Add/Sub/Bit-Or : Input Atomic-Type Ptr/Ref (Cpp Does Not Support std::atomic_float::fetch_add )

#ifdef METAL_CODE
inline uint atomic_add(ThreadGroup ATOMIC_UINT& data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit(&data, add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      ATOMIC_UINT& data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit(&data, add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup ATOMIC_UINT& data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit(&data, sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      ATOMIC_UINT& data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit(&data, sub_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup ATOMIC_UINT& data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit(&data, or_value, memory_order_relaxed); }
inline uint atomic_or (GLOBAL      ATOMIC_UINT& data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit(&data, or_value, memory_order_relaxed); }
inline uint atomic_and(ThreadGroup ATOMIC_UINT& data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit(&data, and_value, memory_order_relaxed); }
inline uint atomic_and(GLOBAL      ATOMIC_UINT& data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit(&data, and_value, memory_order_relaxed); }
#else
inline uint atomic_add(ATOMIC_UINT& data, ConstRef(uint) add_value){ return std::atomic_fetch_add(&data, add_value); }
inline uint atomic_sub(ATOMIC_UINT& data, ConstRef(uint) sub_value){ return std::atomic_fetch_sub(&data, sub_value); }
inline uint atomic_or (ATOMIC_UINT& data, ConstRef(uint) or_value) { return std::atomic_fetch_or(&data, or_value);   }
inline uint atomic_and(ATOMIC_UINT& data, ConstRef(uint) and_value){ return std::atomic_fetch_and(&data, and_value);   }
#endif
#ifdef METAL_CODE
inline uint atomic_add(ThreadGroup ATOMIC_UINT* data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit(data, add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      ATOMIC_UINT* data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit(data, add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup ATOMIC_UINT* data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit(data, sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      ATOMIC_UINT* data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit(data, sub_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup ATOMIC_UINT* data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit(data, or_value, memory_order_relaxed); }
inline uint atomic_or (GLOBAL      ATOMIC_UINT* data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit(data, or_value, memory_order_relaxed); }
inline uint atomic_and(ThreadGroup ATOMIC_UINT* data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit(data, and_value, memory_order_relaxed); }
inline uint atomic_and(GLOBAL      ATOMIC_UINT* data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit(data, and_value, memory_order_relaxed); }
#else
inline uint atomic_add(ATOMIC_UINT* data, ConstRef(uint) add_value){ return std::atomic_fetch_add(data, add_value); }
inline uint atomic_sub(ATOMIC_UINT* data, ConstRef(uint) sub_value){ return std::atomic_fetch_sub(data, sub_value); }
inline uint atomic_or (ATOMIC_UINT* data, ConstRef(uint) or_value) { return std::atomic_fetch_or (data, or_value);   }
inline uint atomic_and(ATOMIC_UINT* data, ConstRef(uint) and_value){ return std::atomic_fetch_or (data, and_value);   }
#endif

// AtomicUint Add/Sub/Bit-Or : Input NonAtomic-Type Ptr/Ref

#ifdef METAL_CODE
// uint reference
inline uint atomic_add(ThreadGroup uint& data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit((ThreadGroup ATOMIC_UINT*)(&data), add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      uint& data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit((GLOBAL      ATOMIC_UINT*)(&data), add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup uint& data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit((ThreadGroup ATOMIC_UINT*)(&data), sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      uint& data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit((GLOBAL      ATOMIC_UINT*)(&data), sub_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup uint& data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit ((ThreadGroup ATOMIC_UINT*)(&data), or_value,  memory_order_relaxed); }
inline uint atomic_or (GLOBAL      uint& data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit ((GLOBAL      ATOMIC_UINT*)(&data), or_value,  memory_order_relaxed); }
inline uint atomic_and(ThreadGroup uint& data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit((ThreadGroup ATOMIC_UINT*)(&data), and_value,  memory_order_relaxed); }
inline uint atomic_and(GLOBAL      uint& data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit((GLOBAL      ATOMIC_UINT*)(&data), and_value,  memory_order_relaxed); }
inline uint atomic_cas(ThreadGroup uint& data, uint compare_value, ConstRef(uint) exchange_value) { atomic_compare_exchange_weak_explicit((ThreadGroup ATOMIC_UINT*)(&data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }
inline uint atomic_cas(GLOBAL      uint& data, uint compare_value, ConstRef(uint) exchange_value) { atomic_compare_exchange_weak_explicit((GLOBAL      ATOMIC_UINT*)(&data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }

// int reference
inline uint atomic_add(ThreadGroup int& data, ConstRef(int) add_value){ return atomic_fetch_add_explicit((ThreadGroup ATOMIC_INT*)(&data), add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      int& data, ConstRef(int) add_value){ return atomic_fetch_add_explicit((GLOBAL      ATOMIC_INT*)(&data), add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup int& data, ConstRef(int) sub_value){ return atomic_fetch_sub_explicit((ThreadGroup ATOMIC_INT*)(&data), sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      int& data, ConstRef(int) sub_value){ return atomic_fetch_sub_explicit((GLOBAL      ATOMIC_INT*)(&data), sub_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup int& data, ConstRef(int) or_value) { return atomic_fetch_or_explicit ((ThreadGroup ATOMIC_INT*)(&data), or_value,  memory_order_relaxed); }
inline uint atomic_or (GLOBAL      int& data, ConstRef(int) or_value) { return atomic_fetch_or_explicit ((GLOBAL      ATOMIC_INT*)(&data), or_value,  memory_order_relaxed); }
inline uint atomic_and(ThreadGroup int& data, ConstRef(int) and_value){ return atomic_fetch_and_explicit((ThreadGroup ATOMIC_INT*)(&data), and_value,  memory_order_relaxed); }
inline uint atomic_and(GLOBAL      int& data, ConstRef(int) and_value){ return atomic_fetch_and_explicit((GLOBAL      ATOMIC_INT*)(&data), and_value,  memory_order_relaxed); }
inline uint atomic_cas(ThreadGroup int& data, int compare_value, ConstRef(int) exchange_value) { atomic_compare_exchange_weak_explicit((ThreadGroup ATOMIC_INT*)(&data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }
inline uint atomic_cas(GLOBAL      int& data, int compare_value, ConstRef(int) exchange_value) { atomic_compare_exchange_weak_explicit((GLOBAL      ATOMIC_INT*)(&data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }

#else
// uint reference
inline uint atomic_add(uint& data, ConstRef(uint) add_value){ return std::atomic_fetch_add((ATOMIC_UINT*)(&data), add_value); }
inline uint atomic_sub(uint& data, ConstRef(uint) sub_value){ return std::atomic_fetch_sub((ATOMIC_UINT*)(&data), sub_value); }
inline uint atomic_and(uint& data, ConstRef(uint) and_value){ return std::atomic_fetch_and((ATOMIC_UINT*)(&data), and_value); }
inline uint atomic_or (uint& data, ConstRef(uint) or_value) { return std::atomic_fetch_or ((ATOMIC_UINT*)(&data), or_value);  }
inline uint atomic_cas(uint& data, uint compare_value, ConstRef(uint) exchange_value)  { std::atomic_compare_exchange_strong((ATOMIC_UINT*)(&data), &compare_value, exchange_value); return compare_value;  }

// int reference
inline uint atomic_add(int& data, ConstRef(int) add_value){ return std::atomic_fetch_add((ATOMIC_INT*)(&data), add_value); }
inline uint atomic_sub(int& data, ConstRef(int) sub_value){ return std::atomic_fetch_sub((ATOMIC_INT*)(&data), sub_value); }
inline uint atomic_and(int& data, ConstRef(int) and_value){ return std::atomic_fetch_and((ATOMIC_INT*)(&data), and_value); }
inline uint atomic_or (int& data, ConstRef(int) or_value) { return std::atomic_fetch_or ((ATOMIC_INT*)(&data), or_value);  }
inline uint atomic_cas(int& data, int compare_value, ConstRef(int) exchange_value)     { std::atomic_compare_exchange_strong((ATOMIC_INT*)(&data), &compare_value, exchange_value); return compare_value;  }
#endif

#ifdef METAL_CODE
inline uint atomic_add(ThreadGroup uint* data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit((ThreadGroup ATOMIC_UINT*)(data), add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      uint* data, ConstRef(uint) add_value){ return atomic_fetch_add_explicit((GLOBAL      ATOMIC_UINT*)(data), add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup uint* data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit((ThreadGroup ATOMIC_UINT*)(data), sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      uint* data, ConstRef(uint) sub_value){ return atomic_fetch_sub_explicit((GLOBAL      ATOMIC_UINT*)(data), sub_value, memory_order_relaxed); }
inline uint atomic_and(ThreadGroup uint* data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit((ThreadGroup ATOMIC_UINT*)(data), and_value, memory_order_relaxed); }
inline uint atomic_and(GLOBAL      uint* data, ConstRef(uint) and_value){ return atomic_fetch_and_explicit((GLOBAL      ATOMIC_UINT*)(data), and_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup uint* data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit ((ThreadGroup ATOMIC_UINT*)(data), or_value,  memory_order_relaxed); }
inline uint atomic_or (GLOBAL      uint* data, ConstRef(uint) or_value) { return atomic_fetch_or_explicit ((GLOBAL      ATOMIC_UINT*)(data), or_value,  memory_order_relaxed); }
inline uint atomic_cas(ThreadGroup uint* data, uint compare_value, ConstRef(uint) exchange_value) { atomic_compare_exchange_weak_explicit((ThreadGroup ATOMIC_UINT*)(data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }
inline uint atomic_cas(GLOBAL      uint* data, uint compare_value, ConstRef(uint) exchange_value) { atomic_compare_exchange_weak_explicit((GLOBAL      ATOMIC_UINT*)(data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }

inline uint atomic_add(ThreadGroup int* data, ConstRef(int) add_value){ return atomic_fetch_add_explicit((ThreadGroup ATOMIC_INT*)(data), add_value, memory_order_relaxed); }
inline uint atomic_add(GLOBAL      int* data, ConstRef(int) add_value){ return atomic_fetch_add_explicit((GLOBAL      ATOMIC_INT*)(data), add_value, memory_order_relaxed); }
inline uint atomic_sub(ThreadGroup int* data, ConstRef(int) sub_value){ return atomic_fetch_sub_explicit((ThreadGroup ATOMIC_INT*)(data), sub_value, memory_order_relaxed); }
inline uint atomic_sub(GLOBAL      int* data, ConstRef(int) sub_value){ return atomic_fetch_sub_explicit((GLOBAL      ATOMIC_INT*)(data), sub_value, memory_order_relaxed); }
inline uint atomic_and(ThreadGroup int* data, ConstRef(int) and_value){ return atomic_fetch_and_explicit((ThreadGroup ATOMIC_INT*)(data), and_value, memory_order_relaxed); }
inline uint atomic_and(GLOBAL      int* data, ConstRef(int) and_value){ return atomic_fetch_and_explicit((GLOBAL      ATOMIC_INT*)(data), and_value, memory_order_relaxed); }
inline uint atomic_or (ThreadGroup int* data, ConstRef(int) or_value) { return atomic_fetch_or_explicit ((ThreadGroup ATOMIC_INT*)(data), or_value,  memory_order_relaxed); }
inline uint atomic_or (GLOBAL      int* data, ConstRef(int) or_value) { return atomic_fetch_or_explicit ((GLOBAL      ATOMIC_INT*)(data), or_value,  memory_order_relaxed); }
inline uint atomic_cas(ThreadGroup int* data, int compare_value, ConstRef(int) exchange_value) { atomic_compare_exchange_weak_explicit((ThreadGroup ATOMIC_INT*)(data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }
inline uint atomic_cas(GLOBAL      int* data, int compare_value, ConstRef(int) exchange_value) { atomic_compare_exchange_weak_explicit((GLOBAL      ATOMIC_INT*)(data), &compare_value, exchange_value, memory_order_relaxed, memory_order_relaxed); return compare_value; }

#else
inline uint atomic_add(uint* data, ConstRef(uint) add_value){ return std::atomic_fetch_add((ATOMIC_UINT*)(data), add_value); }
inline uint atomic_sub(uint* data, ConstRef(uint) sub_value){ return std::atomic_fetch_sub((ATOMIC_UINT*)(data), sub_value); }
inline uint atomic_and(uint* data, ConstRef(uint) and_value){ return std::atomic_fetch_and((ATOMIC_UINT*)(data), and_value); }
inline uint atomic_or (uint* data, ConstRef(uint) or_value) { return std::atomic_fetch_or ((ATOMIC_UINT*)(data), or_value);  }
inline uint atomic_cas(uint* data, uint compare_value, ConstRef(uint) exchange_value)   { std::atomic_compare_exchange_strong((ATOMIC_UINT*)(&data), &compare_value, exchange_value); return compare_value;  }

inline uint atomic_add(int* data, ConstRef(int) add_value){ return std::atomic_fetch_add((ATOMIC_UINT*)(data), add_value); }
inline uint atomic_sub(int* data, ConstRef(int) sub_value){ return std::atomic_fetch_sub((ATOMIC_UINT*)(data), sub_value); }
inline uint atomic_and(int* data, ConstRef(int) and_value){ return std::atomic_fetch_and((ATOMIC_UINT*)(data), and_value); }
inline uint atomic_or (int* data, ConstRef(int) or_value) { return std::atomic_fetch_or ((ATOMIC_UINT*)(data), or_value);  }
inline uint atomic_cas(int* data, int compare_value, ConstRef(int) exchange_value)      { std::atomic_compare_exchange_strong((ATOMIC_INT*)(&data), &compare_value, exchange_value); return compare_value;  }
#endif


//
// AtomicTemplate Add/Sub Is Dangerous...
//

// #ifdef METAL_CODE
// template<typename T, typename AtomicT> inline T atomic_add(ThreadGroup AtomicT& data, ConstRef(T) add_value){ return atomic_fetch_add_explicit(&data, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_add(GLOBAL      AtomicT& data, ConstRef(T) add_value){ return atomic_fetch_add_explicit(&data, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(ThreadGroup AtomicT& data, ConstRef(T) sub_value){ return atomic_fetch_sub_explicit(&data, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(GLOBAL      AtomicT& data, ConstRef(T) sub_value){ return atomic_fetch_sub_explicit(&data, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (ThreadGroup AtomicT& data, ConstRef(T) or_value) { return atomic_fetch_or_explicit(&data, or_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (GLOBAL      AtomicT& data, ConstRef(T) or_value) { return atomic_fetch_or_explicit(&data, or_value, memory_order_relaxed); }
// #else
// template<typename T, typename AtomicT> inline T atomic_add(ATOMIC_UINT& data, ConstRef(T) add_value){ return std::atomic_fetch_add(&data, add_value); }
// template<typename T, typename AtomicT> inline T atomic_sub(ATOMIC_UINT& data, ConstRef(T) sub_value){ return std::atomic_fetch_sub(&data, sub_value); }
// template<typename T, typename AtomicT> inline T atomic_or (ATOMIC_UINT& data, ConstRef(T) or_value) { return std::atomic_fetch_or(&data, or_value);   }
// #endif
// #ifdef METAL_CODE
// template<typename T, typename AtomicT> inline T atomic_add(ThreadGroup AtomicT* data, ConstRef(T) add_value){ return atomic_fetch_add_explicit(data, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_add(GLOBAL      AtomicT* data, ConstRef(T) add_value){ return atomic_fetch_add_explicit(data, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(ThreadGroup AtomicT* data, ConstRef(T) sub_value){ return atomic_fetch_sub_explicit(data, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(GLOBAL      AtomicT* data, ConstRef(T) sub_value){ return atomic_fetch_sub_explicit(data, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (ThreadGroup AtomicT* data, ConstRef(T) or_value) { return atomic_fetch_or_explicit(data, or_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (GLOBAL      AtomicT* data, ConstRef(T) or_value) { return atomic_fetch_or_explicit(data, or_value, memory_order_relaxed); }
// #else
// template<typename T, typename AtomicT> inline T atomic_add(AtomicT* data, ConstRef(T) add_value){ return std::atomic_fetch_add(data, add_value); }
// template<typename T, typename AtomicT> inline T atomic_sub(AtomicT* data, ConstRef(T) sub_value){ return std::atomic_fetch_sub(data, sub_value); }
// template<typename T, typename AtomicT> inline T atomic_or (AtomicT* data, ConstRef(T) or_value) { return std::atomic_fetch_or (data, or_value);   }
// #endif

// #ifdef METAL_CODE
// template<typename T, typename AtomicT> inline uint atomic_add(ThreadGroup uint& data, ConstRef(uint) add_value){ ThreadGroup ATOMIC_UINT* tmp = (ThreadGroup ATOMIC_UINT*)(&data); return atomic_fetch_add_explicit(tmp, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline uint atomic_add(GLOBAL      uint& data, ConstRef(uint) add_value){ GLOBAL      ATOMIC_UINT* tmp = (GLOBAL      ATOMIC_UINT*)(&data); return atomic_fetch_add_explicit(tmp, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline uint atomic_sub(ThreadGroup uint& data, ConstRef(uint) sub_value){ ThreadGroup ATOMIC_UINT* tmp = (ThreadGroup ATOMIC_UINT*)(&data); return atomic_fetch_sub_explicit(tmp, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline uint atomic_sub(GLOBAL      uint& data, ConstRef(uint) sub_value){ GLOBAL      ATOMIC_UINT* tmp = (GLOBAL      ATOMIC_UINT*)(&data); return atomic_fetch_sub_explicit(tmp, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline uint atomic_or (ThreadGroup uint& data, ConstRef(uint) or_value) { ThreadGroup ATOMIC_UINT* tmp = (ThreadGroup ATOMIC_UINT*)(&data); return atomic_fetch_or_explicit (tmp, or_value,  memory_order_relaxed); }
// template<typename T, typename AtomicT> inline uint atomic_or (GLOBAL      uint& data, ConstRef(uint) or_value) { GLOBAL      ATOMIC_UINT* tmp = (GLOBAL      ATOMIC_UINT*)(&data); return atomic_fetch_or_explicit (tmp, or_value,  memory_order_relaxed); }
// #else
// template<typename T, typename AtomicT> inline uint atomic_add(uint& data, ConstRef(uint) add_value){ ATOMIC_UINT* tmp = (ATOMIC_UINT*)(&data); return std::atomic_fetch_add(tmp, add_value); }
// template<typename T, typename AtomicT> inline uint atomic_sub(uint& data, ConstRef(uint) sub_value){ ATOMIC_UINT* tmp = (ATOMIC_UINT*)(&data); return std::atomic_fetch_sub(tmp, sub_value); }
// template<typename T, typename AtomicT> inline uint atomic_or (uint& data, ConstRef(uint) or_value) { ATOMIC_UINT* tmp = (ATOMIC_UINT*)(&data); return std::atomic_fetch_or (tmp, or_value);  }
// #endif
// #ifdef METAL_CODE
// template<typename T, typename AtomicT> inline T atomic_add(ThreadGroup T* data, ConstRef(T) add_value){ ThreadGroup AtomicT* tmp = (ThreadGroup AtomicT*)(data); return atomic_fetch_add_explicit(tmp, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_add(GLOBAL      T* data, ConstRef(T) add_value){ GLOBAL      AtomicT* tmp = (GLOBAL      AtomicT*)(data); return atomic_fetch_add_explicit(tmp, add_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(ThreadGroup T* data, ConstRef(T) sub_value){ ThreadGroup AtomicT* tmp = (ThreadGroup AtomicT*)(data); return atomic_fetch_sub_explicit(tmp, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_sub(GLOBAL      T* data, ConstRef(T) sub_value){ GLOBAL      AtomicT* tmp = (GLOBAL      AtomicT*)(data); return atomic_fetch_sub_explicit(tmp, sub_value, memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (ThreadGroup T* data, ConstRef(T) or_value) { ThreadGroup AtomicT* tmp = (ThreadGroup AtomicT*)(data); return atomic_fetch_or_explicit (tmp, or_value,  memory_order_relaxed); }
// template<typename T, typename AtomicT> inline T atomic_or (GLOBAL      T* data, ConstRef(T) or_value) { GLOBAL      AtomicT* tmp = (GLOBAL      AtomicT*)(data); return atomic_fetch_or_explicit (tmp, or_value,  memory_order_relaxed); }
// #else
// template<typename T, typename AtomicT> inline T atomic_add(T* data, ConstRef(T) add_value){ AtomicT* tmp = (AtomicT*)(data); return std::atomic_fetch_add(tmp, add_value); }
// template<typename T, typename AtomicT> inline T atomic_sub(T* data, ConstRef(T) sub_value){ AtomicT* tmp = (AtomicT*)(data); return std::atomic_fetch_sub(tmp, sub_value); }
// template<typename T, typename AtomicT> inline T atomic_or (T* data, ConstRef(T) or_value) { AtomicT* tmp = (AtomicT*)(data); return std::atomic_fetch_or (tmp, or_value);  }
// #endif

