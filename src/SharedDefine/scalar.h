#pragma once

#include "address_space.h"
#include "bits_utils.h"
#include "constant_value.h"

#if SIM_USE_SIMD
#include <simd/simd.h>
#elif SIM_USE_GLM
#include <glm/glm.hpp>
#include <glm/detail/qualifier.hpp>
#endif

using REAL = float;

// union Float_t {
//     float f;
//     uint32_t u;};
// inline uint float_to_uint(float f) {
//     Float_t n;
//     n.f = f;
//     return n.u;
// }
// inline float uint_to_float(uint32_t u) {
//     Float_t n;
//     n.u = u;
//     return n.f;
// }

template<typename T> inline constexpr T abs_scalar(CONST(T) value) { return value > T(0) ? value : -value; }
template<typename T> inline constexpr T lerp_scalar(CONST(T) left, CONST(T) right, CONST(T) lerp_value) { return left + lerp_value * (right - left);}
// inline constexpr uint abs_scalar(CONST(int) value) { return value > (0) ? value : -value; }
// inline constexpr float abs_scalar(CONST(float) value) { return value > (0.f) ? value : -value; }


template<typename T1, typename T2> inline constexpr T1 max_scalar(CONST(T1) left, CONST(T2) right) { return left > right ? left : right; }
template<typename T1, typename T2> inline constexpr T1 min_scalar(CONST(T1) left, CONST(T2) right) { return left < right ? left : right; }

template<typename T1, typename T2, typename T3> inline constexpr T1 max_scalar(CONST(T1) v1, CONST(T2) v2, CONST(T3) v3) 
{ return (v1 >= v2 && v1 >= v3) ? v1 : (v2 >= v1 && v2 >= v3) ? v2 : v3; }
template<typename T1, typename T2, typename T3> inline constexpr T1 min_scalar(CONST(T1) v1, CONST(T2) v2, CONST(T3) v3) 
{ return (v1 <= v2 && v1 <= v3) ? v1 : (v2 <= v1 && v2 <= v3) ? v2 : v3; }
template<typename T1, typename T2, typename T3, typename T4> inline constexpr T1 max_scalar(CONST(T1) v1, CONST(T2) v2, CONST(T3) v3, CONST(T4) v4) 
{ return (v1 >= v2 && v1 >= v3 && v1 >= v4) ? v1 : (v2 >= v1 && v2 >= v3 && v2 >= v4) ? v2 : (v3 >= v1 && v3 >= v2 && v3 >= v4) ? v3 : v4; }
template<typename T1, typename T2, typename T3, typename T4> inline constexpr T1 min_scalar(CONST(T1) v1, CONST(T2) v2, CONST(T3) v3, CONST(T4) v4) 
{ return (v1 <= v2 && v1 <= v3 && v1 <= v4) ? v1 : (v2 <= v1 && v2 <= v3 && v2 <= v4) ? v2 : (v3 <= v1 && v3 <= v2 && v3 <= v4) ? v3 : v4; }

template<typename T> inline constexpr void swap_scalar(TREF(T) left, TREF(T) right) { T tmp = left; left = right; right = tmp; }
template<typename T1, typename T2, typename T3>  inline constexpr T1 clamp_scalar(CONST(T1) value, CONST(T2) lower, CONST(T3) upper) { return max_scalar(min_scalar(value, upper), lower);}


#if SIM_USE_SIMD
                     inline constexpr int floor_scalar(CONST(float) value)  { return simd::floor(value); }
template<typename T> inline constexpr T sign_scalar(CONST(T) value)  { return simd::sign(value); }
template<typename T> inline constexpr T sin_scalar(CONST(T) value)   { return simd::sin(value); }
template<typename T> inline constexpr T cos_scalar(CONST(T) value)   { return simd::cos(value); }
template<typename T> inline constexpr T tan_scalar(CONST(T) value)   { return simd::tan(value); }
template<typename T> inline constexpr T asin_scalar(CONST(T) value)   { return simd::asin(value); }
template<typename T> inline constexpr T acos_scalar(CONST(T) value)   { return simd::acos(value); }
template<typename T> inline constexpr T atan_scalar(CONST(T) value)   { return simd::atan(value); }
template<typename T> inline constexpr T atan2_scalar(CONST(T) y, CONST(T) x)   { return x != T(0) ? simd::atan2(y, x) : T(0); }
template<typename T> inline constexpr T sqrt_scalar(CONST(T) value)  { return simd::sqrt(value); }
template<typename T> inline constexpr T rsqrt_scalar(CONST(T) value) { return simd::rsqrt(value); }
template<typename T> inline constexpr T is_inf_scalar(CONST(T) value) { return simd::isinf(value); }
template<typename T> inline constexpr T is_nan_scalar(CONST(T) value) { return simd::isnan(value); }
template<typename T1, typename T2> inline constexpr T1 pow_scalar(CONST(T1) x, CONST(T2) y) {return simd::pow(x, y);}

#elif SIM_USE_GLM
                     inline constexpr int floor_scalar(CONST(float) value)  { return glm::floor(value); }
template<typename T> inline T sign_scalar(CONST(T) value)  { return glm::sign(value);}
template<typename T> inline constexpr T sin_scalar(CONST(T) value)   { return glm::sin(value); }
template<typename T> inline constexpr T cos_scalar(CONST(T) value)   { return glm::cos(value); }
template<typename T> inline constexpr T tan_scalar(CONST(T) value)   { return glm::tan(value); }
template<typename T> inline constexpr T asin_scalar(CONST(T) value)   { return glm::asin(value); }
template<typename T> inline constexpr T acos_scalar(CONST(T) value)   { return glm::acos(value); }
template<typename T> inline constexpr T atan_scalar(CONST(T) value)   { return glm::atan(value); }
template<typename T> inline constexpr T atan2_scalar(CONST(T) y, CONST(T) x)   { return x != T(0) ? glm::atan2(y, x) : T(0); }
template<typename T> inline constexpr T sqrt_scalar(CONST(T) value)  { return glm::sqrt(value);}
template<typename T> inline constexpr T rsqrt_scalar(CONST(T) value) { return 1.f / glm::sqrt(value);}
template<typename T> inline constexpr T is_inf_scalar(CONST(T) value) { return glm::isinf(value); }
template<typename T> inline constexpr T is_nan_scalar(CONST(T) value) { return glm::isnan(value); }
template<typename T1, typename T2> inline constexpr T1 pow_scalar(CONST(T1) x, CONST(T2) y)  {return glm::pow(x, y);}

#endif

template<typename T> inline constexpr T square_scalar(CONST(T) value) { return value * value; }

template<typename T> inline constexpr T is_equal_scalar(CONST(T) left, CONST(T) right)   { return abs_scalar(left - right) < Epsilon; }




