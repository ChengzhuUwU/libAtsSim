#pragma once

#include "utils.h"
#include <cstdio>

#if __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#endif

#if __APPLE__
MTL::Device*& get_device();
#endif

void create_device();



// #if __APPLE__
// static inline MTL::Device* shared_device;
// #endif

// #if __APPLE__
// inline MTL::Device*& get_device(){
//     return shared_device;
// }
// #endif

// inline void create_device(){
// #if __APPLE__
//     shared_device = MTL::CreateSystemDefaultDevice();
//     fast_check_null(shared_device);
// #else
//     printf("not apple device\n");
// #endif
// }