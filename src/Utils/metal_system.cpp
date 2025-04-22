#include "metal_system.h"

#if  __APPLE__
MTL::Device* shared_device;

MTL::Device*& get_device(){
    return shared_device;
}
#endif

void create_device(){
    #if  __APPLE__
    shared_device = MTL::CreateSystemDefaultDevice();
    fast_check_null(shared_device);
    #endif
}