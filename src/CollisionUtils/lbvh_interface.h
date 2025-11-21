#pragma once

#include "collision_data.h"
#include "lbvh_cpu.h"
#include "lbvh_gpu.h"

template<LBVHUpdateType update_type>
struct LbvhFaceEdge {
    LbvhCpu vert_cpu;
    LbvhCpu face_cpu;
    LbvhCpu edge_cpu;
    LbvhGpu vert_gpu;
    LbvhGpu face_gpu;
    LbvhGpu edge_gpu;
};