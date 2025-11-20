#pragma once

#include "collision_data.h"
#include "lbvh_cpu.h"
#include "lbvh_gpu.h"

struct LbvhFaceEdgeData {
    LbvhData vert_tree;
    LbvhData face_tree;
    LbvhData edge_tree;
};

template<LBVHUpdateType update_type>
struct LbvhFaceEdge {
    LbvhCpu vert_cpu;
    LbvhCpu face_cpu;
    LbvhCpu edge_cpu;
    LbvhGpu vert_gpu;
    LbvhGpu face_gpu;
    LbvhGpu edge_gpu;
};