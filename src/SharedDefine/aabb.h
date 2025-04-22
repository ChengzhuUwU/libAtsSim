#pragma once

#include "float_n.h"
#include "float_n_n.h"
#include "geometry.h"

struct AABB{
    Float3 min_pos;
    Float3 max_pos;
    
    AABB(){
        min_pos = make<Float3>(Float_max);
        max_pos = make<Float3>(Float_min);
    }
    AABB(CREF(AABB) aabb){
        min_pos = aabb.min_pos;
        max_pos = aabb.max_pos;
    }
    // AABB(volatile CREF(AABB) aabb){
    //     min_pos = aabb.min_pos;
    //     max_pos = aabb.max_pos;
    // }
#ifdef METAL_CODE
    AABB(DEVICE const AABB& aabb){
        min_pos = aabb.min_pos;
        max_pos = aabb.max_pos;
    }
    AABB(DEVICE const Float3& pos){
        min_pos = pos;
        max_pos = pos;
    }
    // AABB(DEVICE volatile const AABB& aabb){
    //     min_pos = aabb.min_pos;
    //     max_pos = aabb.max_pos;
    // }
#endif
    AABB(CREF(Float3) pos){
        min_pos = pos;
        max_pos = pos;
    }
    AABB(CREF(Float3) pos1, CREF(Float3) pos2){
        min_pos = min_vec(pos1, pos2);
        max_pos = max_vec(pos1, pos2);
    }
    AABB(CREF(Float3) pos1, CREF(Float3) pos2, CREF(Float3) pos3){
        min_pos = min_vec(min_vec(pos1, pos2), pos3);
        max_pos = max_vec(max_vec(pos1, pos2), pos3);
    }
    
    // for ccd
    AABB(CREF(Float3) v0, CREF(Float3) v1, 
         CREF(Float3) v2, CREF(Float3) v3)
    {
        min_pos = min_vec(min_vec(v0, v1), min_vec(v2, v3));
        max_pos = max_vec(max_vec(v0, v1), max_vec(v2, v3));
    }
    AABB(CREF(Float3) v0, CREF(Float3) v1, CREF(Float3) v2, 
         CREF(Float3) v3, CREF(Float3) v4, CREF(Float3) v5)
    {
        Float3 min_pos_t0 = min_vec(min_vec(v0, v1), v2);
        Float3 min_pos_t1 = min_vec(min_vec(v3, v4), v5);
        Float3 max_pos_t0 = max_vec(max_vec(v0, v1), v2);
        Float3 max_pos_t1 = max_vec(max_vec(v3, v4), v5);
        min_pos = min_vec(min_pos_t0, min_pos_t1);
        max_pos = max_vec(max_pos_t0, max_pos_t1);
    }
    Float3 center() const{
        return (min_pos + max_pos) * 0.5f;
    }
    Float3 range() const{
        return max_vec(make<Float3>(Epsilon), max_pos - min_pos);
    }
    Float3 range_inv() const{
        return reverse_vec(range());
    }
    float volume(){
        Float3 wid = range();
        return wid.x * wid.y + wid.z;
    }
    AABB operator+(CREF(AABB) input_aabb) const{
        AABB tmp;
        tmp.min_pos = min_vec(min_pos, input_aabb.min_pos);
        tmp.max_pos = max_vec(max_pos, input_aabb.max_pos);
        return tmp;
    }
    void operator+=(CREF(AABB) input_aabb){
        min_pos = min_vec(min_pos, input_aabb.min_pos);
        max_pos = max_vec(max_pos, input_aabb.max_pos);
    }
    void operator+=(CREF(Float3) input_vec){
        min_pos = min_vec(min_pos, input_vec);
        max_pos = max_vec(max_pos, input_vec);
    }
    void operator+=(float input_val){
        min_pos -= make<Float3>(input_val);
        max_pos += make<Float3>(input_val);
    }
    AABB operator+(float eps){
        min_pos -= make<Float3>(eps);
        max_pos += make<Float3>(eps);
        return *this;
    }
    bool is_overlap(CREF(Float3) vert_pos){
        return all_vec(is_greater_equal_than(vert_pos, min_pos) && is_smaller_equal_than(vert_pos, max_pos));
    }
    bool is_overlap(CREF(Float2x3) edge_pos){
	    Float3 foot = SimGeometry::foot_ve(center(), get(edge_pos, 0), get(edge_pos, 1));
        return is_overlap(foot);
    }
    bool is_overlap(CREF(AABB) aabb){
        if (min_pos.x >= aabb.max_pos.x || aabb.min_pos.x >= max_pos.x)
            return false;
        if (min_pos.y >= aabb.max_pos.y || aabb.min_pos.y >= max_pos.y)
            return false;
        if (min_pos.z >= aabb.max_pos.z || aabb.min_pos.z >= max_pos.z)
            return false;
        return true;
    }
};

class OBB{
    Float3 center;
    Float3 dir;
    Float3 width;
};