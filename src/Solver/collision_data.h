#pragma once

#include "aabb.h"
#include "address_space.h"
#include "atomic.h"
#include "morton.h"

CONSTEXPR uint mask_is_negative = 1u << 31;

struct ProximityVV
{
    inline uint get_vert1() const { return indeces[0]; }
    inline uint get_vert2() const { return indeces[1]; }
    inline Float2 get_weight() const{ return makeFloat2(-1, 1); }
    inline float get_weight(uint idx) const{ return idx == 0 ? -1 : 1; }

    // normal : Normal of v2, Which Is The Same Direction To Push v1
    inline Float3 get_normal() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); } 
    inline Float3x3 get_txt() const { Float3 t = get_normal(); return outer_product(t, t); }
    inline float get_stiff() const { return vec2[3]; }
    inline Int2 get_indices() const { return makeInt2(indeces[0], indeces[1]); }

    // Self Collision Is Always Positive (You Dont Know The Correct Direction ~ , Just Fix It In E-F Pair Or CCD-Based Response)
    template<uint NV> 
    inline void get_indices(THREAD uint tet[NV]) const 
    { 
        static_assert(NV == 1 || NV == 2, "Wrong NumVerts In Collision Pair, Should Be 1 or 2");
        for(uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative; 
    }

    inline void set_negative()      { for(uint i = 0; i < 2; i++) indeces[i] |= mask_is_negative; }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int2 get_indices_negative() const { Int2 tmp; for(uint i = 0; i < 2; i++) { tmp[i] = indeces[i] & ~mask_is_negative; } return tmp; }
    inline uint get_vert1_negative() const { return indeces[0] & ~mask_is_negative; }
    inline uint get_vert2_negative() const { return indeces[1] & ~mask_is_negative; }

    ProximityVV(){}
    ProximityVV(CREF(ProximityVV) right) : indeces(right.indeces), vec2(right.vec2) {}

#ifdef METAL_CODE
    ProximityVV(DEVICE const ProximityVV& right) : indeces(right.indeces), vec2(right.vec2) {  }
#endif

    // 'normal' Is The Normal of vid2, Because We Want That 'project(v0 - v1, normal)' Is Their Distance 
    // (a(x) - b(x), In Paper Small Steps)
    // Or 'normal' Is The Direction To Push vid1 Away
    ProximityVV(CREF(uint) vid1, CREF(uint) vid2, CREF(float) stiffness, CREF(float) area, CREF(Float3) normal)
    {
        indeces = makeInt4(vid1, vid2, 0, 0);
        vec2 = makeFloat4(normal[0], normal[1], normal[2], stiffness * area);
    }

protected:
    Int4 indeces;
    Float4 vec2; // normal(3), stiff(1) = stiffness * area
};


struct ProximityVF{
    inline uint get_vert() const { return indeces[0]; }
    inline Int3 get_face() const { return makeInt3(indeces[0], indeces[1], indeces[2]); }
    inline Float4 get_weight() const{ return vec1; }
    inline float get_weight(uint idx) const{ return get_weight()[idx]; }
    inline float get_vert_weight(uint idx) const{ return 1.0f; }
    inline Float3 get_face_weight() const{ return makeFloat3(vec1[1], vec1[2], vec1[3]); }

    template<uint NV> 
    inline void get_weights(THREAD float weight[NV]) const { 
        static_assert(NV == 1 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 1 or 4");
        for(uint i = 0; i < NV; i++) weight[i] = vec1[i]; 
    }
    inline Float3 get_t() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3 get_normal() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const { Float3 t = get_t(); return outer_product(t, t); }
    inline float get_area() const { return vec2[3]; }
    inline float get_stiff() const { return vec2[3]; }
    inline Int4 get_indices() const { return indeces; }

    // Self Collision Is Always Positive (You Dont Know The Correct Direction ~ , Just Fix It In E-F Pair Or CCD-Based Response)
    template<uint NV> 
    inline void get_indices(THREAD uint tet[NV]) const { 
        static_assert(NV == 1 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 1 or 4");
        for(uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative; 
    }

    inline void set_negative()      { for(uint i = 0; i < 4; i++) indeces[i] |= mask_is_negative; }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int4 get_indices_negative() const { Int4 tmp; for(uint i = 0; i < 4; i++) { tmp[i] = indeces[i] & ~mask_is_negative; } return tmp; }
    inline uint get_vert_negative() const { return indeces[0] & ~mask_is_negative; }

    ProximityVF(){}
    ProximityVF(CREF(uint) vid, CREF(Int3) f_vid, CREF(Float4) weight, CREF(float) area, CREF(Float3) t){
        indeces = makeInt4(vid, f_vid[0], f_vid[1], f_vid[2]);
        vec1 = weight;
        vec2 = makeFloat4(t[0], t[1], t[2], area);
    }
    ProximityVF(CREF(uint) vid, CREF(Int3) f_vid, CREF(Float4) weight, CREF(float) stiffness, CREF(float) area, CREF(Float3) normal){
        indeces = makeInt4(vid, f_vid[0], f_vid[1], f_vid[2]);
        vec1 = weight;
        vec2 = makeFloat4(normal[0], normal[1], normal[2], stiffness * area);
    }

private:
    Int4 indeces;
    Float4 vec1; // weight = (1, -bary[0], -bary[1], -bary[2])
    Float4 vec2; // t(3), area(1)
};

struct ProximityEE{
    inline Int2 get_edge1() const { return make<Int2>(indeces[0], indeces[1]); }
    inline Int2 get_edge2() const { return make<Int2>(indeces[2], indeces[3]); }
    inline Int4 get_indices() const { return indeces; }

    template<uint NV> 
    inline void get_indices(THREAD uint tet[NV]) const { 
        static_assert(NV == 2 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 2 or 4");
        for(uint i = 0; i < NV; i++) tet[i] = indeces[i] & ~mask_is_negative; 
    }

    inline Float4 get_weight() const { return vec1; }
    inline float get_weight(uint idx) const{ return vec1[idx]; }

    template<uint NV> 
    inline void get_weights(THREAD float weight[NV]) const { 
        static_assert(NV == 2 || NV == 4, "Wrong NumVerts In Collision Pair, Should Be 2 or 4");
        for(uint i = 0; i < NV; i++) weight[i] = vec1[i]; 
    }

    inline float get_area() const { return vec2[3]; }
    inline Float3 get_t() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const { Float3 t = get_t(); return outer_product(t, t); }
    inline Float2 get_a() const { return make<Float2>(-vec1[0], -vec1[1]); }
    inline Float2 get_b() const { return make<Float2>(vec1[2], vec1[3]); }

    inline void set_negative()      { for(uint i = 0; i < 4; i++) indeces[i] |= mask_is_negative; }
    inline bool is_negative() const { return indeces[0] & mask_is_negative; }
    inline Int4 get_indices_negative() const { Int4 tmp; for(uint i = 0; i < 4; i++) { tmp[i] = indeces[i] & ~mask_is_negative;} return tmp; }
    // inline Int2 get_edge1_negative()   const { return makeInt2(indeces[0] & ~mask_is_negative, indeces[1] & ~mask_is_negative); }

    ProximityEE(){}
    ProximityEE(CREF(Int2) edge1, CREF(Int2) edge2, CREF(Float4) weight, CREF(float) area, CREF(Float3) t){
        indeces = make<Int4>(edge1[0], edge1[1], edge2[0], edge2[1]);
        vec1 = weight;
        vec2 = make<Float4>(t[0], t[1], t[2], area);
    }

private:
    Int4 indeces;
    Float4 vec1; // weight = (-a[0], -a[1], b[0], b[1])
    Float4 vec2; // t(3), area(1)
};

struct ProximityEF{
    inline Int2 get_edge() const { return make<Int2>(indeces[0], indeces[1]); }
    inline Int3 get_face() const { return makeInt3(indeces[2], indeces[3], indeces[4]); }
    
    template<uint NV> 
    inline void get_indices(THREAD uint tet[NV]) const {  
        static_assert(NV == 2 || NV == 5, "Wrong NumVerts In Collision Pair, Should Be 2 or 5");
        for(uint i = 0; i < NV; i++) tet[i] = indeces[i]; 
    }
    template<uint NV> 
    inline void get_weights(THREAD float weight[NV]) const { 
        static_assert(NV == 2 || NV == 5, "Wrong NumVerts In Collision Pair, Should Be 2 or 5");
        weight[0] = vec1.x; 
        weight[1] = 1.f - vec1.x; 
        if(NV == 5){
            weight[2] = vec1.y; 
            weight[3] = vec1.z; 
            weight[4] = 1.f - vec1.y - vec1.z; 
        }   
    }
    inline float get_weight(uint idx) const{ 
        switch (idx) {
            case 0:
                return vec1.x;
            case 1:
                return 1.f - vec1.x;
            case 2:
                return vec1.y;
            case 3:
                return vec1.z;
            case 4:
                return 1.f - vec1.y - vec1.z;
            default:
                return 0.f;
        }
    }
    inline float get_area() const { return vec2[3]; }
    inline Float3 get_G() const { return makeFloat3(vec2[0], vec2[1], vec2[2]); }
    inline Float3x3 get_txt() const { Float3 t = get_G(); return outer_product(t, t); }

    ProximityEF(){}
    ProximityEF(const uint tet[5], CREF(Float3) weight, CREF(float) area, CREF(Float3) G){
        indeces[0] = tet[0];
        indeces[1] = tet[1];
        indeces[2] = tet[2];
        indeces[3] = tet[3];
        indeces[4] = tet[4];
        vec1 = weight;
        vec2 = make<Float4>(G[0], G[1], G[2], area);
    }

private:
    uint indeces[5];
    Float3 vec1; // (x, y, z) => weight = (x, 1-x, y, z, 1-y-z)
    Float4 vec2; // t(3), area(1)
};

