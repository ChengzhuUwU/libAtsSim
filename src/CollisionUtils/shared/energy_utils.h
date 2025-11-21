#pragma once

#include "large_matrix.h"

namespace SimContactEnergy {

ConstExpr float _eps = 0.015f;
ConstExpr float _inverseEps = 1e-6;
ConstExpr float _separationEps = 1e-4;
ConstExpr float _tooSmall = 1e-6;
ConstExpr float _mu = 1e3;
ConstExpr float _ccd_eps = 0.01f;

///////////////////////////////////////////////////////////////////////
// get the barycentric coordinate of the projection of v[0] onto the triangle
// formed by v[1], v[2], v[3]
///////////////////////////////////////////////////////////////////////
inline Float3 getBarycentricCoordinates(const Float3 vertices[4]) {
    Float3 v0 = vertices[1];
    Float3 v1 = vertices[2];
    Float3 v2 = vertices[3];

    Float3 e1 = v1 - v0;
    Float3 e2 = v2 - v0;
    const Float3 n = cross_vec(e1, e2);
    const Float3 nHat = normalize_vec(n);                                   // 面法线
    const Float3 v = vertices[0] - (dot_vec(nHat, vertices[0] - v0)) * nHat;// 点在面上投影点

    // get the barycentric coordinates
    const Float3 na = cross_vec(v2 - v1, v - v1);
    const Float3 nb = cross_vec(v0 - v2, v - v2);
    const Float3 nc = cross_vec(v1 - v0, v - v0);
    const Float3 barycentric = make<Float3>(
        dot_vec(n, na) / length_squared_vec(n),
        dot_vec(n, nb) / length_squared_vec(n),
        dot_vec(n, nc) / length_squared_vec(n));

    return barycentric;
}

///////////////////////////////////////////////////////////////////////
// get the barycentric coordinate of the projection of v[0] onto the triangle
// formed by v[1], v[2], v[3]
///////////////////////////////////////////////////////////////////////
inline Float3 getBarycentricCoordinates(const Thread LargeVector<12> &vertices) {
    Float3 vs[4];
    for (int x = 0; x < 4; x++) {
        vs[x] = vertices[x];
    }
    return getBarycentricCoordinates(vs);
}

///////////////////////////////////////////////////////////////////////
// find the distance from a line segment (v1, v2) to a point (v0)
///////////////////////////////////////////////////////////////////////
inline REAL pointLineDistance(ConstRef(Float3) v0, ConstRef(Float3) v1, ConstRef(Float3) v2) {
    const Float3 e0 = v0 - v1;
    const Float3 e1 = v2 - v1;
    const Float3 e1hat = normalize_vec(e1);
    const REAL projection = dot_vec(e0, e1hat);

    // if it projects onto the line segment, use that length
    if (projection > 0.0 && projection < length_vec(e1)) {
        const Float3 normal = e0 - projection * e1hat;
        return length_vec(normal);
    }

    // if it doesn't, find the point-point distances
    const REAL diff01 = length_vec(v0 - v1);
    const REAL diff02 = length_vec(v0 - v2);

    return (diff01 < diff02) ? diff01 : diff02;
}

///////////////////////////////////////////////////////////////////////
// get the linear interpolation coordinates from v0 to the line segment
// between v1 and v2
///////////////////////////////////////////////////////////////////////
inline Float2 getLerp(ConstRef(Float3) v0, ConstRef(Float3) v1, ConstRef(Float3) v2) {
    const Float3 e0 = v0 - v1;
    const Float3 e1 = v2 - v1;
    const Float3 e1hat = normalize_vec(e1);
    const REAL projection = dot_vec(e0, e1hat);

    if (projection < 0.0)
        return makeFloat2(1.0f, 0.0f);

    if (projection >= length_vec(e1))
        return makeFloat2(0.0f, 1.0f);

    const REAL ratio = projection / length_vec(e1);
    return makeFloat2(1.0f - ratio, ratio);
}

///////////////////////////////////////////////////////////////////////
// get the barycentric coordinate of the projection of v[0] onto the triangle
// formed by v[1], v[2], v[3]
//
// but, if the projection is actually outside, project to all of the
// edges and find the closest point that's still inside the triangle
///////////////////////////////////////////////////////////////////////
inline Float3 getInsideBarycentricCoordinates(const Thread Float3 vertices[4], Float3 barycentric) {
    // if it's already inside, we're all done
    if (barycentric[0] >= 0.0f &&
        barycentric[1] >= 0.0f &&
        barycentric[2] >= 0.0f)
        return barycentric;

    // TODO : optimize
    // const Float3 e0 = v1 - v0;
    // const Float3 e1 = v2 - v1;
    // const Float3 e2 = v0 - v2;
    // const Float3 d0 = v2 - v1;
    // const Float3 e1hat = normalize_vec(e1);
    // const REAL projection = dot_vec(e0, e1hat);
    // if (projection > 0.0 && projection < length_vec(e1))
    // {
    //   const Float3 normal = e0 - projection * e1hat;
    //   return length_vec(normal);
    // }
    // const REAL diff01 = length_vec(v0 - v1);
    // const REAL diff02 = length_vec(v0 - v2);
    // return (diff01 < diff02) ? diff01 : diff02;

    // find distance to all the line segments
    //
    // there's lots of redundant computation between here and getLerp,
    // but let's get it working and see if it fixes the actual
    // artifact before optimizing
    REAL distance12 = pointLineDistance(vertices[0], vertices[1], vertices[2]);
    REAL distance23 = pointLineDistance(vertices[0], vertices[2], vertices[3]);
    REAL distance31 = pointLineDistance(vertices[0], vertices[3], vertices[1]);

    // less than or equal is important here, otherwise fallthrough breaks
    if (distance12 <= distance23 && distance12 <= distance31) {
        Float2 lerp = getLerp(vertices[0], vertices[1], vertices[2]);
        barycentric[0] = lerp[0];
        barycentric[1] = lerp[1];
        barycentric[2] = 0.0f;
        return barycentric;
    }

    // less than or equal is important here, otherwise fallthrough breaks
    if (distance23 <= distance12 && distance23 <= distance31) {
        Float2 lerp = getLerp(vertices[0], vertices[2], vertices[3]);
        barycentric[0] = 0.0f;
        barycentric[1] = lerp[0];
        barycentric[2] = lerp[1];
        return barycentric;
    }

    // else it must be the 31 case
    Float2 lerp = getLerp(vertices[0], vertices[3], vertices[1]);
    barycentric[0] = lerp[1];
    barycentric[1] = 0.0f;
    barycentric[2] = lerp[0];
    return barycentric;
}

inline Float3 getInsideBarycentricCoordinates(const Thread Float3 vertices[4]) {
    Float3 barycentric = getBarycentricCoordinates(vertices);
    return getInsideBarycentricCoordinates(vertices, barycentric);
}

inline Float3 getInsideBarycentricCoordinates(const Thread VECTOR12 &vertices) {
    return getInsideBarycentricCoordinates(vertices.ptr());
}

}// namespace SimContactEnergy