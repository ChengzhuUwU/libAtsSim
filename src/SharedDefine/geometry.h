#pragma once

#include "float_n.h"
#include "float_n_n.h"

namespace SimGeometry{

// 点-边 的垂足
inline Float3 foot_ve(CREF(Float3) target, CREF(Float3) line_start, CREF(Float3) line_end){
	Float3 delta_A = target - line_start; // A -> P
	// Float3 delta_B = target - line_end; // B -> P
	Float3 delta_AB = line_end - line_start; // A -> B

	float r = dot_vec(delta_A, delta_AB) / dot_vec(delta_AB, delta_AB); // AP在线段方向上的投影（向量）

	if (r <= 0) {
		return line_start;
	}
	else if (r >= 1) {
		return line_end;
	}
	else {
		Float3 tmp = line_start + delta_AB * r; // 垂点坐标
		return tmp;
	}
}


namespace DistanceType{

// int Point_Triangle_Distance_Type(
//     CREF(Float3) p, 
//     CREF(Float3) t0, 
//     CREF(Float3) t1,
//     CREF(Float3) t2)
// {
//     Float3x2 basis;
//     Float3 e0 = t1 - t0;
//     Float3 e1 = t2 - t0;
//     set(basis, 0, 0, e0.x);
//     set(basis, 1, 0, e0.y);
//     set(basis, 2, 0, e0.z);
//     set(basis, 0, 1, e1.x);
//     set(basis, 1, 1, e1.y);
//     set(basis, 2, 1, e1.z);

//     const Float3 nVec = cross_vec(e0, e1);

//     Float3x2 param;

//     basis.row(1) = cross_vec(e0, nVec) ;// basis.row(0).cross(nVec);
//     param.col(0) = (basis * basis.transpose()).ldlt().solve(basis * (p - t0));
//     if (param(0, 0) > 0.0 && param(0, 0) < 1.0 && param(1, 0) >= 0.0) {
//         return 3; // PE t0t1
//     }
//     else {
//         basis.row(0) = (t2 - t1).transpose();

//         basis.row(1) = basis.row(0).cross(nVec);
//         param.col(1) = (basis * basis.transpose()).ldlt().solve(basis * (p - t1));
//         if (param(0, 1) > 0.0 && param(0, 1) < 1.0 && param(1, 1) >= 0.0) {
//             return 4; // PE t1t2
//         }
//         else {
//             basis.row(0) = (t0 - t2).transpose();

//             basis.row(1) = basis.row(0).cross(nVec);
//             param.col(2) = (basis * basis.transpose()).ldlt().solve(basis * (p - t2));
//             if (param(0, 2) > 0.0 && param(0, 2) < 1.0 && param(1, 2) >= 0.0) {
//                 return 5; // PE t2t0
//             }
//             else {
//                 if (param(0, 0) <= 0.0 && param(0, 2) >= 1.0) {
//                     return 0; // PP t0
//                 }
//                 else if (param(0, 1) <= 0.0 && param(0, 0) >= 1.0) {
//                     return 1; // PP t1
//                 }
//                 else if (param(0, 2) <= 0.0 && param(0, 1) >= 1.0) {
//                     return 2; // PP t2
//                 }
//                 else {
//                     return 6; // PT
//                 }
//             }
//         }
//     }
// }

// a more robust implementation of http://geomalgorithms.com/a07-_distance.html
inline int Edge_Edge_Distance_Type(
    CREF(Float3) ea0,
    CREF(Float3) ea1,
    CREF(Float3) eb0,
    CREF(Float3) eb1)
{
    Float3 u = ea1 - ea0;
    Float3 v = eb1 - eb0;
    Float3 w = ea0 - eb0;
    REAL a = length_squared_vec(u); // always >= 0
    REAL b = dot_vec(u, v);
    REAL c = length_squared_vec(v); // always >= 0
    REAL d = dot_vec(u, w);
    REAL e = dot_vec(v, w);
    REAL D = a * c - b * b; // always >= 0
    REAL tD = D; // tc = tN / tD, default tD = D >= 0
    REAL sN, tN;

    int defaultCase = 8;

    // compute the line parameters of the two closest points
    sN = (b * e - c * d);
    if (sN <= 0.0) { // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;
        defaultCase = 2;
    }
    else if (sN >= D) { // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        defaultCase = 5;
    }
    else {
        tN = (a * e - b * d);
        if (tN > 0.0 && tN < tD && (dot_vec(cross_vec(u, v), w) == 0.0 || length_squared_vec(cross_vec(u, v)) < 1.0e-20 * a * c)) {
            // if (tN > 0.0 && tN < tD && (u.cross(v).dot(w) == 0.0 || u.cross(v).squaredNorm() == 0.0)) {
            // std::cout << u.cross(v).squaredNorm() / (a * c) << ": " << sN << " " << D << ", " << tN << " " << tD << std::endl;
            // avoid coplanar or nearly parallel EE
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            }
            else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
        // else defaultCase stays as 8
    }

    if (tN <= 0.0) { // tc < 0 => the t=0 edge is visible
        // recompute sc for this edge
        if (-d <= 0.0) {
            return 0;
        }
        else if (-d >= a) {
            return 3;
        }
        else {
            return 6;
        }
    }
    else if (tN >= tD) { // tc > 1  => the t=1 edge is visible
        // recompute sc for this edge
        if ((-d + b) <= 0.0) {
            return 1;
        }
        else if ((-d + b) >= a) {
            return 4;
        }
        else {
            return 7;
        }
    }

    return defaultCase;
}

} // namespace DistanceType

inline void Point_Point_Distance(
	CREF(Float3) a, 
    CREF(Float3) b, 
    TREF(REAL) dist2)
{
    dist2 = length_squared_vec(a - b);
}

inline void Point_Edge_Distance(
	CREF(Float3) p, 
    CREF(Float3) e0, 
    CREF(Float3) e1, 
    TREF(REAL) dist2
	// ,TREF(Float3) outerPoint, TREF(Float3) innerPoint
	)
{
    // if constexpr (dim == 2) {
    //     const Eigen::Matrix<REAL, 2, 1> e = e1 - e0;
    //     REAL numerator = (e[1] * p[0] - e[0] * p[1] + e1[0] * e0[1] - e1[1] * e0[0]);
    //     dist2 = numerator * numerator / e.squaredNorm();
    // }
    {
        dist2 = length_squared_vec(cross_vec(e0 - p, e1 - p)) / length_squared_vec(e1 - e0);
    }
}


// void Point_Triangle_Distance_Unclassified(
//     Float3& p,
//     Float3& t0,
//     Float3& t1,
//     Float3& t2,
//     REAL& dist2)
// {
//     switch (DistanceType::Point_Triangle_Distance_Type(p, t0, t1, t2)) {
//     case 0: {
//         Point_Point_Distance(p, t0, dist2);
//         break;
//     }
//     case 1: {
//         Point_Point_Distance(p, t1, dist2);
//         break;
//     }
//     case 2: {
//         Point_Point_Distance(p, t2, dist2);
//         break;
//     }
//     case 3: {
//         Point_Edge_Distance(p, t0, t1, dist2);
//         break;
//     }
//     case 4: {
//         Point_Edge_Distance(p, t1, t2, dist2);
//         break;
//     }
//     case 5: {
//         Point_Edge_Distance(p, t2, t0, dist2);
//         break;
//     }
//     case 6: {
//         Point_Triangle_Distance(p, t0, t1, t2, dist2);
//         break;
//     }
//     default:
//         break;
//     }
// }


static inline void Edge_Edge_Distance(
    CREF(Float3) ea0, // ea
    CREF(Float3) ea1,
    CREF(Float3) eb0, // eb
    CREF(Float3) eb1,
    TREF(REAL)  dist2)
{
    Float3 b = cross_vec(ea1 - ea0, eb1 - eb0);
    REAL aTb = dot_vec(eb0 - ea0, b); // ea上一点到eb上的投影长度
    dist2 = aTb * aTb / length_squared_vec(b);
}

inline void Edge_Edge_Distance_Unclassified(
    CREF(Float3) ea0,
    CREF(Float3) ea1,
    CREF(Float3) eb0,
    CREF(Float3) eb1,
    TREF(REAL) dist2)
{
    switch (DistanceType::Edge_Edge_Distance_Type(ea0, ea1, eb0, eb1)) {
    case 0: {
        Point_Point_Distance(ea0, eb0, dist2);
        break;
    }
    case 1: {
        Point_Point_Distance(ea0, eb1, dist2);
        break;
    }
    case 2: {
        Point_Edge_Distance(ea0, eb0, eb1, dist2);
        break;
    }
    case 3: {
        Point_Point_Distance(ea1, eb0, dist2);
        break;
    }
    case 4: {
        Point_Point_Distance(ea1, eb1, dist2);
        break;
    }
    case 5: {
        Point_Edge_Distance(ea1, eb0, eb1, dist2);
        break;
    }
    case 6: {
        Point_Edge_Distance(eb0, ea0, ea1, dist2);
        break;
    }
    case 7: {
        Point_Edge_Distance(eb1, ea0, ea1, dist2);
        break;
    }
    case 8: {
        Edge_Edge_Distance(ea0, ea1, eb0, eb1, dist2);
        break;
    }
    default:
        break;
    }
}







// template<typename VecType> 
// static inline float stp (const VecType &u, const VecType &v, const VecType &w) {
//     return dot_vec(u, cross_vec(v, w));
// }
// template<typename ScalarType> 
// static inline ScalarType norm2(ScalarType value){ return value * value; }

// inline float signed_vf_distance(const Float3 &x,
//                           const Float3 &y0, const Float3 &y1, const Float3 &y2,
//                           Float3& normal, float *w)
// {
//     normal = cross_vec(normalize_vec(y1 - y0), normalize_vec(y2 - y0));
//     if (length_squared_vec(normal) < 1e-6)
//         return Float_max; // return ifinity
//     normal = normalize_vec(normal);
//     float h = dot_vec(x - y0, normal);
//     float b0 = stp(y1 - x, y2 - x, normal);
//     float b1 = stp(y2 - x, y0 - x, normal);
//     float b2 = stp(y0 - x, y1 - x, normal);
//     w[0] = 1;
//     w[1] = -b0 / (b0 + b1 + b2);
//     w[2] = -b1 / (b0 + b1 + b2);
//     w[3] = -b2 / (b0 + b1 + b2);
//     return h;
// }

// inline float signed_ve_distance(const Float3 &x, const Float3 &y0, const Float3 &y1,
//                           Float3& normal, float *w)
// {
//     Float3 e = y1 - y0;
//     float d = dot_vec(x - y0, e) / length_squared_vec(e);
//     if (d < 0 || d > 1.0)
//         return Float_max; // return ifinity
//     if (w)
//     {
//         w[0] = 1;
//         w[1] = -(1.0 - d);
//         w[2] = -d;
//         w[3] = 0;
//     }
//     Float3 dist = x - (y0 + d * e);
//     float l = length_vec(dist);
//     if (l > Epsilon)
//         normal = dist / l;
//     return l;
// }

// // x -> y
// inline float signed_ee_distance(const Float3 &x0, const Float3 &x1,
//                           const Float3 &y0, const Float3 &y1,
//                           Float3& normal, float *w)
// {
//     normal = cross_vec(normalize_vec(x1 - x0), normalize_vec(y1 - y0));
//     // 平行？
//     if (length_squared_vec(normal) < 1e-8)
//     {
//         // special case: parallel lines
//         Float3 e0 = normalize_vec(x1 - x0), e1 = normalize_vec(y1 - y0);

//         float p0min = dot_vec(x0, e0);
//         float p0max = dot_vec(x1, e0);
//         float p1min = dot_vec(y0, e0);
//         float p1max = dot_vec(y1, e0);
//         if (p1max < p1min)
//             swap_scalar(p1max, p1min);

//         float a = max_scalar(p0min, p1min), b = min_scalar(p0max, p1max), c = 0.5 * (a + b);
//         if (a > b)
//             return Float_max; // return ifinity

//         Float3 d = (y0 - x0) - dot_vec(y0 - x0, e0) * e0;
//         normal = normalize_vec(-d);

//         if (w)
//         {
//             w[1] = (c - dot_vec(x0, e0)) / length_vec(x1 - x0);
//             w[0] = 1.0 - w[1];
//             w[3] = -(dot_vec(e0, e1) * c - dot_vec(y0, e1)) / length_vec(y1 - y0);
//             w[2] = -1.0 - w[3];
//         }
//         return length_vec(d);
//     }
//     normal = normalize_vec(normal);
//     float h = dot_vec(x0 - y0, normal);
//     float a0 = stp(y1 - x1, y0 - x1, normal), a1 = stp(y0 - x0, y1 - x0, normal);
//     float b0 = stp(x0 - y1, x1 - y1, normal), b1 = stp(x1 - y0, x0 - y0, normal);
//     w[0] = a0 / (a0 + a1);
//     w[1] = a1 / (a0 + a1);
//     w[2] = -b0 / (b0 + b1);
//     w[3] = -b1 / (b0 + b1);
//     return h;
// }

// inline bool set_unsigned_ve_distance(const Float3 &x, const Float3 &y0, const Float3 &y1,
//                               float& _d, Float3& _n,
//                               float& _wx, float& _wy0, float& _wy1)
// {
//     // float t = clamp(dot_vec(x-y0, y1-y0)/dot_vec(y1-y0, y1-y0), 0., 1.);
//     float t = clamp_scalar(dot_vec(x - y0, y1 - y0) / dot_vec(y1 - y0, y1 - y0), 0.f, 1.f);
//     Float3 y = y0 + t * (y1 - y0);
//     float d = length_vec(x - y);
//     if (d < _d)
//     {
//         _d = d;
//         _n = normalize_vec(x - y);
//         _wx = 1;
//         _wy0 = 1 - t;
//         _wy1 = t;
//         return true;
//     }
//     return false;
// }

// inline bool set_unsigned_vf_distance(const Float3 &x,
//                               const Float3 &y0, const Float3 &y1, const Float3 &y2,
//                               float& _d, Float3& _n,
//                               float& _wx,
//                               float& _wy0, float& _wy1, float& _wy2)
// {
//     Float3 n = normalize_vec(cross_vec(normalize_vec(y1 - y0), normalize_vec(y2 - y0)));
//     float d = abs_scalar(dot_vec(x - y0, n));
//     float b0 = stp(y1 - x, y2 - x, n);
//     float b1 = stp(y2 - x, y0 - x, n);
//     float b2 = stp(y0 - x, y1 - x, n);
//     if (d < _d && b0 >= 0 && b1 >= 0 && b2 >= 0)
//     {
//         _d = d;
//         _n = n;
//         _wx = 1;
//         _wy0 = -b0 / (b0 + b1 + b2);
//         _wy1 = -b1 / (b0 + b1 + b2);
//         _wy2 = -b2 / (b0 + b1 + b2);
//         return true;
//     }
//     bool success = false;
//     if (b0 < 0 && set_unsigned_ve_distance(x, y1, y2, _d, _n, _wx, _wy1, _wy2))
//     {
//         success = true;
//         _wy0 = 0;
//     }
//     if (b1 < 0 && set_unsigned_ve_distance(x, y2, y0, _d, _n, _wx, _wy2, _wy0))
//     {
//         success = true;
//         _wy1 = 0;
//     }
//     if (b2 < 0 && set_unsigned_ve_distance(x, y0, y1, _d, _n, _wx, _wy0, _wy1))
//     {
//         success = true;
//         _wy2 = 0;
//     }
//     return success;
// }

// inline bool set_unsigned_ee_distance(const Float3 &x0, const Float3 &x1,
//                               const Float3 &y0, const Float3 &y1,
//                               float& _d, Float3& _n,
//                               float& _wx0, float& _wx1,
//                               float& _wy0, float& _wy1)
// {
//     Float3 n = normalize_vec(cross_vec(normalize_vec(x1 - x0), normalize_vec(y1 - y0)));
//     float d = abs_scalar(dot_vec(x0 - y0, n));
//     float a0 = stp(y1 - x1, y0 - x1, n), a1 = stp(y0 - x0, y1 - x0, n);
//     float b0 = stp(x0 - y1, x1 - y1, n), b1 = stp(x1 - y0, x0 - y0, n);
//     if (d < _d && a0 >= 0 && a1 >= 0 && b0 >= 0 && b1 >= 0)
//     {
//         _d = d;
//         _n = n;
//         _wx0 = a0 / (a0 + a1);
//         _wx1 = a1 / (a0 + a1);
//         _wy0 = -b0 / (b0 + b1);
//         _wy1 = -b1 / (b0 + b1);
//         return true;
//     }
//     bool success = false;
//     if (a0 < 0 && set_unsigned_ve_distance(x1, y0, y1, _d, _n, _wx1, _wy0, _wy1))
//     {
//         success = true;
//         _wx0 = 0;
//     }
//     if (a1 < 0 && set_unsigned_ve_distance(x0, y0, y1, _d, _n, _wx0, _wy0, _wy1))
//     {
//         success = true;
//         _wx1 = 0;
//     }
//     if (b0 < 0 && set_unsigned_ve_distance(y1, x0, x1, _d, _n, _wy1, _wx0, _wx1))
//     {
//         success = true;
//         _wy0 = 0;
//         _n = -_n;
//     }
//     if (b1 < 0 && set_unsigned_ve_distance(y0, x0, x1, _d, _n, _wy0, _wx0, _wx1))
//     {
//         success = true;
//         _wy1 = 0;
//         _n = -_n;
//     }
//     return success;
// }

// inline float unsigned_vf_distance(const Float3 &x,
//                             const Float3 &y0, const Float3 &y1, const Float3 &y2,
//                             Float3& normal, float w[4])
// {
//     float _w[4];
//     if (!w)
//         w = _w;
//     float d = Float_max; // 0
//     set_unsigned_vf_distance(x, y0, y1, y2, d, normal, w[0], w[1], w[2], w[3]);
//     return d;
// }

// inline float unsigned_ee_distance(const Float3 &x0, const Float3 &x1,
//                             const Float3 &y0, const Float3 &y1,
//                             Float3& normal, float w[4])
// {
//     float _w[4];
//     if (!w)
//         w = _w;
//     float d = Float_max;
//     set_unsigned_ee_distance(x0, x1, y0, y1, d, normal, w[0], w[1], w[2], w[3]);
//     return d;
// }

// inline Float3 get_barycentric_coords(ARRAY(Float2) metarial_position, const Float2 &point, const Face& face)
// {
//     // Compute vectors
//     Float2 v0 = (metarial_position[face[0]] - metarial_position[face[2]]);
//     Float2 v1 = (metarial_position[face[1]] - metarial_position[face[2]]);
//     Float2 v2 = point - (f->v[2]->u);
//     // Compute dot_vec products
//     float dot00 = dot_vec(v0, v0);
//     float dot01 = dot_vec(v0, v1);
//     float dot02 = dot_vec(v0, v2);
//     float dot11 = dot_vec(v1, v1);
//     float dot12 = dot_vec(v1, v2);
//     // Compute barycentric coordinates
//     float invDenom = 1.f / (dot00 * dot11 - dot01 * dot01);
//     float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
//     float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
//     return make<Float3>(u, v, 1 - u - v);
// }



//
// Habok
//


} // namespace SimGeometry