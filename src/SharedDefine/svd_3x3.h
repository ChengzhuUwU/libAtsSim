#pragma once

#include "float_n.h"
#include "float_n_n.h"

namespace sim 
{

namespace Local
{
    ConstExpr float _gamma  = 5.828427124 ;  // FOUR_GAMMA_SQUARED = sqrt(8)+3;
ConstExpr float _cstar  = 0.923879532 ;  // cos(pi/8)
ConstExpr float _sstar  = 0.3826834323;  // sin(p/8)
ConstExpr float EPSILON = 1e-6;

// #include <cuda.h>
// #include "math.h" // CUDA math library
// CUDA's rsqrt_scalar seems to be faster than the inlined approximation?

inline
float accurateSqrt(float x)
{
    return x * rsqrt_scalar(x);
}

inline
void condSwap(bool c, Thread float &X, Thread float &Y)
{
    // used in step 2
    float Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

inline
void condNegSwap(bool c, Thread float &X, Thread float &Y)
{
    // used in step 2 and 3
    float Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

// matrix multiplication M = A * B
inline
void multAB(float a11, float a12, float a13,
          float a21, float a22, float a23,
          float a31, float a32, float a33,
          //
          float b11, float b12, float b13,
          float b21, float b22, float b23,
          float b31, float b32, float b33,
          //
          Thread float &m11, Thread float &m12, Thread float &m13,
          Thread float &m21, Thread float &m22, Thread float &m23,
          Thread float &m31, Thread float &m32, Thread float &m33)
{

    m11=a11*b11 + a12*b21 + a13*b31; m12=a11*b12 + a12*b22 + a13*b32; m13=a11*b13 + a12*b23 + a13*b33;
    m21=a21*b11 + a22*b21 + a23*b31; m22=a21*b12 + a22*b22 + a23*b32; m23=a21*b13 + a22*b23 + a23*b33;
    m31=a31*b11 + a32*b21 + a33*b31; m32=a31*b12 + a32*b22 + a33*b32; m33=a31*b13 + a32*b23 + a33*b33;
}

// matrix multiplication M = Transpose[A] * B
inline
void multAtB(float a11, float a12, float a13,
          float a21, float a22, float a23,
          float a31, float a32, float a33,
          //
          float b11, float b12, float b13,
          float b21, float b22, float b23,
          float b31, float b32, float b33,
          //
          Thread float &m11, Thread float &m12, Thread float &m13,
          Thread float &m21, Thread float &m22, Thread float &m23,
          Thread float &m31, Thread float &m32, Thread float &m33)
{
  m11=a11*b11 + a21*b21 + a31*b31; m12=a11*b12 + a21*b22 + a31*b32; m13=a11*b13 + a21*b23 + a31*b33;
  m21=a12*b11 + a22*b21 + a32*b31; m22=a12*b12 + a22*b22 + a32*b32; m23=a12*b13 + a22*b23 + a32*b33;
  m31=a13*b11 + a23*b21 + a33*b31; m32=a13*b12 + a23*b22 + a33*b32; m33=a13*b13 + a23*b23 + a33*b33;
}

inline
void quatToMat3(const Thread float * qV,
Thread float &m11, Thread float &m12, Thread float &m13,
Thread float &m21, Thread float &m22, Thread float &m23,
Thread float &m31, Thread float &m32, Thread float &m33
)
{
    float w = qV[3];
    float x = qV[0];
    float y = qV[1];
    float z = qV[2];

    float qxx = x*x;
    float qyy = y*y;
    float qzz = z*z;
    float qxz = x*z;
    float qxy = x*y;
    float qyz = y*z;
    float qwx = w*x;
    float qwy = w*y;
    float qwz = w*z;

     m11=1 - 2*(qyy + qzz); m12=2*(qxy - qwz); m13=2*(qxz + qwy);
    m21=2*(qxy + qwz); m22=1 - 2*(qxx + qzz); m23=2*(qyz - qwx);
    m31=2*(qxz - qwy); m32=2*(qyz + qwx); m33=1 - 2*(qxx + qyy);
}

inline
void approximateGivensQuaternion(float a11, float a12, float a22, Thread float &ch, Thread float &sh)
{
/*
     * Given givens angle computed by approximateGivensAngles,
     * compute the corresponding rotation quaternion.
     */
    ch = 2*(a11-a22);
    sh = a12;
    bool b = _gamma*sh*sh < ch*ch;
    float w = rsqrt_scalar(ch*ch+sh*sh);
    ch=b?w*ch:_cstar;
    sh=b?w*sh:_sstar;
}

inline
void jacobiConjugation( const int x, const int y, const int z,
                        Thread float &s11,
                        Thread float &s21, Thread float &s22,
                        Thread float &s31, Thread float &s32, Thread float &s33,
                        Thread float * qV)
{
    float ch,sh;
    approximateGivensQuaternion(s11,s21,s22,ch,sh);

    float scale = ch*ch+sh*sh;
    float a = (ch*ch-sh*sh)/scale;
    float b = (2*sh*ch)/scale;

    // make temp copy of S
    float _s11 = s11;
    float _s21 = s21; float _s22 = s22;
    float _s31 = s31; float _s32 = s32; float _s33 = s33;

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    s11 =a*(a*_s11 + b*_s21) + b*(a*_s21 + b*_s22);
    s21 =a*(-b*_s11 + a*_s21) + b*(-b*_s21 + a*_s22);	s22=-b*(-b*_s11 + a*_s21) + a*(-b*_s21 + a*_s22);
    s31 =a*_s31 + b*_s32;								s32=-b*_s31 + a*_s32; s33=_s33;

    // update cumulative rotation qV
    float tmp[3];
    tmp[0]=qV[0]*sh;
    tmp[1]=qV[1]*sh;
    tmp[2]=qV[2]*sh;
    sh *= qV[3];

    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qV[z] += sh;
    qV[3] -= tmp[z]; // w
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    _s11 = s22;
    _s21 = s32; _s22 = s33;
    _s31 = s21; _s32 = s31; _s33 = s11;
    s11 = _s11;
    s21 = _s21; s22 = _s22;
    s31 = _s31; s32 = _s32; s33 = _s33;

}

inline
float dist2(float x, float y, float z)
{
    return x*x+y*y+z*z;
}

// finds transformation that diagonalizes a symmetric matrix
inline
void jacobiEigenanlysis( // symmetric matrix
                                Thread float &s11,
                                Thread float &s21, Thread float &s22,
                                Thread float &s31, Thread float &s32, Thread float &s33,
                                // quaternion representation of V
                                Thread float * qV)
{
    qV[3]=1; qV[0]=0;qV[1]=0;qV[2]=0; // follow same indexing convention as GLM
    for (int i=0;i<10;i++)
    {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        jacobiConjugation(0,1,2,s11,s21,s22,s31,s32,s33,qV); // p,q = 0,1
        jacobiConjugation(1,2,0,s11,s21,s22,s31,s32,s33,qV); // p,q = 1,2
        jacobiConjugation(2,0,1,s11,s21,s22,s31,s32,s33,qV); // p,q = 0,2
    }
}

inline
void sortSingularValues(// matrix that we want to decompose
                            Thread float &b11, Thread float &b12, Thread float &b13,
                            Thread float &b21, Thread float &b22, Thread float &b23,
                            Thread float &b31, Thread float &b32, Thread float &b33,
                          // sort V simultaneously
                            Thread float &v11, Thread float &v12, Thread float &v13,
                            Thread float &v21, Thread float &v22, Thread float &v23,
                            Thread float &v31, Thread float &v32, Thread float &v33)
{
    float rho1 = dist2(b11,b21,b31);
    float rho2 = dist2(b12,b22,b23);
    float rho3 = dist2(b13,b23,b33);
    bool c;
    c = rho1 < rho2;
    condNegSwap(c,b11,b12); condNegSwap(c,v11,v12);
    condNegSwap(c,b21,b22); condNegSwap(c,v21,v22);
    condNegSwap(c,b31,b32); condNegSwap(c,v31,v32);
    condSwap(c,rho1,rho2);
    c = rho1 < rho3;
    condNegSwap(c,b11,b13); condNegSwap(c,v11,v13);
    condNegSwap(c,b21,b23); condNegSwap(c,v21,v23);
    condNegSwap(c,b31,b33); condNegSwap(c,v31,v33);
    condSwap(c,rho1,rho3);
    c = rho2 < rho3;
    condNegSwap(c,b12,b13); condNegSwap(c,v12,v13);
    condNegSwap(c,b22,b23); condNegSwap(c,v22,v23);
    condNegSwap(c,b32,b33); condNegSwap(c,v32,v33);
}

inline
void QRGivensQuaternion(float a1, float a2, Thread float &ch, Thread float &sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float epsilon = EPSILON;
    float rho = accurateSqrt(a1*a1 + a2*a2);

    sh = rho > epsilon ? a2 : 0;
    ch = fabs(a1) + fmax(rho,epsilon);
    bool b = a1 < 0;
    condSwap(b,sh,ch);
    float w = rsqrt_scalar(ch*ch+sh*sh);
    ch *= w;
    sh *= w;
}

inline
void QRDecomposition(// matrix that we want to decompose
                            float b11, float b12, float b13,
                            float b21, float b22, float b23,
                            float b31, float b32, float b33,
                            // output Q
                            Thread float &q11, Thread float &q12, Thread float &q13,
                            Thread float &q21, Thread float &q22, Thread float &q23,
                            Thread float &q31, Thread float &q32, Thread float &q33,
                            // output R
                            Thread float &r11, Thread float &r12, Thread float &r13,
                            Thread float &r21, Thread float &r22, Thread float &r23,
                            Thread float &r31, Thread float &r32, Thread float &r33)
{
    float ch1,sh1,ch2,sh2,ch3,sh3;
    float a,b;

    // first givens rotation (ch,0,0,sh)
    QRGivensQuaternion(b11,b21,ch1,sh1);
    a=1-2*sh1*sh1;
    b=2*ch1*sh1;
    // apply B = Q' * B
    r11=a*b11+b*b21;  r12=a*b12+b*b22;  r13=a*b13+b*b23;
    r21=-b*b11+a*b21; r22=-b*b12+a*b22; r23=-b*b13+a*b23;
    r31=b31;          r32=b32;          r33=b33;

    // second givens rotation (ch,0,-sh,0)
    QRGivensQuaternion(r11,r31,ch2,sh2);
    a=1-2*sh2*sh2;
    b=2*ch2*sh2;
    // apply B = Q' * B;
    b11=a*r11+b*r31;  b12=a*r12+b*r32;  b13=a*r13+b*r33;
    b21=r21;           b22=r22;           b23=r23;
    b31=-b*r11+a*r31; b32=-b*r12+a*r32; b33=-b*r13+a*r33;

    // third givens rotation (ch,sh,0,0)
    QRGivensQuaternion(b22,b32,ch3,sh3);
    a=1-2*sh3*sh3;
    b=2*ch3*sh3;
    // R is now set to desired value
    r11=b11;             r12=b12;           r13=b13;
    r21=a*b21+b*b31;     r22=a*b22+b*b32;   r23=a*b23+b*b33;
    r31=-b*b21+a*b31;    r32=-b*b22+a*b32;  r33=-b*b23+a*b33;

    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    // the number of floating point operations for three quaternion multiplications
    // is more or less comparable to the explicit form of the joined matrix.
    // certainly more memory-efficient!
    float sh12=sh1*sh1;
    float sh22=sh2*sh2;
    float sh32=sh3*sh3;

    q11=(-1+2*sh12)*(-1+2*sh22);
    q12=4*ch2*ch3*(-1+2*sh12)*sh2*sh3+2*ch1*sh1*(-1+2*sh32);
    q13=4*ch1*ch3*sh1*sh3-2*ch2*(-1+2*sh12)*sh2*(-1+2*sh32);

    q21=2*ch1*sh1*(1-2*sh22);
    q22=-8*ch1*ch2*ch3*sh1*sh2*sh3+(-1+2*sh12)*(-1+2*sh32);
    q23=-2*ch3*sh3+4*sh1*(ch3*sh1*sh3+ch1*ch2*sh2*(-1+2*sh32));

    q31=2*ch2*sh2;
    q32=2*ch3*(1-2*sh22)*sh3;
    q33=(-1+2*sh22)*(-1+2*sh32);
}

static inline
void svd_built_in(// input A
        float a11, float a12, float a13,
        float a21, float a22, float a23,
        float a31, float a32, float a33,
        // output U
        Thread float &u11, Thread float &u12, Thread float &u13,
        Thread float &u21, Thread float &u22, Thread float &u23,
        Thread float &u31, Thread float &u32, Thread float &u33,
        // output S
        Thread float &s11, Thread float &s12, Thread float &s13,
        Thread float &s21, Thread float &s22, Thread float &s23,
        Thread float &s31, Thread float &s32, Thread float &s33,
        // output V
        Thread float &v11, Thread float &v12, Thread float &v13,
        Thread float &v21, Thread float &v22, Thread float &v23,
        Thread float &v31, Thread float &v32, Thread float &v33)
{
    // normal equations matrix
    float ATA11, ATA12, ATA13;
    float ATA21, ATA22, ATA23;
    float ATA31, ATA32, ATA33;

    multAtB(a11,a12,a13,a21,a22,a23,a31,a32,a33,
          a11,a12,a13,a21,a22,a23,a31,a32,a33,
          ATA11,ATA12,ATA13,ATA21,ATA22,ATA23,ATA31,ATA32,ATA33);

    // symmetric eigenalysis
    float qV[4];
    jacobiEigenanlysis( ATA11,ATA21,ATA22, ATA31,ATA32,ATA33,qV);
    quatToMat3(qV,v11,v12,v13,v21,v22,v23,v31,v32,v33);

    float b11, b12, b13;
    float b21, b22, b23;
    float b31, b32, b33;
    multAB(a11,a12,a13,a21,a22,a23,a31,a32,a33,
        v11,v12,v13,v21,v22,v23,v31,v32,v33,
        b11, b12, b13, b21, b22, b23, b31, b32, b33);

    // sort singular values and find V
    sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33,
                        v11,v12,v13,v21,v22,v23,v31,v32,v33);

    // QR decomposition
    QRDecomposition(b11, b12, b13, b21, b22, b23, b31, b32, b33,
    u11, u12, u13, u21, u22, u23, u31, u32, u33,
    s11, s12, s13, s21, s22, s23, s31, s32, s33
    );
}

}

inline void svd(ConstRef(Float3x3) F, ThreadRef(Float3x3) U, ThreadRef(Float3) Sigma, ThreadRef(Float3x3) V)
{
    Matrix3x3f tmpF(F);
    Matrix3x3f tmpU;
    Matrix3x3f tmpSigma;
    Matrix3x3f tmpV;
    Local::svd_built_in(
        tmpF(0, 0), tmpF(0, 1), tmpF(0, 2), 
        tmpF(1, 0), tmpF(1, 1), tmpF(1, 2), 
        tmpF(2, 0), tmpF(2, 1), tmpF(2, 2), 
        tmpU(0, 0), tmpU(0, 1), tmpU(0, 2), 
        tmpU(1, 0), tmpU(1, 1), tmpU(1, 2), 
        tmpU(2, 0), tmpU(2, 1), tmpU(2, 2), 
        tmpSigma(0, 0), tmpSigma(0, 1), tmpSigma(0, 2), 
        tmpSigma(1, 0), tmpSigma(1, 1), tmpSigma(1, 2), 
        tmpSigma(2, 0), tmpSigma(2, 1), tmpSigma(2, 2), 
        tmpV(0, 0), tmpV(0, 1), tmpV(0, 2), 
        tmpV(1, 0), tmpV(1, 1), tmpV(1, 2), 
        tmpV(2, 0), tmpV(2, 1), tmpV(2, 2));
    
    U = tmpU.get_mat();
    Sigma = tmpSigma.get_diag();
    V = tmpV.get_mat();
}   


}