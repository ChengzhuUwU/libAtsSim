#pragma once

#include "float_n_n.h"
#include "float_n.h"
#include "bits_utils.h"
#include <large_matrix.h>
#include <svd_3x3.h>

namespace FEM
{
    /// \brief Extract the rotational part of the given square matrix by performing polar decomposion.
    ///
    /// \details This implementation is based on SVD. It checks the determinant to avoid any reflection.
    template <typename Derived>
    inline Float3x3 extractRotation(const THREAD Float3x3& F)
    {
        Float3x3 U, V;
        Float3 Sigma;
        sim::svd(F, U, Sigma, V);
        const Float3x3 R = (U * transpose_mat(V));
        // assert(std::abs(std::abs(R.determinant()) - 1.0) < 1e-04);
        return (determinant_mat(R) > 0) ? R : -1.f * R;

        // if constexpr (Derived::RowsAtCompileTime == 2)
        // {
        //     // Just ignore reflection (if any)
        //     // TODO: Discuss whether this is a good strategy, or not
        //     return R;
        // }
        // else if constexpr (Derived::RowsAtCompileTime == 3)
        // {
        //     // Correct reflection (if any)
        //     return (determinant_mat(R) > 0) ? R : -R;
        // }
    }


    /// \brief Calculate the first Lame parameter, $\lambda$.
    ///
    /// \details Reference: [1]
    template <typename Scalar> constexpr Scalar 
    calcFirstLame(const Scalar youngs_modulus, const Scalar poisson_ratio)
    {
        return youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    }

    /// \brief Calculate the second Lame parameter, $\mu$. This is the 'stretch' term in cloth simulation
    ///
    /// \details Reference: [1]
    template <typename Scalar> constexpr Scalar 
    calcSecondLame(const Scalar youngs_modulus, const Scalar poisson_ratio)
    {
        return youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    }


    /// \brief Calculate either the "deformed" shape matrix (D_s) or "reference" shape matrix (D_m) of a triangle.
    ///
    /// \param x_0 A 2D vector.
    ///
    /// \param x_1 A 2D vector.
    ///
    /// \param x_2 A 2D vector.
    ///
    /// \details Reference: [1]
    inline Float2x3
    calc2dShapeMatrix(const THREAD Float3& x_0,
                      const THREAD Float3& x_1,
                      const THREAD Float3& x_2)
    {
        // using Mat = Eigen::Matrix<typename Derived::Scalar, 2, 2>;

        Float2x3 shape_matrix;
        set(shape_matrix, 0, x_1 - x_0);
        set(shape_matrix, 1, x_2 - x_0);
        return shape_matrix;
    }




    /// \brief Calculate the Green strain tensor for a finite element.
    ///
    /// \param deform_grad The deformation gradient matrix, which should be either 2-by-2 (2D element in 2D), 3-by-3 (3D
    /// element in 3D), or 3-by-2 (2D element in 3D).
    inline Float2x2 calcGreenStrain(const THREAD Float2x3& deform_grad)
    {
        // 1/2 * (F^T - I)
        return 0.5f * (transpose_mat(deform_grad) * deform_grad - Identity2x2);
    }
    /// \brief Calculate the Green strain tensor for a finite element.
    ///
    /// \param deform_grad The deformation gradient matrix, which should be either 2-by-2 (2D element in 2D), 3-by-3 (3D
    /// element in 3D), or 3-by-2 (2D element in 3D).
    inline Float3x3 calcGreenStrain(const THREAD Float3x3& deform_grad)
    {
        // 1/2 * (F^T - I)
        return 0.5f * (transpose_mat(deform_grad) * deform_grad - Identity3x3);
    }








    /// \details Eq. 3.4 in [1]
    // inline float calcCoRotationalEnergyDensity(const Float3x3& deform_grad,
    //                                            const float     first_lame,
    //                                            const float     second_lame)
    // {

    //     const auto R     = extractRotation(deform_grad);
    //     const auto S     = R.transpose() * deform_grad;
    //     const auto I     = Mat::Identity();
    //     const auto trace = (S - I).trace();

    //     assert(deform_grad.isApprox(R * S));

    //     return second_lame * (deform_grad - R).squaredNorm() + 0.5 * first_lame * trace * trace;
    // }

    // /// \details Eq. 3.5 in [1]
    // template <typename Derived>
    // Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    // calcCoRotationalPiolaStress(const Eigen::MatrixBase<Derived>& deform_grad,
    //                             const typename Derived::Scalar    first_lame,
    //                             const typename Derived::Scalar    second_lame)
    // {
    //     using Mat = Eigen::Matrix<typename Derived::Scalar, Derived::ColsAtCompileTime, Derived::ColsAtCompileTime>;

    //     const auto R     = extractRotation(deform_grad);
    //     const auto S     = R.transpose() * deform_grad;
    //     const auto I     = Mat::Identity();
    //     const auto trace = (S - I).trace();

    //     assert(deform_grad.isApprox(R * S));

    //     return 2.0 * second_lame * (deform_grad - R) + first_lame * trace * R;
    // }

    /// \details The first equation in Sec. 3.3 in [1]
    inline float 
    calcStVenantKirchhoffEnergyDensity(const THREAD Float2x3& deform_grad, 
                                       const float first_lame, // lambda
                                       const float second_lame) // mu
    {
        //
        // Isotropic Saint Venant-Kirchhoff
        // \Psi_{StVK} = \frac{1}{2} tr(\epsilon)^2 + \mu tr(\epsilon)^2
        //
        const Float2x2 E  = calcGreenStrain(deform_grad); // epsilon
        const float trace = trace_mat(E);
        const float squaredNorm_E = length_squared_vec(get(E, 0)) + length_squared_vec(get(E, 1));
        return 0.5f * first_lame * trace * trace + second_lame * squaredNorm_E;
    }

    /// \details Eq. 3.3 in [1]
    inline Float2x3 
    calcStVenantKirchhoffPiolaStress(const THREAD Float2x3& deform_grad, 
                                     const float first_lame, 
                                     const float second_lame)
    {
        // Piola-Kirchhoff stress tensor 
        // P = lambda * tr(epsilon) + 2 * mu * F * epsilon
        const Float2x2 E = calcGreenStrain(deform_grad);
        return  first_lame * trace_mat(E) * deform_grad + 2.f * second_lame * deform_grad * E;
    }

    inline Float2x3
    calc2dTriangleDeformGrad(const THREAD Float3& x_0,
                             const THREAD Float3& x_1,
                             const THREAD Float3& x_2,
                             const THREAD Float2x2& rest_shape_mat_inv)
    {
        const auto D_s = calc2dShapeMatrix(x_0, x_1, x_2);
        const auto F   = D_s * rest_shape_mat_inv;

        return F;
    }






// namespace Tet
// {
// /// \brief Calculate either the "deformed" shape matrix (D_s) or "reference" shape matrix (D_m) of a tetrahedron.
// ///
// /// \param x_0 A 3D vector.
// ///
// /// \param x_1 A 3D vector.
// ///
// /// \param x_2 A 3D vector.
// ///
// /// \param x_3 A 3D vector.
// ///
// /// \details Reference: [1]
// inline Float3x3 calc3dShapeMatrix(const THREAD Float3& x_0,
//                                   const THREAD Float3& x_1,
//                                   const THREAD Float3& x_2,
//                                   const THREAD Float3& x_3)
// {
//     Float3x3 shape_matrix;
//     set(shape_matrix, 0, x_1 - x_0);
//     set(shape_matrix, 1, x_2 - x_0);
//     set(shape_matrix, 2, x_3 - x_0);
//     return shape_matrix;
// }

// }


}



