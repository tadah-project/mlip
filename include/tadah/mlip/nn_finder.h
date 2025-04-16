#ifndef NN_FINDER_H
#define NN_FINDER_H

#include <tadah/core/context.h>
#include <tadah/core/lapack.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
namespace tadah {
namespace mlip {

/**
 * @class NNFinder
 *
 * Nearest Neighbor Finder that constructs a full nearest neighbor list
 * for every atom in a structure, using:
 *  - a **binned** (linked-cell) approach if every cell dimension ≥ cutoff,
 *  - a **naive** approach otherwise (fallback).
 *
 * The cutoff is from context("RCUTMAX").
 */
class NNFinder {
private:
    double cutoff;      ///< The cutoff distance
    double cutoff_sq;   ///< cutoff^2 for distance checks

    /**
     * Return false if any cell dimension < cutoff => fallback to naive.
     */
    bool check_box(Structure &st);

    /**
     * Invert a 3×3 matrix (column-major) 
     */
/**
 * Invert a 3x3 matrix using the cofactor (adjugate) method,
 * matching the layout:
 *   M(r,c) stored at index [r*3 + c]
 *
 * So, for example:
 *   M(0,0) -> inM[0],
 *   M(1,0) -> inM[3],
 *   M(2,0) -> inM[6],
 *   M(0,1) -> inM[1],
 *   ...
 *
 * The output is also stored in the same pattern.
 *
 * Throws std::runtime_error if the matrix is near-singular.
 */
// inline void inverse_3x3_direct(const double* inM, double* outM)
// {
//     // Map input array -> matrix elements:
//     const double m00 = inM[0];  // M(0,0)
//     const double m10 = inM[1];  // M(1,0)
//     const double m20 = inM[2];  // M(2,0)
//
//     const double m01 = inM[3];  // M(0,1)
//     const double m11 = inM[4];  // M(1,1)
//     const double m21 = inM[5];  // M(2,1)
//
//     const double m02 = inM[6];  // M(0,2)
//     const double m12 = inM[7];  // M(1,2)
//     const double m22 = inM[8];  // M(2,2)
//
//     // ----------------------------------------------------------------------
//     // 1) Compute determinant:
//     //    det(M) = m00*(m11*m22 - m12*m21)
//     //           - m01*(m10*m22 - m12*m20)
//     //           + m02*(m10*m21 - m11*m20)
//     // ----------------------------------------------------------------------
//     double det = m00*(m11*m22 - m12*m21)
//                - m01*(m10*m22 - m12*m20)
//                + m02*(m10*m21 - m11*m20);
//
//     if (std::fabs(det) < 1e-14) {
//         throw std::runtime_error(
//             "inverse_3x3_cofactor_custom: matrix is near-singular."
//         );
//     }
//     double invDet = 1.0 / det;
//
//     // ----------------------------------------------------------------------
//     // 2) Cofactor / adjugate formula:
//     //
//     //    inv(0,0) = (m11*m22 - m12*m21)
//     //    inv(0,1) = -(m01*m22 - m02*m21)
//     //    inv(0,2) = (m01*m12 - m02*m11)
//     //
//     //    inv(1,0) = -(m10*m22 - m12*m20)
//     //    inv(1,1) = (m00*m22 - m02*m20)
//     //    inv(1,2) = -(m00*m12 - m02*m10)
//     //
//     //    inv(2,0) = (m10*m21 - m11*m20)
//     //    inv(2,1) = -(m00*m21 - m01*m20)
//     //    inv(2,2) = (m00*m11 - m01*m10)
//     //
//     // Multiply each cofactor by invDet.
//     // ----------------------------------------------------------------------
//     double i00 =  (m11*m22 - m12*m21) * invDet;
//     double i01 = -(m01*m22 - m02*m21) * invDet;
//     double i02 =  (m01*m12 - m02*m11) * invDet;
//
//     double i10 = -(m10*m22 - m12*m20) * invDet;
//     double i11 =  (m00*m22 - m02*m20) * invDet;
//     double i12 = -(m00*m12 - m02*m10) * invDet;
//
//     double i20 =  (m10*m21 - m11*m20) * invDet;
//     double i21 = -(m00*m21 - m01*m20) * invDet;
//     double i22 =  (m00*m11 - m01*m10) * invDet;
//
//     // ----------------------------------------------------------------------
//     // 3) Store the inverse back into outM in the same layout:
//     //    inv(r,c) => outM[r*3 + c]
//     // ----------------------------------------------------------------------
//     outM[0] = i00;  // inv(0,0)
//     outM[1] = i01;  // inv(0,1)
//     outM[2] = i02;  // inv(0,2)
//
//     outM[3] = i10;  // inv(1,0)
//     outM[4] = i11;  // inv(1,1)
//     outM[5] = i12;  // inv(1,2)
//
//     outM[6] = i20;  // inv(2,0)
//     outM[7] = i21;  // inv(2,1)
//     outM[8] = i22;  // inv(2,2)
// }

    /**
     * Naive approach to build neighbor lists.
     */
    void calc_naive(Structure &st);
void deduplicate(Structure &st);
    /**
     * Binning-based approach. If any dimension < cutoff, fallback to naive.
     */
    void calc_binned(Structure &st);

    /**
     * For naive approach, compute ±N to consider for image shifts.
     */
    void num_shifts(Structure &st, int N[3]);

public:
    /**
     * Construct with cutoff from context("RCUTMAX").
     */
    NNFinder(tadah::core::Context &context);

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses binned approach if possible, else naive fallback.
     */
    void calc(Structure &st);

    /**
     * Build nearest neighbors for each structure in a DB.
     */
    void calc(StructureDB &stdb);
};

} // namespace mlip
} // namespace tadah
#endif // NN_FINDER_H


