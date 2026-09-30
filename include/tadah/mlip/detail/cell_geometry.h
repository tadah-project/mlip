#ifndef TADAH_MLIP_DETAIL_CELL_GEOMETRY_H
#define TADAH_MLIP_DETAIL_CELL_GEOMETRY_H

// Internal helpers shared by NNFinder and nn_cache; not part of the public
// API. Rows of a cell are the lattice vectors a_k (Structure::cell), a
// position is r = f0 a0 + f1 a1 + f2 a2, and the fractional coordinate
// f_k = r . b*_k with the reciprocal vectors b*_k = (a_{k+1} x a_{k+2}) / V.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tadah {
namespace mlip {
namespace detail {

/** Unit roundoff of double, 2^-53. */
constexpr double kUnitRoundoff = 0x1p-53;

/** True unless @p x is an infinity or a NaN. Tested on the exponent bits
 *  because -ffast-math lets the compiler fold std::isfinite to true. */
inline bool is_finite_bits(double x) {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "double must be 64-bit");
  static_assert(std::numeric_limits<double>::is_iec559,
                "double must be IEEE 754");
  std::uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

/** Geometry of the lattice spanned by the rows a[0], a[1], a[2]. */
struct CellGeometry {
  double a[3][3];       ///< a[k][c]: component c of lattice vector a_k
  double cross[3][3];   ///< a_{k+1} x a_{k+2}
  double volume;        ///< a_0 . (a_1 x a_2), signed
  double bstar[3][3];   ///< reciprocal vectors b*_k = cross_k / volume
  double length[3];     ///< |a_k|
  double recip[3];      ///< |b*_k|
  double height[3];     ///< |volume| / |a_{k+1} x a_{k+2}|: spacing of the
                        ///< lattice planes spanned by a_{k+1} and a_{k+2}
  double Q;             ///< orthogonality defect |a0||a1||a2| / |volume|
  double G;             ///< sum_k |a_k| |b*_k| (3 for an orthogonal cell)
};

/** Geometry of the rows @p a. For a singular cell volume is 0 and the
 *  derived quantities are infinite or NaN; callers check. */
inline CellGeometry cell_geometry(const double a[3][3]) {
  CellGeometry g{};
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) g.a[k][c] = a[k][c];
  for (int k = 0; k < 3; ++k) {
    const double *u = a[(k + 1) % 3];
    const double *v = a[(k + 2) % 3];
    g.cross[k][0] = u[1] * v[2] - u[2] * v[1];
    g.cross[k][1] = u[2] * v[0] - u[0] * v[2];
    g.cross[k][2] = u[0] * v[1] - u[1] * v[0];
  }
  g.volume = a[0][0] * g.cross[0][0] + a[0][1] * g.cross[0][1] +
             a[0][2] * g.cross[0][2];
  const double absV = std::fabs(g.volume);
  g.G = 0.0;
  for (int k = 0; k < 3; ++k) {
    for (int c = 0; c < 3; ++c) g.bstar[k][c] = g.cross[k][c] / g.volume;
    const double cn = std::sqrt(g.cross[k][0] * g.cross[k][0] +
                                g.cross[k][1] * g.cross[k][1] +
                                g.cross[k][2] * g.cross[k][2]);
    g.length[k] = std::sqrt(a[k][0] * a[k][0] + a[k][1] * a[k][1] +
                            a[k][2] * a[k][2]);
    g.recip[k] = cn / absV;
    g.height[k] = absV / cn;
    g.G += g.length[k] * g.recip[k];
  }
  g.Q = g.length[0] * g.length[1] * g.length[2] / absV;
  return g;
}

/** Bound on the rounding error of a pair distance |r_i - (r_j + s.A)|
 *  computed in double by NNFinder or by any consumer, whatever the order
 *  of the operations or the use of FMA: 16 eps (G + 2)(R + 2 rho), where
 *  rho = max |r_i| and the shift s is one that can reach distance R
 *  (|s_k| <= (R + 2 rho) |b*_k|). */
inline double distance_rounding_bound(double G, double R, double rho) {
  return 16.0 * kUnitRoundoff * (G + 2.0) * (R + 2.0 * rho);
}

}  // namespace detail
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DETAIL_CELL_GEOMETRY_H
