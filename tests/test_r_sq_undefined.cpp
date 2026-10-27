// The coefficient of determination of reference values that are all equal.
//
// R^2 = 1 - SS_res / SS_tot measures the residual against the spread of
// the reference values about their mean, SS_tot. When every reference
// value is the same (the forces of a dataset of ideal crystals, the energy
// per atom of a dataset of one structure) there is no spread, SS_tot is 0
// (or, after the mean's rounding, a few ulp), and R^2 has no value:
// Statistics::r_sq gives NaN, and so do the per-dataset R^2 of Analytics
// for such a dataset, the other datasets keeping theirs.
//
// The build uses -ffast-math, so NaN is told by its bits, not std::isnan.

#include "catch2/catch.hpp"

#include <tadah/core/core_types.h>
#include <tadah/core/vec3d.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/statistics.h>
#include <tadah/mlip/structure_db.h>

#include <cstdint>
#include <cstring>

using tadah::core::Vec3d;
using tadah::mlip::Analytics;
using tadah::mlip::Statistics;
using tadah::mlip::StructureDB;

namespace {

/* Whether @p x is a NaN, by its bits. */
bool is_nan_bits(double x) {
  std::uint64_t b = 0;
  std::memcpy(&b, &x, sizeof b);
  return (b & 0x7ff0000000000000ull) == 0x7ff0000000000000ull &&
         (b & 0x000fffffffffffffull) != 0;
}

tadah::core::t_type vec(std::initializer_list<double> v) {
  tadah::core::t_type out(v.size());
  std::size_t i = 0;
  for (double x : v) out(i++) = x;
  return out;
}

/* Two datasets of two 2-atom structures each: the first with forces that
 * differ, the second with every force component 0 (ideal crystals), and
 * different energies per atom in each; the prediction moves every force
 * component by 0.1 and every energy by 0.2. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 2, 4};
  db.structures.resize(4);
  for (std::size_t i = 0; i < 4; ++i) {
    db(i).energy = -2.0 * static_cast<double>(i + 1);
    db(i).atoms.resize(2);
    for (std::size_t a = 0; a < 2; ++a)
      db(i).atoms[a].force =
          i < 2 ? Vec3d(0.1 * (a + 1), -0.2 * (i + 1), 0.3 * (a + i))
                : Vec3d(0.0, 0.0, 0.0);
  }
  return db;
}

StructureDB prediction(const StructureDB &ref) {
  StructureDB p;
  p.dbidx = ref.dbidx;
  p.structures.resize(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    p(i).energy = ref(i).energy + 0.2;
    p(i).atoms.resize(2);
    for (std::size_t a = 0; a < 2; ++a)
      p(i).atoms[a].force = ref(i).atoms[a].force + Vec3d(0.1, 0.1, 0.1);
  }
  return p;
}

}  // namespace

// Pins: Statistics::r_sq of reference values that are all equal is NaN,
// whatever the prediction (equal to them or not, and for values whose
// mean is not exact in floating point, 0.1 three times); values that
// differ give 1 - SS_res / SS_tot.
// Before the fix: 1 - SS_res / 0, -inf for a prediction that differs from
// the reference (and NaN, 0/0, for one equal to it); for 0.1 three times
// the mean's rounding left SS_tot at a few 1e-34, and R^2 was about -1e32.
TEST_CASE("Statistics::r_sq of equal reference values has no value",
          "[analytics][r_sq]") {
  CHECK(is_nan_bits(Statistics::r_sq(vec({1.0, 1.0, 1.0}), vec({1.0, 2.0, 3.0}))));
  CHECK(is_nan_bits(Statistics::r_sq(vec({0.0, 0.0}), vec({0.0, 0.0}))));
  CHECK(is_nan_bits(Statistics::r_sq(vec({0.1, 0.1, 0.1}), vec({0.2, 0.1, 0.0}))));
  CHECK(is_nan_bits(Statistics::r_sq(vec({5.0}), vec({4.0}))));
  // 1 - (1 + 0 + 1) / ((-1)^2 + 0 + 1^2) = 0.
  CHECK(Statistics::r_sq(vec({1.0, 2.0, 3.0}), vec({2.0, 2.0, 2.0})) == Approx(0.0).margin(1e-15));
}

// Pins: the per-dataset force R^2 of a dataset whose reference forces are
// all 0 is NaN, and that of the other dataset is its own finite value; the
// per-dataset energy R^2 of a dataset of one structure is NaN.
// Before the fix: the force R^2 of the second dataset was -inf (SS_tot 0),
// and the energy R^2 of one structure -inf.
TEST_CASE("Analytics: the R^2 of a dataset of equal reference values has no "
          "value", "[analytics][r_sq]") {
  const StructureDB ref = reference();
  const StructureDB pred = prediction(ref);
  const auto f = Analytics(ref, pred).calc_f_r_sq();
  REQUIRE(f.size() == 2);
  CHECK_FALSE(is_nan_bits(f[0]));
  CHECK(f[0] < 1.0);
  CHECK(is_nan_bits(f[1]));
  const auto e = Analytics(ref, pred).calc_e_r_sq();
  REQUIRE(e.size() == 2);
  CHECK_FALSE(is_nan_bits(e[0]));
  CHECK_FALSE(is_nan_bits(e[1]));

  StructureDB one = ref;
  one.dbidx = {0, 1, 4};
  StructureDB one_pred = pred;
  one_pred.dbidx = one.dbidx;
  const auto e1 = Analytics(one, one_pred).calc_e_r_sq();
  REQUIRE(e1.size() == 2);
  CHECK(is_nan_bits(e1[0]));
  CHECK_FALSE(is_nan_bits(e1[1]));
}
