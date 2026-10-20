// test_raw_descriptor_weights.cpp
//
// The weights of a linear model with NORM true act on the raw descriptors:
// its fit converts the weights it solves for the normalised descriptors
// with raw_descriptor_map (raw_descriptor_weights.h).

#include "catch2/catch.hpp"

#include <tadah/core/core_types.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/raw_descriptor_weights.h>

using tadah::core::Matrix;
using tadah::mlip::Normaliser;
using tadah::mlip::raw_descriptor_map;

/* Pins: raw_descriptor_map for 2 intercept columns and 2 descriptor columns
 * (means 1, 2 and stdevs 2, 4 of columns 2, 3): J(i,i) = 1/stdev_i, the
 * intercept rows 1, 0, -1/2, -1/2 and 0, 1, -1/2, -1/2 (the same
 * correction on every species' intercept), and with no intercept column it
 * is refused naming BIAS. */
TEST_CASE("raw_descriptor_map maps normalised weights to raw ones",
          "[raw_descriptors]") {
  Normaliser norm;
  norm.mean = tadah::core::v_type(4);
  norm.std_dev = tadah::core::v_type(4);
  norm.mean[0] = norm.mean[1] = 0.0;
  norm.std_dev[0] = norm.std_dev[1] = 0.0;
  norm.mean[2] = 1.0;
  norm.mean[3] = 2.0;
  norm.std_dev[2] = 2.0;
  norm.std_dev[3] = 4.0;
  const Matrix J = raw_descriptor_map(norm, 4, 2);
  const double expect[4][4] = {{1, 0, -0.5, -0.5},
                               {0, 1, -0.5, -0.5},
                               {0, 0, 0.5, 0},
                               {0, 0, 0, 0.25}};
  for (std::size_t a = 0; a < 4; ++a)
    for (std::size_t b = 0; b < 4; ++b) {
      INFO(a << ", " << b);
      CHECK(J(a, b) == expect[a][b]);
    }
  CHECK_THROWS_WITH(raw_descriptor_map(norm, 4, 0), Catch::Contains("BIAS"));
}
