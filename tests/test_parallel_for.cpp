// mlip::parallel_for runs the train/predict kernels over structures, under
// OpenMP when no pool backend is installed. An exception may not leave an
// OpenMP region (the runtime calls std::terminate), so a kernel that refuses
// one structure -- e.g. a neighbour list shorter than the model cutoff --
// would end the process instead of reaching the caller. parallel_for must
// run every index and then rethrow the exception of the lowest failing one.

#include "catch2/catch.hpp"

#include <tadah/mlip/parallel.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("parallel_for carries the exception of the lowest failing index "
          "out of its loop",
          "[parallel_for]") {
  REQUIRE_FALSE(tadah::mlip::has_parallel_backend());

  const std::size_t n = 40;
  std::vector<int> ran(n, 0);
  auto body = [&](std::size_t i) {
    ran[i] = 1;
    if (i == 7 || i == 23 || i == 31)
      throw std::invalid_argument("index " + std::to_string(i));
  };

  bool caught = false;
  std::string what;
  try {
    tadah::mlip::parallel_for(n, body);
  } catch (const std::invalid_argument &e) {
    caught = true;
    what = e.what();
  }
  CHECK(caught);
  CHECK(what == "index 7");
  // Every index ran, failing or not.
  for (std::size_t i = 0; i < n; ++i) {
    INFO("index " << i);
    CHECK(ran[i] == 1);
  }

  // Without a failure nothing is thrown.
  std::fill(ran.begin(), ran.end(), 0);
  CHECK_NOTHROW(tadah::mlip::parallel_for(n, [&](std::size_t i) { ran[i] = 1; }));
  for (std::size_t i = 0; i < n; ++i) CHECK(ran[i] == 1);
}
