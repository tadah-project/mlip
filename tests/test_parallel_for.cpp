// mlip::parallel_for runs the train/predict kernels over structures, under
// OpenMP when no pool backend is installed. An exception may not leave an
// OpenMP region (the runtime calls std::terminate), so a kernel that refuses
// one structure -- e.g. a neighbour list shorter than the model cutoff --
// would end the process instead of reaching the caller. parallel_for must
// run every index and then rethrow the exception of the lowest failing one.

#include "catch2/catch.hpp"

#include <tadah/mlip/parallel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

// With several threads the lowest failing index need not throw first. Here
// it waits until the two higher failing indices have thrown, so an
// implementation that keeps the first exception thrown reports "index 23"
// or "index 31". Hidden: the CTest entry mlip_parallel_for_threads runs it
// with OMP_NUM_THREADS=4 (the suite otherwise runs with one thread, where
// the lowest failing index is always the first to throw).
TEST_CASE("parallel_for with several threads rethrows the lowest failing "
          "index even when it throws last",
          "[.][parallel_for_threads]") {
#ifndef _OPENMP
  FAIL("built without OpenMP: this case needs several threads");
#else
  REQUIRE_FALSE(tadah::mlip::has_parallel_backend());
  const std::size_t n = 40;
  std::vector<int> ran(n, 0);
  std::atomic<int> thrown{0};
  std::atomic<int> team{1};
  std::atomic<bool> waited{false};
  auto body = [&](std::size_t i) {
    ran[i] = 1;
    team.store(omp_get_num_threads());
    if (i == 23 || i == 31) {
      thrown.fetch_add(1);
      throw std::invalid_argument("index " + std::to_string(i));
    }
    if (i == 7) {
      // Wait (at most 20 s) for both higher indices to have thrown.
      const auto t0 = std::chrono::steady_clock::now();
      while (thrown.load() < 2 &&
             std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
        std::this_thread::yield();
      waited.store(thrown.load() == 2);
      throw std::invalid_argument("index 7");
    }
  };
  std::string what;
  try {
    tadah::mlip::parallel_for(n, body);
  } catch (const std::invalid_argument &e) {
    what = e.what();
  }
  INFO("team size " << team.load());
  REQUIRE(team.load() > 1);
  REQUIRE(waited.load());   // index 7 threw after 23 and 31
  CHECK(what == "index 7");
  for (std::size_t i = 0; i < n; ++i) {
    INFO("index " << i);
    CHECK(ran[i] == 1);
  }
#endif
}
