// Catch2 v2 unit tests for the check_bounds dry-run (Phase 7).
//
// Verifies:
//   * parse_hpo_target_bounds extracts (key, lo, hi) from
//     `OPTIM <KEY> (1) lo hi` (scalar) and `OPTIM <KEY> (1-N) lo hi`
//     (vector) lines, and skips comments / blank lines.
//   * sample_corners returns 2^k for k <= 12 with each corner having
//     either lo or hi at every dimension.
//   * sample_corners' Latin-hypercube branch returns ⌈k log₂ k⌉
//     samples for k > 12.
//   * The Phase 7 engine (`tadah analysis check_bounds`) is wired —
//     tested elsewhere via the smoke path; this file focuses on the
//     pure-function library.

#include "catch2/catch.hpp"

#include <tadah/mlip/check_bounds.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>

using tadah::mlip::Bound;
using tadah::mlip::parse_hpo_target_bounds;
using tadah::mlip::sample_corners;

static std::string write_tmp(const std::string& body) {
  std::string path = "/tmp/test_check_bounds_" +
                     std::to_string(std::rand()) + ".tadah";
  std::ofstream f(path);
  f << body;
  f.close();
  return path;
}

TEST_CASE("check_bounds: parse scalar OPTIM lines", "[check_bounds]") {
  const std::string body =
      "# header comment\n"
      "OPTIMIZER\n"
      "  TYPE foo\n"
      "ENDOPTIMIZER\n"
      "\n"
      "OPTIM RCUT2B (1) 3.0 6.0\n"
      "OPTIM CGRIDMB (1) 2.0 4.0\n";
  auto path = write_tmp(body);
  auto bounds = parse_hpo_target_bounds(path);
  std::remove(path.c_str());

  REQUIRE(bounds.size() == 2);
  REQUIRE(bounds[0].key == "RCUT2B");
  REQUIRE(bounds[0].arg_pos == -1);
  REQUIRE(bounds[0].lo == Approx(3.0));
  REQUIRE(bounds[0].hi == Approx(6.0));

  REQUIRE(bounds[1].key == "CGRIDMB");
  REQUIRE(bounds[1].arg_pos == -1);
}

TEST_CASE("check_bounds: parse vector OPTIM lines", "[check_bounds]") {
  const std::string body =
      "OPTIM CGRID2B (1-4) 0.0 6.0\n"
      "OPTIM SGRID2B (1-3) 0.1 5.0\n";
  auto path = write_tmp(body);
  auto bounds = parse_hpo_target_bounds(path);
  std::remove(path.c_str());

  REQUIRE(bounds.size() == 4 + 3);
  // CGRID2B occupies arg_pos 0..3
  for (int i = 0; i < 4; ++i) {
    REQUIRE(bounds[i].key == "CGRID2B");
    REQUIRE(bounds[i].arg_pos == i);
    REQUIRE(bounds[i].lo == Approx(0.0));
    REQUIRE(bounds[i].hi == Approx(6.0));
  }
  // SGRID2B at indices 4..6
  for (int i = 0; i < 3; ++i) {
    REQUIRE(bounds[4 + i].key == "SGRID2B");
    REQUIRE(bounds[4 + i].arg_pos == i);
  }
}

TEST_CASE("check_bounds: parse rejects malformed OPTIM lines", "[check_bounds]") {
  const std::string body = "OPTIM RCUT2B (1)\n";   // missing lo hi
  auto path = write_tmp(body);
  REQUIRE_THROWS_AS(parse_hpo_target_bounds(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("check_bounds: sample_corners full enumeration k<=12", "[check_bounds]") {
  std::vector<Bound> bounds(3);
  for (int i = 0; i < 3; ++i) {
    bounds[i].lo = static_cast<double>(i);
    bounds[i].hi = static_cast<double>(i) + 10.0;
  }
  auto corners = sample_corners(bounds);
  REQUIRE(corners.size() == 8);     // 2^3

  // Every corner has each dimension == lo or hi.
  std::set<std::string> seen;
  for (const auto& c : corners) {
    REQUIRE(c.size() == 3);
    std::string sig;
    for (std::size_t j = 0; j < 3; ++j) {
      const bool is_hi = std::fabs(c[j] - bounds[j].hi) < 1e-12;
      const bool is_lo = std::fabs(c[j] - bounds[j].lo) < 1e-12;
      REQUIRE((is_hi || is_lo));
      sig += is_hi ? "H" : "L";
    }
    seen.insert(sig);
  }
  REQUIRE(seen.size() == 8);   // all distinct
}

TEST_CASE("check_bounds: sample_corners empty returns empty", "[check_bounds]") {
  std::vector<Bound> bounds;
  auto corners = sample_corners(bounds);
  REQUIRE(corners.empty());
}

TEST_CASE("check_bounds: sample_corners LHS branch k=15", "[check_bounds]") {
  std::vector<Bound> bounds(15);
  for (int i = 0; i < 15; ++i) {
    bounds[i].lo = 0.0;
    bounds[i].hi = 1.0;
  }
  auto corners = sample_corners(bounds, /*seed=*/42);
  // ⌈15 · log₂(15)⌉ = ⌈15 × 3.907⌉ = 59
  const auto expected = static_cast<std::size_t>(
      std::ceil(15.0 * std::log2(15.0)));
  REQUIRE(corners.size() == expected);
  // Each corner has 15 entries, all in {0, 1}.
  for (const auto& c : corners) {
    REQUIRE(c.size() == 15);
    for (double v : c) {
      REQUIRE((v == 0.0 || v == 1.0));
    }
  }
}
