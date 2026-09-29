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

// ---------------------------------------------------------------------------
//  check_bounds_at_corners: the density form of a DM_REAM seed, and the
//  corner context of a vector key
// ---------------------------------------------------------------------------

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/dm/rhoform.h>

namespace {

using tadah::core::Context;

Bound box(const std::string &key, int arg_pos, double lo, double hi) {
  Bound b;
  b.key = key;
  b.arg_pos = arg_pos;
  b.lo = lo;
  b.hi = hi;
  return b;
}

/* A D2_Blip5 + DM_REAM seed in the density form `form` (nullptr: no RHOFORM
 * line) with the TYPEMB basis code `code` (3 = Blip5, 1 = Gaussian). */
Context ream_seed(const char *form, int code = 3) {
  tadah::core::PeriodicTable::initialize();
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip5 4 4 Ta Ta");
  c.add("CGRID2B", "2.5 3.2 3.9 4.5");
  c.add("SGRID2B", "3.0 3.0 3.0 3.0");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", code);
  c.add("TYPEMB", "3 3 3 5 5 0 0 Ta Ta");
  c.add("CGRIDMB", "2.8 3.6 4.4");
  c.add("SGRIDMB", "3.0 3.0 3.0");
  c.add("AMPGRIDMB", "0.5 0.3 0.1");
  if (form) c.add("RHOFORM", form);
  c.add("CEMBFUNC", "-0.2 0.3 0.8 1.3 1.8");
  c.add("SEMBFUNC", "2.0 2.0 2.0 2.0 2.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.5);
  c.add("RCUTMB", 5.5);
  c.add("VERBOSE", 0);
  return c;
}

tadah::mlip::CheckBoundsResult run_corners(tadah::mlip::StructureDB &stdb,
                                           const Context &seed,
                                           const std::vector<Bound> &b) {
  return tadah::mlip::check_bounds_at_corners(stdb, seed, b,
                                              sample_corners(b));
}

}  // namespace

TEST_CASE("check_bounds: a DM_REAM seed must declare RHOFORM",
          "[check_bounds][rhoform]") {
  tadah::mlip::StructureDB stdb;
  const std::vector<Bound> b = {box("AMPGRIDMB", -1, -0.1, 0.5)};
  REQUIRE_THROWS_AS(run_corners(stdb, ream_seed(nullptr), b),
                    tadah::models::DensityFormError);
  REQUIRE_THROWS_WITH(run_corners(stdb, ream_seed(nullptr), b),
                      Catch::Contains("DM_REAM: RHOFORM is required"));
  REQUIRE_NOTHROW(run_corners(stdb, ream_seed("SIGNED"), b));
}

TEST_CASE("check_bounds: the corners below the density form's low bounds "
          "fail under NONNEG and MONOTONE, none under SIGNED",
          "[check_bounds][rhoform]") {
  tadah::mlip::StructureDB stdb;
  // AMPGRIDMB (1) reaches below 0: the two corners at its low bound fail
  const std::vector<Bound> b = {box("AMPGRIDMB", -1, -0.1, 0.5),
                                box("CGRIDMB", 1, 3.5, 3.7)};
  for (const char *form : {"NONNEG", "MONOTONE"}) {
    INFO(form);
    const auto res = run_corners(stdb, ream_seed(form), b);
    CHECK(res.n_corners == 4);
    CHECK(res.n_corners_failing == 2);
    for (std::size_t ci : res.failing_corner_indices)
      CHECK(sample_corners(b)[ci][0] == -0.1);
  }
  const auto s = run_corners(stdb, ream_seed("SIGNED"), b);
  CHECK(s.n_corners == 4);
  CHECK(s.n_corners_failing == 0);
}

TEST_CASE("check_bounds: under NONNEG a Gaussian density's SGRIDMB box needs "
          "lo > 0", "[check_bounds][rhoform]") {
  tadah::mlip::StructureDB stdb;
  const std::vector<Bound> b = {box("SGRIDMB", -1, -0.5, 2.0)};
  const auto g = run_corners(stdb, ream_seed("NONNEG", 1), b);
  CHECK(g.n_corners == 2);
  CHECK(g.n_corners_failing == 1);
  // the Blip5 bumps are even in s: NONNEG sets no width rule for them
  CHECK(run_corners(stdb, ream_seed("NONNEG", 3), b).n_corners_failing == 0);
  CHECK(run_corners(stdb, ream_seed("SIGNED", 1), b).n_corners_failing == 0);
  CHECK(run_corners(stdb, ream_seed("MONOTONE", 3), b).n_corners_failing == 1);
}

TEST_CASE("check_bounds: a corner of a vector key changes only its own entry",
          "[check_bounds]") {
  // The corner context took (arg_pos + 1) copies of the corner value for
  // OPTIM KEY (i), and a single value for KEY (1): CGRIDMB (3) on a
  // four-entry grid audited [c, c, c], so every corner failed the DM_Blip
  // CGRIDMB/SGRIDMB size check. Each corner now keeps the seed's other
  // entries (the context tadah hpo evaluates).
  tadah::core::PeriodicTable::initialize();
  Context seed;
  seed.add("ATOM", "Ta");
  seed.add("INIT2B", "false");
  seed.add("INITMB", "true");
  seed.add("TYPEMB", "DM_Blip 0 4 4 Ta Ta");
  seed.add("CGRIDMB", "3.0 3.5 4.0 4.5");
  seed.add("SGRIDMB", "1.0 1.0 1.0 1.0");
  seed.add("RCTYPEMB", "Cut_Cos");
  seed.add("RCUTMB", 5.0);
  seed.add("VERBOSE", 0);
  tadah::mlip::StructureDB stdb;
  stdb.add("tests_data/structure_1.dat");
  REQUIRE(stdb.size() == 1);

  SECTION("OPTIM CGRIDMB (3), CGRIDMB (1) and two lines on one key") {
    for (const auto &b : std::vector<std::vector<Bound>>{
             {box("CGRIDMB", 2, 3.9, 4.1)},
             {box("CGRIDMB", -1, 2.9, 3.1)},
             {box("CGRIDMB", 1, 3.4, 3.6), box("CGRIDMB", 3, 4.4, 4.6)}}) {
      const auto res = run_corners(stdb, seed, b);
      INFO(b.size() << " bound(s) on CGRIDMB, first arg_pos " << b[0].arg_pos);
      CHECK(res.n_corners == (std::size_t{1} << b.size()));
      CHECK(res.n_corners_failing == 0);
    }
  }
  SECTION("the seed's other entries are kept") {
    // entry 1 far outside the data's r_ij range: every corner of a box on
    // entry 3 keeps it, so every corner fails the DM_Blip CGRIDMB check
    Context far = seed;
    far.remove("CGRIDMB");
    far.add("CGRIDMB", "30.0 3.5 4.0 4.5");
    const std::vector<Bound> b = {box("CGRIDMB", 2, 3.9, 4.1)};
    const auto res = run_corners(stdb, far, b);
    CHECK(res.n_corners == 2);
    CHECK(res.n_corners_failing == 2);
  }
}
