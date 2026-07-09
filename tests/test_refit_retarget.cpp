// Unit + round-trip tests for the `tadah refit --retarget` rewriter and the
// lifted hpotarget writer: byte-idempotence against a fresh emission at the
// same percentage, comment-state preservation, per-key overrides, absolute
// --set overrides, clamp repair, and the knot-channel RCUT recentring.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Context;
using namespace tadah::mlip::engines;
namespace fs = std::filesystem;

namespace {

struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

// Small knot-family refit pot Context (mirrors `tadah refit` output shape).
Context make_knot_pot() {
  Context c;
  c.add("ATOM", "Ta");
  for (double v : {5.0, 3.0, 4.2}) c.add("CGRID2B", v);
  c.add("RCUT2B", 5.0);
  c.add("TYPE2B", "D2_Knot5");
  c.add("TYPE2B", 3); c.add("TYPE2B", 0);
  c.add("TYPE2B", "Ta"); c.add("TYPE2B", "Ta");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 2); c.add("TYPEMB", 2); c.add("TYPEMB", 0);
  c.add("TYPEMB", 2); c.add("TYPEMB", 3); c.add("TYPEMB", 0);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta"); c.add("TYPEMB", "Ta");
  for (double v : {4.8, 2.5}) c.add("CGRIDMB", v);
  c.add("RCUTMB", 4.8);
  for (double v : {0.01, -1.2}) c.add("AMPGRIDMB", v);
  for (double v : {0.9, -2.9, 0.1}) c.add("CEMBFUNC", v);
  for (double v : {1.5, -900.0, 12.0, 220.0, -160.0, 5300.0})
    c.add("FIXWEIGHT", v);
  return c;
}

std::string slurp(const std::string &path) {
  std::ifstream is(path);
  return std::string((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
}

std::vector<std::string> read_lines(const std::string &path) {
  std::ifstream is(path);
  std::vector<std::string> lines;
  for (std::string l; std::getline(is, l);) lines.push_back(l);
  return lines;
}

struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_retarget_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScopedTempDir() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "retarget at the emission percentage is byte-idempotent",
                 "[retarget]") {
  ScopedTempDir tmp;
  Context pot = make_knot_pot();
  write_hpotarget("a.hpotarget", pot, 5.0);
  const std::string fresh = slurp("a.hpotarget");
  REQUIRE_FALSE(fresh.empty());

  RetargetOptions opts;  // pct = 5 (the emission default)
  RetargetReport rep = retarget_hpotarget_file("a.hpotarget", "b.hpotarget",
                                               false, opts);
  REQUIRE(rep.n_rewritten > 0);
  REQUIRE(slurp("b.hpotarget") == fresh);

  // and self-idempotence at a different percentage: @10 twice == @10 once
  RetargetOptions o10;
  o10.pct = 10.0;
  retarget_hpotarget_file("a.hpotarget", "c10.hpotarget", false, o10);
  retarget_hpotarget_file("c10.hpotarget", "c10b.hpotarget", false, o10);
  REQUIRE(slurp("c10b.hpotarget") == slurp("c10.hpotarget"));
}

TEST_CASE_METHOD(PTFixture, "retarget rewrites bounds around the centres",
                 "[retarget]") {
  ScopedTempDir tmp;
  write_hpotarget("t.hpotarget", make_knot_pot(), 5.0);

  RetargetOptions opts;
  opts.pct = 10.0;
  opts.pct_per_key["FIXWEIGHT"] = 2.0;
  auto lines = read_lines("t.hpotarget");
  const auto n_before = lines.size();
  RetargetReport rep = retarget_hpotarget_lines(lines, opts);
  REQUIRE(lines.size() == n_before);
  REQUIRE(rep.n_active > 0);      // FIXWEIGHT block
  REQUIRE(rep.n_commented > 0);   // shape blocks

  // FIXWEIGHT (6) centre is 5300 with the per-key 2% -> 5194 .. 5406
  bool checked_fw = false, checked_cg = false, commented_preserved = false;
  for (const auto &l : lines) {
    std::istringstream is(l);
    std::string tok;
    if (l.rfind("OPTIM FIXWEIGHT (6)", 0) == 0) {
      double lo, hi;
      std::string a, b, idx;
      is >> a >> b >> idx >> lo >> hi;
      REQUIRE(lo == Approx(5300.0 * 0.98));
      REQUIRE(hi == Approx(5300.0 * 1.02));
      checked_fw = true;
    }
    if (l.rfind("# OPTIM CGRID2B (1)", 0) == 0) {
      commented_preserved = true;
      std::string h, a, b, idx;
      double lo, hi;
      is >> h >> a >> b >> idx >> lo >> hi;
      REQUIRE(lo == Approx(5.0 * 0.90));  // global 10%
      REQUIRE(hi == Approx(5.0 * 1.10));
      checked_cg = true;
    }
  }
  REQUIRE(checked_fw);
  REQUIRE(checked_cg);
  REQUIRE(commented_preserved);

  // group header comments carry the new percentages
  bool hdr_fw = false, hdr_cg = false;
  for (const auto &l : lines) {
    if (l.rfind("# FIXWEIGHT  (+-2%", 0) == 0) hdr_fw = true;
    if (l.rfind("# CGRID2B  (+-10%", 0) == 0) hdr_cg = true;
  }
  REQUIRE(hdr_fw);
  REQUIRE(hdr_cg);
}

TEST_CASE_METHOD(PTFixture, "retarget --set overrides and errors",
                 "[retarget]") {
  ScopedTempDir tmp;
  write_hpotarget("t.hpotarget", make_knot_pot(), 5.0);
  auto lines = read_lines("t.hpotarget");

  RetargetOptions opts;
  opts.sets.push_back({"FIXWEIGHT", 3, -50.0, -10.0});
  retarget_hpotarget_lines(lines, opts);
  bool found = false;
  for (const auto &l : lines)
    if (l.rfind("OPTIM FIXWEIGHT (3)", 0) == 0) {
      REQUIRE(l.find("-5.00000000e+01 -1.00000000e+01") != std::string::npos);
      found = true;
    }
  REQUIRE(found);

  auto lines2 = read_lines("t.hpotarget");
  RetargetOptions bad;
  bad.sets.push_back({"FIXWEIGHT", 99, 0.0, 1.0});
  REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines2, bad),
                      Catch::Contains("no such OPTIM line"));

  auto lines3 = read_lines("t.hpotarget");
  RetargetOptions inv;
  inv.sets.push_back({"FIXWEIGHT", 1, 2.0, 1.0});
  REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines3, inv),
                      Catch::Contains("hi must be greater"));
}

TEST_CASE_METHOD(PTFixture,
                 "retarget repairs lo-clamped width bounds from the upper bound",
                 "[retarget]") {
  // Synthetic SGRID-style group: the tiny width's lo bound is clamped by
  // the positive-only rule at emission, so its midpoint is NOT the centre.
  std::ostringstream os;
  os << "# Bounds are +-5%\n";
  tadah::core::v_type widths(2);
  widths[0] = 1.2;
  widths[1] = 0.0004;
  emit_optim_group(os, "SGRID2B", widths, 5.0, /*positive_only=*/true);
  std::vector<std::string> lines;
  {
    std::istringstream is(os.str());
    for (std::string l; std::getline(is, l);) lines.push_back(l);
  }

  RetargetOptions opts;  // same pct: a correct repair reproduces the input
  const auto before = lines;
  RetargetReport rep = retarget_hpotarget_lines(lines, opts);
  REQUIRE(lines == before);  // byte-idempotent DESPITE the clamped line
  bool repaired = false;
  for (const auto &w : rep.warnings)
    if (w.find("recovered") != std::string::npos) repaired = true;
  REQUIRE(repaired);
}

TEST_CASE_METHOD(PTFixture,
                 "retarget recentres a stale knot-channel RCUT line",
                 "[retarget]") {
  // Old-style knot hpotargets carry `# OPTIM RCUT2B` centred on the table
  // cutoff; the knot channel's true cutoff is its largest knot (5.0 here).
  std::vector<std::string> lines = {
      "# Bounds are +-5%",
      "# OPTIM RCUT2B (1) 5.30250000e+00 5.86050000e+00",  // centre 5.5815
      "# OPTIM CGRID2B (1) 4.75000000e+00 5.25000000e+00",   // centre 5.0
      "# OPTIM CGRID2B (2) 2.85000000e+00 3.15000000e+00",   // centre 3.0
  };
  RetargetOptions opts;
  RetargetReport rep = retarget_hpotarget_lines(lines, opts);
  bool warned = false;
  for (const auto &w : rep.warnings)
    if (w.find("largest CGRID2B knot") != std::string::npos) warned = true;
  REQUIRE(warned);
  REQUIRE(lines[1].find("4.75000000e+00 5.25000000e+00") != std::string::npos);
  REQUIRE(lines[1].rfind("# OPTIM RCUT2B (1)", 0) == 0);  // stays commented
}

TEST_CASE_METHOD(PTFixture, "retarget file wrapper: in-place and guards",
                 "[retarget]") {
  ScopedTempDir tmp;
  write_hpotarget("t.hpotarget", make_knot_pot(), 5.0);
  const std::string fresh = slurp("t.hpotarget");

  RetargetOptions o10;
  o10.pct = 10.0;
  // out == in without in_place -> refuse
  REQUIRE_THROWS_WITH(
      retarget_hpotarget_file("t.hpotarget", "t.hpotarget", false, o10),
      Catch::Contains("refusing to overwrite"));
  REQUIRE(slurp("t.hpotarget") == fresh);  // untouched

  // in-place works and non-OPTIM lines are preserved byte-for-byte
  retarget_hpotarget_file("t.hpotarget", "", true, o10);
  const auto after = read_lines("t.hpotarget");
  bool has_optimizer_block = false;
  for (const auto &l : after)
    if (l == "OPTIMIZER" || l == "ENDOPTIMIZER") has_optimizer_block = true;
  REQUIRE(has_optimizer_block);
  REQUIRE(slurp("t.hpotarget") != fresh);
}

TEST_CASE_METHOD(PTFixture,
                 "write_hpotarget: knot channels get no RCUT OPTIM line",
                 "[retarget][hpotarget_writer]") {
  ScopedTempDir tmp;
  write_hpotarget("k.hpotarget", make_knot_pot(), 5.0);
  const std::string body = slurp("k.hpotarget");
  REQUIRE(body.find("OPTIM RCUT2B") == std::string::npos);
  REQUIRE(body.find("OPTIM RCUTMB") == std::string::npos);
  REQUIRE(body.find("RCUT2B is DERIVED") != std::string::npos);
  REQUIRE(body.find("OPTIM FIXWEIGHT (6)") != std::string::npos);
}
