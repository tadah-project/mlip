// Unit + round-trip tests for the `tadah refit --retarget` rewriter and the
// lifted hpotarget writer: byte-idempotence against a fresh emission at the
// same percentage, comment-state preservation, per-key overrides, absolute
// --set overrides, clamp repair, and the knot-channel RCUT recentring.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/refit_modes.h>

#include <algorithm>
#include <cmath>
#include <limits>

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

// Small BUMP-family (D2_Blip, w = 2) refit pot with a hard Cut_Dummy cutoff.
// Both blips sit exactly at the support edge (c + 2/eta == RCUT2B == 5), so
// the pot is gate-compliant but every symmetric OPTIM box corner overhangs —
// emitted at full width; `tadah hpo`'s per-candidate gate owns the invariant.
Context make_blip_pot(double c2 = 4.5, double e2 = 4.0) {
  Context c;
  c.add("ATOM", "Cu");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 2); c.add("TYPE2B", 2);
  c.add("TYPE2B", "Cu"); c.add("TYPE2B", "Cu");
  c.add("CGRID2B", 3.0); c.add("CGRID2B", c2);
  c.add("SGRID2B", 1.0); c.add("SGRID2B", e2);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("FIXWEIGHT", 1.5); c.add("FIXWEIGHT", -900.0);
  return c;
}

// max over CGRID2B/SGRID2B OPTIM pairs of (c_hi + w/e_lo) in `lines`.
double bump_corner_reach(const std::vector<std::string> &lines, double w) {
  std::vector<double> chi, elo;
  for (const auto &l : lines) {
    std::istringstream is(l);
    std::string h, a, key, idx;
    double lo, hi;
    if (l.rfind("# OPTIM CGRID2B", 0) == 0 &&
        (is >> h >> a >> key >> idx >> lo >> hi))
      chi.push_back(hi);
    std::istringstream is2(l);
    if (l.rfind("# OPTIM SGRID2B", 0) == 0 &&
        (is2 >> h >> a >> key >> idx >> lo >> hi))
      elo.push_back(lo);
  }
  REQUIRE(chi.size() == elo.size());
  REQUIRE_FALSE(chi.empty());
  double reach = 0.0;
  for (std::size_t i = 0; i < chi.size(); ++i)
    reach = std::max(reach, chi[i] + w / elo[i]);
  return reach;
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
  // knot channels carry no bump-support contract
  REQUIRE(body.find("bump-support contract") == std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "write_hpotarget: bump boxes are emitted at full width",
                 "[retarget][hpotarget_writer][bumpcap]") {
  ScopedTempDir tmp;
  const double w = 2.0, rcut = 5.0;
  const double tol = 1e-6 * rcut;

  SECTION("edge seeds: contract recorded, symmetric boxes may overhang") {
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    const std::string body = slurp("b.hpotarget");
    REQUIRE(body.find("RCUT2B bump-support contract: w = 2") !=
            std::string::npos);
    REQUIRE(body.find("rejected per evaluation") != std::string::npos);
    REQUIRE(body.find("# OPTIM RCUT2B") != std::string::npos);
    // blip 2 (c=4.5, pct 10) keeps its symmetric bounds
    REQUIRE(body.find("# OPTIM CGRID2B (2) 4.05000000e+00 4.95000000e+00") !=
            std::string::npos);
    const auto lines = read_lines("b.hpotarget");
    // edge seeds + full-width boxes: the worst corner overhangs — the
    // per-candidate gate in `tadah hpo`, not the writer, enforces the
    // invariant now
    REQUIRE(bump_corner_reach(lines, w) > rcut + tol);
    // boxes stay alive (lo < hi)
    for (const auto &l : lines) {
      if (l.rfind("# OPTIM CGRID2B", 0) != 0 &&
          l.rfind("# OPTIM SGRID2B", 0) != 0)
        continue;
      std::istringstream is(l);
      std::string h, a, key, idx;
      double lo, hi;
      REQUIRE((is >> h >> a >> key >> idx >> lo >> hi));
      REQUIRE(lo < hi);
    }
  }

  SECTION("a seed well inside rcut also keeps symmetric bounds (no cap)") {
    // c=3.5, eta=2.0 (reach 4.5), pct 20: symmetric hi_c 4.2, e_lo 1.6
    // (corner 5.45 > 5 — previously capped to 3.75, now left at full
    // width for the per-candidate gate).
    Context pot;
    pot.add("ATOM", "Cu");
    pot.add("TYPE2B", "D2_Blip");
    pot.add("TYPE2B", 1); pot.add("TYPE2B", 1);
    pot.add("TYPE2B", "Cu"); pot.add("TYPE2B", "Cu");
    pot.add("CGRID2B", 3.5);
    pot.add("SGRID2B", 2.0);
    pot.add("RCUT2B", 5.0);
    pot.add("RCTYPE2B", "Cut_Dummy");
    pot.add("FIXWEIGHT", 1.0);
    write_hpotarget("c.hpotarget", pot, 20.0);
    const auto lines = read_lines("c.hpotarget");
    bool cg = false, sg = false;
    for (const auto &l : lines) {
      std::istringstream is(l);
      std::string h, a, key, idx;
      double lo, hi;
      if (l.rfind("# OPTIM CGRID2B (1)", 0) == 0) {
        REQUIRE((is >> h >> a >> key >> idx >> lo >> hi));
        REQUIRE(hi == Approx(4.2));    // symmetric, not capped
        REQUIRE(lo == Approx(2.8));
        cg = true;
      }
      if (l.rfind("# OPTIM SGRID2B (1)", 0) == 0) {
        REQUIRE((is >> h >> a >> key >> idx >> lo >> hi));
        REQUIRE(lo == Approx(1.6));    // untouched
        sg = true;
      }
    }
    REQUIRE(cg);
    REQUIRE(sg);
  }

  SECTION("a smooth cutoff disables the contract and the cap") {
    Context pot = make_blip_pot();
    pot.remove("RCTYPE2B");
    pot.add("RCTYPE2B", "Cut_Poly");
    write_hpotarget("s.hpotarget", pot, 10.0);
    const std::string body = slurp("s.hpotarget");
    REQUIRE(body.find("bump-support contract") == std::string::npos);
    // symmetric bounds: blip 2 (c=4.5, pct 10) keeps hi = 4.95
    REQUIRE(body.find("# OPTIM CGRID2B (2) 4.05000000e+00 4.95000000e+00") !=
            std::string::npos);
  }

  SECTION("a pot whose own bump spills is emitted uncapped (warn-only)") {
    write_hpotarget("d.hpotarget", make_blip_pot(4.8, 2.0), 10.0);  // 5.8 > 5
    const std::string body = slurp("d.hpotarget");
    // blip 2 bounds stay symmetric (uncapped): 4.8 +- 0.48
    REQUIRE(body.find("# OPTIM CGRID2B (2) 4.32000000e+00 5.28000000e+00") !=
            std::string::npos);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "retarget: full-width bump boxes and contract diagnostics",
                 "[retarget][bumpcap]") {
  ScopedTempDir tmp;
  const double w = 2.0, rcut = 5.0;
  const double tol = 1e-6 * rcut;

  SECTION("same-pct retarget of a fresh file is byte-idempotent") {
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    const std::string fresh = slurp("b.hpotarget");
    RetargetOptions o10;
    o10.pct = 10.0;
    retarget_hpotarget_file("b.hpotarget", "b2.hpotarget", false, o10);
    REQUIRE(slurp("b2.hpotarget") == fresh);
  }

  SECTION("re-pct retarget keeps full-width boxes and stays a fixed point") {
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    RetargetOptions o25;
    o25.pct = 25.0;
    retarget_hpotarget_file("b.hpotarget", "b25.hpotarget", false, o25);
    // edge seeds at +-25%: the symmetric worst corner overhangs — the
    // per-candidate gate in `tadah hpo` owns the invariant now
    REQUIRE(bump_corner_reach(read_lines("b25.hpotarget"), w) > rcut + tol);
    retarget_hpotarget_file("b25.hpotarget", "b25b.hpotarget", false, o25);
    REQUIRE(slurp("b25b.hpotarget") == slurp("b25.hpotarget"));
  }

  SECTION("--set with an infeasible corner wins but warns") {
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    auto lines = read_lines("b.hpotarget");
    RetargetOptions opts;
    opts.pct = 10.0;
    opts.sets.push_back({"SGRID2B", 2, 1.0, 4.4});  // e_lo 1.0 -> reach 6.5
    RetargetReport rep = retarget_hpotarget_lines(lines, opts);
    bool warned = false, applied = false;
    for (const auto &wmsg : rep.warnings)
      if (wmsg.find("bump-support contract") != std::string::npos)
        warned = true;
    for (const auto &l : lines)
      if (l.rfind("# OPTIM SGRID2B (2)", 0) == 0 &&
          l.find("1.00000000e+00") != std::string::npos)
        applied = true;
    REQUIRE(warned);
    REQUIRE(applied);
  }

  SECTION("an ACTIVE RCUT line draws the wasted-dimension warning") {
    // A hand-added ACTIVE `OPTIM RCUT2B` next to the emitted commented one
    // is usually a wasted search dimension on a bump channel (feasible
    // candidates vanish inside the cutoff regardless of its value): warn,
    // and leave the CGRID/SGRID boxes at full symmetric width.
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    auto lines = read_lines("b.hpotarget");
    for (std::size_t i = 0; i < lines.size(); ++i) {
      if (lines[i].rfind("# OPTIM RCUT2B (1)", 0) == 0) {
        lines.insert(lines.begin() + i + 1, "OPTIM RCUT2B (1) 6.0 6.2");
        break;
      }
    }
    RetargetOptions o10;
    o10.pct = 10.0;
    RetargetReport rep = retarget_hpotarget_lines(lines, o10);
    bool warned = false;
    for (const auto &wmsg : rep.warnings)
      if (wmsg.find("wasted dimension") != std::string::npos)
        warned = true;
    REQUIRE(warned);
    // full-width edge-seed boxes: the worst corner overhangs (uncapped)
    REQUIRE(bump_corner_reach(lines, w) > rcut + tol);
  }

  SECTION("--rcut2b declares the cutoff in one step") {
    // The RCUT2B reference is recentred on 6.0 and a stray ACTIVE line is
    // demoted to a comment. The CGRID/SGRID boxes are symmetric full-width
    // either way (nothing is capped any more).
    write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
    auto lines = read_lines("b.hpotarget");
    for (std::size_t i = 0; i < lines.size(); ++i) {
      if (lines[i].rfind("# OPTIM RCUT2B (1)", 0) == 0) {
        lines.insert(lines.begin() + i + 1, "OPTIM RCUT2B (1) 5.9 6.1");
        break;
      }
    }
    RetargetOptions o;
    o.pct = 10.0;
    o.rcut_override["2B"] = 6.0;
    RetargetReport rep = retarget_hpotarget_lines(lines, o);
    bool demote_warn = false;
    for (const auto &wmsg : rep.warnings)
      if (wmsg.find("commented out") != std::string::npos) demote_warn = true;
    REQUIRE(demote_warn);
    bool recentred = false;
    for (const auto &l : lines) {
      REQUIRE(l.rfind("OPTIM RCUT2B", 0) != 0);  // no ACTIVE RCUT line left
      if (l.rfind("# OPTIM RCUT2B (1) 5.40000000e+00 6.60000000e+00", 0) == 0)
        recentred = true;  // 6.0 +- 10%
    }
    REQUIRE(recentred);
    // symmetric full-width boxes: the edge-seed corner overhangs the OLD
    // 5.0 cutoff (nothing truncates against it any more)
    REQUIRE(bump_corner_reach(lines, w) > rcut + tol);
  }

  SECTION("inherited seed spill: uncapped rewrite + spill warning") {
    write_hpotarget("d.hpotarget", make_blip_pot(4.8, 2.0), 10.0);
    auto lines = read_lines("d.hpotarget");
    RetargetOptions o5;
    o5.pct = 5.0;
    RetargetReport rep = retarget_hpotarget_lines(lines, o5);
    bool warned = false;
    for (const auto &wmsg : rep.warnings)
      if (wmsg.find("spill past") != std::string::npos) warned = true;
    REQUIRE(warned);
    // blip 2 rewritten symmetric at 5%: 4.8 +- 0.24, NOT capped
    bool uncapped = false;
    for (const auto &l : lines)
      if (l.rfind("# OPTIM CGRID2B (2)", 0) == 0 &&
          l.find("5.04000000e+00") != std::string::npos)
        uncapped = true;
    REQUIRE(uncapped);
  }
}

// ---------------------------------------------------------------------------
//  RHOFORM (FEAT-02): non-negative AMPGRIDMB boxes under NONNEG / MONOTONE
// ---------------------------------------------------------------------------
namespace {
// A DM_REAM Blip5 refit pot (INITMB true) declared in `form`: amplitudes
// include an exact 0 (an NNLS active constraint) and small values whose
// symmetric box would reach below 0.
Context make_ream_blip5_pot(const std::string &form) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip5");
  c.add("TYPE2B", 2); c.add("TYPE2B", 2);
  c.add("TYPE2B", "Ta"); c.add("TYPE2B", "Ta");
  c.add("CGRID2B", 2.5); c.add("CGRID2B", 3.5);
  c.add("SGRID2B", 2.0); c.add("SGRID2B", 2.0);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("TYPEMB", "DM_REAM");
  for (int v : {3, 5, 5, 5, 2, 2, 0, 0}) c.add("TYPEMB", v);
  c.add("TYPEMB", "Ta"); c.add("TYPEMB", "Ta");
  for (double v : {1.5, 2.3, 3.1, 3.9, 4.4}) c.add("CGRIDMB", v);
  for (double v : {2.0, 2.0, 2.0, 2.0, 5.0}) c.add("SGRIDMB", v);
  for (double v : {0.06, 0.0, 2.5e-5, 0.012, 3.0e-7}) c.add("AMPGRIDMB", v);
  c.add("RHOFORM", form);
  c.add("RCUTMB", 5.0);
  c.add("RCTYPEMB", "Cut_Dummy");
  c.add("CEMBFUNC", 0.5); c.add("CEMBFUNC", 1.5);
  c.add("SEMBFUNC", 1.0); c.add("SEMBFUNC", 1.0);
  for (double v : {1.5, -9.0, 0.2, -0.7}) c.add("FIXWEIGHT", v);
  return c;
}

std::vector<std::pair<double, double>> amp_bounds(
    const std::vector<std::string> &lines) {
  std::vector<std::pair<double, double>> out;
  for (const auto &l : lines) {
    const auto p = l.find("OPTIM AMPGRIDMB (");
    if (p == std::string::npos) continue;
    std::istringstream ls(l.substr(l.find(')', p) + 1));
    double lo, hi;
    ls >> lo >> hi;
    out.emplace_back(lo, hi);
  }
  return out;
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "hpotarget under RHOFORM NONNEG / MONOTONE: non-negative "
                 "AMPGRIDMB boxes, retarget fixed point",
                 "[retarget][rhoform]") {
  for (const char *form : {"NONNEG", "MONOTONE"}) {
    INFO(form);
    ScopedTempDir tmp;
    write_hpotarget("n.hpotarget", make_ream_blip5_pot(form), 5.0);
    const std::string fresh = slurp("n.hpotarget");
    const auto lines = read_lines("n.hpotarget");
    // the parsed header line
    REQUIRE(fresh.find(rhoform_bounds_line(tadah::models::parse_rhoform(form)) +
                       "\n") != std::string::npos);
    const auto b = amp_bounds(lines);
    REQUIRE(b.size() == 5);
    for (const auto &x : b) REQUIRE(x.first >= 0.0);
    REQUIRE(b[1].first == 0.0);   // amplitude 0: [0, group floor]
    REQUIRE(b[1].second > 0.0);
    REQUIRE(b[2].first == 0.0);   // 2.5e-5 - floor(6e-5) < 0: clamped
    REQUIRE(b[4].first == 0.0);
    REQUIRE(b[0].first == Approx(0.06 * 0.95));  // unclamped: v - 5%
    // the positive-only width groups are unchanged
    REQUIRE(fresh.find("OPTIM SGRIDMB (1) 1.90000000e+00 2.10000000e+00") !=
            std::string::npos);

    // same percentage: byte-identical (the clamped centres come back from hi)
    RetargetOptions o5;
    const RetargetReport rep =
        retarget_hpotarget_file("n.hpotarget", "r5.hpotarget", false, o5);
    REQUIRE(rep.rhoform == tadah::models::rhoform_name(
                               tadah::models::parse_rhoform(form)));
    REQUIRE(slurp("r5.hpotarget") == fresh);
    // via another percentage and back: still the fresh file
    RetargetOptions o10;
    o10.pct = 10.0;
    retarget_hpotarget_file("n.hpotarget", "r10.hpotarget", false, o10);
    retarget_hpotarget_file("r10.hpotarget", "r10b.hpotarget", false, o10);
    REQUIRE(slurp("r10b.hpotarget") == slurp("r10.hpotarget"));
    for (const auto &x : amp_bounds(read_lines("r10.hpotarget")))
      REQUIRE(x.first >= 0.0);
    retarget_hpotarget_file("r10.hpotarget", "back5.hpotarget", false, o5);
    REQUIRE(slurp("back5.hpotarget") == fresh);
    // a wide percentage clamps every box and still returns
    RetargetOptions o80;
    o80.pct = 80.0;
    retarget_hpotarget_file("n.hpotarget", "r80.hpotarget", false, o80);
    for (const auto &x : amp_bounds(read_lines("r80.hpotarget")))
      REQUIRE(x.first >= 0.0);
    retarget_hpotarget_file("r80.hpotarget", "back5b.hpotarget", false, o5);
    REQUIRE(slurp("back5b.hpotarget") == fresh);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "hpotarget under RHOFORM NONNEG: per-key and above-100% "
                 "retargets keep the fitted amplitudes",
                 "[retarget][rhoform]") {
  const std::vector<double> amps{0.06, 0.0, 2.5e-5, 0.012, 3.0e-7};
  auto contains_amps = [&amps](const std::string &file) {
    const auto b = amp_bounds(read_lines(file));
    REQUIRE(b.size() == amps.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
      INFO(file << " AMPGRIDMB(" << i + 1 << ") [" << b[i].first << ", "
                << b[i].second << "] vs " << amps[i]);
      CHECK(b[i].first >= 0.0);
      CHECK(b[i].first <= amps[i]);
      CHECK(amps[i] <= b[i].second);
    }
  };
  ScopedTempDir tmp;
  write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
  const std::string fresh = slurp("n.hpotarget");
  RetargetOptions o5;
  SECTION("a per-key percentage: the group comment's percentage drives the "
          "repair") {
    for (double k : {50.0, 150.0}) {
      INFO("AMPGRIDMB=" << k);
      RetargetOptions ok;
      ok.pct_per_key["AMPGRIDMB"] = k;
      retarget_hpotarget_file("n.hpotarget", "k1.hpotarget", false, ok);
      // the same command again: a fixed point, with no repair warnings
      const RetargetReport rep =
          retarget_hpotarget_file("k1.hpotarget", "k2.hpotarget", false, ok);
      REQUIRE(slurp("k2.hpotarget") == slurp("k1.hpotarget"));
      for (const auto &w : rep.warnings)
        CHECK(w.find("AMPGRIDMB") == std::string::npos);
      contains_amps("k1.hpotarget");
      retarget_hpotarget_file("k2.hpotarget", "k5.hpotarget", false, o5);
      REQUIRE(slurp("k5.hpotarget") == fresh);
    }
  }
  SECTION("above 100%: every box clamped, the centres still come back") {
    RetargetOptions o150;
    o150.pct = 150.0;
    retarget_hpotarget_file("n.hpotarget", "r150.hpotarget", false, o150);
    for (const auto &x : amp_bounds(read_lines("r150.hpotarget")))
      REQUIRE(x.first == 0.0);
    contains_amps("r150.hpotarget");
    const RetargetReport rep = retarget_hpotarget_file(
        "r150.hpotarget", "r150b.hpotarget", false, o150);
    REQUIRE(slurp("r150b.hpotarget") == slurp("r150.hpotarget"));
    for (const auto &w : rep.warnings)
      CHECK(w.find("AMPGRIDMB") == std::string::npos);
    retarget_hpotarget_file("r150.hpotarget", "back5.hpotarget", false, o5);
    REQUIRE(slurp("back5.hpotarget") == fresh);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "hpotarget RHOFORM: SIGNED keeps the symmetric boxes, the "
                 "form header is enforced",
                 "[retarget][rhoform]") {
  ScopedTempDir tmp;
  SECTION("SIGNED: no header line, the symmetric rule (lo may be < 0)") {
    write_hpotarget("s.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
    const std::string body = slurp("s.hpotarget");
    REQUIRE(body.find("# RHOFORM") == std::string::npos);
    const auto b = amp_bounds(read_lines("s.hpotarget"));
    REQUIRE(b.size() == 5);
    REQUIRE(b[1].first < 0.0);
    RetargetOptions o5;
    const RetargetReport rep =
        retarget_hpotarget_file("s.hpotarget", "r.hpotarget", false, o5);
    REQUIRE(rep.rhoform.empty());
    REQUIRE(slurp("r.hpotarget") == body);
  }
  SECTION("--set: AMPGRIDMB lo < 0 is refused under NONNEG only") {
    write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
    RetargetOptions bad;
    bad.sets.push_back({"AMPGRIDMB", 2, -0.1, 0.2});
    auto lines = read_lines("n.hpotarget");
    REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines, bad),
                        Catch::Contains("--set AMPGRIDMB(2): lo must be >= 0: "
                                        "the file declares RHOFORM NONNEG"));
    RetargetOptions ok;
    ok.sets.push_back({"AMPGRIDMB", 2, 0.0, 0.2});
    auto lines2 = read_lines("n.hpotarget");
    REQUIRE_NOTHROW(retarget_hpotarget_lines(lines2, ok));
    write_hpotarget("s.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
    auto lines3 = read_lines("s.hpotarget");
    REQUIRE_NOTHROW(retarget_hpotarget_lines(lines3, bad));
  }
  SECTION("a RHOFORM header naming no density form is refused") {
    write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
    auto lines = read_lines("n.hpotarget");
    for (auto &l : lines)
      if (l.rfind("# RHOFORM NONNEG:", 0) == 0)
        l = "# RHOFORM POSITIVE: AMPGRIDMB bounds are non-negative";
    RetargetOptions o5;
    REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines, o5),
                        Catch::Contains("names no density form"));
  }
  SECTION("a DM_REAM pot without RHOFORM is refused by the writer") {
    Context pot = make_ream_blip5_pot("SIGNED");
    pot.remove("RHOFORM");
    REQUIRE_THROWS_AS(write_hpotarget("k.hpotarget", pot, 5.0),
                      tadah::models::DensityFormError);
    REQUIRE_THROWS_WITH(write_hpotarget("k.hpotarget", pot, 5.0),
                        Catch::Contains("RHOFORM is required"));
  }
}

// ---------------------------------------------------------------------------
//  Per-key percentages on every line, the --pct / --set checks
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(PTFixture,
                 "retarget: a per-key percentage is a fixed point for a width "
                 "group and for an RCUT scalar line",
                 "[retarget][pct]") {
  ScopedTempDir tmp;
  write_hpotarget("a.hpotarget", make_blip_pot(), 5.0);
  const std::string fresh = slurp("a.hpotarget");
  RetargetOptions o5;
  for (const char *key : {"SGRID2B", "RCUT2B"}) {
    INFO(key << "=50");
    RetargetOptions o;
    o.pct_per_key[key] = 50.0;
    retarget_hpotarget_file("a.hpotarget", "b.hpotarget", false, o);
    // the same command again: byte-identical, no repair warning
    const RetargetReport rep =
        retarget_hpotarget_file("b.hpotarget", "c.hpotarget", false, o);
    REQUIRE(slurp("c.hpotarget") == slurp("b.hpotarget"));
    for (const auto &w : rep.warnings)
      CHECK(w.find("inconsistent") == std::string::npos);
    // back to the emission percentage: the fresh file again
    retarget_hpotarget_file("c.hpotarget", "d.hpotarget", false, o5);
    REQUIRE(slurp("d.hpotarget") == fresh);
  }
  // RCUT2B = 5 at +-50%: [2.5, 7.5]
  RetargetOptions o;
  o.pct_per_key["RCUT2B"] = 50.0;
  retarget_hpotarget_file("a.hpotarget", "r.hpotarget", false, o);
  CHECK(slurp("r.hpotarget").find(
            "# OPTIM RCUT2B (1) 2.50000000e+00 7.50000000e+00") !=
        std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "retarget: --pct KEY=N needs an OPTIM line on KEY",
                 "[retarget][pct]") {
  ScopedTempDir tmp;
  write_hpotarget("a.hpotarget", make_blip_pot(), 5.0);
  auto lines = read_lines("a.hpotarget");
  RetargetOptions o;
  o.pct_per_key["AMPGRIDMB"] = 10.0;  // the pair-only pot has none
  REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines, o),
                      Catch::Contains("--pct AMPGRIDMB=10: the file has no "
                                      "OPTIM AMPGRIDMB line") &&
                          Catch::Contains("SGRID2B"));
  // a commented line counts (it is rewritten)
  RetargetOptions ok;
  ok.pct_per_key["CGRID2B"] = 10.0;
  auto lines2 = read_lines("a.hpotarget");
  REQUIRE_NOTHROW(retarget_hpotarget_lines(lines2, ok));
}

TEST_CASE_METHOD(PTFixture,
                 "retarget: --set bounds must be finite, lo < hi, and lo "
                 "within the density form's rule",
                 "[retarget][set]") {
  ScopedTempDir tmp;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
  write_hpotarget("s.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
  auto refused = [](const std::string &file, RetargetOptions::SetOverride s,
                    const std::string &what) {
    RetargetOptions o;
    o.sets.push_back(s);
    auto lines = read_lines(file);
    INFO(file << " --set " << s.key << "(" << s.index << ")=" << s.lo << ","
              << s.hi);
    CHECK_THROWS_WITH(retarget_hpotarget_lines(lines, o),
                      Catch::Contains(what));
  };
  // NaN passed the lo < 0 test of a non-negative AMPGRIDMB box
  refused("n.hpotarget", {"AMPGRIDMB", 2, nan, 0.2},
          "--set AMPGRIDMB(2): lo must be a finite number");
  refused("n.hpotarget", {"AMPGRIDMB", 2, -0.1, 0.2},
          "--set AMPGRIDMB(2): lo must be >= 0");
  refused("s.hpotarget", {"CGRIDMB", 1, 1.4, nan},
          "--set CGRIDMB(1): hi must be a finite number");
  refused("s.hpotarget", {"CGRIDMB", 1, -inf, 1.6},
          "--set CGRIDMB(1): lo must be a finite number");
  refused("s.hpotarget", {"CGRIDMB", 1, 1.6, 1.4},
          "--set CGRIDMB(1): hi must be greater than lo");
  // the file's form sets the rule: SIGNED takes a negative amplitude
  RetargetOptions o;
  o.sets.push_back({"AMPGRIDMB", 2, -0.1, 0.2});
  auto lines = read_lines("s.hpotarget");
  REQUIRE_NOTHROW(retarget_hpotarget_lines(lines, o));
}

TEST_CASE_METHOD(PTFixture,
                 "retarget: a RHOFORM header naming no density form is an "
                 "error at its line",
                 "[retarget][rhoform]") {
  ScopedTempDir tmp;
  write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
  auto lines = read_lines("n.hpotarget");
  std::size_t at = 0;
  for (std::size_t i = 0; i < lines.size(); ++i)
    if (lines[i].rfind("# RHOFORM NONNEG:", 0) == 0) {
      lines[i] = "# RHOFORM POSITIVE: AMPGRIDMB bounds are non-negative";
      at = i + 1;
    }
  REQUIRE(at > 0);
  RetargetOptions o5;
  REQUIRE_THROWS_WITH(retarget_hpotarget_lines(lines, o5),
                      Catch::Contains("line " + std::to_string(at) + ":") &&
                          Catch::Contains("names no density form"));
}

TEST_CASE_METHOD(PTFixture,
                 "refit --retarget: --pct and --set numbers use the "
                 "hpotarget number grammar",
                 "[retarget][strict]") {
  ScopedTempDir tmp;
  write_hpotarget("a.hpotarget", make_blip_pot(), 5.0);
  auto retarget = [](const char *key, const std::string &tok) {
    Context c;
    c.add("RETARGET", std::string("a.hpotarget"));
    c.add("OUTFILE", std::string("o.hpotarget"));
    c.add(key, tok);
    return tadah::mlip::engines::run_refit_retarget(c);
  };
  // std::stod read a prefix (5x), hexadecimal (0x10), inf, nan and a
  // subnormal (1e-320); the hpotarget grammar refuses them all
  for (const std::string bad : {"5x", "0x10", "inf", "nan", "-inf", "1e-320",
                                "1e400", ""}) {
    INFO("'" << bad << "'");
    if (!bad.empty())  // a Context holds no empty token
      CHECK_THROWS_WITH(retarget("PCT", bad),
                        Catch::Contains("--pct token '" + bad + "'"));
    CHECK_THROWS_WITH(retarget("PCT", "SGRID2B=" + bad),
                      Catch::Contains("--pct token 'SGRID2B=" + bad + "'"));
    CHECK_THROWS_WITH(retarget("SET", "CGRID2B(1)=" + bad + ",3.1"),
                      Catch::Contains("--set token 'CGRID2B(1)=" + bad +
                                      ",3.1'"));
    CHECK_THROWS_WITH(retarget("SET", "CGRID2B(1)=2.9," + bad),
                      Catch::Contains("--set token 'CGRID2B(1)=2.9," + bad +
                                      "'"));
  }
  REQUIRE_FALSE(fs::exists("o.hpotarget"));
  // the grammar's own forms
  for (const std::string ok : {"10", "+7.5", "5.", ".5", "2.5e1"}) {
    INFO("'" << ok << "'");
    CHECK(retarget("PCT", ok) == EXIT_SUCCESS);
    CHECK(retarget("PCT", "SGRID2B=" + ok) == EXIT_SUCCESS);
  }
  CHECK(retarget("SET", "CGRID2B(1)=2.9,3.1") == EXIT_SUCCESS);
  // the refusals give their examples without "e.g." (before the fix:
  // "...; e.g. --pct 10 FIXWEIGHT=2)." and "(e.g. --set ...)")
  CHECK_THROWS_WITH(
      retarget("PCT", "5x"),
      "tadah refit --retarget: cannot parse --pct token '5x' — expected a "
      "positive number or KEY=<positive number> in the number grammar of "
      "the hpotarget file ([+-]digits[.digits][e[+-]digits]; such as --pct "
      "10 FIXWEIGHT=2).");
  CHECK_THROWS_WITH(
      retarget("SET", "CGRID2B=2.9,3.1"),
      "tadah refit --retarget: cannot parse --set token 'CGRID2B=2.9,3.1' — "
      "expected 'KEY(i)=lo,hi', KEY an OPTIM key or component path of the "
      "file (such as --set 'CGRID2B(3)=2.1,2.4' or "
      "'EAM.DENSITY.CENTRES(2)=2.4,2.6').");
  // zero and negative percentages stay refused
  CHECK_THROWS_WITH(retarget("PCT", "0"), Catch::Contains("--pct token '0'"));
  CHECK_THROWS_WITH(retarget("PCT", "SGRID2B=-5"),
                    Catch::Contains("--pct token 'SGRID2B=-5'"));
}

TEST_CASE_METHOD(PTFixture,
                 "refit --retarget: the configuration's OPTIM_PCT is a "
                 "positive number",
                 "[retarget][strict]") {
  /* Pins: the OPTIM_PCT of a --retarget configuration is read as the refit
   * reads it (read_optim_pct: the hpotarget number grammar, > 0), so 0 and
   * -5 are refused naming the token, as --pct refuses them, and 10 runs.
   * Before the fix: Context::get<double> read it unchecked, so OPTIM_PCT -5
   * rewrote the bounds at -5 % (no exception). */
  ScopedTempDir tmp;
  write_hpotarget("a.hpotarget", make_blip_pot(), 5.0);
  auto retarget = [](const std::string &tok) {
    Context c;
    c.add("RETARGET", std::string("a.hpotarget"));
    c.add("OUTFILE", std::string("o.hpotarget"));
    c.add("OPTIM_PCT", tok);
    return tadah::mlip::engines::run_refit_retarget(c);
  };
  for (const std::string bad : {"-5", "0"}) {
    INFO("OPTIM_PCT " << bad);
    CHECK_THROWS_WITH(retarget(bad), "tadah refit: OPTIM_PCT must be a "
                                     "positive number, got '" + bad + "'");
  }
  REQUIRE_FALSE(fs::exists("o.hpotarget"));
  CHECK(retarget("10") == EXIT_SUCCESS);
}

// ---------------------------------------------------------------------------
//  refit --retarget --rhoform
// ---------------------------------------------------------------------------
namespace {
int retarget_with(const std::string &in, const std::string &out,
                  const char *rhoform) {
  Context c;
  c.add("RETARGET", in);
  c.add("OUTFILE", out);
  if (rhoform) c.add("RETARGET_RHOFORM", std::string(rhoform));
  return tadah::mlip::engines::run_refit_retarget(c);
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "refit --retarget --rhoform NONNEG|MONOTONE: non-negative "
                 "AMPGRIDMB boxes and the parsed RHOFORM header",
                 "[retarget][rhoform][rhoform_option]") {
  for (const char *form : {"NONNEG", "MONOTONE"}) {
    INFO(form);
    ScopedTempDir tmp;
    write_hpotarget("s.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
    // an example: AMPGRIDMB (3) = 2.5e-5 with the group
    // floor 6e-5, a SIGNED box reaching below 0
    REQUIRE(slurp("s.hpotarget").find(
                "# OPTIM AMPGRIDMB (3) -3.50000000e-05 8.50000000e-05\n") !=
            std::string::npos);
    REQUIRE(retarget_with("s.hpotarget", "r.hpotarget", form) ==
            EXIT_SUCCESS);
    const std::string body = slurp("r.hpotarget");
    CHECK(body.find("# OPTIM AMPGRIDMB (3) 0.00000000e+00 8.50000000e-05\n") !=
          std::string::npos);
    CHECK(body.find(rhoform_bounds_line(tadah::models::parse_rhoform(form)) +
                    "\n") != std::string::npos);
    for (const auto &x : amp_bounds(read_lines("r.hpotarget")))
      CHECK(x.first >= 0.0);
    // byte for byte the file the writer emits for a potential in that form
    write_hpotarget("f.hpotarget", make_ream_blip5_pot(form), 5.0);
    CHECK(body == slurp("f.hpotarget"));
    // the header now declares the form: the same --rhoform again, or none,
    // is a fixed point
    REQUIRE(retarget_with("r.hpotarget", "r2.hpotarget", form) ==
            EXIT_SUCCESS);
    CHECK(slurp("r2.hpotarget") == body);
    REQUIRE(retarget_with("r.hpotarget", "r3.hpotarget", nullptr) ==
            EXIT_SUCCESS);
    CHECK(slurp("r3.hpotarget") == body);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "refit --retarget --rhoform: a header naming another form is "
                 "an error; SIGNED adds no header",
                 "[retarget][rhoform][rhoform_option]") {
  ScopedTempDir tmp;
  write_hpotarget("n.hpotarget", make_ream_blip5_pot("NONNEG"), 5.0);
  std::size_t at = 0;
  {
    const auto lines = read_lines("n.hpotarget");
    for (std::size_t i = 0; i < lines.size(); ++i)
      if (lines[i].rfind("# RHOFORM NONNEG:", 0) == 0) at = i + 1;
  }
  REQUIRE(at > 0);
  for (const char *other : {"MONOTONE", "SIGNED", "signed"}) {
    INFO(other);
    CHECK_THROWS_WITH(
        retarget_with("n.hpotarget", "o.hpotarget", other),
        Catch::Contains("--rhoform") &&
            Catch::Contains("line " + std::to_string(at) +
                            " declares RHOFORM NONNEG"));
  }
  REQUIRE_FALSE(fs::exists("o.hpotarget"));
  // an unknown form is refused
  CHECK_THROWS_WITH(retarget_with("n.hpotarget", "o.hpotarget", "SQUARE"),
                    Catch::Contains("SQUARE"));
  // SIGNED over a SIGNED file (no header): the plain rewrite, no header
  write_hpotarget("s.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
  REQUIRE(retarget_with("s.hpotarget", "o.hpotarget", "SIGNED") ==
          EXIT_SUCCESS);
  CHECK(slurp("o.hpotarget") == slurp("s.hpotarget"));
  CHECK(slurp("o.hpotarget").find("# RHOFORM") == std::string::npos);
}

/* Pins: the hpotarget of a DM_REAM potential under EMBF0 ZERO (the
 * make_ream_blip5_pot("SIGNED") potential, which has no EMBF0 line and so
 * reads as ZERO) carries the note, as consecutive lines,
 *   "# EMBF0 ZERO: F(0) = 0 exactly, so with BIAS false the model has no"
 *   "# per-atom energy constant. For absolute reference energies, such as"
 *   "# DFT totals, tadah hpo takes --eshift-atom or --eshift-dbatom (an"
 *   "# ESHIFT_ATOM or ESHIFT_DBATOM line of <config>, a potential, is not"
 *   "# read), or an ESHIFT line in <config>."
 *   "# (Not BIAS true: its intercept columns, one per ATOM species,"
 *   "# precede the weights that FIXINDEX/FIXWEIGHT and the OPTIM"
 *   "# FIXWEIGHT lines below index.)"
 * and no line advising to add ESHIFT_ATOM or ESHIFT_DBATOM to <config>:
 * <config> is the refit potential plus a DBFILE line, a file with WEIGHTS,
 * whose toggle lines tadah hpo does not read.
 * The file begins with the lines
 *   "# Starter target file of tadah hpo, written by tadah refit. It runs
 *   with:"
 *   "#   tadah hpo -c <config> --hpotarget <this file> --validation <val>"
 * and the first note ends "# validation data the loss is measured on (the
 * Nested Fitting chapter of the documentation)."; no line has "e.g.".
 * Before the fix: the note said "# DFT totals) add ESHIFT_ATOM,
 * ESHIFT_DBATOM or ESHIFT to <config>.", advice that applies no shift (the
 * toggle line is not read, with a WARNING saying so): the sequence is not
 * found. Before the fix of the voice: "(e.g.\n# DFT totals) give tadah hpo
 * ...", "... or add an ESHIFT line to <config>.", a file that began "# HPO
 * starter target written by `tadah refit`. Run:" and "See the Tadah docs
 * (HPO).". */
TEST_CASE_METHOD(PTFixture,
                 "write_hpotarget: the EMBF0 ZERO note says how tadah hpo "
                 "derives a per-atom reference energy",
                 "[retarget][embf0]") {
  ScopedTempDir tmp;
  write_hpotarget("z.hpotarget", make_ream_blip5_pot("SIGNED"), 5.0);
  const std::vector<std::string> lines = read_lines("z.hpotarget");
  const std::vector<std::string> note = {
      "# EMBF0 ZERO: F(0) = 0 exactly, so with BIAS false the model has no",
      "# per-atom energy constant. For absolute reference energies, such as",
      "# DFT totals, tadah hpo takes --eshift-atom or --eshift-dbatom (an",
      "# ESHIFT_ATOM or ESHIFT_DBATOM line of <config>, a potential, is not",
      "# read), or an ESHIFT line in <config>.",
      "# (Not BIAS true: its intercept columns, one per ATOM species,",
      "# precede the weights that FIXINDEX/FIXWEIGHT and the OPTIM",
      "# FIXWEIGHT lines below index.)"};
  INFO("z.hpotarget:\n" << slurp("z.hpotarget"));
  const auto first = std::find(lines.begin(), lines.end(), note[0]);
  REQUIRE(first != lines.end());
  REQUIRE(static_cast<std::size_t>(lines.end() - first) >= note.size());
  CHECK(std::vector<std::string>(first, first + note.size()) == note);
  CHECK(slurp("z.hpotarget").find("add ESHIFT_ATOM") == std::string::npos);
  const std::vector<std::string> head = {
      "# Starter target file of tadah hpo, written by tadah refit. It runs "
      "with:",
      "#   tadah hpo -c <config> --hpotarget <this file> --validation <val>",
      "# <config>: the refit potential with a DBFILE line appended naming a",
      "# training dataset; <val>: a file with DBFILE line(s) naming the",
      "# validation data the loss is measured on (the Nested Fitting",
      "# chapter of the documentation)."};
  REQUIRE(lines.size() >= head.size());
  CHECK(std::vector<std::string>(lines.begin(), lines.begin() + head.size()) ==
        head);
  CHECK(slurp("z.hpotarget").find("e.g.") == std::string::npos);
}

/* Pins: the warnings of --rcut2b say what the run does, without an
 * instruction:
 *  - on a knot channel (CGRID2B, no SGRID2B): "--rcut2b is ignored:
 *    CGRID2B is a knot channel, and RCUT2B is its largest knot, which the
 *    CGRID2B boxes move.";
 *  - on a bump channel whose file has no RCUT2B reference line: "RCUT2B:
 *    no OPTIM reference line in the file — the declared cutoff
 *    6.00000000e+00 still binds the bump-support cap; the fit takes RCUT2B
 *    from the training configuration, which --rcut2b does not change.";
 *  - on an ACTIVE RCUT2B line: "OPTIM RCUT2B (1): ACTIVE line commented out
 *    — ... (the bump-support gate keeps every bump inside the cutoff's LOW
 *    bound); the fit takes RCUT2B from the training configuration, which
 *    --rcut2b does not change.".
 * Before the fix: they ended "; move the knots instead.", "; remember to
 * set RCUT2B in the training config." and ". Set RCUT2B in the training
 * config.". */
TEST_CASE_METHOD(PTFixture,
                 "refit --retarget --rcut2b: the warnings say what the run "
                 "does",
                 "[retarget][rcut][messages]") {
  ScopedTempDir tmp;
  const auto warnings_of = [](std::vector<std::string> lines) {
    RetargetOptions o;
    o.pct = 10.0;
    o.rcut_override["2B"] = 6.0;
    return retarget_hpotarget_lines(lines, o).warnings;
  };
  const auto has = [](const std::vector<std::string> &w,
                      const std::string &text) {
    return std::find(w.begin(), w.end(), text) != w.end();
  };
  write_hpotarget("k.hpotarget", make_knot_pot(), 10.0);
  const auto knot = warnings_of(read_lines("k.hpotarget"));
  INFO("knot warnings:\n" << [&] {
    std::string t;
    for (const auto &w : knot) t += w + "\n";
    return t;
  }());
  CHECK(has(knot, "--rcut2b is ignored: CGRID2B is a knot channel, and "
                  "RCUT2B is its largest knot, which the CGRID2B boxes "
                  "move."));
  write_hpotarget("b.hpotarget", make_blip_pot(), 10.0);
  std::vector<std::string> blip = read_lines("b.hpotarget");
  std::vector<std::string> no_reference;
  for (const auto &l : blip)
    if (l.find("OPTIM RCUT2B") == std::string::npos) no_reference.push_back(l);
  REQUIRE(no_reference.size() < blip.size());
  const auto missing = warnings_of(no_reference);
  INFO("warnings without a reference line:\n" << [&] {
    std::string t;
    for (const auto &w : missing) t += w + "\n";
    return t;
  }());
  CHECK(has(missing, "RCUT2B: no OPTIM reference line in the file — the "
                     "declared cutoff 6.00000000e+00 still binds the "
                     "bump-support cap; the fit takes RCUT2B from the "
                     "training configuration, which --rcut2b does not "
                     "change."));
  for (std::size_t i = 0; i < blip.size(); ++i)
    if (blip[i].rfind("# OPTIM RCUT2B (1)", 0) == 0) {
      blip.insert(blip.begin() + i + 1, "OPTIM RCUT2B (1) 5.9 6.1");
      break;
    }
  const auto active = warnings_of(blip);
  INFO("warnings with an ACTIVE line:\n" << [&] {
    std::string t;
    for (const auto &w : active) t += w + "\n";
    return t;
  }());
  CHECK(has(active, "OPTIM RCUT2B (1): ACTIVE line commented out — --rcut2b "
                    "declares a FIXED cutoff, and searching RCUT on a bump "
                    "channel is a wasted dimension (the bump-support gate "
                    "keeps every bump inside the cutoff's LOW bound); the "
                    "fit takes RCUT2B from the training configuration, which "
                    "--rcut2b does not change."));
}
