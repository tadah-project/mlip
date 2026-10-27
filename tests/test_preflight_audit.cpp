// Catch2 v2 unit tests for the pre-flight audit.
//
// Verifies:
//   * parse_audit_mode handles warn / error / off case-insensitively;
//     throws on garbage.
//   * audit_mode_from_context defaults to Off when AUDIT is absent.
//   * AuditReport::any_fail / any_warn / log() are sane.
//   * enforce_audit_mode throws iff (mode==Error && any_fail()).
//   * run_critical_optim_audit flags Critical seed params absent from
//     the OPTIM key set of a descriptor that has a key in it, ignores
//     those that are covered, and flags nothing for a descriptor with no
//     key in the search space.
//   * run_preflight_soft on a synthetic ρ-collapse data set fires the
//     DM_mBlipEnv cross_param_check (the silent ρ-collapse regression).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/bound_suggester.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/models/descriptors/d2/d2_eam.h>
#include <tadah/models/descriptors/dm/dm_eam.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <tadah/models/descriptors/param_meta.h>

#include <map>
#include <set>
#include <sstream>

using namespace tadah::mlip;
using tadah::core::Context;
using tadah::core::stats::DataStats;
using tadah::core::stats::PerPairStats;
using tadah::core::stats::IDX_P1;
using tadah::core::stats::IDX_P10;
using tadah::core::stats::IDX_P50;
using tadah::core::stats::IDX_P90;
using tadah::core::stats::IDX_P99;
using tadah::models::DescriptorMeta;
using tadah::models::ParamMeta;
using tadah::models::SeedView;
using tadah::models::Scale;
using tadah::models::Criticality;
using tadah::models::PairScope;
using tadah::models::HealthOk;
using tadah::models::HealthWarn;
using tadah::models::HealthFail;
using tadah::models::HealthVerdict;

// Tests use metadata-based overloads (run_preflight_soft_with_metas,
// run_critical_optim_audit_with_metas) so they don't depend on the
// static descriptor registrations being live in the test binary
// (which gets linker-stripped from libtadah.models in some setups).

static DescriptorMeta make_dm_mblipenv_like_meta() {
  DescriptorMeta dm;
  dm.label = "DM_mBlipEnv";

  ParamMeta cgridmb;
  cgridmb.key      = "CGRIDMB";
  cgridmb.arg_pos  = -1;
  cgridmb.scale    = Scale::Distance;
  cgridmb.units    = "Å";
  cgridmb.role     = "Centre of the radial GTO";
  cgridmb.crit     = Criticality::Critical;
  cgridmb.scope    = PairScope::PerElementPair;
  cgridmb.check = [](const DataStats& d, const SeedView& s) -> HealthVerdict {
    if (!s.contains("CGRIDMB")) return HealthOk{};
    double mu = s.get<double>("CGRIDMB");
    auto& pp = d.per_pair[0];
    if (mu < pp.r_min || mu > pp.r_max)
      return HealthFail{"CGRIDMB outside the dataset's r_ij range — ρ collapses."};
    return HealthOk{};
  };
  dm.params.push_back(std::move(cgridmb));

  ParamMeta sgridmb;
  sgridmb.key      = "SGRIDMB";
  sgridmb.arg_pos  = -1;
  sgridmb.scale    = Scale::InverseDistanceSq;
  sgridmb.units    = "Å^-2";
  sgridmb.role     = "Inverse Gaussian width";
  sgridmb.crit     = Criticality::Critical;
  sgridmb.scope    = PairScope::Global;
  dm.params.push_back(std::move(sgridmb));

  ParamMeta rcutmb;
  rcutmb.key       = "RCUTMB";
  rcutmb.arg_pos   = -1;
  rcutmb.scale     = Scale::Distance;
  rcutmb.units     = "Å";
  rcutmb.role      = "Real-space cutoff";
  rcutmb.crit      = Criticality::Critical;
  rcutmb.scope     = PairScope::Global;
  dm.params.push_back(std::move(rcutmb));

  // A key of the descriptor that is not Critical: searching it puts the
  // descriptor in the search space without covering a Critical key.
  ParamMeta rcutenv;
  rcutenv.key      = "RCUTENV";
  rcutenv.arg_pos  = -1;
  rcutenv.scale    = Scale::Density;
  rcutenv.units    = "ρ";
  rcutenv.role     = "Inverse-cutoff envelope on ρ";
  rcutenv.crit     = Criticality::Recommended;
  rcutenv.scope    = PairScope::Global;
  dm.params.push_back(std::move(rcutenv));

  // ρ-collapse cross-param check.
  dm.cross_param_checks.push_back(
    [](const DataStats& d, const SeedView&) -> HealthVerdict {
      auto it = d.rho_percentiles_per_descriptor.find("DM_mBlipEnv");
      if (it == d.rho_percentiles_per_descriptor.end()) return HealthOk{};
      const auto& rho = it->second;
      if (rho[IDX_P99] < 1.0e-6) {
        return HealthFail{
          "ρ collapses to ≤ 1e-6 for 99% of atoms — radial GTO does not "
          "overlap the neighbour shells."};
      }
      return HealthOk{};
    });

  return dm;
}

using MetaPair = std::pair<std::string, DescriptorMeta>;

TEST_CASE("preflight_audit: parse_audit_mode", "[preflight_audit]") {
  REQUIRE(parse_audit_mode("warn")  == AuditMode::Warn);
  REQUIRE(parse_audit_mode("WARN")  == AuditMode::Warn);
  REQUIRE(parse_audit_mode("error") == AuditMode::Error);
  REQUIRE(parse_audit_mode("off")   == AuditMode::Off);
  REQUIRE_THROWS(parse_audit_mode("loud"));
  REQUIRE_THROWS(parse_audit_mode(""));
}

TEST_CASE("preflight_audit: audit_mode_from_context defaults to Off", "[preflight_audit]") {
  Context c;
  REQUIRE(audit_mode_from_context(c) == AuditMode::Off);
  c.add<std::string>("AUDIT", "warn");
  REQUIRE(audit_mode_from_context(c) == AuditMode::Warn);
  c.remove("AUDIT");
  c.add<std::string>("AUDIT", "error");
  REQUIRE(audit_mode_from_context(c) == AuditMode::Error);
}

TEST_CASE("preflight_audit: enforce_audit_mode throws on FAIL+Error", "[preflight_audit]") {
  AuditReport rep;
  AuditMessage m;
  m.level = AuditMessage::Level::Fail;
  m.descriptor_label = "DM_mBlipEnv";
  m.text = "ρ collapse";
  rep.messages.push_back(m);

  REQUIRE_NOTHROW(enforce_audit_mode(rep, AuditMode::Off));
  REQUIRE_NOTHROW(enforce_audit_mode(rep, AuditMode::Warn));
  REQUIRE_THROWS_AS(enforce_audit_mode(rep, AuditMode::Error),
                    std::runtime_error);

  // Empty report -> never throws regardless of mode.
  AuditReport empty;
  REQUIRE_NOTHROW(enforce_audit_mode(empty, AuditMode::Error));

  // Only WARN findings -> never throws even on Error.
  AuditReport warn_only;
  AuditMessage w;
  w.level = AuditMessage::Level::Warn;
  w.text  = "warn only";
  warn_only.messages.push_back(w);
  REQUIRE_NOTHROW(enforce_audit_mode(warn_only, AuditMode::Error));
}

TEST_CASE("preflight_audit: AuditReport::any_fail / any_warn", "[preflight_audit]") {
  AuditReport rep;
  REQUIRE_FALSE(rep.any_fail());
  REQUIRE_FALSE(rep.any_warn());

  AuditMessage w;
  w.level = AuditMessage::Level::Warn;
  rep.messages.push_back(w);
  REQUIRE(rep.any_warn());
  REQUIRE_FALSE(rep.any_fail());

  AuditMessage f;
  f.level = AuditMessage::Level::Fail;
  rep.messages.push_back(f);
  REQUIRE(rep.any_warn());
  REQUIRE(rep.any_fail());
}

/* Pins: with a key of the descriptor in the search space (RCUTENV, not
 * Critical), its fixed Critical keys left out of it (CGRIDMB 2.9, RCUTMB)
 * are flagged at WARN, each message naming its own key; with them in the
 * search space, none is.
 * Before the fix: passes as well (a pin; the case searched nothing, and
 * every Critical key of every descriptor was flagged). */
TEST_CASE("preflight_audit: critical_optim_audit flags a fixed CGRIDMB left out of OPTIM", "[preflight_audit]") {
  // Minimal seed: fixed CGRIDMB, RCUTMB, SGRIDMB; the search space holds
  // RCUTENV, a key of the descriptor that is not Critical.
  // We supply the descriptor meta directly so the test does not depend
  // on dm_all.cpp's static registrations being live in the test binary.
  /* A centre away from the origin: one that can miss the neighbour shells
   * (the case below has the centre at 0). */
  Context seed;
  seed.add<double>("CGRIDMB", 2.9);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  // The descriptor is searched (RCUTENV): every Critical fixed param
  // left out of the search should be flagged.
  const std::vector<std::string> rcutenv_only = {"RCUTENV"};
  auto rep = run_critical_optim_audit_with_metas(seed, rcutenv_only, metas);

  // We expect at least CGRIDMB to be flagged: a fixed CGRIDMB outside the
  // data's support collapses the density block without any error.
  const std::vector<std::string> critical_keys = {"CGRIDMB", "SGRIDMB", "RCUTMB"};
  bool saw_cgridmb = false;
  bool saw_rcutmb  = false;
  for (const auto& m : rep.messages) {
    INFO(m.param_key << ": " << m.text);
    if (m.param_key == "CGRIDMB") saw_cgridmb = true;
    if (m.param_key == "RCUTMB")  saw_rcutmb  = true;
    REQUIRE(m.level == AuditMessage::Level::Warn);
    // The hint explains the failure in terms of the data and names the
    // remedy for this key; it does not refer to the removed BASINS HPO
    // strategy, and it does not illustrate the risk with another key
    // (the text is shared by every Critical parameter).
    CHECK_THAT(m.text, Catch::Matchers::Contains(
        "misses the neighbour distances of the training data"));
    CHECK_THAT(m.text, Catch::Matchers::Contains("OPTIM " + m.param_key));
    CHECK_THAT(m.text, !Catch::Matchers::Contains("BASINS"));
    for (const auto& other : critical_keys)
      if (other != m.param_key)
        CHECK_THAT(m.text, !Catch::Matchers::Contains(other));
  }
  REQUIRE(saw_cgridmb);
  REQUIRE(saw_rcutmb);

  // When CGRIDMB and RCUTMB are in the OPTIM set, they should NOT be flagged.
  std::vector<std::string> with_optim = {"CGRIDMB", "RCUTMB", "SGRIDMB",
                                        "RCUTENV"};
  auto rep2 = run_critical_optim_audit_with_metas(seed, with_optim, metas);
  for (const auto& m : rep2.messages) {
    REQUIRE(m.param_key != "CGRIDMB");
    REQUIRE(m.param_key != "RCUTMB");
    REQUIRE(m.param_key != "SGRIDMB");
  }
}

/* A radial centre at the origin cannot miss the neighbour shells: the
 * basis falls off from r = 0, so it is largest at the nearest neighbours
 * whatever their distance, and its width alone sets its reach. A fixed
 * CGRIDMB whose every value is 0 is not flagged; the width (SGRIDMB) and
 * the cutoff (RCUTMB) still are, and a list with one non-zero centre is
 * flagged.
 * The search space holds RCUTENV, a key of the descriptor that is not
 * Critical, so that the descriptor is searched.
 * Before the fix: CGRIDMB 0 was flagged as a centre that can miss the
 * neighbour distances (the case above pinned it with CGRIDMB 0; it sets
 * 2.9 now). The case searched nothing then, which flagged every Critical
 * key of every descriptor. */
TEST_CASE("preflight_audit: critical_optim_audit does not flag a fixed "
          "CGRIDMB of centres at the origin", "[preflight_audit]") {
  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());
  const std::vector<std::string> no_optim = {"RCUTENV"};
  auto flagged = [](const AuditReport& r, const std::string& key) {
    for (const auto& m : r.messages)
      if (m.param_key == key) return true;
    return false;
  };
  Context seed;
  seed.add<double>("CGRIDMB", 0.0);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);
  auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
  CHECK_FALSE(flagged(rep, "CGRIDMB"));
  CHECK(flagged(rep, "SGRIDMB"));
  CHECK(flagged(rep, "RCUTMB"));

  seed.add<double>("CGRIDMB", 2.9);   /* centres 0 and 2.9 */
  rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
  CHECK(flagged(rep, "CGRIDMB"));
}

// Bump-family metas: RCUT* Critical as in the real D2_Blip5 / DM_REAM
// meta(); CGRID2B is promoted to Critical here (the real meta says
// Recommended) only to prove that the skip is RCUT-specific.
static DescriptorMeta make_bump_pair_like_meta(const std::string& label) {
  DescriptorMeta dm;
  dm.label = label;
  ParamMeta rcut;
  rcut.key   = "RCUT2B";
  rcut.arg_pos = -1;
  rcut.scale = Scale::Distance;
  rcut.units = "Å";
  rcut.role  = "Outer cutoff for the 2-body neighbour sum.";
  rcut.crit  = Criticality::Critical;
  rcut.scope = PairScope::Global;
  dm.params.push_back(std::move(rcut));
  ParamMeta cgrid;
  cgrid.key   = "CGRID2B";
  cgrid.arg_pos = -1;
  cgrid.scale = Scale::Distance;
  cgrid.units = "Å";
  cgrid.role  = "Bump centres";
  cgrid.crit  = Criticality::Critical;
  cgrid.scope = PairScope::PerElementPair;
  dm.params.push_back(std::move(cgrid));
  // A key of the descriptor that is not Critical: searching it puts the
  // descriptor in the search space.
  ParamMeta sgrid;
  sgrid.key   = "SGRID2B";
  sgrid.arg_pos = -1;
  sgrid.scale = Scale::InverseDistance;
  sgrid.units = "Å^-1";
  sgrid.role  = "Bump widths";
  sgrid.crit  = Criticality::Recommended;
  sgrid.scope = PairScope::Global;
  dm.params.push_back(std::move(sgrid));
  return dm;
}

static DescriptorMeta make_dm_ream_like_meta() {
  DescriptorMeta dm;
  dm.label = "DM_REAM";
  ParamMeta rcut;
  rcut.key   = "RCUTMB";
  rcut.arg_pos = -1;
  rcut.scale = Scale::Distance;
  rcut.units = "Å";
  rcut.role  = "Real-space cutoff for the EAM density neighbour sum.";
  rcut.crit  = Criticality::Critical;
  rcut.scope = PairScope::Global;
  dm.params.push_back(std::move(rcut));
  // A key of the descriptor that is not Critical: searching it puts the
  // descriptor in the search space.
  ParamMeta sgrid;
  sgrid.key   = "SGRIDMB";
  sgrid.arg_pos = -1;
  sgrid.scale = Scale::InverseDistance;
  sgrid.units = "Å^-1";
  sgrid.role  = "Density bump widths";
  sgrid.crit  = Criticality::Recommended;
  sgrid.scope = PairScope::Global;
  dm.params.push_back(std::move(sgrid));
  return dm;
}

// A refit-style bump seed: TYPE2B D2_Blip5 <n> <n> Ta Ta, TYPEMB DM_REAM 3
// (Blip5 code) <sizes...> Ta Ta, hard cutoffs unless the caller adds RCTYPE*.
static Context make_bump_seed(const std::string& pair_family) {
  Context seed;
  seed.add("TYPE2B", pair_family);
  seed.add("TYPE2B", 2);
  seed.add("TYPE2B", 2);
  seed.add("TYPE2B", "Ta");
  seed.add("TYPE2B", "Ta");
  seed.add("TYPEMB", "DM_REAM");
  seed.add("TYPEMB", 3);      // basis code 3 = Blip5 (bump family)
  seed.add("TYPEMB", 2);
  seed.add("TYPEMB", 2);
  seed.add("TYPEMB", 2);
  seed.add("TYPEMB", 2);
  seed.add("TYPEMB", 2);
  seed.add("TYPEMB", 0);
  seed.add("TYPEMB", 0);
  seed.add("TYPEMB", "Ta");
  seed.add("TYPEMB", "Ta");
  seed.add<double>("RCUT2B", 5.58);
  seed.add<double>("RCUTMB", 5.58);
  seed.add<double>("CGRID2B", 2.0);
  seed.add<double>("CGRID2B", 3.0);
  return seed;
}

TEST_CASE("preflight_audit: critical_optim_audit does not flag RCUT of bump channels under a hard cutoff",
          "[preflight_audit]") {
  // For D2_Blip/D2_Blip5 pair channels and DM_REAM bump densities the stated
  // RCUT is the fitted truncation (every bump support ends at or before it),
  // so "add OPTIM RCUT2B" is the wrong advice: the dimension is flat above
  // the seed value and only trips the bump-support gate below it. Knot
  // channels were already exempt (RCUT derived from the largest knot).
  // Both descriptors are in the search space through a width key that is
  // not Critical (SGRID2B, SGRIDMB); before the fix the case searched
  // nothing, which flagged every Critical key of every descriptor.
  std::vector<MetaPair> metas;
  metas.emplace_back("D2_Blip5", make_bump_pair_like_meta("D2_Blip5"));
  metas.emplace_back("DM_REAM",  make_dm_ream_like_meta());
  const std::vector<std::string> no_optim = {"SGRID2B", "SGRIDMB"};
  auto flagged = [](const AuditReport& r, const std::string& key) {
    for (const auto& m : r.messages)
      if (m.param_key == key) return true;
    return false;
  };

  SECTION("RCTYPE absent (constructor default Cut_Dummy): RCUT2B/RCUTMB skipped, CGRID2B still flagged") {
    Context seed = make_bump_seed("D2_Blip5");
    auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
    CHECK_FALSE(flagged(rep, "RCUT2B"));
    CHECK_FALSE(flagged(rep, "RCUTMB"));
    CHECK(flagged(rep, "CGRID2B"));
  }
  SECTION("explicit Cut_Dummy on both channels: skipped") {
    Context seed = make_bump_seed("D2_Blip5");
    seed.add("RCTYPE2B", "Cut_Dummy");
    seed.add("RCTYPEMB", "Cut_Dummy");
    auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
    CHECK_FALSE(flagged(rep, "RCUT2B"));
    CHECK_FALSE(flagged(rep, "RCUTMB"));
  }
  SECTION("cubic D2_Blip pair channel is a bump family too: skipped") {
    metas[0] = MetaPair("D2_Blip", make_bump_pair_like_meta("D2_Blip"));
    Context seed = make_bump_seed("D2_Blip");
    auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
    CHECK_FALSE(flagged(rep, "RCUT2B"));
  }
  SECTION("smooth cutoff (Cut_Cos): the damping tail depends on RCUT, the warning stays") {
    Context seed = make_bump_seed("D2_Blip5");
    seed.add("RCTYPE2B", "Cut_Cos");
    seed.add("RCTYPEMB", "Cut_Cos");
    REQUIRE_FALSE(tadah::models::potedit::channel_cutoff_is_discontinuous(seed, "2B"));
    auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
    CHECK(flagged(rep, "RCUT2B"));
    CHECK(flagged(rep, "RCUTMB"));
  }
  SECTION("non-bump pair family (D2_BP) is unaffected: RCUT2B flagged") {
    metas[0] = MetaPair("D2_BP", make_bump_pair_like_meta("D2_BP"));
    Context seed = make_bump_seed("D2_BP");
    auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);
    CHECK(flagged(rep, "RCUT2B"));
    CHECK_FALSE(flagged(rep, "RCUTMB"));   // the DM_REAM density is still a bump channel
  }
  SECTION("OPTIM coverage still silences everything") {
    Context seed = make_bump_seed("D2_Blip5");
    seed.add("RCTYPE2B", "Cut_Cos");
    const std::vector<std::string> with_optim = {"RCUT2B", "CGRID2B", "RCUTMB"};
    auto rep = run_critical_optim_audit_with_metas(seed, with_optim, metas);
    CHECK(rep.messages.empty());
  }
}

// Build a synthetic DataStats that triggers the DM_mBlipEnv ρ-collapse
// cross_param_check: rho_percentiles_per_descriptor for "DM_mBlipEnv"
// has p99 < 1e-6.
static DataStats make_collapsed_rho_data() {
  DataStats d;
  d.n_atoms = 100;
  d.n_structures = 10;
  d.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 200;
  pp.r_min = 2.5;
  pp.r_max = 6.0;
  for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i) {
    pp.r_percentiles[i] = 2.5 + 0.4 * static_cast<double>(i);
  }
  d.per_pair.push_back(pp);
  // ρ collapsed near zero across percentiles — the collapse signature.
  std::array<double, 11> rho{};
  for (auto& v : rho) v = 1e-12;
  d.rho_percentiles_per_descriptor["DM_mBlipEnv"] = rho;
  return d;
}

TEST_CASE("preflight_audit: run_preflight_soft fires on collapsed ρ", "[preflight_audit]") {
  auto data = make_collapsed_rho_data();
  Context seed;
  seed.add<double>("CGRIDMB", 3.0);   // inside [r_min, r_max]
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);
  seed.add<double>("RCUTENV", 5.0);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  auto rep = run_preflight_soft_with_metas(data, seed, metas, AuditMode::Warn);

  // At least one DM_mBlipEnv finding with text mentioning "ρ" or "collapse".
  bool saw_collapse = false;
  for (const auto& m : rep.messages) {
    if (m.descriptor_label != "DM_mBlipEnv") continue;
    if (m.text.find("collapse") != std::string::npos ||
        m.text.find("\xCF\x81")  != std::string::npos /* ρ in UTF-8 */ ) {
      saw_collapse = true;
      break;
    }
  }
  REQUIRE(saw_collapse);
  REQUIRE(rep.any_fail());
}

TEST_CASE("preflight_audit: per-param check flags out-of-range CGRIDMB", "[preflight_audit]") {
  // Healthy ρ stats but CGRIDMB at 0.0 — outside r_min/r_max.
  DataStats data;
  data.n_atoms = 10;
  data.n_structures = 1;
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.r_min = 2.5;
  pp.r_max = 6.0;
  for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i) {
    pp.r_percentiles[i] = 2.5 + 0.4 * static_cast<double>(i);
  }
  data.per_pair.push_back(pp);

  Context seed;
  seed.add<double>("CGRIDMB", 0.0);  // outside [2.5, 6.0]
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  auto rep = run_preflight_soft_with_metas(data, seed, metas, AuditMode::Warn);
  bool saw = false;
  for (const auto& m : rep.messages) {
    if (m.param_key == "CGRIDMB" &&
        m.level == AuditMessage::Level::Fail) {
      saw = true;
      REQUIRE(m.text.find("CGRIDMB") != std::string::npos);
      break;
    }
  }
  REQUIRE(saw);
}

TEST_CASE("preflight_audit: Off mode returns empty report", "[preflight_audit]") {
  auto data = make_collapsed_rho_data();
  Context seed;
  seed.add<double>("CGRIDMB", 3.0);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  auto rep = run_preflight_soft_with_metas(data, seed, metas, AuditMode::Off);
  REQUIRE(rep.messages.empty());
}

/* Pins: for a D2_EAM pair term and a DM_EAM many-body term (their own
 * meta()), with RCUT2B and RCUTMB set and no OPTIM line, the critical-OPTIM
 * audit flags neither cutoff, and suggest_bounds_for gives no RCUT2B or
 * RCUTMB box: the setfl file fixes the cutoff of both descriptors (they
 * refuse another value), so it is not a search target.
 * Before the fix: both cutoffs were flagged ("Consider adding `OPTIM RCUT2B
 * (...)` from `tadah analysis suggest_bounds`") and each got a suggested
 * box, a search tadah hpo now refuses at startup and whose every candidate
 * but the seed failed before. */
TEST_CASE("preflight_audit: the setfl cutoff of D2_EAM and DM_EAM is not "
          "flagged or suggested",
          "[preflight_audit][eam]") {
  Context seed;
  seed.add("TYPE2B", "D2_EAM 0 Ta Ta");
  seed.add("TYPEMB", "DM_EAM 0 Ta Ta");
  seed.add<double>("RCUT2B", 5.58190742024);
  seed.add<double>("RCUTMB", 5.58190742024);
  std::vector<MetaPair> metas;
  const auto d2 = tadah::models::D2_EAM().meta();
  const auto dm = tadah::models::DM_EAM().meta();
  REQUIRE(d2.has_value());
  REQUIRE(dm.has_value());
  metas.emplace_back("D2_EAM", *d2);
  metas.emplace_back("DM_EAM", *dm);
  const auto rep = run_critical_optim_audit_with_metas(seed, {}, metas);
  for (const auto &m : rep.messages) {
    INFO(m.descriptor_label << "." << m.param_key << ": " << m.text);
    CHECK(m.param_key != "RCUT2B");
    CHECK(m.param_key != "RCUTMB");
  }
  // The cutoff is the only key of either descriptor, and it is not
  // Critical: no search space makes the audit flag it.
  for (const auto &meta : {*d2, *dm})
    for (const auto &p : meta.params) {
      INFO(meta.label << "." << p.key);
      CHECK(p.crit != Criticality::Critical);
    }

  DataStats data;
  data.n_atoms = 16;
  data.n_structures = 1;
  data.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 64;
  pp.r_min = 2.5;
  pp.r_max = 6.0;
  for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i)
    pp.r_percentiles[i] = 2.5 + 0.3 * static_cast<double>(i);
  data.per_pair.push_back(pp);
  for (const auto &meta : {*d2, *dm}) {
    INFO(meta.label);
    CHECK(suggest_bounds_for(data, meta, seed).empty());
  }
}

/* Pins: the audit flags nothing for a descriptor with no key in the search
 * space: a search of FIXWEIGHT alone, or of nothing, gives an empty report
 * for a DM_mBlipEnv-like and a D2_BP-like descriptor whose Critical keys
 * (CGRIDMB 2.9, SGRIDMB, RCUTMB, RCUT2B, CGRID2B) are all fixed; a search of
 * SGRID2B alone flags the D2_BP keys (RCUT2B, CGRID2B) and none of the
 * DM_mBlipEnv-like descriptor's.
 * Before the fix: every Critical key of every configured descriptor was
 * flagged whatever the search space (the warning then read "Critical
 * hyperparameter is fixed in the seed config and is not in any OPTIM block
 * ..."), also for a search
 * of FIXWEIGHT alone, which tadah hpo printed on every run, AUDIT off
 * included. */
TEST_CASE("preflight_audit: critical_optim_audit flags nothing for a "
          "descriptor with no key in the search space",
          "[preflight_audit]") {
  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());
  metas.emplace_back("D2_BP", make_bump_pair_like_meta("D2_BP"));
  Context seed;
  seed.add<double>("CGRIDMB", 2.9);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);
  seed.add("TYPE2B", "D2_BP 2 2 Ta Ta");
  seed.add<double>("RCUT2B", 5.6);
  seed.add<double>("CGRID2B", 2.0);
  seed.add<double>("CGRID2B", 3.0);
  for (const std::vector<std::string> &space :
       {std::vector<std::string>{"FIXWEIGHT"}, std::vector<std::string>{}}) {
    INFO("search space of " << space.size() << " key(s)");
    const auto rep = run_critical_optim_audit_with_metas(seed, space, metas);
    std::string flagged;
    for (const auto &m : rep.messages)
      flagged += " " + m.descriptor_label + "." + m.param_key;
    INFO("flagged:" << flagged);
    CHECK(rep.messages.empty());
  }
  const auto rep =
      run_critical_optim_audit_with_metas(seed, {"SGRID2B"}, metas);
  std::set<std::string> flagged;
  for (const auto &m : rep.messages)
    flagged.insert(m.descriptor_label + "." + m.param_key);
  CAPTURE(flagged);
  CHECK(flagged == std::set<std::string>{"D2_BP.CGRID2B", "D2_BP.RCUT2B"});
}

// ---------------------------------------------------------------------------
//  A joined line: each descriptor is judged by its own channels and cutoff
// ---------------------------------------------------------------------------

namespace {

/* Neighbour distances of bcc Ta: r_min 2.25 A, shells at 2.86 and 3.30 A,
 * r_max 5.6 A. */
DataStats bcc_ta_distances() {
  DataStats d;
  d.n_atoms = 100;
  d.n_structures = 10;
  d.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 4000;
  pp.r_min = 2.25;
  pp.r_max = 5.6;
  pp.r_percentiles = {2.6, 2.8, 2.85, 2.9, 3.4, 4.7, 5.2, 5.4, 5.55,
                      4.0, 1.0};
  tadah::core::stats::KdePeak first, second;
  first.position = 2.86;
  first.density = 1.0;
  first.fwhm = 0.7;
  second.position = 3.30;
  second.density = 0.7;
  second.fwhm = 0.3;
  pp.r_peaks = {first, second};
  d.per_pair.push_back(pp);
  return d;
}

/* TYPEMB DM_mJoin of a DM_EAD Gaussian channel (CGRIDMB(1)) and two
 * DM_Blip blips (CGRIDMB(2), CGRIDMB(3)), every SGRIDMB 1. */
Context joined_density_seed(const std::string& cgridmb,
                            const std::string& rcutmb) {
  Context c;
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_mJoin");
  c.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  c.add("TYPEMB", "DM_Blip 0 2 2 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  c.add("RCUTMB", rcutmb);
  c.add("CGRIDMB", cgridmb);
  c.add("SGRIDMB", "1.0 1.0 1.0");
  return c;
}

std::string fails_of(const AuditReport& rep) {
  std::string s;
  for (const auto& m : rep.messages)
    if (m.level == AuditMessage::Level::Fail)
      s += m.descriptor_label + "." + m.param_key + ": " + m.text + "\n";
  return s;
}

}  // namespace

/* Pins: the pre-flight audit judges each descriptor of a DM_mJoin line by
 * its own channels and its own RCUTMB, and names a channel by its entry of
 * the whole key, as an OPTIM line does: for DM_EAD 0 1 1 (a Gaussian,
 * CGRIDMB(1) = 0) joined with DM_Blip 0 2 2 (blips CGRIDMB(2) and (3)) on
 * bcc Ta distances (r_min 2.25 A), every SGRIDMB 1:
 *  - CGRIDMB 0 3 3.5, RCUTMB 5.6 5.6: no FAIL (the Gaussian is 6e-3 at
 *    r_min, each blip's support |r - c| < 2 holds it);
 *  - CGRIDMB 0 3 30: one FAIL, DM_Blip.CGRIDMB, naming "CGRIDMB(3) = 30"
 *    and the blip;
 *  - CGRIDMB 0 3 3.5, RCUTMB 5.6 2.0: one FAIL, DM_Blip's, naming "RCUTMB
 *    = 2", its own cutoff, below every neighbour distance.
 * Before the fix: each descriptor judged every channel of the line with
 * its own form and the first RCUTMB: DM_Blip failed the Gaussian at the
 * origin as a blip ("CGRIDMB(1) = 0 with SGRIDMB(1) = 1: the support of
 * the blip ... holds no neighbour distance"), DM_EAD failed the blip at 30
 * as a Gaussian, and DM_Blip was judged at RCUTMB 5.6. */
TEST_CASE("preflight_audit: each descriptor of a DM_mJoin line is judged by "
          "its own channels and cutoff",
          "[preflight_audit][join]") {
  const DataStats data = bcc_ta_distances();
  SECTION("a Gaussian at the origin and two blips pass") {
    const auto rep = run_preflight_soft(
        data, joined_density_seed("0.0 3.0 3.5", "5.6 5.6"), AuditMode::Error);
    INFO(fails_of(rep));
    CHECK_FALSE(rep.any_fail());
  }
  SECTION("a blip far out fails, named by its entry of the whole key") {
    const auto rep = run_preflight_soft(
        data, joined_density_seed("0.0 3.0 30", "5.6 5.6"), AuditMode::Error);
    const std::string fails = fails_of(rep);
    INFO(fails);
    std::size_t n = 0;
    for (const auto& m : rep.messages) {
      if (m.level != AuditMessage::Level::Fail) continue;
      ++n;
      CHECK(m.descriptor_label == "DM_Blip");
      CHECK(m.text.find("CGRIDMB(3) = 30") != std::string::npos);
      CHECK(m.text.find("blip") != std::string::npos);
    }
    CHECK(n == 1u);
  }
  SECTION("a blip's own cutoff below every distance fails") {
    const auto rep = run_preflight_soft(
        data, joined_density_seed("0.0 3.0 3.5", "5.6 2.0"), AuditMode::Error);
    const std::string fails = fails_of(rep);
    INFO(fails);
    std::size_t n = 0;
    for (const auto& m : rep.messages) {
      if (m.level != AuditMessage::Level::Fail) continue;
      ++n;
      CHECK(m.descriptor_label == "DM_Blip");
      CHECK(m.text.find("RCUTMB = 2 ") != std::string::npos);
    }
    CHECK(n == 1u);
  }
}

// ---------------------------------------------------------------------------
//  The critical-OPTIM audit of a joined line, and a cutoff that holds no
//  neighbour
// ---------------------------------------------------------------------------

#include <algorithm>
#include <string>
#include <vector>

namespace {

/* The findings of @p rep as "label.key: text", sorted. */
std::vector<std::string> findings_of(const AuditReport& rep) {
  std::vector<std::string> out;
  for (const auto& m : rep.messages)
    out.push_back(m.descriptor_label + "." + m.param_key + ": " + m.text);
  std::sort(out.begin(), out.end());
  return out;
}

/* The findings of @p rep as "label.key", sorted. */
std::vector<std::string> keys_of(const AuditReport& rep) {
  std::vector<std::string> out;
  for (const auto& m : rep.messages)
    out.push_back(m.descriptor_label + "." + m.param_key);
  std::sort(out.begin(), out.end());
  return out;
}

/* TYPEMB DM_EAD 0 1 1 Ta Ta, one Gaussian channel at @p centre, SGRIDMB 1,
 * RCUTMB 5.6. */
Context single_ead_seed(const std::string& centre) {
  Context c;
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUTMB", 5.6);
  c.add("CGRIDMB", centre);
  c.add("SGRIDMB", 1.0);
  return c;
}

}  // namespace

/* Pins: the critical-OPTIM audit (tadah hpo under AUDIT warn and error)
 * judges each descriptor of a DM_mJoin line by its own values: for DM_mJoin
 * of DM_EAD with its Gaussian at the origin (CGRIDMB(1) = 0) and DM_EAD with
 * its Gaussian on the first shell (CGRIDMB(2) = 2.9), RCUTMB 5.6 5.6, a
 * search of SGRIDMB, the findings are those of the two descriptors alone,
 * together: RCUTMB of each and CGRIDMB of the second (a centre at the
 * origin is not flagged), three findings, each naming its descriptor and
 * key as alone (its text names its entries of the whole key: see the next
 * test case).
 * Before the fix: each descriptor read the whole line: CGRIDMB 0 2.9 is not
 * every centre at the origin, so both descriptors' CGRIDMB were flagged,
 * four findings. */
TEST_CASE("preflight_audit: critical_optim_audit judges each descriptor of a "
          "joined line by its own values",
          "[preflight_audit][join]") {
  Context joined;
  joined.add("INITMB", "true");
  joined.add("TYPEMB", "DM_mJoin");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  joined.add("RCUTMB", "5.6 5.6");
  joined.add("CGRIDMB", "0 2.9");
  joined.add("SGRIDMB", "1 1");
  const std::vector<std::string> space = {"SGRIDMB"};
  std::vector<std::string> alone =
      keys_of(run_critical_optim_audit(single_ead_seed("0"), space));
  const std::vector<std::string> second =
      keys_of(run_critical_optim_audit(single_ead_seed("2.9"), space));
  alone.insert(alone.end(), second.begin(), second.end());
  std::sort(alone.begin(), alone.end());
  const std::vector<std::string> together =
      keys_of(run_critical_optim_audit(joined, space));
  CAPTURE(alone, together);
  CHECK(together == alone);
  CHECK(together.size() == 3u);
}

/* Pins: a critical-parameter warning names the entries of its descriptor
 * as an OPTIM line numbers them, and the OPTIM line that searches them, in
 * impersonal words: for DM_mJoin of two DM_EAD (CGRIDMB 0 2.9, RCUTMB 5.6
 * 5.6) and a search of SGRIDMB the three texts begin "RCUTMB(1) is a
 * critical parameter that the seed fixes and no OPTIM line searches",
 * "RCUTMB(2) ..." and "CGRIDMB(2) ...", and name `OPTIM RCUTMB (1)`,
 * `OPTIM RCUTMB (2)` and `OPTIM CGRIDMB (2)` and tadah analysis
 * suggest_bounds; for a DM_EAD alone with centres 2.9 and 3.3, the CGRIDMB
 * text begins "CGRIDMB(1-2) is a critical parameter" and names `OPTIM
 * CGRIDMB (1-2)`. No text says "Consider", "OPTIM block" or
 * "hyperparameter".
 * Before the fix: every text read "Critical hyperparameter is fixed in the
 * seed config and is not in any OPTIM block. ... Consider adding `OPTIM
 * RCUTMB (...) <lo> <hi>` from `tadah analysis suggest_bounds`.", the
 * warnings about the two cutoffs of the join alike. */
TEST_CASE("preflight_audit: a critical-parameter warning names the entries "
          "of its descriptor and the OPTIM line that searches them",
          "[preflight_audit][join]") {
  Context joined;
  joined.add("INITMB", "true");
  joined.add("TYPEMB", "DM_mJoin");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  joined.add("RCUTMB", "5.6 5.6");
  joined.add("CGRIDMB", "0 2.9");
  joined.add("SGRIDMB", "1 1");
  const std::string fixed =
      " is a critical parameter that the seed fixes and no OPTIM line "
      "searches";
  std::vector<std::string> texts;
  for (const auto& m : run_critical_optim_audit(joined, {"SGRIDMB"}).messages)
    texts.push_back(m.text);
  std::sort(texts.begin(), texts.end());
  CAPTURE(texts);
  REQUIRE(texts.size() == 3u);
  const char* const entries[] = {"CGRIDMB(2)", "RCUTMB(1)", "RCUTMB(2)"};
  const char* const lines[] = {"`OPTIM CGRIDMB (2)", "`OPTIM RCUTMB (1)",
                               "`OPTIM RCUTMB (2)"};
  for (std::size_t i = 0; i < 3; ++i) {
    CAPTURE(entries[i]);
    CHECK(texts[i].rfind(entries[i] + fixed, 0) == 0);
    CHECK(texts[i].find(lines[i]) != std::string::npos);
    CHECK(texts[i].find("tadah analysis suggest_bounds") != std::string::npos);
  }

  Context lone;
  lone.add("INITMB", "true");
  lone.add("TYPEMB", "DM_EAD 0 2 2 Ta Ta");
  lone.add("RCTYPEMB", "Cut_Cos");
  lone.add("RCUTMB", 5.6);
  lone.add("CGRIDMB", "2.9 3.3");
  lone.add("SGRIDMB", "1 1");
  std::size_t centres = 0;
  for (const auto& m : run_critical_optim_audit(lone, {"SGRIDMB"}).messages) {
    texts.push_back(m.text);
    if (m.param_key != "CGRIDMB") continue;
    ++centres;
    CAPTURE(m.text);
    CHECK(m.text.rfind("CGRIDMB(1-2)" + fixed, 0) == 0);
    CHECK(m.text.find("`OPTIM CGRIDMB (1-2)") != std::string::npos);
  }
  CHECK(centres == 1u);
  for (const auto& text : texts) {
    CAPTURE(text);
    CHECK(text.find("Consider") == std::string::npos);
    CHECK(text.find("OPTIM block") == std::string::npos);
    CHECK(text.find("hyperparameter") == std::string::npos);
  }
}

namespace {

/* TYPE2B D2_Knot5 3 3 Ta Ta alone: knots 2.5 3.5 4.5 (KDEG2B 5),
 * Cut_Dummy, RCUT2B 4.5. */
Context knot_alone_seed() {
  Context c;
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Knot5 3 3 Ta Ta");
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("RCUT2B", 4.5);
  c.add("CGRID2B", "2.5 3.5 4.5");
  c.add("KDEG2B", "5 5 5");
  return c;
}

/* TYPE2B D2_Blip 3 3 Ta Ta alone: centres 2.5 3 3.5, SGRID2B 1, under
 * @p rctype, RCUT2B 5. */
Context blip_alone_seed(const std::string& rctype) {
  Context c;
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("RCTYPE2B", rctype);
  c.add("RCUT2B", 5.0);
  c.add("CGRID2B", "2.5 3.0 3.5");
  c.add("SGRID2B", "1 1 1");
  return c;
}

/* TYPE2B D2_mJoin of the two above, under @p rctype ("Cut_Cos Cut_Cos"),
 * RCUT2B 4.5 5. */
Context knot_blip_join_seed(const std::string& rctype) {
  Context c;
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_mJoin");
  c.add("TYPE2B", "D2_Knot5 3 3 Ta Ta");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("RCTYPE2B", rctype);
  c.add("RCUT2B", "4.5 5.0");
  c.add("CGRID2B", "2.5 3.5 4.5 2.5 3.0 3.5");
  c.add("KDEG2B", "5 5 5");
  c.add("SGRID2B", "1 1 1");
  return c;
}

}  // namespace

/* Pins: the critical-OPTIM audit leaves out the cutoff of a knot channel,
 * and that of a bump channel under a hard cutoff, wherever tadah hpo
 * derives it from the largest knot or holds the bump supports within it:
 * on a line of that descriptor alone and for that descriptor of a joined
 * line, whose own RCUT entry tadah hpo derives or holds. For a search of
 * CGRID2B:
 *  - D2_Knot5 alone (Cut_Dummy) and D2_Blip alone under Cut_Dummy give no
 *    finding, and D2_Blip alone under Cut_Cos gives D2_Blip.RCUT2B;
 *  - D2_mJoin of D2_Knot5 and D2_Blip gives what the two give alone:
 *    D2_Blip.RCUT2B under Cut_Cos Cut_Cos, no finding under Cut_Dummy
 *    Cut_Dummy.
 * Before the fix (tadah hpo derived and held neither on a joined line):
 * the joined line gave D2_Blip.RCUT2B and D2_Knot5.RCUT2B under both. */
TEST_CASE("preflight_audit: critical_optim_audit leaves out a knot or bump "
          "cutoff for each descriptor of a joined line as for one alone",
          "[preflight_audit][join][knot]") {
  const std::vector<std::string> space = {"CGRID2B"};
  CHECK(keys_of(run_critical_optim_audit(knot_alone_seed(), space)).empty());
  CHECK(keys_of(run_critical_optim_audit(blip_alone_seed("Cut_Dummy"), space))
            .empty());
  CHECK(keys_of(run_critical_optim_audit(blip_alone_seed("Cut_Cos"), space)) ==
        std::vector<std::string>{"D2_Blip.RCUT2B"});
  CHECK(keys_of(run_critical_optim_audit(knot_blip_join_seed("Cut_Cos Cut_Cos"),
                                         space)) ==
        std::vector<std::string>{"D2_Blip.RCUT2B"});
  CHECK(keys_of(run_critical_optim_audit(
                    knot_blip_join_seed("Cut_Dummy Cut_Dummy"), space))
            .empty());
}

/* Pins: given the searched entries of each key (the overload tadah hpo
 * calls, with HPOTargets::optimap's opt flags), the audit counts a key as
 * searched for a descriptor when the search holds one of that
 * descriptor's own entries of it, and a descriptor as searched when it
 * holds one of its entries of any key. DM_mJoin of two DM_EAD (centres
 * 2.5 and 2.9, RCUTMB 5.6 5.6), searched CGRIDMB (1) and SGRIDMB (2):
 * four findings, RCUTMB(1) and SGRIDMB(1) of the first, RCUTMB(2) and
 * CGRIDMB(2) of the second; searched SGRIDMB (2) alone: the first is not
 * searched, two findings, RCUTMB(2) and CGRIDMB(2). A line of one DM_EAD
 * (centres 2.5 2.9) with CGRIDMB (1) searched: CGRIDMB counts as searched,
 * two findings, RCUTMB and SGRIDMB(1-2). The key-name overload counts every
 * entry of a named key: CGRIDMB and SGRIDMB give RCUTMB(1) and RCUTMB(2).
 * Before the fix: no overload took the searched entries; tadah hpo passed
 * key names, so OPTIM CGRIDMB (1) counted as a search of the second
 * descriptor's CGRIDMB too. */
TEST_CASE("preflight_audit: critical_optim_audit judges the search of a "
          "joined line by its descriptors' own entries",
          "[preflight_audit][join]") {
  Context joined;
  joined.add("INITMB", "true");
  joined.add("TYPEMB", "DM_mJoin");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  joined.add("RCUTMB", "5.6 5.6");
  joined.add("CGRIDMB", "2.5 2.9");
  joined.add("SGRIDMB", "1 1");
  auto entries_of = [](const AuditReport& rep) {
    std::vector<std::string> out;
    for (const auto& m : rep.messages)
      out.push_back(m.text.substr(0, m.text.find(' ')));
    std::sort(out.begin(), out.end());
    return out;
  };
  const std::map<std::string, std::vector<bool>> both = {
      {"CGRIDMB", {true, false}}, {"SGRIDMB", {false, true}}};
  CHECK(entries_of(run_critical_optim_audit(joined, both)) ==
        std::vector<std::string>{"CGRIDMB(2)", "RCUTMB(1)", "RCUTMB(2)",
                                 "SGRIDMB(1)"});
  const std::map<std::string, std::vector<bool>> second = {
      {"SGRIDMB", {false, true}}};
  CHECK(entries_of(run_critical_optim_audit(joined, second)) ==
        std::vector<std::string>{"CGRIDMB(2)", "RCUTMB(2)"});
  const std::vector<std::string> named = {"CGRIDMB", "SGRIDMB"};
  CHECK(entries_of(run_critical_optim_audit(joined, named)) ==
        std::vector<std::string>{"RCUTMB(1)", "RCUTMB(2)"});

  Context lone;
  lone.add("INITMB", "true");
  lone.add("TYPEMB", "DM_EAD 0 2 2 Ta Ta");
  lone.add("RCTYPEMB", "Cut_Cos");
  lone.add("RCUTMB", 5.6);
  lone.add("CGRIDMB", "2.5 2.9");
  lone.add("SGRIDMB", "1 1");
  const std::map<std::string, std::vector<bool>> first = {
      {"CGRIDMB", {true, false}}};
  CHECK(entries_of(run_critical_optim_audit(lone, first)) ==
        std::vector<std::string>{"RCUTMB(1)", "SGRIDMB(1-2)"});
}

/* Pins: the pre-flight audit fails a descriptor whose cutoff holds no
 * neighbour distance, whichever descriptor it is: D2_Knot5 with RCUT2B 2
 * (knots 1.5 and 2) on bcc Ta distances (r_min 2.25 A) gives one FAIL,
 * naming RCUT2B and 2, under AUDIT error.
 * Before the fix: D2_Knot5's cutoff had no check: no FAIL, and tadah
 * train trained a descriptor that is 0 for every structure. */
TEST_CASE("preflight_audit: a cutoff that holds no neighbour distance is a "
          "FAIL finding",
          "[preflight_audit][cutoff]") {
  Context seed;
  seed.add("INIT2B", "true");
  seed.add("TYPE2B", "D2_Knot5 2 2 Ta Ta");
  seed.add("RCTYPE2B", "Cut_Dummy");
  seed.add("RCUT2B", 2.0);
  seed.add("CGRID2B", "1.5 2.0");
  seed.add("KDEG2B", "5 5");
  const auto rep = run_preflight_soft(bcc_ta_distances(), seed,
                                      AuditMode::Error);
  const std::string fails = fails_of(rep);
  INFO(fails);
  std::size_t n = 0;
  for (const auto& m : rep.messages) {
    if (m.level != AuditMessage::Level::Fail) continue;
    ++n;
    CHECK(m.descriptor_label == "D2_Knot5");
    CHECK(m.param_key == "RCUT2B");
    CHECK(m.text.find("RCUT2B = 2 ") != std::string::npos);
  }
  CHECK(n == 1u);
}

#include <tadah/core/logger.h>
#include <iostream>

/* Pins: the log line of a finding (AuditReport::log, at WARNING) reads
 * "AUDIT finding WARN D2_Blip.RCUT2B: <text>", "AUDIT finding FAIL
 * DM_mBlipEnv: <text>" for a check of the whole descriptor; the report
 * (AuditReport::print) lists "  WARN D2_Blip.RCUT2B: <text>": the key AUDIT
 * that runs the audit, the level and the parameter, without the name of
 * the code that wrote them.
 * Before the fix: "preflight_audit: [WARN] D2_Blip.RCUT2B — <text>" and
 * "  [WARN] D2_Blip.RCUT2B — <text>". */
TEST_CASE("preflight_audit: a finding names the key AUDIT, its level and "
          "its parameter",
          "[preflight_audit]") {
  AuditReport rep;
  AuditMessage w;
  w.level = AuditMessage::Level::Warn;
  w.descriptor_label = "D2_Blip";
  w.param_key = "RCUT2B";
  w.text = "a warning";
  rep.messages.push_back(w);
  AuditMessage f;
  f.level = AuditMessage::Level::Fail;
  f.descriptor_label = "DM_mBlipEnv";
  f.text = "a failure";
  rep.messages.push_back(f);

  auto &logger = tadah::core::Logger::instance();
  const auto level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream logged;
  std::streambuf *old = std::cout.rdbuf(logged.rdbuf());
  rep.log();
  std::cout.rdbuf(old);
  logger.setLogLevel(level);
  INFO("log:\n" << logged.str());
  CHECK(logged.str().find("WARNING: AUDIT finding WARN D2_Blip.RCUT2B: a "
                          "warning\n") != std::string::npos);
  CHECK(logged.str().find("WARNING: AUDIT finding FAIL DM_mBlipEnv: a "
                          "failure\n") != std::string::npos);
  CHECK(logged.str().find("preflight_audit") == std::string::npos);

  std::ostringstream printed;
  rep.print(printed);
  INFO("report:\n" << printed.str());
  CHECK(printed.str().find("\n  WARN D2_Blip.RCUT2B: a warning\n") !=
        std::string::npos);
  CHECK(printed.str().find("\n  FAIL DM_mBlipEnv: a failure\n") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
//  The element pairs of a descriptor's line: a descriptor sums over the
//  neighbours of the element pairs its TYPE2B or TYPEMB line names, and the
//  audit judges it by their distances alone
// ---------------------------------------------------------------------------

namespace {

/* Ta-W data (DataStats sorts per_pair by element pair): the (Ta, Ta) pairs
 * from 2.86 to 3.3 A, the (Ta, W) pairs from 2.4 to 2.6 A and the (W, W)
 * pairs from 2.5 to 2.7 A. */
DataStats ta_w_distances() {
  DataStats d;
  d.n_atoms = 40;
  d.n_structures = 4;
  d.elements = {"Ta", "W"};
  auto pair = [](const char* a, const char* b, double lo, double hi) {
    PerPairStats pp;
    pp.elements = {a, b};
    pp.n_pairs = 100;
    pp.r_min = lo;
    pp.r_max = hi;
    for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i)
      pp.r_percentiles[i] = lo + (hi - lo) * static_cast<double>(i) / 10.0;
    return pp;
  };
  d.per_pair = {pair("Ta", "Ta", 2.86, 3.3), pair("Ta", "W", 2.4, 2.6),
                pair("W", "W", 2.5, 2.7)};
  return d;
}

/* TYPE2B D2_Blip 2 2 <pairs> alone, RCUT2B @p rcut, centres 1 and 2,
 * SGRID2B 1 1, Cut_Cos. */
Context blip_line(const std::string& pairs, double rcut) {
  Context c;
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip 2 2 " + pairs);
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", rcut);
  c.add("CGRID2B", "1 2");
  c.add("SGRID2B", "1 1");
  return c;
}

}  // namespace

/* Pins: the pre-flight audit judges the cutoff of a descriptor by the
 * neighbour distances of the element pairs its line names, on Ta-W data
 * whose (Ta, Ta) pairs start at 2.86 A, (Ta, W) at 2.4 A and (W, W) at
 * 2.5 A (ta_w_distances()), for D2_Blip 2 2 <pairs> with RCUT2B r:
 *  - "Ta Ta", r = 2.6: one FAIL, "D2_Blip.RCUT2B: RCUT2B = 2.6 is below the
 *    smallest (Ta, Ta) neighbour distance of the dataset (2.86 Å): the
 *    descriptor sums over no neighbour and is 0 for every atom.";
 *  - "W W", r = 2.45: one FAIL naming "(W, W)" and 2.5 Å;
 *  - "Ta Ta W W", r = 2.45: one FAIL naming "(Ta, Ta) or (W, W)" and 2.5 Å;
 *  - "Mo Mo", r = 5: one FAIL, "RCUT2B = 5: no (Mo, Mo) pair of atoms of the
 *    dataset lies within it, ...";
 *  - "W W" and "Ta Ta W W" at 2.6, "Ta W" and "* *" at 2.45: no finding (a
 *    pin: no false FAIL, as since the check read every element pair).
 * Before the fix: the check read the smallest neighbour distance of every
 * element pair of the data, 2.4 A: none of the four FAILs was given, and
 * AUDIT error trained descriptors that are 0 for every atom. */
TEST_CASE("preflight_audit: a cutoff is judged by the element pairs of its "
          "descriptor's line",
          "[preflight_audit][cutoff][line_pairs]") {
  const DataStats data = ta_w_distances();
  auto fails = [&](const std::string& pairs, double rcut) {
    std::vector<std::string> out;
    for (const auto& m :
         run_preflight_soft(data, blip_line(pairs, rcut), AuditMode::Error)
             .messages)
      if (m.level == AuditMessage::Level::Fail)
        out.push_back(m.descriptor_label + "." + m.param_key + ": " + m.text);
    return out;
  };
  const std::string none =
      ": the descriptor sums over no neighbour and is 0 for every atom.";
  CHECK(fails("Ta Ta", 2.6) ==
        std::vector<std::string>{
            "D2_Blip.RCUT2B: RCUT2B = 2.6 is below the smallest (Ta, Ta) "
            "neighbour distance of the dataset (2.86 Å)" + none});
  CHECK(fails("W W", 2.45) ==
        std::vector<std::string>{
            "D2_Blip.RCUT2B: RCUT2B = 2.45 is below the smallest (W, W) "
            "neighbour distance of the dataset (2.5 Å)" + none});
  CHECK(fails("Ta Ta W W", 2.45) ==
        std::vector<std::string>{
            "D2_Blip.RCUT2B: RCUT2B = 2.45 is below the smallest (Ta, Ta) or "
            "(W, W) neighbour distance of the dataset (2.5 Å)" + none});
  CHECK(fails("Mo Mo", 5.0) ==
        std::vector<std::string>{
            "D2_Blip.RCUT2B: RCUT2B = 5: no (Mo, Mo) pair of atoms of the "
            "dataset lies within it, so the descriptor sums over no "
            "neighbour and is 0 for every atom."});
  CHECK(fails("W W", 2.6).empty());
  CHECK(fails("Ta Ta W W", 2.6).empty());
  CHECK(fails("Ta W", 2.45).empty());
  CHECK(fails("* *", 2.45).empty());
}

/* Pins: a Gaussian channel of a many-body density is judged by the
 * neighbour distances of the element pairs of its descriptor's line: on
 * the Ta-W data, CGRIDMB 3.1 with SGRIDMB 100 (below 1e-6 farther than
 * 0.37 A from 3.1 A) under RCUTMB 3.5:
 *  - DM_EAD 0 1 1 W W: one FAIL, DM_EAD.CGRIDMB, "... below 1e-6 at every
 *    (W, W) neighbour distance within RCUTMB = 3.5 (2.5 to 2.7 Å) ...";
 *  - DM_EAD 0 1 1 Ta Ta: no FAIL (the (Ta, Ta) pairs reach 3.3 A);
 *  - DM_mJoin of the two (CGRIDMB 3.1 3.1, SGRIDMB 100 100, RCUTMB 3.5
 *    3.5): one FAIL, the W W descriptor's, naming CGRIDMB(2).
 * Before the fix: the channel was judged by the distances of every element
 * pair (2.4 to 3.3 A), which hold 3.1 A: no FAIL. */
TEST_CASE("preflight_audit: a density channel is judged by the element pairs "
          "of its descriptor's line",
          "[preflight_audit][density_reach][line_pairs]") {
  const DataStats data = ta_w_distances();
  auto ead = [](const std::string& pairs) {
    Context c;
    c.add("INITMB", "true");
    c.add("TYPEMB", "DM_EAD 0 1 1 " + pairs);
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", 3.5);
    c.add("CGRIDMB", 3.1);
    c.add("SGRIDMB", 100.0);
    return c;
  };
  auto fails = [&](const Context& seed) {
    std::vector<std::string> out;
    for (const auto& m :
         run_preflight_soft(data, seed, AuditMode::Error).messages)
      if (m.level == AuditMessage::Level::Fail)
        out.push_back(m.descriptor_label + "." + m.param_key + ": " + m.text);
    return out;
  };
  const auto ww = fails(ead("W W"));
  REQUIRE(ww.size() == 1u);
  CHECK(ww[0].rfind("DM_EAD.CGRIDMB: CGRIDMB(1) = 3.1 with SGRIDMB(1) = 100",
                    0) == 0);
  CHECK(ww[0].find("is below 1e-6 at every (W, W) neighbour distance within "
                   "RCUTMB = 3.5 (2.5 to 2.7 Å)") != std::string::npos);
  CHECK(fails(ead("Ta Ta")).empty());

  Context joined;
  joined.add("INITMB", "true");
  joined.add("TYPEMB", "DM_mJoin");
  joined.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  joined.add("TYPEMB", "DM_EAD 0 1 1 W W");
  joined.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  joined.add("RCUTMB", "3.5 3.5");
  joined.add("CGRIDMB", "3.1 3.1");
  joined.add("SGRIDMB", "100 100");
  const auto j = fails(joined);
  INFO(j.size());
  REQUIRE(j.size() == 1u);
  CHECK(j[0].rfind("DM_EAD.CGRIDMB: CGRIDMB(2) = 3.1", 0) == 0);
  CHECK(j[0].find("(W, W)") != std::string::npos);
}

// ---------------------------------------------------------------------------
//  DESCRIPTOR blocks: a block is judged by the element pairs of its PAIRS,
//  and a descriptor placed among blocks on TYPEMB by those of its own line
// ---------------------------------------------------------------------------

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>

namespace {

/* A DESCRIPTOR block "ta" of a DM_Embed descriptor of the element pairs
 * @p pairs, whose density cutoff has RADIUS @p radius. */
std::string embed_block(const std::string& pairs, const std::string& radius) {
  return "DESCRIPTOR ta DM_Embed\n"
         "  PAIRS " + pairs + "\n"
         "  DENSITY\n"
         "    BASIS Blip5\n"
         "    FORM NONNEG\n"
         "    CENTRES 1.0 2.5 4.0\n"
         "    ETA 1.5 1.5 1.5\n"
         "    AMPLITUDES 0.4 0.2 0.1\n"
         "    CUTOFF\n"
         "      TYPE Cut_Poly\n"
         "      RADIUS " + radius + "\n"
         "    ENDCUTOFF\n"
         "  ENDDENSITY\n"
         "  EMBEDDING\n"
         "    BASIS Gaussian\n"
         "    CENTRES 0.0 0.5 1.0 1.5\n"
         "    WIDTHS 0.5 0.5 0.5 0.5\n"
         "    REFERENCE ZERO\n"
         "  ENDEMBEDDING\n"
         "ENDDESCRIPTOR\n";
}

/* The configuration of the lines @p text (ATOM Ta W, INITMB true before
 * them), as a training configuration file gives it. */
Context embed_seed(const std::string& text) {
  tadah::core::PeriodicTable::initialize();
  std::istringstream in("ATOM Ta W\nINIT2B false\nINITMB true\n" + text);
  return tadah::core::CoreUserConfigParser().parse(in);
}

/* The FAIL findings of the audit of @p seed on the Ta-W data
 * (ta_w_distances()), each "<label>.<key>: <text>"; "threw: <what>" alone
 * when the audit throws. */
std::vector<std::string> embed_fails(const Context& seed) {
  std::vector<std::string> out;
  try {
    for (const auto& m :
         run_preflight_soft(ta_w_distances(), seed, AuditMode::Error).messages)
      if (m.level == AuditMessage::Level::Fail)
        out.push_back(m.descriptor_label + "." + m.param_key + ": " + m.text);
  } catch (const std::exception& e) {
    return {std::string("threw: ") + e.what()};
  }
  return out;
}

}  // namespace

/* Pins: a DM_Embed block is judged by the neighbour distances of the
 * element pairs of its PAIRS on the Ta-W data (ta_w_distances(): (Ta, Ta)
 * from 2.86 A, (Ta, W) from 2.4 A, (W, W) from 2.5 A), a block alone on
 * TYPEMB (no TYPEMB line):
 *  - PAIRS Ta Ta, RADIUS 2.6: one FAIL, "DM_Embed.TA.DENSITY.CUTOFF.RADIUS:
 *    TA.DENSITY.CUTOFF.RADIUS = 2.6 is below the smallest (Ta, Ta) neighbour
 *    distance of the dataset (2.86 Å): the descriptor sums over no
 *    neighbour and is 0 for every atom.";
 *  - PAIRS W W, RADIUS 2.6: no FAIL.
 * Before the fix: passes as well (a pin of the judgement of a block by its
 * PAIRS, which no test on data of several elements covered: judged by every
 * element pair, PAIRS Ta Ta at 2.6 passed, its radius holding the (Ta, W)
 * distance of 2.4 A). */
TEST_CASE("preflight_audit: a DESCRIPTOR block is judged by the element "
          "pairs of its PAIRS",
          "[preflight_audit][cutoff][line_pairs][embed]") {
  CHECK(embed_fails(embed_seed(embed_block("Ta Ta", "2.6"))) ==
        std::vector<std::string>{
            "DM_Embed.TA.DENSITY.CUTOFF.RADIUS: TA.DENSITY.CUTOFF.RADIUS = "
            "2.6 is below the smallest (Ta, Ta) neighbour distance of the "
            "dataset (2.86 Å): the descriptor sums over no neighbour and is 0 "
            "for every atom."});
  CHECK(embed_fails(embed_seed(embed_block("W W", "2.6"))).empty());
}

/* Pins: a descriptor that TYPEMB places among DESCRIPTOR blocks is judged
 * by the element pairs of its own line and by its own values (the context
 * the model builds it from: its TYPEMB line, its RCTYPEMB and RCUTMB value
 * and its share of the list keys), on the Ta-W data, with the block of
 * PAIRS W W and RADIUS 2.6 (which passes) and DM_EAD 0 1 1 Ta Ta with
 * RCUTMB 2.6, CGRIDMB 2.9 and SGRIDMB 1, its TYPEMB lines:
 *  - "DM_EAD 0 1 1 Ta Ta" then "@ta"; "@ta" then "DM_EAD 0 1 1 Ta Ta";
 *    "DM_mJoin", "@ta", "DM_EAD 0 1 1 Ta Ta"; and "DM_mJoin", "DM_EAD 0 1 1
 *    Ta Ta", "@ta": each one FAIL, "DM_EAD.RCUTMB: RCUTMB = 2.6 is below the
 *    smallest (Ta, Ta) neighbour distance of the dataset (2.86 Å): the
 *    descriptor sums over no neighbour and is 0 for every atom.";
 *  - "DM_mJoin", "DM_EAD 0 1 1 Ta Ta", "@ta", "DM_EAD 0 1 1 W W" with RCUTMB
 *    3.5 2.45 (CGRIDMB 2.9 2.6, SGRIDMB 1 1): one FAIL, the second
 *    descriptor's own cutoff, "DM_EAD.RCUTMB: RCUTMB = 2.45 is below the
 *    smallest (W, W) neighbour distance of the dataset (2.5 Å): ...";
 *  - "DM_EAD 0 1 1 W W" then "@ta" with the block of PAIRS Ta Ta and RADIUS
 *    2.6: one FAIL, the block's, "DM_Embed.TA.DENSITY.CUTOFF.RADIUS: ...
 *    below the smallest (Ta, Ta) neighbour distance of the dataset (2.86
 *    Å): ...".
 * Before the fix: the element pairs of such a descriptor were read from the
 * whole TYPEMB key, its @ta and DM_mJoin values among them: the audit threw
 * ("the element types of a descriptor's line come in pairs, and the line
 * has 3 of them, Ta Ta @ta", "... 1 Ta Ta", "TYPEMB DM_mJoin takes 3
 * parameters before its element pairs; parameter 2, the number of CGRIDMB
 * values, must be a non-negative integer, not \"DM_EAD\": ..."), or judged
 * the descriptor by pairs that are not elements ("RCUTMB = 2.6: no (1, Ta)
 * or (Ta, @ta) pair of atoms of the dataset lies within it, ..."); the line
 * of two DM_EAD descriptors gave no FAIL, each judged by the first RCUTMB
 * value, 3.5, and by pairs of the whole key. */
TEST_CASE("preflight_audit: a descriptor placed among DESCRIPTOR blocks is "
          "judged by the element pairs of its own line",
          "[preflight_audit][cutoff][line_pairs][embed]") {
  const std::string keys =
      "RCTYPEMB Cut_Cos\nRCUTMB 2.6\nCGRIDMB 2.9\nSGRIDMB 1\n";
  const std::string ta_ta =
      "DM_EAD.RCUTMB: RCUTMB = 2.6 is below the smallest (Ta, Ta) neighbour "
      "distance of the dataset (2.86 Å): the descriptor sums over no "
      "neighbour and is 0 for every atom.";
  for (const char* lines :
       {"TYPEMB DM_EAD 0 1 1 Ta Ta\nTYPEMB @ta\n",
        "TYPEMB @ta\nTYPEMB DM_EAD 0 1 1 Ta Ta\n",
        "TYPEMB DM_mJoin\nTYPEMB @ta\nTYPEMB DM_EAD 0 1 1 Ta Ta\n",
        "TYPEMB DM_mJoin\nTYPEMB DM_EAD 0 1 1 Ta Ta\nTYPEMB @ta\n"}) {
    CAPTURE(lines);
    CHECK(embed_fails(embed_seed(lines + keys +
                                 embed_block("W W", "2.6"))) ==
          std::vector<std::string>{ta_ta});
  }
  CHECK(embed_fails(embed_seed(
            "TYPEMB DM_mJoin\nTYPEMB DM_EAD 0 1 1 Ta Ta\nTYPEMB @ta\n"
            "TYPEMB DM_EAD 0 1 1 W W\nRCTYPEMB Cut_Cos Cut_Cos\n"
            "RCUTMB 3.5 2.45\nCGRIDMB 2.9 2.6\nSGRIDMB 1 1\n" +
            embed_block("W W", "2.6"))) ==
        std::vector<std::string>{
            "DM_EAD.RCUTMB: RCUTMB = 2.45 is below the smallest (W, W) "
            "neighbour distance of the dataset (2.5 Å): the descriptor sums "
            "over no neighbour and is 0 for every atom."});
  {
    const std::vector<std::string> fails = embed_fails(embed_seed(
        "TYPEMB DM_EAD 0 1 1 W W\nTYPEMB @ta\n" + keys +
        embed_block("Ta Ta", "2.6")));
    CHECK(fails ==
          std::vector<std::string>{
              "DM_Embed.TA.DENSITY.CUTOFF.RADIUS: TA.DENSITY.CUTOFF.RADIUS = "
              "2.6 is below the smallest (Ta, Ta) neighbour distance of the "
              "dataset (2.86 Å): the descriptor sums over no neighbour and is "
              "0 for every atom."});
  }
}
