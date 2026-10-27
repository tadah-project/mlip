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
 * flagged whatever the search space ("Critical hyperparameter is fixed in
 * the seed config and is not in any OPTIM block ..."), also for a search
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
 * origin is not flagged), three findings.
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
  std::vector<std::string> alone = findings_of(
      run_critical_optim_audit(single_ead_seed("0"), space));
  const std::vector<std::string> second = findings_of(
      run_critical_optim_audit(single_ead_seed("2.9"), space));
  alone.insert(alone.end(), second.begin(), second.end());
  std::sort(alone.begin(), alone.end());
  const std::vector<std::string> together =
      findings_of(run_critical_optim_audit(joined, space));
  CAPTURE(alone, together);
  CHECK(together == alone);
  CHECK(together.size() == 3u);
}

namespace {

/* The findings of @p rep as "label.key", sorted. */
std::vector<std::string> keys_of(const AuditReport& rep) {
  std::vector<std::string> out;
  for (const auto& m : rep.messages)
    out.push_back(m.descriptor_label + "." + m.param_key);
  std::sort(out.begin(), out.end());
  return out;
}

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
 * and that of a bump channel under a hard cutoff, only where tadah hpo
 * derives it from the largest knot or holds the bump supports within it:
 * on a line of that descriptor alone. For a search of CGRID2B:
 *  - D2_Knot5 alone (Cut_Dummy) and D2_Blip alone under Cut_Dummy give no
 *    finding, and D2_Blip alone under Cut_Cos gives D2_Blip.RCUT2B;
 *  - D2_mJoin of D2_Knot5 and D2_Blip gives D2_Blip.RCUT2B and
 *    D2_Knot5.RCUT2B under Cut_Cos Cut_Cos and under Cut_Dummy Cut_Dummy:
 *    tadah hpo neither derives the cutoff of a knot channel of a joined
 *    line nor holds its bump supports within the cutoff, so the cutoffs of
 *    a joined line are fixed values of the search.
 * Before the fix: each descriptor of the joined line was judged as if it
 * were alone: Cut_Cos Cut_Cos gave D2_Blip.RCUT2B only, and Cut_Dummy
 * Cut_Dummy no finding. */
TEST_CASE("preflight_audit: critical_optim_audit leaves out a knot or bump "
          "cutoff only on a line of that descriptor alone",
          "[preflight_audit][join][knot]") {
  const std::vector<std::string> space = {"CGRID2B"};
  CHECK(keys_of(run_critical_optim_audit(knot_alone_seed(), space)).empty());
  CHECK(keys_of(run_critical_optim_audit(blip_alone_seed("Cut_Dummy"), space))
            .empty());
  CHECK(keys_of(run_critical_optim_audit(blip_alone_seed("Cut_Cos"), space)) ==
        std::vector<std::string>{"D2_Blip.RCUT2B"});
  for (const std::string rctype : {"Cut_Cos Cut_Cos", "Cut_Dummy Cut_Dummy"}) {
    CAPTURE(rctype);
    CHECK(keys_of(run_critical_optim_audit(knot_blip_join_seed(rctype),
                                           space)) ==
          std::vector<std::string>{"D2_Blip.RCUT2B", "D2_Knot5.RCUT2B"});
  }
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
