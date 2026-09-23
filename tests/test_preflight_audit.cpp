// Catch2 v2 unit tests for the pre-flight audit (Phase 5).
//
// Verifies:
//   * parse_audit_mode handles warn / error / off case-insensitively;
//     throws on garbage.
//   * audit_mode_from_context defaults to Off when AUDIT is absent.
//   * AuditReport::any_fail / any_warn / log() are sane.
//   * enforce_audit_mode throws iff (mode==Error && any_fail()).
//   * run_critical_optim_audit flags Critical seed params absent from
//     the OPTIM key set, ignores those that are covered.
//   * run_preflight_soft on a synthetic ρ-collapse data set fires the
//     DM_mBlipEnv cross_param_check (the silent ρ-collapse regression).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/models/eam/pot_curve_edit.h>
#include <tadah/models/descriptors/param_meta.h>

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

TEST_CASE("preflight_audit: critical_optim_audit flags a fixed CGRIDMB left out of OPTIM", "[preflight_audit]") {
  // Minimal seed: fixed CGRIDMB, RCUTMB, SGRIDMB and no OPTIM coverage.
  // We supply the descriptor meta directly so the test does not depend
  // on dm_all.cpp's static registrations being live in the test binary.
  Context seed;
  seed.add<double>("CGRIDMB", 0.0);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  // No OPTIM keys at all — every Critical fixed param should be flagged.
  std::vector<std::string> no_optim;
  auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);

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
  std::vector<std::string> with_optim = {"CGRIDMB", "RCUTMB", "SGRIDMB"};
  auto rep2 = run_critical_optim_audit_with_metas(seed, with_optim, metas);
  for (const auto& m : rep2.messages) {
    REQUIRE(m.param_key != "CGRIDMB");
    REQUIRE(m.param_key != "RCUTMB");
    REQUIRE(m.param_key != "SGRIDMB");
  }
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
  std::vector<MetaPair> metas;
  metas.emplace_back("D2_Blip5", make_bump_pair_like_meta("D2_Blip5"));
  metas.emplace_back("DM_REAM",  make_dm_ream_like_meta());
  const std::vector<std::string> no_optim;
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
