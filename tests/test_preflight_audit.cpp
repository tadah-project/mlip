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
//     DM_mBlipEnv cross_param_check (the BASINS regression).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/preflight_audit.h>
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

  // ρ-collapse cross-param check (the BASINS test).
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

TEST_CASE("preflight_audit: critical_optim_audit flags BASINS-style fixed CGRIDMB", "[preflight_audit]") {
  // BASINS-style minimal seed: fixed CGRIDMB, RCUTMB, SGRIDMB and no
  // OPTIM coverage. We supply the descriptor meta directly so the
  // test does not depend on dm_all.cpp's static registrations being
  // live in the test binary.
  Context seed;
  seed.add<double>("CGRIDMB", 0.0);
  seed.add<double>("SGRIDMB", 0.2);
  seed.add<double>("RCUTMB", 5.6);

  std::vector<MetaPair> metas;
  metas.emplace_back("DM_mBlipEnv", make_dm_mblipenv_like_meta());

  // No OPTIM keys at all — every Critical fixed param should be flagged.
  std::vector<std::string> no_optim;
  auto rep = run_critical_optim_audit_with_metas(seed, no_optim, metas);

  // We expect at least CGRIDMB to be flagged (the BASINS smoking gun).
  bool saw_cgridmb = false;
  bool saw_rcutmb  = false;
  for (const auto& m : rep.messages) {
    if (m.param_key == "CGRIDMB") saw_cgridmb = true;
    if (m.param_key == "RCUTMB")  saw_rcutmb  = true;
    REQUIRE(m.level == AuditMessage::Level::Warn);
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
  // ρ collapsed near zero across percentiles — the BASINS signature.
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
