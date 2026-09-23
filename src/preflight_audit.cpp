#include <tadah/mlip/preflight_audit.h>

#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tadah/core/logger.h>
#include <tadah/core/registry.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace mlip {

namespace {

const char* level_str(AuditMessage::Level l) {
  switch (l) {
    case AuditMessage::Level::Ok:   return "OK";
    case AuditMessage::Level::Warn: return "WARN";
    case AuditMessage::Level::Fail: return "FAIL";
  }
  return "?";
}

// Walk seed for TYPE2B / TYPEMB labels and instantiate via the no-arg
// D_Base factory (the only path that covers both D2_* and DM_*).
// Returns the labels paired with their meta() — descriptors that fail
// to instantiate or whose meta() is nullopt are silently skipped (the
// audit is best-effort across the configured descriptor set).
std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>
collect_descriptor_metas(const tadah::core::Context& seed) {
  std::vector<std::pair<std::string, tadah::models::DescriptorMeta>> out;
  for (const char* type_key : {"TYPE2B", "TYPEMB"}) {
    if (!seed.contains(type_key)) continue;
    const std::size_t n = seed.size(type_key);
    for (std::size_t i = 0; i < n; ++i) {
      std::string label;
      try {
        label = seed.get<std::string>(type_key, i);
      } catch (...) { continue; }
      try {
        std::unique_ptr<tadah::models::D_Base> obj(
            tadah::core::factory<tadah::models::D_Base>(label));
        auto m = obj->meta();
        if (m.has_value()) out.emplace_back(label, std::move(*m));
      } catch (...) {
        // Descriptor class not registered with no-arg factory or meta()
        // not yet implemented — skip; suggester / audit cover other
        // descriptors uniformly.
      }
    }
  }
  return out;
}

}  // namespace

AuditMode parse_audit_mode(const std::string& s) {
  std::string v;
  v.reserve(s.size());
  for (char c : s) v.push_back(static_cast<char>(::tolower(c)));
  if (v == "off")   return AuditMode::Off;
  if (v == "warn")  return AuditMode::Warn;
  if (v == "error") return AuditMode::Error;
  throw std::runtime_error(
      "Invalid AUDIT value '" + s +
      "'. Expected one of: off, warn, error.");
}

AuditMode audit_mode_from_context(const tadah::core::Context& ctx) {
  if (!ctx.contains("AUDIT")) return AuditMode::Off;
  return parse_audit_mode(ctx.get<std::string>("AUDIT"));
}

bool AuditReport::any_fail() const noexcept {
  for (const auto& m : messages)
    if (m.level == AuditMessage::Level::Fail) return true;
  return false;
}

bool AuditReport::any_warn() const noexcept {
  for (const auto& m : messages)
    if (m.level == AuditMessage::Level::Warn) return true;
  return false;
}

void AuditReport::print(std::ostream& os) const {
  if (messages.empty()) {
    os << "# pre-flight audit: no findings\n";
    return;
  }
  os << "# pre-flight audit findings\n";
  for (const auto& m : messages) {
    os << "  [" << level_str(m.level) << "] "
       << m.descriptor_label;
    if (!m.param_key.empty()) os << "." << m.param_key;
    os << " — " << m.text << "\n";
  }
}

void AuditReport::log() const {
  for (const auto& m : messages) {
    if (m.level == AuditMessage::Level::Ok) continue;
    std::string prefix = "preflight_audit: [" +
                         std::string(level_str(m.level)) + "] " +
                         m.descriptor_label;
    if (!m.param_key.empty()) prefix += "." + m.param_key;
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        prefix + " — " + m.text);
  }
}

namespace {

AuditReport run_preflight_with_metas_impl(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas,
    AuditMode                            mode) {
  AuditReport report;
  if (mode == AuditMode::Off) return report;
  tadah::models::SeedView sv(seed);

  for (const auto& [label, meta] : metas) {
    // Per-parameter HealthChecks.
    for (const auto& pm : meta.params) {
      if (!pm.check) continue;
      // Skip if the seed lacks the param entirely — the per-param
      // check expects the value to exist (the cross_param_checks
      // handle absent-but-derivable cases).
      if (!seed.contains(pm.key)) continue;

      auto v = pm.check(data, sv);
      if (tadah::models::is_ok(v)) continue;
      AuditMessage msg;
      msg.descriptor_label = label;
      msg.param_key        = pm.key;
      msg.text             = tadah::models::verdict_message(v);
      msg.level = tadah::models::is_fail(v)
                  ? AuditMessage::Level::Fail
                  : AuditMessage::Level::Warn;
      report.messages.push_back(std::move(msg));
    }

    // Whole-descriptor cross-parameter checks (e.g. ρ-collapse).
    for (const auto& chk : meta.cross_param_checks) {
      if (!chk) continue;
      auto v = chk(data, sv);
      if (tadah::models::is_ok(v)) continue;
      AuditMessage msg;
      msg.descriptor_label = label;
      msg.text             = tadah::models::verdict_message(v);
      msg.level = tadah::models::is_fail(v)
                  ? AuditMessage::Level::Fail
                  : AuditMessage::Level::Warn;
      report.messages.push_back(std::move(msg));
    }
  }
  return report;
}

}  // namespace

AuditReport run_preflight_soft(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    AuditMode                            mode) {
  if (mode == AuditMode::Off) return {};
  const auto metas = collect_descriptor_metas(seed);
  return run_preflight_with_metas_impl(data, seed, metas, mode);
}

AuditReport run_preflight_soft_with_metas(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas,
    AuditMode                            mode) {
  return run_preflight_with_metas_impl(data, seed, metas, mode);
}

AuditReport run_preflight_from_db(
    StructureDB&                stdb,
    const tadah::core::Context& seed,
    AuditMode                   mode) {
  AuditReport report;
  if (mode == AuditMode::Off) return report;
  // Default: run the cheap stats (no per-descriptor ρ percentiles).
  // Cross-parameter checks that require ρ percentiles will degrade
  // gracefully (they look up the descriptor in
  // `rho_percentiles_per_descriptor` and return Ok when absent).
  // compute_data_stats takes a mutable context (it may set RCUTMAX or
  // run NN finder). The audit must not mutate the seed itself, so use
  // a working copy.
  tadah::core::Context work = seed;

  // AUDIT_SAMPLE bounds the cost of the dataset scan. 0 means "use
  // every structure"; the blueprint default is 256, which is enough
  // to drive percentile-based health checks without a full-DB NN
  // sweep on large training sets. A negative value (entered manually)
  // is treated as "use all" for safety.
  std::size_t sample = 256;
  if (seed.contains("AUDIT_SAMPLE")) {
    const int v = seed.get<int>("AUDIT_SAMPLE");
    sample = (v <= 0) ? 0 : static_cast<std::size_t>(v);
  }

  auto data = compute_data_stats(stdb, work,
                                 /*with_descriptor_stats=*/false,
                                 sample);
  return run_preflight_soft(data, seed, mode);
}

AuditReport run_critical_optim_audit_with_metas(
    const tadah::core::Context&     seed,
    const std::vector<std::string>& optim_keys,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas) {
  AuditReport report;
  if (metas.empty()) return report;

  std::unordered_set<std::string> optim_set(optim_keys.begin(),
                                            optim_keys.end());

  for (const auto& [label, meta] : metas) {
    for (const auto& pm : meta.params) {
      if (pm.crit != tadah::models::Criticality::Critical) continue;
      if (!seed.contains(pm.key)) continue;        // not configured
      if (optim_set.count(pm.key) > 0) continue;   // covered by HPO
      // Knot channels: RCUT2B/RCUTMB are DERIVED from the largest knot and
      // re-derived on every HPO evaluation — recommending an OPTIM block on
      // them would contradict that contract (the search knob is CGRID*).
      if ((pm.key == "RCUT2B" &&
           tadah::models::potedit::derived_knot_rcut(seed, "2B")) ||
          (pm.key == "RCUTMB" &&
           tadah::models::potedit::derived_knot_rcut(seed, "MB")))
        continue;
      // Bump channels (D2_Blip/D2_Blip5, DM_REAM bump density) under a HARD
      // cutoff: RCUT is the fitted truncation and every bump support ends at
      // or before it (the bump-support gate keeps it so), so a larger cutoff
      // changes no prediction and a smaller one only trips the gate --
      // searching it is a flat dimension with a cliff, and `tadah hpo`
      // already warns when someone does. The range is searched through the
      // bump centres/widths (CGRID*/SGRID*); the per-candidate bump-support
      // gate in tadah hpo is what keeps them consistent with RCUT. Under a
      // SMOOTH cutoff the damping tail depends on RCUT, so the warning stays.
      if ((pm.key == "RCUT2B" &&
           tadah::models::potedit::bump_channel_halfwidth(seed, "2B") &&
           tadah::models::potedit::channel_cutoff_is_discontinuous(seed, "2B")) ||
          (pm.key == "RCUTMB" &&
           tadah::models::potedit::bump_channel_halfwidth(seed, "MB") &&
           tadah::models::potedit::channel_cutoff_is_discontinuous(seed, "MB")))
        continue;

      AuditMessage msg;
      msg.descriptor_label = label;
      msg.param_key        = pm.key;
      msg.text =
          "Critical hyperparameter is fixed in the seed config and is "
          "not in any OPTIM block. If the seed value misses the "
          "neighbour distances of the training data (e.g. a cutoff or "
          "radial grid that does not overlap the neighbour shells), the "
          "descriptor block can collapse silently. "
          "Consider adding `OPTIM " + pm.key + " (...) <lo> <hi>` "
          "from `tadah analysis suggest_bounds`.";
      msg.level = AuditMessage::Level::Warn;
      report.messages.push_back(std::move(msg));
    }
  }
  return report;
}

AuditReport run_critical_optim_audit(
    const tadah::core::Context&     seed,
    const std::vector<std::string>& optim_keys) {
  return run_critical_optim_audit_with_metas(
      seed, optim_keys, collect_descriptor_metas(seed));
}

void enforce_audit_mode(const AuditReport& report, AuditMode mode) {
  if (mode != AuditMode::Error) return;
  if (!report.any_fail())       return;
  std::string out =
      "preflight_audit: AUDIT=error and one or more FAIL findings:";
  for (const auto& m : report.messages) {
    if (m.level != AuditMessage::Level::Fail) continue;
    out += "\n  - " + m.descriptor_label;
    if (!m.param_key.empty()) out += "." + m.param_key;
    out += ": " + m.text;
  }
  throw std::runtime_error(out);
}

}  // namespace mlip
}  // namespace tadah
