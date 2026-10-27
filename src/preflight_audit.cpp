#include <tadah/mlip/preflight_audit.h>

#include <algorithm>
#include <tadah/core/number_format.h>

#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tadah/core/logger.h>
#include <tadah/core/registry.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/joined_descriptors.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/dm/dm_all.h>
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

/* A descriptor of the seed with the context its parameters are read from:
 * the seed for a single descriptor, a descriptor's own context for one of
 * a joined line (joined_descriptors()), whose entries the messages number
 * as the whole key numbers them (SeedView::entry_number()), and the element
 * pairs its line names, by whose neighbour statistics it is judged (none:
 * metadata given without a line, judged by every pair of the data). */
struct DescriptorCheck {
  std::string label;
  tadah::models::DescriptorMeta meta;
  tadah::core::Context context;
  std::map<std::string, std::size_t> first_entries;
  std::optional<tadah::models::LinePairs> pairs;
};

// Walk seed for TYPE2B / TYPEMB labels and instantiate via the no-arg
// D_Base factory (the only path that covers both D2_* and DM_*), and the
// DESCRIPTOR blocks of the seed. Returns each descriptor with its meta()
// and the element pairs of its line (D_Base::get_init_atoms() of the seed:
// a TYPE key of one descriptor, the joined lines and the TYPEMB key of
// blocks being collect_descriptor_checks()'s; a block's PAIRS) —
// descriptors that fail to instantiate or whose meta() is nullopt are
// silently skipped (the audit is best-effort across the configured
// descriptor set).
std::vector<DescriptorCheck>
collect_descriptor_metas(const tadah::core::Context& seed) {
  std::vector<DescriptorCheck> out;
  for (const char* type_key : {"TYPE2B", "TYPEMB"}) {
    if (!seed.contains(type_key)) continue;
    const std::size_t n = seed.size(type_key);
    for (std::size_t i = 0; i < n; ++i) {
      std::string label;
      try {
        label = seed.get<std::string>(type_key, i);
      } catch (...) { continue; }
      std::unique_ptr<tadah::models::D_Base> obj;
      std::optional<tadah::models::DescriptorMeta> m;
      try {
        obj.reset(tadah::core::factory<tadah::models::D_Base>(label));
        m = obj->meta();
      } catch (...) {
        // Descriptor class not registered with no-arg factory or meta()
        // not yet implemented — skip; suggester / audit cover other
        // descriptors uniformly.
        continue;
      }
      if (!m.has_value()) continue;
      out.push_back(DescriptorCheck{
          label, std::move(*m), seed, {},
          tadah::models::LinePairs(obj->get_init_atoms(seed))});
    }
  }
  if(!seed.blocks("DESCRIPTOR").empty())
    for(const auto &child:models::embed_contexts(seed))if(!child.blocks("DESCRIPTOR").empty()) {
      std::unique_ptr<models::DM_Base> descriptor(core::factory<models::DM_Base,const core::Context&>(child.get<std::string>("TYPEMB"),child));
      if(auto meta=descriptor->meta()) {
        std::string label = meta->label;
        out.push_back(DescriptorCheck{
            std::move(label), std::move(*meta), seed, {},
            tadah::models::LinePairs(descriptor->get_init_atoms(child))});
      }
    }
  return out;
}

/* The descriptors of the seed as collect_descriptor_metas() finds them,
 * with each descriptor of a joined TYPE2B or TYPEMB line on its own
 * context: a joined line's descriptors share its RCUT values and its list
 * keys, each taking its own (D_mJoin::parse_context()); and so each
 * descriptor that a TYPEMB key places among DESCRIPTOR blocks
 * (block_line_descriptors(): the model's context of each), whose element
 * pairs are those of its own line, not of the whole key. */
std::vector<DescriptorCheck>
collect_descriptor_checks(const tadah::core::Context& seed) {
  std::vector<DescriptorCheck> out;
  tadah::core::Context single = seed;
  for (const char* type_key : {"TYPE2B", "TYPEMB"}) {
    const bool blocks = std::string(type_key) == "TYPEMB" &&
                        !seed.blocks("DESCRIPTOR").empty();
    if (!blocks && !is_joined_line(seed, type_key)) continue;
    for (auto& jd : blocks ? block_line_descriptors(seed)
                           : joined_descriptors(seed, type_key)) {
      std::unique_ptr<tadah::models::D_Base> obj(
          tadah::core::factory<tadah::models::D_Base>(jd.label));
      auto m = obj->meta();
      if (!m.has_value()) continue;
      tadah::models::LinePairs pairs(obj->get_init_atoms(jd.context));
      out.push_back(DescriptorCheck{jd.label, std::move(*m),
                                    std::move(jd.context),
                                    std::move(jd.first_entries),
                                    std::move(pairs)});
    }
    // The blocks of a TYPEMB key stay: collect_descriptor_metas() finds
    // them without it.
    single.remove(type_key);
  }
  // The descriptors of the other lines, each with the seed as its context.
  for (auto& c : collect_descriptor_metas(single)) {
    c.context = seed;
    out.push_back(std::move(c));
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
    os << "  " << level_str(m.level) << " " << m.descriptor_label;
    if (!m.param_key.empty()) os << "." << m.param_key;
    os << ": " << m.text << "\n";
  }
}

void AuditReport::log() const {
  for (const auto& m : messages) {
    if (m.level == AuditMessage::Level::Ok) continue;
    // "AUDIT finding WARN D2_Blip.RCUT2B: ...": the key that runs the
    // audit, the level of the finding and its parameter.
    std::string prefix = "AUDIT finding " +
                         std::string(level_str(m.level)) + " " +
                         m.descriptor_label;
    if (!m.param_key.empty()) prefix += "." + m.param_key;
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        prefix + ": " + m.text);
  }
}

namespace {

AuditReport run_preflight_checks(
    const tadah::core::stats::DataStats&  data,
    const std::vector<DescriptorCheck>&   checks,
    AuditMode                             mode) {
  AuditReport report;
  if (mode == AuditMode::Off) return report;

  for (const auto& c : checks) {
    // A descriptor sums over the neighbours of the element pairs its line
    // names: its checks read their statistics, judge each element pair the
    // line stands for on its own (the reach of a cutoff, or of the channels
    // of a density), and name them.
    const tadah::models::SeedView sv =
        c.pairs ? tadah::models::SeedView(c.context, c.first_entries, *c.pairs)
                : tadah::models::SeedView(c.context, c.first_entries);
    const tadah::core::stats::DataStats line_data =
        c.pairs ? c.pairs->stats(data) : data;
    // Per-parameter HealthChecks.
    for (const auto& pm : c.meta.params) {
      if (!pm.check) continue;
      // Skip if the seed lacks the param entirely — the per-param
      // check expects the value to exist (the cross_param_checks
      // handle absent-but-derivable cases).
      if (!sv.contains(pm.key)) continue;

      auto v = pm.check(line_data, sv);
      if (tadah::models::is_ok(v)) continue;
      AuditMessage msg;
      msg.descriptor_label = c.label;
      msg.param_key        = pm.key;
      msg.text             = tadah::models::verdict_message(v);
      msg.level = tadah::models::is_fail(v)
                  ? AuditMessage::Level::Fail
                  : AuditMessage::Level::Warn;
      report.messages.push_back(std::move(msg));
    }

    // Whole-descriptor cross-parameter checks (e.g. ρ-collapse).
    for (const auto& chk : c.meta.cross_param_checks) {
      if (!chk) continue;
      auto v = chk(line_data, sv);
      if (tadah::models::is_ok(v)) continue;
      AuditMessage msg;
      msg.descriptor_label = c.label;
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
  return run_preflight_checks(data, collect_descriptor_checks(seed), mode);
}

AuditReport run_preflight_soft_with_metas(
    const tadah::core::stats::DataStats& data,
    const tadah::core::Context&          seed,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas,
    AuditMode                            mode) {
  if (mode == AuditMode::Off) return {};
  std::vector<DescriptorCheck> checks;
  for (const auto& [label, meta] : metas)
    checks.push_back(DescriptorCheck{label, meta, seed, {}, std::nullopt});
  return run_preflight_checks(data, checks, mode);
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
  // is treated as "use all" for safety. A FAIL of the sample is judged on
  // every structure (below).
  std::size_t sample = 256;
  if (seed.contains("AUDIT_SAMPLE")) {
    const int v = seed.get<int>("AUDIT_SAMPLE");
    sample = (v <= 0) ? 0 : static_cast<std::size_t>(v);
  }

  auto data = compute_data_stats(stdb, work,
                                 /*with_descriptor_stats=*/false,
                                 sample);
  report = run_preflight_soft(data, seed, mode);
  // A FAIL refuses the run under AUDIT error, and its text speaks of the
  // dataset ("no (W, W) pair of atoms of the dataset lies within it"): a
  // FAIL of a sample of the structures, which may leave out the only
  // structures of a pair, is judged again on every structure.
  if (report.any_fail() && sample > 0 && sample < stdb.size()) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::INFO,
        "pre-flight audit: the scan of " + std::to_string(sample) +
            " of the " + std::to_string(stdb.size()) +
            " structures (AUDIT_SAMPLE " + std::to_string(sample) +
            ") gave a FAIL finding, which is judged on every structure");
    data = compute_data_stats(stdb, work, /*with_descriptor_stats=*/false,
                              /*sample_size=*/0);
    report = run_preflight_soft(data, seed, mode);
  }
  return report;
}

namespace {
/* Whether every value of the radial-centre key `key` of `seed` is the
 * number 0 (a grid generator such as LIN is not). */
bool centres_at_origin(const tadah::core::Context& seed,
                       const std::string& key) {
  const std::size_t n = seed.size(key);
  if (n == 0) return false;
  for (std::size_t i = 0; i < n; ++i) {
    double v = 0.0;
    if (!tadah::core::parse_number(seed.get<std::string>(key, i), v) ||
        v != 0.0)
      return false;
  }
  return true;
}
}  // namespace

namespace {

/* The search space as the audit reads it: every entry of a named key
 * (run_critical_optim_audit with key names), or the entries flagged for
 * each key (with the searched entries, as HPOTargets::optimap's opt). */
struct SearchSpace {
  std::unordered_set<std::string> names;
  const std::map<std::string, std::vector<bool>>* entries = nullptr;

  /* Whether the search holds one of the entries of @p key that the
   * descriptor @p c has (@p sv): any entry for a line of one descriptor,
   * its own (SeedView::entry_number) for a descriptor of a joined line. */
  bool holds(const DescriptorCheck& c, const tadah::models::SeedView& sv,
             const std::string& key) const {
    if (entries == nullptr) return names.count(key) > 0;
    const auto it = entries->find(key);
    if (it == entries->end()) return false;
    const std::vector<bool>& flags = it->second;
    if (c.first_entries.empty())
      return std::find(flags.begin(), flags.end(), true) != flags.end();
    for (std::size_t i = 0; i < sv.size(key); ++i) {
      const std::size_t e = sv.entry_number(key, i) - 1;
      if (e < flags.size() && flags[e]) return true;
    }
    return false;
  }
};

/* The critical-OPTIM audit of @p checks against the search @p space, each
 * descriptor judged by its own values of a key (every centre at the
 * origin, in its own context) and by its own entries of the search; the
 * cutoff of a knot channel, and of a bump channel under a hard cutoff, is
 * left out where tadah hpo derives it or holds the bump supports within
 * it: on a line of one such descriptor and for such a descriptor of a
 * joined line (D2_mJoin, DM_mJoin), whose own RCUT entry tadah hpo derives
 * or holds (derived_knot_rcut and bump_channel_halfwidth on the
 * descriptor's own context). */
AuditReport critical_optim_audit(const std::vector<DescriptorCheck>& checks,
                                 const SearchSpace& space) {
  AuditReport report;

  for (const auto& c : checks) {
    const auto& meta = c.meta;
    const tadah::core::Context& ctx = c.context;
    const tadah::models::SeedView sv(ctx, c.first_entries);
    // A descriptor with no entry of a key in the search space is not
    // searched: its fixed keys are the seed's choice, and no warning is
    // given about them (a search of FIXWEIGHT alone, or of another
    // descriptor's keys or entries).
    const bool searched =
        std::any_of(meta.params.begin(), meta.params.end(),
                    [&](const tadah::models::ParamMeta& pm) {
                      return space.holds(c, sv, pm.key);
                    });
    if (!searched) continue;
    for (const auto& pm : meta.params) {
      if (pm.crit != tadah::models::Criticality::Critical) continue;
      if (!sv.contains(pm.key)) continue;           // not configured
      if (space.holds(c, sv, pm.key)) continue;      // covered by HPO
      // Knot channels: RCUT2B/RCUTMB are DERIVED from the largest knot and
      // re-derived on every HPO evaluation — recommending an OPTIM block on
      // them would contradict that contract (the search knob is CGRID*).
      // tadah hpo derives them for a line of one knot descriptor and for
      // the own RCUT entry of a knot descriptor of a joined line.
      if ((pm.key == "RCUT2B" &&
           tadah::models::potedit::derived_knot_rcut(ctx, "2B")) ||
          (pm.key == "RCUTMB" &&
           tadah::models::potedit::derived_knot_rcut(ctx, "MB")))
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
      // The gate holds the bumps of a line of one bump descriptor and of a
      // bump descriptor of a joined line, within its own RCUT entry.
      if ((pm.key == "RCUT2B" &&
           tadah::models::potedit::bump_channel_halfwidth(ctx, "2B") &&
           tadah::models::potedit::channel_cutoff_is_discontinuous(ctx, "2B")) ||
          (pm.key == "RCUTMB" &&
           tadah::models::potedit::bump_channel_halfwidth(ctx, "MB") &&
           tadah::models::potedit::channel_cutoff_is_discontinuous(ctx, "MB")))
        continue;

      // Radial centres at the origin (every CGRID* value 0): the basis is
      // largest at r = 0 and falls off with distance, so its width
      // (SGRID*), not its centre, decides whether it reaches the neighbour
      // shells (a compact blip reaches 2/eta); a fixed width is audited on
      // its own.
      if (pm.key.rfind("CGRID", 0) == 0 && centres_at_origin(ctx, pm.key))
        continue;

      // The descriptor's entries of the key, as an OPTIM line numbers them
      // (SeedView::entry_number: those of a joined line's descriptor follow
      // the entries of the descriptors before it): "(2)", "(1-7)".
      const std::size_t n = sv.size(pm.key);
      const std::size_t first = sv.entry_number(pm.key, 0);
      const std::size_t last = sv.entry_number(pm.key, n > 0 ? n - 1 : 0);
      const std::string entries =
          first == last ? std::to_string(first)
                        : std::to_string(first) + "-" + std::to_string(last);
      AuditMessage msg;
      msg.descriptor_label = c.label;
      msg.param_key        = pm.key;
      msg.text =
          pm.key + "(" + entries + ") is a critical parameter that the seed "
          "fixes and no OPTIM line searches: a seed value that misses the "
          "neighbour distances of the training data can make the descriptor "
          "0 for every atom. `tadah analysis suggest_bounds` suggests a "
          "range for an `OPTIM " + pm.key + " (" + entries + ")` line.";
      msg.level = AuditMessage::Level::Warn;
      report.messages.push_back(std::move(msg));
    }
  }
  return report;
}

}  // namespace

AuditReport run_critical_optim_audit_with_metas(
    const tadah::core::Context&     seed,
    const std::vector<std::string>& optim_keys,
    const std::vector<std::pair<std::string, tadah::models::DescriptorMeta>>& metas) {
  std::vector<DescriptorCheck> checks;
  for (const auto& [label, meta] : metas)
    checks.push_back(DescriptorCheck{label, meta, seed, {}, std::nullopt});
  SearchSpace space;
  space.names.insert(optim_keys.begin(), optim_keys.end());
  return critical_optim_audit(checks, space);
}

AuditReport run_critical_optim_audit(
    const tadah::core::Context&     seed,
    const std::vector<std::string>& optim_keys) {
  SearchSpace space;
  space.names.insert(optim_keys.begin(), optim_keys.end());
  return critical_optim_audit(collect_descriptor_checks(seed), space);
}

AuditReport run_critical_optim_audit(
    const tadah::core::Context& seed,
    const std::map<std::string, std::vector<bool>>& searched_entries) {
  SearchSpace space;
  space.entries = &searched_entries;
  return critical_optim_audit(collect_descriptor_checks(seed), space);
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
