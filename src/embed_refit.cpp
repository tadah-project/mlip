#include <cmath>
#include <set>
#include <sstream>
#include <tadah/core/number_format.h>
#include <tadah/mlip/engines/embed_refit.h>
#include <tadah/models/descriptors/d_base.h>

namespace tadah {
namespace mlip {
namespace engines {
namespace {
[[noreturn]] void bad(const std::string &s) {
  throw std::runtime_error("tadah refit: " + s);
}
std::vector<std::string> words(const std::string &s) {
  std::istringstream in(s.substr(0, s.find('#')));
  std::string t;
  std::vector<std::string> out;
  while (in >> t)
    out.push_back(t);
  return out;
}
std::string line_values(const std::string &key, const std::vector<double> &v) {
  std::string out = key;
  for (double x : v)
    out += " " + core::format_roundtrip(x);
  return out;
}
} // namespace
void validate_embed_refit(const core::Context &c) {
  if (c.blocks("DESCRIPTOR").size() != 1)
    bad("single-element refit accepts exactly one DESCRIPTOR block");
  for (const char *key :
       {"REFIT_BASIS", "REFIT_CONSTRAIN_F0", "CGRIDMB", "SGRIDMB", "AMPGRIDMB",
        "CEMBFUNC", "SEMBFUNC", "KDEGMB", "KDEGEMB", "RHOFORM", "EMBF0",
        "RCUTMB", "RCTYPEMB"})
    if (c.has_stored(key))
      bad(std::string(key) + " conflicts with component blocks; declare the "
                             "setting in its component");
  // A component fit fits the basis its block declares: it does not
  // escalate (the dense fit's escalation grows the global NRHO_BASIS and
  // NF_BASIS counts, which component grids replace).
  if (c.has_stored("REFIT_MAX_ESCALATIONS"))
    bad("REFIT_MAX_ESCALATIONS has no effect with component blocks: a "
        "component fit does not escalate its basis; its size is the "
        "component's DENSITY and EMBEDDING CENTRES (CENTRES AUTO <count>, or "
        "an explicit grid), so remove REFIT_MAX_ESCALATIONS and give the "
        "component more functions there");
  if (c.has_stored("REFIT_PAIR_BASIS"))
    (void)models::basis_kind_from_string(
        c.get<std::string>("REFIT_PAIR_BASIS"));
}
void prepare_embed_refit(const core::Context &c, const models::EAMSetflIO &eam,
                         models::EAMRefitConfig &cfg) {
  validate_embed_refit(c);
  auto block = c.blocks("DESCRIPTOR")[0];
  struct Entry {
    std::size_t line;
    std::vector<std::string> values;
  };
  std::map<std::string, Entry> fields;
  std::vector<std::string> path;
  for (std::size_t i = 1; i + 1 < block.lines.size(); ++i) {
    auto w = words(block.lines[i]);
    if (w.empty())
      continue;
    auto key = core::to_upper(w[0]);
    if (key.rfind("END", 0) == 0) {
      if (path.empty())
        bad("unexpected boundary");
      path.pop_back();
      continue;
    }
    std::string full;
    for (const auto &p : path)
      full += p + ".";
    full += key;
    if (w.size() == 1) {
      path.push_back(key);
      continue;
    }
    w.erase(w.begin());
    if (!fields.emplace(full, Entry{i, w}).second)
      bad("duplicate " + full);
  }
  auto entry = [&](const std::string &key) -> Entry & {
    auto it = fields.find(key);
    if (it == fields.end())
      bad("missing " + key);
    return it->second;
  };
  auto is_auto = [](const Entry &e) {
    return !e.values.empty() && core::to_upper(e.values[0]) == "AUTO";
  };
  auto replace = [&](const std::string &key, const std::vector<double> &v) {
    auto &e = entry(key);
    block.lines[e.line] = line_values(key.substr(key.find_last_of('.') + 1), v);
    e.values.clear();
    for (double x : v)
      e.values.push_back(core::format_roundtrip(x));
  };
  auto radius = entry("DENSITY.CUTOFF.RADIUS");
  if (is_auto(radius)) {
    if (radius.values.size() != 1)
      bad("RADIUS AUTO takes no arguments");
    replace("DENSITY.CUTOFF.RADIUS", {eam.rcut()});
  }
  double rcut = 0;
  if (!core::parse_decimal_finite(entry("DENSITY.CUTOFF.RADIUS").values[0],
                                  rcut) ||
      !(rcut > 0))
    bad("invalid density RADIUS");
  for (bool density : {true, false}) {
    const std::string prefix = density ? "DENSITY." : "EMBEDDING.";
    auto basis = entry(prefix + "BASIS").values;
    if (basis.size() != 1)
      bad(prefix + "BASIS requires one family");
    auto family = models::embed_basis_from_name(basis[0]);
    bool knot = family == models::EmbedBasis::Knot5;
    const double lo = density ? std::max(cfg.r_lo_frac * rcut, eam.dr()) : 0;
    const double hi = density ? rcut : eam.rhomax();
    auto centres = entry(prefix + "CENTRES");
    int count = density ? cfg.nrho : cfg.nF;
    double spacing = 0;
    if (is_auto(centres)) {
      if (centres.values.size() > 2)
        bad("CENTRES AUTO accepts at most one count");
      if (centres.values.size() == 2) {
        double n = 0;
        if (!core::parse_decimal_finite(centres.values[1], n) || n < 1 ||
            n > 100000 || std::floor(n) != n)
          bad("invalid AUTO count");
        if (c.has_stored(density ? "NRHO_BASIS" : "NF_BASIS"))
          bad("AUTO count conflicts with a global basis count");
        count = static_cast<int>(n);
      }
      const double support =
          density && core::to_upper(entry("DENSITY.CUTOFF.TYPE").values[0]) ==
                         "NONE"
              ? models::embed_support(family)
              : 0;
      spacing = (hi - lo) / std::max(1.0, count - 1 + support);
      std::vector<double> grid;
      for (int i = 0; i < count; ++i)
        grid.push_back(lo + i * spacing);
      replace(prefix + "CENTRES", grid);
    } else {
      if (c.has_stored(density ? "NRHO_BASIS" : "NF_BASIS"))
        bad("explicit component grid conflicts with a global basis count");
      const auto &tokens = centres.values;
      for (std::size_t i = 0; i < tokens.size(); ++i)
        if (core::to_upper(tokens[i]) == "LIN" ||
            core::to_upper(tokens[i]) == "GEOM") {
          if (i + 3 >= tokens.size())
            bad("incomplete component grid");
          i += 3;
        }
      auto grid = models::expand_embed_grid(tokens, prefix + "CENTRES");
      count = static_cast<int>(grid.size());
      if (count < 1)
        bad("empty grid");
      spacing = (hi - lo) / std::max(1, count - 1);
      for (std::size_t i = 1; i < grid.size(); ++i)
        if (std::fabs(grid[i] - grid[i - 1]) > 0)
          spacing = std::min(spacing, std::fabs(grid[i] - grid[i - 1]));
      replace(prefix + "CENTRES", grid);
    }
    if (!knot) {
      const bool widths = fields.count(prefix + "WIDTHS");
      if (widths && fields.count(prefix + "ETA"))
        bad(prefix + "ETA and WIDTHS are mutually exclusive");
      auto scale = entry(prefix + (widths ? "WIDTHS" : "ETA"));
      if (is_auto(scale)) {
        if (scale.values.size() != 1)
          bad("scale AUTO takes no arguments");
        if (!(spacing > 0) || !std::isfinite(spacing))
          bad("AUTO scale cannot be derived from this grid");
        double value = widths ? spacing
                       : family == models::EmbedBasis::Gaussian
                           ? 1 / (spacing * spacing)
                           : 1 / spacing;
        replace(prefix + (widths ? "WIDTHS" : "ETA"),
                std::vector<double>(count, value));
      }
    }
    if (density) {
      auto amp = entry(prefix + "AMPLITUDES");
      cfg.fit_density_amplitudes = is_auto(amp);
      if (cfg.fit_density_amplitudes) {
        if (amp.values.size() != 1)
          bad("AMPLITUDES AUTO takes no arguments");
        replace(prefix + "AMPLITUDES", std::vector<double>(count, 0));
      }
    }
  }
  auto resolved = c;
  resolved.replace_blocks({block});
  auto specs = models::embed_specs(resolved);
  cfg.component = specs.at(0);
  cfg.nrho = static_cast<int>(cfg.component->density.centres.size());
  cfg.nF = static_cast<int>(cfg.component->embedding.centres.size());
  cfg.unified_output = true;
  for (const auto &p : cfg.component->pairs)
    if (p != eam.element())
      bad("descriptor PAIRS must match the single source element " +
          eam.element());
  cfg.rho_form = cfg.component->form;
  cfg.embf0_zero = cfg.component->reference == models::EmbF0Mode::Zero;
  cfg.basis = c.has_stored("REFIT_PAIR_BASIS")
                  ? c.get<std::string>("REFIT_PAIR_BASIS")
                  : "Blip5";
  if (core::to_upper(c.stored_value("REFIT_MODE").value_or("compact")) ==
          "DENSE" &&
      (models::basis_kind_from_string(cfg.basis) == models::BasisKind::Knot5 ||
       cfg.component->density.basis == models::EmbedBasis::Knot5 ||
       cfg.component->embedding.basis == models::EmbedBasis::Knot5))
    bad("REFIT_MODE dense fits no knot component; REFIT_MODE compact fits "
        "knot components");
}
} // namespace engines
} // namespace mlip
} // namespace tadah
