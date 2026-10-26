#include <tadah/mlip/bound_suggester.h>
#include <tadah/mlip/stream_format_guard.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace tadah {
namespace mlip {

namespace {
const char* scale_str(models::Scale s) {
  using S = models::Scale;
  switch (s) {
    case S::Distance:           return "Å";
    case S::InverseDistanceSq:  return "Å^-2";
    case S::InverseDistance:    return "Å^-1";
    case S::Density:            return "ρ";
    case S::InverseDensity:     return "ρ^-1";
    case S::InverseDensitySq:   return "ρ^-2";
    case S::Energy:             return "eV";
    case S::Force:              return "eV/Å";
    case S::Count:              return "n";
    default:                    return "-";
  }
}

/* @p text without a final full stop, so that a clause can follow it. */
std::string open_ended(std::string text) {
  if (!text.empty() && text.back() == '.') text.pop_back();
  return text;
}

/* A range holds the seed's value: tadah hpo starts a search from the seed
 * and refuses a starting point outside its box. A range a suggester gives
 * without the seed's value (a cutoff's range ends at the 99th percentile of
 * the distances within the seed's own cutoff) is widened to it, and its
 * rationale says so. */
void hold_seed(OptimSuggestion &s) {
  if (s.none || !s.seed_value) return;
  const double v = *s.seed_value;
  const double lo = std::min(s.lo, s.hi);
  const double hi = std::max(s.lo, s.hi);
  if (v >= lo && v <= hi) return;
  s.lo = std::min(lo, v);
  s.hi = std::max(hi, v);
  std::ostringstream value;
  value << v;
  s.rationale = open_ended(s.rationale) + "; widened to hold the seed's "
                "value " + value.str() + ", where tadah hpo starts";
}
}  // namespace

std::vector<OptimSuggestion>
suggest_bounds_for(const tadah::core::stats::DataStats& data,
                   const models::DescriptorMeta&        meta,
                   const tadah::core::Context&          seed) {
  std::vector<OptimSuggestion> out;
  models::SeedView sv(seed);

  for (const auto& pm : meta.params) {
    // A value of the descriptor's TYPE line (arg_pos > 0: the value at
    // index arg_pos - 1 of its line, as the screening parameters of D2_ZBL
    // are values 2 to 5 of "TYPE2B D2_ZBL a0 s0 p0 p1 Ta Ta"): a row
    // without an OPTIM line, as tadah hpo searches no value of a TYPE2B or
    // TYPEMB line (it refuses OPTIM TYPE2B at its line). Its arg_pos is its
    // place after the descriptor's name; the seed's value is read from the
    // descriptor's line, the first one of the key its name opens.
    if (pm.arg_pos > 0) {
      OptimSuggestion s;
      s.descriptor_label = meta.label;
      s.key      = pm.key;
      s.arg_pos  = std::max(0, pm.arg_pos - 2);
      s.none     = true;
      s.crit     = pm.crit;
      s.scale    = pm.scale;
      s.units    = pm.units;
      if (sv.contains(pm.key)) {
        const std::size_t n = sv.size(pm.key);
        for (std::size_t i = 0; i < n; ++i) {
          if (sv.get<std::string>(pm.key, i) != meta.label) continue;
          const std::size_t at = i + static_cast<std::size_t>(pm.arg_pos) - 1;
          if (at < n) s.seed_value = sv.get<double>(pm.key, at);
          break;
        }
      }
      s.rationale = open_ended(pm.role) + ": a value of the " + meta.label +
                    " line of " + pm.key + ", which tadah hpo does not "
                    "search (it refuses OPTIM " + pm.key + "); the seed's " +
                    pm.key + " line sets it";
      out.push_back(std::move(s));
      continue;
    }
    if (!pm.suggest) continue;

    // Determine how many entries this ParamMeta covers given the
    // current seed. arg_pos == -1 -> scalar (one suggestion). arg_pos
    // == 0 -> vector entry; iterate over `sv.size(pm.key)`.
    std::size_t n_entries = 1;
    int base_arg_pos = -1;
    if (pm.arg_pos == 0) {
      n_entries = sv.contains(pm.key) ? sv.size(pm.key) : 0;
      if (n_entries == 0) {
        // No seed entry yet — emit a single suggestion at arg_pos 0
        // with a generic rationale ("seed config has no entries; emit
        // one representative bound").
        n_entries = 1;
        base_arg_pos = 0;
      }
    }

    for (std::size_t i = 0; i < n_entries; ++i) {
      const int ap = (pm.arg_pos < 0) ? -1
                                      : static_cast<int>(base_arg_pos < 0 ? i : base_arg_pos);
      const models::BoundSuggestion bracket = pm.suggest(data, sv, ap);

      OptimSuggestion s;
      s.descriptor_label = meta.label;
      s.key      = pm.key;
      s.arg_pos  = ap;
      s.lo       = bracket.first;
      s.hi       = bracket.second;
      s.fixed    = bracket.fixed;
      s.none     = bracket.none;
      s.crit     = pm.crit;
      s.scale    = pm.scale;
      s.units    = pm.units;

      // The seed value, as a number (the grids are expanded): a value
      // that is none is an error of the seed, not a value to leave out.
      if (sv.contains(pm.key) &&
          (ap < 0 || static_cast<std::size_t>(ap) < sv.size(pm.key)))
        s.seed_value = (ap < 0)
            ? sv.get<double>(pm.key)
            : sv.get<double>(pm.key, static_cast<std::size_t>(ap));

      // Rationale: the suggester's (the form it assumes and the data it
      // comes from), else the scale and the role of the parameter.
      if (!bracket.rationale.empty()) {
        s.rationale = bracket.rationale;
      } else {
        std::ostringstream rs;
        rs << "scale=" << scale_str(pm.scale) << "; role=" << pm.role;
        s.rationale = rs.str();
      }
      hold_seed(s);

      out.push_back(std::move(s));
    }
  }
  return out;
}

void print_suggestions(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os) {
  const StreamFormatGuard keep(os);
  os << "# Suggested OPTIM bounds (review before use)\n";
  os << "#\n";
  os << "#   col 1: descriptor.key[arg_pos]\n";
  os << "#   col 2: lo .. hi   (units in column 3)\n";
  os << "#   col 3: scale\n";
  os << "#   col 4: criticality\n";
  os << "#   col 5: rationale: the form a range assumes and the data it\n";
  os << "#          comes from, \"fixed range\" and why, or\n";
  os << "#          the scale and the role of the parameter; a range\n";
  os << "#          widened to hold the seed's value says so\n";
  os << "#\n";
  for (const auto& s : sug) {
    std::ostringstream tag;
    tag << s.descriptor_label << "." << s.key;
    if (s.arg_pos >= 0) tag << "[" << s.arg_pos << "]";
    const char* crit =
        (s.crit == models::Criticality::Critical)    ? "CRITICAL"
      : (s.crit == models::Criticality::Recommended) ? "recommended"
      :                                                 "optional";
    os << std::left << std::setw(28) << tag.str() << " ";
    if (s.none) {
      os << "no OPTIM line: " << s.rationale << "\n";
      continue;
    }
    os << std::scientific << std::setprecision(4)
       << s.lo << " .. " << s.hi
       << "   " << std::setw(6) << scale_str(s.scale)
       << "   " << std::setw(11) << crit
       << "   " << s.rationale << "\n";
  }
}

void write_optim_block(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os) {
  const StreamFormatGuard keep(os);
  os << "# Written by `tadah analysis suggest_bounds`. Each line\n";
  os << "# carries a one-line rationale comment. Review and edit\n";
  os << "# before running tadah hpo.\n\n";

  // One OPTIM line per run of consecutive entries of a (descriptor, key)
  // with the same range, in the order of `sug`: "(1)" for a scalar,
  // "(i-j)" for entries i to j of a vector key. An entry without a range
  // gets a comment line.
  struct Group {
    std::string label;
    std::string key;
    int min_arg_pos;
    int max_arg_pos;
    double lo, hi;
    std::string rationale;
    bool none;
  };
  std::vector<Group> groups;
  for (const auto& s : sug) {
    if (!groups.empty()) {
      Group& g = groups.back();
      if (g.label == s.descriptor_label && g.key == s.key &&
          s.arg_pos >= 0 && g.min_arg_pos >= 0 &&
          s.arg_pos == g.max_arg_pos + 1 && g.none == s.none &&
          (s.none || (g.lo == s.lo && g.hi == s.hi)) &&
          g.rationale == s.rationale) {
        g.max_arg_pos = s.arg_pos;
        continue;
      }
    }
    // Each suggester is free to return (lo, hi) as it pleases; the OPTIM
    // block requires lo <= hi (e.g. RCUT bracketing 2NN..p99 on a dataset
    // clamped at p99).
    groups.push_back(Group{s.descriptor_label, s.key, s.arg_pos, s.arg_pos,
                           std::min(s.lo, s.hi), std::max(s.lo, s.hi),
                           s.rationale, s.none});
  }

  for (const auto& g : groups) {
    std::ostringstream index;
    if (g.min_arg_pos < 0) {
      index << "(1)";
    } else if (g.min_arg_pos == g.max_arg_pos) {
      index << "(" << (g.min_arg_pos + 1) << ")";
    } else {
      // OPTIM uses 1-based indexing: arg_pos 0..N-1 -> (1-N)
      index << "(" << (g.min_arg_pos + 1) << "-" << (g.max_arg_pos + 1) << ")";
    }
    if (g.none) {
      os << "# " << g.label << " :: " << g.key << " " << index.str()
         << ": no OPTIM line: " << g.rationale << "\n\n";
      continue;
    }
    os << "# " << g.label << " :: " << g.rationale << "\n";
    os << "OPTIM " << g.key << " " << index.str()
       << " " << std::scientific << std::setprecision(6) << g.lo
       << " " << std::scientific << std::setprecision(6) << g.hi
       << "\n\n";
  }
}

}  // namespace mlip
}  // namespace tadah
