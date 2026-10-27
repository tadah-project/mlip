#include <tadah/mlip/bound_suggester.h>
#include <tadah/mlip/joined_descriptors.h>
#include <tadah/mlip/stream_format_guard.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <tadah/core/number_format.h>
#include <tadah/models/eam/pot_curve_edit.h>

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

/* @p v as the OPTIM block writes a bound (scientific, 7 significant
 * digits), rounded down (@p up false) or up (@p up true): the bound read
 * back holds @p v, so that a box holds its range and a seed's value at an
 * end of it (rounding to the nearest can put the bound past the value). */
std::string bound_text(double v, bool up) {
  auto text_of = [](double x) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(6) << x;
    return os.str();
  };
  std::string text = text_of(v);
  for (int i = 0; i < 3; ++i) {
    const double back = std::stod(text);
    if (up ? back >= v : back <= v) return text;
    // One unit of the 7th significant digit outward.
    const double step = std::pow(
        10.0, std::floor(std::log10(std::fabs(back))) - 6.0);
    text = text_of(up ? back + step : back - step);
  }
  throw std::logic_error("write_optim_block: no 7-digit bound holds " +
                         text_of(v));
}

/* A range of zero width (both ends one value, the seed's or none) is no
 * box: tadah hpo refuses it. Such a row has no OPTIM line, and its
 * rationale says why. */
void refuse_zero_width(OptimSuggestion &s) {
  if (s.none || s.lo != s.hi) return;
  std::ostringstream value;
  value << s.lo;
  s.none = true;
  s.rationale = open_ended(s.rationale) + ": a range of zero width (" +
                value.str() + "), which tadah hpo refuses";
}

/* The suggestion of entry @p ap of @p pm on the statistics @p line of the
 * element pairs of a descriptor's line: the suggester's on the statistics of
 * each pair (one per_pair entry), combined (no row when no pair gives one):
 * for a parameter marked ParamMeta::reach_each_pair (a cutoff, which its
 * check judges at each pair), the range from the largest low end of the
 * pairs' ranges to the largest high end, so that each value in it reaches
 * each pair; for any other, the range that holds the range of each. Without
 * a pair, the suggester's on @p line (no row, naming the pairs of the line).
 * The rationale gives each pair's once (none when the suggester gives none)
 * after the rule that combines them; a range is fixed when every pair's
 * is. */
models::BoundSuggestion suggest_over_pairs(
    const models::ParamMeta &pm, const tadah::core::stats::DataStats &line,
    const models::SeedView &sv, int ap) {
  if (line.per_pair.size() < 2) return pm.suggest(line, sv, ap);
  std::vector<models::BoundSuggestion> each;
  for (const auto &pp : line.per_pair) {
    tadah::core::stats::DataStats one = line;
    one.per_pair = {pp};
    each.push_back(pm.suggest(one, sv, ap));
  }
  std::vector<std::string> why;
  for (const auto &b : each)
    if (!b.rationale.empty() &&
        std::find(why.begin(), why.end(), b.rationale) == why.end())
      why.push_back(b.rationale);
  std::string joined;
  for (const auto &w : why)
    joined += (joined.empty() ? "" : "; ") + open_ended(w);
  bool ranged = false, fixed = true;
  double lo = 0.0, hi = 0.0;
  for (const auto &b : each) {
    if (b.none) continue;
    const double l = std::min(b.first, b.second);
    const double h = std::max(b.first, b.second);
    lo = !ranged ? l : pm.reach_each_pair ? std::max(lo, l) : std::min(lo, l);
    hi = ranged ? std::max(hi, h) : h;
    fixed = fixed && b.fixed;
    ranged = true;
  }
  if (why.size() > 1)
    joined = (pm.reach_each_pair
                  ? "the range from the largest low end to the largest high "
                    "end of the ranges of the element pairs of the "
                    "descriptor's line, so that each value in it reaches each "
                    "pair: "
                  : "the range that holds the range of each element pair of "
                    "the descriptor's line: ") +
             joined;
  if (!ranged) return models::BoundSuggestion::no_row(joined);
  models::BoundSuggestion out(lo, hi, joined);
  out.fixed = fixed;
  return out;
}
}  // namespace

std::vector<OptimSuggestion>
suggest_bounds_for(const tadah::core::stats::DataStats& data,
                   const models::DescriptorMeta&        meta,
                   const tadah::core::Context&          seed,
                   const std::optional<models::LinePairs>& pairs) {
  std::vector<OptimSuggestion> out;
  // A descriptor sums over the neighbours of the element pairs its line
  // names: its ranges come from their statistics, which its rationales
  // name.
  const models::SeedView sv = pairs ? models::SeedView(seed, {}, *pairs)
                                    : models::SeedView(seed);
  const tadah::core::stats::DataStats line =
      pairs ? pairs->stats(data) : data;

  for (const auto& pm : meta.params) {
    // A value of the descriptor's TYPE line (ParamMeta::arg_pos > 0: its
    // position on the descriptor's line, from 1, the descriptor's name being
    // 1, as the screening length a0 of "TYPE2B D2_ZBL a0 s0 p0 p1 Ta Ta" is
    // 2): a row without an OPTIM line, as tadah hpo searches no value of a
    // TYPE2B or TYPEMB line (it refuses OPTIM TYPE2B at its line). The row
    // numbers the value as an OPTIM line numbers the values of a key, by
    // the numbers among them (a0 is (1)), and its seed value is read from
    // the descriptor's line, which @p seed gives (its own context for a
    // descriptor of a joined line). A seed without the line has no such
    // value, and no row.
    if (pm.arg_pos > 0) {
      if (!sv.contains(pm.key)) continue;
      const std::size_t at = static_cast<std::size_t>(pm.arg_pos - 1);
      const std::size_t n = sv.size(pm.key);
      if (n == 0 || sv.get<std::string>(pm.key, 0) != meta.label || at >= n)
        throw std::logic_error(
            "suggest_bounds_for: the " + pm.key + " line of the seed is not "
            "the line of " + meta.label + " with its value " +
            std::to_string(pm.arg_pos) + " (ParamMeta arg_pos)");
      std::size_t numbers = 0;
      for (std::size_t i = 0; i <= at; ++i) {
        double v = 0.0;
        if (tadah::core::parse_number(sv.get<std::string>(pm.key, i), v))
          ++numbers;
      }
      OptimSuggestion s;
      s.descriptor_label = meta.label;
      s.key      = pm.key;
      s.none     = true;
      s.crit     = pm.crit;
      s.scale    = pm.scale;
      s.units    = pm.units;
      s.seed_value = sv.get<double>(pm.key, at);
      s.arg_pos  = static_cast<int>(numbers) - 1;
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
      const models::BoundSuggestion bracket =
          pairs ? suggest_over_pairs(pm, line, sv, ap)
                : pm.suggest(data, sv, ap);

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
      refuse_zero_width(s);

      out.push_back(std::move(s));
    }
  }
  return out;
}

void merge_shared_rows(std::vector<OptimSuggestion>& rows) {
  std::vector<OptimSuggestion> out;
  std::map<std::pair<std::string, int>, std::size_t> row_of;
  for (auto& r : rows) {
    // "(1)" names entry 1 of a key of one value (arg_pos -1) and of a list.
    const auto entry = std::make_pair(r.key, std::max(0, r.arg_pos));
    const auto it = row_of.find(entry);
    if (it == row_of.end()) {
      row_of.emplace(entry, out.size());
      out.push_back(std::move(r));
      continue;
    }
    OptimSuggestion& f = out[it->second];
    const bool same = f.none == r.none && f.fixed == r.fixed &&
                      (f.none || (f.lo == r.lo && f.hi == r.hi)) &&
                      f.rationale == r.rationale;
    std::string why = open_ended(f.rationale);
    if (!same) {
      if (f.none && !r.none) {
        f.lo = r.lo;
        f.hi = r.hi;
        f.fixed = r.fixed;
        f.none = false;
      } else if (!r.none) {
        f.lo = std::min(f.lo, r.lo);
        f.hi = std::max(f.hi, r.hi);
        f.fixed = f.fixed && r.fixed;
      }
      why += "; with that of " + r.descriptor_label + ": " +
             open_ended(r.rationale);
    }
    if (f.descriptor_label != r.descriptor_label &&
        ("+" + f.descriptor_label + "+").find("+" + r.descriptor_label +
                                              "+") == std::string::npos)
      f.descriptor_label += "+" + r.descriptor_label;
    const std::string shared = "; a key the descriptors of the line share";
    if (why.size() < shared.size() ||
        why.compare(why.size() - shared.size(), shared.size(), shared) != 0)
      why += shared;
    f.rationale = why;
  }
  rows = std::move(out);
}

void leave_out_derived_cutoffs(std::vector<OptimSuggestion>& rows,
                               const tadah::core::Context& seed) {
  // The row of entry `entry` of `key` gets none, its rationale naming the
  // largest knot `knot` of the channel `what` and the OPTIM line tadah hpo
  // warns about: of the entry on a joined line (`joined`), whose other
  // entries hold the cutoffs of other descriptors, "OPTIM RCUT2B (1)".
  auto leave_out = [&](const std::string& key, std::size_t entry,
                       bool joined, const std::string& what, double knot) {
    std::ostringstream value;
    value << knot;
    const std::string optim =
        joined ? key + " (" + std::to_string(entry + 1) + ")" : key;
    for (auto& r : rows) {
      if (r.key != key ||
          static_cast<std::size_t>(std::max(0, r.arg_pos)) != entry)
        continue;
      r.none = true;
      r.fixed = false;
      r.rationale = "the cutoff of " + what + " is its largest knot (" +
                    value.str() + " Å), which tadah train and tadah hpo "
                    "derive: tadah hpo warns that an OPTIM " + optim +
                    " line has no effect";
    }
  };
  for (const char* sfx : {"2B", "MB"}) {
    const std::string key = std::string("RCUT") + sfx;
    const std::string type_key = std::string("TYPE") + sfx;
    if (is_joined_line(seed, type_key)) {
      // each knot descriptor of the line: its own RCUT entry
      std::vector<JoinedDescriptor> parts;
      try {
        parts = joined_descriptors(seed, type_key);
      } catch (const std::exception&) {
        continue;  // a line D_mJoin refuses: its own error elsewhere
      }
      const std::string line = seed.get<std::string>(type_key, 0);
      for (const auto& jd : parts) {
        const auto knot = models::potedit::derived_knot_rcut(jd.context, sfx);
        const auto first = jd.first_entries.find(key);
        if (!knot || first == jd.first_entries.end()) continue;
        leave_out(key, first->second, true,
                  std::string(sfx) == "2B"
                      ? "a D2_Knot5 descriptor of a " + line + " line"
                      : "a knot DM_REAM density of a " + line + " line",
                  *knot);
      }
      continue;
    }
    const auto knot = models::potedit::derived_knot_rcut(seed, sfx);
    if (!knot) continue;
    leave_out(key, 0, false,
              std::string(sfx) == "2B"
                  ? "a D2_Knot5 line alone on TYPE2B"
                  : "a knot DM_REAM density alone on TYPEMB",
              *knot);
  }
}

void print_suggestions(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os) {
  const StreamFormatGuard keep(os);
  os << "# Suggested OPTIM bounds (each range is a suggestion, to be "
        "checked before use)\n";
  os << "#\n";
  os << "#   col 1: descriptor.key(i), i the entry as an OPTIM line numbers "
        "it\n";
  os << "#   col 2: lo .. hi   (units in column 3)\n";
  os << "#   col 3: scale\n";
  os << "#   col 4: criticality\n";
  os << "#   col 5: rationale: the form a range assumes and the data it\n";
  os << "#          comes from, \"fixed range\" and why, or\n";
  os << "#          the scale and the role of the parameter; a range\n";
  os << "#          widened to hold the seed's value says so\n";
  os << "#\n";
  for (const auto& s : sug) {
    // The entry as an OPTIM line numbers it, from 1 ("(1)" for a key of
    // one value), as write_optim_block writes it.
    std::ostringstream tag;
    tag << s.descriptor_label << "." << s.key << "("
        << (s.arg_pos >= 0 ? s.arg_pos + 1 : 1) << ")";
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
  os << "# Written by `tadah analysis suggest_bounds`, one rationale\n";
  os << "# comment per line. The ranges are suggestions, to be checked\n";
  os << "# before tadah hpo runs.\n\n";

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
    os << "OPTIM " << g.key << " " << index.str() << " "
       << bound_text(g.lo, false) << " " << bound_text(g.hi, true)
       << "\n\n";
  }
}

}  // namespace mlip
}  // namespace tadah
