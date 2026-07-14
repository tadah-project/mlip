#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>

#include <tadah/models/descriptors/basis_kind.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

struct OptimLine {
  std::size_t line = 0;        // index into `lines`
  std::string indent;          // leading whitespace (verbatim)
  std::string prefix;          // "#"-comment prefix (verbatim, may be empty)
  std::string key;
  std::size_t index = 0;       // 1-based
  double lo = 0.0, hi = 0.0;
  double centre = 0.0;         // recovered value
  double v = 0.0, q = 0.0;     // grid-snapped centre + bound grid (rewrite)
  bool is_rcut = false;
};

const std::regex kOptimRe(
    R"(^(\s*)(#\s*)?OPTIM\s+([A-Za-z0-9_]+)\s*\(\s*(\d+)\s*\)\s+(\S+)\s+(\S+)\s*$)");
const std::regex kHeaderPctRe(R"(Bounds are \+-([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)%)");
const std::regex kGroupHeaderRe(
    R"(^(#\s*)([A-Za-z0-9_]+)(\s+\(\+-)([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)(%.*)$)");
// The bump-support contract the writer records next to a bump channel's
// commented RCUT scalar: `# RCUT2B bump-support contract: w = 3 (...)`.
// Files without it (knot channels, pre-contract emissions) get no capping.
const std::regex kBumpContractRe(
    R"(^#\s*RCUT(2B|MB)\s+bump-support contract:\s*w\s*=\s*([0-9]*\.?[0-9]+).*$)");

std::string sci8(double v) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(8) << v;
  return os.str();
}

std::string pct_str(double p) {
  std::ostringstream os;
  os << std::defaultfloat << p;
  return os.str();
}

bool group_positive_only(const std::string &key, bool &known) {
  for (const auto &s : hpotarget_group_specs()) {
    if (key == s.key) {
      known = true;
      return s.positive_only;
    }
  }
  known = false;
  return false;
}

}  // namespace

RetargetReport retarget_hpotarget_lines(std::vector<std::string> &lines,
                                        const RetargetOptions &opts) {
  RetargetReport rep;

  // ---- collect every OPTIM line (active and commented) ----
  std::vector<OptimLine> ols;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (!std::regex_match(lines[i], m, kOptimRe)) continue;
    OptimLine ol;
    ol.line = i;
    ol.indent = m[1].str();
    ol.prefix = m[2].str();
    ol.key = m[3].str();
    ol.index = std::stoul(m[4].str());
    try {
      ol.lo = std::stod(m[5].str());
      ol.hi = std::stod(m[6].str());
    } catch (const std::exception &) {
      rep.warnings.push_back("line " + std::to_string(i + 1) +
                             ": non-numeric OPTIM bounds — left untouched.");
      continue;
    }
    ol.is_rcut = (ol.key == "RCUT2B" || ol.key == "RCUTMB");
    ol.centre = 0.5 * (ol.lo + ol.hi);
    ols.push_back(ol);
  }
  if (ols.empty()) {
    rep.warnings.push_back("no OPTIM lines found — nothing to rewrite.");
    return rep;
  }

  // ---- bump-support contracts (writer-recorded metadata comments) ----
  std::map<std::string, double> bump_w;  // suffix "2B"/"MB" -> half-width w
  for (const auto &l : lines) {
    std::smatch m;
    if (std::regex_match(l, m, kBumpContractRe))
      bump_w[m[1].str()] = std::stod(m[2].str());
  }
  auto bump_cgrid_key = [&bump_w](const std::string &key) {
    return (key == "CGRID2B" && bump_w.count("2B")) ||
           (key == "CGRIDMB" && bump_w.count("MB"));
  };

  // ---- emission percentage from the file header (for clamp repair) ----
  double hdr_pct = -1.0;
  for (const auto &l : lines) {
    std::smatch m;
    if (std::regex_search(l, m, kHeaderPctRe)) {
      hdr_pct = std::stod(m[1].str());
      break;
    }
  }
  if (hdr_pct <= 0.0)
    rep.warnings.push_back(
        "no 'Bounds are +-X%' header found — asymmetric (clamped) bounds "
        "cannot be repaired exactly; midpoints are used as centres.");

  // ---- group by key; repair clamped centres for positive-only groups ----
  std::map<std::string, std::vector<OptimLine *>> groups;
  for (auto &ol : ols) groups[ol.key].push_back(&ol);

  for (auto &[key, group] : groups) {
    bool known = false;
    const bool positive_only = group_positive_only(key, known);
    const bool is_rcut = group.front()->is_rcut;
    if (!known && !is_rcut)
      rep.warnings.push_back(
          "OPTIM key '" + key +
          "' is not a standard refit hpotarget group — rewritten with the "
          "plain symmetric rule.");

    // contiguity check: a hand-pruned group shifts gmax/floor vs a fresh
    // emission (bounds still valid, just not identical to a full re-emit)
    if (!is_rcut) {
      std::vector<std::size_t> idx;
      for (auto *ol : group) idx.push_back(ol->index);
      std::sort(idx.begin(), idx.end());
      for (std::size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] != i + 1) {
          rep.warnings.push_back(
              "group '" + key + "' has non-contiguous indices — its scale "
              "floor is computed from the surviving lines only.");
          break;
        }
      }
    }

    // Midpoints are exact except for: positive-only groups (lo can be
    // clamped), RCUT scalars, and bump-contract CGRID groups (hi can be
    // capped by the bump-support rule) — those need the rule-consistency
    // check + one-sided recovery below.
    const bool is_bump_cgrid = bump_cgrid_key(key);
    if (!positive_only && !is_rcut && !is_bump_cgrid) continue;
    if (hdr_pct <= 0.0) continue;  // cannot check/repair without the pct

    // group scale from the midpoints — the max element is (essentially)
    // never lo-clamped, so it is unbiased
    double gmax = 0.0;
    for (auto *ol : group) gmax = std::max(gmax, std::fabs(ol->centre));
    const double floor = is_rcut ? 1e-6 : optim_group_floor(gmax);

    // A clamped line cannot be detected by bound asymmetry (every interval
    // is symmetric about its own midpoint): instead check whether the
    // OBSERVED half-width matches the emission rule at the midpoint. When
    // it does not, one bound was moved off the symmetric rule and the OTHER
    // one is authoritative: positive-only groups / RCUT get the lo clamp
    // (hi = v + max(p·|v|, floor) recovers v); bump-contract CGRID groups
    // get the hi cap (lo = v - max(p·|v|, floor) recovers v).
    const double p = hdr_pct / 100.0;
    for (auto *ol : group) {
      const double d_obs = 0.5 * (ol->hi - ol->lo);
      const double d_raw = std::max(p * std::fabs(ol->centre), floor);
      const double qd = optim_bound_quantum(ol->centre, d_raw);
      const double d_exp = quantize_optim(
          std::max(p * std::fabs(quantize_optim(ol->centre, qd)), floor), qd);
      if (std::fabs(d_obs - d_exp) <= 1e-6 * std::max(d_obs, d_exp))
        continue;  // consistent with the rule: midpoint is the centre
      double v;
      const bool from_lo = is_bump_cgrid && p < 1.0;
      if (from_lo) {
        v = ol->lo / (1.0 - p);              // branch p·v >= floor
        if (p * std::fabs(v) < floor) v = ol->lo + floor;  // floor branch
      } else {
        v = ol->hi / (1.0 + p);              // branch p·v >= floor
        if (p * std::fabs(v) < floor) v = ol->hi - floor;  // floor branch
      }
      v = quantize_optim(v, optim_bound_quantum(v, std::max(p * std::fabs(v),
                                                            floor)));
      rep.warnings.push_back(
          "OPTIM " + ol->key + " (" + std::to_string(ol->index) +
          "): bounds inconsistent with the +-" + pct_str(hdr_pct) +
          (from_lo
               ? "% emission rule (hi capped by the bump-support contract "
                 "or hand-edited) — centre recovered from the lower bound: "
               : "% emission rule (lo-clamped or hand-edited) — centre "
                 "recovered from the upper bound: ") +
          sci8(v) + ".");
      ol->centre = v;
    }
  }

  // ---- F2: knot channels — recentre RCUT lines on the largest knot ----
  // A knot hpotarget carries CGRID* lines but NO SGRID* lines (eta gauge).
  for (const char *sfx : {"2B", "MB"}) {
    const std::string ckey = std::string("CGRID") + sfx;
    const std::string skey = std::string("SGRID") + sfx;
    const std::string rkey = std::string("RCUT") + sfx;
    if (!groups.count(ckey) || groups.count(skey) || !groups.count(rkey))
      continue;
    double mx = -std::numeric_limits<double>::max();
    for (auto *ol : groups[ckey]) mx = std::max(mx, ol->centre);
    for (auto *ol : groups[rkey]) {
      if (std::fabs(ol->centre - mx) <= 1e-9 * std::max(1.0, std::fabs(mx)))
        continue;
      rep.warnings.push_back(
          rkey + " centre " + sci8(ol->centre) +
          " differs from the largest " + ckey + " knot " + sci8(mx) +
          " — recentred on the knot (the effective cutoff of a knot channel "
          "is its largest knot; vary " + ckey + " to change it).");
      ol->centre = mx;
    }
  }

  // ---- rewrite bounds ----
  for (auto &ol : ols) {
    bool known = false;
    const bool positive_only = group_positive_only(ol.key, known);
    const auto it = opts.pct_per_key.find(ol.key);
    const double pct = it != opts.pct_per_key.end() ? it->second : opts.pct;
    double gmax = 0.0;
    for (auto *g : groups[ol.key]) gmax = std::max(gmax, std::fabs(g->centre));

    // Same centre/half-width grid snapping as the emitter (see
    // optim_bound_quantum): the printed bounds are exact decimals, so a
    // subsequent retarget recovers this centre exactly (fixed point).
    const double floor = ol.is_rcut ? 1e-6 : optim_group_floor(gmax);
    const double d_raw = std::max(pct / 100.0 * std::fabs(ol.centre), floor);
    const double q = optim_bound_quantum(ol.centre, d_raw);
    const double v = quantize_optim(ol.centre, q);
    const double d =
        quantize_optim(std::max(pct / 100.0 * std::fabs(v), floor), q);
    double lo, hi;
    if (ol.is_rcut) {
      lo = std::max(0.0, v - d);
      hi = v + d;
      if (hi <= lo) hi = lo + 1e-3;
    } else {
      lo = v - d;
      hi = v + d;
      if (positive_only && lo <= 0.0) lo = std::max(1e-6, 0.5 * std::fabs(v));
      if (hi <= lo) hi = lo + std::max(floor, 1e-6);
    }
    ol.lo = lo;
    ol.hi = hi;
    ol.v = v;
    ol.q = q;
  }

  // ---- bump-support cap re-application (contract files only) ----
  // Rewritten bounds are symmetric again; re-tighten CGRID hi / SGRID lo per
  // index exactly like a fresh emission (shared apply_bump_cap), so
  // retarget(fresh, same pct) stays byte-identical and any retargeted file
  // is a fixed point.
  for (const auto &[sfx, w] : bump_w) {
    const std::string ckey = "CGRID" + sfx;
    const std::string skey = "SGRID" + sfx;
    const std::string rkey = "RCUT" + sfx;
    if (!groups.count(ckey) && !groups.count(skey)) continue;
    if (!groups.count(rkey) || !groups.count(ckey) || !groups.count(skey)) {
      rep.warnings.push_back(
          "bump-support contract for " + rkey + " is present but the " +
          (groups.count(rkey) ? (groups.count(ckey) ? skey : ckey) : rkey) +
          " lines are missing (hand-pruned?) — bounds left uncapped; the "
          "`tadah hpo` startup gate is the backstop.");
      continue;
    }
    // The binding cutoff for the cap. Normally the group is ONE commented
    // scalar whose centre IS the fixed rcut. An ACTIVE OPTIM RCUT line makes
    // rcut a search dimension — `tadah hpo`'s bump-support gate then binds
    // at its LOW bound — and duplicate lines (e.g. a hand-added active line
    // next to the emitted commented one) are ambiguous. Bind against the
    // smallest reachable cutoff and say so, instead of silently trusting
    // whichever line happens to come first in the file.
    double rcut = std::numeric_limits<double>::max();
    bool any_active = false;
    for (const auto *ol : groups[rkey]) {
      rcut = std::min(rcut, ol->prefix.empty() ? ol->lo : ol->centre);
      any_active |= ol->prefix.empty();
    }
    if (groups[rkey].size() > 1 || any_active)
      rep.warnings.push_back(
          rkey + ": " + std::to_string(groups[rkey].size()) +
          " OPTIM line(s)" + (any_active ? ", some ACTIVE" : "") +
          " — the bump-support cap binds against the smallest reachable "
          "cutoff, " + sci8(rcut) + ". An active OPTIM " + rkey +
          " is a wasted dimension on a bump channel (the support gate keeps "
          "every bump inside its LOW bound, so the cutoff value cannot "
          "affect the model): keep ONE commented line centred on the fixed " +
          rkey + " and set " + rkey + " in the config.");
    std::map<std::size_t, OptimLine *> smap;
    for (auto *ol : groups[skey]) smap[ol->index] = ol;
    std::string spilled, orphans;
    for (auto *c : groups[ckey]) {
      const auto s = smap.find(c->index);
      if (s == smap.end()) {
        orphans += (orphans.empty() ? "" : ", ") + std::to_string(c->index);
        continue;
      }
      const BumpCapResult r =
          apply_bump_cap(c->v, c->hi, c->q, s->second->v, s->second->lo,
                         s->second->q, w, rcut);
      if (r.seed_spill)
        spilled += (spilled.empty() ? "" : ", ") + std::to_string(c->index);
    }
    if (!orphans.empty())
      rep.warnings.push_back(
          ckey + " index(es) " + orphans + " have no matching " + skey +
          " line — left uncapped; the `tadah hpo` startup gate is the "
          "backstop.");
    if (!spilled.empty())
      rep.warnings.push_back(
          ckey + " bump(s) " + spilled + " spill past " + rkey + " = " +
          sci8(rcut) + " at their own centres — bounds left uncapped; "
          "`tadah hpo` will reject this file until the pot is repaired "
          "(tadah refit --edit).");
  }

  // ---- absolute --set overrides (applied last) ----
  for (const auto &s : opts.sets) {
    bool found = false;
    for (auto &ol : ols) {
      if (ol.key == s.key && ol.index == s.index) {
        if (s.hi <= s.lo)
          throw std::runtime_error("--set " + s.key + "(" +
                                   std::to_string(s.index) +
                                   "): hi must be greater than lo.");
        ol.lo = s.lo;
        ol.hi = s.hi;
        found = true;
        break;
      }
    }
    if (!found) {
      std::string avail;
      for (const auto &[key, group] : groups) {
        avail += "\n  " + key + " (1-" + std::to_string(group.size()) + ")";
      }
      throw std::runtime_error(
          "--set " + s.key + "(" + std::to_string(s.index) +
          "): no such OPTIM line in the file. Available:" + avail);
    }
  }

  // A --set on a bump channel wins over the cap, but warn when the chosen
  // box violates the contract corner (`tadah hpo` will reject the file).
  for (const auto &s : opts.sets) {
    std::string sfx;
    if ((s.key == "CGRID2B" || s.key == "SGRID2B") && bump_w.count("2B"))
      sfx = "2B";
    else if ((s.key == "CGRIDMB" || s.key == "SGRIDMB") && bump_w.count("MB"))
      sfx = "MB";
    else
      continue;
    const std::string ckey = "CGRID" + sfx;
    const std::string skey = "SGRID" + sfx;
    const std::string rkey = "RCUT" + sfx;
    if (!groups.count(ckey) || !groups.count(skey) || !groups.count(rkey))
      continue;
    const double w = bump_w.at(sfx);
    const double rcut = groups[rkey].front()->centre;
    OptimLine *c = nullptr, *e = nullptr;
    for (auto *ol : groups[ckey]) if (ol->index == s.index) c = ol;
    for (auto *ol : groups[skey]) if (ol->index == s.index) e = ol;
    if (!c || !e || e->lo <= 0.0) continue;
    const double reach = c->hi + w / e->lo;
    if (reach > rcut + tadah::models::bump_rcut_tol(rcut))
      rep.warnings.push_back(
          "--set " + s.key + "(" + std::to_string(s.index) +
          "): the resulting box lets bump " + std::to_string(s.index) +
          " reach " + sci8(reach) + ", past " + rkey + " = " + sci8(rcut) +
          " — `tadah hpo` will reject this file (bump-support contract).");
  }

  // ---- serialise the rewritten lines ----
  for (const auto &ol : ols) {
    lines[ol.line] = ol.indent + ol.prefix + "OPTIM " + ol.key + " (" +
                     std::to_string(ol.index) + ") " + sci8(ol.lo) + " " +
                     sci8(ol.hi);
    ++rep.n_rewritten;
    if (ol.prefix.empty()) ++rep.n_active;
    else ++rep.n_commented;
  }

  // ---- refresh the percentage in the header + group comments ----
  for (auto &l : lines) {
    std::smatch m;
    if (std::regex_search(l, m, kHeaderPctRe)) {
      l = m.prefix().str() + "Bounds are +-" + pct_str(opts.pct) + "%" +
          m.suffix().str();
      continue;
    }
    if (std::regex_match(l, m, kGroupHeaderRe) && groups.count(m[2].str())) {
      const auto it = opts.pct_per_key.find(m[2].str());
      const double pct = it != opts.pct_per_key.end() ? it->second : opts.pct;
      l = m[1].str() + m[2].str() + m[3].str() + pct_str(pct) + m[5].str();
    }
  }

  return rep;
}

RetargetReport retarget_hpotarget_file(const std::string &in,
                                       const std::string &out, bool in_place,
                                       const RetargetOptions &opts) {
  namespace fs = std::filesystem;
  std::ifstream is(in);
  if (!is)
    throw std::runtime_error("retarget: cannot read " + in);
  std::vector<std::string> lines;
  for (std::string l; std::getline(is, l);) lines.push_back(l);
  is.close();

  const std::string target = in_place ? in : out;
  if (!in_place && fs::exists(out) && fs::exists(in) &&
      fs::equivalent(out, in))
    throw std::runtime_error(
        "retarget: refusing to overwrite the input file " + in +
        " — pass --in-place to rewrite it, or -o with a different name.");

  RetargetReport rep = retarget_hpotarget_lines(lines, opts);

  // atomic write: temp file in the same directory, then rename
  const fs::path tpath =
      fs::path(target).parent_path() /
      (fs::path(target).filename().string() + ".retarget.tmp");
  {
    std::ofstream os(tpath);
    if (!os)
      throw std::runtime_error("retarget: cannot write " + tpath.string());
    for (const auto &l : lines) os << l << "\n";
  }
  fs::rename(tpath, target);
  return rep;
}

} // namespace engines
} // namespace mlip
} // namespace tadah
