#include <tadah/mlip/engines/hpotarget_retarget.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/hpotarget_syntax.h>
#include <tadah/core/number_format.h>
#include <tadah/core/utils.h>

#include <tadah/models/descriptors/basis_kind.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
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
  std::string key;             // upper-cased: groups, --pct, --set
  std::string key_written;     // as written: the rewritten line keeps it
  std::string trailing;        // a comment after <high> (verbatim, kept)
  std::size_t index = 0;       // 1-based
  double lo = 0.0, hi = 0.0;
  double centre = 0.0;         // recovered value
  double v = 0.0, q = 0.0;     // grid-snapped centre + bound grid (rewrite)
  bool is_rcut = false;
};

// A commented OPTIM line: its indent, its comment prefix and its text.
const std::regex kCommentedOptimRe(R"(^(\s*)(#\s*)(OPTIM\s.*)$)");
const std::regex kHeaderPctRe(R"(Bounds are \+-([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)%)");
const std::regex kGroupHeaderRe(
    R"(^(#\s*)([A-Za-z0-9_.]+)(\s+\(\+-)([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)(%.*)$)");
// The knot-support cap the writer records for the knot centres of a DM_Embed
// density with CUTOFF TYPE NONE (knot_support_cap_line):
// `# TA.DENSITY.CENTRES knot-support cap: RADIUS = 4 (...)`.
const std::regex kKnotCapRe(
    R"(^#\s*([A-Za-z0-9_.]+)\s+knot-support cap:\s*RADIUS\s*=\s*(\S+).*$)");
// The bump-support contract the writer records next to a bump channel's
// commented RCUT scalar: `# RCUT2B bump-support contract: w = 3 (...)`.
// Files without it (knot channels, pre-contract emissions) get no capping.
const std::regex kBumpContractRe(
    R"(^#\s*RCUT(2B|MB)\s+bump-support contract:\s*w\s*=\s*([0-9]*\.?[0-9]+).*$)");
// The positive density form the writer records (rhoform_bounds_line):
// `# RHOFORM NONNEG: AMPGRIDMB bounds are non-negative (lo = max(0, v - d))`.
const std::regex kRhoFormRe(
    R"(^#\s*RHOFORM\s+(\S+)\s*:\s*AMPGRIDMB bounds are non-negative.*$)");

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

std::string rtrimmed(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

// " and "-joined, the last two; ", " between the others.
std::string join_reasons(const std::vector<std::string> &r) {
  std::string out;
  for (std::size_t i = 0; i < r.size(); ++i)
    out += (i == 0 ? "" : i + 1 == r.size() ? " and " : ", ") + r[i];
  return out;
}

// An OPTIM line --retarget keeps as written: the physical line (1-based)
// and its text, and the key and indices it selects where its syntax reads
// (empty key: unknown).
struct KeptLine {
  std::size_t line = 0;
  std::string text;
  std::string key;
  std::vector<std::size_t> indices;
};

// The OPTIM lines of `lines`: those --retarget rewrites, the lines
// `OPTIM <KEY> (<index>) <low> <high>` (active or commented: one physical
// line that tadah hpo's syntax reads, an explicit box, one index, no shape
// tag), and those it keeps as written, each with its WARNING. An active
// line is a logical line of the file whose key is OPTIM, as tadah hpo reads
// it; a commented one a physical line `# OPTIM ...` whose text (cut at the
// next '#') the syntax reads, a comment it refuses being prose.
struct Classified {
  std::vector<OptimLine> rewritten;
  std::vector<KeptLine> kept;
  std::vector<std::string> warnings;  // one per kept line, in file order
};

Classified classify_optim_lines(const std::vector<std::string> &lines) {
  static const char *const kForm =
      "--retarget rewrites the lines OPTIM <KEY> (<index>) <low> <high>";
  struct Entry { OptimLine ol; bool rewrite = false; KeptLine kept;
                 std::string warning; };
  std::map<std::size_t, Entry> entries;  // by physical line, 0-based

  // The reasons a line read by the syntax is not of the rewritten form.
  auto form_reasons = [](const OptimSyntax &syn) {
    std::vector<std::string> r;
    if (syn.box.auto_box) r.push_back("has an AUTO box");
    if (syn.selector.indices.size() != 1)
      r.push_back("selects " + std::to_string(syn.selector.indices.size()) +
                  " indices");
    if (syn.box.tagged)
      r.push_back("has the shape tag " +
                  std::string(syn.box.shape == OptimShape::Log     ? "LOG"
                              : syn.box.shape == OptimShape::Asinh ? "ASINH " +
                                                                         syn.box.c_tok
                                                                   : "LIN"));
    return r;
  };
  auto kept_warning = [](std::size_t line, const std::string &text,
                         const std::vector<std::string> &reasons) {
    return "line " + std::to_string(line) + " ('" + text +
           "') is kept as written: " + kForm + ", and this line " +
           join_reasons(reasons) + ".";
  };
  // The text after the box of a one-line OPTIM line: its comment.
  auto trailing_of = [](const std::string &physical, std::size_t text_end) {
    const auto hash = physical.find(kTargetComment, text_end);
    return hash == std::string::npos ? std::string()
                                     : physical.substr(text_end);
  };

  // The active OPTIM lines, as tadah hpo reads the file.
  std::string joined;
  for (const auto &l : lines) joined += l + "\n";
  std::istringstream in(joined);
  for (const TargetLine &tl : target_logical_lines(in)) {
    if (tl.key != "OPTIM") continue;
    const std::size_t p = tl.first - 1;
    Entry e;
    e.kept.line = tl.first;
    e.kept.text = rtrimmed(lines[p]);
    std::vector<std::string> reasons;
    if (tl.last > tl.first)
      reasons.push_back(tl.last == tl.first + 1
                            ? "continues on line " + std::to_string(tl.last)
                            : "continues on lines " +
                                  std::to_string(tl.first + 1) + " to " +
                                  std::to_string(tl.last));
    std::optional<OptimSyntax> syn;
    if (tl.unfinished) {
      reasons.push_back(std::string("is one tadah hpo refuses (") +
                        kTargetUnfinishedLineMsg + ")");
    } else {
      std::istringstream rest(tl.text);
      std::string word;
      rest >> word;
      try {
        syn = read_optim_syntax(rest);
      } catch (const std::exception &x) {
        reasons.push_back(std::string("is one tadah hpo refuses (") +
                          x.what() + ")");
        std::istringstream ks(rest.str());
        std::string w, k;
        if (ks >> w >> k) e.kept.key = tadah::core::to_upper(k);
      }
    }
    if (syn) {
      e.kept.key = syn->key;
      e.kept.indices = syn->selector.indices;
      for (auto &r : form_reasons(*syn)) reasons.push_back(r);
    }
    if (reasons.empty()) {
      const std::string &l = lines[p];
      e.rewrite = true;
      e.ol.line = p;
      e.ol.indent = l.substr(0, l.find_first_not_of(" \t"));
      e.ol.key = syn->key;
      e.ol.key_written = syn->key_as_written;
      e.ol.index = syn->selector.indices.front();
      e.ol.lo = syn->box.low;
      e.ol.hi = syn->box.high;
      e.ol.trailing = trailing_of(l, tl.text.size());
    } else {
      e.warning = kept_warning(tl.first, e.kept.text, reasons);
    }
    entries[p] = std::move(e);
  }

  // The commented OPTIM lines (a comment never continues).
  for (std::size_t p = 0; p < lines.size(); ++p) {
    std::smatch m;
    if (!std::regex_match(lines[p], m, kCommentedOptimRe)) continue;
    std::string text = m[3].str();
    const auto hash = text.find(kTargetComment);
    if (hash != std::string::npos) text.resize(hash);
    text = rtrimmed(text);
    std::istringstream rest(text);
    std::string word;
    rest >> word;
    OptimSyntax syn;
    try {
      syn = read_optim_syntax(rest);
    } catch (const std::exception &) {
      continue;  // prose, or a line tadah hpo would refuse: a comment
    }
    Entry e;
    e.kept.line = p + 1;
    e.kept.text = rtrimmed(lines[p]);
    e.kept.key = syn.key;
    e.kept.indices = syn.selector.indices;
    const std::vector<std::string> reasons = form_reasons(syn);
    if (reasons.empty()) {
      e.rewrite = true;
      e.ol.line = p;
      e.ol.indent = m[1].str();
      e.ol.prefix = m[2].str();
      e.ol.key = syn.key;
      e.ol.key_written = syn.key_as_written;
      e.ol.index = syn.selector.indices.front();
      e.ol.lo = syn.box.low;
      e.ol.hi = syn.box.high;
      e.ol.trailing = trailing_of(lines[p], m.position(3) + text.size());
    } else {
      e.warning = kept_warning(p + 1, e.kept.text, reasons);
    }
    entries[p] = std::move(e);
  }

  Classified out;
  for (auto &[p, e] : entries) {
    if (e.rewrite) {
      out.rewritten.push_back(std::move(e.ol));
    } else {
      out.kept.push_back(std::move(e.kept));
      out.warnings.push_back(std::move(e.warning));
    }
  }
  return out;
}

bool group_positive_only(const std::string &key, bool &known) {
  if(key.find('.')!=std::string::npos) {
    const auto leaf=tadah::core::to_upper(key.substr(key.find_last_of('.')+1));
    known=leaf=="CENTRES" || leaf=="AMPLITUDES" || leaf=="ETA" || leaf=="WIDTHS" || leaf=="RADIUS" || leaf=="CUTOFF";
    return leaf=="ETA" || leaf=="WIDTHS" || leaf=="RADIUS" || leaf=="CUTOFF";
  }
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
  std::set<std::string> nonnegative_components;
  std::set<std::string> component_ids;  // the IDs of COMPONENT_PARAMETER paths
  std::string amplitude_paths;          // their <ID>.DENSITY.AMPLITUDES
  for(const auto &line:lines) {
    std::istringstream in(line);std::string comment,marker,path,rule;
    if(!(in>>comment>>marker>>path>>rule) || comment!="#" || marker!="COMPONENT_PARAMETER") continue;
    path=tadah::core::to_upper(path);
    if(rule=="NONNEG") nonnegative_components.insert(path);
    const std::string id=path.substr(0,path.find('.'));
    if(component_ids.insert(id).second)
      amplitude_paths+=(amplitude_paths.empty()?"":", ")+id+".DENSITY.AMPLITUDES";
  }
  // The knot-support caps (writer-recorded): path -> RADIUS.
  std::map<std::string, double> knot_cap;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (!std::regex_match(lines[i], m, kKnotCapRe)) continue;
    double r = 0.0;
    if (!tadah::core::parse_decimal_finite(m[2].str(), r) || !(r > 0.0))
      throw std::runtime_error("retarget: line " + std::to_string(i + 1) +
                               ": the knot-support cap '" + lines[i] +
                               "' names no positive RADIUS.");
    knot_cap[tadah::core::to_upper(m[1].str())] = r;
  }

  // --rhoform and --rcutmb name the legacy many-body keys (AMPGRIDMB,
  // RCUTMB): on a file whose many-body parameters are DM_Embed components
  // only, they have nothing to act on. The form of a component is its FORM,
  // recorded in the COMPONENT_PARAMETER rule of <ID>.DENSITY.AMPLITUDES;
  // its radius is the CUTOFF RADIUS, the path <ID>.DENSITY.CUTOFF.RADIUS.
  if (!component_ids.empty()) {
    bool legacy_amplitudes = false, legacy_density = false;
    for (const auto &ol : classify_optim_lines(lines).rewritten) {
      const std::string &key = ol.key;
      legacy_amplitudes |= key == "AMPGRIDMB";
      legacy_density |= key == "CGRIDMB" || key == "SGRIDMB" || key == "RCUTMB";
    }
    std::string radius_paths;
    for (const auto &id : component_ids)
      radius_paths += (radius_paths.empty() ? "" : ", ") + id +
                      ".DENSITY.CUTOFF.RADIUS";
    if (opts.rhoform && !legacy_amplitudes)
      throw std::runtime_error(
          std::string("retarget: --rhoform ") +
          tadah::models::rhoform_name(*opts.rhoform) +
          " has nothing to act on in this file: its density amplitudes are "
          "the DM_Embed component path(s) " + amplitude_paths + ", whose "
          "form is the component's FORM (the DENSITY FORM line of the "
          "potential's block), recorded in their COMPONENT_PARAMETER rule "
          "(NONNEG for FORM NONNEG or MONOTONE, FREE for SIGNED), and there "
          "is no AMPGRIDMB group; retarget without --rhoform, or write the "
          "target from a potential of that FORM (tadah refit --edit).");
    if (opts.rcut_override.count("MB") && !legacy_density)
      throw std::runtime_error(
          "retarget: --rcutmb has nothing to act on in this file: its "
          "many-body parameters are DM_Embed component paths, and a "
          "component's radius is its CUTOFF RADIUS, the path(s) " +
          radius_paths + " (set RADIUS in the potential's block, or search "
          "it with an OPTIM line on that path when its CUTOFF TYPE is not "
          "NONE); there is no CGRIDMB, SGRIDMB or RCUTMB group; retarget "
          "without --rcutmb.");
  }

  // ---- the density form (writer-recorded header line, or --rhoform) ----
  // Under NONNEG/MONOTONE the AMPGRIDMB boxes are non-negative: a lo-clamped
  // line is repaired from its upper bound and every rewrite keeps lo >= 0.
  std::optional<tadah::models::RhoForm> form;
  std::size_t form_line = 0;  // 1-based; 0 = no header line
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (!std::regex_match(lines[i], m, kRhoFormRe)) continue;
    try {
      form = tadah::models::parse_rhoform(m[1].str());
    } catch (const std::exception &e) {
      throw std::runtime_error("retarget: line " + std::to_string(i + 1) +
                               ": the RHOFORM header line '" + lines[i] +
                               "' names no density form: " + e.what());
    }
    form_line = i + 1;
    break;
  }
  // --rhoform: the form of the potential the file searches around. A header
  // records a positive form only, so a header that names another form, or
  // any header under --rhoform SIGNED, is a conflict.
  if (opts.rhoform) {
    const char *want = tadah::models::rhoform_name(*opts.rhoform);
    if (form && *form != *opts.rhoform)
      throw std::runtime_error(
          std::string("retarget: --rhoform ") + want +
          " conflicts with the file: line " + std::to_string(form_line) +
          " declares RHOFORM " + tadah::models::rhoform_name(*form) +
          " (the form its AMPGRIDMB bounds were written for). An hpotarget "
          "searches around one potential, whose density form does not "
          "change; write the target from that potential (tadah refit "
          "--edit) or retarget without --rhoform.");
    if (!form && tadah::models::rhoform_is_positive(*opts.rhoform)) {
      // The writer's place: after the intro and EMBF0 comments, before
      // the '#' that opens the "Search space: ... Bounds are +-X%" block.
      std::size_t at = lines.size();
      for (std::size_t i = 0; i < lines.size(); ++i)
        if (std::regex_search(lines[i], kHeaderPctRe)) {
          at = (i > 0 && lines[i - 1] == "#") ? i - 1 : i;
          break;
        }
      if (at == lines.size()) {  // no such block: end of the leading comments
        at = 0;
        while (at < lines.size() && !lines[at].empty() && lines[at][0] == '#')
          ++at;
      }
      lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(at),
                   rhoform_bounds_line(*opts.rhoform));
      form = opts.rhoform;
    }
  }
  if (form) rep.rhoform = tadah::models::rhoform_name(*form);

  // ---- collect every OPTIM line (active and commented) ----
  // The lines OPTIM <KEY> (<index>) <low> <high> are rewritten; an OPTIM
  // line of another form (an AUTO box, several indices, a shape tag, a
  // continued line, a line tadah hpo refuses) is kept as written, with a
  // WARNING naming it.
  Classified classified = classify_optim_lines(lines);
  std::vector<OptimLine> &ols = classified.rewritten;
  for (auto &w : classified.warnings) rep.warnings.push_back(std::move(w));
  for (auto &ol : ols) {
    ol.is_rcut = (ol.key == "RCUT2B" || ol.key == "RCUTMB");
    ol.centre = 0.5 * (ol.lo + ol.hi);
  }
  // The lines of KEY kept as written, "line N ('...')" each.
  auto kept_of = [&classified](const std::string &key) {
    std::vector<std::string> out;
    for (const auto &k : classified.kept)
      if (k.key == key)
        out.push_back("line " + std::to_string(k.line) + " ('" + k.text +
                      "')");
    return out;
  };
  // --pct KEY=N applies to the OPTIM lines on KEY (active or commented):
  // a key without one is a mistake the rewrite would otherwise hide.
  {
    std::set<std::string> keys;
    for (const auto &ol : ols) keys.insert(ol.key);
    for (const auto &[key, p] : opts.pct_per_key) {
      if (keys.count(key)) continue;
      const std::vector<std::string> kept = kept_of(key);
      if (!kept.empty())
        throw std::runtime_error(
            "--pct " + key + "=" + pct_str(p) + ": no OPTIM " + key +
            " line of the file is rewritten: " + join_reasons(kept) +
            (kept.size() == 1 ? " is" : " are") + " kept as written.");
      std::string avail;
      for (const auto &k : keys) avail += (avail.empty() ? "" : ", ") + k;
      throw std::runtime_error(
          "--pct " + key + "=" + pct_str(p) + ": the file has no OPTIM " + key +
          " line (active or commented); its OPTIM keys: " +
          (avail.empty() ? std::string("none") : avail) + ".");
    }
  }
  if (ols.empty()) {
    // An OPTIM line kept as written is named by its WARNING above.
    if (!classified.kept.empty()) return rep;
    rep.warnings.push_back("no OPTIM lines found — nothing to rewrite.");
    return rep;
  }

  // ---- bump-support contracts (writer-recorded metadata comments) ----
  std::map<std::string, double> bump_w;  // suffix "2B"/"MB" -> half-width w
  for (const auto &l : lines) {
    std::smatch m;
    if (std::regex_match(l, m, kBumpContractRe))
      bump_w[m[1].str()] = tadah::core::to_double(m[2].str());
  }
  auto bump_cgrid_key = [&bump_w](const std::string &key) {
    return (key == "CGRID2B" && bump_w.count("2B")) ||
           (key == "CGRIDMB" && bump_w.count("MB"));
  };

  auto nonneg_key = [&form,&nonnegative_components](const std::string &key) {
    if(nonnegative_components.count(tadah::core::to_upper(key)))return true;
    return tadah::models::optim_key_lower_bound(key, form) ==
           tadah::models::OptimLowerBound::NonNegative;
  };

  // ---- emission percentage from the file header (for clamp repair) ----
  double hdr_pct = -1.0;
  for (const auto &l : lines) {
    std::smatch m;
    if (std::regex_search(l, m, kHeaderPctRe)) {
      hdr_pct = tadah::core::to_double(m[1].str());
      break;
    }
  }
  if (hdr_pct <= 0.0)
    rep.warnings.push_back(
        "no 'Bounds are +-X%' header found — asymmetric (clamped) bounds "
        "cannot be repaired exactly; midpoints are used as centres.");

  // ---- per-group percentages (the `# KEY  (+-X%` group comments) ----
  // A per-key retarget (--pct 10 AMPGRIDMB=50) rewrites the group's own
  // comment below, so a group's lines follow its comment's percentage, not
  // the file header's. Groups without a comment (RCUT scalars) fall back to
  // the header.
  std::map<std::string, double> group_pct;
  for (const auto &l : lines) {
    std::smatch m;
    if (std::regex_match(l, m, kGroupHeaderRe))
      group_pct[tadah::core::to_upper(m[2].str())] =
          tadah::core::to_double(m[4].str());
  }

  // ---- group by key; repair clamped centres for positive-only groups ----
  std::map<std::string, std::vector<OptimLine *>> groups;
  for (auto &ol : ols) groups[ol.key].push_back(&ol);

  // ---- --rcut2b/--rcutmb: declared FIXED cutoffs for bump channels ----
  // Forced centres for the RCUT reference lines (the repair loop skips
  // them); the bump-support cap below binds against the declared value.
  std::set<std::string> forced_rcut;  // rkey names with forced centres
  for (const auto &[sfx, r] : opts.rcut_override) {
    const std::string flag = std::string("--rcut") + (sfx == "2B" ? "2b" : "mb");
    const std::string ckey = "CGRID" + sfx;
    const std::string skey = "SGRID" + sfx;
    const std::string rkey = "RCUT" + sfx;
    if (groups.count(ckey) && !groups.count(skey)) {
      rep.warnings.push_back(
          flag + " is ignored: " + ckey + " is a knot channel, and " + rkey +
          " is its largest knot, which the " + ckey + " boxes move.");
      continue;
    }
    forced_rcut.insert(rkey);
    if (!groups.count(rkey)) {
      rep.warnings.push_back(
          rkey + ": no OPTIM reference line in the file — the declared "
          "cutoff " + sci8(r) + " still binds the bump-support cap; the fit "
          "takes " + rkey + " from the training configuration, which " +
          flag + " does not change.");
      continue;
    }
    for (auto *ol : groups[rkey]) {
      if (ol->prefix.empty()) {
        ol->prefix = "# ";
        rep.warnings.push_back(
            "OPTIM " + rkey + " (" + std::to_string(ol->index) +
            "): ACTIVE line commented out — " + flag + " declares a FIXED "
            "cutoff, and searching RCUT on a bump channel is a wasted "
            "dimension (the bump-support gate keeps every bump inside the "
            "cutoff's LOW bound); the fit takes " + rkey + " from the "
            "training configuration, which " + flag + " does not change.");
      }
      ol->centre = r;
    }
  }

  for (auto &[key, group] : groups) {
    bool known = false;
    const bool positive_only = group_positive_only(key, known);
    const bool nonneg = nonneg_key(key);
    const bool is_rcut = group.front()->is_rcut;
    if (is_rcut && forced_rcut.count(key))
      continue;  // centre declared via --rcut2b/--rcutmb: nothing to recover
    if (!known && !is_rcut)
      rep.warnings.push_back(
          "OPTIM key '" + group.front()->key_written +
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
              "group '" + group.front()->key_written +
              "' has non-contiguous indices — its scale "
              "floor is computed from the surviving lines only.");
          break;
        }
      }
    }

    // Midpoints are exact except for: positive-only and non-negative groups
    // (lo can be clamped), RCUT scalars, and bump-contract CGRID groups (hi
    // can be capped by the bump-support rule) — those need the
    // rule-consistency check + one-sided recovery below.
    const auto cap = knot_cap.find(tadah::core::to_upper(key));
    const bool is_capped = cap != knot_cap.end();
    const bool is_bump_cgrid = bump_cgrid_key(key) || is_capped;
    if (!positive_only && !nonneg && !is_rcut && !is_bump_cgrid) continue;
    const auto gp = group_pct.find(key);
    const double pct = gp != group_pct.end() ? gp->second : hdr_pct;
    if (pct <= 0.0) continue;  // cannot check/repair without the pct
    const double p = pct / 100.0;

    // group scale from the midpoints — below 100% the max element is
    // (essentially) never lo-clamped, so it is unbiased. From 100% on a
    // non-negative group clamps every box at lo = 0, the largest included:
    // its midpoint is (1 + p) v / 2, while hi / (1 + p) = v exactly. The
    // smaller of the two is v for a clamped box with d = p v and never above
    // the midpoint, so the max stays exact.
    double gmax = 0.0;
    for (auto *ol : group) {
      double s = std::fabs(ol->centre);
      if (nonneg && ol->lo == 0.0) s = std::min(s, ol->hi / (1.0 + p));
      gmax = std::max(gmax, s);
    }
    const double floor = is_rcut ? 1e-6 : optim_group_floor(gmax);

    // A clamped line cannot be detected by bound asymmetry (every interval
    // is symmetric about its own midpoint): instead check whether the
    // OBSERVED half-width matches the emission rule at the midpoint. When
    // it does not, one bound was moved off the symmetric rule and the OTHER
    // one is authoritative: positive-only and non-negative groups / RCUT get
    // the lo clamp (hi = v + max(p·|v|, floor) recovers v); bump-contract
    // CGRID groups get the hi cap (lo = v - max(p·|v|, floor) recovers v).
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
      if (nonneg) v = std::max(0.0, v);  // a non-negative amplitude
      const double qv = optim_bound_quantum(v, std::max(p * std::fabs(v),
                                                        floor));
      v = quantize_optim(v, qv);
      ol->centre = v;
      // lo = 0 in a non-negative group is the clamp the file's RHOFORM line
      // declares (lo = max(0, v - d)): recovered without a warning
      if (nonneg && ol->lo == 0.0) continue;
      // hi at the knot-support cap is the cap the file declares
      if (is_capped && from_lo &&
          ol->hi == quantize_optim_down(cap->second, qv))
        continue;
      rep.warnings.push_back(
          "OPTIM " + ol->key_written + " (" + std::to_string(ol->index) +
          "): bounds inconsistent with the +-" + pct_str(pct) +
          (from_lo
               ? "% emission rule (hi capped by the bump-support contract "
                 "or hand-edited) — centre recovered from the lower bound: "
               : "% emission rule (lo-clamped or hand-edited) — centre "
                 "recovered from the upper bound: ") +
          sci8(v) + ".");
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
    const bool nonneg = nonneg_key(ol.key);
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
      if (nonneg && lo < 0.0) lo = 0.0;  // the writer's lo = max(0, v - d)
      if (hi <= lo) hi = lo + std::max(floor, 1e-6);
      // the writer's knot-support cap: the box ends at RADIUS
      if (const auto cap = knot_cap.find(tadah::core::to_upper(ol.key));
          cap != knot_cap.end()) {
        hi = std::min(hi, quantize_optim_down(cap->second, q));
        if (!(hi > lo))
          throw std::runtime_error(
              "retarget: OPTIM " + ol.key_written + " (" +
              std::to_string(ol.index) + "): the knot centre " + sci8(v) + " is not inside the "
              "knot-support cap RADIUS = " + sci8(cap->second) +
              " the file declares.");
      }
    }
    ol.lo = lo;
    ol.hi = hi;
    ol.v = v;
    ol.q = q;
  }

  // ---- bump-support contract diagnostics (contract files only) ----
  // Bounds are no longer capped: `tadah hpo` (BUMPGATE CANDIDATE, the
  // default) accepts partially infeasible boxes at full width and rejects
  // violating candidates per evaluation, so retarget rewrites symmetric
  // bounds untouched — retarget(fresh, same pct) stays a fixed point
  // trivially. The parsed contract still drives two warnings: a seed
  // whose own bumps spill (hpo's startup SEED check rejects such a file),
  // and the wasted-dimension note for an ACTIVE OPTIM RCUT line.
  for (const auto &[sfx, w] : bump_w) {
    const std::string ckey = "CGRID" + sfx;
    const std::string skey = "SGRID" + sfx;
    const std::string rkey = "RCUT" + sfx;
    const auto ov = opts.rcut_override.find(sfx);
    const bool has_ov = ov != opts.rcut_override.end();
    if (!groups.count(ckey) || !groups.count(skey)) continue;
    if (!groups.count(rkey) && !has_ov) continue;
    // The seed's cutoff: an explicit --rcut2b/--rcutmb wins. Otherwise the
    // group is normally ONE commented scalar whose centre IS the fixed
    // rcut; with active/duplicate lines, use the smallest reachable value.
    double rcut = std::numeric_limits<double>::max();
    if (has_ov) {
      rcut = ov->second;
    } else {
      bool any_active = false;
      for (const auto *ol : groups[rkey]) {
        rcut = std::min(rcut, ol->prefix.empty() ? ol->lo : ol->centre);
        any_active |= ol->prefix.empty();
      }
      if (any_active)
        rep.warnings.push_back(
            rkey + ": an ACTIVE OPTIM " + rkey + " line is usually a "
            "wasted dimension on a bump channel — every feasible "
            "candidate's bumps vanish at their own support edge inside "
            "the cutoff, so the sampled value only widens or narrows the "
            "feasible (c, eta) region. Keep ONE commented line centred on "
            "the fixed " + rkey + " and set " + rkey + " in the config — "
            "or declare it in one step with --rcut2b/--rcutmb.");
    }
    const double tol = tadah::models::bump_rcut_tol(rcut);
    std::map<std::size_t, OptimLine *> smap;
    for (auto *ol : groups[skey]) smap[ol->index] = ol;
    std::string spilled;
    for (auto *c : groups[ckey]) {
      const auto s = smap.find(c->index);
      if (s == smap.end()) continue;
      if (s->second->v <= 0.0 || c->v + w / s->second->v > rcut + tol)
        spilled += (spilled.empty() ? "" : ", ") + std::to_string(c->index);
    }
    if (!spilled.empty())
      rep.warnings.push_back(
          ckey + " bump(s) " + spilled + " spill past " + rkey + " = " +
          sci8(rcut) + " at their own centres — the pot itself is invalid "
          "under a hard cutoff; `tadah hpo` rejects a spilling seed at "
          "startup until it is repaired (tadah refit --edit).");
  }

  // ---- absolute --set overrides (applied last) ----
  for (const auto &s : opts.sets) {
    bool found = false;
    for (auto &ol : ols) {
      if (ol.key == s.key && ol.index == s.index) {
        const std::string what =
            "--set " + s.key + "(" + std::to_string(s.index) + "): ";
        if (!std::isfinite(s.lo))
          throw std::runtime_error(what + "lo must be a finite number.");
        if (!std::isfinite(s.hi))
          throw std::runtime_error(what + "hi must be a finite number.");
        if (!(s.hi > s.lo))
          throw std::runtime_error(what + "hi must be greater than lo.");
        // The density form's rule on the low bound (the file names no
        // potential, so no basis: models::optim_low_bound_rule).
        if (const auto rule =
                tadah::models::optim_low_bound_rule(s.key, form, std::nullopt);
            rule && !rule->admits(s.lo))
          throw std::runtime_error(
              what + "lo must be " + (rule->strict ? "> 0" : ">= 0") +
              ": the file declares RHOFORM " + rep.rhoform + ", which "
              "requires " + rule->requirement + " (`tadah hpo` refuses such "
              "a box under that form).");
        ol.lo = s.lo;
        ol.hi = s.hi;
        found = true;
        break;
      }
    }
    if (!found) {
      for (const auto &k : classified.kept)
        if (k.key == s.key &&
            std::find(k.indices.begin(), k.indices.end(), s.index) !=
                k.indices.end())
          throw std::runtime_error(
              "--set " + s.key + "(" + std::to_string(s.index) + "): line " +
              std::to_string(k.line) + " ('" + k.text + "'), which selects " +
              s.key + " (" + std::to_string(s.index) + "), is kept as "
              "written: --set applies to the lines OPTIM <KEY> (<index>) "
              "<low> <high>.");
      std::string avail;
      for (const auto &[key, group] : groups) {
        avail += "\n  " + key + " (1-" + std::to_string(group.size()) + ")";
      }
      throw std::runtime_error(
          "--set " + s.key + "(" + std::to_string(s.index) +
          "): no such OPTIM line in the file. Available:" + avail);
    }
  }

  // A --set on a bump channel always wins, but warn when the chosen box
  // has an infeasible corner: those combinations are per-candidate rejects
  // in `tadah hpo` (and a startup error under BUMPGATE STRICT).
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
          " (bump-support contract) — `tadah hpo` rejects those "
          "combinations per evaluation (or the whole file at startup "
          "under BUMPGATE STRICT).");
  }

  // ---- serialise the rewritten lines ----
  for (const auto &ol : ols) {
    lines[ol.line] = ol.indent + ol.prefix + "OPTIM " + ol.key_written +
                     " (" + std::to_string(ol.index) + ") " + sci8(ol.lo) +
                     " " + sci8(ol.hi) + ol.trailing;
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
    if (std::regex_match(l, m, kGroupHeaderRe) &&
        groups.count(tadah::core::to_upper(m[2].str()))) {
      const auto it =
          opts.pct_per_key.find(tadah::core::to_upper(m[2].str()));
      const double pct = it != opts.pct_per_key.end() ? it->second : opts.pct;
      l = m[1].str() + m[2].str() + m[3].str() + pct_str(pct) + m[5].str();
    }
  }

  // ---- the percentage of an RCUT scalar line ----
  // The writer gives an RCUT line no group comment: its percentage is the
  // file header's. A per-key one (--pct RCUT2B=50) is recorded in a group
  // comment above the line, "# RCUT2B  (+-50%, 1 params)", which the next
  // retarget reads (group_pct) to recover the centre; the comment goes
  // when the line's percentage is the global one again, so the file is
  // then the writer's.
  for (const char *rkey : {"RCUT2B", "RCUTMB"}) {
    if (!groups.count(rkey)) continue;
    const auto it = opts.pct_per_key.find(rkey);
    const bool own = it != opts.pct_per_key.end() && it->second != opts.pct;
    std::set<std::size_t> rcut_lines;  // the rewritten lines of rkey
    for (const auto *ol : groups.at(rkey)) rcut_lines.insert(ol->line);
    std::vector<std::string> out;
    out.reserve(lines.size() + 1);
    // Where each line goes, so the rewritten lines of the next key are
    // found where they are after this one's comment comes or goes.
    std::vector<std::size_t> moved_to(lines.size(), 0);
    bool placed = false;
    for (std::size_t i = 0; i < lines.size(); ++i) {
      const std::string &l = lines[i];
      std::smatch m;
      if (std::regex_match(l, m, kGroupHeaderRe) &&
          tadah::core::to_upper(m[2].str()) == rkey) {
        if (own && !placed) {
          moved_to[i] = out.size();
          out.push_back(l);  // refreshed above
          placed = true;
        }
        continue;
      }
      if (own && !placed && rcut_lines.count(i)) {
        out.push_back("# " + std::string(rkey) + "  (+-" +
                      pct_str(it->second) + "%, " +
                      std::to_string(groups.at(rkey).size()) + " params)");
        placed = true;
      }
      moved_to[i] = out.size();
      out.push_back(l);
    }
    lines.swap(out);
    for (auto &ol : ols) ol.line = moved_to[ol.line];
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
