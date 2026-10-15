#include <tadah/mlip/engines/refit_keys.h>

#include <tadah/core/number_format.h>
#include <tadah/models/descriptors/basis_kind.h>
#include <tadah/models/descriptors/dm/embf0.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

using tadah::core::Context;

const std::string kWho = "tadah refit: ";
constexpr long long kIntMax = std::numeric_limits<int>::max();

// The tokens `key` stores, none when it stores none (no blueprint default is
// inserted: Context::has_stored, stored_value).
std::vector<std::string> tokens(const Context &c, const std::string &key) {
  std::vector<std::string> out;
  if (!c.has_stored(key)) return out;
  for (std::size_t i = 0;; ++i) {
    const std::optional<std::string> t = c.stored_value(key, i);
    if (!t) break;
    out.push_back(*t);
  }
  return out;
}

// The token of a scalar key, nullopt when it is absent. (Its blueprint
// value_N is 1: the config parser takes one value, and Context::add keeps one,
// overwriting it.)
std::optional<std::string> single(const Context &c, const std::string &key) {
  return c.stored_value(key, 0);
}

// " (too small: ...)" for a token in the number grammar whose magnitude is
// below the smallest normal double (parse_decimal_finite takes it as the
// subnormal or 0 it underflows to, parse_decimal_normal refuses it), as
// tadah hpo's number_underflow_note says it.
std::string too_small_note(const std::string &tok) {
  double v = 0.0;
  if (!core::parse_decimal_finite(tok, v) || core::parse_decimal_normal(tok, v))
    return "";
  return " (too small: its magnitude is below the smallest normal double, "
         "2.2250738585072014e-308)";
}

// TenthToOne: REFIT_REACH_SMIN. Its compressed lattice a = smin * a0 sets
// the neighbour lists of the self-test and the E-V scans, whose length grows
// as smin^-3 (a/a0 0.01 took a minute, 1e-6 did not finish), and the
// reachability probe (models::EAMRefit) lists neighbours out to rcut / 0.1
// only, so a compression past a/a0 0.1 is not supported.
enum class Range { Positive, NonNegative, UpToOne, OpenUnit, TenthToOne };

const char *range_text(Range r) {
  switch (r) {
    case Range::Positive: return "a positive number";
    case Range::NonNegative: return "a number >= 0";
    case Range::UpToOne: return "a number in (0, 1]";
    case Range::OpenUnit: return "a number in (0, 1)";
    case Range::TenthToOne: return "a number from 0.1 to 1";
  }
  return "";
}

bool in_range(double v, Range r) {
  switch (r) {
    case Range::Positive: return v > 0.0;
    case Range::NonNegative: return v >= 0.0;
    case Range::UpToOne: return v > 0.0 && v <= 1.0;
    case Range::OpenUnit: return v > 0.0 && v < 1.0;
    case Range::TenthToOne: return v >= 0.1 && v <= 1.0;
  }
  return false;
}

double real_token(const std::string &key, const std::string &tok, Range r) {
  double v = 0.0;
  const bool number = core::parse_decimal_normal(tok, v);
  if (number && in_range(v, r)) return v;
  throw std::runtime_error(kWho + key + " must be " + range_text(r) +
                           ", got '" + tok + "'" +
                           (number ? std::string() : too_small_note(tok)));
}

// A real key: absent -> nullopt.
std::optional<double> real(const Context &c, const std::string &key, Range r) {
  const std::optional<std::string> tok = single(c, key);
  if (!tok) return std::nullopt;
  return real_token(key, *tok, r);
}

// An integer: decimal digits only (std::from_chars, as tadah hpo's
// parse_uint: no sign, fraction, exponent or trailing character), within
// [lo, hi].
unsigned long long integer_token(const std::string &key, const std::string &tok,
                                 unsigned long long lo, unsigned long long hi) {
  unsigned long long v = 0;
  const char *first = tok.data(), *last = first + tok.size();
  const auto res = std::from_chars(first, last, v);
  if (!tok.empty() && res.ec == std::errc{} && res.ptr == last && v >= lo &&
      v <= hi)
    return v;
  throw std::runtime_error(kWho + key + " must be an integer from " +
                           std::to_string(lo) + " to " + std::to_string(hi) +
                           ", got '" + tok + "'");
}

std::optional<int> integer(const Context &c, const std::string &key,
                           long long lo, long long hi) {
  const std::optional<std::string> tok = single(c, key);
  if (!tok) return std::nullopt;
  return static_cast<int>(integer_token(key, *tok, lo, hi));
}

// ---- the mode rule ----

// Every key of a refit configuration that one REFIT_MODE reads and the other
// does not (check_refit_mode_keys); the others are read in both. Dense: the
// windowed least squares, models::EAMRefit::fit (its basis sizes, their
// preset and escalation, the placement, the soft F(0) pin of EMBF0 FREE),
// which compact mode never runs. Compact: the compact fitter of the HPO
// library and its stage blocks (RefitParser), which dense mode never runs.
struct ModeKey {
  const char *key;
  RefitMode mode;  ///< the one mode that reads it
  bool block;      ///< a multi-line block (KEY ... ENDKEY)
};
constexpr ModeKey kModeKeys[] = {
    {"REFIT_LEVEL", RefitMode::Dense, false},
    {"N2B", RefitMode::Dense, false},
    {"NRHO_BASIS", RefitMode::Dense, false},
    {"NF_BASIS", RefitMode::Dense, false},
    {"REFIT_MAX_ESCALATIONS", RefitMode::Dense, false},
    {"REFIT_PLACEMENT", RefitMode::Dense, false},
    {"REFIT_CONSTRAIN_F0", RefitMode::Dense, false},
    {"REFIT_COMPACT_NMAX", RefitMode::Compact, false},
    {"REFIT_COMPACT_SAMPLES", RefitMode::Compact, false},
    {"REFIT_COMPACT_OPT_EVERY", RefitMode::Compact, false},
    {"REFIT_COMPACT_STALL", RefitMode::Compact, false},
    {"REFIT_COMPACT_INSERT_TOPK", RefitMode::Compact, false},
    {"REFIT_COMPACT_JITTERS", RefitMode::Compact, false},
    {"REFIT_COMPACT_SEED", RefitMode::Compact, false},
    {"REFIT_COMPACT_SHARE", RefitMode::Compact, false},
    {"REFIT_COMPACT_MERGE", RefitMode::Compact, false},
    {"REFIT_CURVE_DERIV_WEIGHT", RefitMode::Compact, false},
    {"REFIT_CURVE_WEIGHTING", RefitMode::Compact, false},
    {"REFIT_CURVE_RELAX", RefitMode::Compact, false},
    {"REFIT_COMPONENT_ITERATIONS", RefitMode::Compact, false},
    {"INIT", RefitMode::Compact, true},
    {"EXPLORE", RefitMode::Compact, true},
    {"REFINE", RefitMode::Compact, true},
    {"FUSE", RefitMode::Compact, true}};
// (A top-level OPTIMIZER block, the grammar before the stage blocks, is read
// by neither mode: check_refit_mode_keys refuses it first, in both, with
// top_level_optimizer_refusal().)

// Read by a refit of an EAM file in either REFIT_MODE, and by neither
// --edit nor --retarget (check_refit_output_mode_keys); EAM itself has its
// own refusal there (validate_refit_mode_keys).
constexpr const char *kFitKeys[] = {
    "REFIT_MODE",         "REFIT_BASIS",          "REFIT_TOL_AMBIENT",
    "REFIT_TOL_COMPRESSED", "REFIT_REACH_SMIN",   "REFIT_REACH_MARGIN",
    "REFIT_R_LO_FRAC",    "REFIT_OBJECTIVE",      "OUTPUT_GRID_SCALE",
    "SETFL_OUT",          "REFIT_LOG"};

// Read by a refit of an EAM file and by --edit (the hpotarget they write),
// not by --retarget: its output is OUTFILE (-o), or its input with
// --in-place. (NUMERIC, the precision of the files a command writes, is no
// key of the refit and is not refused: --retarget writes only an
// hpotarget, whose bounds every hpotarget writer prints %.8e whatever
// NUMERIC is, so it has nothing to apply it to; a task file's NUMERIC
// above its first TASK line applies to every task.)
constexpr const char *kFitAndEditKeys[] = {"HPOFILEOUT"};

// Read by the knot families only (compact mode): the compact fitter applies
// the coalescence guard (CompactBlipFitter::merge_close_knots and the
// anti-coalescence factor), the insertion-only rungs and the top-k
// insertion to the one-sided knot kinds only; for the Blip families it
// takes no guard, every rung optimised and one insertion site.
constexpr const char *kKnotFamilyKeys[] = {
    "REFIT_COMPACT_MERGE", "REFIT_COMPACT_OPT_EVERY",
    "REFIT_COMPACT_INSERT_TOPK"};

// Read by REFIT_OBJECTIVE curve only (compact mode): the compact fitter
// reads the derivative weight in the curve objective's solve and score and
// the weighting and its floor in the curve weights only.
constexpr const char *kCurveObjectiveKeys[] = {
    "REFIT_CURVE_WEIGHTING", "REFIT_CURVE_RELAX", "REFIT_CURVE_DERIV_WEIGHT"};

const char *mode_name(RefitMode m) {
  return m == RefitMode::Dense ? "dense" : "compact";
}

std::string lower(std::string s) {
  for (auto &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

// The configuration sets `key`: it stores it, and not as the blueprint
// default the task factory added (KeyProvenance; none recorded: every stored
// key is the configuration's).
bool configured(const Context &c, const KeyProvenance *provenance,
                const std::string &key) {
  return refit_key_configured(c, provenance, key);
}

// The family a REFIT_BASIS spelling names (models::basis_kind_from_name, the
// one table: the fit reads it too), or none.
bool basis_is(const std::string &b, tadah::models::BasisKind kind) {
  const auto k = tadah::models::basis_kind_from_name(b);
  return k && *k == kind;
}

// A word key (REFIT_LEVEL, REFIT_PLACEMENT) in any letter case, one of
// `allowed`, else refused naming them; `fallback` without the key.
std::string word(const Context &c, const std::string &key,
                 const std::vector<std::string> &allowed,
                 const std::string &fallback) {
  const std::optional<std::string> tok = single(c, key);
  if (!tok) return fallback;
  const std::string v = lower(*tok);
  for (const std::string &a : allowed)
    if (v == a) return v;
  std::string list;
  for (std::size_t i = 0; i < allowed.size(); ++i) {
    list += (i == 0 ? "" : i + 1 == allowed.size() ? " or " : ", ") +
            ("'" + allowed[i] + "'") +
            (allowed[i] == fallback ? " (the default)" : "");
  }
  throw std::runtime_error(kWho + key + " must be " + list +
                           " (in any letter case), got '" + *tok + "'");
}

}  // namespace

bool refit_key_configured(const Context &c, const KeyProvenance *provenance,
                          const std::string &key) {
  return c.has_stored(key) &&
         !(provenance &&
           provenance->origin(key).kind == KeyOrigin::Kind::Default);
}

bool refit_basis_is_knot_family(const std::string &b) {
  return basis_is(b, tadah::models::BasisKind::Knot5);
}

bool refit_basis_is_blip(const std::string &b) {
  return basis_is(b, tadah::models::BasisKind::Blip);
}

bool refit_basis_is_blip5(const std::string &b) {
  return basis_is(b, tadah::models::BasisKind::Blip5);
}

bool refit_basis_is_gaussian(const std::string &b) {
  return basis_is(b, tadah::models::BasisKind::Gaussian);
}

void check_refit_basis(const Context &c) {
  if(c.blocks("DESCRIPTOR").empty() && c.has_stored("REFIT_PAIR_BASIS"))
    throw std::runtime_error("tadah refit: REFIT_PAIR_BASIS requires a DESCRIPTOR block; REFIT_BASIS controls all legacy components");
  const std::optional<std::string> b = single(c, "REFIT_BASIS");
  if (!b || tadah::models::basis_kind_from_name(*b)) return;
  throw std::runtime_error(
      kWho + "REFIT_BASIS '" + *b + "' is no basis family of tadah refit: "
      "Blip (Blip, BLIP, blip), Blip5 (Blip5, BLIP5, blip5, M6), the knot "
      "families Knot5, Knot3 and KnotMix (Knot5, KNOT5, knot5, Knot3, KNOT3, "
      "knot3, KnotMix, KNOTMIX, knotmix, Mixed, MIXED), or, under REFIT_MODE "
      "dense only, the experimental Gaussian (Gaussian, gaussian, BP, "
      "D2_BP)");
}

std::string read_refit_level(const Context &c) {
  return word(c, "REFIT_LEVEL", {"coarse", "balanced", "fine", "accurate"},
              "balanced");
}

std::string read_refit_placement(const Context &c) {
  return word(c, "REFIT_PLACEMENT", {"window", "uniform"}, "window");
}

RefitMode read_refit_mode(const Context &c) {
  const std::optional<std::string> tok = single(c, "REFIT_MODE");
  if (!tok) return RefitMode::Compact;
  const std::string m = lower(*tok);
  if (m == "compact") return RefitMode::Compact;
  if (m == "dense") return RefitMode::Dense;
  throw std::runtime_error(kWho +
                           "REFIT_MODE must be 'compact' (the default) or "
                           "'dense', got '" + *tok + "'");
}

std::string top_level_optimizer_refusal() {
  return kWho +
         "a top-level OPTIMIZER block is no longer supported. The stage "
         "pipeline is now INIT -> EXPLORE -> REFINE -> FUSE, each its own "
         "block, with the OPTIMIZER block NESTED inside EXPLORE/REFINE/FUSE. "
         "Migration: GLOBAL folded into EXPLORE rung 0 (set its global ALGO "
         "in EXPLORE's OPTIMIZER, with an INNER local); POLISH -> REFINE; "
         "JOINT -> FUSE. The Refitting page of the documentation has "
         "examples (Stage blocks).";
}

bool read_refit_curve_objective(const Context &c) {
  const std::optional<std::string> tok = single(c, "REFIT_OBJECTIVE");
  const std::string obj = tok ? lower(*tok) : std::string("energy");
  if (obj == "curve") return true;
  if (obj == "energy") return false;
  throw std::runtime_error(kWho +
                           "REFIT_OBJECTIVE must be 'energy' (default) or "
                           "'curve'; got '" + obj + "'");
}

void check_refit_mode_keys(const Context &c, RefitMode mode,
                           const KeyProvenance *provenance) {
  // neither mode reads a top-level OPTIMIZER block (its stages are the
  // compact fitter's stage blocks now, which dense mode does not read)
  if (configured(c, provenance, "OPTIMIZER"))
    throw std::runtime_error(
        top_level_optimizer_refusal() +
        (mode == RefitMode::Dense
             ? " Under REFIT_MODE dense, which reads no stage block, remove "
               "the OPTIMIZER block."
             : ""));
  const RefitMode other =
      mode == RefitMode::Dense ? RefitMode::Compact : RefitMode::Dense;
  const std::string runs = std::string(mode_name(mode)) +
                           (c.has_stored("REFIT_MODE") ? "" : " (the default)");
  const auto refuse = [&](const std::string &what) {
    throw std::runtime_error(kWho + what + " has no effect under REFIT_MODE " +
                             runs + ": only REFIT_MODE " + mode_name(other) +
                             " reads it; remove " + what +
                             ", or set REFIT_MODE " + mode_name(other));
  };
  for (const ModeKey &k : kModeKeys)
    if (k.mode != mode && configured(c, provenance, k.key))
      refuse(k.block ? std::string("the ") + k.key + " block"
                     : std::string(k.key));
  // the iteration limit of the component compact fit (DM_Embed blocks),
  // which a configuration without blocks (the compact ladder) does not run
  if (c.blocks("DESCRIPTOR").empty() &&
      configured(c, provenance, "REFIT_COMPONENT_ITERATIONS"))
    throw std::runtime_error(
        kWho + "REFIT_COMPONENT_ITERATIONS has no effect without DESCRIPTOR "
        "blocks: only the compact fit of DM_Embed components reads it (the "
        "compact ladder's budgets are the MAXEVAL of its stage blocks); "
        "remove REFIT_COMPONENT_ITERATIONS, or declare the many-body term "
        "as a DESCRIPTOR block");
  const bool curve = read_refit_curve_objective(c);
  if (mode == RefitMode::Dense) {
    if (curve) refuse("REFIT_OBJECTIVE curve");
    // REFIT_CONSTRAIN_F0 is the F(0) pin of an EMBF0 FREE fit: under EMBF0
    // ZERO (the default) F(0) = 0 by construction, and the key has no
    // effect, true or false (it was accepted, false with a WARNING)
    const std::optional<tadah::models::EmbF0Mode> embf0 =
        tadah::models::stored_embf0(c);
    if (embf0.value_or(tadah::models::EmbF0Mode::Zero) ==
            tadah::models::EmbF0Mode::Zero &&
        configured(c, provenance, "REFIT_CONSTRAIN_F0"))
      throw std::runtime_error(
          kWho + "REFIT_CONSTRAIN_F0 has no effect under EMBF0 ZERO" +
          (embf0 ? "" : " (the default)") +
          ": only EMBF0 FREE reads it; remove REFIT_CONSTRAIN_F0, or set "
          "EMBF0 FREE");
    return;
  }
  // the knot-family keys under a Blip family (REFIT_BASIS as written; no
  // REFIT_BASIS is the compact default, Knot5)
  if (const std::optional<std::string> basis = single(c, "REFIT_BASIS");
      basis && (refit_basis_is_blip(*basis) || refit_basis_is_blip5(*basis)))
    for (const char *key : kKnotFamilyKeys)
      if (configured(c, provenance, key))
        throw std::runtime_error(
            kWho + key + " has no effect under REFIT_BASIS " + *basis +
            ": only the knot families (REFIT_BASIS Knot5, the compact "
            "default, Knot3 or KnotMix) read it; remove " + key +
            ", or set REFIT_BASIS Knot5");
  if (!curve) {
    const std::string energy =
        std::string("energy") +
        (configured(c, provenance, "REFIT_OBJECTIVE") ? "" : " (the default)");
    for (const char *key : kCurveObjectiveKeys)
      if (configured(c, provenance, key))
        throw std::runtime_error(
            kWho + key + " has no effect under REFIT_OBJECTIVE " + energy +
            ": only REFIT_OBJECTIVE curve reads it; remove " + key +
            ", or set REFIT_OBJECTIVE curve");
    return;
  }
  // REFIT_CURVE_RELAX is the floor of the curve weighting, which
  // REFIT_CURVE_WEIGHTING false switches off (its default is true)
  const bool weighting = c.has_stored("REFIT_CURVE_WEIGHTING")
                             ? c.get<bool>("REFIT_CURVE_WEIGHTING")
                             : true;
  if (!weighting && configured(c, provenance, "REFIT_CURVE_RELAX"))
    throw std::runtime_error(
        kWho +
        "REFIT_CURVE_RELAX has no effect under REFIT_CURVE_WEIGHTING false: "
        "only REFIT_CURVE_WEIGHTING true (the default) reads it; remove "
        "REFIT_CURVE_RELAX, or set REFIT_CURVE_WEIGHTING true");
}

void check_refit_output_mode_keys(const Context &c, RefitOutputMode mode,
                                  const KeyProvenance *provenance) {
  const std::string flag =
      mode == RefitOutputMode::Edit ? "--edit" : "--retarget";
  const auto refuse = [&](const std::string &what, const std::string &reader,
                          const std::string &tail) {
    throw std::runtime_error("tadah refit " + flag + ": " + what +
                             " has no effect under " + flag + ": " + reader +
                             "; remove " + what + tail);
  };
  for (const char *key : kFitKeys)
    if (configured(c, provenance, key))
      refuse(key, "only a refit of an EAM file reads it", "");
  for (const ModeKey &k : kModeKeys)
    if (configured(c, provenance, k.key))
      refuse(k.block ? std::string("the ") + k.key + " block"
                     : std::string(k.key),
             std::string("only a refit of an EAM file in REFIT_MODE ") +
                 mode_name(k.mode) + " reads it",
             "");
  if (configured(c, provenance, "OPTIMIZER"))
    refuse("the OPTIMIZER block",
           "no mode of tadah refit reads a top-level OPTIMIZER block", "");
  if (mode == RefitOutputMode::Retarget)
    for (const char *key : kFitAndEditKeys)
      if (configured(c, provenance, key))
        refuse(key, "only a refit of an EAM file and --edit read it",
               " (the file --retarget writes is -o, OUTFILE)");
}

RefitNumbers read_refit_numbers(const Context &c) {
  RefitNumbers n;
  const auto set = [](auto &field, const auto &value) {
    if (value) field = *value;
  };
  set(n.tol_ambient, real(c, "REFIT_TOL_AMBIENT", Range::Positive));
  set(n.tol_compressed, real(c, "REFIT_TOL_COMPRESSED", Range::Positive));
  n.reach_smin = real(c, "REFIT_REACH_SMIN", Range::TenthToOne);
  n.reach_margin = real(c, "REFIT_REACH_MARGIN", Range::Positive);
  n.r_lo_frac = real(c, "REFIT_R_LO_FRAC", Range::OpenUnit);
  set(n.output_grid_scale, real(c, "OUTPUT_GRID_SCALE", Range::Positive));
  n.optim_pct = read_optim_pct(c);

  // N2B, NRHO_BASIS, NF_BASIS: the dense escalation's per-function cap is
  // 1024 basis functions (RefitEngine's n_cap).
  n.n2b = integer(c, "N2B", 1, 1024);
  n.nrho = integer(c, "NRHO_BASIS", 1, 1024);
  n.nF = integer(c, "NF_BASIS", 1, 1024);
  set(n.max_escalations, integer(c, "REFIT_MAX_ESCALATIONS", 0, kIntMax));

  // REFIT_COMPACT_NMAX: one value for all three functions, or phi rho F.
  const std::vector<std::string> nmax = tokens(c, "REFIT_COMPACT_NMAX");
  if (nmax.size() == 1) {
    n.nmax.fill(static_cast<int>(
        integer_token("REFIT_COMPACT_NMAX", nmax[0], 1, kIntMax)));
  } else if (nmax.size() == 3) {
    for (std::size_t i = 0; i < 3; ++i)
      n.nmax[i] = static_cast<int>(
          integer_token("REFIT_COMPACT_NMAX", nmax[i], 1, kIntMax));
  } else if (!nmax.empty()) {
    throw std::runtime_error(kWho +
                             "REFIT_COMPACT_NMAX takes 1 value (all "
                             "functions) or 3 (phi rho F), got " +
                             std::to_string(nmax.size()));
  }
  // The compact fit's curve sample grid has at least 4 x NMAX points per
  // function (REFIT_COMPACT_SAMPLES is raised to that), an int: computed in
  // 64 bits, the largest NMAX's must fit (4 * NMAX overflowed above
  // 536870911).
  {
    const int nmax_all = std::max({n.nmax[0], n.nmax[1], n.nmax[2]});
    const long long grid_min = 4LL * nmax_all;
    if (grid_min > kIntMax)
      throw std::runtime_error(
          kWho + "REFIT_COMPACT_NMAX " + std::to_string(nmax_all) +
          " needs a curve sample grid of 4 x " + std::to_string(nmax_all) +
          " = " + std::to_string(grid_min) +
          " points per function (REFIT_COMPACT_SAMPLES is raised to at least "
          "4 x REFIT_COMPACT_NMAX), above " + std::to_string(kIntMax) +
          "; lower REFIT_COMPACT_NMAX");
  }
  set(n.samples, integer(c, "REFIT_COMPACT_SAMPLES", 1, kIntMax));
  set(n.opt_every, integer(c, "REFIT_COMPACT_OPT_EVERY", 1, kIntMax));
  n.stall = integer(c, "REFIT_COMPACT_STALL", 1, kIntMax);
  n.insert_topk = integer(c, "REFIT_COMPACT_INSERT_TOPK", 1, kIntMax);
  // jitters + 1 candidates per rung, an int
  set(n.jitters, integer(c, "REFIT_COMPACT_JITTERS", 0, kIntMax - 1));
  if (const auto tok = single(c, "REFIT_COMPACT_SEED"))
    n.seed = static_cast<unsigned>(
        integer_token("REFIT_COMPACT_SEED", *tok, 0,
                      std::numeric_limits<std::uint32_t>::max()));
  set(n.share, real(c, "REFIT_COMPACT_SHARE", Range::Positive));
  set(n.curve_deriv_weight,
      real(c, "REFIT_CURVE_DERIV_WEIGHT", Range::NonNegative));
  set(n.curve_relax, real(c, "REFIT_CURVE_RELAX", Range::UpToOne));
  set(n.component_iterations,
      integer(c, "REFIT_COMPONENT_ITERATIONS", 1, kIntMax));
  return n;
}

double read_optim_pct(const Context &c) {
  const std::optional<double> v = real(c, "OPTIM_PCT", Range::Positive);
  return v ? *v : 5.0;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
