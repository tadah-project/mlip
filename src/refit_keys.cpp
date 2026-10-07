#include <tadah/mlip/engines/refit_keys.h>

#include <tadah/core/number_format.h>

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

}  // namespace

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
  return n;
}

double read_optim_pct(const Context &c) {
  const std::optional<double> v = real(c, "OPTIM_PCT", Range::Positive);
  return v ? *v : 5.0;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
