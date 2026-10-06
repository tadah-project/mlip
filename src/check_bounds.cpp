#include <tadah/mlip/check_bounds.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/dm/rhoform.h>

namespace tadah {
namespace mlip {

namespace {

// Parse `(N)` or `(lo-hi)` index spec. Returns the inclusive range.
// `(1)` means scalar — represented as (-1, -1). `(1-4)` means vector
// entries with arg_pos 0..3 (1-based indexing in the file).
std::pair<int, int> parse_index_spec(const std::string& s) {
  // Expect one of: "(N)", "(lo-hi)", or no parens (scalar).
  if (s.empty()) return {-1, -1};
  std::string body = s;
  if (body.front() == '(') body.erase(0, 1);
  if (!body.empty() && body.back() == ')') body.pop_back();
  if (body.empty()) return {-1, -1};
  auto dash = body.find('-');
  if (dash == std::string::npos) {
    // Single index N (1-based) -> arg_pos = N-1.
    int n = std::stoi(body);
    if (n == 1) return {-1, -1};   // (1) is the scalar case
    return {n - 1, n - 1};
  }
  int lo = std::stoi(body.substr(0, dash));
  int hi = std::stoi(body.substr(dash + 1));
  return {lo - 1, hi - 1};
}

}  // namespace

std::vector<Bound> parse_hpo_target_bounds(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(
        "check_bounds: cannot open HPOTARGET file '" + path + "'");
  }
  std::vector<Bound> out;
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    // Strip leading whitespace and comments.
    std::size_t p = line.find_first_not_of(" \t");
    if (p == std::string::npos) continue;
    if (line[p] == '#') continue;
    if (line.compare(p, 5, "OPTIM") != 0) continue;
    // Require a whitespace separator after `OPTIM` so the parser
    // doesn't false-match `OPTIMIZER` / `OPTIMIZER_ENDOPTIMIZER`.
    if (p + 5 >= line.size()) continue;
    const char next = line[p + 5];
    if (next != ' ' && next != '\t') continue;
    std::istringstream iss(line.substr(p + 5));
    std::string key, idx;
    if (!(iss >> key)) continue;
    if (!(iss >> idx)) {
      throw std::runtime_error(
          "check_bounds: malformed OPTIM line " + std::to_string(lineno) +
          " in " + path + " — missing index spec");
    }
    double lo = 0, hi = 0;
    if (!(iss >> lo >> hi)) {
      throw std::runtime_error(
          "check_bounds: malformed OPTIM line " + std::to_string(lineno) +
          " in " + path + " — missing <lo> <hi>");
    }
    auto range = parse_index_spec(idx);
    if (range.first < 0 && range.second < 0) {
      Bound b;
      b.key = key;
      b.arg_pos = -1;
      b.lo = lo;
      b.hi = hi;
      out.push_back(std::move(b));
    } else {
      for (int i = range.first; i <= range.second; ++i) {
        Bound b;
        b.key = key;
        b.arg_pos = i;
        b.lo = lo;
        b.hi = hi;
        out.push_back(std::move(b));
      }
    }
  }
  return out;
}

std::vector<std::vector<double>>
sample_corners(const std::vector<Bound>& bounds, std::uint64_t seed) {
  const std::size_t k = bounds.size();
  if (k == 0) return {};

  // Cap dimensionality at 12 for full-corner enumeration. For higher
  // k, we draw `min(2^k, ⌈k·log₂(k)⌉)` Latin-hypercube samples.
  // The "k log₂ k" budget is data-derived from the dimensionality
  // itself, not a tuning constant — see plan §"No-magic-numbers".
  if (k <= 12) {
    const std::uint64_t N = std::uint64_t{1} << k;
    std::vector<std::vector<double>> out(N, std::vector<double>(k, 0.0));
    for (std::uint64_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        out[i][j] = ((i >> j) & 1u) ? bounds[j].hi : bounds[j].lo;
      }
    }
    return out;
  }

  // High-d branch: LHS in {lo, hi}^k.
  const double kd = static_cast<double>(k);
  const std::uint64_t budget =
      std::min<std::uint64_t>(
          std::uint64_t{1} << std::min<std::size_t>(k, 62),
          static_cast<std::uint64_t>(std::ceil(kd * std::log2(kd))));
  std::mt19937_64 rng(seed);
  std::vector<std::vector<double>> out(budget, std::vector<double>(k, 0.0));
  // Per-dimension permutation: half the samples take lo, half take hi.
  for (std::size_t j = 0; j < k; ++j) {
    std::vector<int> perm(budget);
    for (std::uint64_t r = 0; r < budget; ++r) {
      perm[r] = (r < budget / 2) ? 0 : 1;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    for (std::uint64_t r = 0; r < budget; ++r) {
      out[r][j] = perm[r] ? bounds[j].hi : bounds[j].lo;
    }
  }
  return out;
}

tadah::core::Context corner_context(const tadah::core::Context& seed,
                                    const std::vector<Bound>& bounds,
                                    const std::vector<double>& corner) {
  if (corner.size() != bounds.size())
    throw std::invalid_argument(
        "check_bounds: a corner has " + std::to_string(corner.size()) +
        " values for " + std::to_string(bounds.size()) + " bounds");
  tadah::core::Context ctx = seed;
  for (std::size_t j = 0; j < bounds.size(); ++j) {
    const Bound& b = bounds[j];
    // OPTIM KEY (i) sets entry i of the key; (1) (arg_pos -1) is entry 1.
    const std::size_t entry =
        b.arg_pos < 0 ? 0 : static_cast<std::size_t>(b.arg_pos);
    std::vector<std::string> tokens;
    for (std::size_t n = 0;; ++n) {
      const auto t = ctx.stored_value(b.key, n);
      if (!t) break;
      tokens.push_back(*t);
    }
    if (tokens.empty() ? entry > 0 : entry >= tokens.size())
      throw std::runtime_error(
          "check_bounds: OPTIM " + b.key + " (" + std::to_string(entry + 1) +
          "): the seed configuration has " + std::to_string(tokens.size()) +
          " value(s) for " + b.key);
    // A key the seed does not set takes the corner value alone.
    if (tokens.empty()) tokens.emplace_back();
    tokens[entry] = tadah::core::format_roundtrip(corner[j]);
    ctx.remove(b.key);
    for (const auto& t : tokens) ctx.add(b.key, t);
  }
  return ctx;
}

CheckBoundsResult check_bounds_at_corners(
    StructureDB&                     stdb,
    tadah::core::Context             seed,
    const std::vector<Bound>&        bounds,
    const std::vector<std::vector<double>>& corners) {
  CheckBoundsResult res;
  res.n_corners = corners.size();
  res.n_corners_failing = 0;

  // Pre-compute DataStats ONCE up front. The dataset's per-pair r_ij
  // distribution does not depend on the corner's parameter choices —
  // only on the cutoff used to walk neighbours. We pick the maximum
  // upper bound of any cutoff key (RCUT2B / RCUTMB / RCUTENV) in
  // `bounds` so the NN walk is a superset of every corner's neighbour
  // set. Per-param `RCUT < r_min` checks still fire correctly because
  // `r_min` is the smallest pair distance seen regardless of cutoff.
  // The previous implementation re-ran `compute_data_stats` per
  // corner — O(corners × n_atoms) — which on a 38-dim blip search box
  // (200 LHS samples × 326k atoms) takes ~80 minutes. With the
  // pre-compute it drops to one full-dataset pass plus one O(K) audit
  // per corner.
  tadah::core::Context stats_ctx = seed;
  double rcutmax_for_stats = 0.0;
  if (stats_ctx.contains("RCUTMAX")) {
    rcutmax_for_stats = stats_ctx.get<double>("RCUTMAX");
  }
  for (const auto& b : bounds) {
    if (b.key == "RCUT2B" || b.key == "RCUTMB" || b.key == "RCUTENV") {
      if (b.hi > rcutmax_for_stats) rcutmax_for_stats = b.hi;
    }
  }
  if (rcutmax_for_stats > 0.0) {
    stats_ctx.remove("RCUTMAX");
    try {
      stats_ctx.addInternal<double>("RCUTMAX", rcutmax_for_stats);
    } catch (...) {
      // Some Context implementations may not allow re-adding a removed
      // internal key; fall back to letting compute_data_stats use
      // whatever was already there.
    }
  }
  tadah::core::stats::DataStats data;
  try {
    data = compute_data_stats(stdb, stats_ctx, /*with_descriptor_stats=*/false);
  } catch (const std::invalid_argument&) {
    // The data cannot be used (two atoms at one position, a singular
    // cell, ...): refused, as by training and HPO, not audited around.
    throw;
  } catch (const std::exception& e) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        std::string("check_bounds: dataset stats failed (") + e.what() +
        "); per-corner audits will run with empty stats.");
  }

  // The low-bound rules of the seed (models::optim_low_bound_rule(key,
  // ctx)): under RHOFORM NONNEG / MONOTONE AMPGRIDMB >= 0, WATOM > 0 and
  // SGRIDMB > 0 under MONOTONE and, for the Gaussian basis, under NONNEG
  // (densities DM_REAM refuses), and in every form every radial bump width
  // > 0 (SGRIDMB of a Blip/Blip5 density, SGRID2B of a D2_Blip/D2_Blip5
  // pair term). tadah hpo refuses a box below them at startup, so a corner
  // below one is a FAIL finding whatever the dataset says. A DM_REAM seed
  // must declare its form (require_rhoform: the migration text), as tadah
  // hpo requires.
  if (tadah::models::context_has_ream(seed))
    (void)tadah::models::require_rhoform(seed, "tadah analysis check_bounds");
  auto form_findings = [&](const std::vector<double>& c) {
    std::vector<std::string> out;
    for (std::size_t j = 0; j < bounds.size(); ++j) {
      const auto rule = tadah::models::optim_low_bound_rule(bounds[j].key, seed);
      if (!rule || rule->admits(c[j])) continue;
      std::ostringstream os;
      os << bounds[j].key << "("
         << (bounds[j].arg_pos < 0 ? 1 : bounds[j].arg_pos + 1) << ") = "
         << c[j] << ": " << rule->by << " requires " << rule->requirement
         << " (tadah hpo refuses this OPTIM box)";
      out.push_back(os.str());
    }
    return out;
  };

  for (std::size_t ci = 0; ci < corners.size(); ++ci) {
    const auto& c = corners[ci];
    const tadah::core::Context corner_ctx = corner_context(seed, bounds, c);

    // Audit this corner against the pre-computed dataset stats.
    AuditReport rep = run_preflight_soft(data, corner_ctx, AuditMode::Warn);
    const std::vector<std::string> form_fails = form_findings(c);
    if (rep.any_fail() || !form_fails.empty()) {
      ++res.n_corners_failing;
      res.failing_corner_indices.push_back(ci);
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::INFO,
          "check_bounds: corner #" + std::to_string(ci) + " failed audit");
      // Log the actual findings for the first failing corner so the
      // user can see *why* the bounds need narrowing without re-running
      // the whole sweep at higher verbosity. Subsequent corners just
      // bump the counter — we trust the user not to need every detail.
      if (res.n_corners_failing == 1) {
        for (const auto& f : form_fails)
          tadah::core::Logger::instance().log(
              tadah::core::LogLevel::WARNING,
              "check_bounds: first-failing corner — " + f);
        for (const auto& m : rep.messages) {
          if (m.level != AuditMessage::Level::Fail) continue;
          std::string txt = "check_bounds: first-failing corner — " +
                            m.descriptor_label;
          if (!m.param_key.empty()) txt += "." + m.param_key;
          txt += ": " + m.text;
          tadah::core::Logger::instance().log(
              tadah::core::LogLevel::WARNING, txt);
        }
      }
    }
  }

  if (res.n_corners_failing > 0) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        "check_bounds: " + std::to_string(res.n_corners_failing) +
        "/" + std::to_string(res.n_corners) +
        " corners produced FAIL findings — bounds need narrowing. "
        "See first-failing-corner findings above.");
  } else {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::INFO,
        "check_bounds: all " + std::to_string(res.n_corners) +
        " corners passed — bounds are safe at the box endpoints.");
  }
  return res;
}

}  // namespace mlip
}  // namespace tadah
