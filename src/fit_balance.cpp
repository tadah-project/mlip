#include <tadah/mlip/fit_balance.h>

#include <tadah/core/number_format.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/models/zero_rows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {

namespace {

const char *const kQuantity[3] = {"energy", "force", "stress"};
const char *const kUnit[3] = {"meV/atom", "eV/Å", "GPa"};
const char *const kColumns[9] = {"quantity", "dataset", "n",
                                 "sigma",    "RMSE",    "unit",
                                 "r/sigma",  "share",   "gamma"};
// Whether a column's entries are aligned to the right (numbers).
const bool kRight[9] = {false, false, true, false, true,
                        false, true,  true, true};
const char *const kHeader = "Fit balance: ";

/* The characters of @p s as a terminal shows them (UTF-8 code points). */
std::size_t display_width(const std::string &s) {
  std::size_t n = 0;
  for (unsigned char ch : s)
    if ((ch & 0xC0) != 0x80) ++n;
  return n;
}

/* A number of the table in at most 8 characters: 4 significant digits
 * from 0.001 to 9999, else 3 in exponent form; "-" for none. */
std::string number(double v) {
  if (std::isnan(v)) return "-";
  if (v == 0.0) return "0";
  const double a = std::fabs(v);
  char buf[32];
  if (a >= 1e-3 && a < 1e4)
    std::snprintf(buf, sizeof buf, "%.4g", v);
  else
    std::snprintf(buf, sizeof buf, "%.2e", v);
  return buf;
}

/* "a b" of an error scale key (b 0 when absent), as its values are. */
std::string scale_text(const core::Context &c, const std::string &key,
                       bool two) {
  std::string t = core::format_roundtrip(c.get<double>(key, 0));
  if (two)
    t += " " + core::format_roundtrip(c.size(key) > 1 ? c.get<double>(key, 1)
                                                      : 0.0);
  return t;
}

std::vector<std::string> split(const std::string &line) {
  std::istringstream in(line);
  std::vector<std::string> t;
  for (std::string w; in >> w;) t.push_back(w);
  return t;
}

bool starts_with(const std::string &s, const std::string &p) {
  return s.compare(0, p.size(), p) == 0;
}

[[noreturn]] void unreadable(const std::string &line, const std::string &why) {
  throw std::runtime_error("The fit balance table of the training record "
                           "cannot be read: '" + line + "': " + why + ".");
}

double read_number(const std::string &t, const std::string &line) {
  if (t == "-") return std::numeric_limits<double>::quiet_NaN();
  std::size_t used = 0;
  double v = 0.0;
  try {
    v = std::stod(t, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used != t.size()) unreadable(line, "'" + t + "' is not a number");
  return v;
}

std::size_t read_count(const std::string &t, const std::string &line) {
  if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos)
    unreadable(line, "'" + t + "' is not a count");
  return static_cast<std::size_t>(std::stoull(t));
}

std::atomic<bool> legend_given{false};

}  // namespace

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

const FitBalanceRow *FitBalance::row(std::size_t quantity,
                                     std::size_t dataset) const {
  for (const FitBalanceRow &r : rows)
    if (r.quantity == quantity && r.dataset == dataset) return &r;
  return nullptr;
}

FitBalance no_fit_balance(const std::string &why) {
  FitBalance b;
  b.none = why.empty() ? std::string("no reason given") : why;
  return b;
}

FitSolve FitSolve::of(const core::Context &c, std::size_t cols,
                      const std::vector<std::size_t> &intercepts) {
  FitSolve s;
  s.lambda = c.get<double>("LAMBDA", 0);
  s.lambda_used = s.lambda;
  if (s.lambda < 0.0)  // LAMBDA -1: the evidence approximation's
    s.lambda_used = c.get<double>("ALPHA") / c.get<double>("BETA");
  s.oalgo = c.get<int>("OALGO", 0);
  s.rcond = c.size("OALGO") == 2 ? c.get<double>("OALGO", 1) : 1e-8;
  for (std::size_t i : core::parse_fixindex(c, cols)) s.pinned.push_back(i - 1);
  s.intercepts = intercepts;
  return s;
}

FitBalance fit_balance(const core::phi_type &Phi, std::size_t data_rows,
                       const core::t_type &T, const core::t_type &factors,
                       const core::t_type &labels,
                       const std::vector<std::size_t> &row_dataset,
                       std::size_t n_datasets,
                       const std::vector<FitRowScale> &scale,
                       const core::t_type &w, const FitSolve &solve,
                       const core::Context &c) {
  const std::size_t cols = Phi.cols();
  if (Phi.rows() < data_rows || T.size() < data_rows ||
      factors.size() < data_rows || labels.size() < data_rows ||
      row_dataset.size() < data_rows || scale.size() < data_rows ||
      w.size() != cols || n_datasets == 0)
    throw std::logic_error(
        "fit_balance: " + std::to_string(data_rows) + " data rows of a "
        "design matrix of " + std::to_string(Phi.rows()) + " x " +
        std::to_string(cols) + " with " + std::to_string(T.size()) +
        " targets, " + std::to_string(factors.size()) + " factors, " +
        std::to_string(labels.size()) + " labels, " +
        std::to_string(row_dataset.size()) + " datasets, " +
        std::to_string(scale.size()) + " scales and " +
        std::to_string(w.size()) + " coefficients");

  FitBalance b;
  b.lambda = solve.lambda;
  b.lambda_used = solve.lambda_used;
  b.datasets = n_datasets;
  b.sigma = {scale_text(c, "ESIGMA", false), scale_text(c, "FSIGMA", true),
             scale_text(c, "SSIGMA", true)};

  // The free columns (those FIXINDEX does not pin) and, among them, those
  // the ridge penalises: every one under OALGO 3 (a uniform ridge), every
  // one but the intercept columns under the others.
  std::vector<char> pinned(cols, 0), intercept(cols, 0);
  for (std::size_t j : solve.pinned)
    if (j < cols) pinned[j] = 1;
  for (std::size_t j : solve.intercepts)
    if (j < cols) intercept[j] = 1;
  std::vector<std::size_t> free_cols, unpenalised;
  for (std::size_t j = 0; j < cols; ++j) {
    if (pinned[j]) continue;
    if (intercept[j] && solve.oalgo != 3) unpenalised.push_back(free_cols.size());
    free_cols.push_back(j);
  }
  b.coefficients = free_cols.size();
  b.pinned = cols - free_cols.size();

  // The observations and their residuals.
  const std::vector<char> zero = models::zero_rows(Phi, data_rows);
  struct Sum {
    std::size_t n = 0;
    double m = 0, e2 = 0, z2 = 0, loss = 0;
  };
  std::vector<std::array<Sum, 3>> sum(n_datasets + 1);  // [0]: every dataset
  std::vector<std::size_t> obs;
  for (std::size_t i = 0; i < data_rows; ++i) {
    if (zero[i] || !(factors(i) > 0.0)) continue;
    const double t = labels(i);
    if (t < 0.0 || t > 2.0 || row_dataset[i] >= n_datasets)
      throw std::logic_error("fit_balance: row " + std::to_string(i) +
                             " has a type or dataset out of range");
    obs.push_back(i);
    double pred = 0.0;
    for (std::size_t j = 0; j < cols; ++j) pred += Phi(i, j) * w(j);
    const double r = T(i) - pred;
    const FitRowScale &s = scale[i];
    const double e = r * s.to_metric, z = r / s.sigma;
    const double f = factors(i);
    const std::size_t q = static_cast<std::size_t>(t);
    for (std::size_t d : {std::size_t{0}, row_dataset[i] + 1}) {
      Sum &a = sum[d][q];
      ++a.n;
      a.m += s.m;
      a.e2 += s.m * e * e;
      a.z2 += s.m * z * z;
      a.loss += f * f * r * r;
    }
  }

  // The prior: lambda |w|^2 over the penalised free columns.
  double prior = 0.0;
  if (solve.lambda_used != 0.0) {
    std::vector<char> unpen(cols, 0);
    for (std::size_t k : unpenalised) unpen[free_cols[k]] = 1;
    for (std::size_t j : free_cols)
      if (!unpen[j]) prior += w(j) * w(j);
    prior *= solve.lambda_used;
  }
  double total = prior;
  for (const Sum &a : sum[0]) total += a.loss;

  // The effective parameters of the weighted observations on the free
  // columns.
  EffectiveParameters g;
  g.by_dataset.assign(n_datasets, std::array<double, 3>{{0.0, 0.0, 0.0}});
  if (!free_cols.empty() && !obs.empty()) {
    core::phi_type Pw(obs.size(), free_cols.size());
    core::t_type lab(obs.size());
    std::vector<std::size_t> ds(obs.size());
    for (std::size_t k = 0; k < obs.size(); ++k) {
      const std::size_t i = obs[k];
      for (std::size_t j = 0; j < free_cols.size(); ++j)
        Pw(k, j) = factors(i) * Phi(i, free_cols[j]);
      lab(k) = labels(i);
      ds[k] = row_dataset[i];
    }
    g = effective_parameters(Pw, lab, ds, n_datasets, solve.lambda_used,
                             unpenalised, solve.oalgo, solve.rcond);
  }
  b.retained = g.retained;
  b.prior_gamma = g.prior;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const bool on[3] = {c.get<bool>("ENERGY"), c.get<bool>("FORCE"),
                      c.get<bool>("STRESS")};
  for (std::size_t q = 0; q < 3; ++q) {
    if (!on[q]) continue;
    for (std::size_t d = 0; d <= n_datasets; ++d) {
      if (d > 0 && n_datasets == 1) break;
      const Sum &a = sum[d][q];
      FitBalanceRow r;
      r.quantity = q;
      r.dataset = d;
      r.n = a.n;
      r.rmse = a.n > 0 ? std::sqrt(a.e2 / a.m) : nan;
      r.rms_z = a.n > 0 ? std::sqrt(a.z2 / a.m) : nan;
      r.share = total > 0.0 ? a.loss / total : nan;
      r.gamma = d == 0 ? g.by_type[q] : g.by_dataset[d - 1][q];
      b.rows.push_back(r);
    }
  }
  b.prior_share = total > 0.0 ? prior / total : nan;
  return b;
}

// ---------------------------------------------------------------------------
// Its lines
// ---------------------------------------------------------------------------

std::vector<std::string> fit_balance_lines(const FitBalance &b) {
  if (!b.computed()) return {std::string(kHeader) + "none (" + b.none + ")"};
  std::string head = std::string(kHeader) + "LAMBDA " +
                     core::format_roundtrip(b.lambda);
  if (b.lambda < 0.0)
    head += " (evidence approximation: " + core::format_sig(b.lambda_used, 6) +
            ")";
  head += "; " + std::to_string(b.coefficients) + " coefficients";
  if (b.pinned > 0)
    head += " (FIXINDEX pins " + std::to_string(b.pinned) + " more)";
  head += ", rank " + std::to_string(b.retained);

  std::vector<std::array<std::string, 9>> cells;
  {
    std::array<std::string, 9> h;
    for (std::size_t k = 0; k < 9; ++k) h[k] = kColumns[k];
    cells.push_back(h);
  }
  for (const FitBalanceRow &r : b.rows)
    cells.push_back({kQuantity[r.quantity],
                     r.dataset == 0 ? std::string("all")
                                    : std::to_string(r.dataset),
                     std::to_string(r.n), b.sigma[r.quantity], number(r.rmse),
                     kUnit[r.quantity], number(r.rms_z), number(r.share),
                     number(r.gamma)});
  cells.push_back({"prior", "", "", "", "", "", "", number(b.prior_share),
                   number(b.prior_gamma)});
  std::array<std::size_t, 9> width{};
  for (const auto &row : cells)
    for (std::size_t k = 0; k < 9; ++k)
      width[k] = std::max(width[k], display_width(row[k]));
  std::vector<std::string> lines = {head};
  for (const auto &row : cells) {
    std::string line = " ";
    for (std::size_t k = 0; k < 9; ++k) {
      const std::string pad(width[k] - display_width(row[k]), ' ');
      line += " ";
      line += kRight[k] ? pad + row[k] : row[k] + pad;
      if (k + 1 < 9) line += " ";
    }
    line.erase(line.find_last_not_of(' ') + 1);
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::string> fit_balance_legend() {
  return {
      "  quantity  energy per atom, force component or stress tensor",
      "  dataset   all: every dataset; k: the k-th DBFILE entry",
      "  n         observations of the fit (six per stress tensor)",
      "  sigma     error scale: ESIGMA, FSIGMA a b or SSIGMA a b",
      "  RMSE      root mean square residual (stress: the nine entries)",
      "  unit      unit of RMSE: meV/atom, eV/Å or GPa",
      "  r/sigma   root mean square of residual / error scale",
      "  share     part of the minimised sum; prior: the LAMBDA penalty",
      "  gamma     effective parameters; with the prior's, the rank",
  };
}

void add_fit_balance(TrainingRecord &record, const FitBalance &b) {
  for (const std::string &l : fit_balance_lines(b)) record.lines.push_back(l);
}

std::vector<std::string> fit_balance_record_lines(
    const TrainingRecord &record) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < record.lines.size(); ++i) {
    if (!starts_with(record.lines[i], kHeader)) continue;
    out.push_back(record.lines[i]);
    if (starts_with(record.lines[i], std::string(kHeader) + "none (")) break;
    for (std::size_t k = i + 1;
         k < record.lines.size() && starts_with(record.lines[k], "  "); ++k) {
      out.push_back(record.lines[k]);
      if (starts_with(record.lines[k], "  prior")) break;
    }
    break;
  }
  return out;
}

std::optional<FitBalance> read_fit_balance(const TrainingRecord &record) {
  const std::vector<std::string> lines = fit_balance_record_lines(record);
  if (lines.empty()) return std::nullopt;
  const std::string &head = lines[0];
  const std::string none = std::string(kHeader) + "none (";
  if (starts_with(head, none)) {
    if (head.back() != ')') unreadable(head, "no closing parenthesis");
    return no_fit_balance(head.substr(none.size(),
                                      head.size() - none.size() - 1));
  }
  FitBalance b;
  {
    // Fit balance: LAMBDA <l>[ (evidence approximation: <x>)]; <p>
    // coefficients[ (FIXINDEX pins <k> more)], rank <r>
    std::string rest = head.substr(std::string(kHeader).size());
    const std::vector<std::string> t = split(rest);
    std::size_t k = 0;
    auto take = [&](const std::string &word) {
      if (k >= t.size() || t[k] != word)
        unreadable(head, "'" + word + "' expected");
      ++k;
    };
    take("LAMBDA");
    if (k >= t.size()) unreadable(head, "no LAMBDA");
    std::string l = t[k++];
    const bool ea = l.back() != ';';
    if (!ea) l.pop_back();
    b.lambda = read_number(l, head);
    b.lambda_used = b.lambda;
    if (ea) {
      take("(evidence");
      take("approximation:");
      if (k >= t.size() || t[k].size() < 3) unreadable(head, "no lambda");
      std::string x = t[k++];
      if (x.substr(x.size() - 2) != ");") unreadable(head, "');' expected");
      b.lambda_used = read_number(x.substr(0, x.size() - 2), head);
    }
    if (k >= t.size()) unreadable(head, "no coefficients");
    b.coefficients = read_count(t[k++], head);
    if (k < t.size() && t[k] == "coefficients,") {
      ++k;
      take("rank");
    } else {
      take("coefficients");
      take("(FIXINDEX");
      take("pins");
      if (k >= t.size()) unreadable(head, "no pinned coefficients");
      b.pinned = read_count(t[k++], head);
      take("more),");
      take("rank");
    }
    if (k + 1 != t.size()) unreadable(head, "no rank");
    b.retained = read_count(t[k], head);
  }
  if (lines.size() < 2 || split(lines[1]) != std::vector<std::string>(
                                                  kColumns, kColumns + 9))
    unreadable(lines.size() < 2 ? head : lines[1], "no column header");
  bool prior = false;
  std::size_t datasets = 1;
  for (std::size_t i = 2; i < lines.size(); ++i) {
    const std::string &line = lines[i];
    const std::vector<std::string> t = split(line);
    if (!t.empty() && t[0] == "prior") {
      if (t.size() != 3) unreadable(line, "a prior row has a share and a gamma");
      b.prior_share = read_number(t[1], line);
      b.prior_gamma = read_number(t[2], line);
      prior = true;
      break;
    }
    std::size_t q = 3;
    for (std::size_t k = 0; k < 3; ++k)
      if (!t.empty() && t[0] == kQuantity[k]) q = k;
    if (q == 3) unreadable(line, "no quantity");
    const std::size_t two = q == 0 ? 0 : 1;  // FSIGMA, SSIGMA: a b
    if (t.size() != 9 + two) unreadable(line, "not a row of the table");
    FitBalanceRow r;
    r.quantity = q;
    r.dataset = t[1] == "all" ? 0 : read_count(t[1], line);
    datasets = std::max(datasets, r.dataset);
    r.n = read_count(t[2], line);
    b.sigma[q] = two ? t[3] + " " + t[4] : t[3];
    r.rmse = read_number(t[4 + two], line);
    if (t[5 + two] != kUnit[q]) unreadable(line, "the unit is not " +
                                                     std::string(kUnit[q]));
    r.rms_z = read_number(t[6 + two], line);
    r.share = read_number(t[7 + two], line);
    r.gamma = read_number(t[8 + two], line);
    b.rows.push_back(r);
  }
  if (!prior) unreadable(lines.back(), "the table ends before its prior row");
  b.datasets = datasets;
  return b;
}

void print_fit_balance(std::ostream &os, const std::vector<std::string> &lines,
                       bool legend) {
  for (const std::string &l : lines) os << l << '\n';
  if (legend)
    for (const std::string &l : fit_balance_legend()) os << l << '\n';
  os.flush();
}

bool first_fit_balance_legend() { return !legend_given.exchange(true); }

}  // namespace mlip
}  // namespace tadah
