#include <tadah/mlip/fit_balance.h>

#include <tadah/core/number_format.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/models/zero_rows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tadah {
namespace mlip {

namespace {

const char *const kQuantity[3] = {"energy", "force", "stress"};
const char *const kUnit[3] = {"meV/atom", "eV/Å", "GPa"};
const char *const kScaleKey[3] = {"ESIGMA", "FSIGMA", "SSIGMA"};
constexpr std::size_t kCols = 8;
const char *const kColumns[kCols] = {"quantity", "dataset", "n",
                                     "RMSE",     "unit",    "r/sigma",
                                     "share",    "gamma"};
// Whether a column's entries are aligned to the right (numbers).
const bool kRight[kCols] = {false, false, true, true,
                            false, true,  true, true};
const char *const kHeader = "Fit balance: ";
const char *const kScales = "  error scales: ";
const char *const kContinued = "    ";  // a wrapped line's next lines
const char *const kAllPinned =
    "FIXINDEX pins every coefficient: none was fitted";

/* The characters of @p s as a terminal shows them (UTF-8 code points). */
std::size_t display_width(const std::string &s) {
  std::size_t n = 0;
  for (unsigned char ch : s)
    if ((ch & 0xC0) != 0x80) ++n;
  return n;
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

/* A number of the table in at most 8 characters: 4 significant digits,
 * the trailing zeros kept, from 0.001 to 9999 (0.1520, 1234), else 3 in
 * exponent form (9.88e-04, 1.00e+04 for 9999.7, which rounds to 10000);
 * "0" for 0 and "-" for none. */
std::string number(double v) {
  if (std::isnan(v)) return "-";
  if (v == 0.0) return "0";
  char buf[32];
  if (std::fabs(v) >= 1e-3) {
    std::snprintf(buf, sizeof buf, "%#.4g", v);
    std::string t = buf;
    if (t.find('e') == std::string::npos) {
      if (t.back() == '.') t.pop_back();  // 1234. -> 1234
      return t;
    }
  }
  std::snprintf(buf, sizeof buf, "%.2e", v);
  return buf;
}

/* The values of an error scale key to 4 significant digits, "a" or "a b"
 * (b 0 when absent). */
std::string scale_text(const core::Context &c, const std::string &key,
                       bool two) {
  std::string t = core::format_sig(c.get<double>(key, 0), 4);
  if (two)
    t += " " + core::format_sig(c.size(key) > 1 ? c.get<double>(key, 1) : 0.0,
                                4);
  return t;
}

/* @p first followed by @p clauses joined by "; ", in lines of at most
 * kFitBalanceWidth: a line ends after a "; " where the next clause does not
 * fit, and a clause longer than a line breaks at blanks; each line after
 * the first starts with kContinued. */
std::vector<std::string> wrap(const std::string &first,
                              const std::vector<std::string> &clauses) {
  std::vector<std::string> lines;
  std::string cur = first;
  bool fresh = true;  // nothing of the clauses on the line yet
  auto fits = [](const std::string &s) {
    return display_width(s) <= kFitBalanceWidth;
  };
  for (std::size_t c = 0; c < clauses.size(); ++c) {
    const std::string piece = clauses[c] + (c + 1 < clauses.size() ? ";" : "");
    if (fits(fresh ? cur + piece : cur + " " + piece)) {
      cur += (fresh ? "" : " ") + piece;
      fresh = false;
      continue;
    }
    if (!fresh) {
      lines.push_back(cur);
      cur = kContinued;
      fresh = true;
      if (fits(cur + piece)) {
        cur += piece;
        fresh = false;
        continue;
      }
    }
    for (const std::string &word : split(piece)) {
      if (!fresh && !fits(cur + " " + word)) {
        lines.push_back(cur);
        cur = kContinued;
        fresh = true;
      }
      cur += (fresh ? "" : " ") + word;
      fresh = false;
    }
  }
  lines.push_back(cur);
  return lines;
}

/* The text of the lines from @p k on that make one wrapped line: lines[k]
 * and the lines after it that start with kContinued, joined by a blank;
 * @p k is left at the next line. */
std::string unwrap(const std::vector<std::string> &lines, std::size_t &k) {
  std::string text = lines[k++];
  while (k < lines.size() && starts_with(lines[k], kContinued)) {
    text += " " + lines[k].substr(lines[k].find_first_not_of(' '));
    ++k;
  }
  return text;
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

FitBalance all_pinned_fit_balance() { return no_fit_balance(kAllPinned); }

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
  // A fit of no free column solves for nothing: no balance to show.
  if (free_cols.empty()) return all_pinned_fit_balance();

  FitBalance b;
  b.lambda = solve.lambda;
  b.lambda_used = solve.lambda_used;
  b.enorm_sqrt = core::to_upper(c.get<std::string>("ENORM")) == "SQRT";
  b.datasets = n_datasets;
  b.coefficients = free_cols.size();
  b.pinned = cols - free_cols.size();
  const bool on[3] = {c.get<bool>("ENERGY"), c.get<bool>("FORCE"),
                      c.get<bool>("STRESS")};
  for (std::size_t q = 0; q < 3; ++q)
    if (on[q]) b.sigma[q] = scale_text(c, kScaleKey[q], q > 0);

  // The residuals of every row of priority > 0, in blocks of rows (the
  // predictions column by column: no copy of the rows): RMSE and r/sigma
  // over every such entry, as the metrics; n, the share and gamma over the
  // observations, the rows that are not zero.
  const std::vector<char> zero = models::zero_rows(Phi, data_rows, factors);
  struct Sum {
    std::size_t n = 0;
    double m = 0, e2 = 0, z2 = 0, loss = 0;
  };
  std::vector<std::array<Sum, 3>> sum(n_datasets + 1);  // [0]: every dataset
  std::vector<std::size_t> obs;
  obs.reserve(data_rows);
  constexpr std::size_t kBlock = 1024;
  std::vector<double> pred(std::min(data_rows, kBlock));
  for (std::size_t i0 = 0; i0 < data_rows; i0 += kBlock) {
    const std::size_t nb = std::min(kBlock, data_rows - i0);
    std::fill(pred.begin(), pred.begin() + nb, 0.0);
    for (std::size_t j = 0; j < cols; ++j) {
      const double wj = w(j);
      if (wj == 0.0) continue;
      for (std::size_t k = 0; k < nb; ++k) pred[k] += Phi(i0 + k, j) * wj;
    }
    for (std::size_t k = 0; k < nb; ++k) {
      const std::size_t i = i0 + k;
      if (!(factors(i) > 0.0)) continue;
      const double t = labels(i);
      if (t < 0.0 || t > 2.0 || row_dataset[i] >= n_datasets)
        throw std::logic_error("fit_balance: row " + std::to_string(i) +
                               " has a type or dataset out of range");
      const bool observation = !zero[i];
      if (observation) obs.push_back(i);
      const double r = T(i) - pred[k];
      const FitRowScale &s = scale[i];
      const double e = r * s.to_metric, z = r / s.sigma;
      const double f = factors(i);
      const std::size_t q = static_cast<std::size_t>(t);
      for (std::size_t d : {std::size_t{0}, row_dataset[i] + 1}) {
        Sum &a = sum[d][q];
        a.m += s.m;
        a.e2 += s.m * e * e;
        a.z2 += s.m * z * z;
        if (!observation) continue;
        ++a.n;
        a.loss += f * f * r * r;
      }
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
  // columns, read where they are.
  EffectiveParameters g;
  g.by_dataset.assign(n_datasets, std::array<double, 3>{{0.0, 0.0, 0.0}});
  if (!obs.empty()) {
    WeightedRows rows{Phi, factors, labels, row_dataset, std::move(obs),
                      free_cols};
    g = effective_parameters(rows, n_datasets, solve.lambda_used, unpenalised,
                             solve.oalgo, solve.rcond);
  }
  b.retained = g.retained;
  b.prior_gamma = g.prior;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t q = 0; q < 3; ++q) {
    if (!on[q]) continue;
    for (std::size_t d = 0; d <= n_datasets; ++d) {
      if (d > 0 && n_datasets == 1) break;
      const Sum &a = sum[d][q];
      FitBalanceRow r;
      r.quantity = q;
      r.dataset = d;
      r.n = a.n;
      r.rmse = a.m > 0 ? std::sqrt(a.e2 / a.m) : nan;
      r.rms_z = a.m > 0 ? std::sqrt(a.z2 / a.m) : nan;
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
  if (!b.computed()) return wrap(kHeader, {"none (" + b.none + ")"});
  std::vector<std::string> head;
  head.push_back("LAMBDA " + core::format_roundtrip(b.lambda));
  if (b.lambda < 0.0)
    head.back() += " (evidence approximation: " +
                   core::format_sig(b.lambda_used, 6) + ")";
  if (b.enorm_sqrt) head.push_back("ENORM SQRT");
  std::string fitted = std::to_string(b.coefficients) + " coefficients";
  if (b.pinned > 0)
    fitted += " (FIXINDEX pins " + std::to_string(b.pinned) + " more)";
  head.push_back(fitted + ", rank " + std::to_string(b.retained));
  std::vector<std::string> lines = wrap(kHeader, head);

  // The error scales of the quantities with a row.
  std::vector<std::string> scales;
  for (std::size_t q = 0; q < 3; ++q) {
    bool has_row = false;
    for (const FitBalanceRow &r : b.rows) has_row = has_row || r.quantity == q;
    if (!has_row) continue;
    std::string unit = kUnit[q];
    if (q == 0 && b.enorm_sqrt) unit = "meV per sqrt(atom)";
    scales.push_back(std::string(kScaleKey[q]) + " " + b.sigma[q] + " " +
                     unit);
  }
  if (!scales.empty())
    for (const std::string &l : wrap(kScales, scales)) lines.push_back(l);

  std::vector<std::array<std::string, kCols>> cells;
  {
    std::array<std::string, kCols> h;
    for (std::size_t k = 0; k < kCols; ++k) h[k] = kColumns[k];
    cells.push_back(h);
  }
  for (const FitBalanceRow &r : b.rows)
    cells.push_back({kQuantity[r.quantity],
                     r.dataset == 0 ? std::string("all")
                                    : std::to_string(r.dataset),
                     std::to_string(r.n), number(r.rmse), kUnit[r.quantity],
                     number(r.rms_z), number(r.share), number(r.gamma)});
  cells.push_back({"prior", "", "", "", "", "", number(b.prior_share),
                   number(b.prior_gamma)});
  std::array<std::size_t, kCols> width{};
  for (const auto &row : cells)
    for (std::size_t k = 0; k < kCols; ++k)
      width[k] = std::max(width[k], display_width(row[k]));
  for (const auto &row : cells) {
    std::string line = " ";
    for (std::size_t k = 0; k < kCols; ++k) {
      const std::string pad(width[k] - display_width(row[k]), ' ');
      line += " ";
      line += kRight[k] ? pad + row[k] : row[k] + pad;
      if (k + 1 < kCols) line += " ";
    }
    line.erase(line.find_last_not_of(' ') + 1);
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::string> fit_balance_legend() {
  return {
      "  quantity  energy per atom, force component or stress tensor",
      "  dataset   all: every dataset's entries together; k: the k-th DBFILE entry",
      "  n         observations: entries of priority > 0 not zero by symmetry",
      "  RMSE      root mean square residual of every entry of priority > 0",
      "  unit      of RMSE: meV/atom, eV/Å, GPa (nine entries per stress tensor)",
      "  r/sigma   root mean square of residual / error scale, the same entries",
      "  share     part of the minimised sum; prior: the LAMBDA term",
      "  gamma     effective parameters",
      "  -         RMSE and r/sigma of a row without an entry of priority > 0",
      "  Over the rows 'all' and the prior's: share sums to 1, gamma to the rank.",
  };
}

void add_fit_balance(TrainingRecord &record, const FitBalance &b) {
  for (const std::string &l : fit_balance_lines(b)) record.lines.push_back(l);
}

namespace {

/* Where the table of @p record is: its first line and the line after its
 * last ({0, 0} without one). */
std::pair<std::size_t, std::size_t> table_span(const TrainingRecord &record) {
  for (std::size_t i = 0; i < record.lines.size(); ++i) {
    if (!starts_with(record.lines[i], kHeader)) continue;
    std::size_t k = i + 1;
    if (starts_with(record.lines[i], std::string(kHeader) + "none (")) {
      while (k < record.lines.size() &&
             starts_with(record.lines[k], kContinued))
        ++k;
      return {i, k};
    }
    for (; k < record.lines.size() && starts_with(record.lines[k], "  "); ++k)
      if (starts_with(record.lines[k], "  prior")) return {i, k + 1};
    return {i, k};
  }
  return {0, 0};
}

}  // namespace

std::vector<std::string> fit_balance_record_lines(
    const TrainingRecord &record) {
  const auto span = table_span(record);
  return std::vector<std::string>(record.lines.begin() + span.first,
                                  record.lines.begin() + span.second);
}

TrainingRecord without_fit_balance(const TrainingRecord &record) {
  const auto span = table_span(record);
  TrainingRecord out = record;
  out.lines.erase(out.lines.begin() + span.first,
                  out.lines.begin() + span.second);
  return out;
}

std::optional<FitBalance> read_fit_balance(const TrainingRecord &record) {
  const std::vector<std::string> lines = fit_balance_record_lines(record);
  if (lines.empty()) return std::nullopt;
  std::size_t at = 0;
  const std::string head = unwrap(lines, at);
  const std::string none = std::string(kHeader) + "none (";
  if (starts_with(head, none)) {
    if (head.back() != ')') unreadable(head, "no closing parenthesis");
    return no_fit_balance(head.substr(none.size(),
                                      head.size() - none.size() - 1));
  }
  FitBalance b;
  {
    // Fit balance: LAMBDA <l>[ (evidence approximation: <x>)][; ENORM
    // SQRT]; <p> coefficients[ (FIXINDEX pins <k> more)], rank <r>
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
    if (k + 1 < t.size() && t[k] == "ENORM" && t[k + 1] == "SQRT;") {
      b.enorm_sqrt = true;
      k += 2;
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
  {
    // error scales: ESIGMA <a> <unit>; FSIGMA <a> <b> eV/Å; SSIGMA ...
    if (at >= lines.size() || !starts_with(lines[at], kScales))
      unreadable(at < lines.size() ? lines[at] : head, "no error scales");
    const std::string text = unwrap(lines, at);
    std::string rest = text.substr(std::string(kScales).size());
    std::size_t from = 0;
    while (from <= rest.size()) {
      const std::size_t to = rest.find("; ", from);
      const std::vector<std::string> t =
          split(rest.substr(from, to == std::string::npos ? std::string::npos
                                                          : to - from));
      std::size_t q = 3;
      for (std::size_t k = 0; k < 3; ++k)
        if (!t.empty() && t[0] == kScaleKey[k]) q = k;
      if (q == 3) unreadable(text, "an error scale key expected");
      const std::size_t values = q == 0 ? 1 : 2;
      const std::string unit =
          q == 0 && b.enorm_sqrt ? "meV per sqrt(atom)" : kUnit[q];
      std::string got;
      for (std::size_t k = 1 + values; k < t.size(); ++k)
        got += (got.empty() ? "" : " ") + t[k];
      if (t.size() < 2 + values || got != unit)
        unreadable(text, std::string(kScaleKey[q]) + " with its value" +
                             (values == 2 ? "s" : "") + " and " + unit +
                             " expected");
      for (std::size_t k = 1; k <= values; ++k) (void)read_number(t[k], text);
      b.sigma[q] = values == 2 ? t[1] + " " + t[2] : t[1];
      if (to == std::string::npos) break;
      from = to + 2;
    }
  }
  if (at >= lines.size() ||
      split(lines[at]) != std::vector<std::string>(kColumns, kColumns + kCols))
    unreadable(at < lines.size() ? lines[at] : head, "no column header");
  bool prior = false;
  std::size_t datasets = 1;
  for (std::size_t i = at + 1; i < lines.size(); ++i) {
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
    if (t.size() != kCols) unreadable(line, "not a row of the table");
    if (b.sigma[q].empty())
      unreadable(line, "no error scale of the " + std::string(kQuantity[q]));
    FitBalanceRow r;
    r.quantity = q;
    r.dataset = t[1] == "all" ? 0 : read_count(t[1], line);
    datasets = std::max(datasets, r.dataset);
    r.n = read_count(t[2], line);
    r.rmse = read_number(t[3], line);
    if (t[4] != kUnit[q])
      unreadable(line, "the unit is not " + std::string(kUnit[q]));
    r.rms_z = read_number(t[5], line);
    r.share = read_number(t[6], line);
    r.gamma = read_number(t[7], line);
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
