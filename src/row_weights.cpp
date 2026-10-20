#include <tadah/mlip/row_weights.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/lapack.h>
#include <tadah/core/number_format.h>
#include <tadah/core/retired_keys.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/detail/cell_geometry.h>
#include <tadah/models/zero_rows.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {

namespace {

/* The values of @p key as the Context stores them, or its default, joined
 * by blanks after the key: "FSIGMA 0.2 0.05". */
std::string key_text(const core::Context &c, const std::string &key) {
  std::string t = key;
  for (const std::string &v : c(key)) t += " " + v;
  return t;
}

/* The blueprint default of @p key as doubles. */
std::vector<double> default_doubles(const std::string &key) {
  std::vector<double> out;
  for (const std::string &v :
       core::BlueprintParser::getKeysMapRef().at(key).default_value)
    out.push_back(std::stod(v));
  return out;
}

/* a and b of an error scale "a [b]" (b 0 when absent). */
std::array<double, 2> scale_ab(const core::Context &c, const std::string &key) {
  const double a = c.get<double>(key, 0);
  const double b = c.size(key) > 1 ? c.get<double>(key, 1) : 0.0;
  return {a, b};
}

bool is_default_scale(const core::Context &c, const std::string &key) {
  const std::vector<double> d = default_doubles(key);
  const std::array<double, 2> ab = scale_ab(c, key);
  return ab[0] == d[0] && ab[1] == (d.size() > 1 ? d[1] : 0.0);
}

std::string upper_enorm(const core::Context &c) {
  return core::to_upper(c.get<std::string>("ENORM"));
}

/* Singular values and right singular vectors of the m x n column-major
 * matrix at @p a, of leading dimension @p lda (destroyed): S (min(m, n),
 * descending) and VT, min(m, n) x n column-major. */
void svd_right(double *a, int m, int n, int lda, std::vector<double> &S,
               std::vector<double> &VT) {
  const int k = std::min(m, n);
  S.assign(static_cast<std::size_t>(k), 0.0);
  VT.assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(n), 0.0);
  if (k == 0) return;
  char jobu = 'N', jobvt = 'S';
  int ldu = 1, ldvt = k, info = 0, lwork = -1;
  double u = 0, wq = 0;
  core::dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S.data(), &u, &ldu,
                VT.data(), &ldvt, &wq, &lwork, &info);
  lwork = std::max(1, static_cast<int>(wq));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  core::dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S.data(), &u, &ldu,
                VT.data(), &ldvt, work.data(), &lwork, &info);
  if (info != 0)
    throw std::runtime_error("effective_parameters: the singular value "
                             "decomposition did not converge (DGESVD info " +
                             std::to_string(info) + ")");
}

/* The most rows effective_parameters() reads in one block (at least the
 * columns, so a block holds the ridge rows). */
constexpr std::size_t kBlockRows = 1024;

/* Rows k0 .. k0 + nb - 1 of @p w, with their factors, into rows r0 .. of
 * the column-major array @p buf of leading dimension @p ld. */
void copy_rows(const WeightedRows &w, std::size_t k0, std::size_t nb,
               double *buf, std::size_t ld, std::size_t r0) {
  for (std::size_t j = 0; j < w.cols.size(); ++j) {
    const std::size_t c = w.cols[j];
    double *col = buf + ld * j + r0;
    for (std::size_t k = 0; k < nb; ++k) {
      const std::size_t i = w.rows[k0 + k];
      col[k] = w.factors(i) * w.Phi(i, c);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// ErrorScales
// ---------------------------------------------------------------------------

ErrorScales ErrorScales::of(const core::Context &c) {
  ErrorScales s;
  s.energy = c.get<bool>("ENERGY");
  s.force = c.get<bool>("FORCE");
  s.stress = c.get<bool>("STRESS");
  s.sqrt_norm = upper_enorm(c) == "SQRT";
  s.esigma = c.get<double>("ESIGMA") / 1000.0;
  const std::array<double, 2> f = scale_ab(c, "FSIGMA");
  s.fsigma_a = f[0];
  s.fsigma_b = f[1];
  const std::array<double, 2> st = scale_ab(c, "SSIGMA");
  s.ssigma_a = st[0] * kGPaInEvPerA3;
  s.ssigma_b = st[1];
  return s;
}

double ErrorScales::energy_norm(std::size_t natoms) const {
  const double n = static_cast<double>(natoms);
  return sqrt_norm ? std::sqrt(n) : n;
}

double ErrorScales::absolute(std::size_t type) const {
  if (type == 0) return esigma;
  if (type == 1) return fsigma_a;
  if (type == 2) return ssigma_a;
  throw std::logic_error("ErrorScales::absolute: row type " +
                         std::to_string(type));
}

// ---------------------------------------------------------------------------
// RowWeights
// ---------------------------------------------------------------------------

std::vector<double> dataset_priorities(const core::Context &c,
                                       const StructureDB &stdb) {
  const std::size_t n_db = stdb.dbidx.size() > 1 ? stdb.dbidx.size() - 1 : 1;
  std::vector<double> w(n_db, 1.0);
  if (!c.has_stored("WDBFILE")) return w;
  const std::size_t given = c.size("WDBFILE");
  const std::string text = key_text(c, "WDBFILE");
  if (given != n_db)
    throw std::runtime_error(
        text + ": " + std::to_string(given) + " value" +
        (given == 1 ? "" : "s") + " for " + std::to_string(n_db) +
        " dataset" + (n_db == 1 ? "" : "s") + ": WDBFILE gives one priority "
        "per dataset (DBFILE entry); no coefficients were fitted.");
  for (std::size_t d = 0; d < n_db; ++d) {
    const double v = c.get<double>("WDBFILE", d);
    if (detail::is_finite_bits(v) && v >= 0.0) {
      w[d] = v;
      continue;
    }
    std::string file;
    if (c.has_stored("DBFILE") && c.size("DBFILE") == n_db)
      file = " (" + c("DBFILE")[d] + ")";
    throw std::runtime_error(
        text + ": the priority of dataset " + std::to_string(d + 1) + " of " +
        std::to_string(n_db) + file + ", " + core::format_roundtrip(v) +
        ", is not a finite number >= 0 (0 leaves the dataset out of the "
        "fit); no coefficients were fitted.");
  }
  return w;
}

RowWeights::RowWeights(const core::Context &c, const StructureDB &stdb)
    : scales_(ErrorScales::of(c)), stdb_(stdb) {
  std::vector<std::size_t> dbidx = stdb.dbidx;
  if (dbidx.size() < 2) dbidx = {0, stdb.size()};
  n_datasets_ = dbidx.size() - 1;
  dprio_ = dataset_priorities(c, stdb);
  dataset_.assign(stdb.size(), 0);
  for (std::size_t d = 0; d < n_datasets_; ++d)
    for (std::size_t s = dbidx[d]; s < dbidx[d + 1] && s < stdb.size(); ++s)
      dataset_[s] = d;
  fsigma_.assign(stdb.size(), scales_.fsigma_a);
  ssigma_.assign(stdb.size(), scales_.ssigma_a);
  for (std::size_t s = 0; s < stdb.size(); ++s) {
    const Structure &st = stdb(s);
    if (scales_.fsigma_b != 0.0 && st.natoms() > 0) {
      double sum = 0.0;
      for (const Atom &a : st.atoms)
        for (int k = 0; k < 3; ++k) sum += a.force[k] * a.force[k];
      const double M =
          std::sqrt(sum / (3.0 * static_cast<double>(st.natoms())));
      const double bm = scales_.fsigma_b * M;
      fsigma_[s] = std::sqrt(scales_.fsigma_a * scales_.fsigma_a + bm * bm);
    }
    if (scales_.ssigma_b != 0.0) {
      double sum = 0.0;
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) sum += st.stress(x, y) * st.stress(x, y);
      const double M = std::sqrt(sum / 9.0);
      const double bm = scales_.ssigma_b * M;
      ssigma_[s] = std::sqrt(scales_.ssigma_a * scales_.ssigma_a + bm * bm);
    }
  }
}

double RowWeights::energy_priority(std::size_t s) const {
  if (!scales_.energy) return 0.0;
  return stdb_(s).eweight * dataset_priority(s);
}
double RowWeights::force_priority(std::size_t s) const {
  return stdb_(s).fweight * dataset_priority(s);
}
double RowWeights::stress_priority(std::size_t s) const {
  return stdb_(s).sweight * dataset_priority(s);
}

double RowWeights::energy_sigma(std::size_t s) const {
  return scales_.energy_norm(stdb_(s).natoms()) * scales_.esigma;
}
double RowWeights::force_sigma(std::size_t s) const { return fsigma_[s]; }
double RowWeights::stress_sigma(std::size_t s) const { return ssigma_[s]; }

double RowWeights::energy_factor(std::size_t s) const {
  const double p = energy_priority(s);
  if (!(p > 0.0)) return 0.0;
  return std::sqrt(p) / energy_sigma(s);
}
double RowWeights::force_factor(std::size_t s) const {
  const double p = force_priority(s);
  if (!(p > 0.0)) return 0.0;
  return std::sqrt(p) / fsigma_[s];
}
double RowWeights::stress_factor(std::size_t s, int x, int y) const {
  const double p = stress_priority(s);
  if (!(p > 0.0)) return 0.0;
  const double m = x == y ? 1.0 : 2.0;
  return std::sqrt(p * m) / ssigma_[s];
}

// ---------------------------------------------------------------------------
// The settings of a fit
// ---------------------------------------------------------------------------

void require_fit_settings(const core::Context &c, const std::string &who) {
  core::require_no_retired_keys(c, who);
  const std::string tail = "; no coefficients were fitted.";
  auto finite = [](double v) { return detail::is_finite_bits(v); };

  const double e = c.get<double>("ESIGMA");
  if (!(finite(e) && e > 0.0))
    throw std::runtime_error(
        who + ": " + key_text(c, "ESIGMA") + ": the energy error scale is a "
        "number > 0 (meV/atom, the error of the energy per atom that costs "
        "as much as a force off by FSIGMA)" + tail);
  struct Scale {
    const char *key, *what, *unit, *zero;
  };
  for (const Scale &s :
       {Scale{"FSIGMA", "force", "eV/Å", "forces"},
        Scale{"SSIGMA", "stress", "GPa", "stress"}}) {
    const std::array<double, 2> ab = scale_ab(c, s.key);
    if (!(finite(ab[0]) && ab[0] > 0.0))
      throw std::runtime_error(
          who + ": " + key_text(c, s.key) + ": the absolute part a of the " +
          s.what + " error scale is a number > 0 (" + s.unit + "): it is "
          "the error scale of a structure whose " + s.zero + " are zero" +
          tail);
    if (!(finite(ab[1]) && ab[1] >= 0.0))
      throw std::runtime_error(
          who + ": " + key_text(c, s.key) + ": the relative part b of the " +
          s.what + " error scale is a number >= 0 (0 gives every " +
          s.what + " entry the error scale a)" + tail);
  }
  const std::string enorm = upper_enorm(c);
  if (enorm != "ATOM" && enorm != "SQRT")
    throw std::runtime_error(who + ": " + key_text(c, "ENORM") +
                             ": ENORM is ATOM or SQRT" + tail);

  const bool energy = c.get<bool>("ENERGY");
  const bool force = c.get<bool>("FORCE");
  const bool stress = c.get<bool>("STRESS");
  if (!energy && !force && !stress)
    throw std::runtime_error(
        who + ": ENERGY, FORCE and STRESS are all false: the fit has no "
        "observation. ENERGY true (the default) fits the energies, FORCE "
        "true the forces and STRESS true the stresses" + tail);
  auto inert = [&](const std::string &given, const std::string &sw,
                   const std::string &what, const std::string &key,
                   const std::string &dflt) {
    throw std::runtime_error(
        who + ": " + given + ": with " + sw + " false the fit observes no " +
        what + ", so " + key + " would change nothing, and a fit refuses " +
        key + " other than its default then. " + sw + " true fits the " +
        what + "; " + key + " " + dflt + ", the default (every potential "
        "has the line), is accepted with " + sw + " false" + tail);
  };
  if (!force && !is_default_scale(c, "FSIGMA"))
    inert(key_text(c, "FSIGMA"), "FORCE", "forces", "FSIGMA", "0.1 0");
  if (!stress && !is_default_scale(c, "SSIGMA"))
    inert(key_text(c, "SSIGMA"), "STRESS", "stresses", "SSIGMA", "0.5 0");
  if (!energy && e != default_doubles("ESIGMA")[0])
    inert(key_text(c, "ESIGMA"), "ENERGY", "energies", "ESIGMA", "1");
  if (!energy && enorm != "ATOM")
    inert("ENORM " + enorm, "ENERGY", "energies", "ENORM", "ATOM");
}

void write_fit_scales(core::Context &c) {
  for (const char *key : {"ESIGMA", "FSIGMA", "SSIGMA", "ENERGY"})
    (void)c.contains(key);  // inserts the blueprint default when absent
  const std::string enorm = upper_enorm(c);
  c.remove("ENORM");
  c.add("ENORM", enorm);
}

std::string fit_scales_info(const core::Context &c) {
  const bool sqrt_norm = upper_enorm(c) == "SQRT";
  auto def = [](bool is_default, const char *off) {
    if (is_default && off) return std::string(" (default; ") + off + " false)";
    if (is_default) return std::string(" (default)");
    if (off) return std::string(" (") + off + " false)";
    return std::string();
  };
  const bool energy = c.get<bool>("ENERGY");
  const bool force = c.get<bool>("FORCE");
  const bool stress = c.get<bool>("STRESS");
  const double e = c.get<double>("ESIGMA");
  const std::array<double, 2> f = scale_ab(c, "FSIGMA");
  const std::array<double, 2> s = scale_ab(c, "SSIGMA");
  const std::string enorm = upper_enorm(c);
  std::string line = "Fit error scales: ESIGMA " + core::format_roundtrip(e) +
                     (sqrt_norm ? " meV per sqrt(atom)" : " meV/atom") +
                     def(e == default_doubles("ESIGMA")[0],
                         energy ? nullptr : "ENERGY");
  line += ", ENORM " + enorm + def(enorm == "ATOM", nullptr);
  line += std::string(", ENERGY ") + (energy ? "true" : "false") +
          def(energy, nullptr);
  line += ", FSIGMA " + core::format_roundtrip(f[0]) + " " +
          core::format_roundtrip(f[1]) + " eV/Å" +
          def(is_default_scale(c, "FSIGMA"), force ? nullptr : "FORCE");
  line += ", SSIGMA " + core::format_roundtrip(s[0]) + " " +
          core::format_roundtrip(s[1]) + " GPa" +
          def(is_default_scale(c, "SSIGMA"), stress ? nullptr : "STRESS");
  return line + ".";
}

// ---------------------------------------------------------------------------
// Observations and effective parameters
// ---------------------------------------------------------------------------

ObservationCount count_observations(const core::phi_type &Phi,
                                    std::size_t data_rows,
                                    const core::t_type &factors,
                                    const core::t_type &labels) {
  ObservationCount n;
  const std::vector<char> zero = models::zero_rows(Phi, data_rows);
  for (std::size_t i = 0; i < zero.size(); ++i) {
    if (zero[i] || !(factors(i) > 0.0)) continue;
    const std::size_t t = static_cast<std::size_t>(labels(i));
    if (t > 2)
      throw std::logic_error("count_observations: row type " +
                             std::to_string(t));
    ++n.total;
    ++n.by_type[t];
    n.fitted[t] = true;
  }
  return n;
}

EffectiveParameters effective_parameters(
    const core::phi_type &Phi, const core::t_type &labels,
    const std::vector<std::size_t> &row_dataset, std::size_t n_datasets,
    double lambda, const std::vector<std::size_t> &unpenalised, int oalgo,
    double rcond) {
  const std::size_t m = Phi.rows(), n = Phi.cols();
  if (labels.size() < m || row_dataset.size() < m)
    throw std::logic_error("effective_parameters: " + std::to_string(m) +
                           " rows, " + std::to_string(labels.size()) +
                           " labels and " + std::to_string(row_dataset.size()) +
                           " datasets");
  core::t_type ones(m);  // the rows of Phi carry their factors
  for (std::size_t i = 0; i < m; ++i) ones(i) = 1.0;
  WeightedRows w{Phi, ones, labels, row_dataset, {}, {}};
  w.rows.resize(m);
  for (std::size_t i = 0; i < m; ++i) w.rows[i] = i;
  w.cols.resize(n);
  for (std::size_t j = 0; j < n; ++j) w.cols[j] = j;
  return effective_parameters(w, n_datasets, lambda, unpenalised, oalgo,
                              rcond);
}

EffectiveParameters effective_parameters(
    const WeightedRows &w, std::size_t n_datasets, double lambda,
    const std::vector<std::size_t> &unpenalised, int oalgo, double rcond) {
  const std::size_t m = w.rows.size(), n = w.cols.size();
  for (std::size_t k = 0; k < m; ++k) {
    const std::size_t i = w.rows[k];
    if (i >= w.Phi.rows() || i >= w.factors.size() || i >= w.labels.size() ||
        i >= w.row_dataset.size())
      throw std::logic_error("effective_parameters: row " + std::to_string(i) +
                             " of a design matrix of " +
                             std::to_string(w.Phi.rows()) + " rows, " +
                             std::to_string(w.factors.size()) + " factors, " +
                             std::to_string(w.labels.size()) + " labels and " +
                             std::to_string(w.row_dataset.size()) +
                             " datasets");
    if (w.row_dataset[i] >= n_datasets || w.labels(i) < 0 || w.labels(i) > 2)
      throw std::logic_error("effective_parameters: row " + std::to_string(i) +
                             " has a type or dataset out of range");
  }
  for (std::size_t c : w.cols)
    if (c >= w.Phi.cols())
      throw std::logic_error("effective_parameters: column " +
                             std::to_string(c) + " of " +
                             std::to_string(w.Phi.cols()));
  for (std::size_t j : unpenalised)
    if (j >= n)
      throw std::logic_error("effective_parameters: unpenalised column " +
                             std::to_string(j) + " of " + std::to_string(n));
  if (oalgo == 3 && lambda != 0.0 && !unpenalised.empty())
    throw std::logic_error("effective_parameters: OALGO 3 penalises every "
                           "column alike (LAMBDA != 0 with unpenalised "
                           "columns is refused by the fit)");
  EffectiveParameters g;
  g.by_dataset.assign(n_datasets, std::array<double, 3>{{0.0, 0.0, 0.0}});
  if (m == 0 || n == 0) return g;

  // The directions: the right singular vectors of the weighted rows A
  // (OALGO 3), whose values s give s^2 + lambda, or of A = [rows;
  // sqrt(lambda) P] (the others), whose values give the eigenvalues of H
  // directly. They are those of the n x n triangle R of A = QR, which the
  // blocks of rows build: R of [R; block] for each block in turn.
  const bool uniform = oalgo == 3;
  const bool ridge = !uniform && lambda > 0.0;
  std::vector<char> pen(n, 1);  // the penalised columns
  if (!uniform)
    for (std::size_t j : unpenalised) pen[j] = 0;
  const std::size_t total = m + (ridge ? n : 0);  // the rows of A
  const std::size_t block =
      std::min(std::max(m, ridge ? n : std::size_t{0}),
               std::max(n, kBlockRows));
  const std::size_t ld = n + block;
  std::vector<double> buf(ld * n, 0.0);  // rows [0, n): R; then a block
  std::vector<double> tau(n);
  int N = static_cast<int>(n), LDA = static_cast<int>(ld), info = 0;
  int lwork = -1;
  {
    int M = LDA;
    double wq = 0;
    core::dgeqrf_(&M, &N, buf.data(), &LDA, tau.data(), &wq, &lwork, &info);
    lwork = std::max(N, static_cast<int>(wq));
  }
  std::vector<double> work(static_cast<std::size_t>(lwork));
  auto factor = [&](std::size_t nb) {  // R of [R; the block of nb rows]
    int M = static_cast<int>(n + nb);
    core::dgeqrf_(&M, &N, buf.data(), &LDA, tau.data(), work.data(), &lwork,
                  &info);
    if (info != 0)
      throw std::runtime_error("effective_parameters: the QR factorisation "
                               "failed (DGEQRF info " +
                               std::to_string(info) + ")");
    for (std::size_t j = 0; j < n; ++j)  // R alone: no reflector below it
      for (std::size_t i = j + 1; i < n; ++i) buf[i + ld * j] = 0.0;
  };
  for (std::size_t k0 = 0; k0 < m; k0 += block) {
    const std::size_t nb = std::min(block, m - k0);
    copy_rows(w, k0, nb, buf.data(), ld, n);
    factor(nb);
  }
  if (ridge) {
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i)
        buf[n + i + ld * j] = i == j && pen[j] ? std::sqrt(lambda) : 0.0;
    factor(n);
  }
  std::vector<double> S, VT;
  svd_right(buf.data(), N, N, LDA, S, VT);
  // A has min(total, n) singular values; R's others are those of its zero
  // rows.
  const std::size_t k = std::min(total, n);
  double thresh = 0.0;
  {
    const double smax = S[0];
    if (oalgo == 1)
      thresh = (rcond < 0.0 ? std::numeric_limits<double>::epsilon() : rcond) *
               smax;
    else if (oalgo == 3)
      thresh = rcond > 0.0 ? rcond * smax : 0.0;
    else
      thresh = std::sqrt(static_cast<double>(n) *
                         std::numeric_limits<double>::epsilon()) *
               smax;
  }
  std::vector<std::size_t> kept;
  std::vector<double> denom;
  for (std::size_t i = 0; i < k; ++i) {
    if (!(S[i] > thresh)) continue;
    kept.push_back(i);
    denom.push_back(uniform ? S[i] * S[i] + lambda : S[i] * S[i]);
    // The prior's part of the direction, lambda |P v_i|^2 / denom: what
    // the rest of the rank is, without the rounding of the difference.
    if (lambda > 0.0) {
      double pv = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        if (pen[j]) pv += VT[i + n * j] * VT[i + n * j];
      g.prior += lambda * pv / denom.back();
    }
  }
  g.retained = kept.size();
  if (kept.empty()) return g;

  // Each row's part of the kept directions, |a_k . v_i|^2 / denom_i, by
  // blocks of rows: U = A_block V_kept.
  const std::size_t r = kept.size();
  std::vector<double> V(n * r);  // column-major, n x r
  for (std::size_t c = 0; c < r; ++c)
    for (std::size_t j = 0; j < n; ++j) V[j + n * c] = VT[kept[c] + n * j];
  const std::size_t rows_block = std::min(m, std::max(n, kBlockRows));
  std::vector<double> U(rows_block * r);
  for (std::size_t k0 = 0; k0 < m; k0 += rows_block) {
    const std::size_t nb = std::min(rows_block, m - k0);
    copy_rows(w, k0, nb, buf.data(), ld, 0);
    char tr = 'N';
    int M = static_cast<int>(nb), R = static_cast<int>(r), LDU = M;
    double one = 1.0, zero = 0.0;
    core::dgemm_(&tr, &tr, &M, &R, &N, &one, buf.data(), &LDA, V.data(), &N,
                 &zero, U.data(), &LDU);
    for (std::size_t q = 0; q < nb; ++q) {
      const std::size_t i = w.rows[k0 + q];
      const std::size_t t = static_cast<std::size_t>(w.labels(i));
      double c = 0.0;
      for (std::size_t d = 0; d < r; ++d)
        c += U[q + nb * d] * U[q + nb * d] / denom[d];
      g.by_type[t] += c;
      g.by_dataset[w.row_dataset[i]][t] += c;
    }
  }
  g.total = g.by_type[0] + g.by_type[1] + g.by_type[2];
  return g;
}

}  // namespace mlip
}  // namespace tadah
