#include <tadah/mlip/weight_covariance.h>

#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/models/functions/weights_fit.h>
#include <tadah/models/m_krr_core.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

const char *const kTypeLetter[3] = {"E", "F", "S"};

const std::string kNoCovariance =
    "The potential has no weight covariance (WCOV and WNOISE), which --error "
    "needs";

/* Why the potential of @p c has no WCOV and WNOISE, and what to do: its
 * configuration (another OALGO than 3, M_KRR implementation 2), or else
 * the ways an OALGO 3 potential has none. */
std::string why_no_covariance(const tadah::core::Context &c) {
  const int oalgo = c.get<int>("OALGO");
  if (oalgo != 3)
    return kNoCovariance + ": its fit, OALGO " + std::to_string(oalgo) +
           ", computes none (only OALGO 3, the SVD solve, does); retrain the "
           "potential with OALGO 3.";
  if (is_krr_implementation_2(c))
    return kNoCovariance + ": " +
           tadah::models::krr_implementation2_without_covariance() +
           ", and retrain the potential.";
  return kNoCovariance + ". An OALGO 3 fit writes them, so the potential is "
         "one of these: a potential of an earlier version of Tadah!, whose "
         "SIGMA is ignored (it was computed with a transposed formula): a "
         "retrain, after its retired keys are converted, gives them; the "
         "potential of a fit with no more observations than effective "
         "weights, whose noise variance is undefined (`tadah train` warned "
         "'no weight covariance'): more data with a priority above 0 and a "
         "retrain give them; or weights set from FIXWEIGHT without a fit "
         "(tadah hpo with every weight pinned and no training data), which "
         "have no covariance.";
}

}  // namespace

void remove_weight_covariance(tadah::core::Context &c) {
  for (const char *key : {"SIGMA", "WCOV", "WNOISE"})
    if (c.has_stored(key)) c.remove(key);
}

void write_weight_covariance(tadah::core::Context &c,
                             const tadah::core::Matrix &cov,
                             const PerRowType &noise) {
  remove_weight_covariance(c);
  const std::size_t n = cov.rows();
  if (cov.cols() != n)
    throw std::logic_error("write_weight_covariance: WCOV must be square, not " +
                           std::to_string(cov.rows()) + " x " +
                           std::to_string(cov.cols()));
  c.add("WCOV", n);
  for (std::size_t col = 0; col < n; ++col)
    for (std::size_t row = 0; row < n; ++row) c.add("WCOV", cov(row, col));
  for (std::size_t t = 0; t < 3; ++t) {
    if (!noise.set[t]) continue;
    c.add("WNOISE", std::string(kTypeLetter[t]));
    c.add("WNOISE", noise.value[t]);
  }
}

WeightCovariance read_weight_covariance(const tadah::core::Context &c,
                                        std::size_t n_weights) {
  if (!c.has_stored("WCOV") || !c.has_stored("WNOISE"))
    throw std::runtime_error(why_no_covariance(c));
  // A potential written before the error scales has a WNOISE in other
  // units (the noise per unit row weight over the retired global weights,
  // the energy's per structure), and no ESIGMA line: not read as if it
  // were in the units of ESIGMA, FSIGMA and SSIGMA.
  if (!c.has_stored("ESIGMA"))
    throw std::runtime_error(
        kNoCovariance + " in the units it gives: the potential has WNOISE "
        "but no ESIGMA line, so a fit before the error scales (ESIGMA, "
        "FSIGMA, SSIGMA) wrote it, with the noise of a total energy and in "
        "the units of the retired weights. A retrain with OALGO 3 writes "
        "WCOV and WNOISE with the error scales.");
  const std::string retrain =
      "`tadah train` writes WCOV and WNOISE of its own fit: retrain the "
      "potential";
  const std::size_t stored = c.size("WCOV");
  const std::size_t N = c.get<std::size_t>("WCOV");
  if (N != n_weights)
    throw std::runtime_error(
        "WCOV: N = " + std::to_string(N) + ", but the potential has " +
        std::to_string(n_weights) + " WEIGHTS: the covariance is not of "
        "these weights. " + retrain + ".");
  if (stored != N * N + 1)
    throw std::runtime_error(
        "WCOV: N = " + std::to_string(N) + " needs N*N+1 = " +
        std::to_string(N * N + 1) + " values (N and the N x N matrix), but "
        "WCOV has " + std::to_string(stored) + ". " + retrain + ".");
  WeightCovariance wc;
  wc.cov = tadah::core::Matrix(N, N);
  for (std::size_t col = 0; col < N; ++col)
    for (std::size_t row = 0; row < N; ++row)
      wc.cov(row, col) = c.get<double>("WCOV", 1 + N * col + row);

  const std::size_t tokens = c.size("WNOISE");
  if (tokens % 2 != 0)
    throw std::runtime_error(
        "WNOISE: " + std::to_string(tokens) + " values; it holds pairs of a "
        "row type (E, F or S) and its noise variance.");
  for (std::size_t i = 0; i < tokens; i += 2) {
    const std::string t = c.get<std::string>("WNOISE", i);
    std::size_t k = 3;
    for (std::size_t j = 0; j < 3; ++j)
      if (t == kTypeLetter[j]) k = j;
    if (k == 3)
      throw std::runtime_error("WNOISE: '" + t +
                               "' is not a row type (E, F or S).");
    if (wc.noise.set[k])
      throw std::runtime_error("WNOISE: " + t + " is listed twice.");
    const double v = c.get<double>("WNOISE", i + 1);
    if (!(v >= 0.0))
      throw std::runtime_error("WNOISE: the noise variance of " + t + ", " +
                               c.get<std::string>("WNOISE", i + 1) +
                               ", is negative.");
    wc.noise.value[k] = v;
    wc.noise.set[k] = true;
  }
  return wc;
}

void require_noise_of(const WeightCovariance &wc, bool force, bool stress) {
  // Every prediction has the energy of each structure.
  if (!wc.noise.set[kEnergyRow])
    throw std::runtime_error(
        "--error: every prediction has an energy, but the potential's WNOISE "
        "lists no E: its fit observed no energies (ENERGY false, or every "
        "structure's energy priority 0), so it has no noise variance of "
        "them, and no energy has a predictive standard deviation. A "
        "prediction without --error, or a retrain with energies, gives "
        "the energies.");
  struct Ask {
    bool asked;
    RowType type;
    const char *what, *key, *priority, *flag;
  };
  for (const Ask &a : {Ask{force, kForceRow, "forces", "FORCE", "fweight", "-F"},
                       Ask{stress, kStressRow, "stresses", "STRESS", "sweight", "-S"}}) {
    if (!a.asked || wc.noise.set[a.type]) continue;
    throw std::runtime_error(
        std::string("--error: ") + a.what + " were asked for (" + a.flag +
        ", " + a.key + " true), but the potential's WNOISE lists no " +
        kTypeLetter[a.type] + ": its fit observed no " + a.what + " (" +
        a.key + " false, or every structure's " + a.priority + " 0), so it "
        "has no noise variance of them. The " + a.what + " without --error, "
        "or a retrain with " + a.what + ", give them.");
  }
}

bool weights_on_raw_descriptors(const tadah::core::Context &c,
                                const tadah::models::Function_Base &f) {
  return c.get<bool>("NORM") && tadah::models::is_linear_function(f);
}

tadah::core::Matrix raw_descriptor_map(const Normaliser &norm, std::size_t n,
                                       std::size_t n_intercepts) {
  if (n_intercepts == 0)
    throw std::runtime_error(
        "NORM true with a linear model (BF_Linear, Kern_Linear) requires BIAS "
        "true: the mean correction of the normalised weights goes into the "
        "intercept columns");
  if (n_intercepts > n || norm.std_dev.size() < n || norm.mean.size() < n)
    throw std::logic_error(
        "raw_descriptor_map: " + std::to_string(n) + " weights, " +
        std::to_string(n_intercepts) + " intercept columns, " +
        std::to_string(norm.mean.size()) + " NMEAN and " +
        std::to_string(norm.std_dev.size()) + " NSTDEV values");
  tadah::core::Matrix J(n, n);
  J.set_zero();
  for (std::size_t b = 0; b < n_intercepts; ++b) J(b, b) = 1.0;
  for (std::size_t i = n_intercepts; i < n; ++i) {
    const double scale = norm.std_dev[i] > std::numeric_limits<double>::min()
                             ? 1.0 / norm.std_dev[i]
                             : 1.0;
    J(i, i) = scale;
    for (std::size_t b = 0; b < n_intercepts; ++b)
      J(b, i) = -norm.mean[i] * scale;
  }
  return J;
}

void to_raw_descriptors(tadah::models::M_Core &m, const Normaliser &norm,
                        std::size_t n_intercepts) {
  const std::size_t n = m.weights.size();
  const tadah::core::Matrix J = raw_descriptor_map(norm, n, n_intercepts);
  tadah::core::t_type w(n);
  for (std::size_t i = 0; i < n; ++i) {
    double v = 0.0;
    for (std::size_t k = 0; k < n; ++k) v += J(i, k) * m.weights(k);
    w(i) = v;
  }
  m.weights = w;
  if (!m.has_covariance()) return;
  if (m.Sigma.rows() != n || m.Sigma.cols() != n)
    throw std::logic_error("to_raw_descriptors: a " +
                           std::to_string(m.Sigma.rows()) + " x " +
                           std::to_string(m.Sigma.cols()) +
                           " covariance of " + std::to_string(n) + " weights");
  // T = Sigma J^T, then J T; symmetric by construction, made exactly so.
  tadah::core::Matrix T(n, n), S(n, n);
  for (std::size_t b = 0; b < n; ++b)
    for (std::size_t k = 0; k < n; ++k) {
      double v = 0.0;
      for (std::size_t l = 0; l < n; ++l) v += m.Sigma(k, l) * J(b, l);
      T(k, b) = v;
    }
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = a; b < n; ++b) {
      double v = 0.0;
      for (std::size_t k = 0; k < n; ++k) v += J(a, k) * T(k, b);
      S(a, b) = v;
      S(b, a) = v;
    }
  m.set_covariance(S, m.noise_variance);
}

PerRowType noise_of_fitted_types(const tadah::core::Context &c,
                                 const std::array<bool, 3> &fitted, double s2) {
  PerRowType noise;
  const ErrorScales scales = ErrorScales::of(c);
  for (std::size_t t = 0; t < 3; ++t) {
    if (!fitted[t]) continue;
    const double sigma = scales.absolute(t);
    noise.value[t] = s2 * sigma * sigma;
    noise.set[t] = true;
  }
  return noise;
}

}  // namespace mlip
}  // namespace tadah
