#include <tadah/mlip/raw_descriptor_weights.h>

#include <tadah/mlip/normaliser.h>
#include <tadah/models/functions/weights_fit.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace tadah {
namespace mlip {

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
}

}  // namespace mlip
}  // namespace tadah
