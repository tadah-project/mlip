#ifndef TADAH_MLIP_WEIGHT_COVARIANCE_H
#define TADAH_MLIP_WEIGHT_COVARIANCE_H

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/models/functions/function_base.h>
#include <tadah/models/m_core.h>

#include <array>
#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

class Normaliser;

/** The row types of a design matrix (DesignMatrix::getTlabels()): energies,
 *  force components, stress components; their WNOISE labels E, F, S. */
enum RowType : std::size_t { kEnergyRow = 0, kForceRow = 1, kStressRow = 2 };

/** One value per row type (E, F, S) and whether it is set. */
struct PerRowType {
  std::array<double, 3> value{{0.0, 0.0, 0.0}};
  std::array<bool, 3> set{{false, false, false}};
};

/** The weight covariance of a potential (keys WCOV and WNOISE).
 *
 *  WCOV:   N, then the N x N covariance of the WEIGHTS, column-major, in
 *          their own parameterisation (that of the raw descriptors for a
 *          linear model with NORM true).
 *  WNOISE: for each row type the fit observed, its letter (E, F, S) and
 *          the noise variance of one observation of priority 1 in the units
 *          of the data, s^2 sigma_t^2 (noise_of_fitted_types(): sigma_t the
 *          absolute error scale of the type, ErrorScales::absolute(); the
 *          energy's per atom, or per sqrt(atom) under ENORM SQRT, and the
 *          stress's of a diagonal entry).
 *
 *  An OALGO 3 fit writes both (M_BLR / M_KRR get_param_file); `tadah
 *  predict --error` reads them. They replace SIGMA, which earlier versions
 *  of Tadah! wrote with a transposed formula (a deprecated key now). */
struct WeightCovariance {
  tadah::core::Matrix cov;
  PerRowType noise;
};

/** Removes SIGMA, WCOV and WNOISE from @p c: a potential holds the
 *  covariance of its own fit only. */
void remove_weight_covariance(tadah::core::Context &c);

/** Writes WCOV (@p cov, N x N) and WNOISE (the set entries of @p noise,
 *  in E, F, S order) into @p c, replacing SIGMA, WCOV and WNOISE. */
void write_weight_covariance(tadah::core::Context &c,
                             const tadah::core::Matrix &cov,
                             const PerRowType &noise);

/** Reads WCOV and WNOISE of a potential with @p n_weights WEIGHTS. WNOISE
 *  lists the row types its fit had (a fit without energies has no E: see
 *  require_noise_of()).
 *  @throws std::runtime_error, naming the keys, when either is absent, or
 *  when the potential has no ESIGMA line (a fit before the error scales
 *  wrote its WNOISE, in other units),
 *  with why the potential of @p c has none and what to do: another OALGO
 *  than 3 (retrain with OALGO 3), M_KRR implementation 2
 *  (is_krr_implementation_2(): MODEL M_KRR <kernel> 1), or, for an OALGO 3
 *  potential, a potential of an earlier version (SIGMA), a fit with no
 *  more observations than effective weights, or weights set from
 *  FIXWEIGHT without a fit; when WCOV's N is not @p n_weights, or WCOV does
 *  not hold N*N+1 values (checked before its values are read); when WNOISE
 *  is not pairs of a row type E, F or S, each at most once, and a
 *  variance >= 0. */
WeightCovariance read_weight_covariance(const tadah::core::Context &c,
                                        std::size_t n_weights);

/** Refuses a prediction with errors of a row type WNOISE does not list:
 *  energies, which every prediction has, of a potential whose fit observed
 *  no energies (ENERGY false, or every structure's energy priority 0);
 *  forces (@p force) or stresses (@p stress) of a potential whose fit
 *  observed none of that type.
 *  @throws std::runtime_error naming the quantity, WNOISE, the cause and
 *  the remedy. */
void require_noise_of(const WeightCovariance &wc, bool force, bool stress);

/* is_krr_implementation_2() (weight_provisioning.h) tells M_KRR
 * implementation 2 (MODEL M_KRR <kernel> 2), whose fit computes no weight
 * covariance (models::krr_implementation2_without_covariance()). */

/** True for a model whose WEIGHTS act on the raw descriptors although it
 *  was fitted to normalised ones: NORM true and a linear function
 *  (BF_Linear, Kern_Linear; models::is_linear_function). Its fit converts
 *  the weights (and their covariance) to the raw descriptors
 *  (raw_descriptor_map), and it predicts from raw descriptors; every other
 *  model with NORM true predicts from the normalised descriptors. The one
 *  predicate of both, for tadah predict, its --error and the fit. */
bool weights_on_raw_descriptors(const tadah::core::Context &c,
                                const tadah::models::Function_Base &f);

/** The linear map J from the weights of the normalised descriptors to
 *  those of the raw descriptors, w_raw = J w (and covariance J Sigma
 *  J^T), of a linear model with NORM true and @p n_intercepts intercept
 *  columns (the first ones, one per ATOM species with BIAS true): an
 *  @p n x @p n matrix with J(i,i) = 1/stdev_i for a descriptor column i
 *  (1 when its stdev is 0) and, in each intercept row b, J(b,b) = 1 and
 *  J(b,i) = -mean_i J(i,i) (the constant the normalisation moves into
 *  the intercepts).
 *  @throws std::runtime_error when @p n_intercepts is 0 (BIAS false: no
 *  column absorbs that constant). */
tadah::core::Matrix raw_descriptor_map(const Normaliser &norm, std::size_t n,
                                       std::size_t n_intercepts);

/** The weights of model @p m, fitted to normalised descriptors, mapped to
 *  the raw descriptors (raw_descriptor_map J: w <- J w), and its weight
 *  covariance with them (Sigma <- J Sigma J^T) when it has one: the
 *  conversion of a linear NORM model after its fit, M_BLR and M_KRR
 *  alike. */
void to_raw_descriptors(tadah::models::M_Core &m, const Normaliser &norm,
                        std::size_t n_intercepts);

/** WNOISE of a fit: for each row type in @p fitted (the types the fit
 *  observed, DesignMatrix::fitted_types()), the noise variance of one
 *  observation of priority 1 in the units of the data, @p s2 sigma_t^2:
 *  s2 the noise variance per unit row weight of the fit (a row's weight is
 *  p m / sigma^2, row_weights.h) and sigma_t the absolute error scale of
 *  the type in @p c (ErrorScales::absolute(): ESIGMA/1000 eV, FSIGMA a
 *  eV/A, SSIGMA a in eV/A^3). The energy's is that of the energy per atom
 *  (per sqrt(atom) under ENORM SQRT), the stress's that of a diagonal
 *  entry (an off-diagonal one, which the fit counts twice, has half of
 *  it); neither a structure's priority nor the relative part of FSIGMA and
 *  SSIGMA is in it. */
PerRowType noise_of_fitted_types(const tadah::core::Context &c,
                                 const std::array<bool, 3> &fitted, double s2);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_WEIGHT_COVARIANCE_H
