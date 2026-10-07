#ifndef TADAH_MLIP_MODELS_PREDICT_WITH_ERROR_H
#define TADAH_MLIP_MODELS_PREDICT_WITH_ERROR_H

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/lapack.h>
#include <tadah/mlip/descriptors_calc_base.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/weight_covariance.h>

#include <cmath>
#include <cstddef>

namespace tadah {
namespace mlip {

/** The values a linear-in-the-weights model predicts for @p stdb, with the
 *  predictive standard deviation of each: M_BLR and M_KRR predict(ctx,
 *  stdb, dc, err), `tadah predict --error`.
 *
 *  The rows phi_i of the design matrix of the function @p f are built as
 *  `tadah predict` evaluates the model: unweighted, without augmentation
 *  rows, from the raw descriptors for a model whose @p weights act on them
 *  (weights_on_raw_descriptors: NORM true and a linear function), else
 *  normalised with @p norm when NORM is true. Each value is phi_i . w,
 *  and its error
 *
 *      err_i = sqrt(phi_i^T WCOV phi_i + WNOISE_t),
 *
 *  t the row type (energy of a structure, force component, stress
 *  component): the variance of the predicted value plus the noise variance
 *  of an observation of it. @p err holds one entry per row, in the order
 *  of the design matrix (each structure's energy, then its 3 force
 *  components per atom with FORCE, then its 6 stress components with
 *  STRESS), in raw units; the energy is that of the structure.
 *
 *  WCOV and WNOISE are read from @p c and checked first
 *  (read_weight_covariance: N against the WEIGHTS, the counts), and a
 *  prediction of a row type WNOISE has no noise of is refused
 *  (require_noise_of): every prediction of a potential fitted without
 *  energies (no E: EWEIGHT 0), and a force or stress prediction (FORCE /
 *  STRESS true in @p c) of one fitted without them. With @p with_noise
 *  false too: the parts of tadah properties pairwise go with the total.
 *
 *  With @p with_noise false each error is sqrt(phi_i^T WCOV phi_i), the
 *  standard deviation of the predicted value alone: for a part of a
 *  potential, which is not an observed quantity (tadah properties pairwise's
 *  two-body and many-body parts).
 *
 *  F is the function type of the model, as its DesignMatrix takes it (a
 *  reference type such as DM_Function_Base& for the models the factory
 *  builds): name it, predict_with_error<BF>(...). */
template <typename F>
StructureDB predict_with_error(const tadah::core::Context &c, F &f,
                               const tadah::core::t_type &weights,
                               Normaliser &norm, StructureDB &stdb,
                               DC_Base &dc, tadah::core::aed_type &err,
                               bool with_noise = true) {
  const WeightCovariance wc = read_weight_covariance(c, weights.size());
  const bool force = c.get<bool>("FORCE");
  const bool stress = c.get<bool>("STRESS");
  require_noise_of(wc, force, stress);

  DesignMatrix<F> dm(f, c);
  dm.apply_weights = false;  // predict the values, not the weighted rows
  dm.augment = false;        // the rows of the structures only
  dm.normalise_descriptors = !weights_on_raw_descriptors(c, f);
  bool wcopy = false;
  dm.build(stdb, norm, dc, wcopy);

  tadah::core::phi_type &Phi = dm.getPhi();
  const tadah::core::t_type &labels = dm.getTlabels();
  const tadah::core::aed_type var = tadah::core::T_MDMT_sym(Phi, wc.cov);
  tadah::core::aed_type w = weights;
  tadah::core::aed_type Tpred = tadah::core::T_dgemv(Phi, w);

  err = tadah::core::aed_type(Phi.rows());
  for (std::size_t i = 0; i < Phi.rows(); ++i) {
    const std::size_t t = static_cast<std::size_t>(labels(i));
    err(i) = std::sqrt(var(i) + (with_noise ? wc.noise.value[t] : 0.0));
  }

  StructureDB stdb_;
  stdb_.structures.resize(stdb.size());
  std::size_t i = 0;
  for (std::size_t s = 0; s < stdb.size(); ++s) {
    stdb_(s).atoms.resize(stdb(s).natoms());
    stdb_(s).energy = Tpred(i++);
    if (force)
      for (std::size_t a = 0; a < stdb(s).natoms(); ++a)
        for (std::size_t k = 0; k < 3; ++k) stdb_(s).atoms[a].force[k] = Tpred(i++);
    if (stress)
      for (std::size_t x = 0; x < 3; ++x)
        for (std::size_t y = x; y < 3; ++y) {
          stdb_(s).stress(x, y) = Tpred(i++);
          if (x != y) stdb_(s).stress(y, x) = stdb_(s).stress(x, y);
        }
  }
  return stdb_;
}

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_MODELS_PREDICT_WITH_ERROR_H
