#ifndef TADAH_MLIP_RAW_DESCRIPTOR_WEIGHTS_H
#define TADAH_MLIP_RAW_DESCRIPTOR_WEIGHTS_H

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/function_base.h>
#include <tadah/models/m_core.h>

#include <cstddef>

namespace tadah {
namespace mlip {

class Normaliser;

/** True for a model whose WEIGHTS act on the raw descriptors although it
 *  was fitted to normalised ones: NORM true and a linear function
 *  (BF_Linear, Kern_Linear; models::is_linear_function). Its fit converts
 *  the weights to the raw descriptors (raw_descriptor_map), and it predicts
 *  from raw descriptors; every other model with NORM true predicts from
 *  the normalised descriptors. The one predicate of tadah predict and the
 *  fit. */
bool weights_on_raw_descriptors(const tadah::core::Context &c,
                                const tadah::models::Function_Base &f);

/** The linear map J from the weights of the normalised descriptors to
 *  those of the raw descriptors, w_raw = J w, of a linear model with NORM
 *  true and @p n_intercepts intercept columns (the first ones, one per
 *  ATOM species with BIAS true): an @p n x @p n matrix with J(i,i) =
 *  1/stdev_i for a descriptor column i (1 when its stdev is 0) and, in each
 *  intercept row b, J(b,b) = 1 and J(b,i) = -mean_i J(i,i) (the constant
 *  the normalisation moves into the intercepts).
 *  @throws std::runtime_error when @p n_intercepts is 0 (BIAS false: no
 *  column absorbs that constant). */
tadah::core::Matrix raw_descriptor_map(const Normaliser &norm, std::size_t n,
                                       std::size_t n_intercepts);

/** The weights of model @p m, fitted to normalised descriptors, mapped to
 *  the raw descriptors (raw_descriptor_map J: w <- J w): the conversion of
 *  a linear NORM model after its fit, M_BLR and M_KRR alike. */
void to_raw_descriptors(tadah::models::M_Core &m, const Normaliser &norm,
                        std::size_t n_intercepts);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_RAW_DESCRIPTOR_WEIGHTS_H
