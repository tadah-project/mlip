#ifndef TADAH_MLIP_WEIGHT_PROVISIONING_H
#define TADAH_MLIP_WEIGHT_PROVISIONING_H

#include <tadah/core/context.h>
#include <tadah/mlip/models/m_tadah_base.h>

#include <cstddef>

namespace tadah {
namespace mlip {

/** Where the model's linear weights come from for a given configuration.
 *
 * Invariants shared by the FIXINDEX/FIXWEIGHT machinery:
 *  - FIXWEIGHT[k] pairs with FIXINDEX[k] in user declaration order
 *    (core::parse_indices preserves order and removes duplicates);
 *  - FIXINDEX entries are 1-based model-weight indices in [1, n_cols];
 *  - a lone FIXINDEX or FIXWEIGHT is an error — there is no auto-seeding
 *    from WEIGHTS;
 *  - after provision_fixed_weights() the model satisfies
 *    get_weights().size() == n_cols and is_trained() == true;
 *  - on cached trainers, reinit() (which resets the trained flag) must
 *    precede re-provisioning, mirroring the train() contract.
 */
enum class WeightSource {
  FullLS,          ///< no FIXINDEX/FIXWEIGHT; all weights LS-fitted
  PartialLS,       ///< FIXINDEX pins a proper subset; the rest are LS-fitted
  AllFixedData,    ///< FIXINDEX covers every weight; training data present
  FixedOnlyNoData  ///< FIXINDEX covers every weight; no training DBFILE
};

/** Classify how the model weights are provisioned for this configuration.
 *
 * Throws std::runtime_error with an actionable message on illegal
 * combinations: a lone FIXINDEX or FIXWEIGHT, a FIXINDEX/FIXWEIGHT size
 * mismatch, an out-of-bounds index, no DBFILE without any pinning, or a
 * missing DBFILE with incomplete FIXINDEX coverage.
 *
 * @param ctx    training configuration
 * @param n_cols total number of model weights K (fb->get_phi_cols(ctx))
 */
WeightSource classify_weight_source(const tadah::core::Context &ctx,
                                    std::size_t n_cols);

/** Set the model weights directly from FIXINDEX/FIXWEIGHT — no training data.
 *
 * Scatters w[FIXINDEX[k]-1] = FIXWEIGHT[k], sets the weights on the model
 * and marks it trained. Requires FIXINDEX to cover every weight 1..n_cols;
 * throws listing the uncovered weight indices otherwise.
 */
void provision_fixed_weights(M_Tadah_Base &model,
                             const tadah::core::Context &ctx,
                             std::size_t n_cols);

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_WEIGHT_PROVISIONING_H
