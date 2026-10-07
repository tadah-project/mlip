#ifndef TADAH_MLIP_WEIGHT_PROVISIONING_H
#define TADAH_MLIP_WEIGHT_PROVISIONING_H

#include <tadah/core/context.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>

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

/** True when the model draws its basis vectors from the training data:
 *  M_KRR with a non-linear kernel (`MODEL M_KRR <kernel>` with a kernel
 *  other than Kern_Linear). Weight k belongs to basis vector k (column 0
 *  the bias vector e0, the others drawn from the training AEDs by the
 *  fixed draw of Basis), so such a model needs its training data even when
 *  FIXINDEX pins every weight. */
bool basis_drawn_from_training_data(const tadah::core::Context &ctx);

/** The M_KRR implementation of `MODEL M_KRR <kernel> [1|2]`: its third
 *  token, 1 when there is none (M_KRR::train). */
int krr_implementation(const tadah::core::Context &ctx);

/** The number of model weights K a training of this configuration fits:
 *  the design-matrix width fb.get_phi_cols(ctx), except for a model whose
 *  basis is drawn from the training data (basis_drawn_from_training_data),
 *  whose K is SBASIS, the number of basis vectors training draws (before
 *  training fb has none, and its width is 0). Throws, naming SBASIS, when
 *  such a model has no SBASIS or SBASIS < 2. */
std::size_t model_weight_count(DM_Function_Base &fb,
                               const tadah::core::Context &ctx);

/** Classify how the model weights are provisioned for this configuration.
 *
 * Throws std::runtime_error with an actionable message on illegal
 * combinations: a lone FIXINDEX or FIXWEIGHT, a FIXINDEX/FIXWEIGHT size
 * mismatch, an out-of-bounds index, no DBFILE without any pinning, a
 * missing DBFILE with incomplete FIXINDEX coverage, or a missing DBFILE
 * for a model whose basis is drawn from the training data.
 *
 * @param ctx    training configuration
 * @param n_cols total number of model weights K (model_weight_count())
 */
WeightSource classify_weight_source(const tadah::core::Context &ctx,
                                    std::size_t n_cols);

/** Set the model weights directly from FIXINDEX/FIXWEIGHT — no training data.
 *
 * Scatters w[FIXINDEX[k]-1] = FIXWEIGHT[k], sets the weights on the model
 * and marks it trained. Requires FIXINDEX to cover every weight 1..n_cols;
 * throws listing the uncovered weight indices otherwise. Refuses a model
 * whose basis is drawn from the training data
 * (basis_drawn_from_training_data): without training it has no basis
 * vectors for its weights.
 */
void provision_fixed_weights(M_Tadah_Base &model,
                             const tadah::core::Context &ctx,
                             std::size_t n_cols);

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_WEIGHT_PROVISIONING_H
