#ifndef TADAH_MLIP_WEIGHT_PROVISIONING_H
#define TADAH_MLIP_WEIGHT_PROVISIONING_H

#include <tadah/core/context.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/models/functions/function_base.h>

#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

/** Where the model's linear weights come from for a given configuration.
 *
 * Invariants shared by the FIXINDEX/FIXWEIGHT machinery:
 *  - FIXWEIGHT[k] pairs with FIXINDEX[k] in user declaration order
 *    (core::parse_fixindex reads FIXINDEX: it keeps that order and refuses
 *    a weight named twice and a count other than FIXWEIGHT's);
 *  - FIXINDEX entries are 1-based model-weight indices in [1, n_cols];
 *  - a lone FIXINDEX or FIXWEIGHT is an error — there is no auto-seeding
 *    from WEIGHTS;
 *  - after provision_fixed_weights() the model satisfies
 *    get_weights().size() == n_cols and is_trained() == true; a model is
 *    provisioned once, as it is trained once (tadah hpo builds a Trainer
 *    for every evaluation).
 */
enum class WeightSource {
  FullLS,          ///< no FIXINDEX/FIXWEIGHT; all weights LS-fitted
  PartialLS,       ///< FIXINDEX pins a proper subset; the rest are LS-fitted
  AllFixedData,    ///< FIXINDEX covers every weight; training data present
  FixedOnlyNoData  ///< FIXINDEX covers every weight; no training DBFILE
};

/** True when the model draws its basis vectors from the training data:
 *  M_KRR with a non-linear kernel (`MODEL M_KRR <kernel>` with a kernel
 *  other than Kern_Linear). Weight k belongs to basis vector k (columns 0
 *  .. n0-1 the intercept vectors, one per ATOM species with BIAS true, e0
 *  alone otherwise; the others drawn from the training data by the fixed
 *  draw of Basis), so such a model needs its training data even when
 *  FIXINDEX pins every weight. */
bool basis_drawn_from_training_data(const tadah::core::Context &ctx);

/** The M_KRR implementation of `MODEL M_KRR <kernel> [1|2]`: its third
 *  token, 1 when there is none (M_KRR::train). */
int krr_implementation(const tadah::core::Context &ctx);

/** Refuses an M_KRR implementation other than 1 and 2 (the third value of
 *  `MODEL M_KRR <kernel> [1|2]`): "MODEL M_KRR <kernel> <n>: M_KRR has
 *  implementation 1 ... and implementation 2 ...; <n> is neither." A
 *  MODEL without the third value (implementation 1) and another model are
 *  accepted. M_KRR::train calls it; tadah hpo calls it when a run starts.
 *  @throws std::runtime_error */
void require_krr_implementation(const tadah::core::Context &ctx);

/** The LAMBDA rules of the fit of @p ctx, checked without a fit, with the
 *  fit's messages: those of M_KRR implementation 2 (MODELS
 *  require_krr_implementation2_lambda) for `MODEL M_KRR <kernel> 2`, else
 *  those of LinearRegressor::train (require_lambda_for_oalgo), which fits
 *  M_BLR and M_KRR implementation 1. tadah hpo calls it when a run starts.
 *  @throws std::runtime_error naming the value. */
void require_fit_lambda(const tadah::core::Context &ctx);

/** True for M_KRR implementation 2, `MODEL M_KRR <kernel> 2` with any
 *  kernel, Kern_Linear too: its solve (MODELS M_KRR_Train::train2) fits
 *  the targets of SBASIS basis vectors drawn from the training structures,
 *  every weight at once, and pins none. */
bool is_krr_implementation_2(const tadah::core::Context &ctx);

/** True when the fit of this configuration draws SBASIS basis vectors
 *  from the training data (Basis): M_KRR with a non-linear kernel
 *  (basis_drawn_from_training_data), and M_KRR implementation 2 with any
 *  kernel, Kern_Linear too (is_krr_implementation_2: its solve, MODELS
 *  M_KRR_Train::train2, fits the targets of SBASIS basis vectors). */
bool fit_draws_basis(const tadah::core::Context &ctx);

/** The SBASIS of a fit that draws basis vectors (fit_draws_basis),
 *  checked: present, and at least the n0 intercept vectors plus one drawn
 *  vector (n0 + 1: the bound of the basis draw,
 *  sbasis_lower_bound_problem; 3 with BIAS true on two species). Throws
 *  naming the model, "M_KRR <kernel> <implementation>" ("M_KRR Kern_Linear
 *  2 needs SBASIS, ..."; "M_KRR Kern_RBF 1: SBASIS 1: the fit needs at
 *  least 2 basis vectors ..."). The upper bound, the number of candidates,
 *  needs the data and is the draw's (Basis::check_sbasis). @p ctx holds
 *  ATOM when BIAS is true (n_intercept_basis_vectors). */
std::size_t checked_sbasis(const tadah::core::Context &ctx);

/** The number of model weights K a training of this configuration fits:
 *  the design-matrix width, f.weights_count(DSIZE) (as
 *  DM_Function_Base::get_phi_cols(ctx)), except for a model whose basis is
 *  drawn from the training data (basis_drawn_from_training_data), whose K
 *  is SBASIS, the number of basis vectors training draws (before training
 *  f has none, and its width is 0). M_KRR Kern_Linear implementation 2 has
 *  DSIZE: its solve finds the coefficients of SBASIS basis vectors, and its
 *  weights are the descriptor weights they make (MODELS
 *  M_KRR_Train::train2). For every fit that draws basis vectors
 *  (fit_draws_basis), Kern_Linear 2 included, SBASIS is checked first
 *  (checked_sbasis, which throws naming the model and SBASIS). @p ctx
 *  holds DSIZE (DescriptorsCalc sets it). */
std::size_t model_weight_count(const tadah::models::Function_Base &f,
                               const tadah::core::Context &ctx);

/** Classify how the model weights are provisioned for this configuration.
 *
 * Throws std::runtime_error with an actionable message on illegal
 * combinations: FIXINDEX or FIXWEIGHT with M_KRR implementation 2
 * (is_krr_implementation_2; MODELS refuse_fixed_weights_in_train2's
 * message, before FIXINDEX is read), what core::parse_fixindex refuses (a
 * lone FIXINDEX or FIXWEIGHT, a malformed or descending entry, an index
 * outside 1..n_cols, a FIXINDEX/FIXWEIGHT count mismatch, a weight named
 * twice), a FIXWEIGHT value that is not a number, no DBFILE without any
 * pinning, a missing DBFILE with incomplete FIXINDEX coverage, or a
 * missing DBFILE for a model whose basis is drawn from the training data.
 * Each refusal of a missing DBFILE ends with
 * potential_has_no_datasets_advice() (training_datasets.h): a potential,
 * which has no DBFILE line, is the likely configuration.
 *
 * @param ctx    training configuration
 * @param n_cols total number of model weights K (model_weight_count())
 */
WeightSource classify_weight_source(const tadah::core::Context &ctx,
                                    std::size_t n_cols);

/** The WARNING of a training whose FIXINDEX and FIXWEIGHT pin every weight
 *  (classify_weight_source() is WeightSource::AllFixedData): the fit
 *  changes none of them (MODELS M_BLR_Train and M_KRR_Train skip the
 *  regressor), so the potential written has the FIXWEIGHT values as its
 *  WEIGHTS. "tadah train: FIXINDEX and FIXWEIGHT pin every weight (all K
 *  of them), so the fit changes none: the WEIGHTS of the potential written
 *  are the FIXWEIGHT values. tadah refit and refit --edit write FIXINDEX
 *  and FIXWEIGHT for every weight of their potentials, the seed of a tadah
 *  hpo coefficient search, and so does a tadah hpo run that pins every
 *  weight in its best_pot.tadah: to fit the weights to the DBFILE data,
 *  delete both lines from the training configuration (the Training page
 *  of the documentation, Retraining from a potential)."; "" without
 *  FIXINDEX and FIXWEIGHT or when a weight is left to fit. A warning, not a
 *  refusal: such a training can be deliberate. TrainEngine::run logs it
 *  before any dataset is read; tadah hpo pins every weight on purpose (its
 *  coefficient search) and does not. Throws what model_weight_count() and
 *  classify_weight_source() throw. @p ctx holds DSIZE (DescriptorsCalc
 *  sets it) and the training DBFILE. */
std::string all_weights_pinned_warning(const tadah::core::Context &ctx,
                                       const tadah::models::Function_Base &f);

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
