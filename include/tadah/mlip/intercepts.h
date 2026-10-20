#ifndef TADAH_MLIP_INTERCEPTS_H
#define TADAH_MLIP_INTERCEPTS_H

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/functions/function_base.h>

#include <cstddef>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/** \file intercepts.h
 *  \brief The rules of BIAS true: one intercept column per ATOM species.
 *
 *  With BIAS true every atom's descriptor starts with N = size(ATOM)
 *  one-hot intercept columns in ATOM order (core::n_intercept_columns,
 *  core::intercept_columns_by_Z). Which weights of a model are intercepts
 *  is the function's (models::Function_Base::intercept_columns): for the
 *  linear models (BF_Linear, Kern_Linear) weights 1..N (FIXINDEX
 *  numbering) are the per-species energy constants and the descriptor
 *  weights follow; BF_Polynomial2's are the products (k,k); a dot-product
 *  kernel's those of the intercept basis vectors. */

/** \brief The NORM/BIAS rules of a configuration to be trained.
 *
 *  - NORM true with a linear model (BF_Linear, Kern_Linear) needs BIAS
 *    true: the weights fitted to the normalised descriptors are converted
 *    to weights of the raw descriptors, and the constant this moves (the
 *    descriptor means times the weights) goes into the intercept columns;
 *  - BIAS true needs a non-empty ATOM list;
 *  - FIXINDEX/FIXWEIGHT is refused with NORM true: the fit pins weights of
 *    the normalised descriptors, which the conversion to raw-descriptor
 *    weights moves.
 *
 *  Called by `tadah train`, by `tadah hpo` at startup and by M_BLR/M_KRR
 *  before they fit.
 *  @throws std::runtime_error */
void check_bias_constraints(const tadah::core::Context &ctx);

/** \brief The startup WARNING of FIXINDEX with BIAS true and two or more
 *  species, for the model's function @p f (@p ctx holds DSIZE): which
 *  weights are the intercepts (f.intercept_columns: for a linear model
 *  weights 1 to N, and the descriptor weights start at index N + 1; they
 *  started at 2 when BIAS true had one intercept column shared by all
 *  species, before 1.6.0). For a model whose basis its fit draws
 *  (basis_drawn_from_training_data) those of that basis, which starts with
 *  the intercept vectors e_0 .. e_{N-1}: a dot-product kernel's weights 1
 *  to N, named by species; Kern_RBF has none. Empty when it does not
 *  apply. */
std::string fixindex_intercept_warning(const tadah::core::Context &ctx,
                                       const tadah::models::Function_Base &f);

/** \brief Refuses an atom whose element @p e ATOM does not list (BIAS
 *  true: it has no intercept column), naming the element, the ATOM list
 *  and @p where (e.g. the structure).
 *  @throws std::runtime_error */
[[noreturn]] void throw_element_not_in_atom(const tadah::core::Element &e,
                                            const tadah::core::Context &ctx,
                                            const std::string &where);

/** \brief Refuses BIAS true training data that cannot determine the
 *  intercepts of the model's function @p f.
 *
 *  An intercept column (f.intercept_columns) depends on an atom only
 *  through its species, so only energies constrain it (its force and
 *  stress rows are zero), and the ridge does not penalise it. Over the
 *  structures whose energies the fit observes (ENERGY true and an energy
 *  priority, eweight x WDBFILE, above 0: RowWeights) the energy rows of
 *  the intercept columns that FIXINDEX/FIXWEIGHT do not pin must therefore
 *  have full rank: with every structure of one composition (e.g. a fixed
 *  stoichiometry), or without any energy observation, a fit would split
 *  the energy between the species arbitrarily (or overflow).
 *  The value of intercept column j for an atom of species s is f's
 *  energy of the unit weight j at the descriptor with intercept component
 *  s alone (@p ctx holds DSIZE).
 *
 *  For a model whose basis vectors its fit draws from the training data
 *  (basis_drawn_from_training_data: MODEL M_KRR with a non-linear kernel)
 *  the intercepts are those of the basis the fit draws, whatever basis @p f
 *  holds (none before the fit, or a potential's): every such basis starts
 *  with the intercept vectors e_0 .. e_{N-1}, so a dot-product kernel's
 *  intercepts are its weights 1 to N, before the fit as after it. M_KRR
 *  implementation 2 (is_krr_implementation_2: any kernel, Kern_Linear
 *  too), whose solve (K + LAMBDA I) w = T fits the basis vectors' targets,
 *  not the structures' energies, and pins no weight, is not checked.
 *
 *  A priority of the data that is negative or not finite, and a WDBFILE
 *  without one value per dataset, are refused first
 *  (require_valid_weights), for BIAS false too. Otherwise a no-op for BIAS
 *  false and for a function without intercept columns (Kern_RBF). An atom
 *  of an element not in ATOM is refused
 *  (throw_element_not_in_atom). FIXINDEX is read as the fit reads it,
 *  with core::parse_fixindex and the fit's weight count
 *  (model_weight_count), and what that refuses, an index past the weights
 *  included, is thrown here, before the fit, with the fit's message.
 *  @throws std::runtime_error naming the species, the rank found, the
 *          intercept weights and the remedies, or core::parse_fixindex's
 *          refusal of FIXINDEX/FIXWEIGHT, or require_valid_weights' of a
 *          weight
 *  @throws std::logic_error for a kernel whose basis vectors do not have
 *          DSIZE values (they would be evaluated past the descriptor) */
void require_intercepts_determined(const StructureDB &stdb,
                                   const tadah::core::Context &ctx,
                                   const tadah::models::Function_Base &f);

/** \brief The intercept columns the fit of @p ctx fits, 0-based, before any
 *  fit: those of the fit's function (for a model whose basis its fit
 *  draws, a kernel's of that basis, which starts with the intercept
 *  vectors) that FIXINDEX and FIXWEIGHT do not pin, FIXINDEX read as the
 *  fit reads it (core::parse_fixindex with model_weight_count; what that
 *  refuses is thrown here, with its message). None for BIAS false and for
 *  M_KRR implementation 2, whose solve runs no least squares over them.
 *  @p f is the model's function, built from @p ctx, which holds DSIZE (a
 *  Trainer's).
 *  @throws std::runtime_error */
std::vector<std::size_t>
fitted_intercept_columns(const tadah::core::Context &ctx,
                         const tadah::models::Function_Base &f);

/** \brief Refuses OALGO 3 with a LAMBDA other than 0 when the fit of
 *  @p ctx has an intercept column it fits, before any fit, with the fit's
 *  message (models::LinearRegressor::require_oalgo3_intercepts: OALGO 3
 *  penalises every weight alike, and a ridge penalty leaves the intercept
 *  columns out; LAMBDA -1, the evidence approximation, needs BIAS false or
 *  every intercept weight pinned).
 *
 *  The intercept columns are counted as the fit has them: those of the
 *  fit's function (for a model whose basis its fit draws, a kernel's of
 *  that basis, which starts with the intercept vectors) that FIXINDEX and
 *  FIXWEIGHT do not pin, FIXINDEX read as the fit reads it (core::parse_
 *  fixindex with model_weight_count; what that refuses is thrown here, with
 *  its message). @p f is the model's function, built from @p ctx, which
 *  holds DSIZE (a Trainer's). M_KRR implementation 2, whose solve does not
 *  run the least squares of LinearRegressor, is not checked; nor is BIAS
 *  false, which has no intercept column. tadah hpo calls it when a run
 *  starts.
 *  @throws std::runtime_error */
void require_oalgo3_intercepts(const tadah::core::Context &ctx,
                               const tadah::models::Function_Base &f);

/** \brief The intercept weights of @p f as text, FIXINDEX numbering: "1
 *  (Nb) and 6 (Ta)" (a weight is labelled with its species when it is
 *  non-zero for that species alone). */
std::string intercept_weights_text(const tadah::core::Context &ctx,
                                   const tadah::models::Function_Base &f);

/** \brief Refuses descriptor builders that disagree: the DC_Selector
 *  (pair_style tadah's descriptor) and the DescriptorsCalc built with
 *  @p ctx (its DSIZE) must give the same descriptor size.
 *  @throws std::logic_error naming @p where */
void require_descriptor_layout(const tadah::models::DC_Selector &dcs,
                               const tadah::core::Context &ctx,
                               const std::string &where);

} // namespace mlip
} // namespace tadah

#endif
