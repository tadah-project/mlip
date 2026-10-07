#ifndef TADAH_MLIP_ENGINES_REFIT_MODES_H
#define TADAH_MLIP_ENGINES_REFIT_MODES_H

// The two `tadah refit` modes that operate on EXISTING refit outputs instead
// of a tabulated EAM file:
//  --edit <pot.tadah>      add/remove one basis function (EDIT/FUNC/ADD/
//                          REMOVE/ETA/DEG keys) -> edited pot + regenerated
//                          hpotarget;
//  --retarget <hpotarget>  rewrite OPTIM bounds (RETARGET/PCT/SET/IN_PLACE).
// Dispatched from RefitEngine::run; mode-consistency validation lives in
// validate_refit_mode_keys (also called from RefitEngine::validate so the
// config-file path is covered — CLI needs/excludes only guard flags).

#include <tadah/core/context.h>
#include <tadah/mlip/key_provenance.h>

namespace tadah {
namespace mlip {
namespace engines {

/** \brief Re-validate the mode key combinations that the CLI cannot see
 *  (config-file path): edit XOR retarget XOR normal(EAM); edit needs FUNC +
 *  exactly one of ADD/REMOVE; PCT, SET, RETARGET_RCUT2B, RETARGET_RCUTMB,
 *  RETARGET_RHOFORM and IN_PLACE only with retarget; ETA/DEG only with
 *  ADD; RHOFORM and EMBF0
 *  are refused with retarget (they have no effect on an hpotarget; the
 *  density form is --rhoform); with edit or retarget, a key that a refit
 *  of an EAM file reads and the mode does not
 *  (check_refit_output_mode_keys, refit_keys.h; @p provenance tells the
 *  blueprint defaults the task factory stored, which are not refused, and
 *  without it every stored key counts). Throws std::runtime_error on
 *  violation. */
void validate_refit_mode_keys(const tadah::core::Context &c,
                              const KeyProvenance *provenance = nullptr);

/** \brief `tadah refit --edit`. A RHOFORM or EMBF0 in @p c (the
 *  configuration) that differs from the potential's is a conflict, as in
 *  tadah predict (models::check_rhoform_precedence /
 *  check_embf0_precedence). @p provenance as for validate_refit_mode_keys.
 *  Returns EXIT_SUCCESS. */
int run_refit_edit(tadah::core::Context &c,
                   const KeyProvenance *provenance = nullptr);

/** \brief `tadah refit --retarget`. @p provenance as for
 *  validate_refit_mode_keys. Returns EXIT_SUCCESS. */
int run_refit_retarget(tadah::core::Context &c,
                       const KeyProvenance *provenance = nullptr);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_REFIT_MODES_H
