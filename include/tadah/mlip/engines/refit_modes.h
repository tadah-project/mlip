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

namespace tadah {
namespace mlip {
namespace engines {

/** \brief Re-validate the mode key combinations that the CLI cannot see
 *  (config-file path): edit XOR retarget XOR normal(EAM); edit needs FUNC +
 *  exactly one of ADD/REMOVE; PCT/SET/IN_PLACE only with retarget; ETA/DEG
 *  only with ADD. Throws std::runtime_error on violation. */
void validate_refit_mode_keys(const tadah::core::Context &c);

/** \brief `tadah refit --edit`. Returns EXIT_SUCCESS. */
int run_refit_edit(tadah::core::Context &c);

/** \brief `tadah refit --retarget`. Returns EXIT_SUCCESS. */
int run_refit_retarget(tadah::core::Context &c);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_REFIT_MODES_H
