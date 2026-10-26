#ifndef TADAH_MLIP_COMMAND_OPTIONS_H
#define TADAH_MLIP_COMMAND_OPTIONS_H

#include <tadah/core/context.h>

#include <string>

namespace tadah {
namespace mlip {

/**
 * @brief Refuses a run of a command whose model is the potential's alone
 *        (tadah predict, tadah properties pairwise, tadah analysis
 *        descriptor) when its context @p run has a key that is not an
 *        option of the blueprint task @p task ("PREDICT",
 *        "PROPERTIES.pairwise", "ANALYSIS.descriptor").
 *
 * The options are the task's blueprint keys and those of its groups; the
 * internal keys a Context derives itself are none of the run's. A key the
 * run stores that is no option -- a key of the model such as RCUT2B, a
 * DESCRIPTOR block, a key that sets a fit such as WDBFILE, EMBF0 or
 * RHOFORM, the line of a retired key -- would change what the potential
 * predicts, or nothing: the command line gives the options alone, so only
 * a program reaches the refusal.
 *
 * @throws std::runtime_error naming each such key, upper case and sorted,
 *         and each block: "tadah predict takes no keys RCUT2B and WDBFILE
 *         and no DESCRIPTOR block: the model is the potential's, as it was
 *         fitted (its keys are those of the potential file)" ("no key
 *         RCUT2B" for one).
 */
void require_command_options(const core::Context &run,
                             const std::string &task);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_COMMAND_OPTIONS_H
