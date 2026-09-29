#ifndef TADAH_MLIP_ENGINES_TRAIN_ENGINE_H
#define TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core training functionality.
 */
class TrainEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
  /** Writes @p context as a potential file (NUMERIC digits, runtime keys
   *  removed, EMBF0 and RHOFORM normalised). A context with a DM_REAM block
   *  but no RHOFORM throws models::DensityFormError before the file is
   *  opened. */
  static void dumpPotFile(const std::string &outfile, const core::Context &context);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

