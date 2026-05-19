#ifndef TADAH_MLIP_ENGINES_ANALYSIS_SUGGEST_BOUNDS_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_SUGGEST_BOUNDS_ENGINE_H

#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Produce data-driven OPTIM bound proposals for the
 * descriptors configured in the user's training config.
 *
 * Reads `-c train.tadah`, computes a `DataStats` over the configured
 * DBFILEs, instantiates each descriptor declared by TYPE2B / TYPEMB,
 * walks its `D_Base::meta()`, and emits one suggestion per parameter.
 * Prints a human table to stdout and (when --outfile is set) writes a
 * compilable `OPTIM` block to disk.
 */
class AnalysisSuggestBoundsEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif  // TADAH_MLIP_ENGINES_ANALYSIS_SUGGEST_BOUNDS_ENGINE_H
