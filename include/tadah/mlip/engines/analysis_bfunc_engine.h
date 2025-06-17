#ifndef TADAH_MLIP_ENGINES_ANALYSIS_BFUNC_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_BFUNC_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Computes and prints basis functions for D2 and DM descriptors.
 */
class AnalysisBfuncEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ANALYSIS_BFUNC_ENGINE_H
