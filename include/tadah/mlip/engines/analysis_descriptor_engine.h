#ifndef TADAH_MLIP_ENGINES_ANALYSIS_DESCRIPTOR_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_DESCRIPTOR_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Copmutes and prints descriptors for given structure(s).
 */
class AnalysisDescriptorEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ANALYSIS_DESCRIPTOR_ENGINE_H
