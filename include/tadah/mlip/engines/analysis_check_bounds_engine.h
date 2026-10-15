#ifndef TADAH_MLIP_ENGINES_ANALYSIS_CHECK_BOUNDS_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_CHECK_BOUNDS_ENGINE_H

#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/** `tadah analysis check_bounds`. Loads HPOTARGET, parses OPTIM
 *  bounds, samples corners, runs the pre-flight audit at each corner.
 *  Returns non-zero exit code if any corner fails. */
class AnalysisCheckBoundsEngine final : public IEngine {
 public:
  void validate(const tadah::mlip::Task& task) override;
  int  run(tadah::mlip::Task& task) override;
};

}  // namespace engines
}  // namespace mlip
}  // namespace tadah

#endif
