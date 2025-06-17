#ifndef TADAH_MLIP_ENGINES_DATA_CONVERT_ENGINE_H
#define TADAH_MLIP_ENGINES_DATA_CONVERT_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core training functionality.
 */
class DataConvertEngine : public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_DATA_CONVERT_ENGINE_H
