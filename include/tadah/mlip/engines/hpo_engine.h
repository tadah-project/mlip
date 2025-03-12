#ifndef TADAH_MLIP_ENGINES_HPO_ENGINE_H
#define TADAH_MLIP_ENGINES_HPO_ENGINE_H

#include <tadah/core/config.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides hyperparameter optimization functionality.
 *
 * Executes HPO routines using the provided configuration.
 *
 * @param config The configuration object.
 * @return int EXIT_SUCCESS or EXIT_FAILURE.
 */
class HPOEngine {
public:
  static int optimize(const tadah::core::Config &config);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPO_ENGINE_H

