#ifndef TADAH_MLIP_ENGINES_HPO_ENGINE_H
#define TADAH_MLIP_ENGINES_HPO_ENGINE_H

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides hyperparameter optimization functionality.
 *
 * Executes HPO routines using the provided contexturation.
 *
 * @param context The contexturation object.
 * @return int EXIT_SUCCESS or EXIT_FAILURE.
 */
class HPOEngine {
public:
  static int optimize(const tadah::core::Context &context);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPO_ENGINE_H

