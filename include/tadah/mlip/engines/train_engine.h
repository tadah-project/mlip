#ifndef TADAH_MLIP_ENGINES_TRAIN_ENGINE_H
#define TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core training functionality.
 *
 * Delegates model training to the underlying Trainer.
 *
 * @param context The contexturation object.
 * @return int EXIT_SUCCESS or EXIT_FAILURE.
 */
class TrainEngine {
public:
  static int train(tadah::core::Context &context);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_TRAIN_ENGINE_H

