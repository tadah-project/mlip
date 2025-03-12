#ifndef TADAH_MLIP_ENGINES_PREDICT_ENGINE_H
#define TADAH_MLIP_ENGINES_PREDICT_ENGINE_H

#include <tadah/core/config.h>
#include <tadah/mlip/structure_db.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core prediction functionality.
 *
 * Uses the trained model to predict properties.
 *
 * @param pot_config The potential configuration.
 * @param stdb The structure database.
 * @return int EXIT_SUCCESS or EXIT_FAILURE.
 */
class PredictEngine {
public:
  static int predict(const tadah::core::Config &pot_config, const tadah::mlip::StructureDB &stdb);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_PREDICT_ENGINE_H

