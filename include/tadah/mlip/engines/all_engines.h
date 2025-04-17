#ifndef TADAH_MLIP_ENGINES_ALL_ENGINES_H
#define TADAH_MLIP_ENGINES_ALL_ENGINES_H

#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/explain_engine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Registers all known engines with the EngineFactory.
 *
 * Each engine is associated with its dot-separated task name
 * (e.g. "TRAIN" ,"DATA.merge", "PROPERTIES.twobody"). The CLI calls engine_factory::create(...)
 * using that name to construct the correct engine.
 */
void registerAllEngines();

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ALL_ENGINES_H
