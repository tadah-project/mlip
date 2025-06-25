#ifndef TADAH_MLIP_ENGINES_ALL_ENGINES_H
#define TADAH_MLIP_ENGINES_ALL_ENGINES_H

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
void registerAllMLIPEngines();

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ALL_ENGINES_H
