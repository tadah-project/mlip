#ifndef TADAH_MLIP_ALL_ENGINES_H
#define TADAH_MLIP_ALL_ENGINES_H

#include <tadah/mlip/iengine.h>

/**
 * This header declares multiple engine classes, each implementing IEngine.
 * For example, "TrainEngine", "PredictEngine", "DataConvertEngine", etc.
 *
 * Each matches a blueprint task, e.g.:
 *   - "TRAIN"
 *   - "DATA.CONVERT"
 *   - "PROPERTIES.TWOBODY"
 *   - ...
 *
 * Implementation stubs are in all_engines.cpp.
 */

namespace tadah {
namespace mlip {

// Forward declarations of each engine class
class TrainEngine;
class PredictEngine;
class HpoEngine;
class DataEngine;
class DataConvertEngine;
class DataReadEngine;
class DataWriteEngine;
class DataMergeEngine;
class DataSplitEngine;
class DataDedupEngine;
class DataSampleEngine;
class DataBalanceEngine;
class TransformEngine;
class PropertiesTwobodyEngine;
class PropertiesEcurveEngine;
class PropertiesEosEngine;
class PropertiesDefectEngine;
class PropertiesSurfaceEngine;
class PropertiesMechanicsEngine;
class AnalysisEngine;
class AnalysisDescriptorEngine;
class AnalysisBfuncEngine;
class AnalysisCutoffEngine;

/**
 * @brief Registers all known engines with the EngineFactory.
 *
 * Each engine is associated with its uppercase, dot-separated task name
 * (e.g. "DATA.MERGE", "PROPERTIES.TWOBODY"). The CLI calls engine_factory::create(...)
 * using that name to construct the correct engine.
 */
void registerAllEngines();

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ALL_ENGINES_H
