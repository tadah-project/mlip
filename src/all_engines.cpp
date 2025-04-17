#include <tadah/mlip/engines/all_engines.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <stdexcept>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

void registerAllEngines() {
  using tadah::mlip::engines::EngineFactory;
  using tadah::mlip::engines::TrainEngine;
  EngineFactory::registerEngine("TRAIN", [](){ return std::make_unique<TrainEngine>(); });
  EngineFactory::registerEngine("PREDICT", [](){ return std::make_unique<PredictEngine>(); });
  EngineFactory::registerEngine("EXPLAIN", [](){ return std::make_unique<ExplainEngine>(); });
  // EngineFactory::registerEngine("HPO", [](){ return std::make_unique<HpoEngine>(); });
  //
  // EngineFactory::registerEngine("DATA", [](){ return std::make_unique<DataEngine>(); });
  // EngineFactory::registerEngine("DATA.convert", [](){ return std::make_unique<DataConvertEngine>(); });
  // EngineFactory::registerEngine("DATA.read", [](){ return std::make_unique<DataReadEngine>(); });
  // EngineFactory::registerEngine("DATA.write", [](){ return std::make_unique<DataWriteEngine>(); });
  // EngineFactory::registerEngine("DATA.merge", [](){ return std::make_unique<DataMergeEngine>(); });
  // EngineFactory::registerEngine("DATA.split", [](){ return std::make_unique<DataSplitEngine>(); });
  // EngineFactory::registerEngine("DATA.dedup", [](){ return std::make_unique<DataDedupEngine>(); });
  // EngineFactory::registerEngine("DATA.sample", [](){ return std::make_unique<DataSampleEngine>(); });
  // EngineFactory::registerEngine("DATA.balance", [](){ return std::make_unique<DataBalanceEngine>(); });
  //
  // EngineFactory::registerEngine("transform", [](){ return std::make_unique<TransformEngine>(); });
  //
  // EngineFactory::registerEngine("PROPERTIES.twobody", [](){ return std::make_unique<PropertiesTwobodyEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.ecurve", [](){ return std::make_unique<PropertiesEcurveEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.eos", [](){ return std::make_unique<PropertiesEosEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.defect", [](){ return std::make_unique<PropertiesDefectEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.surface", [](){ return std::make_unique<PropertiesSurfaceEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.mechanics", [](){ return std::make_unique<PropertiesMechanicsEngine>(); });
  //
  // EngineFactory::registerEngine("ANALYSIS", [](){ return std::make_unique<AnalysisEngine>(); });
  // EngineFactory::registerEngine("ANALYSIS.descriptor", [](){ return std::make_unique<AnalysisDescriptorEngine>(); });
  // EngineFactory::registerEngine("ANALYSIS.bfunc", [](){ return std::make_unique<AnalysisBfuncEngine>(); });
  // EngineFactory::registerEngine("ANALYSIS.cutoff", [](){ return std::make_unique<AnalysisCutoffEngine>(); });

}

} // namespace engines
} // namespace mlip
} // namespace tadah
