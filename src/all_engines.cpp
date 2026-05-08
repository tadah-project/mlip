#include <tadah/mlip/engines/all_engines.h>
#include <tadah/mlip/engines/engine_factory.h>

#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/mlip/engines/data_convert_engine.h>
#include <tadah/mlip/engines/data_print_engine.h>
#include <tadah/mlip/engines/data_write_engine.h>
#include <tadah/mlip/engines/data_merge_engine.h>
#include <tadah/mlip/engines/data_split_engine.h>
#include <tadah/mlip/engines/data_dedup_engine.h>
#include <tadah/mlip/engines/data_sample_engine.h>
#include <tadah/mlip/engines/data_balance_engine.h>

#include <tadah/mlip/engines/properties_pairwise_engine.h>

#include <tadah/mlip/engines/analysis_descriptor_engine.h>
#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/mlip/engines/analysis_cutoff_engine.h>
#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>
#include <tadah/mlip/engines/analysis_check_bounds_engine.h>
#include <tadah/mlip/engines/analysis_suggest_bounds_engine.h>

#include <stdexcept>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

void registerAllMLIPEngines() {
  EngineFactory::registerEngine("TRAIN", [](){ return std::make_unique<TrainEngine>(); });
  EngineFactory::registerEngine("PREDICT", [](){ return std::make_unique<PredictEngine>(); });
  EngineFactory::registerEngine("EXPLAIN", [](){ return std::make_unique<ExplainEngine>(); });

  EngineFactory::registerEngine("DATA.convert", [](){ return std::make_unique<DataConvertEngine>(); });
  EngineFactory::registerEngine("DATA.print", [](){ return std::make_unique<DataPrintEngine>(); });
  EngineFactory::registerEngine("DATA.write", [](){ return std::make_unique<DataWriteEngine>(); });
  EngineFactory::registerEngine("DATA.merge", [](){ return std::make_unique<DataMergeEngine>(); });
  EngineFactory::registerEngine("DATA.split", [](){ return std::make_unique<DataSplitEngine>(); });
  EngineFactory::registerEngine("DATA.dedup", [](){ return std::make_unique<DataDedupEngine>(); });
  EngineFactory::registerEngine("DATA.sample", [](){ return std::make_unique<DataSampleEngine>(); });
  EngineFactory::registerEngine("DATA.balance", [](){ return std::make_unique<DataBalanceEngine>(); });
  
  // EngineFactory::registerEngine("transform", [](){ return std::make_unique<TransformEngine>(); });
  
  EngineFactory::registerEngine("PROPERTIES.pairwise", [](){ return std::make_unique<PropertiesPairwiseEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.ecurve", [](){ return std::make_unique<PropertiesEcurveEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.eos", [](){ return std::make_unique<PropertiesEosEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.defect", [](){ return std::make_unique<PropertiesDefectEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.surface", [](){ return std::make_unique<PropertiesSurfaceEngine>(); });
  // EngineFactory::registerEngine("PROPERTIES.mechanics", [](){ return std::make_unique<PropertiesMechanicsEngine>(); });
  //
  EngineFactory::registerEngine("ANALYSIS.descriptor", [](){ return std::make_unique<AnalysisDescriptorEngine>(); });
  EngineFactory::registerEngine("ANALYSIS.bfunc", [](){ return std::make_unique<AnalysisBfuncEngine>(); });
  EngineFactory::registerEngine("ANALYSIS.cutoff", [](){ return std::make_unique<AnalysisCutoffEngine>(); });
  EngineFactory::registerEngine("ANALYSIS.dataset_stats", [](){ return std::make_unique<AnalysisDatasetStatsEngine>(); });
  EngineFactory::registerEngine("ANALYSIS.suggest_bounds", [](){ return std::make_unique<AnalysisSuggestBoundsEngine>(); });
  EngineFactory::registerEngine("ANALYSIS.check_bounds", [](){ return std::make_unique<AnalysisCheckBoundsEngine>(); });

}

} // namespace engines
} // namespace mlip
} // namespace tadah
