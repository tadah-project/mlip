#include <tadah/mlip/all_engines.h>
#include <tadah/mlip/engine_factory.h> // EngineFactory
#include <tadah/core/logger.h>
#include <stdexcept>
#include <iostream>

// Place each engine’s class definition + registration in here

namespace tadah {
namespace mlip {

//------------------------------------------------------------------------------
// TrainEngine
//------------------------------------------------------------------------------
class TrainEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "TrainEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "TrainEngine validate() called. No mandatory keys in stub.");
  }
};

//------------------------------------------------------------------------------
// PredictEngine
//------------------------------------------------------------------------------
class PredictEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    std::cout << "PredictEngine run() called. Stub action here." << std::endl;
    core::Logger::instance().log(core::LogLevel::INFO,
      "PredictEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {
    std::cout << "PredictEngine validate() called. Stub action here." << std::endl;
    // Example: ensure "POTENTIAL" is present
    core::Logger::instance().log(core::LogLevel::INFO,
      "PredictEngine validate() stub. Checking required keys if needed.");
  }
};

//------------------------------------------------------------------------------
// HpoEngine
//------------------------------------------------------------------------------
class HpoEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "HpoEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "HpoEngine validate() stub. Checking required keys if needed.");
  }
};
//------------------------------------------------------------------------------
// DataEngine
//------------------------------------------------------------------------------
class DataEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataEngine validate() stub. Checking required keys if needed.");
  }
};

//------------------------------------------------------------------------------
// DataConvertEngine
//------------------------------------------------------------------------------
class DataConvertEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataConvertEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataConvertEngine validate() stub. Checking required keys if needed.");
  }
};

//------------------------------------------------------------------------------
// DataReadEngine
//------------------------------------------------------------------------------
class DataReadEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataReadEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataWriteEngine
//------------------------------------------------------------------------------
class DataWriteEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataWriteEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataMergeEngine
//------------------------------------------------------------------------------
class DataMergeEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataMergeEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataSplitEngine
//------------------------------------------------------------------------------
class DataSplitEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataSplitEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataDedupEngine
//------------------------------------------------------------------------------
class DataDedupEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataDedupEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataSampleEngine
//------------------------------------------------------------------------------
class DataSampleEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataSampleEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// DataBalanceEngine
//------------------------------------------------------------------------------
class DataBalanceEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "DataBalanceEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// TransformEngine
//------------------------------------------------------------------------------
class TransformEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "TransformEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesTwobodyEngine
//------------------------------------------------------------------------------
class PropertiesTwobodyEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesTwobodyEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesEcurveEngine
//------------------------------------------------------------------------------
class PropertiesEcurveEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesEcurveEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesEosEngine
//------------------------------------------------------------------------------
class PropertiesEosEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesEosEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesDefectEngine
//------------------------------------------------------------------------------
class PropertiesDefectEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesDefectEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesSurfaceEngine
//------------------------------------------------------------------------------
class PropertiesSurfaceEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesSurfaceEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// PropertiesMechanicsEngine
//------------------------------------------------------------------------------
class PropertiesMechanicsEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "PropertiesMechanicsEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// AnalysisEngine
//------------------------------------------------------------------------------
class AnalysisEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "AnalysisEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};
//------------------------------------------------------------------------------
// AnalysisDescriptorEngine
//------------------------------------------------------------------------------
class AnalysisDescriptorEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "AnalysisDescriptorEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// AnalysisBfuncEngine
//------------------------------------------------------------------------------
class AnalysisBfuncEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "AnalysisBfuncEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

//------------------------------------------------------------------------------
// AnalysisCutoffEngine
//------------------------------------------------------------------------------
class AnalysisCutoffEngine : public IEngine {
public:
  int run(const tadah::core::Context &ctx) override {
    core::Logger::instance().log(core::LogLevel::INFO,
      "AnalysisCutoffEngine run() called. Stub action here.");
    return 0;
  }
  void validate(const tadah::core::Context &ctx) override {}
};

/**
 * @brief One-time registration of all engine classes in the EngineFactory.
 */
void registerAllEngines() {
  using tadah::mlip::EngineFactory;

  // The blueprint tasks are uppercase, e.g. "TRAIN" or "DATA.MERGE"
  EngineFactory::registerEngine("train", [](){ return std::make_unique<TrainEngine>(); });
  EngineFactory::registerEngine("predict", [](){ return std::make_unique<PredictEngine>(); });
  EngineFactory::registerEngine("hpo", [](){ return std::make_unique<HpoEngine>(); });

  EngineFactory::registerEngine("data", [](){ return std::make_unique<DataEngine>(); });
  EngineFactory::registerEngine("data.convert", [](){ return std::make_unique<DataConvertEngine>(); });
  EngineFactory::registerEngine("data.read", [](){ return std::make_unique<DataReadEngine>(); });
  EngineFactory::registerEngine("data.write", [](){ return std::make_unique<DataWriteEngine>(); });
  EngineFactory::registerEngine("data.merge", [](){ return std::make_unique<DataMergeEngine>(); });
  EngineFactory::registerEngine("data.split", [](){ return std::make_unique<DataSplitEngine>(); });
  EngineFactory::registerEngine("data.dedup", [](){ return std::make_unique<DataDedupEngine>(); });
  EngineFactory::registerEngine("data.sample", [](){ return std::make_unique<DataSampleEngine>(); });
  EngineFactory::registerEngine("data.balance", [](){ return std::make_unique<DataBalanceEngine>(); });

  EngineFactory::registerEngine("transform", [](){ return std::make_unique<TransformEngine>(); });

  EngineFactory::registerEngine("properties.twobody", [](){ return std::make_unique<PropertiesTwobodyEngine>(); });
  EngineFactory::registerEngine("properties.ecurve", [](){ return std::make_unique<PropertiesEcurveEngine>(); });
  EngineFactory::registerEngine("properties.eos", [](){ return std::make_unique<PropertiesEosEngine>(); });
  EngineFactory::registerEngine("properties.defect", [](){ return std::make_unique<PropertiesDefectEngine>(); });
  EngineFactory::registerEngine("properties.surface", [](){ return std::make_unique<PropertiesSurfaceEngine>(); });
  EngineFactory::registerEngine("properties.mechanics", [](){ return std::make_unique<PropertiesMechanicsEngine>(); });

  EngineFactory::registerEngine("analysis", [](){ return std::make_unique<AnalysisEngine>(); });
  EngineFactory::registerEngine("analysis.descriptor", [](){ return std::make_unique<AnalysisDescriptorEngine>(); });
  EngineFactory::registerEngine("analysis.bfunc", [](){ return std::make_unique<AnalysisBfuncEngine>(); });
  EngineFactory::registerEngine("analysis.cutoff", [](){ return std::make_unique<AnalysisCutoffEngine>(); });

  // Additional tasks or subcommands can be added here
}

} // namespace mlip
} // namespace tadah
